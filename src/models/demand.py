"""Task 1: Station-level demand forecasting models."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from skforecast.recursive import ForecasterRecursiveMultiSeries

from src.evaluate import regression_metrics


def time_split(df: pd.DataFrame):
    """Split data chronologically into train/val/test (70/15/15)."""
    n = len(df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    return df.iloc[:train_end], df.iloc[train_end:val_end], df.iloc[val_end:]


FEATURE_COLS_DEMAND = [
    "hour_of_day", "day_of_week", "is_weekend", "is_rush_hour",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "departures_lag_1h", "departures_lag_2h", "departures_lag_3h",
    "departures_lag_24h", "departures_lag_168h",
    "departures_roll_mean_3h", "departures_roll_mean_6h",
    "departures_roll_mean_12h", "departures_roll_mean_24h",
    "departures_roll_std_3h", "departures_roll_std_6h",
    "departures_roll_std_12h", "departures_roll_std_24h",
    "departures_diff_1h",
    "area_total_departures", "area_active_stations",
    "latitude", "longitude",
    "temperature_lag1", "humidity_lag1", "precipitation_lag1", "wind_speed_lag1",
]


def train_demand_models(df: pd.DataFrame, target: str = "departures") -> list[dict]:
    """Train and evaluate multiple demand forecasting models.

    Returns list of result dicts with metrics and trained model objects.
    """
    available_features = [c for c in FEATURE_COLS_DEMAND if c in df.columns]
    train, val, test = time_split(df)

    X_train, y_train = train[available_features], train[target]
    X_val, y_val = val[available_features], val[target]
    X_test, y_test = test[available_features], test[target]

    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        "XGBoost": XGBRegressor(n_estimators=200, learning_rate=0.1, random_state=42, n_jobs=-1),
    }

    results = []

    # Naive baseline: predict last observed value
    y_pred_naive = y_test.shift(1).fillna(0)
    results.append(regression_metrics(y_test, y_pred_naive, label="Naive"))

    # Historical average by station and hour
    avg = train.groupby(["station_id", "hour_of_day"])[target].mean()
    y_pred_avg = test.apply(
        lambda r: avg.get((r["station_id"], r["hour_of_day"]), y_train.mean()), axis=1
    )
    results.append(regression_metrics(y_test, y_pred_avg, label="HistAvg"))

    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        metrics = regression_metrics(y_test, y_pred, label=name)
        metrics["model_obj"] = model
        results.append(metrics)

    return results


def train_skforecast_model(df: pd.DataFrame, target: str = "departures") -> dict:
    """Train a skforecast ForecasterRecursiveMultiSeries model.

    Fits on train+val to predict the test period, matching the same time split.
    Formats data as skforecast expects:
    - series: DataFrame with datetime index, one column per station_id
    - exog: DataFrame with datetime index, exogenous features
    """
    train, val, test = time_split(df)
    fit_data = pd.concat([train, val])

    available_features = [c for c in FEATURE_COLS_DEMAND if c in df.columns]
    exog_cols = [c for c in available_features if c not in ("latitude", "longitude")]

    # Pivot to wide format: index=hour, columns=station_id, values=target
    fit_series = fit_data.pivot(index="hour", columns="station_id", values=target).asfreq("h").ffill()
    test_series = test.pivot(index="hour", columns="station_id", values=target).asfreq("h").ffill()

    # Exogenous features: average across stations per hour
    fit_exog = fit_data.groupby("hour")[exog_cols].mean().asfreq("h").ffill()
    test_exog = test.groupby("hour")[exog_cols].mean().asfreq("h").ffill()

    print("Training skforecast ForecasterRecursiveMultiSeries...")
    forecaster = ForecasterRecursiveMultiSeries(
        estimator=XGBRegressor(n_estimators=200, learning_rate=0.1, random_state=42, n_jobs=-1),
        lags=[1, 2, 3, 24, 168],
        transformer_series=StandardScaler(),
    )
    forecaster.fit(series=fit_series, exog=fit_exog)

    # Predict test period
    steps = len(test_exog)
    y_pred = forecaster.predict(steps=steps, exog=test_exog)

    # y_pred is long-format: columns = ['level', 'pred']
    # Pivot to wide format matching test_series
    pred_wide = y_pred.pivot(columns="level", values="pred")
    # Align columns
    common_cols = list(set(pred_wide.columns) & set(test_series.columns))
    pred_aligned = pred_wide[common_cols].sort_index(axis=1)
    test_aligned = test_series[common_cols].sort_index(axis=1)

    y_pred_flat = pred_aligned.values.flatten().astype(float)
    y_test_flat = test_aligned.values.flatten().astype(float)

    mask = ~(np.isnan(y_pred_flat) | np.isnan(y_test_flat))
    metrics = regression_metrics(
        pd.Series(y_test_flat[mask]),
        pd.Series(y_pred_flat[mask]),
        label="skforecast-MultiSeries",
    )
    metrics["model_obj"] = forecaster
    print(f"  skforecast-MultiSeries  MAE={metrics['MAE']:.4f}  RMSE={metrics['RMSE']:.4f}")
    return metrics
