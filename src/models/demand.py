"""Task 1: Station-level demand forecasting models."""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from skforecast.recursive import ForecasterRecursiveMultiSeries

from src.evaluate import regression_metrics


def time_split(df: pd.DataFrame):
    """Split data chronologically using global timestamp cutoffs (70/15/15).

    Ensures all stations share the same date boundaries for train/val/test.
    """
    timestamps = df["hour"].sort_values().unique()
    n = len(timestamps)
    train_end = timestamps[int(n * 0.70)]
    val_end = timestamps[int(n * 0.85)]
    train = df[df["hour"] < train_end].copy()
    val = df[(df["hour"] >= train_end) & (df["hour"] < val_end)].copy()
    test = df[df["hour"] >= val_end].copy()
    return train, val, test


FEATURE_COLS_DEMAND = [
    "hour_of_day", "day_of_week", "is_weekend", "is_rush_hour",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "weekend_x_hour", "rush_x_weekend",
    "departures_lag_1h", "departures_lag_2h", "departures_lag_3h",
    "departures_lag_12h", "departures_lag_24h", "departures_lag_48h",
    "departures_lag_168h",
    "departures_roll_mean_3h", "departures_roll_mean_6h",
    "departures_roll_mean_12h", "departures_roll_mean_24h",
    "departures_roll_std_3h", "departures_roll_std_6h",
    "departures_roll_std_12h", "departures_roll_std_24h",
    "departures_diff_1h",
    "area_total_departures", "area_active_stations",
    "latitude", "longitude",
    "temperature", "humidity", "precipitation", "wind_speed",
    "is_precipitating", "precipitation_x_weekend",
    "station_mean_demand", "station_hour_mean", "station_weekend_ratio",
]


def train_demand_models(df: pd.DataFrame, target: str = "departures") -> list[dict]:
    """Train and evaluate multiple demand forecasting models.

    Uses global timestamp split. XGBoost uses early stopping on validation set.
    Returns list of result dicts with metrics and trained model objects.
    """
    available_features = [c for c in FEATURE_COLS_DEMAND if c in df.columns]
    train, val, test = time_split(df)

    X_train, y_train = train[available_features], train[target]
    X_val, y_val = val[available_features], val[target]
    X_test, y_test = test[available_features], test[target]

    models = {
        "Ridge": Ridge(alpha=1.0),
        "RandomForest": RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1),
        "XGBoost": XGBRegressor(
            n_estimators=500, learning_rate=0.1, random_state=42, n_jobs=-1,
            early_stopping_rounds=20,
        ),
    }

    results = []

    # Naive baseline: use per-station lag-1 from test set
    test_sorted = test.sort_values(["station_id", "hour"])
    y_pred_naive = test_sorted.groupby("station_id")[target].shift(1).fillna(0)
    results.append(regression_metrics(y_test, y_pred_naive, label="Naive"))

    # Historical average by station and hour
    avg = train.groupby(["station_id", "hour_of_day"])[target].mean()
    y_pred_avg = test.apply(
        lambda r: avg.get((r["station_id"], r["hour_of_day"]), y_train.mean()), axis=1
    )
    results.append(regression_metrics(y_test, y_pred_avg, label="HistAvg"))

    for name, model in models.items():
        print(f"Training {name}...")
        if name == "XGBoost":
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        else:
            model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        metrics = regression_metrics(y_test, y_pred, label=name)
        metrics["model_obj"] = model
        results.append(metrics)

    return results


def train_skforecast_model(df: pd.DataFrame, target: str = "departures") -> dict:
    """Train a skforecast ForecasterRecursiveMultiSeries model.

    Fits on train+val to predict the test period. Uses per-station exogenous
    features via dict format, which preserves station-level variation.
    """
    train, val, test = time_split(df)
    fit_data = pd.concat([train, val])

    available_features = [c for c in FEATURE_COLS_DEMAND if c in df.columns]
    exog_cols = [c for c in available_features if c not in ("latitude", "longitude")]

    # Build series: DataFrame with datetime index and frequency
    fit_series = fit_data.pivot(index="hour", columns="station_id", values=target)
    fit_series = fit_series.asfreq("h").ffill()
    test_series = test.pivot(index="hour", columns="station_id", values=target)
    test_series = test_series.asfreq("h").ffill()

    # Build per-station exog dicts
    fit_exog_dict = {}
    test_exog_dict = {}
    for sid in fit_series.columns:
        station_fit = fit_data[fit_data["station_id"] == sid].set_index("hour")[exog_cols]
        station_test = test[test["station_id"] == sid].set_index("hour")[exog_cols]
        if len(station_fit) > 0:
            fit_exog_dict[sid] = station_fit
        if len(station_test) > 0:
            test_exog_dict[sid] = station_test

    print("Training skforecast ForecasterRecursiveMultiSeries...")
    forecaster = ForecasterRecursiveMultiSeries(
        estimator=XGBRegressor(n_estimators=200, learning_rate=0.1, random_state=42, n_jobs=-1),
        lags=[1, 2, 3, 24, 168],
        transformer_series=StandardScaler(),
        transformer_exog=StandardScaler(),
    )
    forecaster.fit(
        series=fit_series, exog=fit_exog_dict, suppress_warnings=True,
    )

    # Predict test period
    steps = len(test_series)
    y_pred = forecaster.predict(
        steps=steps, exog=test_exog_dict, suppress_warnings=True,
    )

    # y_pred is long-format: columns = ['level', 'pred']
    pred_wide = y_pred.pivot(columns="level", values="pred")
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
