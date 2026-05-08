"""Task 1: Station-level demand forecasting models."""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from skforecast.recursive import ForecasterRecursiveMultiSeries
from skforecast.model_selection import TimeSeriesFold, grid_search_forecaster_multiseries

from src.evaluate import regression_metrics
import optuna
from lightgbm import early_stopping as lgbm_early_stopping


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
    # Temporal
    "hour_of_day", "day_of_week", "month", "is_weekend", "is_rush_hour",
    "is_daylight",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos",
    "weekend_x_hour", "rush_x_weekend",
    # Fourier multi-seasonality
    "fourier_24h_sin_1", "fourier_24h_cos_1",
    "fourier_24h_sin_2", "fourier_24h_cos_2",
    "fourier_24h_sin_3", "fourier_24h_cos_3",
    "fourier_168h_sin_1", "fourier_168h_cos_1",
    "fourier_168h_sin_2", "fourier_168h_cos_2",
    "fourier_yearly_sin_1", "fourier_yearly_cos_1",
    # Holiday
    "is_holiday", "is_day_before_holiday", "is_day_after_holiday",
    # Lag
    "departures_lag_1h", "departures_lag_2h", "departures_lag_3h",
    "departures_lag_12h", "departures_lag_24h", "departures_lag_48h",
    "departures_lag_168h", "departures_lag_336h", "departures_lag_672h",
    # Rolling
    "departures_roll_mean_3h", "departures_roll_mean_6h",
    "departures_roll_mean_12h", "departures_roll_mean_24h",
    "departures_roll_std_3h", "departures_roll_std_6h",
    "departures_roll_std_12h", "departures_roll_std_24h",
    # Trend
    "departures_diff_1h",
    # Seasonal same-hour lags
    "departures_same_hour_roll_mean_3d", "departures_same_hour_roll_mean_7d",
    # Cross-station
    "area_total_departures", "area_total_arrivals", "area_active_stations",
    # Spatial
    "latitude", "longitude", "neighborhood_cluster",
    "cluster_departures_mean_lag_1h", "cluster_departures_mean_lag_24h",
    # Weather
    "temperature", "humidity", "precipitation", "wind_speed",
    "is_precipitating", "precip_roll_sum_3h", "precip_roll_sum_6h",
    "apparent_temp",
    "precipitation_x_weekend", "precipitation_x_rush",
    "temp_x_hour_sin", "temp_x_humidity",
    # Station encoding
    "station_mean_demand", "station_hour_mean", "station_weekend_ratio",
    # KNN spatial lags
    "knn_departures_mean_lag_1h",
    "knn_departures_mean_lag_3h",
    "knn_departures_mean_lag_24h",
    # Anomaly
    "anomaly_score",
]

# Feature subsets for ensemble diversity
_FEATURE_LAGS = [
    "departures_lag_1h", "departures_lag_2h", "departures_lag_3h",
    "departures_lag_12h", "departures_lag_24h", "departures_lag_48h",
    "departures_lag_168h", "departures_lag_336h", "departures_lag_672h",
    "departures_roll_mean_3h", "departures_roll_mean_6h",
    "departures_roll_mean_12h", "departures_roll_mean_24h",
    "departures_roll_std_3h", "departures_roll_std_6h",
    "departures_roll_std_12h", "departures_roll_std_24h",
    "departures_diff_1h",
    "departures_same_hour_roll_mean_3d", "departures_same_hour_roll_mean_7d",
]

_FEATURE_CALENDAR_WEATHER = [
    "hour_of_day", "day_of_week", "month", "is_weekend", "is_rush_hour",
    "is_daylight",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos",
    "fourier_24h_sin_1", "fourier_24h_cos_1", "fourier_24h_sin_2", "fourier_24h_cos_2",
    "fourier_24h_sin_3", "fourier_24h_cos_3",
    "fourier_168h_sin_1", "fourier_168h_cos_1", "fourier_168h_sin_2", "fourier_168h_cos_2",
    "fourier_yearly_sin_1", "fourier_yearly_cos_1",
    "weekend_x_hour", "rush_x_weekend",
    "is_holiday", "is_day_before_holiday", "is_day_after_holiday",
    "temperature", "humidity", "precipitation", "wind_speed",
    "is_precipitating", "precip_roll_sum_3h", "precip_roll_sum_6h",
    "apparent_temp",
    "precipitation_x_weekend", "precipitation_x_rush",
    "temp_x_hour_sin", "temp_x_humidity",
]


def train_demand_models(df: pd.DataFrame, target: str = "departures") -> list[dict]:
    """Train and evaluate demand forecasting models with GridSearchCV.

    Uses global timestamp split. GridSearchCV with TimeSeriesSplit for
    hyperparameter tuning on RF and XGBoost.
    Returns list of result dicts with metrics and trained model objects.
    """
    available_features = [c for c in FEATURE_COLS_DEMAND if c in df.columns]
    train, val, test = time_split(df)

    X_train, y_train = train[available_features], train[target]
    X_val, y_val = val[available_features], val[target]
    X_test, y_test = test[available_features], test[target]

    # Feature subsets for diversity (lag-heavy vs calendar+weather-heavy)
    avail_lags = [c for c in _FEATURE_LAGS if c in df.columns]
    avail_cal_weather = [c for c in _FEATURE_CALENDAR_WEATHER if c in df.columns]

    # Combine train+val for GridSearchCV (CV splits internally)
    X_trainval = pd.concat([X_train, X_val])
    y_trainval = pd.concat([y_train, y_val])
    tscv = TimeSeriesSplit(n_splits=3)

    results = []

    # Naive baseline
    test_sorted = test.sort_values(["station_id", "hour"])
    y_pred_naive = test_sorted.groupby("station_id")[target].shift(1).fillna(0)
    results.append(regression_metrics(y_test, y_pred_naive, label="Naive"))

    # Historical average
    avg = train.groupby(["station_id", "hour_of_day"])[target].mean()
    y_pred_avg = test.apply(
        lambda r: avg.get((r["station_id"], r["hour_of_day"]), y_train.mean()), axis=1
    )
    results.append(regression_metrics(y_test, y_pred_avg, label="HistAvg"))

    # Ridge
    print("Training Ridge...")
    ridge = Ridge()
    ridge_grid = {"alpha": [0.1, 1.0, 10.0, 100.0]}
    ridge_search = GridSearchCV(ridge, ridge_grid, cv=tscv, scoring="neg_mean_absolute_error")
    ridge_search.fit(X_trainval, y_trainval)
    print(f"  Best params: {ridge_search.best_params_}")
    y_pred = ridge_search.predict(X_test)
    metrics = regression_metrics(y_test, y_pred, label="Ridge")
    metrics["model_obj"] = ridge_search.best_estimator_
    results.append(metrics)

    # RandomForest with GridSearchCV (calendar+weather features for diversity)
    print("Training RandomForest (GridSearchCV, calendar+weather features)...")
    rf = RandomForestRegressor(random_state=42, n_jobs=-1)
    rf_grid = {
        "n_estimators": [100, 200],
        "max_depth": [15, 25],
        "min_samples_leaf": [5],
    }
    rf_search = GridSearchCV(rf, rf_grid, cv=tscv, scoring="neg_mean_absolute_error", verbose=0)
    rf_search.fit(X_trainval[avail_cal_weather], y_trainval)
    print(f"  Best params: {rf_search.best_params_}")
    y_pred = rf_search.predict(X_test[avail_cal_weather])
    metrics = regression_metrics(y_test, y_pred, label="RandomForest")
    metrics["model_obj"] = rf_search.best_estimator_
    metrics["feature_subset"] = avail_cal_weather
    results.append(metrics)

    # XGBoost with GridSearchCV + early stopping (lag features for diversity)
    print("Training XGBoost (GridSearchCV, lag features)...")
    xgb = XGBRegressor(
        random_state=42, n_jobs=-1, early_stopping_rounds=20,
        objective="reg:tweedie", tweedie_variance_power=1.5,
    )
    xgb_grid = {
        "n_estimators": [200, 400],
        "max_depth": [3, 6],
        "learning_rate": [0.05, 0.1],
    }
    xgb_search = GridSearchCV(
        xgb, xgb_grid, cv=tscv, scoring="neg_mean_absolute_error", verbose=0,
    )
    xgb_search.fit(X_trainval[avail_lags], y_trainval, eval_set=[(X_val[avail_lags], y_val)], verbose=False)
    print(f"  Best params: {xgb_search.best_params_}")
    y_pred = xgb_search.predict(X_test[avail_lags])
    metrics = regression_metrics(y_test, y_pred, label="XGBoost")
    metrics["model_obj"] = xgb_search.best_estimator_
    metrics["feature_subset"] = avail_lags
    results.append(metrics)

    # LightGBM with Optuna (Tweedie + HyperbandPruner)
    print("Training LightGBM (Optuna, Tweedie)...")
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def _lgbm_objective(trial):
        tweedie_p = trial.suggest_float("tweedie_variance_power", 1.1, 1.9)
        params = {
            "objective": "tweedie",
            "tweedie_variance_power": tweedie_p,
            "n_estimators": 1000,
            "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.15, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 31, 128),
            "max_depth": trial.suggest_int("max_depth", 5, 12),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
            "subsample": trial.suggest_float("subsample", 0.7, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1,
        }
        model = LGBMRegressor(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgbm_early_stopping(stopping_rounds=10, verbose=False)],
        )
        trial.set_user_attr("best_iteration", model.best_iteration_)
        preds = model.predict(X_val)
        return np.mean(np.abs(y_val.values - preds))

    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.HyperbandPruner(min_resource=50, max_resource=1000, reduction_factor=3),
    )
    study.optimize(_lgbm_objective, n_trials=20, show_progress_bar=False)
    best_params = study.best_params
    best_iteration = study.best_trial.user_attrs["best_iteration"]
    best_tweedie_p = best_params.pop("tweedie_variance_power")
    best_params.update({
        "objective": "tweedie",
        "tweedie_variance_power": best_tweedie_p,
        "n_estimators": best_iteration,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    })
    print(f"  Best params: {study.best_params}")
    print(f"  Best iteration: {best_iteration}")
    print(f"  Best val MAE: {study.best_value:.4f}")

    lgbm_best = LGBMRegressor(**best_params)
    lgbm_best.fit(X_trainval, y_trainval)
    y_pred = lgbm_best.predict(X_test)
    metrics = regression_metrics(y_test, y_pred, label="LightGBM")
    metrics["model_obj"] = lgbm_best
    results.append(metrics)

    return results


def train_skforecast_model(df: pd.DataFrame, target: str = "departures") -> dict:
    """Train a skforecast ForecasterRecursiveMultiSeries with grid search.

    Uses per-station exog dicts with station-specific encoding features.
    Compact grid (2 lags x 4 params) to keep runtime manageable.
    """
    train, val, test = time_split(df)
    fit_data = pd.concat([train, val])

    # Exog features: calendar + weather + station-specific encoding + holiday
    shared_exog_cols = [
        "hour_of_day", "day_of_week", "is_weekend", "is_rush_hour",
        "hour_sin", "hour_cos", "dow_sin", "dow_cos",
        "temperature", "precipitation", "wind_speed",
        "is_holiday", "is_day_before_holiday", "is_day_after_holiday",
    ]
    station_exog_cols = [
        "station_mean_demand", "station_hour_mean", "station_weekend_ratio",
    ]
    all_exog = [c for c in shared_exog_cols + station_exog_cols if c in df.columns]

    # Build series with datetime index and frequency
    fit_series = fit_data.pivot(index="hour", columns="station_id", values=target).asfreq("h").ffill()
    test_series = test.pivot(index="hour", columns="station_id", values=target).asfreq("h").ffill()

    # Build per-station exog dicts
    def _build_exog_dict(data, series_index):
        exog_dict = {}
        for sid in series_index.columns:
            st = data[data["station_id"] == sid].set_index("hour")
            exog_df = st[all_exog].reindex(series_index.index).ffill().bfill()
            exog_dict[sid] = exog_df
        return exog_dict

    fit_exog = _build_exog_dict(fit_data, fit_series)
    test_exog = _build_exog_dict(test, test_series)

    # Grid search
    print("Grid search for skforecast ForecasterRecursiveMultiSeries...")
    forecaster = ForecasterRecursiveMultiSeries(
        estimator=XGBRegressor(random_state=42, n_jobs=-1),
        lags=5,
        transformer_series=StandardScaler(),
        transformer_exog=StandardScaler(),
    )

    lags_grid = [[1, 2, 3, 24], [1, 2, 3, 24, 168]]
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [3, 6],
    }

    cv = TimeSeriesFold(initial_train_size=int(len(fit_series) * 0.85), steps=24, refit=False)

    grid_results = grid_search_forecaster_multiseries(
        forecaster=forecaster,
        series=fit_series,
        exog=fit_exog,
        lags_grid=lags_grid,
        param_grid=param_grid,
        metric="mean_absolute_error",
        cv=cv,
        return_best=True,
        suppress_warnings=True,
        show_progress=True,
    )
    print(f"  Best lags: {forecaster.lags}")
    print(f"  Best params: {grid_results.iloc[0]['params']}")

    # Predict test period with best forecaster
    steps = len(test_series)
    y_pred = forecaster.predict(
        steps=steps, exog=test_exog, suppress_warnings=True,
    )

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


def train_multi_target_models(
    df: pd.DataFrame, targets: list = None,
) -> dict:
    """Train multi-target models predicting departures and arrivals jointly.

    Uses MultiOutputRegressor wrapping XGBoost (Tweedie) and LightGBM.
    Returns per-target MAE and predicted net flow.
    """
    if targets is None:
        targets = ["departures", "arrivals"]

    available_features = [c for c in FEATURE_COLS_DEMAND if c in df.columns]
    train, val, test = time_split(df)

    X_trainval = pd.concat([train[available_features], val[available_features]])
    y_trainval = pd.concat([train[targets], val[targets]])
    X_test = test[available_features]
    y_test = test[targets]
    tscv = TimeSeriesSplit(n_splits=3)

    results = {"model_name": "MultiTarget"}

    for base_cls, name, grid in [
        (XGBRegressor, "MultiTarget-XGB", {
            "estimator__n_estimators": [200, 400],
            "estimator__max_depth": [3, 6],
            "estimator__learning_rate": [0.05, 0.1],
        }),
        (LGBMRegressor, "MultiTarget-LGBM", {
            "estimator__n_estimators": [200, 400],
            "estimator__max_depth": [3, 6, -1],
            "estimator__learning_rate": [0.05, 0.1],
        }),
    ]:
        if base_cls == XGBRegressor:
            base = base_cls(
                random_state=42, n_jobs=-1,
                objective="reg:tweedie", tweedie_variance_power=1.5,
            )
        else:
            base = base_cls(
                random_state=42, n_jobs=-1, verbose=-1,
                objective="tweedie", tweedie_variance_power=1.5,
            )
        mor = MultiOutputRegressor(base)
        search = GridSearchCV(
            mor, grid, cv=tscv,
            scoring="neg_mean_absolute_error", verbose=0,
        )
        print(f"Training {name} (GridSearchCV)...")
        search.fit(X_trainval, y_trainval)
        print(f"  Best params: {search.best_params_}")

        y_pred = search.predict(X_test)
        per_target = {}
        for i, t in enumerate(targets):
            mae = np.mean(np.abs(y_test[t].values - y_pred[:, i]))
            per_target[t] = mae
            print(f"  {name} {t} MAE={mae:.4f}")

        results[f"{name}_obj"] = search.best_estimator_
        results[f"{name}_pred"] = y_pred
        results[f"{name}_mae"] = per_target

    # Use best multi-target model (lower sum of MAEs) for net flow
    best_key = None
    best_sum = float("inf")
    for key in [k for k in results if k.endswith("_mae")]:
        mae_sum = sum(results[key].values())
        if mae_sum < best_sum:
            best_sum = mae_sum
            best_key = key

    best_prefix = best_key.replace("_mae", "")
    best_pred = results[f"{best_prefix}_pred"]
    arrivals_idx = targets.index("arrivals") if "arrivals" in targets else 1
    departures_idx = targets.index("departures") if "departures" in targets else 0
    net_flow_pred = best_pred[:, arrivals_idx] - best_pred[:, departures_idx]

    results["y_pred_departures"] = best_pred[:, departures_idx]
    results["y_pred_arrivals"] = best_pred[:, arrivals_idx]
    results["y_pred_net_flow"] = net_flow_pred
    results["best_model_obj"] = results[f"{best_prefix}_obj"]

    return results
