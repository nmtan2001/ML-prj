"""Task 2: Station risk classification models."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import f1_score, make_scorer
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from src.evaluate import classification_metrics


def _focal_loss_xgb(labels, preds, gamma=2.0, alpha=0.25):
    """Focal loss objective for XGBoost binary classification."""
    preds = 1.0 / (1.0 + np.exp(-preds))
    eps = 1e-7
    preds = np.clip(preds, eps, 1 - eps)

    p_t = np.where(labels == 1, preds, 1 - preds)
    alpha_t = np.where(labels == 1, alpha, 1 - alpha)
    focal_weight = alpha_t * (1 - p_t) ** gamma

    grad = focal_weight * (preds - labels)
    hess = focal_weight * preds * (1 - preds)
    return grad, hess


FEATURE_COLS_RISK = [
    "hour_of_day", "day_of_week", "is_weekend", "is_rush_hour",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "weekend_x_hour", "rush_x_weekend",
    "is_holiday", "is_day_before_holiday", "is_day_after_holiday",
    "departures_lag_1h", "departures_lag_2h", "departures_lag_3h",
    "departures_lag_12h", "departures_lag_24h", "departures_lag_48h",
    "departures_lag_168h", "departures_lag_336h", "departures_lag_672h",
    "departures_roll_mean_3h", "departures_roll_mean_6h",
    "departures_roll_mean_12h", "departures_roll_mean_24h",
    "departures_roll_std_3h", "departures_roll_std_6h",
    "departures_diff_1h",
    "area_total_departures", "area_active_stations",
    "latitude", "longitude",
    "bikes_available_lag_1h",
    "dock_utilization_lag_1h",
    "temperature", "humidity", "precipitation", "wind_speed",
    "is_precipitating", "precipitation_x_weekend",
    "station_mean_demand", "station_hour_mean", "station_weekend_ratio",
    "knn_departures_sum_lag_1h", "knn_departures_mean_lag_1h",
    "knn_departures_sum_lag_3h", "knn_departures_mean_lag_3h",
    "knn_departures_sum_lag_24h", "knn_departures_mean_lag_24h",
    "forecasted_departures", "forecasted_net_flow", "forecasted_bikes_available",
]


def _find_best_threshold(model, X_val, y_val) -> float:
    """Find decision threshold that maximizes F1 on validation set."""
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_val)[:, 1]
    else:
        return 0.5
    best_thresh, best_f1 = 0.5, 0.0
    for thresh in np.arange(0.1, 0.9, 0.05):
        y_pred = (y_proba >= thresh).astype(int)
        f1 = f1_score(y_val, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
    return best_thresh


def train_risk_models(df: pd.DataFrame, target: str = "is_high_risk") -> list[dict]:
    """Train risk classification models with GridSearchCV and threshold tuning."""
    from src.models.demand import time_split

    available_features = [c for c in FEATURE_COLS_RISK if c in df.columns]
    train, val, test = time_split(df)

    X_train, y_train = train[available_features], train[target]
    X_val, y_val = val[available_features], val[target]
    X_test, y_test = test[available_features], test[target]

    # Combine train+val for GridSearchCV
    X_trainval = pd.concat([X_train, X_val])
    y_trainval = pd.concat([y_train, y_val])
    tscv = TimeSeriesSplit(n_splits=3)

    neg_count = (y_trainval == 0).sum()
    pos_count = (y_trainval == 1).sum()
    scale_ratio = neg_count / pos_count if pos_count > 0 else 1.0

    f1_scorer = make_scorer(f1_score, zero_division=0)

    # LogisticRegression GridSearchCV
    print("Training LogisticRegression (GridSearchCV)...")
    lr_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(class_weight="balanced", solver="lbfgs", max_iter=2000)),
    ])
    lr_grid = {"clf__C": [0.01, 0.1, 1.0, 10.0]}
    lr_search = GridSearchCV(lr_pipe, lr_grid, cv=tscv, scoring=f1_scorer, verbose=0)
    lr_search.fit(X_trainval, y_trainval)
    print(f"  Best params: {lr_search.best_params_}")

    # RandomForest GridSearchCV
    print("Training RandomForest (GridSearchCV)...")
    rf = RandomForestClassifier(random_state=42, n_jobs=-1, class_weight="balanced")
    rf_grid = {
        "n_estimators": [100, 200],
        "max_depth": [10, 20, None],
        "min_samples_leaf": [1, 5],
    }
    rf_search = GridSearchCV(rf, rf_grid, cv=tscv, scoring=f1_scorer, verbose=0)
    rf_search.fit(X_trainval, y_trainval)
    print(f"  Best params: {rf_search.best_params_}")

    # XGBoost GridSearchCV
    print("Training XGBoost (GridSearchCV)...")
    xgb = XGBClassifier(random_state=42, scale_pos_weight=scale_ratio, early_stopping_rounds=20)
    xgb_grid = {
        "n_estimators": [200, 500],
        "max_depth": [3, 6, 10],
        "learning_rate": [0.05, 0.1],
    }
    xgb_search = GridSearchCV(xgb, xgb_grid, cv=tscv, scoring=f1_scorer, verbose=0)
    xgb_search.fit(X_trainval, y_trainval, eval_set=[(X_val, y_val)], verbose=False)
    print(f"  Best params: {xgb_search.best_params_}")

    # LightGBM GridSearchCV
    print("Training LightGBM (GridSearchCV)...")
    lgbm = LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1,
                           scale_pos_weight=scale_ratio)
    lgbm_grid = {
        "n_estimators": [200, 500],
        "max_depth": [3, 6, 10, -1],
        "learning_rate": [0.05, 0.1],
    }
    lgbm_search = GridSearchCV(lgbm, lgbm_grid, cv=tscv, scoring=f1_scorer, verbose=0)
    lgbm_search.fit(X_trainval, y_trainval)
    print(f"  Best params: {lgbm_search.best_params_}")

    # XGBoost with focal loss (uses best params from standard XGBoost)
    print("Training XGBoost-FocalLoss...")
    best_xgb_params = xgb_search.best_params_
    focal_model = XGBClassifier(
        random_state=42,
        scale_pos_weight=scale_ratio,
        early_stopping_rounds=20,
        objective=_focal_loss_xgb,
        **best_xgb_params,
    )
    focal_model.fit(
        X_trainval, y_trainval,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    # Evaluate all with threshold tuning
    results = []
    models_to_eval = [
        ("LogisticRegression", lr_search.best_estimator_),
        ("RandomForest", rf_search.best_estimator_),
        ("XGBoost", xgb_search.best_estimator_),
        ("LightGBM", lgbm_search.best_estimator_),
        ("XGBoost-FocalLoss", focal_model),
    ]
    for name, model in models_to_eval:
        best_thresh = _find_best_threshold(model, X_val, y_val)
        print(f"  {name} optimal threshold: {best_thresh:.2f}")

        if hasattr(model, "predict_proba") and best_thresh != 0.5:
            y_proba = model.predict_proba(X_test)[:, 1]
            y_pred = (y_proba >= best_thresh).astype(int)
        else:
            y_pred = model.predict(X_test)

        metrics = classification_metrics(y_test, y_pred, label=name)
        metrics["model_obj"] = model
        metrics["y_test"] = y_test
        metrics["y_pred"] = y_pred
        results.append(metrics)

    return results
