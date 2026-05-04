"""Task 2: Station risk classification models."""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.evaluate import classification_metrics


FEATURE_COLS_RISK = [
    "hour_of_day", "day_of_week", "is_weekend", "is_rush_hour",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "departures_lag_1h", "departures_lag_2h", "departures_lag_3h",
    "departures_lag_24h", "departures_lag_168h",
    "departures_roll_mean_3h", "departures_roll_mean_6h",
    "departures_roll_mean_12h", "departures_roll_mean_24h",
    "departures_roll_std_3h", "departures_roll_std_6h",
    "departures_diff_1h",
    "area_total_departures", "area_active_stations",
    "latitude", "longitude",
    "bikes_available_lag_1h",
    "dock_utilization_lag_1h",
    "temperature_lag1", "humidity_lag1", "precipitation_lag1", "wind_speed_lag1",
]


def train_risk_models(df: pd.DataFrame, target: str = "is_high_risk") -> list[dict]:
    """Train and evaluate risk classification models.

    Uses StandardScaler for LogisticRegression via Pipeline.
    Computes class imbalance ratio for XGBoost scale_pos_weight.
    Returns results with y_test and y_pred for confusion matrix plotting.
    """
    from src.models.demand import time_split

    available_features = [c for c in FEATURE_COLS_RISK if c in df.columns]
    train, val, test = time_split(df)

    X_train, y_train = train[available_features], train[target]
    X_test, y_test = test[available_features], test[target]

    # Compute class imbalance ratio for XGBoost
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_ratio = neg_count / pos_count if pos_count > 0 else 1.0

    models = {
        "LogisticRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs")),
        ]),
        "RandomForest": RandomForestClassifier(
            n_estimators=100, random_state=42, n_jobs=-1, class_weight="balanced",
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200, learning_rate=0.1, random_state=42,
            scale_pos_weight=scale_ratio,
        ),
    }

    results = []
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        metrics = classification_metrics(y_test, y_pred, label=name)
        metrics["model_obj"] = model
        metrics["y_test"] = y_test
        metrics["y_pred"] = y_pred
        results.append(metrics)

    return results
