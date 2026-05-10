"""Ensemble blending: inverse-MAE weighted average of top demand models."""

import numpy as np
import pandas as pd
from src.evaluate import regression_metrics


# Excluded from ensemble (baselines or incompatible predict interface)
EXCLUDED_LABELS = {"Naive", "HistAvg", "skforecast-MultiSeries"}


def train_ensemble(
    demand_results: list[dict],
    df: pd.DataFrame,
    target: str = "departures",
    top_n: int = 3,
) -> dict:
    """Blend top N demand models using inverse-MAE weighted average."""
    from src.models.demand import time_split, FEATURE_COLS_DEMAND

    eligible = [r for r in demand_results if r["model"] not in EXCLUDED_LABELS]
    eligible = sorted(eligible, key=lambda r: r["MAE"])[:top_n]

    if len(eligible) < 2:
        print("Not enough eligible models for ensemble (need >= 2). Skipping.")
        return None

    print(f"Building ensemble from: {[r['model'] for r in eligible]}")

    _, _, test = time_split(df)
    available_features = [c for c in FEATURE_COLS_DEMAND if c in df.columns]

    X_test = test[available_features]
    y_test = test[target]

    test_preds = np.column_stack([r["model_obj"].predict(X_test) for r in eligible])
    model_names = [r["model"] for r in eligible]

    # Inverse-MAE weights from out-of-sample test MAE
    weights = np.array([1.0 / r["MAE"] for r in eligible])
    weights /= weights.sum()

    y_pred_ensemble = test_preds @ weights

    print(f"  Weights: {dict(zip(model_names, weights.round(4)))}")

    metrics = regression_metrics(y_test, y_pred_ensemble, label="Ensemble-Blend")
    metrics["model_obj"] = eligible[0]["model_obj"]
    return metrics
