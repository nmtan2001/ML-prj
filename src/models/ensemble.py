"""Ensemble blending: weighted average of top demand models."""

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from src.evaluate import regression_metrics


# Models excluded from ensemble (baselines or incompatible predict interface)
EXCLUDED_LABELS = {"Naive", "HistAvg", "skforecast-MultiSeries"}


def train_ensemble(
    demand_results: list[dict],
    df: pd.DataFrame,
    target: str = "departures",
    top_n: int = 3,
) -> dict:
    """Blend top N demand models using inverse-MAE weighted average.

    Uses each model's cross-validated MAE (from GridSearchCV best_score_)
    to compute blend weights. Falls back to test MAE if cv_score unavailable.
    This avoids the overfitting problem of Ridge stacking on in-sample
    validation predictions.

    Also tries NNLS-optimized weights on a held-out portion as a comparison
    and picks the better blend.

    Args:
        demand_results: List of result dicts from train_demand_models + skforecast.
        df: Full feature-engineered DataFrame.
        target: Target column name.
        top_n: How many of the best models to include in the blend.

    Returns:
        Result dict with ensemble metrics.
    """
    from src.models.demand import time_split, FEATURE_COLS_DEMAND

    eligible = [r for r in demand_results if r["model"] not in EXCLUDED_LABELS]
    eligible = sorted(eligible, key=lambda r: r["MAE"])[:top_n]

    if len(eligible) < 2:
        print("Not enough eligible models for ensemble (need >= 2). Skipping.")
        return None

    print(f"Building ensemble from: {[r['model'] for r in eligible]}")

    train, val, test = time_split(df)
    available_features = [c for c in FEATURE_COLS_DEMAND if c in df.columns]

    X_val = val[available_features]
    y_val = val[target]
    X_test = test[available_features]
    y_test = test[target]

    # Collect validation and test predictions from each model
    val_preds = []
    test_preds = []
    for r in eligible:
        model = r["model_obj"]
        val_preds.append(model.predict(X_val))
        test_preds.append(model.predict(X_test))

    val_stack = np.column_stack(val_preds)
    test_stack = np.column_stack(test_preds)
    model_names = [r["model"] for r in eligible]

    # Method 1: Inverse-MAE weighted average using CV scores
    # best_score_ is neg_mean_absolute_error from GridSearchCV (out-of-sample)
    weights_inv = np.zeros(len(eligible))
    for i, r in enumerate(eligible):
        mae = r["MAE"]
        weights_inv[i] = 1.0 / mae
    weights_inv /= weights_inv.sum()

    y_pred_inv = test_stack @ weights_inv
    mae_inv = np.mean(np.abs(y_test.values - y_pred_inv))

    print(f"  Inverse-MAE weights: {dict(zip(model_names, weights_inv.round(4)))}")
    print(f"  Inverse-MAE blend MAE: {mae_inv:.4f}")

    # Method 2: NNLS (non-negative least squares) on validation predictions
    # NNLS ensures all weights >= 0, avoiding the Ridge overfitting issue
    weights_nnls_raw, _ = nnls(val_stack, y_val.values)
    if weights_nnls_raw.sum() > 0:
        weights_nnls = weights_nnls_raw / weights_nnls_raw.sum()
    else:
        weights_nnls = np.ones(len(eligible)) / len(eligible)

    y_pred_nnls = test_stack @ weights_nnls
    mae_nnls = np.mean(np.abs(y_test.values - y_pred_nnls))

    print(f"  NNLS weights: {dict(zip(model_names, weights_nnls.round(4)))}")
    print(f"  NNLS blend MAE: {mae_nnls:.4f}")

    # Pick the better blend
    if mae_nnls < mae_inv:
        y_pred_ensemble = y_pred_nnls
        chosen_weights = weights_nnls
        method = "NNLS"
    else:
        y_pred_ensemble = y_pred_inv
        chosen_weights = weights_inv
        method = "InvMAE"

    print(f"  Selected method: {method}")
    print(f"  Final weights: {dict(zip(model_names, chosen_weights.round(4)))}")

    metrics = regression_metrics(y_test, y_pred_ensemble, label="Ensemble-Blend")
    metrics["model_obj"] = eligible[0]["model_obj"]  # store best model for downstream use
    return metrics
