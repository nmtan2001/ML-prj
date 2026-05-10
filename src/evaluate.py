"""Evaluation utilities: metrics, plots, model comparison."""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    precision_score,
    recall_score,
    root_mean_squared_error,
    r2_score,
    roc_auc_score,
    average_precision_score,
)
import matplotlib.pyplot as plt
import seaborn as sns


def regression_metrics(y_true: pd.Series, y_pred: pd.Series, label: str = "") -> dict:
    """Compute regression metrics including WAPE."""
    mae = mean_absolute_error(y_true, y_pred)
    denom = np.sum(np.abs(y_true))
    wape = np.sum(np.abs(y_true - y_pred)) / denom * 100 if denom > 0 else 0
    return {
        "model": label,
        "MAE": mae,
        "RMSE": root_mean_squared_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
        "WAPE": wape,
    }


def classification_metrics(y_true: pd.Series, y_pred: pd.Series, label: str = "",
                           y_proba: pd.Series = None) -> dict:
    """Compute classification metrics."""
    result = {
        "model": label,
        "Accuracy": accuracy_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
    }
    if y_proba is not None:
        try:
            result["PR-AUC"] = average_precision_score(y_true, y_proba)
        except ValueError:
            pass
    return result


def plot_confusion_matrix(y_true, y_pred, labels=None, title="Confusion Matrix"):
    """Plot a confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    return fig


def plot_model_comparison(results: list[dict], metric_key: str, title: str = ""):
    """Bar chart comparing models on a given metric."""
    df = pd.DataFrame(results)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(df["model"], df[metric_key])
    ax.set_xlabel(metric_key)
    ax.set_title(title or f"Model Comparison: {metric_key}")
    plt.tight_layout()
    return fig


def plot_predictions(y_true, y_pred, title="Predicted vs Actual"):
    """Scatter plot of predicted vs actual values."""
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_true, y_pred, alpha=0.3, s=10)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, "r--", linewidth=1)
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title(title)
    return fig
