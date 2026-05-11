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


# -- EDA plots --


def plot_temporal_patterns(hourly: pd.DataFrame) -> plt.Figure:
    """Average demand by hour of day (line) and by day of week (bar)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    by_hour = hourly.groupby(hourly["hour"].dt.hour)[["departures", "arrivals"]].mean()
    by_hour.plot(ax=axes[0], marker="o")
    axes[0].set_xlabel("Hour of Day")
    axes[0].set_ylabel("Average Trips")
    axes[0].set_title("Average Demand by Hour of Day")

    by_dow = hourly.groupby(hourly["hour"].dt.dayofweek)[["departures", "arrivals"]].mean()
    by_dow.index = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    by_dow.plot(ax=axes[1], kind="bar")
    axes[1].set_xlabel("Day of Week")
    axes[1].set_ylabel("Average Trips")
    axes[1].set_title("Average Demand by Day of Week")
    plt.tight_layout()
    return fig


def plot_weekly_trend(hourly: pd.DataFrame) -> plt.Figure:
    """Weekly total departures/arrivals over time."""
    weekly = hourly.set_index("hour").resample("W")[["departures", "arrivals"]].sum()
    fig, ax = plt.subplots(figsize=(14, 5))
    weekly.plot(ax=ax)
    ax.set_title("Weekly Total Demand Over Time")
    ax.set_xlabel("Week")
    ax.set_ylabel("Total Trips")
    plt.tight_layout()
    return fig


def plot_top_stations(hourly: pd.DataFrame, n: int = 15) -> plt.Figure:
    """Top N stations by total departures (horizontal bar)."""
    top = hourly.groupby("station_id").agg(
        total_departures=("departures", "sum"),
    ).sort_values("total_departures", ascending=False)
    fig, ax = plt.subplots(figsize=(12, 6))
    top["total_departures"].head(n).plot(kind="barh", ax=ax)
    ax.set_title(f"Top {n} Stations by Total Departures")
    ax.set_xlabel("Total Departures")
    ax.invert_yaxis()
    plt.tight_layout()
    return fig


def plot_station_map(hourly: pd.DataFrame) -> plt.Figure:
    """Scatter of station lat/lon colored by total departures."""
    station_stats = hourly.groupby("station_id").agg(
        total_departures=("departures", "sum"),
        lat=("latitude", "first"),
        lon=("longitude", "first"),
    )
    station_stats = station_stats[(station_stats["lat"] != 0) & (station_stats["lon"] != 0)]
    fig, ax = plt.subplots(figsize=(10, 10))
    scatter = ax.scatter(
        station_stats["lon"], station_stats["lat"],
        c=station_stats["total_departures"], cmap="YlOrRd",
        s=station_stats["total_departures"] / station_stats["total_departures"].max() * 500,
        alpha=0.7, edgecolors="black", linewidth=0.5,
    )
    plt.colorbar(scatter, label="Total Departures")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Station Locations (size = total departures)")
    plt.tight_layout()
    return fig


def plot_net_flow(hourly: pd.DataFrame, n: int = 15) -> plt.Figure:
    """Top receiving vs draining stations by average net flow."""
    net_flow = hourly.groupby("station_id")["net_flow"].agg(["mean", "std"]).sort_values("mean")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    net_flow["mean"].tail(n).plot(kind="barh", ax=axes[0], color="green", alpha=0.7)
    axes[0].set_title(f"Top {n} Receiving Stations (avg net flow)")
    axes[0].set_xlabel("Avg Net Flow (arrivals - departures)")
    net_flow["mean"].head(n).plot(kind="barh", ax=axes[1], color="red", alpha=0.7)
    axes[1].set_title(f"Top {n} Draining Stations (avg net flow)")
    axes[1].set_xlabel("Avg Net Flow (arrivals - departures)")
    plt.tight_layout()
    return fig


def plot_dock_utilization(hourly_sim: pd.DataFrame) -> plt.Figure:
    """Dock utilization time series for the busiest station with threshold lines."""
    sample_station = hourly_sim["station_id"].value_counts().index[0]
    station_data = hourly_sim[hourly_sim["station_id"] == sample_station].set_index("hour")
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(station_data.index, station_data["dock_utilization"], linewidth=0.8)
    ax.axhline(0.9, color="red", linestyle="--", label="Near-full threshold")
    cap = station_data["capacity"].iloc[0]
    ax.axhline(2 / cap, color="orange", linestyle="--", label="Near-empty threshold")
    ax.set_title(f"Dock Utilization Over Time - Station {sample_station}")
    ax.set_ylabel("Dock Utilization")
    ax.legend()
    plt.tight_layout()
    return fig


def plot_risk_analysis(hourly_sim: pd.DataFrame, n: int = 15) -> plt.Figure:
    """Risk distribution bar chart and top risk stations by frequency."""
    risk_counts = hourly_sim["is_high_risk"].value_counts()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    risk_counts.plot(kind="bar", ax=axes[0], color=["steelblue", "salmon"])
    axes[0].set_title("Overall Risk Distribution")
    axes[0].set_xticklabels(["Normal", "High Risk"], rotation=0)
    axes[0].set_ylabel("Hours")

    risk_by_station = hourly_sim.groupby("station_id")["is_high_risk"].mean().sort_values(ascending=False)
    risk_by_station.head(n).plot(kind="barh", ax=axes[1], color="salmon")
    axes[1].set_title(f"Top {n} Stations by Risk Frequency")
    axes[1].set_xlabel("Fraction of Hours at Risk")
    plt.tight_layout()
    return fig


def plot_correlation_heatmap(df: pd.DataFrame) -> plt.Figure:
    """Numeric feature correlation heatmap."""
    num_cols = df.select_dtypes(include=[np.number]).columns
    corr = df[num_cols].corr()
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(corr, cmap="RdBu_r", center=0, annot=False, ax=ax)
    ax.set_title("Feature Correlation Heatmap")
    plt.tight_layout()
    return fig


# -- Result plots --


def plot_demand_comparison(demand_results: list[dict]) -> plt.Figure:
    """Side-by-side MAE and RMSE bar charts for demand models."""
    demand_df = pd.DataFrame(demand_results)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    demand_df.plot(x="model", y="MAE", kind="barh", ax=axes[0], legend=False, color="steelblue")
    axes[0].set_title("MAE by Model")
    axes[0].invert_yaxis()
    demand_df.plot(x="model", y="RMSE", kind="barh", ax=axes[1], legend=False, color="coral")
    axes[1].set_title("RMSE by Model")
    axes[1].invert_yaxis()
    plt.tight_layout()
    return fig


def plot_all_risk_confusion(risk_results: list[dict]) -> plt.Figure:
    """Side-by-side confusion matrices for all risk models."""
    n = len(risk_results)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]
    for ax, r in zip(axes, risk_results):
        cm = confusion_matrix(r["y_test"], r["y_pred"])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["Normal", "Risk"], yticklabels=["Normal", "Risk"])
        ax.set_title(f"{r['model']}\nAcc={r['Accuracy']:.3f} F1={r['F1']:.3f}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    plt.tight_layout()
    return fig


def plot_prioritization(ranked: pd.DataFrame, n: int = 15) -> plt.Figure:
    """Priority barh + scatter map colored by priority score."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ranked.head(n).plot(x="station_id", y="priority_score", kind="barh", ax=axes[0], color="coral")
    axes[0].set_title(f"Top {n} Priority Stations")
    axes[0].set_xlabel("Priority Score")
    axes[0].invert_yaxis()

    plot_data = ranked[(ranked["lat"] != 0) & (ranked["lon"] != 0)]
    scatter = axes[1].scatter(
        plot_data["lon"], plot_data["lat"],
        c=plot_data["priority_score"], cmap="YlOrRd",
        s=plot_data["priority_score"] * 500,
        alpha=0.7, edgecolors="black", linewidth=0.5,
    )
    plt.colorbar(scatter, label="Priority Score", ax=axes[1])
    axes[1].set_title("Station Priority Map")
    axes[1].set_xlabel("Longitude")
    axes[1].set_ylabel("Latitude")
    plt.tight_layout()
    return fig
