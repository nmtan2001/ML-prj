"""Task 3: Capacity prioritization using predictions from Tasks 1 and 2."""

import numpy as np
import pandas as pd


def _max_consecutive_risk(risk_series: pd.Series) -> int:
    """Longest streak of consecutive high-risk hours."""
    values = risk_series.values
    max_streak = streak = 0
    for v in values:
        if v:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0
    return max_streak


def compute_priority_score(
    station_summary: pd.DataFrame,
    w_demand: float = 0.35,
    w_risk: float = 0.35,
    w_sustained: float = 0.15,
    w_importance: float = 0.15,
) -> pd.DataFrame:
    """Weighted priority score from demand, risk, sustained imbalance, and importance."""
    df = station_summary.copy()

    components = ["predicted_demand", "risk_frequency", "max_consecutive_risk_hours", "capacity_strain"]
    for col in components:
        col_min, col_max = df[col].min(), df[col].max()
        if col_max > col_min:
            df[f"{col}_norm"] = (df[col] - col_min) / (col_max - col_min)
        else:
            df[f"{col}_norm"] = 0.0

    df["priority_score"] = (
        w_demand * df["predicted_demand_norm"]
        + w_risk * df["risk_frequency_norm"]
        + w_sustained * df["max_consecutive_risk_hours_norm"]
        + w_importance * df["capacity_strain_norm"]
    )

    return df.sort_values("priority_score", ascending=False).reset_index(drop=True)


def generate_station_summary(
    hourly_with_preds: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate hourly predictions to per-station summary for prioritization."""
    summary = hourly_with_preds.groupby("station_id").agg(
        predicted_demand=("predicted_departures", "mean"),
        risk_frequency=("predicted_risk", "mean"),
        max_consecutive_risk_hours=("predicted_risk", _max_consecutive_risk),
        capacity=("capacity", "first"),
        lat=("latitude", "first"),
        lon=("longitude", "first"),
    ).reset_index()

    # Capacity strain: predicted demand relative to dock capacity.
    # High strain = station is too small for its demand.
    summary["capacity_strain"] = summary["predicted_demand"] / summary["capacity"]

    # Distance to nearest neighbor (stations far from alternatives are more critical)
    _add_nearest_neighbor_distance(summary)

    return summary


def _add_nearest_neighbor_distance(df: pd.DataFrame) -> None:
    """Add min haversine distance (km) to nearest other station."""
    valid = df[(df["lat"] != 0) & (df["lon"] != 0)]
    if len(valid) < 2:
        df["nearest_neighbor_km"] = 0.0
        return

    lat_rad = np.radians(df["lat"].values)
    lon_rad = np.radians(df["lon"].values)
    R = 6371.0  # Earth radius in km

    distances = np.full(len(df), np.inf)
    for i in range(len(df)):
        if df["lat"].iloc[i] == 0:
            distances[i] = 0.0
            continue
        dlat = lat_rad - lat_rad[i]
        dlon = lon_rad - lon_rad[i]
        a = np.sin(dlat / 2) ** 2 + np.cos(lat_rad[i]) * np.cos(lat_rad) * np.sin(dlon / 2) ** 2
        dist = R * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
        dist[i] = np.inf  # exclude self
        distances[i] = dist.min()

    df["nearest_neighbor_km"] = distances
