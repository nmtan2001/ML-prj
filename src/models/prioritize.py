"""Task 3: Capacity prioritization using predictions from Tasks 1 and 2."""

import pandas as pd


def compute_priority_score(
    station_summary: pd.DataFrame,
    w_demand: float = 0.4,
    w_risk: float = 0.4,
    w_importance: float = 0.2,
) -> pd.DataFrame:
    """Weighted priority score from demand, risk frequency, and station importance."""
    df = station_summary.copy()

    # Normalize each component to [0, 1]
    for col in ["predicted_demand", "risk_frequency", "avg_daily_demand"]:
        col_min, col_max = df[col].min(), df[col].max()
        if col_max > col_min:
            df[f"{col}_norm"] = (df[col] - col_min) / (col_max - col_min)
        else:
            df[f"{col}_norm"] = 0.0

    df["priority_score"] = (
        w_demand * df["predicted_demand_norm"]
        + w_risk * df["risk_frequency_norm"]
        + w_importance * df["avg_daily_demand_norm"]
    )

    return df.sort_values("priority_score", ascending=False).reset_index(drop=True)


def generate_station_summary(
    hourly_with_preds: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate hourly predictions to per-station summary for prioritization."""
    summary = hourly_with_preds.groupby("station_id").agg(
        predicted_demand=("predicted_departures", "mean"),
        risk_frequency=("predicted_risk", "mean"),
        avg_daily_demand=("departures", lambda x: x.sum() / hourly_with_preds.loc[x.index, "hour"].dt.date.nunique()),
    ).reset_index()

    return summary
