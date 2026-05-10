"""Simulate dock inventory levels from trip data for risk labeling."""

import numpy as np
import pandas as pd

from src.config import DATA_PROCESSED, DEFAULT_STATION_CAPACITY, NEAR_EMPTY_THRESHOLD, NEAR_FULL_RATIO


def simulate_inventory(hourly: pd.DataFrame, station_info: pd.DataFrame) -> pd.DataFrame:
    """Simulate bike inventory using mean-reverting model."""
    alpha = 0.3
    cap_map = station_info.set_index("station_id")["capacity"].to_dict()
    hourly = hourly.sort_values(["station_id", "hour"]).copy()
    hourly["capacity"] = hourly["station_id"].map(cap_map).fillna(DEFAULT_STATION_CAPACITY).astype(int)

    bikes_list = []
    for station_id, group in hourly.groupby("station_id"):
        cap = cap_map.get(station_id, DEFAULT_STATION_CAPACITY)
        target = cap / 2
        current = target
        station_bikes = []
        for _, row in group.iterrows():
            current = current + row["net_flow"] + alpha * (target - current)
            current = float(np.clip(current, 0, cap))
            station_bikes.append(current)
        group = group.copy()
        group["bikes_available"] = station_bikes
        bikes_list.append(group)

    hourly = pd.concat(bikes_list, ignore_index=True)
    hourly["dock_utilization"] = hourly["bikes_available"] / hourly["capacity"]

    return hourly


def label_risk(hourly: pd.DataFrame) -> pd.DataFrame:
    """Label near-empty (<=2 bikes) and near-full (>=90% utilization) as high risk."""
    hourly["is_near_empty"] = (hourly["bikes_available"] <= NEAR_EMPTY_THRESHOLD).astype(int)
    hourly["is_near_full"] = (hourly["dock_utilization"] >= NEAR_FULL_RATIO).astype(int)
    hourly["is_high_risk"] = ((hourly["is_near_empty"] == 1) | (hourly["is_near_full"] == 1)).astype(int)
    return hourly


if __name__ == "__main__":
    from src.preprocess import load_processed

    hourly, station_info = load_processed()
    hourly = simulate_inventory(hourly, station_info)
    hourly = label_risk(hourly)
    print(hourly[["station_id", "hour", "bikes_available", "is_high_risk"]].head(20))


def simulate_and_label_cached(hourly: pd.DataFrame, station_info: pd.DataFrame) -> pd.DataFrame:
    """Simulate + label with parquet caching."""
    cache_path = DATA_PROCESSED / "hourly_simulated.parquet"
    if cache_path.exists():
        print("Loading cached simulated dock data...")
        result = pd.read_parquet(cache_path)
        print(f"Loaded {len(result)} records from cache.")
        return result
    hourly = simulate_inventory(hourly, station_info)
    hourly = label_risk(hourly)
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    hourly.to_parquet(cache_path, index=False)
    print(f"Cached simulated dock data ({len(hourly)} records).")
    return hourly
