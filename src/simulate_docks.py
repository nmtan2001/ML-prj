"""Simulate dock inventory levels from trip data for risk labeling."""

import numpy as np
import pandas as pd

from src.config import DEFAULT_STATION_CAPACITY, NEAR_EMPTY_THRESHOLD, NEAR_FULL_RATIO


def simulate_inventory(hourly: pd.DataFrame, station_info: pd.DataFrame) -> pd.DataFrame:
    """
    Simulate bike inventory at each station using mean-reverting model.

    Models operational rebalancing by reverting toward 50% capacity each hour.
    This prevents the unrealistic accumulation/depletion of pure cumulative models.
    """
    alpha = 0.3  # reversion strength toward 50% capacity
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
    """Add binary risk label based on dock levels.

    is_high_risk = 1 if near-empty (<=2 bikes) or near-full (>=90% utilized).
    """
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
