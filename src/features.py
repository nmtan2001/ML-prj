"""Feature engineering for Citi Bike demand and risk models."""

import numpy as np
import pandas as pd


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add hour, day_of_week, cyclical encodings, and rush hour flag."""
    df = df.copy()
    df["hour_of_day"] = df["hour"].dt.hour
    df["day_of_week"] = df["hour"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    # Rush hour only on weekdays
    df["is_rush_hour"] = (
        df["hour_of_day"].isin([7, 8, 9, 17, 18, 19]) & (df["is_weekend"] == 0)
    ).astype(int)

    # Cyclical encodings
    df["hour_sin"] = np.sin(2 * np.pi * df["hour_of_day"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour_of_day"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

    # Key interactions
    df["weekend_x_hour"] = df["is_weekend"] * df["hour_of_day"]
    df["rush_x_weekend"] = df["is_rush_hour"] * df["is_weekend"]

    return df


def add_lag_features(df: pd.DataFrame, target: str = "departures") -> pd.DataFrame:
    """Add lag features: t-1h, t-2h, t-3h, t-24h, t-168h."""
    df = df.sort_values(["station_id", "hour"]).copy()
    for lag in [1, 2, 3, 12, 24, 48, 168]:
        col = f"{target}_lag_{lag}h"
        df[col] = df.groupby("station_id")[target].shift(lag)
    # Also lag dock-level features for risk classification
    for col in ["bikes_available", "dock_utilization"]:
        if col in df.columns:
            df[f"{col}_lag_1h"] = df.groupby("station_id")[col].shift(1)
    return df


def add_rolling_features(df: pd.DataFrame, target: str = "departures") -> pd.DataFrame:
    """Add rolling mean and std over recent windows, shifted by 1 to avoid leakage."""
    df = df.sort_values(["station_id", "hour"]).copy()
    for window in [3, 6, 12, 24]:
        rolled = df.groupby("station_id")[target].transform(
            lambda s: s.shift(1).rolling(window=window, min_periods=1).mean()
        )
        df[f"{target}_roll_mean_{window}h"] = rolled

        rolled_std = df.groupby("station_id")[target].transform(
            lambda s: s.shift(1).rolling(window=window, min_periods=1).std()
        )
        df[f"{target}_roll_std_{window}h"] = rolled_std
    return df


def add_trend_features(df: pd.DataFrame, target: str = "departures") -> pd.DataFrame:
    """Add lagged difference (t-1 minus t-2) to avoid leakage."""
    df = df.sort_values(["station_id", "hour"]).copy()
    df[f"{target}_diff_1h"] = df.groupby("station_id")[target].diff(1).shift(1)
    return df


def add_cross_station_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add area-level demand features across nearby stations, shifted by 1 to avoid leakage."""
    df = df.copy()
    area = df.groupby("hour").agg(
        area_total_departures=("departures", "sum"),
        area_total_arrivals=("arrivals", "sum"),
        area_active_stations=("departures", lambda x: (x > 0).sum()),
    ).reset_index()
    area["area_total_departures"] = area["area_total_departures"].shift(1)
    area["area_total_arrivals"] = area["area_total_arrivals"].shift(1)
    area["area_active_stations"] = area["area_active_stations"].shift(1)
    df = df.merge(area, on="hour", how="left")
    return df


def add_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    """Merge weather features from cached parquet file."""
    from src.config import DATA_PROCESSED

    weather_path = DATA_PROCESSED / "weather.parquet"
    if not weather_path.exists():
        print("Weather data not found. Run src/weather.py first. Skipping weather features.")
        return df

    weather = pd.read_parquet(weather_path)
    df = df.merge(weather, on="hour", how="left")
    # Fill any missing weather with forward fill
    weather_cols = [c for c in weather.columns if c != "hour"]
    df[weather_cols] = df[weather_cols].ffill().bfill()
    # Weather interactions
    if "precipitation" in df.columns and "is_weekend" in df.columns:
        df["precipitation_x_weekend"] = df["precipitation"] * df["is_weekend"]
    return df


def add_station_encoding(df: pd.DataFrame) -> pd.DataFrame:
    """Add target-encoded station features computed from training-period data only.

    station_mean_demand: average departures per station (smoothed toward global mean).
    station_hour_mean: average departures per (station, hour_of_day).
    station_weekend_ratio: ratio of weekend to weekday avg demand per station.
    """
    # Use first 70% of time range for encoding to avoid leakage
    timestamps = df["hour"].sort_values().unique()
    cutoff = timestamps[int(len(timestamps) * 0.70)]
    train_mask = df["hour"] < cutoff
    train_df = df[train_mask]

    global_mean = train_df["departures"].mean()

    # Smoothed target encoding for station
    station_stats = train_df.groupby("station_id")["departures"].agg(["mean", "count"])
    smoothing = 50
    station_stats["station_mean_demand"] = (
        (station_stats["count"] * station_stats["mean"] + smoothing * global_mean)
        / (station_stats["count"] + smoothing)
    )
    df = df.merge(
        station_stats[["station_mean_demand"]].reset_index(),
        on="station_id", how="left",
    )

    # Station x hour target encoding
    station_hour = train_df.groupby(["station_id", "hour_of_day"])["departures"].mean()
    station_hour_map = station_hour.to_dict()
    df["station_hour_mean"] = df.apply(
        lambda r: station_hour_map.get((r["station_id"], r["hour_of_day"]), global_mean),
        axis=1,
    )

    # Station weekend ratio
    weekend_demand = train_df[train_df["is_weekend"] == 1].groupby("station_id")["departures"].mean()
    weekday_demand = train_df[train_df["is_weekend"] == 0].groupby("station_id")["departures"].mean()
    ratio = (weekend_demand / weekday_demand).fillna(1.0)
    df = df.merge(
        ratio.rename("station_weekend_ratio").reset_index(),
        on="station_id", how="left",
    )
    df["station_weekend_ratio"] = df["station_weekend_ratio"].fillna(1.0)

    return df


def build_features(hourly: pd.DataFrame, target: str = "departures") -> pd.DataFrame:
    """
    Run full feature engineering pipeline.
    """
    print("Adding temporal features...")
    df = add_temporal_features(hourly)

    print("Adding lag features...")
    df = add_lag_features(df, target=target)

    print("Adding rolling features...")
    df = add_rolling_features(df, target=target)

    print("Adding trend features...")
    df = add_trend_features(df, target=target)

    print("Adding cross-station features...")
    df = add_cross_station_features(df)

    print("Adding weather features...")
    df = add_weather_features(df)

    print("Adding station encoding features...")
    df = add_station_encoding(df)

    # Drop rows with NaN from lag/rolling
    df = df.dropna().reset_index(drop=True)
    print(f"Feature matrix shape: {df.shape}")
    return df
