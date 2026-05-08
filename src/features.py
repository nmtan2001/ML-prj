"""Feature engineering for Citi Bike demand and risk models."""

import numpy as np
import pandas as pd
import holidays as hol
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add hour, day_of_week, cyclical encodings, and rush hour flag."""
    df = df.copy()
    df["hour_of_day"] = df["hour"].dt.hour
    df["day_of_week"] = df["hour"].dt.dayofweek
    df["month"] = df["hour"].dt.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_daylight"] = ((df["hour_of_day"] >= 6) & (df["hour_of_day"] <= 19)).astype(int)
    # Rush hour only on weekdays
    df["is_rush_hour"] = (
        df["hour_of_day"].isin([7, 8, 9, 17, 18, 19]) & (df["is_weekend"] == 0)
    ).astype(int)

    # Cyclical encodings
    df["hour_sin"] = np.sin(2 * np.pi * df["hour_of_day"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour_of_day"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # Key interactions
    df["weekend_x_hour"] = df["is_weekend"] * df["hour_of_day"]
    df["rush_x_weekend"] = df["is_rush_hour"] * df["is_weekend"]

    return df


def add_lag_features(df: pd.DataFrame, target: str = "departures") -> pd.DataFrame:
    """Add lag features: t-1h, t-2h, t-3h, t-24h, t-168h."""
    df = df.sort_values(["station_id", "hour"]).copy()
    for lag in [1, 2, 3, 12, 24, 48, 168, 336, 672]:
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


def add_seasonal_lag_features(df: pd.DataFrame, target: str = "departures") -> pd.DataFrame:
    """Add same-hour, same-day-of-week rolling averages over recent weeks."""
    df = df.sort_values(["station_id", "hour"]).copy()
    # Same hour of day, rolling over previous same-hour values
    for window_days in [3, 7]:
        col = f"{target}_same_hour_roll_mean_{window_days}d"
        rolled = df.groupby(["station_id", "hour_of_day"])[target].transform(
            lambda s: s.shift(1).rolling(window=window_days, min_periods=1).mean()
        )
        df[col] = rolled
    return df


def add_holiday_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add US NY-state holiday flags: is_holiday, is_day_before_holiday, is_day_after_holiday."""
    df = df.copy()
    years = df["hour"].dt.year.unique()
    ny_holidays = hol.US(state="NY", years=sorted(years))
    holiday_dates = set(ny_holidays.keys())

    dates = df["hour"].dt.date
    df["is_holiday"] = dates.isin(holiday_dates).astype(int)
    df["is_day_before_holiday"] = dates.map(
        lambda d: (d + pd.Timedelta(days=1)) in holiday_dates
    ).astype(int)
    df["is_day_after_holiday"] = dates.map(
        lambda d: (d - pd.Timedelta(days=1)) in holiday_dates
    ).astype(int)
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


def add_spatial_clusters(df: pd.DataFrame, n_clusters: int = 8) -> pd.DataFrame:
    """Cluster stations by normalized hourly demand profile for shared learning."""
    df = df.copy()

    # Use only training period for clustering to avoid leakage
    timestamps = df["hour"].sort_values().unique()
    cutoff = timestamps[int(len(timestamps) * 0.70)]
    train_df = df[df["hour"] < cutoff]

    profile = train_df.pivot_table(
        index="station_id", columns=train_df["hour"].dt.hour,
        values="departures", aggfunc="mean",
    ).fillna(0)
    norms = np.linalg.norm(profile.values, axis=1, keepdims=True)
    norms[norms == 0] = 1
    profile_normed = profile.values / norms

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(profile_normed)
    cluster_map = dict(zip(profile.index, clusters))
    df["neighborhood_cluster"] = df["station_id"].map(cluster_map).astype(int)

    # Compute hourly cluster mean, shift it, then merge back
    df = df.sort_values(["station_id", "hour"])
    cluster_hourly = df.groupby(["neighborhood_cluster", "hour"])["departures"].mean().reset_index()
    cluster_hourly = cluster_hourly.rename(columns={"departures": "_cluster_mean"})
    cluster_hourly = cluster_hourly.sort_values(["neighborhood_cluster", "hour"])
    for lag in [1, 24]:
        col = f"cluster_departures_mean_lag_{lag}h"
        cluster_hourly[f"_shifted_{lag}"] = cluster_hourly.groupby("neighborhood_cluster")["_cluster_mean"].shift(lag)
        merge_col = cluster_hourly[["neighborhood_cluster", "hour", f"_shifted_{lag}"]].rename(columns={f"_shifted_{lag}": col})
        df = df.merge(merge_col, on=["neighborhood_cluster", "hour"], how="left")

    return df


def add_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    """Merge weather features from cached parquet file with enhanced features."""
    from src.config import DATA_PROCESSED

    weather_path = DATA_PROCESSED / "weather.parquet"
    if not weather_path.exists():
        print("Weather data not found. Run src/weather.py first. Skipping weather features.")
        return df

    weather = pd.read_parquet(weather_path)
    weather = weather.sort_values("hour")

    # Precipitation rolling sums before merge
    weather["precip_roll_sum_3h"] = weather["precipitation"].rolling(3, min_periods=1).sum()
    weather["precip_roll_sum_6h"] = weather["precipitation"].rolling(6, min_periods=1).sum()

    df = df.merge(weather, on="hour", how="left")
    weather_cols = [c for c in weather.columns if c != "hour"]
    df[weather_cols] = df[weather_cols].ffill().bfill()

    # Apparent temperature (wind chill / heat index approximation)
    df["apparent_temp"] = df["temperature"] - (0.1 * df["wind_speed"]) + (0.05 * df["humidity"])

    # Weather interactions
    if "precipitation" in df.columns and "is_weekend" in df.columns:
        df["precipitation_x_weekend"] = df["precipitation"] * df["is_weekend"]
    if "precipitation" in df.columns and "is_rush_hour" in df.columns:
        df["precipitation_x_rush"] = df["precipitation"] * df["is_rush_hour"]
    if "temperature" in df.columns and "hour_sin" in df.columns:
        df["temp_x_hour_sin"] = df["temperature"] * df["hour_sin"]
    if "temperature" in df.columns and "humidity" in df.columns:
        df["temp_x_humidity"] = df["temperature"] * df["humidity"]

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


def add_spatial_lag_features(
    df: pd.DataFrame, target: str = "departures", k: int = 5, lag_hours: list = None,
) -> pd.DataFrame:
    """Add KNN spatial lag features using nearest-station demand at lagged hours."""
    if lag_hours is None:
        lag_hours = [1, 3, 24]

    df = df.copy()
    station_coords = df.groupby("station_id")[["latitude", "longitude"]].first()
    if len(station_coords) < k + 1:
        k = max(1, len(station_coords) - 1)

    # Find K nearest neighbors per station
    nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
    nn.fit(station_coords.values)
    _, indices = nn.kneighbors(station_coords.values)
    # Exclude self (index 0 is always self)
    neighbor_map = {
        sid: station_coords.index[indices[i, 1:]].tolist()
        for i, sid in enumerate(station_coords.index)
    }

    # Pivot target to wide form (hour x station) with hourly frequency
    wide = df.pivot_table(index="hour", columns="station_id", values=target, aggfunc="sum").sort_index()
    wide = wide.asfreq("h").sort_index()

    for lag in lag_hours:
        shifted = wide.shift(lag)
        mean_col = f"knn_{target}_mean_lag_{lag}h"

        mean_vals = np.full(len(df), np.nan)

        for sid, neighbors in neighbor_map.items():
            mask = df["station_id"] == sid
            hours = df.loc[mask, "hour"]
            if not neighbors or sid not in shifted.columns:
                continue
            neighbor_data = shifted[neighbors].loc[hours.values]
            mean_vals[mask.values] = neighbor_data.mean(axis=1, skipna=True).values

        df[mean_col] = mean_vals

    return df


def build_features(hourly: pd.DataFrame, target: str = "departures") -> pd.DataFrame:
    """
    Run full feature engineering pipeline.
    """
    print("Adding temporal features...")
    df = add_temporal_features(hourly)

    print("Adding holiday features...")
    df = add_holiday_features(df)

    print("Adding lag features...")
    df = add_lag_features(df, target=target)

    print("Adding rolling features...")
    df = add_rolling_features(df, target=target)

    print("Adding trend features...")
    df = add_trend_features(df, target=target)

    print("Adding seasonal lag features...")
    df = add_seasonal_lag_features(df, target=target)

    print("Adding cross-station features...")
    df = add_cross_station_features(df)

    print("Adding spatial clusters...")
    df = add_spatial_clusters(df)

    print("Adding KNN spatial lag features...")
    df = add_spatial_lag_features(df, target=target)

    print("Adding weather features...")
    df = add_weather_features(df)

    print("Adding station encoding features...")
    df = add_station_encoding(df)

    # Drop rows with NaN from lag/rolling
    df = df.dropna().reset_index(drop=True)
    print(f"Feature matrix shape: {df.shape}")
    return df
