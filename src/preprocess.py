"""Preprocess raw Citi Bike trip data into hourly station-level observations."""

import pandas as pd

from src.config import (
    DATA_PROCESSED,
    DATA_RAW,
    DEFAULT_STATION_CAPACITY,
    MAX_STATIONS,
    MIN_TRIPS_THRESHOLD,
    MONTHS,
    STATION_BOUNDS,
)


def load_raw_data() -> pd.DataFrame:
    """Load and concatenate all raw CSV files incrementally to limit memory."""
    frames = []
    for month in MONTHS:
        month_dir = DATA_RAW / month
        if not month_dir.exists():
            raise FileNotFoundError(f"Raw data not found for {month}. Run download.py first.")
        for csv_file in sorted(month_dir.glob("*.csv")):
            print(f"  Loading {csv_file.name}...")
            df = pd.read_csv(csv_file, low_memory=False)
            df = clean_column_names(df)
            df = parse_timestamps(df)
            df = filter_stations(df)
            frames.append(df)
    return pd.concat(frames, ignore_index=True)


def preprocess_incremental() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Process months incrementally to limit memory usage.

    First pass: count total activity per station across all months.
    Second pass: filter to selected stations and aggregate to hourly.
    """
    # Pass 1: accumulate station activity counts to select top stations
    print("Pass 1: Counting station activity across all months...")
    start_counts_all = pd.Series(dtype="int64")
    end_counts_all = pd.Series(dtype="int64")
    bounds = STATION_BOUNDS

    for month in MONTHS:
        month_dir = DATA_RAW / month
        if not month_dir.exists():
            raise FileNotFoundError(f"Raw data not found for {month}. Run download.py first.")
        for csv_file in sorted(month_dir.glob("*.csv")):
            print(f"  Scanning {csv_file.name}...")
            df = pd.read_csv(csv_file, low_memory=False)
            df = clean_column_names(df)

            lat_col = "start_lat" if "start_lat" in df.columns else "latitude"
            lon_col = "start_lng" if "start_lng" in df.columns else "longitude"
            in_box = (
                (df[lat_col] >= bounds["lat_min"])
                & (df[lat_col] <= bounds["lat_max"])
                & (df[lon_col] >= bounds["lon_min"])
                & (df[lon_col] <= bounds["lon_max"])
            )
            box_df = df[in_box]
            start_counts_all = start_counts_all.add(box_df["start_station_id"].value_counts(), fill_value=0)
            end_counts_all = end_counts_all.add(
                df[df["end_station_id"].isin(box_df["start_station_id"].unique())]["end_station_id"].value_counts(),
                fill_value=0,
            )
            del df, box_df

    total_activity = start_counts_all.add(end_counts_all, fill_value=0)
    active = total_activity[total_activity >= MIN_TRIPS_THRESHOLD].sort_values(ascending=False)
    if len(active) > MAX_STATIONS:
        active = active.head(MAX_STATIONS)
    selected = set(active.index)
    print(f"  Selected {len(selected)} stations.")

    # Pass 2: aggregate to hourly for selected stations only
    print("Pass 2: Aggregating selected stations to hourly...")
    hourly_frames = []
    station_coords = {}

    for month in MONTHS:
        month_dir = DATA_RAW / month
        for csv_file in sorted(month_dir.glob("*.csv")):
            print(f"  Processing {csv_file.name}...")
            df = pd.read_csv(csv_file, low_memory=False)
            df = clean_column_names(df)
            df = parse_timestamps(df)
            df = df[df["start_station_id"].isin(selected) | df["end_station_id"].isin(selected)].copy()

            df["hour"] = df["started_at"].dt.floor("h")

            # Departures
            df_dep = df[df["start_station_id"].isin(selected)]
            dep = (
                df_dep.groupby(["start_station_id", "hour"])
                .agg(departures=("ride_id", "count"))
                .reset_index()
                .rename(columns={"start_station_id": "station_id"})
            )

            # Capture coordinates
            for sid in selected:
                if sid not in station_coords:
                    rows = df_dep[df_dep["start_station_id"] == sid]
                    if len(rows) > 0:
                        lat_col = "start_lat" if "start_lat" in df.columns else "latitude"
                        lon_col = "start_lng" if "start_lng" in df.columns else "longitude"
                        station_coords[sid] = (rows.iloc[0][lat_col], rows.iloc[0][lon_col])

            # Arrivals
            df_arr = df[df["end_station_id"].isin(selected)]
            arr = (
                df_arr.groupby(["end_station_id", "hour"])
                .agg(arrivals=("ride_id", "count"))
                .reset_index()
                .rename(columns={"end_station_id": "station_id"})
            )

            merged = dep.merge(arr, on=["station_id", "hour"], how="outer").fillna(0)
            merged["departures"] = merged["departures"].astype(int)
            merged["arrivals"] = merged["arrivals"].astype(int)
            hourly_frames.append(merged)
            del df, df_dep, df_arr, dep, arr, merged

    hourly = pd.concat(hourly_frames, ignore_index=True)
    hourly = hourly.groupby(["station_id", "hour"]).sum().reset_index()
    hourly["net_flow"] = hourly["arrivals"] - hourly["departures"]
    hourly["latitude"] = hourly["station_id"].map(lambda s: station_coords.get(s, (0, 0))[0])
    hourly["longitude"] = hourly["station_id"].map(lambda s: station_coords.get(s, (0, 0))[1])

    station_info = build_station_info(hourly)

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    hourly.to_parquet(DATA_PROCESSED / "hourly.parquet", index=False)
    station_info.to_parquet(DATA_PROCESSED / "station_info.parquet", index=False)

    print(f"Saved {len(hourly)} hourly records for {station_info.shape[0]} stations.")
    return hourly, station_info


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names to snake_case."""
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("(", "")
        .str.replace(")", "")
    )
    return df


def parse_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """Parse started_at and ended_at to datetime."""
    # Handle both old and new column naming conventions
    start_col = "started_at" if "started_at" in df.columns else "start_time"
    end_col = "ended_at" if "ended_at" in df.columns else "stop_time"

    df[start_col] = pd.to_datetime(df[start_col], errors="coerce")
    df[end_col] = pd.to_datetime(df[end_col], errors="coerce")
    df = df.rename(columns={start_col: "started_at", end_col: "ended_at"})
    return df


def filter_stations(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to stations within the configured bounding box.

    Matches trips where either the start OR end station is in the bounding box.
    """
    lat_col = "start_lat" if "start_lat" in df.columns else "latitude"
    lon_col = "start_lng" if "start_lng" in df.columns else "longitude"
    end_lat_col = "end_lat" if "end_lat" in df.columns else None

    bounds = STATION_BOUNDS
    start_in_box = (
        (df[lat_col] >= bounds["lat_min"])
        & (df[lat_col] <= bounds["lat_max"])
        & (df[lon_col] >= bounds["lon_min"])
        & (df[lon_col] <= bounds["lon_max"])
    )
    if end_lat_col and end_lat_col in df.columns:
        end_lon_col = "end_lng" if "end_lng" in df.columns else None
        end_in_box = (
            (df[end_lat_col] >= bounds["lat_min"])
            & (df[end_lat_col] <= bounds["lat_max"])
            & (df[end_lon_col] >= bounds["lon_min"])
            & (df[end_lon_col] <= bounds["lon_max"])
        )
        return df[start_in_box | end_in_box].copy()
    return df[start_in_box].copy()


def select_active_stations(df: pd.DataFrame) -> tuple[pd.DataFrame, set]:
    """Identify target stations and filter data to trips touching them.

    Selects by total activity (departures + arrivals) to get balanced stations.
    Returns filtered dataframe and the set of selected station IDs.
    """
    start_counts = df["start_station_id"].value_counts()
    end_counts = df["end_station_id"].value_counts()
    total_activity = start_counts.add(end_counts, fill_value=0)

    # Filter by bounding box stations only
    bounds = STATION_BOUNDS
    lat_col = "start_lat" if "start_lat" in df.columns else "latitude"
    lon_col = "start_lng" if "start_lng" in df.columns else "longitude"
    box_stations = set(
        df[
            (df[lat_col] >= bounds["lat_min"])
            & (df[lat_col] <= bounds["lat_max"])
            & (df[lon_col] >= bounds["lon_min"])
            & (df[lon_col] <= bounds["lon_max"])
        ]["start_station_id"].unique()
    )
    total_activity = total_activity[total_activity.index.isin(box_stations)]

    active = total_activity[total_activity >= MIN_TRIPS_THRESHOLD].sort_values(ascending=False)
    if len(active) > MAX_STATIONS:
        active = active.head(MAX_STATIONS)
    selected = set(active.index)

    # Keep trips where start OR end is at a selected station
    df = df[df["start_station_id"].isin(selected) | df["end_station_id"].isin(selected)].copy()
    return df, selected


def aggregate_to_hourly(df: pd.DataFrame, selected_stations: set) -> pd.DataFrame:
    """Aggregate trips to hourly departures and arrivals per station.

    Only aggregates for stations in selected_stations.
    """
    df["hour"] = df["started_at"].dt.floor("h")

    # Departures from selected stations
    df_dep = df[df["start_station_id"].isin(selected_stations)]
    departures = (
        df_dep.groupby(["start_station_id", "hour"])
        .agg(
            departures=("ride_id", "count"),
            latitude=("start_lat", "first"),
            longitude=("start_lng", "first"),
        )
        .reset_index()
        .rename(columns={"start_station_id": "station_id"})
    )

    # Arrivals at selected stations (from ANY origin, not just selected)
    df_arr = df[df["end_station_id"].isin(selected_stations)]
    arrivals = (
        df_arr.groupby(["end_station_id", "hour"])
        .agg(arrivals=("ride_id", "count"))
        .reset_index()
        .rename(columns={"end_station_id": "station_id"})
    )

    hourly = departures.merge(arrivals, on=["station_id", "hour"], how="outer").fillna(0)
    hourly["departures"] = hourly["departures"].astype(int)
    hourly["arrivals"] = hourly["arrivals"].astype(int)
    hourly["net_flow"] = hourly["arrivals"] - hourly["departures"]
    hourly["latitude"] = hourly.groupby("station_id")["latitude"].transform("first")
    hourly["longitude"] = hourly.groupby("station_id")["longitude"].transform("first")

    return hourly


def build_station_info(hourly: pd.DataFrame) -> pd.DataFrame:
    """Build a static station info table (id, lat, lon, avg_daily_demand)."""
    station_info = (
        hourly.groupby("station_id")
        .agg(
            latitude=("latitude", "first"),
            longitude=("longitude", "first"),
            avg_daily_demand=("departures", lambda x: x.sum() / max(hourly.loc[x.index, "hour"].dt.date.nunique(), 1)),
        )
        .reset_index()
    )
    station_info["capacity"] = DEFAULT_STATION_CAPACITY
    return station_info


def preprocess() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run full preprocessing pipeline using incremental loading.

    Returns:
        Tuple of (hourly_data, station_info) DataFrames.
    """
    return preprocess_incremental()


if __name__ == "__main__":
    preprocess()
