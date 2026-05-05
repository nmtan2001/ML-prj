"""Fetch historical weather data from Open-Meteo API."""

import pandas as pd
import openmeteo_requests
from src.config import WEATHER_LAT, WEATHER_LON


def fetch_weather(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch hourly weather data for the configured location.
    """
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": WEATHER_LAT,
        "longitude": WEATHER_LON,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ["temperature_2m", "relative_humidity_2m", "precipitation", "wind_speed_10m"],
        "timezone": "America/New_York",
    }

    om = openmeteo_requests.Client()
    responses = om.weather_api(url, params=params)
    response = responses[0]

    hourly = response.Hourly()
    time_range = pd.date_range(
        start=pd.to_datetime(hourly.Time(), unit="s"),
        end=pd.to_datetime(hourly.TimeEnd(), unit="s"),
        freq="h",
        inclusive="left",
    )

    df = pd.DataFrame({
        "hour": time_range,
        "temperature": hourly.Variables(0).ValuesAsNumpy(),
        "humidity": hourly.Variables(1).ValuesAsNumpy(),
        "precipitation": hourly.Variables(2).ValuesAsNumpy(),
        "wind_speed": hourly.Variables(3).ValuesAsNumpy(),
    })

    # Current-hour weather is available at prediction time (exogenous)
    # Add derived features
    df["is_precipitating"] = (df["precipitation"] > 0).astype(int)

    return df


def fetch_and_cache_weather(start_date: str, end_date: str, cache_path=None) -> pd.DataFrame:
    """
    Fetch weather data with parquet caching.
    Validates cached date range covers requested range; re-fetches if stale.
    """
    from src.config import DATA_PROCESSED

    if cache_path is None:
        cache_path = DATA_PROCESSED / "weather.parquet"

    if cache_path.exists():
        cached = pd.read_parquet(cache_path)
        cached_start = pd.to_datetime(cached["hour"].min())
        cached_end = pd.to_datetime(cached["hour"].max())
        req_start = pd.to_datetime(start_date)
        req_end = pd.to_datetime(end_date)
        if cached_start <= req_start and cached_end >= req_end:
            print(f"Loading cached weather data from {cache_path}")
            return cached
        print(f"Cached weather range [{cached_start.date()}, {cached_end.date()}] "
              f"does not cover [{req_start.date()}, {req_end.date()}]. Re-fetching.")

    print(f"Fetching weather data from Open-Meteo ({start_date} to {end_date})...")
    df = fetch_weather(start_date, end_date)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path, index=False)
    print(f"Weather data cached to {cache_path}")
    return df


if __name__ == "__main__":
    df = fetch_and_cache_weather("2025-04-01", "2026-03-31")
    print(df.head())
    print(df.describe())
