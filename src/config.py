from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"

# Citi Bike S3 base URL for trip data
CITIBIKE_S3_BASE = "https://s3.amazonaws.com/tripdata"

# Months to download (YYYYMM format) -- Apr 2025 through Mar 2026
MONTHS = [
    "202504", "202505", "202506", "202507", "202508", "202509",
    "202510", "202511", "202512", "202601", "202602", "202603",
]

# Station selection: bounding box for Manhattan below Central Park
STATION_BOUNDS = {
    "lat_min": 40.70,
    "lat_max": 40.77,
    "lon_min": -74.02,
    "lon_max": -73.96,
}

# Minimum trips threshold to include a station
MIN_TRIPS_THRESHOLD = 1000

# Max number of top stations to keep (by trip count)
MAX_STATIONS = 50

# Train/val/test split ratios (chronological)
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Risk thresholds for Task 2
NEAR_EMPTY_THRESHOLD = 2       # bikes
NEAR_FULL_RATIO = 0.90         # fraction of capacity

# Default station capacity (when real data unavailable)
DEFAULT_STATION_CAPACITY = 35

# Forecast horizons (in hours)
FORECAST_HORIZONS = [2, 4, 6]

# Weather data location (NYC Central Park approximate coordinates)
WEATHER_LAT = 40.73
WEATHER_LON = -73.99
