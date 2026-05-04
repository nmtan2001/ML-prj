# Citi Bike Station Demand - ML Pipeline

Predict hourly station-level demand, classify risk of dock overflow/emptiness, and prioritize stations for capacity intervention using NYC Citi Bike trip data (Jan-Mar 2026, 30 stations in lower Manhattan).

## Tasks

| Task | Goal | Models |
|------|------|--------|
| 1 - Demand Forecasting | Predict hourly departures per station | Ridge, RandomForest, XGBoost, skforecast MultiSeries |
| 2 - Risk Classification | Classify high-risk dock states | LogisticRegression, RandomForest, XGBoost |
| 3 - Capacity Prioritization | Rank stations by urgency | Composite priority score |

## Results

| Model | MAE | RMSE | R2 |
|-------|-----|------|----|
| skforecast-MultiSeries | 3.67 | 5.64 | - |
| XGBoost | 3.96 | 6.53 | - |
| RandomForest | 4.09 | 6.74 | - |

| Model | Accuracy | F1 | Precision | Recall |
|-------|----------|-----|-----------|--------|
| RandomForest | 0.888 | 0.590 | 0.595 | 0.586 |
| XGBoost | 0.872 | 0.577 | 0.529 | 0.635 |
| LogisticRegression | 0.777 | 0.429 | 0.330 | 0.610 |

## Project Structure

```
citibike-ml/
  main.py                  # Orchestration pipeline
  src/
    config.py              # Paths and constants
    download.py            # Download Citi Bike CSV data
    preprocess.py          # Load, clean, aggregate to hourly
    simulate_docks.py      # Simulate dock inventory and risk labels
    features.py            # Temporal, lag, rolling, weather, station encoding
    weather.py             # Fetch weather from Open-Meteo API
    evaluate.py            # Metrics (MAE, RMSE, MAPE, F1, Precision, Recall)
    models/
      demand.py            # Demand forecasting models + skforecast
      risk.py              # Risk classification with threshold tuning
      prioritize.py        # Priority score computation
  notebooks/
    01_eda.ipynb           # Exploratory data analysis
    02_results.ipynb       # Model results and comparison plots
  data/
    raw/                   # Original Citi Bike CSVs
    processed/             # Parquet caches (hourly, station_info, weather)
```

## Setup

```bash
uv sync
```

## Usage

```bash
# Run full pipeline
uv run python main.py

# Fetch weather data
uv run python src/weather.py

# Run notebooks
uv run jupyter nbconvert --to notebook --execute notebooks/01_eda.ipynb --output 01_eda.ipynb
uv run jupyter nbconvert --to notebook --execute notebooks/02_results.ipynb --output 02_results.ipynb
```

## Tech Stack

- **Data:** pandas, pyarrow, openmeteo-requests
- **ML:** scikit-learn, xgboost, skforecast 0.19
- **Visualization:** matplotlib, seaborn
- **Environment:** uv, Python 3.12, Jupyter

## Key Design Decisions

- **Global timestamp split** -- train/val/test uses a single date cutoff across all stations, preventing temporal leakage
- **Threshold tuning** -- classification thresholds optimized on validation set (not default 0.5)
- **Per-station exogenous features** for skforecast -- station-specific lag/rolling features instead of area averages
- **Current-hour weather** -- exogenous weather is available at prediction time, no lag needed
- **Target encoding** -- station-level demand profiles (mean, hour-mean, weekend ratio) computed on training data only
