# Citi Bike Station Demand - ML Pipeline

Predict hourly station-level demand, classify risk of dock overflow/emptiness, and prioritize stations for capacity intervention using NYC Citi Bike trip data (Jan-Mar 2026, 30 stations in lower Manhattan).

## Tasks

| Task | Goal | Models |
|------|------|--------|
| 1 - Demand Forecasting | Predict hourly departures per station | Ridge, RandomForest, XGBoost, skforecast MultiSeries |
| 2 - Risk Classification | Classify high-risk dock states | LogisticRegression, RandomForest, XGBoost |
| 3 - Capacity Prioritization | Rank stations by urgency | Composite priority score |

## Results

| Model | MAE | RMSE | MAPE |
|-------|-----|------|------|
| XGBoost | 3.56 | 5.38 | 42.5% |
| RandomForest | 3.65 | 5.61 | 43.4% |
| Ridge | 4.39 | 6.93 | 60.2% |
| skforecast-MultiSeries | 4.76 | 7.37 | -- |
| Naive | 5.65 | 8.93 | 67.7% |
| HistAvg | 7.14 | 11.34 | 56.9% |

| Model | Accuracy | F1 | Precision | Recall |
|-------|----------|-----|-----------|--------|
| XGBoost | 0.881 | 0.609 | 0.554 | 0.677 |
| RandomForest | 0.894 | 0.602 | 0.622 | 0.583 |
| LogisticRegression | 0.819 | 0.448 | 0.385 | 0.535 |

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
