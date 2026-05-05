# Citi Bike Station Demand - ML Pipeline

Predict hourly station-level demand, classify risk of dock overflow/emptiness, and prioritize stations for capacity intervention using NYC Citi Bike trip data (Apr 2025-Mar 2026, 50 stations in lower Manhattan).

## Tasks

| Task | Goal | Models |
|------|------|--------|
| 1 - Demand Forecasting | Predict hourly departures per station | Ridge, RandomForest, XGBoost, LightGBM, skforecast MultiSeries, Ensemble |
| 2 - Risk Classification | Classify high-risk dock states | LogisticRegression, RandomForest, XGBoost, LightGBM |
| 3 - Capacity Prioritization | Rank stations by urgency | Composite priority score |

## Results

| Model | MAE | RMSE | MAPE |
|-------|-----|------|------|
| Ensemble-Blend | 2.76 | 4.12 | 49.7% |
| XGBoost | 2.78 | 4.13 | 50.0% |
| LightGBM | 2.79 | 4.14 | 50.5% |
| RandomForest | 2.85 | 4.27 | 51.5% |
| Ridge | 3.86 | 5.47 | 83.3% |
| skforecast-MultiSeries | 3.88 | 5.67 | -- |
| Naive | 4.23 | 6.65 | 71.1% |
| HistAvg | 8.78 | 11.55 | 220.6% |

| Model | Accuracy | F1 | Precision | Recall |
|-------|----------|-----|-----------|--------|
| LightGBM | 0.925 | 0.599 | 0.549 | 0.659 |
| XGBoost | 0.925 | 0.599 | 0.551 | 0.655 |
| RandomForest | 0.923 | 0.578 | 0.539 | 0.623 |
| LogisticRegression | 0.853 | 0.399 | 0.306 | 0.575 |

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
      ensemble.py          # Inverse-MAE weighted ensemble blending
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
- **ML:** scikit-learn, xgboost, lightgbm, skforecast 0.19
- **Visualization:** matplotlib, seaborn
- **Environment:** uv, Python 3.12, Jupyter

## Key Design Decisions

- **Global timestamp split** -- train/val/test uses a single date cutoff across all stations, preventing temporal leakage
- **GridSearchCV** -- TimeSeriesSplit CV for hyperparameter tuning on all models
- **Threshold tuning** -- classification thresholds optimized on validation set (not default 0.5)
- **Inverse-MAE ensemble** -- weighted average of top 3 models, avoids Ridge stacking overfitting
- **Per-station exogenous features** for skforecast -- station-specific encoding features
- **Holiday features** -- NY state holidays, day-before/after flags
- **Current-hour weather** -- exogenous weather is available at prediction time, no lag needed
- **Target encoding** -- station-level demand profiles (mean, hour-mean, weekend ratio) computed on training data only
