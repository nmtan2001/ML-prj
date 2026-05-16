# Citi Bike Station Demand - ML Pipeline

Predict hourly station-level demand, classify risk of dock overflow/emptiness, and prioritize stations for capacity intervention using NYC Citi Bike trip data (Apr 2025-Mar 2026, 50 stations in lower Manhattan).

## Tasks

| Task | Goal | Models |
|------|------|--------|
| 1 - Demand Forecasting | Predict hourly departures per station | Ridge, RandomForest, XGBoost, LightGBM, skforecast MultiSeries, Ensemble |
| 2 - Risk Classification | Classify high-risk dock states | LogisticRegression, RandomForest, XGBoost, LightGBM |
| 3 - Capacity Prioritization | Rank stations by urgency | Composite priority score |

## Results

### Task 1: Demand Forecasting

| Model | MAE | RMSE | WAPE |
|-------|-----|------|------|
| LightGBM | 2.81 | 4.26 | 31.1% |
| Ensemble-Blend | 2.82 | 4.27 | 31.2% |
| XGBoost | 2.88 | 4.36 | 31.8% |
| RandomForest | 2.95 | 4.49 | 32.7% |
| Ridge | 3.42 | 5.03 | 37.9% |
| skforecast-MultiSeries | 3.75 | 5.46 | -- |
| Naive | 4.31 | 6.74 | 47.7% |
| HistAvg | 8.91 | 11.68 | 98.6% |

### Task 2: Risk Classification

| Model | Accuracy | F1 | Precision | Recall |
|-------|----------|-----|-----------|--------|
| LightGBM | 0.933 | 0.614 | 0.616 | 0.611 |
| XGBoost | 0.935 | 0.607 | 0.634 | 0.582 |
| RandomForest | 0.936 | 0.599 | 0.653 | 0.553 |
| XGBoost-FocalLoss | 0.937 | 0.574 | 0.693 | 0.490 |
| LogisticRegression | 0.867 | 0.406 | 0.331 | 0.525 |

## Feature Engineering

70+ features engineered across five categories. All lag/rolling features are shifted to prevent temporal leakage.

| Category | Features |
|----------|----------|
| **Temporal** | Hour, day-of-week, month; cyclical sin/cos encodings; Fourier terms (24h, 168h, yearly); rush hour & weekend flags |
| **Lag** | t-1h, 2h, 3h, 12h, 24h, 48h, 168h, 336h, 672h; same-hour rolling mean (3d, 7d); first-order difference |
| **Rolling Stats** | Mean & std over 3h, 6h, 12h, 24h windows (shifted by 1); trend via lagged diff |
| **Weather** | Temperature, humidity, precipitation, wind speed, apparent temperature; weather x time interactions |
| **Spatial / Station** | KNN lag features (k=5); KMeans clusters (k=8) with cluster mean demand lags; smoothed target encoding; station x hour mean demand; weekend ratio per station; IsolationForest anomaly score |

Holiday flags for NY state (including day-before/after). Target encoding computed on training data only.

## Interpretation & Key Findings

### Demand Patterns

- **Seasonality:** 3.4x swing between peak (Sep: +60% above annual avg) and off-peak (Feb: -53%). Peak season runs Jun-Oct (40-60% above average).
- **Rush hours:** Evening rush is 76% stronger than morning (26.7 vs 15.2 departures/hr), reflecting commuter one-way patterns -- residential in AM, business district in PM.
- **Model insight:** Lag and temporal features are most predictive; weather features provide marginal additional gain.

### Risk Hotspots

High-risk = near-empty (<=2 bikes) OR near-full (>=90% dock utilization). 13.2% of station-hours are at risk overall.

| Station | % Hours at Risk | Imbalance Type |
|---------|-----------------|----------------|
| 5980.10 | 33.8% | Mostly near-full |
| 6756.01 | 28.7% | Mixed |
| 6602.03 | 24.9% | Mostly near-empty |
| 6173.08 | 24.4% | Mixed |
| 6233.04 | 22.9% | Mostly near-full |

23,701 near-empty hours (bikes unavailable) and 28,572 near-full hours (cannot return bikes). Estimated ~$1.3M unrealized revenue at $4.50/trip, ~30% reducible via proactive rebalancing.

### Capacity Prioritization (Top 5)

| Priority | Station | Score | Demand | Risk Freq | Strain |
|----------|---------|-------|--------|-----------|--------|
| #1 | 6233.04 | 0.884 | 14.4/hr | 23.0% | 0.41 |
| #2 | 6140.05 | 0.846 | 14.0/hr | 21.0% | 0.40 |
| #3 | 6331.01 | 0.686 | 10.7/hr | 20.6% | 0.30 |
| #4 | 5788.13 | 0.685 | 11.9/hr | 18.1% | 0.34 |
| #5 | 6492.08 | 0.665 | 10.8/hr | 21.9% | 0.31 |

Top 2 stations score >0.84 and are clear expansion candidates -- 2x average demand with the same 35-dock capacity. Bottom 10 stations score <0.15.

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
