"""Main pipeline: orchestrate preprocess, feature engineering, modeling."""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from preprocess import preprocess
from simulate_docks import label_risk, simulate_inventory, simulate_and_label_cached
from features import build_features
from models.demand import train_demand_models, train_multi_target_models, time_split, FEATURE_COLS_DEMAND
from models.risk import train_risk_models
from models.ensemble import train_ensemble
from models.prioritize import compute_priority_score
from evaluate import (
    plot_model_comparison,
    plot_confusion_matrix,
    plot_predictions,
    plot_temporal_patterns,
    plot_weekly_trend,
    plot_top_stations,
    plot_station_map,
    plot_net_flow,
    plot_dock_utilization,
    plot_risk_analysis,
    plot_correlation_heatmap,
    plot_demand_comparison,
    plot_all_risk_confusion,
    plot_prioritization,
)
import matplotlib.pyplot as plt


def _save(fig: plt.Figure, path: str):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    # Step 1: Preprocess raw data
    print("=" * 60)
    print("STEP 1: Preprocessing")
    print("=" * 60)
    hourly, station_info = preprocess()

    # Step 1.5: Fetch weather data
    from weather import fetch_and_cache_weather
    date_range = hourly["hour"].agg(["min", "max"])
    fetch_and_cache_weather(
        date_range["min"].strftime("%Y-%m-%d"),
        date_range["max"].strftime("%Y-%m-%d"),
    )

    # Step 2: Simulate dock levels and label risk
    print("\n" + "=" * 60)
    print("STEP 2: Simulating dock inventory")
    print("=" * 60)
    hourly = simulate_and_label_cached(hourly, station_info)

    # EDA plots
    _save(plot_temporal_patterns(hourly), "eda_temporal_patterns.png")
    _save(plot_weekly_trend(hourly), "eda_weekly_trend.png")
    _save(plot_top_stations(hourly), "eda_top_stations.png")
    _save(plot_station_map(hourly), "eda_station_map.png")
    _save(plot_net_flow(hourly), "eda_net_flow.png")
    _save(plot_dock_utilization(hourly), "eda_dock_utilization.png")
    _save(plot_risk_analysis(hourly), "eda_risk_analysis.png")

    # Step 3: Feature engineering
    print("\n" + "=" * 60)
    print("STEP 3: Feature engineering")
    print("=" * 60)
    df = build_features(hourly, target="departures")

    _save(plot_correlation_heatmap(df), "eda_correlation.png")

    # Step 4: Task 1 - Demand forecasting
    print("\n" + "=" * 60)
    print("STEP 4: Task 1 - Demand forecasting")
    print("=" * 60)
    demand_results = train_demand_models(df, target="departures")
    for r in demand_results:
        print(f"  {r['model']:20s} MAE={r['MAE']:.4f}  RMSE={r['RMSE']:.4f}  WAPE={r['WAPE']:.1f}%")

    fig = plot_model_comparison(demand_results, "MAE", title="Task 1: MAE Comparison")
    fig.savefig("output_demand_mae.png", dpi=150, bbox_inches="tight")

    # skforecast model
    from models.demand import train_skforecast_model
    sk_result = train_skforecast_model(df, target="departures")
    demand_results.append(sk_result)

    # Ensemble blending
    ensemble_result = train_ensemble(demand_results, df, target="departures")
    if ensemble_result is not None:
        demand_results.append(ensemble_result)
        print(f"  {'Ensemble-Blend':20s} MAE={ensemble_result['MAE']:.4f}  "
              f"RMSE={ensemble_result['RMSE']:.4f}  WAPE={ensemble_result['WAPE']:.1f}%")

    _save(plot_demand_comparison(demand_results), "output_demand_comparison.png")

    # Scatter: best sklearn model predictions vs actual
    _excluded_scatter = {"Naive", "HistAvg", "Ensemble-Blend", "skforecast-MultiSeries"}
    sklearn_results = [r for r in demand_results if r["model"] not in _excluded_scatter]
    if sklearn_results:
        best_sk = min(sklearn_results, key=lambda r: r["MAE"])
        avail = [c for c in FEATURE_COLS_DEMAND if c in df.columns]
        _, _, test_df_scatter = time_split(df)
        y_pred_scatter = best_sk["model_obj"].predict(test_df_scatter[avail])
        _save(
            plot_predictions(test_df_scatter["departures"], y_pred_scatter,
                             title=f"Predicted vs Actual - {best_sk['model']}"),
            "output_demand_scatter.png",
        )

    # Step 4.5: Multi-target forecasting (departures + arrivals)
    print("\n" + "=" * 60)
    print("STEP 4.5: Multi-target forecasting")
    print("=" * 60)
    mt_results = train_multi_target_models(df)

    # Compute forecasted features for risk model
    print("Computing forecasted features for risk model...")
    _, _, test_df_mt = time_split(df)
    avail = [c for c in FEATURE_COLS_DEMAND if c in df.columns]

    # Use best demand model for forecasted_departures on all rows
    _excluded = {"Naive", "HistAvg", "Ensemble-Blend", "skforecast-MultiSeries"}
    best_demand = min(
        [r for r in demand_results if r["model"] not in _excluded],
        key=lambda r: r["MAE"],
    )
    df["forecasted_departures"] = best_demand["model_obj"].predict(df[avail])

    # Use multi-target model for net flow on all rows
    df["forecasted_arrivals"] = mt_results["best_model_obj"].predict(df[avail])[:, 1]
    df["forecasted_net_flow"] = df["forecasted_arrivals"] - df["forecasted_departures"]

    # Forecasted bikes available = lag + net flow
    if "bikes_available_lag_1h" in df.columns:
        df["forecasted_bikes_available"] = df["bikes_available_lag_1h"] + df["forecasted_net_flow"]
    else:
        df["forecasted_bikes_available"] = 0.0

    # Step 5: Task 2 - Risk classification
    print("\n" + "=" * 60)
    print("STEP 5: Task 2 - Risk classification")
    print("=" * 60)
    risk_results = train_risk_models(df, target="is_high_risk")
    for r in risk_results:
        print(f"  {r['model']:20s} Acc={r['Accuracy']:.4f}  F1={r['F1']:.4f}  Prec={r['Precision']:.4f}  Rec={r['Recall']:.4f}")

    best_risk = max(risk_results, key=lambda r: r["F1"])
    fig_cm = plot_confusion_matrix(
        best_risk["y_test"], best_risk["y_pred"],
        title=f"Confusion Matrix - {best_risk['model']}",
    )
    fig_cm.savefig("output_risk_confusion.png", dpi=150, bbox_inches="tight")

    _save(plot_all_risk_confusion(risk_results), "output_risk_all_confusion.png")

    # Step 6: Task 3 - Capacity prioritization (using model predictions)
    print("\n" + "=" * 60)
    print("STEP 6: Task 3 - Capacity prioritization")
    print("=" * 60)
    _, _, test_df = time_split(df)
    avail_test = [c for c in FEATURE_COLS_DEMAND if c in test_df.columns]
    test_df["predicted_departures"] = best_demand["model_obj"].predict(test_df[avail_test])

    # Use best risk model predictions
    best_risk_model = max(risk_results, key=lambda r: r["F1"])
    test_df["predicted_risk"] = best_risk_model["y_pred"]

    from models.prioritize import generate_station_summary
    summary = generate_station_summary(test_df)
    ranked = compute_priority_score(summary)

    display_cols = ["station_id", "priority_score", "predicted_demand", "risk_frequency",
                    "max_consecutive_risk_hours", "capacity_strain", "nearest_neighbor_km",
                    "capacity", "lat", "lon"]
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    print(ranked[display_cols].to_string(index=False))

    ranked.to_csv("output_prioritization.csv", index=False)
    print(f"\nSaved full ranking to output_prioritization.csv ({len(ranked)} stations)")

    _save(plot_prioritization(ranked), "output_prioritization.png")

    print("\nDone.")


if __name__ == "__main__":
    main()
