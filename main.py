"""Main pipeline: orchestrate preprocess, feature engineering, modeling."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from preprocess import preprocess
from simulate_docks import label_risk, simulate_inventory
from features import build_features
from models.demand import train_demand_models
from models.risk import train_risk_models
from models.prioritize import compute_priority_score
from evaluate import plot_model_comparison
import matplotlib.pyplot as plt


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
    hourly = simulate_inventory(hourly, station_info)
    hourly = label_risk(hourly)

    # Step 3: Feature engineering
    print("\n" + "=" * 60)
    print("STEP 3: Feature engineering")
    print("=" * 60)
    df = build_features(hourly, target="departures")

    # Step 4: Task 1 - Demand forecasting
    print("\n" + "=" * 60)
    print("STEP 4: Task 1 - Demand forecasting")
    print("=" * 60)
    demand_results = train_demand_models(df, target="departures")
    for r in demand_results:
        print(f"  {r['model']:20s} MAE={r['MAE']:.4f}  RMSE={r['RMSE']:.4f}  MAPE={r['MAPE']:.1f}%")

    fig = plot_model_comparison(demand_results, "MAE", title="Task 1: MAE Comparison")
    fig.savefig("output_demand_mae.png", dpi=150, bbox_inches="tight")

    # skforecast model
    from models.demand import train_skforecast_model
    sk_result = train_skforecast_model(df, target="departures")
    demand_results.append(sk_result)

    # Step 5: Task 2 - Risk classification
    print("\n" + "=" * 60)
    print("STEP 5: Task 2 - Risk classification")
    print("=" * 60)
    risk_results = train_risk_models(df, target="is_high_risk")
    for r in risk_results:
        print(f"  {r['model']:20s} Acc={r['Accuracy']:.4f}  F1={r['F1']:.4f}  Prec={r['Precision']:.4f}  Rec={r['Recall']:.4f}")

    from evaluate import plot_confusion_matrix
    best_risk = max(risk_results, key=lambda r: r["F1"])
    fig_cm = plot_confusion_matrix(
        best_risk["y_test"], best_risk["y_pred"],
        title=f"Confusion Matrix - {best_risk['model']}",
    )
    fig_cm.savefig("output_risk_confusion.png", dpi=150, bbox_inches="tight")

    # Step 6: Task 3 - Capacity prioritization
    print("\n" + "=" * 60)
    print("STEP 6: Task 3 - Capacity prioritization")
    print("=" * 60)
    summary = df.groupby("station_id").agg(
        predicted_demand=("departures", "mean"),
        risk_frequency=("is_high_risk", "mean"),
        avg_daily_demand=("departures", lambda x: x.sum() / df.loc[x.index, "hour"].dt.date.nunique()),
    ).reset_index()
    ranked = compute_priority_score(summary)
    print(ranked[["station_id", "priority_score"]].head(10))

    print("\nDone.")


if __name__ == "__main__":
    main()
