"""
Demo: simulate an AutoResearch agent loop with 8 iterations.

This script demonstrates the full keep/discard workflow:
  1. Run baseline model
  2. Try modifications one by one
  3. Keep improvements, discard regressions
  4. Plot the trajectory

Usage: python demo.py
"""
import os
import time
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.compose import TransformedTargetRegressor

from prepare import load_data, evaluate, log_result, plot_results, RESULTS_FILE


# ── Each "iteration" is a model the agent might try ────────
ITERATIONS = [
    {
        "id": 1,
        "description": "baseline: LinearRegression",
        "model": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ]),
    },
    {
        "id": 2,
        "description": "Ridge(alpha=1.0)",
        "model": Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0)),
        ]),
    },
    {
        "id": 3,
        "description": "Lasso(alpha=0.01)",
        "model": Pipeline([
            ("scaler", StandardScaler()),
            ("model", Lasso(alpha=0.01)),
        ]),
    },
    {
        "id": 4,
        "description": "PolyFeatures(2) + Ridge",
        "model": Pipeline([
            ("scaler", StandardScaler()),
            ("poly", PolynomialFeatures(degree=2, interaction_only=True)),
            ("model", Ridge(alpha=1.0)),
        ]),
    },
    {
        "id": 5,
        "description": "PolyFeatures(3) + Ridge -- overshoot",
        "model": Pipeline([
            ("scaler", StandardScaler()),
            ("poly", PolynomialFeatures(degree=3)),
            ("model", Ridge(alpha=0.1)),
        ]),
    },
    {
        "id": 6,
        "description": "RandomForest(n=100)",
        "model": Pipeline([
            ("model", RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)),
        ]),
    },
    {
        "id": 7,
        "description": "GradientBoosting(n=200)",
        "model": Pipeline([
            ("model", GradientBoostingRegressor(
                n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42
            )),
        ]),
    },
    {
        "id": 8,
        "description": "HistGBT(n=300, tuned)",
        "model": Pipeline([
            ("model", HistGradientBoostingRegressor(
                max_iter=300, max_depth=8, learning_rate=0.08,
                min_samples_leaf=20, random_state=42
            )),
        ]),
    },
]


def main():
    # Clean previous results
    if os.path.exists(RESULTS_FILE):
        os.remove(RESULTS_FILE)
        print(f"Cleared previous {RESULTS_FILE}\n")

    # Load data once
    X_train, y_train, X_val, y_val, feature_names = load_data()
    print(f"Dataset: California Housing")
    print(f"  Train: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"  Val:   {X_val.shape[0]} samples")
    print(f"  Features: {list(feature_names)}")
    print(f"{'=' * 70}\n")

    best_rmse = float("inf")

    for it in ITERATIONS:
        exp_id = it["id"]
        desc = it["description"]
        model = it["model"]

        print(f"── Experiment {exp_id}: {desc}")

        # Train
        t0 = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - t0

        # Evaluate
        val_rmse, val_r2 = evaluate(model, X_val, y_val)

        # Keep / discard decision
        if exp_id == 1:
            status = "baseline"
            best_rmse = val_rmse
            decision_msg = "BASELINE established"
        elif val_rmse < best_rmse:
            status = "keep"
            improvement = (best_rmse - val_rmse) / best_rmse * 100
            decision_msg = f"KEEP  (improved {improvement:.1f}% over best)"
            best_rmse = val_rmse
        else:
            status = "discard"
            regression = (val_rmse - best_rmse) / best_rmse * 100
            decision_msg = f"DISCARD (regressed {regression:.1f}% vs best)"

        # Log
        log_result(f"exp-{exp_id:03d}", val_rmse, val_r2, status, desc)

        # Print
        print(f"   RMSE:  {val_rmse:.6f}  |  R²: {val_r2:.4f}  |  Time: {train_time:.2f}s")
        print(f"   >>> {decision_msg}")
        print()

    # Summary
    print(f"{'=' * 70}")
    print(f"Best RMSE achieved: {best_rmse:.6f}")
    print(f"Results saved to:   {RESULTS_FILE}")
    print()

    # Plot
    plot_results("performance.png")
    print("\nDone. Open performance.png to see the trajectory.")


if __name__ == "__main__":
    main()
