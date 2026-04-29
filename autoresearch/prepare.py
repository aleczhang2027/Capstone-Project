"""
FROZEN -- Do not modify this file.
Data loading, train/val split, evaluation metrics, and plotting
for NFL coverage classification (pff_passCoverage_collapsed).
"""
import csv
import os
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt

# ── Constants ──────────────────────────────────────────────
RANDOM_SEED = 42
VAL_FRACTION = 0.2
RESULTS_FILE = "results.tsv"
PRIMARY_TARGET = "pff_passCoverage_collapsed"

DATA_DIR = Path(__file__).parent.parent / "Data"

FEATURE_COLUMNS = [
    "x", "y", "s", "a", "dis", "o", "dir", "event", "sideofball",
    "quarter", "down", "yardsToGo", "gameClock", "absoluteYardlineNumber",
    "yardlineSide", "yardlineNumber", "playClockAtSnap", "preSnapHomeScore",
    "preSnapVisitorScore", "preSnapHomeTeamWinProbability",
    "preSnapVisitorTeamWinProbability", "expectedPoints", "possessionTeam",
    "defensiveTeam", "offenseFormation", "receiverAlignment",
]

# ── Data ───────────────────────────────────────────────────
def load_data():
    """Load pre-processed NFL tracking data (last BEFORE_SNAP frame per play).

    Expects week*_tracking_enhanced.csv files already processed by
    Additional_data_processing.ipynb: one row per play, gameClock in seconds,
    pff_passCoverage_collapsed and play_key columns present.

    Target: pff_passCoverage_collapsed (9-class coverage family).
    Split: GroupShuffleSplit on play_key so no play leaks across train/val.
    """
    tracking_files = sorted(DATA_DIR.glob("week*_tracking_enhanced.csv"))
    if not tracking_files:
        raise FileNotFoundError(f"No tracking files found in {DATA_DIR}")

    wanted = {"play_key", PRIMARY_TARGET, *FEATURE_COLUMNS}
    frames = [
        pd.read_csv(path, usecols=lambda c: c in wanted, low_memory=False)
        for path in tracking_files
    ]
    df = pd.concat(frames, ignore_index=True).dropna(subset=[PRIMARY_TARGET])

    X = df[FEATURE_COLUMNS]
    y = df[PRIMARY_TARGET]
    groups = df["play_key"]

    splitter = GroupShuffleSplit(n_splits=1, test_size=VAL_FRACTION, random_state=RANDOM_SEED)
    train_idx, val_idx = next(splitter.split(X, y, groups=groups))

    return (
        X.iloc[train_idx].reset_index(drop=True),
        y.iloc[train_idx].reset_index(drop=True),
        X.iloc[val_idx].reset_index(drop=True),
        y.iloc[val_idx].reset_index(drop=True),
        FEATURE_COLUMNS,
    )


# ── Evaluation (frozen metric) ─────────────────────────────
def evaluate(model, X_val, y_val):
    """Return (accuracy, macro_f1, weighted_f1) on the validation set."""
    preds = model.predict(X_val)
    acc = float(accuracy_score(y_val, preds))
    macro_f1 = float(f1_score(y_val, preds, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_val, preds, average="weighted", zero_division=0))
    return acc, macro_f1, weighted_f1


# ── Logging ────────────────────────────────────────────────
def log_result(experiment_id, val_acc, val_macro_f1, val_weighted_f1, status, description):
    """Append one row to results.tsv."""
    file_exists = os.path.exists(RESULTS_FILE)
    with open(RESULTS_FILE, "a", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        if not file_exists:
            writer.writerow([
                "experiment", "val_acc", "val_macro_f1", "val_weighted_f1",
                "status", "description",
            ])
        writer.writerow([
            experiment_id,
            f"{val_acc:.6f}",
            f"{val_macro_f1:.6f}",
            f"{val_weighted_f1:.6f}",
            status,
            description,
        ])


# ── Plotting ───────────────────────────────────────────────
def plot_results(save_path="performance.png"):
    """Plot validation macro F1 and accuracy over experiments from results.tsv."""
    if not os.path.exists(RESULTS_FILE):
        print("No results.tsv found. Run some experiments first.")
        return

    experiments, accs, macro_f1s, weighted_f1s, statuses, descriptions = [], [], [], [], [], []
    with open(RESULTS_FILE) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            experiments.append(row["experiment"])
            accs.append(float(row["val_acc"]))
            macro_f1s.append(float(row["val_macro_f1"]))
            weighted_f1s.append(float(row["val_weighted_f1"]))
            statuses.append(row["status"])
            descriptions.append(row["description"])

    color_map = {"keep": "#2ecc71", "discard": "#e74c3c", "baseline": "#3498db"}
    colors = [color_map.get(s, "#95a5a6") for s in statuses]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

    # ── Top: Macro F1 (primary metric) ──
    ax1.scatter(range(len(macro_f1s)), macro_f1s, c=colors, s=80, zorder=3,
                edgecolors="white", linewidth=0.5)
    ax1.plot(range(len(macro_f1s)), macro_f1s, "k--", alpha=0.2, zorder=2)

    best_so_far = []
    current_best = -float("inf")
    for v in macro_f1s:
        current_best = max(current_best, v)
        best_so_far.append(current_best)
    ax1.plot(range(len(macro_f1s)), best_so_far, color="#2ecc71", linewidth=2.5, label="Best so far")

    ax1.set_ylim(0, 1)
    ax1.set_ylabel("Validation Macro F1 (higher is better)", fontsize=12)
    ax1.set_title(
        "AutoResearch: NFL Coverage Classification (pff_passCoverage_collapsed)",
        fontsize=13, fontweight="bold",
    )
    ax1.grid(True, alpha=0.3)

    # ── Bottom: Accuracy ──
    ax2.scatter(range(len(accs)), accs, c=colors, s=80, zorder=3,
                edgecolors="white", linewidth=0.5)
    ax2.plot(range(len(accs)), accs, "k--", alpha=0.2, zorder=2)

    best_acc = []
    current_best_acc = -float("inf")
    for v in accs:
        current_best_acc = max(current_best_acc, v)
        best_acc.append(current_best_acc)
    ax2.plot(range(len(accs)), best_acc, color="#2ecc71", linewidth=2.5, label="Best so far")

    ax2.set_ylim(0, 1)
    ax2.set_xlabel("Experiment #", fontsize=12)
    ax2.set_ylabel("Validation Accuracy (higher is better)", fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    short_labels = [d[:22] + ".." if len(d) > 24 else d for d in descriptions]
    ax2.set_xticks(range(len(accs)))
    ax2.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=8)

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#3498db",
               markersize=10, label="baseline"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#2ecc71",
               markersize=10, label="keep (improved)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#e74c3c",
               markersize=10, label="discard (regressed)"),
        Line2D([0], [0], color="#2ecc71", linewidth=2.5, label="Best so far"),
    ]
    ax1.legend(handles=legend_elements, loc="lower right", fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved {save_path}")


if __name__ == "__main__":
    plot_results()
