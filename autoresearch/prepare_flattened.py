"""
Data loading for the flattened tracking experiment.
Loads week*_tracking_flattened.csv files produced by Additional_data_processing.ipynb.
One row per play: 2,640 positional tracking columns + play context + labels.
"""
import csv
import os
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, f1_score

RANDOM_SEED = 42
VAL_FRACTION = 0.2
RESULTS_FILE = "results_flattened.tsv"
PRIMARY_TARGET = "pff_passCoverage_collapsed"

DATA_DIR = Path(__file__).parent.parent / "Data"

CONTEXT_COLUMNS = [
    "quarter", "down", "yardsToGo", "gameClock", "absoluteYardlineNumber",
    "playClockAtSnap", "preSnapHomeScore", "preSnapVisitorScore",
    "preSnapHomeTeamWinProbability", "preSnapVisitorTeamWinProbability",
    "expectedPoints", "possessionTeam", "defensiveTeam",
    "offenseFormation", "receiverAlignment",
]


def load_data():
    """Load flattened NFL tracking data (last 40 BEFORE_SNAP frames per play).

    Expects week*_tracking_flattened.csv files produced by Additional_data_processing.ipynb.
    Tracking feature columns are named off_p01_f01_x ... def_p11_f40_s.
    Target: pff_passCoverage_collapsed (9-class coverage family).
    Split: GroupShuffleSplit on play_key so no play leaks across train/val.
    """
    tracking_files = sorted(DATA_DIR.glob("week*_tracking_flattened.csv"))
    if not tracking_files:
        raise FileNotFoundError(f"No flattened tracking files found in {DATA_DIR}")

    frames = [pd.read_csv(path, low_memory=False) for path in tracking_files]
    df = pd.concat(frames, ignore_index=True).dropna(subset=[PRIMARY_TARGET])

    tracking_cols = sorted(c for c in df.columns if c.startswith(("off_p", "def_p")))
    feature_columns = tracking_cols + CONTEXT_COLUMNS

    X = df[feature_columns]
    y = df[PRIMARY_TARGET]
    groups = df["play_key"]

    splitter = GroupShuffleSplit(n_splits=1, test_size=VAL_FRACTION, random_state=RANDOM_SEED)
    train_idx, val_idx = next(splitter.split(X, y, groups=groups))

    return (
        X.iloc[train_idx].reset_index(drop=True),
        y.iloc[train_idx].reset_index(drop=True),
        X.iloc[val_idx].reset_index(drop=True),
        y.iloc[val_idx].reset_index(drop=True),
        feature_columns,
    )


def evaluate(model, X_val, y_val):
    preds = model.predict(X_val)
    acc = float(accuracy_score(y_val, preds))
    macro_f1 = float(f1_score(y_val, preds, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_val, preds, average="weighted", zero_division=0))
    return acc, macro_f1, weighted_f1


def log_result(experiment_id, val_acc, val_macro_f1, val_weighted_f1, status, description):
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
