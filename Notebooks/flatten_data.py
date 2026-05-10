"""
Reconstruct week*_tracking_flattened.csv from week*_tracking_enhanced.csv.

One row per play: last 40 BEFORE_SNAP frames, up to 11 offense + 11 defense players,
position-ordered by x at the final frame. Features: x, y, s per player per frame.
Column names: off_p01_f01_x ... def_p11_f40_s  (2,640 tracking columns total).
"""
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

DATA_DIR = Path("/Users/alecxszhang/Desktop/Stat 390/Data")

N_PLAYERS = 11
N_FRAMES  = 40
FEATURES  = ["x", "y", "s"]

COVERAGE_MAP = {
    "Cover-0":              "Cover-0",
    "Goal Line":            "Cover-0",
    "Cover-1":              "Cover-1",
    "Cover-1 Double":       "Cover-1",
    "2-Man":                "Cover-2",
    "Cover-2":              "Cover-2",
    "Cover-3":              "Cover-3",
    "Cover-3 Seam":         "Cover-3",
    "Cover-3 Cloud Left":   "Cover-3",
    "Cover-3 Cloud Right":  "Cover-3",
    "Cover-3 Double Cloud": "Cover-3",
    "Quarters":             "Quarters",
    "Cover-6 Right":        "Cover-6",
    "Cover 6-Left":         "Cover-6",
    "Bracket":              "Bracket",
    "Prevent":              "Prevent",
    "Miscellaneous":        "Other",
    "Red Zone":             "Other",
}

CONTEXT_COLS = [
    "quarter", "down", "yardsToGo", "gameClock", "absoluteYardlineNumber",
    "playClockAtSnap", "preSnapHomeScore", "preSnapVisitorScore",
    "preSnapHomeTeamWinProbability", "preSnapVisitorTeamWinProbability",
    "expectedPoints", "possessionTeam", "defensiveTeam",
    "offenseFormation", "receiverAlignment",
]

# Build flat column names in the same sorted order prepare_flattened.py expects
TRACKING_COLS = sorted(
    f"{side}_p{p:02d}_f{f:02d}_{feat}"
    for side in ("def", "off")
    for p in range(1, N_PLAYERS + 1)
    for f in range(1, N_FRAMES + 1)
    for feat in FEATURES
)


def gameclock_to_seconds(clock_str):
    try:
        m, s = str(clock_str).split(":")
        return int(m) * 60 + int(s)
    except Exception:
        return np.nan


def flatten_play(group):
    """Return a dict of tracking columns for one play."""
    before = group[group["frameType"] == "BEFORE_SNAP"].copy()
    before = before.sort_values("frameId")

    row = {}
    for side_label, side_name in [("off", "Offense"), ("def", "Defense")]:
        side_df = before[before["sideofball"] == side_name]

        # Order players by x at the final frame, then take up to N_PLAYERS
        last_frame = side_df[side_df["frameId"] == side_df["frameId"].max()]
        ordered_players = last_frame.sort_values("x")["nflId"].tolist()[:N_PLAYERS]

        for p_idx, nfl_id in enumerate(ordered_players, start=1):
            player_frames = side_df[side_df["nflId"] == nfl_id].sort_values("frameId")
            # Take last N_FRAMES frames
            player_frames = player_frames.tail(N_FRAMES)
            vals = player_frames[FEATURES].values  # shape (actual_frames, 3)

            # Pad front with NaN if fewer than N_FRAMES
            pad = N_FRAMES - len(vals)
            if pad > 0:
                vals = np.vstack([np.full((pad, len(FEATURES)), np.nan), vals])

            for f_idx in range(N_FRAMES):
                for feat_idx, feat in enumerate(FEATURES):
                    col = f"{side_label}_p{p_idx:02d}_f{f_idx + 1:02d}_{feat}"
                    row[col] = vals[f_idx, feat_idx]

        # Fill missing players with NaN
        for p_idx in range(len(ordered_players) + 1, N_PLAYERS + 1):
            for f_idx in range(N_FRAMES):
                for feat in FEATURES:
                    row[f"{side_label}_p{p_idx:02d}_f{f_idx + 1:02d}_{feat}"] = np.nan

    return row


def process_week(wk):
    in_path  = DATA_DIR / f"week{wk}_tracking_enhanced.csv"
    out_path = DATA_DIR / f"week{wk}_tracking_flattened.csv"

    print(f"\nWeek {wk}: loading...", flush=True)
    df = pd.read_csv(in_path, low_memory=False)

    # Derived columns
    df["play_key"] = df["gameId"].astype(str) + "_" + df["playId"].astype(str)
    df["gameClock"] = df["gameClock"].apply(gameclock_to_seconds)
    df["pff_passCoverage_collapsed"] = df["pff_passCoverage"].map(COVERAGE_MAP)

    plays = df.groupby("play_key", sort=False)
    n_plays = len(plays)

    rows = []
    for play_key, group in tqdm(plays, total=n_plays, desc=f"Week {wk}", unit="play"):
        # Pull one row for context (same for all rows in this play)
        ctx = group.iloc[0]
        record = {"play_key": play_key}
        for col in CONTEXT_COLS:
            record[col] = ctx[col]
        record["pff_passCoverage_collapsed"] = ctx["pff_passCoverage_collapsed"]
        record["pff_manZone"] = ctx["pff_manZone"]
        record.update(flatten_play(group))
        rows.append(record)

    out_cols = ["play_key"] + CONTEXT_COLS + ["pff_passCoverage_collapsed", "pff_manZone"] + TRACKING_COLS
    result = pd.DataFrame(rows, columns=out_cols)
    result.to_csv(out_path, index=False)
    print(f"  Saved: {out_path.name} | {len(result):,} plays x {result.shape[1]} cols")


if __name__ == "__main__":
    weeks_to_run = [
        wk for wk in range(1, 10)
        if not (DATA_DIR / f"week{wk}_tracking_flattened.csv").exists()
    ]
    if not weeks_to_run:
        print("All weeks already flattened.")
    else:
        print(f"Weeks to flatten: {weeks_to_run}")
        for wk in weeks_to_run:
            process_week(wk)
    print("\nDone.")
