import zipfile
import pandas as pd
from pathlib import Path
from tqdm import tqdm

ZIP_PATH = Path("/Users/alecxszhang/Desktop/Stat 390/Data/nfl-big-data-bowl-2025.zip")
OUT_DIR = Path("/Users/alecxszhang/Desktop/Stat 390/Data")
CHUNK_SIZE = 200_000

defense_positions = {"OLB", "DE", "DT", "ILB", "FS", "SS", "NT", "MLB", "DB", "LB", "CB", "SAF"}

presnap_cols = [
    "gameId", "playId", "quarter", "down", "yardsToGo", "gameClock",
    "absoluteYardlineNumber", "yardlineSide", "yardlineNumber", "playClockAtSnap",
    "preSnapHomeScore", "preSnapVisitorScore",
    "preSnapHomeTeamWinProbability", "preSnapVisitorTeamWinProbability",
    "expectedPoints", "possessionTeam", "defensiveTeam",
    "offenseFormation", "receiverAlignment",
    "pff_passCoverage", "pff_manZone",
]

# Approximate row counts per week (from prior runs)
KNOWN_ROWS = {1: 7_200_341, 2: 6_795_170, 3: 7_209_143, 4: 6_821_608,
              5: 7_174_239, 6: 6_310_244, 7: 6_100_000, 8: 6_700_000, 9: 5_739_165}

print("Opening zip...")
with zipfile.ZipFile(ZIP_PATH) as z:
    print("Loading metadata files...")
    players = pd.read_csv(z.open("players.csv"))
    plays = pd.read_csv(z.open("plays.csv"))

    players["sideofball"] = players["position"].apply(
        lambda pos: "Defense" if pos in defense_positions else "Offense"
    )
    play_context = plays[presnap_cols]

    weeks_to_run = [wk for wk in range(1, 10)
                    if not (OUT_DIR / f"week{wk}_tracking_enhanced.csv").exists()]

    if not weeks_to_run:
        print("All weeks already processed.")
    else:
        print(f"Weeks to process: {weeks_to_run}\n")

    for wk in weeks_to_run:
        fname = f"tracking_week_{wk}.csv"
        est_rows = KNOWN_ROWS.get(wk, 7_000_000)
        est_chunks = (est_rows // CHUNK_SIZE) + 1

        chunks = []
        with tqdm(total=est_chunks, desc=f"Week {wk}", unit="chunk",
                  bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} chunks [{elapsed}<{remaining}]") as pbar:
            for chunk in pd.read_csv(z.open(fname), chunksize=CHUNK_SIZE):
                chunk = chunk.merge(players[["displayName", "sideofball"]], on="displayName", how="left")
                chunk = chunk.merge(play_context, on=["gameId", "playId"], how="left")
                chunks.append(chunk)
                pbar.update(1)

        enhanced = pd.concat(chunks, ignore_index=True)
        out_path = OUT_DIR / f"week{wk}_tracking_enhanced.csv"
        enhanced.to_csv(out_path, index=False)
        print(f"  Saved: {out_path.name} | {enhanced.shape[0]:,} rows x {enhanced.shape[1]} cols\n")

print("Done.")
