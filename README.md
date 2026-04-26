# Capstone Project

This repo contains our NFL coverage-classification baseline built from the enhanced Big Data Bowl tracking files in `Data/` and the analysis notebooks used for preprocessing and modeling.

## Repository Contents

- `Baseline.ipynb`: baseline modeling notebook for predicting collapsed pass coverage and `pff_manZone`.
- `data_processing.ipynb`: preprocessing notebook that builds the enhanced weekly tracking files used by the baseline model.
- `Data/`: local data directory with `games.csv`, `plays.csv`, `players.csv`, `player_play.csv`, and `week*_tracking_enhanced.csv`.
- `pyproject.toml`: Poetry environment definition.

## Reproducible Setup

This project was developed in Python 3.12 with Poetry.

1. Clone the repository and enter the project directory.
2. Install dependencies:

```bash
poetry install
```

3. Start Jupyter from the Poetry environment:

```bash
poetry run jupyter notebook
```

4. Open `Baseline.ipynb`.
5. Make sure the notebook kernel is the Poetry environment for this project.
6. Run the notebook from top to bottom.

If you prefer to activate the virtual environment directly, you can also run:

```bash
poetry shell
jupyter notebook
```

## Data Requirements

The baseline notebook expects the following files to already exist in `Data/`:

- `games.csv`
- `plays.csv`
- `players.csv`
- `player_play.csv`
- `week1_tracking_enhanced.csv` through `week9_tracking_enhanced.csv`

The baseline experiment uses the nine `week*_tracking_enhanced.csv` files and filters to the final `BEFORE_SNAP` frame for each play.

## Preprocessing Workflow

`data_processing.ipynb` is the notebook used to create the enhanced weekly tracking files that feed the baseline experiment.

In that notebook, the workflow:

- loads raw weekly tracking data plus `games.csv`, `plays.csv`, `players.csv`, and `player_play.csv`
- creates a `sideofball` feature from player position metadata
- merges player-level information into tracking rows
- merges pre-snap play context such as down, distance, yard line, scores, win probability, formation, `pff_passCoverage`, and `pff_manZone`
- exports enhanced files such as `week1_tracking_enhanced.csv` through `week9_tracking_enhanced.csv`

Those enhanced files are then consumed by `Baseline.ipynb`.

## Baseline Experiment

### Objective

Train a simple baseline classifier on pre-snap tracking and play-context features to predict:

- `pff_passCoverage_collapsed`
- `pff_manZone`

### Coverage Label Collapse

The current baseline compresses raw `pff_passCoverage` labels into broader families before training and evaluation. For example:

- all `Cover 2` variants map to `Cover 2`
- all `Cover 3` variants map to `Cover 3`
- `Cover 4` and `Quarters` variants map to `Quarters/C4`
- `Cover 6` variants map to `Cover 6`
- `Red Zone`, `Goal Line`, and `Miscellaneous` map to `Exotic/GoalLine`

This collapse is applied before the train/test split and is also the label shown in the final output reports.

### Modeling Pipeline

- Input rows: final pre-snap frame per play
- Features: positional, motion, game-state, and categorical context fields from `Baseline.ipynb`
- Numeric preprocessing: median imputation + standardization
- Categorical preprocessing: most-frequent imputation + one-hot encoding
- Model: `LogisticRegression`
- Solver: `saga`
- Class weighting: `balanced`
- Split strategy: `GroupShuffleSplit` with `play_key` grouping and `test_size=0.2`

## Experiment Log

### Iteration: Baseline Logistic Regression

- Notebook: `Baseline.ipynb`
- Data used: `week1_tracking_enhanced.csv` through `week9_tracking_enhanced.csv`
- Evaluation rows: 16,119 total plays after baseline frame selection
- Train rows: 12,741
- Test rows: 3,186
- Random seed: `42`
- Targets: `pff_passCoverage_collapsed`, `pff_manZone`

Saved notebook metrics:

| Target | Classes | Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|---:|
| `pff_passCoverage_collapsed` | 9 | 0.31 | 0.28 | 0.32 |
| `pff_manZone` | 3 | 0.608600 | 0.525982 | 0.631265 |

Updated saved classification report for `pff_passCoverage_collapsed`:

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| `Bracket` | 0.04 | 0.47 | 0.07 | 15 |
| `Cover 0` | 0.16 | 0.42 | 0.24 | 115 |
| `Cover 1` | 0.35 | 0.22 | 0.27 | 685 |
| `Cover 2` | 0.31 | 0.39 | 0.34 | 440 |
| `Cover 3` | 0.56 | 0.26 | 0.35 | 1133 |
| `Cover 6` | 0.21 | 0.40 | 0.27 | 250 |
| `Exotic/GoalLine` | 0.39 | 0.71 | 0.50 | 132 |
| `Prevent` | 0.11 | 0.64 | 0.19 | 14 |
| `Quarters/C4` | 0.27 | 0.26 | 0.26 | 402 |

This updated run reflects the revised label-collapsing logic in `Baseline.ipynb`, which reduces the coverage target from many punctuation and naming variants down to 9 broader classes for evaluation, with `Cover 6` kept separate from `Quarters/C4`.

## Runtime and Budget

### Runtime for This Iteration

- Compute environment: local laptop / local Python environment
- Hardware budget: CPU-only baseline, no cloud training budget used
- GPU required: no
- Measured in `Baseline.ipynb` using `time.perf_counter()`
- `pff_manZone`: `4.335892` seconds total (`4.325504` fit, `0.010388` predict)
- `pff_passCoverage_collapsed`: `7.422823` seconds total (`7.410392` fit, `0.012431` predict)
- Combined baseline evaluation loop runtime: about `11.76` seconds

These timings should be treated as local baseline measurements on a CPU-only setup, so they are useful for relative comparison but may vary across machines.

### Budget Summary

- Direct cloud cost: `$0`
- External API cost: `$0`
- Training budget class: low-cost baseline

## Autoresearch Experiment Log

The autoresearch loop ran 14 automated experiments against the `pff_passCoverage_collapsed` target using macro F1 as the primary metric. The five experiments that produced improvements, in the order they ran:

### 1. HistGradientBoosting with ordinal-encoded categoricals
- **Macro F1:** 0.3829 (baseline was 0.2786 — **+37%**)
- Replaced LogisticRegression with `HistGradientBoostingClassifier`. Categorical features were ordinal-encoded and passed as native categorical splits; numeric features were passed through as-is (HGB handles NaN natively). The biggest single gain of the entire run.

### 2. HGB hyperparameter tuning
- **Macro F1:** 0.3985 (+4% over prior best)
- Tuned the HGB: `max_iter=500`, `learning_rate=0.05`, `max_depth=8`, `min_samples_leaf=10`, `l2_regularization=0.1`. More iterations at a moderate learning rate meaningfully improved over the default HGB settings.

### 3. VotingClassifier: soft vote HGB + RandomForest
- **Macro F1:** 0.4187 (+4.8% over prior best)
- Combined the tuned HGB with a `RandomForestClassifier` (200 trees, `balanced` class weight) using soft voting. The two models make different types of errors, so combining their probability outputs was better than either alone.

### 4. Extended feature engineering
- **Macro F1:** 0.4219 (+0.5% over prior best)
- Added nine engineered features on top of the VotingClassifier: `score_diff`, `wp_diff`, `is_redzone`, `down_x_yards`, `is_long_yardage`, `is_short_yardage`, `two_min_warning`, `field_zone`, and `score_sign`. Game-state context that is implicit in the raw columns became explicit and immediately usable.

### 5. QuantileTransformer on RF numeric branch *(best overall)*
- **Macro F1:** 0.4264 (+0.4% over prior best)
- Applied `QuantileTransformer(output_distribution="normal")` to the RandomForest's numeric inputs instead of `StandardScaler`. Skewed tracking features like speed and acceleration are better normalized by a quantile mapping. Final best: **macro F1 = 0.4264**, a **+53% improvement** over the LogisticRegression baseline.

## Notes

- If the notebook kernel throws an import error such as `RuntimeError: CPU dispatcher tracer already initlized`, restart the kernel fully and rerun the notebook from the top.
- The repository currently stores data locally under `Data/`, so anyone reproducing the run needs those files in place before opening the notebook.
