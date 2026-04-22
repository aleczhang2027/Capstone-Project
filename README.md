# Capstone Project

This repo contains our NFL coverage-classification baseline built from the enhanced Big Data Bowl tracking files in `Data/` and the analysis notebooks used for preprocessing and modeling.

## Repository Contents

- `Baseline.ipynb`: baseline modeling notebook for predicting collapsed pass coverage and `pff_manZone`.
- `data_processing.ipynb`: data preparation notebook.
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

## Baseline Experiment

### Objective

Train a simple baseline classifier on pre-snap tracking and play-context features to predict:

- `pff_passCoverage_collapsed`
- `pff_manZone`

### Coverage Label Collapse

The current baseline compresses raw `pff_passCoverage` labels into broader families before training and evaluation. For example:

- all `Cover 2` variants map to `Cover 2`
- all `Cover 3` variants map to `Cover 3`
- `Cover 4`, `Cover 6`, and `Quarters` variants map to `Quarters/C4`
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
| `pff_passCoverage_collapsed` | 17 | 0.212806 | 0.152841 | 0.240724 |
| `pff_manZone` | 3 | 0.608600 | 0.525982 | 0.631265 |

## Runtime and Budget

### Runtime for This Iteration

- Compute environment: local laptop / local Python environment
- Hardware budget: CPU-only baseline, no cloud training budget used
- GPU required: no
- Saved notebook outputs include model metrics, but wall-clock runtime was not explicitly logged in the notebook for this iteration

Because runtime was not recorded in the notebook itself, this iteration should be treated as a local low-cost baseline run rather than a timed benchmark.

### Budget Summary

- Direct cloud cost: `$0`
- External API cost: `$0`
- Training budget class: low-cost baseline

## Notes

- If the notebook kernel throws an import error such as `RuntimeError: CPU dispatcher tracer already initlized`, restart the kernel fully and rerun the notebook from the top.
- The repository currently stores data locally under `Data/`, so anyone reproducing the run needs those files in place before opening the notebook.
