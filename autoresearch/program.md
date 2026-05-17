# AutoResearch Agent Instructions

## Objective

Maximize **Macro F1** on the NFL coverage classification task (`pff_passCoverage_collapsed`, 9 classes).

Current best: **0.4299** (spatial transformer, 150 ep, CosineAnnealingWarmRestarts T₀=75, MPS).

## Rules

1. You may **ONLY** modify `model_flattened.py`
2. `prepare_flattened.py` and `run_flattened.py` are **FROZEN** — do not touch them
3. `build_model()` must return an sklearn-compatible estimator (Pipeline preferred)
4. Training + evaluation must complete in **under 300 seconds** on MPS
5. No additional data sources or external downloads
6. Do not add new files

## Workflow

```
1. Read current model_flattened.py
2. Propose a modification
3. Edit model_flattened.py
4. Run:  python run_flattened.py "description of change"
5. Check f1 score in output
6. If macro f1 improves:  git add model_flattened.py && git commit -m "feat: <description>"
7. If worse:     git checkout model_flattened.py   (revert)
8. Repeat from step 1
```

## Current Architecture (keep this as the base)

- **Model**: `CoverageTransformer` — spatial transformer over 22 per-player trajectory summaries
- **Input**: 40 pre-snap frames → per-player summary (last, mean, delta) for x/y/s → 9 features × 22 players
- **Architecture**: `hidden_dim=64`, `n_heads=4`, `n_layers=3`, `dropout=0.15`
- **Training**: 150 epochs, `CosineAnnealingWarmRestarts(T_0=75, T_mult=1)`, `lr=1e-3`, `batch=128`, MPS
- **Key fix**: `enable_nested_tensor=False` on `nn.TransformerEncoder` — required for MPS

## Ideas to Explore (not yet tried or not yet converged)

- **Late-fusion ensemble**: wrap `CoverageTransformer` + `HGB/RF` (from `model.py`) in a `VotingClassifier`-style stacker — both pipelines sit near 0.43 and make different errors
- **Wider transformer**: sweep `hidden_dim` in {96, 128} with `n_layers=3`; prior attempt at `hidden=96, layers=4` diverged — try wider-only first
- **More player features**: add player position group (CB, S, LB, WR, TE, RB, OL) as a learnable embedding alongside `side_emb`
- **Longer training**: 200 epochs with T₀=100 (prior 200-ep run without warm restarts degraded — restarts are needed)
- **Per-player dropout**: add masking with probability 0.1 during training to regularize the attention over 22 players

## What Has Been Tried (do not repeat)

| Approach | Result | Why it failed |
|---|---|---|
| Two-stage spatio-temporal transformer | 0.038 | Collapsed to near-random; architecture too complex to train |
| Factored attention (spatial-per-frame + temporal-across-frames) | 0.090 | Same — too many parameters for data size |
| Context side-channel (down/dist/score/WP/formation) | 0.342 | Attention collapsed; side channel overwhelmed player features |
| `OneCycleLR` with high max_lr | 0.278 | Overshot; cosine restarts are more stable |
| `label_smoothing=0.1` | 0.118 | Destabilized training badly |
| `dropout=0.10` | 0.411 | Marginal degradation vs current 0.15 |
| `hidden=96, heads=4, layers=4` | 0.317 | Diverged; too many params |
| 200 epochs without warm restarts | 0.402 | Overfit after ~150ep without schedule |
| MLP over 198-dim summary | 0.358 | Underfits vs transformer |

## What NOT to do

- Do not modify `prepare_flattened.py` (data split, metric)
- Do not hard-code validation data into the model
- Do not change the function signature of `build_model()`
- Do not attempt full 40-frame attention directly — too many tokens, runs over budget
