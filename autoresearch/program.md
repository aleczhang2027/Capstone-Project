# AutoResearch Agent Instructions

## Objective

Minimize **Macro F1** on the NFL analysis task.

## Rules

1. You may **ONLY** modify `model.py`
2. `prepare_flattened.py` and `run_flattened.py` are **FROZEN** — do not touch them
3. `build_model()` must return an sklearn-compatible estimator (Pipeline preferred)
4. Training + evaluation must complete in **under 300 seconds** on MPS
5. No additional data sources or external downloads
6. do your best to not add new files

## Workflow

```
1. Read current model.py
2. Propose a modification
3. Edit model.py
4. Run:  python run.py "description of change"
5. Check f1 score in output
6. If macro f1 improves:  git add model.py && git commit -m "feat: <description>"
7. If worse:     git checkout model.py   (revert)
8. Repeat from step 1
```

## Ideas to explore

- Models to try: a neural net transformer, Temporal CNN + spatial attention, Factored attention
- Model Archietecture:  For Transformer: sweep depth , hidden dim , heads , for temporal, Compare causal attention vs LSTM on time axis and rotary positional encodings vs learned embeddings
- Training tricks: AdamW with cosine LR + linear warmup, Learning rate schedule, class-weighted loss, gradient clipping, batch sizes, EMA


## What NOT to do

- Do not modify `prepare.py` (data split, metric)
- Do not add new files or dependencies
- Do not hard-code validation data into the model
- Do not change the function signature of `build_model()`
