# Ablation / Comparison Table

| Pipeline | Model | Macro F1 | Notes |
|---|---|---:|---|
| **Single-frame** | Logistic Regression (baseline) | 0.279 | Single pre-snap frame, ~20 features |
| Single-frame | HGB (tuned) | 0.399 | Native categorical splits, 500 iters |
| Single-frame | VotingClassifier HGB+RF | 0.419 | Soft vote, balanced RF |
| Single-frame | VotingClassifier + feature engineering | 0.422 | score_diff, is_redzone, wp_diff, etc. |
| Single-frame | **VotingClassifier + QuantileTransformer** | **0.426** | Best single-frame result |
| **Flattened** | HGB baseline | 0.367 | 40 frames × 22 players × 3 feats |
| Flattened | HGB tuned (depth=8, l2=0.05) | 0.393 | |
| Flattened | Spatial transformer (60 ep, CPU) | 0.391 | Before MPS fix |
| Flattened | Spatial transformer (60 ep, MPS fix) | 0.402 | `enable_nested_tensor=False` unlock |
| Flattened | Spatial transformer (100 ep) | 0.410 | |
| Flattened | Spatial transformer (150 ep) | 0.416 | |
| Flattened | Spatial transformer + CosineWarmRestarts T₀=50 | 0.429 | |
| Flattened | **Spatial transformer + CosineWarmRestarts T₀=75** | **0.430** | Best overall |
| Flattened | Spatial transformer hidden=96, layers=4 | 0.317 | Diverged |
| Flattened | Factored spatio-temporal attention | 0.090 | Collapsed |
| Flattened | Context side-channel + transformer | 0.342 | Attention collapsed |
