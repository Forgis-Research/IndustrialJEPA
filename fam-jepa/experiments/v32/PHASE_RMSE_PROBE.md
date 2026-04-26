# V32 Phase: MSE-RUL Probe on Frozen v30 Encoder (~30 min)

## Goal
Train a direct-MSE regression head on the frozen v30 encoder to get competitive C-MAPSS RMSE numbers. The paper currently reports RMSE 36.5/44.1/39.5 (derived from probability surface inversion). A direct MSE head should close ~85% of the gap to supervised SOTA.

## What to do

Freeze the v30 pretrained encoder. Train a small MLP head that maps h_t -> scalar RUL estimate, optimized with MSE loss. No probability surface, no BCE, no hazard CDF. Just regression.

### Architecture
```
h_t (256-d) -> LayerNorm -> Linear(256, 128) -> ReLU -> Linear(128, 1) -> clamp(0, 125)
```
~33K trainable params. RUL cap = 125 (standard C-MAPSS protocol).

### Training
- Datasets: FD001, FD002, FD003 (use v30 pretrained checkpoints)
- Seeds: 42, 123, 456
- Labels: RUL target = min(cycles_remaining, 125)
- Loss: MSE
- Optimizer: Adam, lr=1e-3, weight_decay=1e-4
- Epochs: 100 (early stop on val RMSE, patience 15)
- Batch size: 64
- Val split: last 20% of training engines

### Evaluation
Standard C-MAPSS RMSE protocol: predict RUL at the LAST cycle of each test engine, compute sqrt(mean((pred - true)^2)).

### Expected results
- FD001: ~17 RMSE (floor: 27.1 from surface inversion; ceiling: 13.8 from v11 E2E)
- FD002: ~26 RMSE (floor: 34.5; ceiling: 24.5)
- FD003: ~18 RMSE (floor: 30.5; ceiling: 15.4)

### Quick MLP sweep (optional, 10 min extra)
If time: try hidden_dim in {64, 128, 256} and pick best val RMSE. But 128 should be fine.

### Output
Save results as `results/rmse_probe.json`:
```json
{
  "FD001": {"s42": X, "s123": X, "s456": X, "mean": X, "std": X},
  "FD002": {...},
  "FD003": {...}
}
```

### Paper impact
Replace Table 4 RMSE column: current "36.5 / 44.1 / 39.5" -> new MSE-probe numbers. Add footnote: "RMSE from frozen-encoder MSE probe (33K params); probability surface optimizes a different objective." Narrative: "FAM's encoder carries the RUL signal; with an MSE-aligned head, it closes ~85% of the gap to supervised SOTA without additional pretraining."
