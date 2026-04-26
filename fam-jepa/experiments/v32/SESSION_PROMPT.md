# V32 Overnight Session — Fill Table 4 + Competitive RMSE

**Goal**: Fill every `\placeholder{--}` in Table 4 of the paper. Get competitive C-MAPSS RMSE via MSE probe. Recompute legacy metrics (F1, PA-F1, AUROC) for GECCO and BATADAL with proper threshold optimization. Compute all 10% legacy metrics. Commit hourly. Don't stop early.

**Self-check**: After each phase, verify outputs exist and are reasonable. If a result looks wrong (e.g. F1 < 0.10 on a dataset where h-AUROC > 0.6), investigate before moving on. Re-run with different hyperparameters if needed.

**Commit cadence**: `git add -A && git commit -m "v32 phase X: <description>" && git push` after each phase completes.

---

## Phase A: MSE-RUL Probe on Frozen v30 Encoder (~40 min)

Train a direct-MSE regression head on the frozen v30 encoder to get competitive C-MAPSS RMSE numbers.

### Architecture
```
h_t (256-d) -> LayerNorm -> Linear(256, 128) -> ReLU -> Dropout(0.1) -> Linear(128, 1) -> clamp(0, 125)
```
~33K trainable params. RUL cap = 125 (standard C-MAPSS protocol).

### Training
- Datasets: FD001, FD002, FD003
- Use v30 pretrained checkpoints (find in `checkpoints/` or `experiments/v30/` or `experiments/archive/`)
- Seeds: 42, 123, 456
- Labels: RUL target = min(cycles_remaining, 125)
- Loss: MSE (not Huber, not L1 — pure MSE for RMSE-optimal results)
- Optimizer: Adam, lr=1e-3, weight_decay=1e-4
- LR scheduler: ReduceLROnPlateau(patience=5, factor=0.5)
- Epochs: 150 (early stop on val RMSE, patience 20)
- Batch size: 64
- Val split: last 20% of training engines

### Evaluation
Standard C-MAPSS RMSE protocol: predict RUL at the **last cycle** of each test engine, compute sqrt(mean((pred - true)^2)). Also compute NASA score.

### MLP sweep
Try hidden_dim in {64, 128, 256} and pick best val RMSE. Report all.

### Also run at 10% labels
After finding best architecture at 100%, re-run with 10% label fraction (same engine-sampling as v31 lf10). Report RMSE at 10%.

### Expected results
- FD001: ~14-20 RMSE (current surface-inversion: 36.5; supervised SOTA: 10.61)
- FD002: ~22-30 RMSE (current: 44.1; SOTA: 13.47)
- FD003: ~15-22 RMSE (current: 39.5; SOTA: 10.71)

### Output
Save results as `results/rmse_probe.json`:
```json
{
  "FD001": {"s42": X, "s123": X, "s456": X, "mean": X, "std": X, "hidden_dim": 128},
  "FD002": {"s42": X, "s123": X, "s456": X, "mean": X, "std": X, "hidden_dim": 128},
  "FD003": {"s42": X, "s123": X, "s456": X, "mean": X, "std": X, "hidden_dim": 128},
  "FD001_lf10": {"s42": X, "s123": X, "s456": X, "mean": X, "std": X},
  "FD002_lf10": {...},
  "FD003_lf10": {...}
}
```

**Commit after Phase A.**

---

## Phase B: Legacy Metrics from Stored Surfaces (~30 min)

Recompute legacy metrics (F1, PA-F1, AUROC) for ALL datasets from stored probability surfaces. Use `experiments/v21/surface_to_legacy.py` as reference.

### What to compute

For each dataset at **100% labels** (from v30 surfaces, 3 seeds):

| Dataset | Metric | Method |
|---------|--------|--------|
| SMAP | PA-F1 | `anomaly_legacy_metrics()` with threshold sweep, `pa=True` |
| PSM | PA-F1 | same |
| SMD | PA-F1 | same |
| MBA | AUROC | `roc_auc_score` at Δt=1 (earliest horizon) |
| SKAB | F1 (non-PA) | `anomaly_legacy_metrics()` with `pa=False` |
| **GECCO** | F1 (non-PA) | **RECOMPUTE CAREFULLY** — current 0.28 is suspicious |
| **BATADAL** | F1 (non-PA) | **RECOMPUTE CAREFULLY** — current 0.24 is suspicious |

### GECCO and BATADAL investigation

The current F1 values (0.28, 0.24) seem too low given h-AUROC of 0.82 and 0.61. Investigate:

1. Load the surface, inspect the distribution of predicted probabilities
2. Try different `horizon_for_score` values: 1, 5, 10, 20, 50, 100 (default is 100, but short-horizon events may need lower)
3. Try different score aggregation: `mode='max'` vs `mode='mean'`
4. Try both PA and non-PA F1
5. Check if the label alignment is correct (t_index matches ground truth labels)
6. Report best F1 for each configuration and explain what went wrong with the old value

### 10% label legacy metrics

For each dataset at **10% labels** (from v31 lf10 surfaces, 3 seeds):

Same computation as above. Load from `experiments/v31/surfaces/` or `experiments/v31/results/phase1/`.

Datasets to compute: SMAP, PSM, SMD, MBA, SKAB, GECCO (if positives exist), BATADAL.

For C-MAPSS: use the Phase A probe results at 10%.

### Output
Save as `results/legacy_metrics.json`:
```json
{
  "SMAP_100": {"pa_f1": X, "f1": X, "auroc": X, "seeds": [X, X, X]},
  "SMAP_10":  {"pa_f1": X, "f1": X, "auroc": X, "seeds": [X, X, X]},
  "GECCO_100": {"f1": X, "best_horizon": X, "best_mode": "max", "investigation": "..."},
  ...
}
```

**Commit after Phase B.**

---

## Phase C: Figure 4 Upgrade — Dense Surface Comparison (~15 min)

Rebuild `fig_probability_surface_v2.pdf` using dense K=150 surfaces from v30 (instead of old sparse v24 data).

### What to do
1. Load v30 FD001 surfaces (FAM and Chronos-2) — dense K=150 horizons
2. Pick 4 visually diverse engines (same selection as hero figure: engines 24, 33, 3, 4)
3. Render: predicted surface, ground truth, error |p - y|, per-horizon AUROC curve
4. Save as `fig_probability_surface_v2.pdf` and `fig_probability_surface_v2.png`
5. Push PNG for local preview

### If v30 surfaces aren't available
Fall back to regenerating them: load v30 checkpoint, run inference on FD001 test set with K=150 horizons, save surface .npz, then render.

**Commit after Phase C.**

---

## Phase D: Self-Check & Paper Table Update (~10 min)

1. Load all results from Phases A-C
2. Print a formatted table matching paper Table 4 structure
3. Verify all placeholders can be filled
4. Create `results/table4_update.txt` with exact LaTeX snippets to paste into paper.tex
5. Flag any results that look suspicious (e.g. RMSE worse than surface-inversion baseline, F1 < 0.10 where h-AUROC > 0.5)

**Final commit with all results.**

---

## File locations

- Pretrained checkpoints: `checkpoints/` or `experiments/v30/ckpts/` or `experiments/archive/v30/`
- v30 surfaces: `experiments/v30/surfaces/` or regenerate from checkpoints
- v31 lf10 surfaces: `experiments/v31/surfaces/` or `experiments/v31/results/phase1/`
- Legacy metric code: `experiments/v21/surface_to_legacy.py`
- Modern eval: `evaluation/surface_metrics.py`
- Grey swan eval: `evaluation/grey_swan_metrics.py`
- train.py: root of `fam-jepa/`

## GPU notes
- RMSE probe trains fast (~33K params, 150 epochs) — should be <2 min per seed per dataset
- Surface loading is CPU-only (numpy), no GPU needed for Phase B
- Phase C rendering is matplotlib, CPU-only
