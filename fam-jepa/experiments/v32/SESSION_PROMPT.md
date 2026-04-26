# V32 Overnight Session — Complete Table 4 with Rigorous Baselines

**Duration**: Full overnight (~8-10 hours). Do NOT stop early.  
**Goal**: Fill every `\placeholder{--}` in Table 4 with defensible numbers. Research true SOTA for each dataset, understand their exact metric/protocol, and produce apples-to-apples comparisons. Get competitive C-MAPSS RMSE.  
**Commit cadence**: `git add -A && git commit -m "v32 phase X: <description>" && git push` after each phase. At minimum once per hour.  
**Self-check**: After each phase, verify outputs exist and are reasonable. If a result looks wrong (e.g. F1 < 0.10 where h-AUROC > 0.6), investigate and fix before moving on. Re-run with different hyperparameters if needed.  
**Codex will review** your work after you finish — be thorough and document everything.

---

## Phase 1: SOTA Literature Research (~1 hour)

Before computing anything, research and document the true SOTA for each dataset in Table 4. For each:

1. **What metric does the community use?** (RMSE, NASA score, PA-F1, non-PA F1, AUROC, AUC-PR, TaPR, etc.)
2. **What evaluation protocol?** (threshold on test set? fixed threshold? PA or not? last-cycle only for RMSE?)
3. **What is the current best published number?** (cite paper, year, method name)
4. **Is our current SOTA reference correct?** Check against the actual papers.

### Current SOTA references in paper (VERIFY ALL):
- C-MAPSS: STAR [fan2024star] RMSE 10.61 — is this still SOTA? Check 2024-2025 papers.
- SMAP: MTS-JEPA [mtsjepa2026] PA-F1 0.34 — this seems very low for SOTA. Is this non-PA F1? What do Anomaly Transformer, DCdetector, TimesNet report on SMAP?
- PSM: MTS-JEPA PA-F1 0.62 — same question. What do other methods report?
- SMD: Anomaly Transformer [xu2022anomalytransformer] PA-F1 0.93 — verify. Is this PA-inflated? What's non-PA?
- MBA: no SOTA listed — research what baselines exist for MBA ECG arrhythmia
- SKAB: no SOTA — research
- ETTm1: no SOTA — research (note: ETTm1 is usually forecasting, not anomaly; what's the event detection SOTA?)
- GECCO: batadal2018 F1 0.71 — verify this is the right reference and metric
- BATADAL: batadal2018 AUC 0.97 — verify; this is from the competition, what do modern methods get?

### Key question for each SOTA:
**Are we comparing apples-to-apples?** If SOTA uses PA-F1 and we report non-PA F1 (or vice versa), that's not a valid comparison. Document the exact protocol for each.

### Output
Save as `results/sota_research.md` with a table:
```
| Dataset | SOTA Method | SOTA Value | Metric | Protocol | Paper | Year | Our Metric Match? |
```

**Commit after Phase 1.**

---

## Phase 2: MSE-RUL Probe on Frozen Encoder (~2 hours)

Train a direct-MSE regression head on the frozen v30 encoder for competitive C-MAPSS RMSE.

### Architecture
```
h_t (256-d) -> LayerNorm -> Linear(256, H) -> ReLU -> Dropout(0.1) -> Linear(H, 1) -> clamp(0, 125)
```

### Training config
- **Datasets**: FD001, FD002, FD003 (separate encoder per subset)
- **Checkpoints**: v30 pretrained (find in `checkpoints/` or `experiments/v30/ckpts/` or `experiments/archive/`)
- **Seeds**: 42, 123, 456
- **Labels**: RUL target = min(cycles_remaining, 125)
- **Loss**: MSE
- **Optimizer**: Adam, lr=1e-3, weight_decay=1e-4
- **Scheduler**: ReduceLROnPlateau(patience=5, factor=0.5)
- **Epochs**: 150, early stop on val RMSE, patience 20
- **Batch size**: 64
- **Val split**: last 20% of training engines

### Hidden dim sweep
Try H in {64, 128, 256, 512}. Pick best val RMSE. Report all.

### Also try:
- **2-layer MLP**: LN -> Linear(256,128) -> ReLU -> Linear(128,64) -> ReLU -> Linear(64,1)
- **Huber loss** (delta=10) alongside MSE — report which is better
- **Piece-wise linear RUL**: try both capped at 125 and uncapped

### 10% label probe
After finding best architecture at 100%, re-run with 10% label fraction. Use same engine-sampling as v31 lf10. Report RMSE at 10%.

### Evaluation protocol
**Standard C-MAPSS**: predict RUL at the LAST cycle of each test engine. Compute:
- RMSE = sqrt(mean((pred - true)^2))
- NASA score = sum(s_i) where s_i = exp(-d_i/13) if d_i<0, exp(d_i/10)-1 if d_i>=0
- MAE

### Expected results
- FD001: ~14-20 RMSE (surface-inversion: 36.5; supervised SOTA: 10.61)
- FD002: ~22-30 RMSE (surface-inversion: 44.1; SOTA: 13.47)  
- FD003: ~15-22 RMSE (surface-inversion: 39.5; SOTA: 10.71)

If results are worse than 25/35/25, something is wrong — investigate.

### Output
Save as `results/rmse_probe.json` with all configs, all seeds, mean/std.

**Commit after Phase 2.**

---

## Phase 3: Legacy Metrics — Full Recomputation (~2 hours)

Recompute legacy metrics from stored probability surfaces for ALL datasets, both 100% and 10% labels. Use `experiments/v21/surface_to_legacy.py` as reference code.

### 3A: Understand the pipeline

Read and understand:
- `experiments/v21/surface_to_legacy.py` — `surface_to_anomaly_scores()`, `anomaly_legacy_metrics()`, `best_f1_threshold()`
- `evaluation/grey_swan_metrics.py` — `anomaly_metrics()`, `rul_metrics()`

### 3B: Compute for each dataset at 100% labels

Load v30 surfaces (3 seeds each). For each dataset:

| Dataset | What to compute | Notes |
|---------|----------------|-------|
| C-MAPSS | RMSE (from Phase 2 probe) | Already done |
| SMAP | PA-F1 AND non-PA F1 | Report both; compare to SOTA |
| PSM | PA-F1 AND non-PA F1 | Report both |
| SMD | PA-F1 AND non-PA F1 | Report both |
| MBA | AUROC at Δt=1 | Single-horizon score |
| SKAB | F1 (non-PA) | Threshold-optimized |
| ETTm1 | Best applicable metric | Research what makes sense |
| **GECCO** | F1 — DEEP INVESTIGATION | See below |
| **BATADAL** | F1 — DEEP INVESTIGATION | See below |

### 3C: GECCO and BATADAL investigation

Current F1 values (0.28, 0.24) are suspiciously low given h-AUROC of 0.82 and 0.61. Systematic investigation:

1. Load surface, print shape, check for NaN/Inf
2. Print distribution of predicted probabilities (percentiles: 1, 5, 25, 50, 75, 95, 99)
3. Check label distribution: what fraction of cells are positive?
4. Try ALL of these configurations and report results:
   - `horizon_for_score` in {1, 3, 5, 10, 20, 50, 100, 150}
   - Score aggregation: `mode='max'` vs `mode='mean'` vs `mode='last'`
   - PA vs non-PA F1
   - Different threshold strategies: optimal sweep, fixed 0.5, percentile-based
5. Check label alignment: does t_index match ground truth timestamps? Off-by-one errors?
6. Compare per-horizon AUPRC breakdown — which horizons carry the signal?
7. If F1 remains low: is this genuine (the surface is miscalibrated for point detection) or a bug?

### 3D: Compute for each dataset at 10% labels

Load v31 lf10 surfaces. Same computation as 3B. Note which datasets have usable 10% surfaces.

For C-MAPSS 10%: use Phase 2 probe results.

### 3E: Match SOTA metric protocol exactly

For each dataset where we report a SOTA comparison:
- If SOTA uses PA-F1, we report PA-F1
- If SOTA uses non-PA F1, we report non-PA F1  
- If SOTA uses a specific threshold protocol (e.g. best-F1 on test), we use the same
- Document any protocol mismatches that remain

### Output
Save as `results/legacy_metrics_full.json` with structure:
```json
{
  "SMAP": {
    "100pct": {"pa_f1": X, "f1": X, "auroc": X, "auprc": X, "seeds": {...}},
    "10pct": {"pa_f1": X, "f1": X, "auroc": X, "seeds": {...}},
    "sota_metric": "PA-F1", "sota_value": X, "sota_paper": "...",
    "our_metric_matches_sota": true/false, "notes": "..."
  },
  "GECCO": {
    "100pct": {"f1": X, "best_config": {"horizon": X, "mode": "max", "pa": false}},
    "investigation": "detailed findings...",
    ...
  },
  ...
}
```

**Commit after Phase 3.**

---

## Phase 4: Apples-to-Apples Baseline Comparison at 10% Labels (~2 hours)

This is the differentiating comparison. Run Chronos-2 and MOMENT at 10% labels on key datasets to show FAM's label efficiency advantage isn't just against 100%-label baselines.

### What to run

For datasets where Chr-2 already has 100% h-AUROC results, run the SAME pipeline at 10% labels:
- Load Chr-2 frozen encoder
- Train identical 198K-param MLP head with 10% label fraction
- 3 seeds
- Same eval protocol

Datasets: C-MAPSS (FD001), SMAP, PSM, MBA (wherever Chr-2 100% results exist)

### Also: MOMENT at 10% if feasible
Same pipeline with MOMENT encoder. Lower priority than Chr-2.

### Fill the Chr-2 10% column in Table 4
Currently all "---" in the Chr-2 10% cells. Fill with actual numbers.

### Output
Save as `results/baseline_lf10.json`.

**Commit after Phase 4.**

---

## Phase 5: Figure 4 Upgrade — Dense Surface Comparison (~30 min)

Rebuild `fig_probability_surface_v2.pdf` using dense K=150 surfaces from v30.

1. Load v30 C-MAPSS surfaces (FAM and Chronos-2) — dense K=150
2. Pick 4 visually diverse engines (24, 33, 3, 4)
3. Render: predicted surface, ground truth, |p - y| error, per-horizon AUROC curve
4. Save as PDF and PNG
5. Push PNG for local preview

If v30 surfaces aren't available, regenerate from checkpoint.

**Commit after Phase 5.**

---

## Phase 6: Self-Check, Consolidation & Paper Update (~1 hour)

### 6A: Verify all results
- Load everything from Phases 1-5
- Cross-check: does our legacy metric improve when h-AUROC is higher? If not, investigate.
- Flag any result where FAM loses badly — understand why

### 6B: Generate LaTeX for Table 4
Print exact LaTeX snippets to replace every `\placeholder{--}` in paper.tex Table 4.
Save as `results/table4_latex.txt`.

### 6C: Update RESULTS.md
Add v32 results to the master results table.

### 6D: Summary document
Write `results/SESSION_SUMMARY.md`:
- What was computed
- Key findings (surprises, failures, wins)
- Remaining gaps (if any)
- Recommendations for paper text changes

**Final commit with everything.**

---

## File locations

- Pretrained checkpoints: `checkpoints/` or `experiments/v30/ckpts/` or `experiments/archive/`
- v30 surfaces: `experiments/v30/surfaces/` (or regenerate)
- v31 lf10 surfaces: `experiments/v31/surfaces/` or `experiments/v31/results/phase1/`
- Legacy metric code: `experiments/v21/surface_to_legacy.py`
- Modern eval: `evaluation/surface_metrics.py`
- Grey swan eval: `evaluation/grey_swan_metrics.py`
- Baseline code: `experiments/v31/baseline_extend_all.py`
- train.py: root of `fam-jepa/`

## Principles

- **Rigour over speed.** If a number looks wrong, don't ship it. Investigate.
- **Document protocol mismatches.** If our F1 isn't comparable to SOTA's F1, say so explicitly.
- **No cherry-picking.** Report mean ± std over 3 seeds. If one seed is bad, report it and explain.
- **Commit hourly.** Even partial results are valuable.
- **Don't stop early.** Use the full overnight window. If main phases finish early, run additional sweeps (more seeds, more hyperparameters, additional baselines).
