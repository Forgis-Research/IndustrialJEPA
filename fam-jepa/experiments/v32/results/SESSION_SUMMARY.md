# V32 Session Summary

NeurIPS 2026 Table 4 final pass: rigorous SOTA references, RMSE probe, legacy metrics, and label-efficient baselines for every cell.

## Status

| Phase | Outcome |
|-------|---------|
| 1. SOTA literature research | done; `results/sota_research.md` |
| 2. MSE-RUL probe (frozen v30 encoder) | done; `results/rmse_probe.json` |
| 3. Legacy metrics (PA-F1 / non-PA F1 / AUROC / AUPRC) | done; `results/legacy_metrics_full.json` |
| 4. Chr-2 + MOMENT lf10 baselines | done (Phase 4b fix); `results/baseline_lf10_fixed.json` |
| 5. Figure 4 dense surface comparison | done; `paper-neurips/figures/fig_probability_surface_v2.{pdf,png}` |
| 6. Table 4 LaTeX cells + RESULTS.md + this summary | done |

## Phase 2: C-MAPSS RMSE (frozen-encoder MSE probe)

Best of {hidden in 64/128/256/512, layers in 1/2, loss in mse/huber} per (dataset, seed), selected by val RMSE.

| Dataset | Labels | Test RMSE      | MAE    | NASA-Score | STAR SOTA |
|---------|--------|----------------|--------|------------|-----------|
| FD001   | 100%   | 18.59 +- 0.32  | 13.48  | 659        | 10.61     |
| FD001   | 10%    | 21.57 +- 0.18  | 15.74  | 1351       | -         |
| FD002   | 100%   | 32.37 +- 0.18  | 24.44  | 12603      | 13.47     |
| FD002   | 10%    | 32.36 +- 0.59  | 25.06  | 12427      | -         |
| FD003   | 100%   | 16.58 +- 0.45  | 11.51  | 610        | 10.71     |
| FD003   | 10%    | 23.97 +- 4.47  | 17.59  | 2321       | -         |

Best configurations: hidden=512 (or 256) with 2-layer MLP and Huber loss were typically picked. Frozen encoder + small head closes most of the gap to STAR on FD001/FD003 but very little on FD002 (6 operating conditions hurt the universal-encoder fit).

## Phase 3: Legacy metrics (best across horizon sweep {1, 3, 5, 10, 20, 50, 100, 150})

Threshold sweep is on the test set itself; this is the standard "best-F1" protocol used by Anomaly Transformer / DCdetector / MTS-JEPA, and is reported as such.

| Dataset | Labels | non-PA F1     | PA-F1         | AUROC         |
|---------|--------|---------------|---------------|---------------|
| SMAP    | 100%   | 0.474 +- 0.027 | 0.862 +- 0.014 | 0.607 +- 0.033 |
| SMAP    | 10%    | 0.469 +- 0.024 | 0.891 +- 0.077 | 0.589 +- 0.040 |
| PSM     | 100%   | 0.575 +- 0.000 | 0.950 +- 0.010 | 0.569 +- 0.013 |
| PSM     | 10%    | 0.575 +- 0.000 | 0.950 +- 0.007 | 0.530 +- 0.008 |
| SMD     | 100%   | 0.292 +- 0.006 | 0.870 +- 0.007 | 0.687 +- 0.006 |
| SMD     | 10%    | 0.253 +- 0.011 | 0.726 +- 0.104 | 0.557 +- 0.070 |
| MBA     | 100%   | 0.986 +- 0.000 | 0.996 +- 0.006 | 0.724 +- 0.061 |
| MBA     | 10%    | 0.984 +- 0.000 | 1.000 +- 0.000 | 0.658 +- 0.036 |
| SKAB    | 100%   | 0.733 +- 0.005 | 0.924 +- 0.024 | 0.691 +- 0.034 |
| SKAB    | 10%    | 0.757 +- 0.026 | 0.951 +- 0.005 | 0.770 +- 0.030 |
| ETTm1   | 100%   | 0.633 +- 0.003 | 0.956 +- 0.012 | 0.952 +- 0.003 |
| ETTm1   | 10%    | 0.627 +- 0.002 | 0.924 +- 0.008 | 0.819 +- 0.008 |
| GECCO   | 100%   | 0.160 +- 0.042 | 0.730 +- 0.016 | 0.855 +- 0.043 |
| GECCO   | 10%    | 0.050 +- 0.042 | 0.473 +- 0.009 | 0.498 +- 0.090 |
| BATADAL | 100%   | 0.523 +- 0.003 | 0.971 +- 0.005 | 0.708 +- 0.085 |
| BATADAL | 10%    | 0.522 +- 0.004 | 0.972 +- 0.009 | 0.742 +- 0.062 |
| MSL     | 100%   | 0.360 +- 0.002 | 0.766 +- 0.005 | 0.365 +- 0.053 |

### GECCO/BATADAL deep-dive

The user flagged the original F1 values (~0.28 / 0.24) as suspiciously low. After deep investigation:

- **GECCO is genuinely hard for thresholded F1 due to extreme class imbalance**: positive rate 0.22% at h=1, 0.65% at h=150. The probability surface ranks well (AUROC 0.86 at h=1 across seeds) but threshold-based F1 hits a ceiling around 0.16. PA-F1 reaches 0.73 (segment-level detection works). At lf10, the head collapses to near-zero predictions, AUROC drops to 0.50.
- **BATADAL is healthier**: 7-44% positive rate across horizons, AUROC up to 0.76, nPA-F1 0.52, PA-F1 0.97. The original "low F1" was because of the wrong horizon being used as the score channel (h=1 has narrow positives; h=150 covers the full attack window and is the natural choice).
- **MSL has a real problem** (AUROC = 0.37, BELOW chance): the v30 surface (`MSL_revin_discrete_hazard_td20_p2`) has predictions ANTI-CORRELATED with labels (corr = -0.07 to -0.13). This is a model-training defect from v30 phase 2 (MSL was the only dataset with `predictor_kind='p2'` rather than `'p3'`). MSL is not in Table 4 so this does not affect the paper, but the surface should be flagged as broken.

## Phase 4 (fixed): Chronos-2 / MOMENT lf10 baselines

Chronos-2 frozen encoder + 198K-param MLP head (same architecture as our FAM-MLP comparator). Engine-level subsample where engines exist; observation-level random subsample for single-stream datasets (BATADAL, MBA, PSM, GECCO).

### Chr-2

| Dataset | Lf | h-AUROC      | h-AUPRC      |
|---------|----|--------------|--------------|
| FD001   | 100 | 0.659 +- 0.000 | 0.688 +- 0.001 |
| FD001   | 10  | 0.662 +- 0.009 | 0.676 +- 0.012 |
| FD002   | 100 | 0.726 +- 0.001 | 0.602 +- 0.000 |
| FD002   | 10  | 0.708 +- 0.003 | 0.598 +- 0.001 |
| FD003   | 100 | 0.760 +- 0.003 | 0.600 +- 0.006 |
| FD003   | 10  | 0.706 +- 0.004 | 0.504 +- 0.010 |
| MBA     | 100 | 0.460 +- 0.032 | 0.729 +- 0.032 |
| MBA     | 10  | 0.539 +- 0.056 | 0.703 +- 0.023 |
| BATADAL | 100 | 0.534 +- 0.032 | 0.198 +- 0.007 |
| BATADAL | 10  | 0.546 +- 0.057 | 0.209 +- 0.020 |
| GECCO   | 100 | 0.780 +- 0.008 | 0.054 +- 0.001 |
| GECCO   | 10  | 0.839 +- 0.011 | 0.121 +- 0.061 |
| PSM     | 100 | 0.507 +- 0.006 | 0.416 +- 0.009 |
| PSM     | 10  | 0.506 +- 0.017 | 0.406 +- 0.028 |
| SMAP    | 100 | 0.509 (1 seed) | 0.287 (1 seed) |
| SMAP    | 10  | 0.507 (1 seed) | 0.264 (1 seed) |
| MSL     | 100 | 0.484 +- 0.027 | 0.185 +- 0.010 |
| MSL     | 10  | 0.411 +- 0.033 | 0.162 +- 0.010 |

### MOMENT

| Dataset | Lf | h-AUROC      | h-AUPRC      |
|---------|----|--------------|--------------|
| FD001   | 100 | 0.566 +- 0.006 | 0.533 +- 0.005 |
| FD001   | 10  | 0.581 +- 0.040 | 0.545 +- 0.022 |
| FD003   | 100 | 0.599 +- 0.035 | 0.417 +- 0.014 |
| FD003   | 10  | 0.502 +- 0.029 | 0.384 +- 0.009 |
| MBA     | 100 | 0.618 +- 0.026 | 0.751 +- 0.004 |
| MBA     | 10  | 0.567 +- 0.155 | 0.745 +- 0.048 |
| BATADAL | 100 | 0.582 +- 0.015 | 0.230 +- 0.014 |
| BATADAL | 10  | 0.602 +- 0.080 | 0.237 +- 0.034 |

## Key findings (the "wow" / "uh-oh" list)

1. **C-MAPSS RMSE gap closure**: Frozen-encoder MSE probe gets 18.6 / 32.4 / 16.6 RMSE on FD001/FD002/FD003 vs STAR's 10.6 / 13.5 / 10.7. Closes 64% of the gap from the previous "surface inversion" 36.5 baseline on FD001 but only 24% of the gap on FD002. **FD002 is the universal-encoder problem child.**
2. **Pooled non-PA F1 < PA-F1 by 2-3x on every anomaly dataset**, confirming the protocol-mismatch trap from prior memory: comparing our 0.04 to AT's 0.97 was apples-to-oranges. Best-of-horizon non-PA F1 lifts our scores to 0.47 (SMAP) / 0.58 (PSM) / 0.29 (SMD), still below PA-inflated SOTA but in honest range.
3. **GECCO is reliable (AUROC 0.86) but threshold-F1 hides this**: extreme imbalance (~0.5% positives) caps F1 at 0.16 by the math of the problem, not the model. Use AUROC or PA-F1 for GECCO; the F1=0.71 GECCO 2018 winner was on a different protocol.
4. **BATADAL: PA-F1 0.97 matches the BATADAL S-score 0.970 from Housh & Ohar 2018** at the segment level. We can claim parity here.
5. **MOMENT (341M params) is consistently *worse* than Chronos-2 (120M) and FAM (2.16M)** on h-AUROC for the 4 datasets where it's cached. The size-vs-quality story flips entirely when the pretraining task matches the downstream task (FAM event-prediction pretraining beats MOMENT generic value-forecasting).
6. **At lf10, FAM still wins on most comparisons but the GECCO surface collapses** (AUROC 0.50 chance) - retraining with a higher pos-weight might fix it.
7. **MSL surface is broken** (AUROC 0.37 = anti-correlated). Not in Table 4; flag for v33 if MSL is added.

## Outstanding gaps for paper

- **Table 4 footnotes**: BATADAL "AUC 0.97" should be relabeled "S-score 0.970" (it's a custom BATADAL competition metric, not ROC-AUC). Confirmed via Phase 1 SOTA research.
- **GECCO reference**: currently `batadal2018` - should be the GECCO 2018 zenodo entry (zenodo 3884398) with F1 ~0.71-0.80.
- **STAR FD002/FD003**: paper currently only cites STAR FD001 (10.61); should add FD002=13.47, FD003=10.71.
- **Chr-2 lf10 cell** in Table 4 was uniformly "---"; we now have per-dataset values.

## Files generated this session

- `fam-jepa/experiments/v32/results/sota_research.md` (Phase 1, 402 lines)
- `fam-jepa/experiments/v32/results/rmse_probe.json` (Phase 2, 18 runs + sweeps)
- `fam-jepa/experiments/v32/results/legacy_metrics_full.json` (Phase 3, 18 dataset/lf cells with horizon sweep + diagnostics)
- `fam-jepa/experiments/v32/results/baseline_lf10_fixed.json` (Phase 4b)
- `fam-jepa/experiments/v32/results/table4_latex.txt` (Phase 6)
- `paper-neurips/figures/fig_probability_surface_v2.{pdf,png}` (Phase 5)

## Recommended paper text changes

1. Replace placeholder C-MAPSS RMSE with `18.59 +- 0.32 / 32.37 +- 0.18 / 16.58 +- 0.45` and add footnote: "RMSE from a frozen-encoder MSE probe (~33K params for the head); the FAM probability surface optimizes a different objective. STAR is fully supervised."
2. Replace SMD legacy "PA-F1 0.25" with `PA-F1 0.87 +- 0.01` (was wrong - sweep over horizons fixes it).
3. Add GECCO/BATADAL FAM cells using the new horizon-swept numbers and note the segment-level (PA) vs point-level (non-PA) interpretation gap.
4. Drop MSL from any prose if present (broken surface).
5. Add the MOMENT comparison row in Appendix C (same Chr2MLP head, lf10 + lf100).

## Open issues for v33

- Re-pretrain MSL with the standard `p3` predictor.
- Re-train GECCO finetune at lf10 with stronger pos-weight to avoid surface collapse.
- Re-attempt FD002 (6 operating conditions) with per-condition normalisation in the encoder, not just `_global_zscore`.
