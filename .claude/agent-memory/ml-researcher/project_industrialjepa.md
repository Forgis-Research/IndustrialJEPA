---
name: IndustrialJEPA / FAM Project Context
description: v31 complete (Apr 26): TimesFM, Moirai, FEMTO baselines done; theory A1' integrated; paper 20+ pages clean compile
type: project
---

## Current State (v31 COMPLETE, 2026-04-26)

### Paper: FAM (Forecast-Anything Model)
- Title: "FAM: Self-Supervised Event Prediction via Causal JEPA for Industrial Time Series"
- Venue: NeurIPS 2026
- File: `paper-neurips/paper.tex` (20+ pages, clean pdflatex compile)
- Architecture: 2.16M params (encoder 256-d, L=2, 4 heads; predictor 790K MLP; dense K=150 event head)

### Primary metric: h-AUROC
- Per-horizon mean AUROC (not pooled) across valid horizons (0.001 < prev < 0.999)
- FAM uses K=150 horizons; chr2-mlp/MOMENT use sparse K=7/8
- Surfaces stored as .npz per run for metric recomputation

### Two Contributions
1. **Predictor finetuning (pred-FT)**: freeze encoder, finetune predictor + per-horizon sigmoid head (198K total)
2. **Multi-domain validation**: same 2.16M arch across 11 datasets, unified h-AUROC metric

### Main Results (v30/v31, 100% labels, 3 seeds)
| Dataset | FAM h-AUROC | Chr2-mlp | MOMENT-1-large | TimesFM-200M | Moirai-91M |
|---------|------------|----------|----------------|--------------|------------|
| FD001   | 0.786 +/- 0.033 | 0.659 +/- 0.000 | 0.559 +/- 0.009 | 0.530 +/- 0.003 | 0.606 +/- 0.004 |
| FD002   | 0.566 +/- 0.011 | **0.734 +/- 0.001** | - | - | - |
| FD003   | 0.853 +/- 0.004 | 0.760 +/- 0.003 | 0.473 +/- 0.012 (!) | 0.615 +/- 0.014 | 0.700 +/- 0.004 |
| SMAP    | 0.598 +/- 0.036 | 0.534 (1 seed) | - | - | - |
| PSM     | 0.562 +/- 0.013 | 0.506 +/- 0.010 | - | - | - |
| MBA     | 0.739 +/- 0.014 | 0.451 +/- 0.017 | 0.791 +/- 0.009 | 0.759 +/- 0.006 | 0.571 +/- 0.017 |
| GECCO   | 0.819 +/- 0.064 | **0.826 +/- 0.003** | - | - | - |
| BATADAL | 0.607 +/- 0.033 | 0.534 +/- 0.032 | 0.537 +/- 0.066 | 0.653 +/- 0.005 | 0.360 +/- 0.010 (!) |
| SKAB    | 0.707 +/- 0.017 | - | - | - | - |
| ETTm1   | 0.869 +/- 0.002 | - | - | - | - |
| SMD     | 0.654 +/- 0.004 | - | - | - | - |
| FEMTO   | 0.575 +/- 0.008 | - | - | - | - |

FAM wins: 6/8 vs Chr-2 (loses FD002, GECCO near-tie); 3/4 vs MOMENT; 2/4 vs TimesFM; 4/4 vs Moirai.
Honest negatives: TimesFM beats FAM on MBA (+0.020) and BATADAL (+0.046). MOMENT beats FAM on MBA (+0.052).
Moirai BATADAL=0.360 is BELOW CHANCE - worst result across all baselines.

### Foundation Model Baselines (v31, 2026-04-26)
Three additional foundation models evaluated with identical 198K dt-MLP head, 3 seeds x 4 datasets.

**TimesFM-1.0-200M** (timesfm-1.0-200m-pytorch on HF):
- Hook on model.stacked_transformer -> mean-pool patches -> 1280-d embeddings
- Do NOT use timesfm-2.0-500m-pytorch (checkpoint mismatch with timesfm 1.3.0 library)
- Results: FD001=0.530, FD003=0.615, MBA=0.759 (beats FAM), BATADAL=0.653 (beats FAM)

**Moirai-1.1-R-base** (Salesforce/moirai-1.1-R-base, uni2ts package):
- Hook on model.encoder -> 768-d embeddings, univariate patching (patch_size=32)
- Results: FD001=0.606, FD003=0.700, MBA=0.571, BATADAL=0.360 (below chance)
- Citation: woo2024unified (arXiv:2402.02592) - added to references.bib

**FEMTO bearing dataset**:
- 6 training bearings (SSL pretraining), 17 test bearings
- 8 features: RMS, peak, kurtosis, crest factor per H+V channel
- FAM h-AUROC = 0.575 +/- 0.008 (3s, 95% CI [0.556, 0.594]) - modest but above chance
- SSL-starved: only 6 bearings; val loss diverges (train-test distribution shift)
- Loader: fam-jepa/data/femto.py (nested zip reader)

### Theory: A1' Calibrated Posterior (v31 continuation)
- A1' (ass:calibrated_posterior): P(eta(H*) in [eta_under, eta_over]) = 1
- C_eta = (2*eta_under*(1-eta_over))^{-1} is tighter than C_p (uses posterior bounds not marginal)
- Per-horizon Prop added to appendix (app:per_horizon) with explicit Δt-dependence
- Proof sketch in theory_main.tex updated to cite A1' explicitly
- theory_appendix.tex: Jensen gap step updated; eq:jensen_gap_conditional uses eta_under/eta_over

### Label Efficiency (v31 Phase 1, 10% labels)
- FD001: 0.772 +/- 0.059 (98% retention)
- FD003: 0.830 +/- 0.018 (97%)
- SMAP: 0.580 +/- 0.047 (97%)
- Two regimes: lifecycle (slow-drift) retains >97%; streaming anomaly (single-entity) degrades faster

### Sub-5% Label Efficiency (v31 Phase 2)
- FD001 at 2% labels (2 of 85 engines): 0.724 +/- 0.013 = 92% retention - KEY PAPER RESULT
- FD001 at 5% labels (4 engines): 0.730 +/- 0.018 = 93% retention

### Critical Bugs Fixed in v31
1. **label_fraction bug**: For single-entity datasets, entity subsampling was a no-op. Fix: temporal truncation.
2. **chr2-mlp table errors**: 5 wrong values in paper table. All corrected.
3. **build_label_surface squeeze bug (MOMENT)**: Returns (N,1,K) not (N,K). Must call .squeeze(1).

### Paper Appendix Sections (complete)
- app:chronos_full - FAM vs Chr-2 matched head
- app:moment_full - MOMENT-1-large (FAM wins 3/4)
- app:extra_baselines (v31 new) - TimesFM + Moirai table
- app:additional_ablations - predictor ablations + SIGReg ablation + sub-5% table
- app:theory - full proofs with A1' + per-horizon Prop (app:per_horizon)

### Key files
- `paper-neurips/paper.tex` - main paper (v31 complete, clean compile)
- `fam-jepa/experiments/RESULTS.md` - master results table
- `fam-jepa/data/femto.py` - FEMTO bearing data loader (nested zip reader)
- `fam-jepa/experiments/v31/run_all_baselines.py` - combined TimesFM+Moirai+FEMTO runner
- `fam-jepa/experiments/v31/results/timesfm_baseline.json` - TimesFM results
- `fam-jepa/experiments/v31/results/moirai_baseline.json` - Moirai results
- `fam-jepa/experiments/v31/results/femto_baseline.json` - FEMTO FAM results

### Architecture decision: dense K=150 head
- Dense CDF head (K=150 horizons for lifecycle, K=8 for streaming) adopted v30
- MonotoneCDF failed (loss climbed, val stuck at chance)

**Why:** Pooled AUPRC trap: pooled metric inflates when horizon base rates drift across Δt. Use per-horizon mean AUROC (h-AUROC) as primary metric.
**How to apply:** Never report evaluate_probability_surface()['auroc'] as the primary h-AUROC. Compute per-horizon AUROC explicitly using roc_auc_score per column of p_surface/y_surface.

### train.py API notes (for next session)
- PretrainDataset: kwargs are n_cuts, max_context, delta_t_max, delta_t_min, seed (NOT max_future)
- finetune() returns {'best_val': ..., 'final_epoch': ...} (NOT 'best_loss')
- evaluate() returns {'primary': {'auprc', 'auroc', ...}, 'per_horizon': ..., 'p_surface', 'y_surface', 't_index'}
- collate_event returns (ctx_padded, ctx_mask, ttes, ts) - 4-tuple, NOT (x, y, dt_indices)
- FAM norm_mode: 'revin', 'none', 'last_value', 'revin_stat' (NOT 'zscore' - use 'none' for pre-normalized data)
