---
name: IndustrialJEPA / FAM Project Context
description: v31 (Apr 26): FAM beats Chr-2 6/8 datasets; MOMENT-1-large (341M) BELOW chance on FD003; label efficiency two-regime; paper at NeurIPS 2026
type: project
---

## Current State (v31, 2026-04-26)

### Paper: FAM (Forecast-Anything Model)
- Title: "FAM: Self-Supervised Event Prediction via Causal JEPA for Industrial Time Series"
- Venue: NeurIPS 2026
- File: `paper-neurips/paper.tex` (20 pages, clean compile)
- Architecture: 2.16M params (encoder 256-d, L=2, 4 heads; predictor 790K MLP; dense K=150 event head)

### Primary metric: h-AUROC
- Per-horizon mean AUROC (not pooled) across valid horizons (0.001 < prev < 0.999)
- FAM uses K=150 horizons; chr2-mlp/MOMENT use sparse K=7/8
- Surfaces stored as .npz per run for metric recomputation

### Two Contributions
1. **Predictor finetuning (pred-FT)**: freeze encoder, finetune predictor + per-horizon sigmoid head (198K total)
2. **Multi-domain validation**: same 2.16M arch across 11 datasets, unified h-AUROC metric

### Main Results (v30/v31, 100% labels, 3 seeds)
| Dataset | FAM h-AUROC | Chr2-mlp | MOMENT-1-large |
|---------|------------|----------|----------------|
| FD001   | 0.786 ± 0.033 | 0.659 ± 0.000 | 0.559 ± 0.009 |
| FD002   | 0.566 ± 0.011 | **0.734 ± 0.001** | - |
| FD003   | 0.853 ± 0.004 | 0.760 ± 0.003 | 0.473 ± 0.012 (below chance!) |
| SMAP    | 0.598 ± 0.036 | 0.534 (1 seed) | - |
| PSM     | 0.562 ± 0.013 | 0.506 ± 0.010 | - |
| MBA     | 0.739 ± 0.014 | 0.451 ± 0.017 | - |
| GECCO   | 0.819 ± 0.064 | **0.826 ± 0.003** | - |
| BATADAL | 0.607 ± 0.033 | 0.534 ± 0.032 | ~0.520 ± 0.060 (3s pending) |
| SKAB    | 0.707 ± 0.017 | - | - |
| ETTm1   | 0.869 ± 0.002 | - | - |
| SMD     | 0.654 ± 0.004 | - | - |

FAM wins 6/8 vs Chr-2 (loses FD002, GECCO near-tie)

### MOMENT Baseline (v31, 2026-04-26)
- MOMENT-1-large (341.2M params, AutonLab/MOMENT-1-large)
- Per-channel univariate embeddings (1024-d) mean-pooled, same 198K MLP head
- KEY FINDING: MOMENT FD003=0.473 BELOW CHANCE despite 158x more params than FAM
- This confirms domain-specific cross-channel JEPA >> scale for event prediction
- MBA skipped (data format incompatibility in MOMENT evaluation harness)

### Label Efficiency (v31 Phase 1, 10% labels)
- FD001: 0.772 ± 0.059 (98% retention)
- FD003: 0.830 ± 0.018 (97%)
- SMAP: 0.580 ± 0.047 (97%)
- Two regimes: lifecycle (slow-drift) retains >97%; streaming anomaly (single-entity) degrades faster

### Critical Bugs Fixed in v31
1. **label_fraction bug**: For single-entity datasets, entity subsampling was a no-op (max(1, round(1*0.1))=1). Fix: temporal truncation to first label_fraction*T steps (min 256).
2. **chr2-mlp table errors**: 5 wrong values in paper table (copied across rows). All corrected.
3. **build_label_surface squeeze bug (MOMENT)**: Returns (N,1,K) not (N,K). Must call .squeeze(1).

### Key files
- `paper-neurips/paper.tex` - main paper (v31 complete)
- `fam-jepa/experiments/RESULTS.md` - master results table
- `fam-jepa/experiments/v31/baseline_moment.py` - MOMENT baseline (with squeeze fix)
- `fam-jepa/experiments/v31/surfaces/` - all NPZ surfaces (gitignored)
- `fam-jepa/experiments/v30/surfaces/` - v30 surfaces including chr2-mlp

### Architecture decision: dense K=150 head
- Dense CDF head (K=150 horizons for lifecycle, K=8 for streaming) adopted v30
- Beats sparse K=8 head by +0.07 h-AUROC on FD001
- MonotoneCDF failed (loss climbed, val stuck at chance)

**Why:** Pooled AUPRC trap: pooled metric inflates when horizon base rates drift across Δt. Use per-horizon mean AUROC (h-AUROC) as primary metric. Report pooled AUPRC as secondary only.
**How to apply:** Never report evaluate_probability_surface()['auroc'] as the primary h-AUROC. Compute per-horizon AUROC explicitly using roc_auc_score per column of p_surface/y_surface.
