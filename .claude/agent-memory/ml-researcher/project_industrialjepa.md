---
name: IndustrialJEPA / FAM Project Context
description: V21 (Apr 22): AUPRC as primary metric, pos-weighted BCE, per-horizon sigmoid. 8 datasets, probability surfaces stored as .npz. Paper reframed from grey swan to event prediction.
type: project
---

## Current State (v21, 2026-04-22)

### Paper: FAM (Forecast-Anything Model)
- Title: "FAM: A Self-Supervised Forecast-Anything Model for Multivariate Time Series Events"
- Venue: NeurIPS 2026
- Repo renamed: mechanical-jepa → fam-jepa

### Two Contributions
1. **Predictor finetuning**: freeze encoder, finetune predictor (790K params) + per-horizon sigmoid head
2. **Multi-domain validation**: same 2.37M arch across 8 datasets, unified AUPRC metric

### Evaluation (v21+)
- **Primary**: AUPRC pooled over probability surface p(t, Δt) — threshold-free, imbalance-robust
- **Secondary**: AUROC pooled over same cells
- **Legacy**: RMSE (C-MAPSS), PA-F1 (anomaly) — derived from stored .npz surfaces
- **Training**: pos-weighted BCE on per-horizon sigmoid (replaces v20's MSE on scalar RUL)
- **Storage**: every run saves p_surface + y_surface as .npz

### Key code locations
- `fam-jepa/evaluation/surface_metrics.py` — evaluate_probability_surface() (PRIMARY)
- `fam-jepa/evaluation/losses.py` — weighted_bce_loss(), build_label_surface()
- `fam-jepa/data/config.py` — central path config
- `fam-jepa/experiments/v21/SESSION_PROMPT.md` — overnight prompt

### Datasets
C-MAPSS FD001/002/003, SMAP, MSL, PSM, SMD, MBA (8 total)

### Best results so far (v20, F1w metric — to be rerun with AUPRC in v21)
- FD001 pred-FT 100%: F1w 0.391±0.085, RMSE 16.90±1.71
- FD001 pred-FT 5%: F1w 0.261±0.165 (beats E2E + scratch)
- SMAP Mahalanobis: PA-F1 0.793±0.014 (beats MTS-JEPA 0.336)
- SIGReg-pred: best pretraining variant (F1w 0.451, RMSE 13.71)
