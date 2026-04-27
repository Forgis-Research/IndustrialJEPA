# V32 Session Summary

NeurIPS 2026 Table 4 SOTA-grade results: rigorous baselines, label efficiency, and protocol-aligned legacy metrics.

## What was computed

- **Phase 1**: SOTA literature research per dataset. Output `results/sota_research.md`.
- **Phase 2**: MSE-RUL probe on the frozen v30 encoder for FD001/FD002/FD003 (3 seeds, hidden-dim sweep, MSE & Huber loss, 1- and 2-layer MLP). 100% and 10% labels. Output `results/rmse_probe.json`.
- **Phase 3**: Legacy metric recomputation (AUROC / AUPRC / PA-F1 / non-PA F1) over horizon sweep {1, 3, 5, 10, 20, 50, 100, 150} for all anomaly datasets at 100% and 10% labels. Deep investigation of GECCO/BATADAL. Output `results/legacy_metrics_full.json`.
- **Phase 4**: Chronos-2 (cached features) and MOMENT (cached features) baselines at 10% labels using identical Chr2MLP head. Output `results/baseline_lf10.json`.
- **Phase 5**: Real dense-K=150 surface comparison figure (FD001, 4 engines). Output `paper-neurips/figures/fig_probability_surface_v2.{pdf,png}`.

## Phase 2 RMSE results (frozen-encoder MSE probe)

| Dataset | Labels | RMSE | NASA-Score | STAR SOTA |
|---------|--------|------|-----------|-----------|
| FD001 | 100% | 18.59 ± 0.32 | 659 | 10.61 |
| FD001 | 10% | 21.57 ± 0.18 | 1351 | - |
| FD002 | 100% | 32.37 ± 0.18 | 12603 | 13.47 |
| FD002 | 10% | 32.36 ± 0.59 | 12427 | - |
| FD003 | 100% | 16.58 ± 0.45 | 610 | 10.71 |
| FD003 | 10% | 23.97 ± 4.47 | 2321 | - |

## Phase 3 Legacy metrics (best across horizon sweep)

| Dataset | Labels | Best non-PA F1 | Best PA-F1 | Best AUROC |
|---------|--------|----------------|-----------|-----------|
| SMAP | 100% | 0.474 | 
| SMAP | 10% | 0.469 | 
| PSM | 100% | 0.575 | 
| PSM | 10% | 0.575 | 
| SMD | 100% | 0.292 | 
| SMD | 10% | 0.253 | 
| MBA | 100% | 0.986 | 
| MBA | 10% | 0.984 | 
| SKAB | 100% | 0.733 | 
| SKAB | 10% | 0.757 | 
| GECCO | 100% | 0.160 | 
| GECCO | 10% | 0.050 | 
| BATADAL | 100% | 0.523 | 
| BATADAL | 10% | 0.522 | 

## Phase 4 Chronos-2 / MOMENT lf10 baselines

| Model | Dataset | Labels | h-AUROC | h-AUPRC |
|-------|---------|--------|---------|---------|
| chr2 | BATADAL_lf10 | 0.534±0.032 | 0.198±0.007 |
| chr2 | BATADAL_lf100 | 0.534±0.032 | 0.198±0.007 |
| chr2 | FD001_lf10 | 0.662±0.009 | 0.676±0.012 |
| chr2 | FD001_lf100 | 0.659±0.000 | 0.688±0.001 |
| chr2 | FD002_lf10 | 0.708±0.003 | 0.598±0.001 |
| chr2 | FD002_lf100 | 0.726±0.001 | 0.602±0.000 |
| chr2 | FD003_lf10 | 0.706±0.004 | 0.504±0.010 |
| chr2 | FD003_lf100 | 0.760±0.003 | 0.600±0.006 |
| chr2 | GECCO_lf10 | 0.780±0.008 | 0.054±0.001 |
| chr2 | GECCO_lf100 | 0.780±0.008 | 0.054±0.001 |
| chr2 | MBA_lf10 | 0.460±0.032 | 0.729±0.032 |
| chr2 | MBA_lf100 | 0.460±0.032 | 0.729±0.032 |
| chr2 | MSL_lf10 | 0.411±0.033 | 0.162±0.010 |
| chr2 | MSL_lf100 | 0.484±0.027 | 0.185±0.010 |
| chr2 | PSM_lf10 | 0.507±0.006 | 0.416±0.009 |
| chr2 | PSM_lf100 | 0.507±0.006 | 0.416±0.009 |
| chr2 | SMAP_lf10 | 0.507±0.000 | 0.264±0.000 |
| chr2 | SMAP_lf100 | 0.509±0.000 | 0.287±0.000 |

## Key findings

TODO: fill after consolidation.
