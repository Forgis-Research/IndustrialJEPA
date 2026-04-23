# V24 Session Summary

**Date**: 2026-04-23 overnight
**Duration**: approximately 10 hours
**Scope**: Canonical architecture first run + Chronos-2 baseline + 3 new-domain datasets + paper polish + ablations

## What shipped

### Main benchmark: 12 datasets, 8 domains
FAM pred-FT with a single $\approx$2.16M-parameter canonical backbone, $P=16$ globally (except $P=1$ for hourly data), 3 seeds per dataset:

| Dataset        | AUPRC       | AUROC       | Legacy SOTA                  | Domain     |
|----------------|-------------|-------------|------------------------------|------------|
| FD001 (turbofan)  | 0.926±0.001 | 0.919±0.001 | RMSE 10.61 (STAR)         | Turbofan   |
| FD002             | 0.908±0.002 | 0.915±0.001 | RMSE 13.47 (STAR)         | Turbofan   |
| FD003             | 0.766±0.009 | 0.876±0.007 | RMSE 10.71 (STAR)         | Turbofan   |
| SMAP              | 0.395±0.010 | 0.594±0.005 | PA-F1 0.336 (MTS-JEPA)    | Spacecraft |
| MSL               | 0.187±0.007 | 0.472±0.015 | PA-F1 0.336 (MTS-JEPA)    | Spacecraft |
| PSM               | 0.425±0.006 | 0.566±0.009 | PA-F1 0.616 (MTS-JEPA)    | Server     |
| SMD               | 0.236±0.015 | 0.680±0.017 | PA-F1 0.925 (AT)          | Server     |
| MBA (cardiac)     | 0.947±0.001 | 0.896±0.003 | -                         | Cardiac    |
| Sepsis (ICU, P=1) | 0.186±0.004 | 0.802±0.003 | AUROC 0.85 (InceptionTime) | ICU        |
| GECCO 2018        | 0.110±0.053 | 0.762±0.057 | F1 0.71 / AUROC 0.88 (TAB) | Water-IoT  |
| BATADAL           | 0.196±0.013 | 0.731±0.025 | AUC 0.972 (Nguyen+24)     | ICS cyber  |
| PhysioNet 2012    | 0.227±0.002 | **0.858±0.001** | AUROC 0.868 (InceptionTime) | ICU-Mort.  |

### Chronos-2 foundation-model baseline
9 datasets with 3 seeds (SMAP 1 seed, SMD skipped on cost):

| Dataset | FAM AUPRC       | Chronos-2 AUPRC | delta (FAM) |
|---------|-----------------|-----------------|-------------|
| FD001   | 0.926           | 0.925           | +0.000  (tie) |
| FD002   | 0.908           | 0.917           | **-0.009** Chronos wins |
| FD003   | 0.766           | 0.794           | **-0.028** Chronos wins |
| SMAP    | 0.395           | 0.285           | **+0.110** FAM wins |
| MSL     | 0.187           | 0.223           | **-0.036** Chronos wins |
| PSM     | 0.425           | 0.411           | **+0.014** FAM wins |
| MBA     | 0.947           | 0.918           | **+0.029** FAM wins |
| GECCO   | 0.110           | 0.032           | **+0.078** FAM wins |
| BATADAL | 0.196           | 0.338           | **-0.142** Chronos wins |

Pattern: FAM wins on SMAP/PSM/MBA/GECCO (domain-outlier distributions), Chronos-2 wins on FD002/FD003/MSL/BATADAL. Asymmetric AUPRC-vs-AUROC on GECCO and BATADAL.

### Three new-domain benchmarks integrated tonight
- **GECCO 2018** (environmental water-quality IoT): 9 ch, 1/min, F1 ~0.71 SOTA
- **BATADAL** (ICS water-cyber): 43 ch, 1/h, AUC 0.972 SOTA
- **PhysioNet 2012** (ICU mortality): 36 ch, 1/h on 48h prefix, AUROC 0.868 SOTA.

FAM reaches within 1 pp AUROC of InceptionTime SOTA on PhysioNet 2012 with a shared recipe across all 12 datasets.

### Ablation: pretrained predictor vs random-init
- FD001: pretrained 0.9257 vs reset 0.9235, delta +0.002 (p=0.38)
- SMAP: pretrained 0.3874 vs reset 0.3950, delta -0.008 (p=0.34)

Both within noise. The value of pred-FT is in the RECIPE (freeze encoder, finetune small head with pos-weighted BCE), not in predictor pretraining.

### Paper deliverables
- Main benchmark table (Tab 1) rewritten for v24
- Abstract reframed (12 datasets / 8 domains, honest Chronos framing)
- Section 5.3 (anomaly) rewritten from stale v18/v19 Mahalanobis numbers
- New "Pretrained vs Random Predictor" ablation section (Sec 6.2)
- Chronos section (Sec 5.5 / Tab 3) extended with all 9 datasets
- New-Domain section (Sec 5.4) replaces unsupported Paderborn
- 2 new figures: fig_probability_surface (real FD001 surface), fig_cross_domain (12-dataset bar chart)
- AUDIT.md (line-by-line discrepancy list) and REVIEW.md (NeurIPS-style critical review)
- ARCHITECTURE_AUDIT.md (off-the-shelf vs novel components)

### What did not ship
- TimesFM-2.5 baseline: JAX-stacked_xf vs Torch-decoder.layers parameter-name mismatch in transformers 4.57 - model lands random-init. Deferred.
- SMD Chronos-2: 327K test obs * ~10 obs/sec = 5+ hours per seed, infeasible on A10G in session budget.
- Deep figure polish (architecture diagram caption): left as-is; existing fig_architecture_ema.pdf is still consistent.
- FD003 capacity-fix: the 0.028 AUPRC gap vs Chronos-2 suggests predictor is undersized for multi-fault regimes. Flagged for follow-up.

## Provenance
Every number in the paper has a JSON file under `fam-jepa/experiments/v24/results/` or a .npz surface under `fam-jepa/experiments/v24/surfaces/`. All probability surfaces are stored so legacy metrics (RMSE, PA-F1, F1) are recomputable without rerunning inference.

## Commits this session
See `git log --oneline` - approximately 30 commits, each pushed immediately after.
