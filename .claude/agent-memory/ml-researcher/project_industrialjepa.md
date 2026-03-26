---
name: IndustrialJEPA Project Context
description: Core project overview — physics-informed attention for industrial time series, 52 experiments complete (Phase 7 overnight session added robotics/JEPA)
type: project
---

IndustrialJEPA is a research project on physics-informed channel grouping in transformer attention for multivariate time series forecasting on mechanical/industrial systems.

**Status as of 2026-03-26**: 52 experiments complete. Physics-mask story confirmed. Phase 7 added robotics datasets (KUKA, AURSAD, Voraus) and JEPA pretraining experiments.

**Core finding**: Physics-masked attention provides principled constraint when physics groups are statistically independent. When groups are coupled (C-MAPSS, ETT) it does not help or hurts.

| System | Physics Mask Effect | Why |
|---|---|---|
| Double Pendulum | +7.4% over Full-Attn (p=0.0002) | Groups are truly independent |
| C-MAPSS | ≈ random mask (p=0.528) | Correlated degradation |
| ETT Weather | -1.3% vs Full-Attn | Thermal couples to all loads |
| All vs CI-Trans | +5–34% | 2D treatment always helps |

**Phase 7 findings (Exp 49-52)**:
- **Exp 49 (KUKA EDA)**: GCS credentials unavailable → physics-based synthetic peg-insertion data. 12% success rate, force_mag_mean=0.617, weak joint-force correlation.
- **Exp 50 (Force prediction)**: CI-Transformer fails on cross-channel causal tasks (MSE ≈ Linear = 0.083). Full-Attn best (0.0052), PhysMask competitive at 2.6x fewer params. Key insight: CI when channels independent, physics grouping when causal structure known.
- **Exp 51 (AURSAD JEPA)**: 30-epoch JEPA on 151k windows → AUROC 0.5482±0.0111 vs random 0.5473±0.0038. Delta +0.0009. NO_BENEFIT.
- **Exp 52 (Voraus JEPA)**: 30-epoch JEPA on 51k windows → AUROC 0.4828±0.0111 vs random 0.5083±0.0192. Delta -0.0254. NO_BENEFIT (JEPA slightly worse).
- **JEPA pattern**: Consistent NO_BENEFIT on two independent industrial datasets. Temporal patch prediction doesn't capture episode-level distributional shift = industrial anomaly. Better approaches: MAE, contrastive learning, or density estimation baselines.

**Engineering notes from Phase 7**:
- FactoryNetDataset OOM on Voraus (60 parquets × 107 MB = 6.25 GB). Fix: stream parquets one at a time, extract windows per-file.
- Stanford KUKA dataset (gs://gresearch/robotics) requires Google Cloud credentials — unavailable on SageMaker.
- Pre-materialization (numpy array from FactoryNetDataset) works for AURSAD (151k windows, ~1.6 GB) but not Voraus.

**Paper title**: "When to Mask: Physics-Informed Attention for Multivariate Time Series"

**Why:** Phase 7 complete 2026-03-26. Physics-mask story confirmed. JEPA results are negative but informative.
**How to apply:** When discussing experiments or datasets, reference the 4-tier narrative and all findings including the Phase 7 robotics/JEPA additions.
