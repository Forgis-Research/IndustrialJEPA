---
name: IndustrialJEPA Project Context
description: Core project overview — physics-informed attention for industrial time series, 48 experiments complete, paper-ready
type: project
---

IndustrialJEPA is a research project on physics-informed channel grouping in transformer attention for multivariate time series forecasting on mechanical/industrial systems.

**Status as of 2026-03-26**: 48 experiments complete across 3 tiers. Story is clear; paper-writing phase.

**Core finding**: Physics-masked attention provides principled constraint when physics groups are statistically independent. When groups are coupled (C-MAPSS, ETT) it does not help or hurts.

| System | Physics Mask Effect | Why |
|---|---|---|
| Double Pendulum | +7.4% over Full-Attn (p=0.0002) | Groups are truly independent |
| C-MAPSS | ≈ random mask (p=0.528) | Correlated degradation |
| ETT Weather | -1.3% vs Full-Attn | Thermal couples to all loads |
| All vs CI-Trans | +5–34% | 2D treatment always helps |

**Paper title**: "When to Mask: Physics-Informed Attention for Multivariate Time Series"

**3 research directions**:
1. SparseGraph-Transformer
2. Slot-Concept Transformer
3. Brain-JEPA inspired (needs large-scale dataset)

**Why:** 48 experiments done, dataset infrastructure built 2026-03-26, ready for paper writing.
**How to apply:** When discussing experiments or datasets, reference the 4-tier narrative and 48-experiment findings.
