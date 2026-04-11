# Next Steps (Updated 2026-03-25 06:00)

## Research Status: COMPLETE for Paper Submission

48 experiments across 3 tiers with comprehensive ablations. The story is clear and nuanced.

---

## Core Finding

**Physics-masked attention provides a principled attention constraint for multivariate time series.**

| System Type | Physics Mask Value | Why |
|-------------|-------------------|-----|
| Independent components (pendulum) | **Strong** (7.4% > Full-Attn, p=0.0002) | Groups match true statistical independence |
| Correlated components (C-MAPSS) | **Moderate** (6.4% > Full-Attn, ns) | Components share failure mode |
| Complex interactions (weather) | **Negative** (-1.3% vs Full-Attn) | Cross-group interactions too important |
| **All systems vs CI** | **Strong** (5-34%, p<0.005) | 2D treatment always helps |

---

## For Paper Writing

### Ready Now
1. All experimental results with 10-seed statistical power
2. Comprehensive ablations (mask type, grouping assignment, data fraction, horizon)
3. Honest negative results (JEPA, slot attention, data efficiency)
4. Clear paper narrative: "When to Mask"

### Still Needed
1. **Figures**: Attention heatmaps comparing physics vs full vs random masks
2. **Writing**: LaTeX paper draft
3. **Related work**: Position vs iTransformer, PatchTST, MTGNN
4. **Additional benchmark** (optional): A second mechanical system (e.g., bearing fault)

---

## Suggested Paper Structure

**Title**: "When to Mask: Physics-Informed Attention for Multivariate Time Series"

1. **Introduction**: The attention spectrum (CI → masked → full)
2. **Method**: PhysMask architecture (2D treatment + physics-informed mask)
3. **Experiments**: 3 tiers × 3+ seeds × 6+ models
4. **When masking helps**: Independence structure predicts effectiveness
5. **Ablations**: Mask type, grouping assignment, horizon, data size
6. **Negative results**: JEPA, pooling, slot attention
7. **Discussion**: Practical guidelines

---

## Experiment Inventory (48 total)

| Phase | Experiments | Key Results |
|-------|------------|-------------|
| C-MAPSS Baselines | Exp 7-20 | Role-Trans 12.22 FD001, beats CI by 27% |
| Transfer & Ablation | Exp 21-38 | Weight sharing is key, JEPA hurts |
| Slot Attention | Exp 39-40 | K=5 optimal but slots collapse |
| Tier 1-3 Validation | Exp 41 | 3/3 tiers physics > CI |
| Multi-horizon | Exp 42 | 5.4% → 9.6% with horizon |
| 10-Seed Stats | Exp 43 | p=0.002 C-MAPSS, p<0.0001 pendulum |
| Data Efficiency | Exp 44 | Full-Attn wins at all sizes |
| Pendulum Grouping | Exp 45 | Wrong grouping 23x worse |
| PhysMask vs Full-Attn | Exp 46 | Pendulum: PhysMask wins. Weather: Full wins |
| Pendulum Mask Ablation | Exp 47 | Physics mask >> random mask (p<0.001) |
| C-MAPSS Mask Ablation | Exp 48 | Physics ≈ random (p=0.528), lowest variance |
