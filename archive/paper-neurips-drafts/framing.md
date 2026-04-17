# Paper Framing: Self-Supervised Learning for Mechanical Grey Swan Prediction

**Date**: 2026-04-09

---

## Title

**Self-Supervised Learning for Mechanical Grey Swan Prediction**

### Rationale
- Clean, direct, memorable
- "Grey swan" is the conceptual hook — rare but physically plausible mechanical failures
- "Self-supervised learning" positions us in the SSL/representation learning community
- Does not oversell (no "foundation model" claim yet)
- Subtitle in abstract can expand: "...via Joint Embedding Predictive Architectures and Temporal Contrastive Learning"

### Rejected Alternatives
- "Forecasting What Matters: ..." — too long, buried lead
- "Mechanical JEPA: ..." — too method-focused, limits framing
- "Beyond Time Series Forecasting: ..." — negative framing, less memorable

---

## Core Framing

### The Grey Swan Paradox
Mechanical component failures are **grey swans** — rare but physically plausible events that follow known degradation physics. Unlike black swans (unforeseeable), grey swans are predictable *in principle* but suffer from a data paradox: the events most critical to predict are the ones with the least training data. A bearing that runs for 10,000 hours before failing provides abundant healthy-state data but perhaps one failure example.

### Forecasting What Matters
Time series foundation models (TimesFM, Chronos-2) forecast the next value. For mechanical prognostics, we don't need to predict every vibration sample — we need to predict the **spectral signature changes** that indicate degradation. Specifically: spectral centroid shift, spectral energy redistribution, and waveform texture changes that precede failure. This is "forecasting what matters" — learning representations aligned with degradation physics rather than reconstructing raw signals.

### The Hybrid Insight
JEPA alone does not beat handcrafted features (p=0.40). But JEPA captures *complementary* features — waveform texture and periodicity that handcrafted spectral statistics miss. The hybrid combination (JEPA+HC) achieves the best result: RMSE 0.055, +75.5% vs time-only, +20.7% vs the best handcrafted-only method. The lesson: self-supervised learning and domain knowledge are synergistic, not competing.

### Cross-Dataset Transfer as the Hard Problem
Within-dataset RUL prediction is well-studied. The unsolved problem is **cross-dataset transfer** — predicting RUL on bearings from a different test rig, operating at different loads and speeds, with different lifetime distributions. Our temporal contrastive approach achieves +38% vs time-only baselines on this task, the first demonstration that self-supervised pretraining enables cross-dataset RUL transfer.

---

## Key Contributions (7 total: 4 delivered, 3 planned)

### Delivered (black text in paper)

1. **Self-supervised framework for mechanical prognostics** that learns degradation-aware representations from unlabeled vibration data, transferable across datasets and operating conditions.

2. **Hybrid JEPA+HC architecture** showing JEPA pretraining captures complementary features to handcrafted spectral indicators. Combined: RMSE 0.055 ± 0.004 (+75.5% vs time-only, +20.7% vs Transformer+HC alone).

3. **Cross-dataset transfer via temporal contrastive learning** — first demonstration for bearing RUL. FEMTO→XJTU-SY: RMSE 0.227 ± 0.015 (+38% vs time-only baseline, p<0.001).

4. **Mechanistic analysis of encoder representations** — JEPA captures waveform texture (PC1 corr w/ spectral centroid: 0.071) while contrastive captures spectral centroid shift (PC1 corr: 0.856), explaining why each excels in different settings and motivating the hybrid approach.

### Planned (blue text in paper, \plannedc{...})

5. ==PLANNED: **Multivariate input with spatiotemporal masking** — physics-aware sensor groupings (radial vibration, axial vibration, thermal, motor current) with structured masking that forces cross-modality prediction.==

6. ==PLANNED: **C-MAPSS turbofan validation** — demonstrating framework generalization beyond bearings to turbofan engines, targeting TCN-Transformer SOTA on FD001-FD004.==

7. ==PLANNED: **Synthetic grey swan augmentation** — physics-informed degradation simulation generating synthetic failure trajectories for rare failure modes, improving prediction of held-out rare events.==

---

## Honest Limitations (acknowledge in paper)

1. **Small dataset**: Only 23 bearing episodes (16 FEMTO + 7 XJTU-SY). Results have non-trivial variance.
2. **JEPA alone < handcrafted**: JEPA+LSTM (0.189) vs HC+LSTM (0.177), not statistically significant (p=0.40). The win is in the *combination*.
3. **Temporal contrastive requires episode structure**: Not purely unsupervised — needs run-to-failure episodes with temporal ordering.
4. **JEPA pretraining instability**: Loss oscillates after epoch 2 on heterogeneous multi-source data. Best checkpoint is always early.
5. **Single-channel only (current)**: Only uses one vibration channel. Multivariate extension is planned.
6. **Two bearing datasets for transfer**: Limited cross-dataset evaluation.

---

## Positioning Against Reviewers

### Likely Reviewer Concern: "Why not just use handcrafted features?"
**Response**: Handcrafted features (spectral centroid, kurtosis, etc.) require domain expertise and don't generalize. Our hybrid result shows JEPA learns *complementary* features — the best result combines both. As we extend to multivariate inputs and new domains (turbofans), handcrafting becomes impractical.

### Likely Reviewer Concern: "Dataset is small"
**Response**: Mechanical run-to-failure datasets are inherently small — each episode requires destroying a bearing. 23 episodes with 5-10 seed evaluation and paired statistical tests is standard for this domain. We report confidence intervals throughout.

### Likely Reviewer Concern: "JEPA alone doesn't beat handcrafted"
**Response**: This is precisely our point. JEPA learns *different* features (waveform texture vs spectral centroid). The mechanistic analysis (Table 5) shows why. The contribution is understanding what each method captures and how to combine them.

### Likely Reviewer Concern: "Why JEPA instead of MAE/contrastive?"
**Response**: We compare directly. JEPA excels within-dataset; contrastive excels cross-dataset. The mechanistic analysis explains why. Both contribute to the full framework.

---

## Figure Plan

1. **Figure 1** (teaser): Grey swan concept — bearing degradation trajectory showing abundant healthy data, sparse failure data, and the prediction task
2. **Figure 2**: Architecture diagram — JEPA pretraining + temporal contrastive + hybrid fusion
3. **Figure 3**: Main results bar chart (already exists as v8_rul_comparison.png)
4. **Figure 4**: Cross-dataset transfer results (already exists as v8_cross_dataset.png)
5. **Figure 5**: Encoder analysis — what JEPA vs contrastive learn (already exists as v8_encoder_analysis.png)
6. **Figure 6**: ==PLANNED: Multivariate architecture with spatiotemporal masking diagram==
7. **Figure 7**: ==PLANNED: C-MAPSS results==

---

## Table Plan

1. **Table 1**: Dataset summary (FEMTO, XJTU-SY, ==PLANNED: C-MAPSS==)
2. **Table 2**: In-domain RUL results (all 11 methods)
3. **Table 3**: Cross-dataset transfer results
4. **Table 4**: Statistical significance tests
5. **Table 5**: Encoder representation analysis
6. **Table 6**: ==PLANNED: Multivariate ablation==
7. **Table 7**: ==PLANNED: Spatiotemporal masking ablation==
8. **Table 8**: ==PLANNED: C-MAPSS comparison with published baselines==
