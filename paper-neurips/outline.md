# Paper Outline: Self-Supervised Learning for Mechanical Grey Swan Prediction

**Page budget**: 9 pages (NeurIPS main paper) + unlimited appendix

---

## Abstract (0.25 pages)

- Grey swan paradox: rare mechanical failures have least training data
- Self-supervised pretraining on unlabeled operational data
- JEPA + temporal contrastive + hybrid with handcrafted features
- Key results: Hybrid RMSE 0.055 (+75.5%), cross-dataset +38%
- ==PLANNED: Multivariate extension, C-MAPSS generalization, synthetic augmentation==

---

## 1. Introduction (1.5 pages)

### 1.1 The Grey Swan Problem (0.4 pages)
- Mechanical failures are rare but catastrophic
- Grey swan = rare but physically plausible (not black swan = unforeseeable)
- Data paradox: most critical events have least data
- **Figure 1**: Grey swan concept figure — degradation trajectory

### 1.2 Forecasting What Matters (0.4 pages)
- Time series FMs forecast next value; we predict degradation indicators
- Spectral centroid shift, energy redistribution → actual failure precursors
- Self-supervised pretraining from abundant unlabeled operational data

### 1.3 Our Approach (0.4 pages)
- Three synergistic components: JEPA + contrastive + hybrid fusion
- ==PLANNED: Multivariate + spatiotemporal masking + synthetic augmentation==
- **Key insight**: SSL and domain knowledge are synergistic, not competing

### 1.4 Contributions (0.3 pages)
- 7 numbered contributions (4 delivered + 3 planned)

---

## 2. Related Work (1.0 pages)

### 2.1 Self-Supervised Learning for Time Series (0.3 pages)
- Contrastive: TS2Vec, TNC, CoST, TS-TCC
- Masked reconstruction: SimMTM, PatchTST, TimeMAE
- Foundation models: TimesFM, Chronos-2, MOMENT
- **Gap**: None target mechanical degradation specifically

### 2.2 Remaining Useful Life Prediction (0.3 pages)
- Classical: Wiener process, particle filters
- Deep learning: CNN-LSTM, TCN-Transformer, attention-based
- Bearing benchmarks: FEMTO, XJTU-SY, IMS
- Turbofan benchmark: C-MAPSS (FD001-FD004)
- **Gap**: Almost all fully supervised + handcrafted features

### 2.3 Joint Embedding Predictive Architectures (0.25 pages)
- I-JEPA (images), V-JEPA (video), Brain-JEPA (fMRI)
- TS-JEPA, MTS-JEPA (time series)
- **Gap**: No JEPA for mechanical systems / prognostics

### 2.4 Rare Event Prediction in Physical Systems (0.15 pages)
- Few-shot fault detection, transfer learning for PHM
- Synthetic data augmentation
- **Gap**: No self-supervised approach for grey swan mechanical events

---

## 3. Method (2.0 pages)

### 3.1 Problem Formulation (0.3 pages)
- RUL% definition: RUL(t) = 1 - t/T_failure
- Input: vibration windows (1024 samples, single channel)
- ==PLANNED: Multivariate extension (8 channels)==
- Grey swan framing: predict rare events from abundant healthy data

### 3.2 JEPA Pretraining (0.5 pages)
- Architecture: ViT encoder (d=256, 4 layers, 4 heads), EMA target encoder
- Patch embedding: 16 patches of 64 samples
- Masking: 62.5% random patch masking
- Loss: L1 on L2-normalized predictions + variance regularization
- **Figure 2a**: JEPA architecture diagram
- ==PLANNED: Spatiotemporal masking for multivariate input==

### 3.3 Temporal Contrastive Learning (0.4 pages)
- Triplet objective: adjacent snapshots positive, distant negative
- Leverages run-to-failure episode temporal structure
- Forces encoder to learn what *changes* over bearing life
- Key: learns spectral centroid shift (the dominant degradation indicator)

### 3.4 Hybrid Architecture (0.3 pages)
- JEPA encoder output + handcrafted spectral features
- Concatenation → Transformer decoder → RUL prediction
- **Figure 2b**: Hybrid fusion diagram
- Why it works: JEPA captures waveform texture, HC captures spectral centroid

### 3.5 ==PLANNED: Multivariate Extension== (0.3 pages)
- ==PLANNED: Multi-channel input with physics-aware sensor groupings==
- ==PLANNED: Spatiotemporal masking (mask by sensor group)==
- ==PLANNED: Frequency-domain dual encoder==

### 3.6 ==PLANNED: Synthetic Grey Swan Augmentation== (0.2 pages)
- ==PLANNED: Physics-informed degradation simulation==
- ==PLANNED: Generating synthetic failure trajectories for rare modes==

---

## 4. Experiments (2.5 pages)

### 4.1 Experimental Setup (0.4 pages)
- Datasets: FEMTO (16 episodes), XJTU-SY (7 episodes), ==PLANNED: C-MAPSS==
- **Table 1**: Dataset summary
- Evaluation: RMSE, paired t-tests, 5-10 seeds
- Baselines: 11 methods from constant mean to Transformer+HC

### 4.2 In-Domain RUL Prediction (0.5 pages)
- **Table 2**: Full 11-method comparison
- Key findings: JEPA+LSTM +15.8% (p=0.010), Hybrid +75.5%
- JEPA matches handcrafted (p=0.40) without domain knowledge
- **Figure 3**: Bar chart comparison

### 4.3 Cross-Dataset Transfer (0.5 pages)
- **Table 3**: FEMTO↔XJTU transfer matrix
- Contrastive outperforms JEPA for transfer (+19%, p<0.001)
- JEPA better within-dataset (especially FEMTO)
- **Figure 4**: Transfer results visualization

### 4.4 Encoder Representation Analysis (0.4 pages)
- **Table 5**: PC1 correlations with RUL and spectral centroid
- JEPA: waveform texture (low spectral centroid correlation: 0.071)
- Contrastive: spectral centroid shift (correlation: 0.856)
- **Figure 5**: Encoder analysis visualization
- Mechanistic explanation for complementary behavior

### 4.5 ==PLANNED: Multivariate Ablation== (0.3 pages)
- ==PLANNED: **Table 6**: 1ch vs 2ch vs 4ch vs 8ch==
- ==PLANNED: **Table 7**: Random vs physics-aware vs spatiotemporal masking==

### 4.6 ==PLANNED: C-MAPSS Turbofan Benchmark== (0.3 pages)
- ==PLANNED: **Table 8**: FD001-FD004 comparison with published SOTA==
- ==PLANNED: Demonstrates generalization beyond bearings==

### 4.7 ==PLANNED: Synthetic Grey Swan Augmentation== (0.1 pages)
- ==PLANNED: Real-only vs augmented training for rare failure modes==

---

## 5. Analysis and Discussion (0.75 pages)

### 5.1 What Do Self-Supervised Encoders Learn? (0.3 pages)
- JEPA → waveform texture and periodicity
- Contrastive → degradation dynamics (spectral centroid shift)
- Hybrid → complementary features from both
- ==PLANNED: Multivariate representation analysis==

### 5.2 When Does Transfer Succeed? (0.25 pages)
- Within-dataset: JEPA preferred (preserves fine-grained texture)
- Cross-dataset: contrastive preferred (captures invariant degradation dynamics)
- Episode lifetime variance as predictor of transfer difficulty

### 5.3 Limitations (0.2 pages)
- Small dataset (23 episodes)
- Single channel (current)
- Temporal contrastive requires episode structure
- JEPA pretraining instability

---

## 6. Conclusion (0.5 pages)
- Summary of contributions
- The grey swan paradox and how SSL addresses it
- ==PLANNED: Vision for mechanical world model==
- Future work

---

## Appendix (unlimited)

### A. Dataset Details
- Bearing specifications, operating conditions, sampling rates
- Episode lifetime distributions
- Preprocessing details

### B. Hyperparameters
- **Table A1**: Complete hyperparameter table for all methods
- JEPA pretraining configurations explored

### C. Additional Results
- Piecewise label ablation
- Full per-episode results
- Additional statistical tests

### D. ==PLANNED: C-MAPSS Experimental Details==

---

## Page Budget Summary

| Section | Pages | Status |
|---------|-------|--------|
| Abstract | 0.25 | Draft |
| Introduction | 1.50 | Polished |
| Related Work | 1.00 | Polished |
| Method | 2.00 | Skeleton + planned |
| Experiments | 2.50 | Partial + planned |
| Analysis | 0.75 | Skeleton |
| Conclusion | 0.50 | Skeleton |
| References | 0.50 | — |
| **Total** | **9.00** | — |
