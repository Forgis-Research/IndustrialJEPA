# Paper Outline: Mechanical-JEPA
*Draft: 2026-04-04*

**Proposed title**: "Self-Supervised Bearing Fault Detection via Joint Embedding Predictive Architecture: Cross-Domain Transfer without Labels"

**Target venue**: ICML 2026 or NeurIPS 2026

---

## Abstract (draft)

We present Mechanical-JEPA, a self-supervised learning framework for industrial bearing fault detection based on Joint Embedding Predictive Architecture (JEPA). By training a Transformer encoder to predict masked patch embeddings in latent space, we learn vibration representations that transfer effectively across machines, sensors, and operating conditions without any fault labels. On the CWRU-to-Paderborn cross-domain benchmark, Mechanical-JEPA achieves a transfer gain of +0.371 ± 0.026 F1 over random initialization — 2.6x better than supervised Transformer pretraining (+0.144) while requiring no labels during pretraining. Remarkably, with only 10 labeled samples per class in the target domain, our method achieves F1=0.735, surpassing a supervised Transformer trained on all available labels (F1=0.689). We also show that spectral auxiliary losses, while improving in-domain accuracy, systematically degrade cross-domain transfer — establishing that latent-space prediction is fundamentally more transferable than physics-informed reconstruction. Our ablation study identifies five critical components that prevent predictor collapse and enable high-quality representation learning at the small-data regime typical of industrial vibration datasets.

---

## 1. Introduction

**Problem**: Industrial machines fail unpredictably. Bearing failures alone account for 40-90% of induction motor failures. Early fault detection requires:
1. Labeled examples of each fault type (expensive to collect)
2. Labeling for each new machine/deployment (doesn't transfer)

**Key insight**: Vibration signals have spatiotemporal structure. A bearing with an inner race fault produces impulse trains at predictable frequencies. This structure can be learned self-supervisedly.

**Our approach**: Apply JEPA — predict masked patch embeddings in latent space rather than reconstructing signals. This forces the encoder to learn semantic representations of fault dynamics, not low-level signal details.

**Contributions**:
1. First application of JEPA to industrial vibration signals
2. 2.6x better cross-domain transfer than supervised pretraining
3. 10-shot performance exceeds supervised Transformer trained on all data
4. Empirical analysis: latent prediction > signal reconstruction for transfer
5. SF-JEPA ablation: spectral auxiliary losses create in-domain/transfer tradeoff

---

## 2. Related Work

### Self-Supervised Learning for Time Series
- TS-TCC (NeurIPS 2021): Contrastive learning + temporal prediction
- TNC: Temporal Neighborhood Coding for physiological signals
- SimMTM (NeurIPS 2023): Masked reconstruction for time series
- **Gap**: None applied JEPA latent prediction to industrial signals

### Bearing Fault Detection
- CNN-based methods achieve near-perfect CWRU accuracy
- Transfer methods: DANN, CORAL, MMD-based domain adaptation
- **Gap**: Most methods require target domain examples or labels

### JEPA Family
- I-JEPA (CVPR 2023): Images, predicts in feature space
- Brain-JEPA (NeurIPS 2024): fMRI, structured masking
- **Gap**: No JEPA application to mechanical/vibration signals

### Nearest Comparisons
- MTS-JEPA: Multivariate time series JEPA (concurrent, limited transfer study)
- LeJEPA/SIGReg: JEPA variant with stop-gradient, 46% worse than EMA on vibration

---

## 3. Method

### 3.1 Problem Setup
- Source: CWRU dataset, K training bearings, 4 fault types, labeled
- Pretraining: Self-supervised on CWRU vibration windows (NO labels used)
- Evaluation: Linear probe on Paderborn bearing dataset (different machine)

### 3.2 Architecture (JEPA V2)
- Input: 3-channel vibration window, 4096 samples at 12kHz
- Patch embedding: 16 non-overlapping patches of 256 samples
- Encoder: 4-layer Transformer, d=512, 4 heads, sinusoidal PE
- Target encoder: EMA copy (momentum=0.996)
- Predictor: 2-layer Transformer, receives context patches + masked positions
- Masking: Random 62.5% of patches masked (10 of 16)

### 3.3 Training Objective
L = L1_JEPA + λ * L_var

Where:
- L1_JEPA = mean L1 between predictor output and target encoder output on masked patches
- L_var = variance regularization penalizing low prediction diversity
- λ = 0.1

### 3.4 Why L1 and not MSE?
MSE creates an incentive for the predictor to predict the "safe" context mean (lower expected MSE than any specific prediction). L1 loss removes this bias — the mean prediction has higher L1 than the true value for any non-symmetric distribution.

### 3.5 Five Critical Components
Each component is necessary; removing any one causes predictor collapse or reduced transfer:
1. **Sinusoidal PE in predictor**: Guarantees position discrimination even at initialization
2. **L1 loss**: Removes mean-prediction shortcut
3. **Variance regularization**: Direct penalty on collapse
4. **Mask ratio 0.625**: Forces content-specific predictions (context average is too poor)
5. **EMA target encoder**: Stable targets prevent representation collapse

---

## 4. Experiments

### 4.1 Datasets
| Dataset | SR | Classes | Windows | Split |
|---------|-----|---------|---------|-------|
| CWRU (source) | 12kHz | 4 | ~2,300 | By bearing ID |
| Paderborn (target) | 64kHz→20kHz | 3 | ~2,280 | By MAT file |

### 4.2 Baselines
- **Random Init**: Untrained JEPA encoder (same architecture)
- **Supervised CNN**: 1D CNN, trained with labels on CWRU
- **Supervised Transformer**: Same encoder, trained with labels on CWRU
- **MAE**: Signal-space reconstruction (masked autoencoder)
- **LeJEPA/SIGReg**: JEPA variant with stop-gradient instead of EMA

### 4.3 Transfer Gain Metric
Transfer Gain = Paderborn F1 (method) - Paderborn F1 (random init, same architecture)

All transfer gains use the same random init baseline (same JEPA architecture, seeds 42/123/456).

### 4.4 Results: Table 1 (Main Transfer Results)
[See CONSOLIDATED_RESULTS.md Section 3]

### 4.5 Results: Figure 1 (Few-Shot Transfer Curves)
Key finding: JEPA@N=10 (0.735) > Transformer@N=all (0.689)
p=0.034, Cohen's d=0.92 (large effect size)

### 4.6 Ablation: Table 2 (5 V2 Components)
[See CONSOLIDATED_RESULTS.md Section 5]

### 4.7 Analysis: SF-JEPA Tradeoff
[See CONSOLIDATED_RESULTS.md Section 8 + fig4_sfjepa_tradeoff]

---

## 5. Analysis

### 5.1 Why JEPA Transfers Better Than Supervised

Empirical finding: Supervised Transformer achieves 0.969 CWRU F1 but only +0.144 transfer gain.
JEPA achieves 0.773 CWRU F1 but +0.371 transfer gain.

Hypothesis: Supervised CWRU training memorizes:
- Motor load frequency components specific to CWRU's test bench
- Bearing defect frequencies at 12kHz (doesn't align with 20kHz Paderborn)
- Sensor placement artifacts from Case Western's lab setup

JEPA's objective — predict masked patches from context — forces the encoder to learn temporal dynamics that are invariant to these dataset-specific confounders.

### 5.2 Physics-Informed Losses and Transfer

SF-JEPA adds spectral features (FFT band energies at 12kHz, RMS, centroid) as auxiliary targets.
Finding: monotone tradeoff — more physics = better CWRU, worse Paderborn.

Interpretation: Spectral band energies are defined relative to CWRU's 12kHz sampling rate. The 4 frequency bands that distinguish CWRU fault types don't correspond to the same physical phenomena at Paderborn's resampled 20kHz. Physics-informed features are inherently dataset-specific.

### 5.3 Component Cross-Type Transfer

Bearing (CWRU) → Gearbox (MCC5-THU): +2.5% gain only.
Gear-pretrained JEPA → Bearing: -3.6% (negative transfer).

Root cause: Bearing faults = periodic impulses at defect frequencies. Gear faults = amplitude/frequency modulation at tooth-mesh frequency. These are fundamentally different signal generating mechanisms. JEPA learns signal dynamics, not domain-agnostic abstractions.

Implication: "Vibration" is not a single modality. Bearing models should be pretrained on bearing data.

---

## 6. Limitations

1. **n=3 seeds**: All claims have 3-seed backing. For camera-ready, increase to 10 seeds.
2. **Linear probe evaluation**: MLP probe likely gives higher absolute F1 but doesn't isolate representation quality.
3. **RUL not solved**: JEPA was designed for fault classification, not degradation monitoring. IMS RUL fails (constant baseline wins due to label imbalance).
4. **CWRU is too easy**: Handcrafted features achieve 0.999 F1. CWRU in-domain results are not meaningful benchmarks for deep learning.
5. **No formal domain adaptation comparison**: DANN, CORAL, etc. were not evaluated.

---

## 7. Conclusion

Mechanical-JEPA demonstrates that self-supervised JEPA pretraining is a strong foundation for cross-machine bearing fault transfer. The key results:
1. +0.371 transfer gain (2.6x better than supervised Transformer)
2. N=10 shot performance exceeds supervised Transformer at full data
3. Latent prediction >> signal reconstruction for transfer (JEPA vs MAE: 0.371 vs 0.001 gain)
4. Five specific architectural components prevent predictor collapse

The broader principle: for industrial vibration, self-supervised pretraining on a well-characterized fault dataset (CWRU) produces more transferable representations than supervised pretraining — because the self-supervised objective avoids memorizing dataset-specific artifacts.

---

## Appendix

### A. Statistical Significance
[All key claims: p < 0.05, large Cohen's d — see statistical_tests.py]

### B. Data Leakage Analysis
[CWRU: split by bearing ID, verified no overlap. Paderborn: split by file, verified 80/20]

### C. Frequency Resampling
[Polyphase resample 64kHz → 20kHz before windowing, matches 12kHz features after patch normalization]

### D. Hyperparameter Sensitivity
[Mask ratio 0.5 vs 0.625 vs 0.75: moderate sensitivity. lr 1e-4 vs 5e-4: moderate. epochs 100 is optimal]
