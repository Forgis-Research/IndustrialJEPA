# DCSSL Replication Specification

**Paper:** "A novel dual-dimensional contrastive self-supervised learning-based framework for rolling bearing remaining useful life prediction"
**Authors:** Zhunan Shen, Chenhao Yang, Liu Cheng, Xiangwei Kong, Zhitong Liu, Kaiyu Su
**Venue:** Scientific Reports (Nature), 2026. DOI: 10.1038/s41598-026-38417-7
**Affiliation:** Northeastern University, Shenyang, China

---

## Goal

Replicate Table 3 from the paper: MSE results on FEMTO/PRONOSTIA bearings.
Then add our JEPA+HC hybrid as an additional column for direct comparison.

---

## Table 3 Target Results (MSE)

Training bearings → Testing bearings:

| Train | Test | InfoTS | USL | CBHRL | SimCLR | SupCon | DCSSL |
|-------|------|--------|-----|-------|--------|--------|-------|
| 1_1 & 1_2 | 1_3 | 0.0037 | 0.0047 | 0.0052 | 0.0029 | 0.0028 | 0.0011 |
| 1_1 & 1_2 | 1_4 | 0.0566 | 0.1003 | 0.0012 | 0.2565 | 0.0080 | 0.0476 |
| 1_1 & 1_2 | 1_5 | 0.0015 | 0.0014 | 0.0005 | 0.0030 | 0.0097 | 0.0005 |
| 1_1 & 1_2 | 1_6 | 0.1095 | 0.0449 | 0.0016 | 0.0560 | 0.0473 | 0.0892 |
| 1_1 & 1_2 | 1_7 | 0.0031 | 0.0044 | 0.2782 | 0.0006 | 0.0040 | 0.0009 |
| 2_1 & 2_2 | 2_3 | 0.0805 | 0.0406 | 0.0018 | 0.0904 | 0.0569 | 0.0027 |
| 2_1 & 2_2 | 2_4 | — | — | 0.0229 | 0.0021 | 0.0046 | 0.0014 |
| 2_1 & 2_2 | 2_5 | — | — | 0.0091 | 0.1849 | 0.0735 | 0.2538 |
| 2_1 & 2_2 | 2_6 | — | — | 0.0425 | 0.0024 | 0.0038 | 0.0012 |
| 2_1 & 2_2 | 2_7 | — | — | — | 0.2577 | 0.0150 | 0.0075 |
| 3_1 & 3_2 | 3_3 | — | — | 0.0619 | 0.0013 | 0.0017 | 0.0068 |
| | **Avg** | — | — | — | 0.0583 | 0.0480 | **0.0375** |

---

## Method: DCSSL Architecture (Reconstructed from Abstract + References)

### Overview
Two-stage framework:
1. **Stage 1 (Self-Supervised Pretraining):** Learn state representations from unlabeled vibration data using dual-dimensional contrastive learning
2. **Stage 2 (Fine-tuning):** Fine-tune with a prediction head for RUL regression on labeled data

### Stage 1: Self-Supervised Pretraining

**Positive pair construction:**
- Random cropping: extract random sub-sequences from the vibration signal
- Timestamp masking: mask portions of the time dimension

**Dual-dimensional contrastive loss:**
1. **Temporal-level contrastive loss:** Within a single bearing's degradation sequence, nearby timestamps should have similar representations, distant timestamps should differ. This captures degradation trend.
2. **Instance-level contrastive loss:** Across different bearing instances at similar degradation stages, representations should be similar. This generalizes across bearings.

**Encoder:** Based on the references (Franceschi et al. 2019 "Unsupervised scalable representation learning for multivariate time series" + Bai et al. 2018 "An empirical evaluation of generic convolutional and recurrent networks"), the encoder is almost certainly a **dilated causal CNN / Temporal Convolutional Network (TCN)**. This is the standard encoder for time series contrastive learning.

Key references for architecture:
- SimCLR (Chen et al. 2020) — contrastive framework
- MoCo (He et al. 2020) — momentum contrast
- CPC (Oord et al. 2018) — contrastive predictive coding
- InfoTS (Luo et al. AAAI 2023) — information-aware augmentations for time series
- USL (Kong et al. RESS 2023) — unsupervised contrastive for RUL
- CBHRL (Zhu et al. RESS 2024) — contrastive BiLSTM health representation learning
- SupCon (Khosla et al. NeurIPS 2020) — supervised contrastive learning

### Stage 2: Fine-tuning / Prediction Head
- Newly constructed prediction head (likely MLP or small LSTM)
- Fine-tuned end-to-end on labeled RUL data
- RUL label: likely piecewise linear (constant at 1.0 during healthy, linear decay to 0.0 at failure) — standard in FEMTO literature

### Likely Hyperparameters (based on related work conventions)
- Temperature τ: 0.05–0.1 (standard for NT-Xent loss)
- Encoder: TCN with 10 residual blocks, kernel size 3, dilation doubling
- Hidden dim: 64–256
- Projection head: 2-layer MLP (hidden → 128 → 64)
- Optimizer: Adam, lr=1e-3 to 1e-4
- Batch size: 32–128
- Pretraining epochs: 100–500
- Fine-tuning epochs: 50–200

---

## Dataset: FEMTO/PRONOSTIA (PHM 2012 Challenge)

**Download:** https://phm-datasets.s3.amazonaws.com/NASA/10.+FEMTO+Bearing.zip

**Structure:**
- 3 operating conditions
  - Condition 1 (1800 RPM, 4000 N): Bearings 1_1 through 1_7
  - Condition 2 (1650 RPM, 4200 N): Bearings 2_1 through 2_7  
  - Condition 3 (1500 RPM, 5000 N): Bearings 3_1 through 3_3
- Training bearings: 1_1, 1_2, 2_1, 2_2, 3_1, 3_2
- Test bearings: 1_3–1_7, 2_3–2_7, 3_3

**Signal:**
- 2 channels: horizontal and vertical accelerometer
- Sampling rate: 25.6 kHz
- Recording: 2560 samples (0.1 sec) per snapshot
- Snapshot interval: every 10 seconds

**Standard preprocessing:**
1. Extract features from each 2560-sample snapshot (or use raw)
2. Construct degradation sequence: ordered list of snapshots over bearing lifetime
3. RUL label: piecewise linear — constant 1.0 during healthy phase, linear decay from FPT to end-of-life

---

## Evaluation Protocol

- **Metric:** MSE between predicted RUL curve and ground truth RUL curve
- **Per-condition training:** Train on condition X training bearings, test on condition X test bearings
  - Condition 1: Train on 1_1 & 1_2, test on 1_3, 1_4, 1_5, 1_6, 1_7
  - Condition 2: Train on 2_1 & 2_2, test on 2_3, 2_4, 2_5, 2_6, 2_7
  - Condition 3: Train on 3_1 & 3_2, test on 3_3
- **RUL is normalized to [0, 1]:** 1.0 = full life remaining, 0.0 = end of life
- **MSE is computed on the full predicted RUL trajectory vs ground truth**

---

## Baselines to Also Implement

1. **InfoTS** (Luo et al. AAAI 2023) — Information-aware augmentations for time series contrastive learning
2. **USL** (Kong et al. RESS 2023) — Unsupervised contrastive framework for RUL
3. **SimCLR** (Chen et al. ICML 2020) — Simple contrastive learning adapted for time series
4. **SupCon** (Khosla et al. NeurIPS 2020) — Supervised contrastive learning
5. **CBHRL** (Zhu et al. RESS 2024) — Contrastive BiLSTM health representation learning
6. **Our JEPA+HC Hybrid** — from mechanical-jepa/v8/

---

## Implementation Priority

1. Download and preprocess FEMTO data
2. Implement DCSSL (core contribution)
3. Implement SimCLR baseline (simplest, validates pipeline)
4. Run all experiments, compare to Table 3
5. Add JEPA+HC hybrid comparison
