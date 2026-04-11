# CNN-GRU-MHA Replication Specification

**Paper:** "Remaining Useful Life of the Rolling Bearings Prediction Method Based on Transfer Learning Integrated with CNN-GRU-MHA"
**Authors:** Jianghong Yu, Jingwei Shao, Xionglu Peng, Tao Liu, Qishui Yao
**Venue:** Applied Sciences, 2024, 14, 9039. DOI: 10.3390/app14199039
**Affiliation:** Hunan University of Technology, Zhuzhou, China

---

## Goal

Replicate Table 4 (FEMTO transfer RMSE) and Table 5 (XJTU-SY transfer RMSE).
Then add our JEPA+HC and DCSSL as additional comparisons on the same protocol.

---

## Table 4 Target Results (FEMTO — per-transfer RMSE loss)

| Source | Target | Loss |
|--------|--------|:----:|
| Bearing 1-3 | Bearing 2-3 | 0.0463 |
| Bearing 1-3 | Bearing 2-4 | 0.0449 |
| Bearing 1-3 | Bearing 3-1 | 0.0427 |
| Bearing 1-3 | Bearing 3-3 | 0.0461 |
| Bearing 2-3 | Bearing 1-3 | 0.0458 |
| Bearing 2-3 | Bearing 1-4 | 0.0426 |
| Bearing 2-3 | Bearing 3-3 | 0.0416 |
| Bearing 2-3 | Bearing 3-3 | 0.0416 |
| Bearing 3-2 | Bearing 1-3 | 0.0382 |
| Bearing 3-2 | Bearing 1-4 | 0.0397 |
| Bearing 3-2 | Bearing 2-3 | 0.0413 |
| Bearing 3-2 | Bearing 2-4 | 0.0418 |
| | **Average** | **0.0433** |

Paper headline RMSE: **0.0443** (Section 5, Conclusions)

## Table 5 Target Results (XJTU-SY — transfer RMSE)

| Source | Target | Loss |
|--------|--------|:----:|
| Bearing 1-3 | Bearing 2-3 | 0.0568 |
| Bearing 1-3 | Bearing 3-2 | 0.0464 |
| Bearing 2-3 | Bearing 1-3 | 0.1138 |
| Bearing 2-3 | Bearing 3-2 | 0.0595 |
| | **Average** | **0.0691** |

---

## Architecture: CNN-GRU-MHA

### Preprocessing Pipeline
1. **DWT noise reduction**: Discrete wavelet transform, wavelet='sym8', 3 decomposition levels. Reconstruct from approximation coefficients (suppress high-freq noise).
2. **Min-max normalization**: X_i = (x_i - x_min) / (x_max - x_min) → [0, 1]
3. **HI construction**: The denoised + normalized vibration signal IS the health indicator (no handcrafted features — the raw processed signal is used directly)

### RUL Labels
Linear degradation: Y_i = (N - i) / N where N = total snapshots, i = current index.
NOT piecewise-linear — simple linear from 1.0 to 0.0 over the entire bearing life.

### CNN Feature Extractor
6 convolutional layers in sequence:

| Block | Layers | Filters | Kernel | Pooling |
|-------|--------|:-------:|:------:|:-------:|
| 1 | Conv + BN + ReLU | 32 | 5×5 | 2×2 MaxPool |
| 2 | Conv + BN + ReLU | 64 | 5×5 | 2×2 MaxPool |
| 3 | Conv + BN + ReLU | 128 | 5×5 | 2×2 MaxPool |
| — | **MHA (2 heads)** | — | — | — |
| 4 | Conv + BN + ReLU | 256 | 3×3 | 2×2 MaxPool |
| 5 | Conv + BN + ReLU | 512 | 3×3 | 2×2 MaxPool |
| 6 | Conv + BN + ReLU | 1024 | 3×3 | 2×2 MaxPool |
| — | **Global Average Pooling** | — | — | — |

Note: Input is 1D vibration signal. The paper describes "5×5 convolutions" but the input is 1D, so these are likely kernel_size=5 and kernel_size=3 for the 1D case. The 2×2 pooling is likely MaxPool1d(2).

### Multi-Head Attention (MHA)
- 2 attention heads
- Inserted after the 3rd CNN layer (before layers 4-6)
- Standard Q/K/V attention with softmax + scaling

### GRU Temporal Model
- 2 GRU layers
- Hidden sizes: [512, 128]
- Processes the sequence of CNN feature vectors over the bearing's lifetime

### Fully Connected Head
- FC1: 64 nodes + ReLU
- FC2: 1 node (RUL output)

### Loss Function
RMSE + L1 regularization:
$$Loss = \sqrt{\frac{1}{m}\sum_{i=1}^{m}(y_i - Y_i)^2} + \alpha \sum_{j=1}^{n}|w_j|$$

### Training Hyperparameters (Table 3)
| Parameter | Value |
|-----------|:-----:|
| Learning rate | 0.001 |
| Sample size | 128 |
| Source domain iterations | 60 |
| Target domain iterations | 100 |
| Batch size | 128 |
| Optimizer | Adam |
| Framework | TensorFlow 2.5.0 (we use PyTorch) |

### Transfer Learning Protocol
1. **Source domain training** (60 iterations): Train full CNN-GRU-MHA on source bearing
2. **Freeze feature extractor**: Freeze CNN + GRU layers (keep parameters)
3. **Target domain fine-tuning** (100 iterations): Train only the FC head on target bearing's test data (with labels)
4. **Evaluate on validation set** (target domain, no labels during inference)

**Critical detail**: The paper splits each target domain bearing 1:1 into test (with labels) and validation (no labels). The FC head is fine-tuned on the test half, and RMSE is evaluated on the validation half.

---

## Dataset Details

### FEMTO/PHM2012
- **Download**: https://phm-datasets.s3.amazonaws.com/NASA/10.+FEMTO+Bearing.zip
- **Channels**: Horizontal + vertical accelerometer (paper uses HORIZONTAL ONLY)
- **Sampling rate**: 25.6 kHz
- **Snapshot**: 2560 samples (0.1s) every 10 seconds
- **3 operating conditions**: see Table 1 in paper

| Condition | RPM | Load (N) | Bearings |
|-----------|:---:|:--------:|----------|
| 1 | 1800 | 4000 | 1_1 through 1_7 |
| 2 | 1650 | 4200 | 2_1 through 2_7 |
| 3 | 1500 | 5000 | 3_1 through 3_3 |

### Transfer Experiment Setup (Table 2)
| Test | Source Bearing | Target Bearings |
|------|:--------------:|-----------------|
| 1 | Bearing 1-3 | 2-3, 2-4, 3-1, 3-3 |
| 2 | Bearing 2-3 | 1-3, 1-4, 3-3, 3-3 |
| 3 | Bearing 3-2 | 1-3, 1-4, 2-3, 2-4 |

### XJTU-SY
- **Reference**: Lei et al. 2019, "XJTU-SY Rolling Element Bearing Accelerated Life Test Datasets: A Tutorial"
- **3 operating conditions**, 15 bearings total (5 per condition)

| Condition | RPM | Load (kN) | Bearings |
|-----------|:---:|:---------:|----------|
| 1 | 2100 | 12 | 1-1 through 1-5 |
| 2 | 2250 | 11 | 2-1 through 2-5 |
| 3 | 2400 | 10 | 3-1 through 3-5 |

### XJTU-SY Transfer Setup (Table 5)
| Experiment | Source | Targets |
|------------|:------:|---------|
| 1 | Bearing 1-3 | 2-3, 3-2 |
| 2 | Bearing 2-3 | 1-3, 3-2 |

---

## Evaluation Protocol

- **Metric**: RMSE on full predicted RUL trajectory vs ground truth (per-bearing)
- **RUL normalized to [0, 1]**: Linear decay Y_i = (N-i)/N
- **Transfer protocol**: Train on source, freeze CNN+GRU, fine-tune FC on target
- **Report**: Per-transfer RMSE + average across all transfers
- **Repeat 5 times** to average out randomness (paper says "repeated five times to take average")

---

## Implementation Priority

1. Download and preprocess FEMTO data (reuse from dcssl-replication/)
2. Implement CNN-GRU-MHA architecture in PyTorch
3. Implement DWT preprocessing + min-max normalization
4. Run FEMTO transfer experiments (Table 4, 12 transfers)
5. Run XJTU-SY transfer experiments (Table 5, 4 transfers)
6. Compare results to paper targets
7. Add JEPA+HC and DCSSL baselines on the same protocol
