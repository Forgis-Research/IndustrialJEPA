# STAR Replication Specification

**Paper**: "A Two-Stage Attention-Based Hierarchical Transformer for Turbofan Engine Remaining Useful Life Prediction"
**Authors**: Zhengyang Fan, Wanru Li, Kuo-Chu Chang
**Affiliation**: Department of Systems Engineering and Operations Research, George Mason University
**Venue**: *Sensors* (MDPI), 2024, 24, 824. DOI: [10.3390/s24030824](https://doi.org/10.3390/s24030824)
**Open access**: [PMC10857698](https://pmc.ncbi.nlm.nih.gov/articles/PMC10857698/)

---

## Goal

Replicate Table 5 from the paper: RMSE and PHM 2008 Score on all four C-MAPSS subsets (FD001, FD002, FD003, FD004). STAR is the current verified supervised SOTA on FD001, FD002, FD003.

---

## Target Results (Table 5 of paper)

| Subset | RMSE | Score |
|:------:|:----:|:-----:|
| FD001  | **10.61** | **169** |
| FD002  | **13.47** | **784** |
| FD003  | **10.71** | 202 |
| FD004  | 15.87 | **1449** |

Second-place on FD003 Score (154, DAST) and FD004 RMSE (15.86, DLformer). Cross-reference on reproducibility: STAR explicitly compares to 12 baselines with both RMSE and Score; numbers are consistent with STAR's claims of SOTA.

---

## Architecture: STAR

### Input
- Multivariate time series $x_{1:T} \in \mathbb{R}^{T \times D}$
- $T$ = input time window length (see hyperparameters)
- $D = 14$ sensors after selection

### Overall Structure (Figure 1 of paper)
```
Input (T, D)
  │
  ├─ Dimension-wise segmentation: each sensor split into K patches of length L
  │  Each patch → affine transform + positional embedding → d_model dims
  │  Result: X^(e) ∈ R^(K × D × d_model)
  │
  │ ┌─────────── Scale 1 ───────────┐
  │ │                               │
  │ ├─ Two-stage encoder            │
  │ │  (temporal attn → sensor attn)│
  │ │                               │
  │ ├─ Two-stage decoder            │
  │ │                               │
  │ ├─ Linear → part of final concat│
  │ │                               │
  │ └─ Patch merging: K → K/2       │
  │                                 │
  │ ┌─────────── Scale 2 ───────────┐
  │ │ ... (same structure)           │
  │ └────────────────────────────────┘
  │
  │ ... continue for S scales ...
  │
  Concatenate all scale outputs → Linear → RUL prediction
```

### Component 1: Dimension-wise Segmentation and Embedding (§3.1)

For each sensor $d \in \{1, ..., D\}$:
- Segment the univariate time series into $K$ disjoint patches of length $L$
- Each patch $x_{k,d} \in \mathbb{R}^L$
- Embedding: $x^{(e)}_{k,d} = A \cdot x_{k,d} + E_{k,d}$
  - $A \in \mathbb{R}^{d_{model} \times L}$ is a learnable affine transformation
  - $E_{k,d} \in \mathbb{R}^{d_{model}}$ is a learnable positional embedding

Output: $X^{(e)} \in \mathbb{R}^{K \times D \times d_{model}}$

**Patch dimensions**:
- For FD001 with time_length=32, the paper doesn't specify L explicitly. Given layers/scales=3, L and K must be chosen such that after 3 halvings of K we still have at least K=1. If T=32, L=4 gives K=8, then 8→4→2 (viable). If L=8 gives K=4, then 4→2→1 (viable). Try L=4 or L=8.
- For FD002 with time_length=64, layers/scales=4: L=4 gives K=16, then 16→8→4→2 (viable).
- For FD003 with time_length=48, layers/scales=1: L can be larger since no merging.
- For FD004 with time_length=64, layers/scales=4: L=4 gives K=16 (same as FD002).

### Component 2: Two-Stage Attention-Based Encoder (§3.2)

This is the key architectural novelty. Each encoder block applies two attention stages sequentially:

**Stage A: Temporal attention within each sensor**

For each sensor $d$:
- Input: $X^{(e)}_{:,d,:} \in \mathbb{R}^{K \times d_{model}}$ (K patches of one sensor)
- Standard multi-head self-attention over the K patch dimension
- LayerNorm, residual connection, feed-forward, LayerNorm, residual (standard Transformer block)
- Output: $X^{temp}_{:,d,:} \in \mathbb{R}^{K \times d_{model}}$

Concatenate across all sensors: $X^{temp} \in \mathbb{R}^{K \times D \times d_{model}}$

**Stage B: Sensor-wise attention at each temporal position**

For each temporal patch index $k$:
- Input: $X^{temp}_{k,:,:} \in \mathbb{R}^{D \times d_{model}}$ (D sensors at one temporal position)
- Standard multi-head self-attention over the D sensor dimension
- Same Transformer block structure
- Output: $X^{enc,s}_{k,:,:} \in \mathbb{R}^{D \times d_{model}}$

Final encoder output at scale $s$: $X^{enc,s} \in \mathbb{R}^{K_s \times D \times d_{model}}$

### Component 3: Patch Merging (§3.3)

Between scales, reduce $K_s \to K_{s+1} = K_s / 2$:

$$X^{enc, s+1}_i = B \cdot [X^{enc, s}_{2i,d}, X^{enc, s}_{2i+1,d}]$$

where $B \in \mathbb{R}^{d_{model} \times 2 d_{model}}$ is a learnable matrix. Applied per sensor.

### Component 4: Two-Stage Attention-Based Decoder (§3.4)

At each scale $s$:
- Input from decoder at scale $s-1$: $X^{dec, s-1}$
- Input from encoder at scale $s$: $X^{enc, s}$

Decoder block:
1. Two-stage self-attention on $X^{dec, s-1}$ (same structure as encoder)
2. Multi-head cross-attention where decoder provides queries and encoder provides keys/values
3. Feed-forward + LayerNorm residuals

Decoder at scale 0 uses a fixed sinusoidal positional encoder as input (standard Vaswani).

### Component 5: Prediction Layer (§3.5)

- Each scale $s$ has a decoder output $X^{dec, s}$
- Pass each through a separate MLP
- Concatenate MLP outputs across all scales
- Final MLP → single RUL value

---

## Best Hyperparameters (Table 4 of paper)

| Parameter | FD001 | FD002 | FD003 | FD004 |
|:----------|:-----:|:-----:|:-----:|:-----:|
| Learning rate | 0.0002 | 0.0002 | 0.0002 | 0.0002 |
| Batch size | 32 | 64 | 32 | 64 |
| Optimizer | Adam | Adam | Adam | Adam |
| Time series length $T$ | 32 | 64 | 48 | 64 |
| Number of layers/scales $S$ | 3 | 4 | 1 | 4 |
| Embedding dim $d_{model}$ | 128 | 64 | 128 | 256 |
| Number of MHA heads | 1 | 4 | 1 | 4 |

Note: FD003 uses only 1 scale (no hierarchical merging). FD001/FD003 use fewer heads despite larger d_model. FD002/FD004 require longer sequences and more layers due to multi-condition complexity.

---

## Data Protocol

### Dataset
- NASA C-MAPSS turbofan engine simulation dataset
- Download: [NASA Prognostics Data Repository](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/) — search for "Turbofan Engine Degradation Simulation Data Set"
- 4 subsets: FD001 (1 condition, 1 fault), FD002 (6 conditions, 1 fault), FD003 (1 condition, 2 faults), FD004 (6 conditions, 2 faults)

### Train/test splits (Table 1 of paper)
| Subset | Train engines | Test engines |
|:------:|:-------------:|:------------:|
| FD001  | 100 | 100 |
| FD002  | 260 | 259 |
| FD003  | 100 | 100 |
| FD004  | 249 | 248 |

### Sensor selection
Use 14 sensors: **2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21**. Drop the others (constant or uninformative).

Sensor names (Table 2 of paper):
- s2=T24, s3=T30, s4=T50, s7=P30, s8=Nf, s9=Nc, s11=Ps30, s12=phi, s13=NRf, s14=NRc, s15=BPR, s17=htBleed, s20=W31, s21=W32

### Normalization
Min-max to [0, 1]:
$$x' = \frac{x - x_{min}}{x_{max} - x_{min}}$$

**IMPORTANT**: The paper does NOT specify per-operating-condition normalization. For FD002/FD004 this is a known issue — the 6 operating conditions produce very different sensor baselines. STAR's result suggests the architecture handles this via the sensor-wise attention without needing per-condition normalization, but implementers should verify by trying both.

### RUL Labels
Piecewise-linear with cap:
$$\text{RUL}(t) = \min(T_{\text{max}} - t, \text{RUL}_{\text{max}})$$

Paper uses $\text{RUL}_{\text{max}} = 125$ (implied by "truncated linear model" and consistency with other papers). Explicit value not stated in the STAR paper — must verify from reference [52,53] cited in §4.1.

### Evaluation Protocol
- **RMSE (primary)**: Computed on test engines at their **last observed sequence window only**. Each test engine has one RUL prediction (at the last available cycle), compared to ground truth provided in `RUL_FDXXX.txt`.
- **Score (PHM 2008)**: Asymmetric exponential penalty
$$\text{Score} = \sum_{i=1}^{N} \begin{cases} e^{-d_i/13} - 1 & d_i < 0 \text{ (early prediction)} \\ e^{d_i/10} - 1 & d_i \geq 0 \text{ (late prediction)} \end{cases}$$
where $d_i = \hat{y}_i - y_i$.

---

## Training Protocol

- **Optimizer**: Adam with learning rate 0.0002
- **Loss**: MSE on RUL predictions
- **Epochs**: Not explicitly stated in the paper — use early stopping on a held-out validation split (15% of training engines) with patience 20.
- **Device**: Paper used 4× NVIDIA RTX 3080. We'll use available GPU(s).

---

## Implementation Priority

1. Download C-MAPSS dataset
2. Write data loader: load CSV files, select 14 sensors, min-max normalize per sensor
3. Compute piecewise-linear RUL labels with cap 125
4. Implement sliding window data generator (stride 1 during training, last window only for test)
5. Implement the two-stage attention encoder block
6. Implement patch merging
7. Implement the two-stage attention decoder block
8. Wire up the hierarchical encoder-decoder with multi-scale outputs
9. Run test pipeline (5 epochs on FD001) to verify no NaN, shapes correct
10. Train FD001 with 5 seeds → verify RMSE ≈ 10.61 (our target)
11. Train FD002, FD003, FD004 → compare all to Table 5

---

## Known Unknowns and Risks

1. **Patch length L**: Not specified per subset. Must infer from "time_length T" and "number of layers/scales S". Try L=4, L=8, L=16.
2. **Number of training epochs**: Not specified. Use early stopping with patience 20.
3. **RUL cap**: Paper implies 125 but doesn't state it explicitly. This is critical — 120 vs 125 vs 130 changes RMSE by ~10%.
4. **Multi-condition handling in FD002/FD004**: Paper does not describe per-condition normalization or conditioning inputs. The 13.47 RMSE on FD002 is very strong — verify whether STAR really achieves this without per-condition normalization.
5. **Sliding window stride**: Probably 1 during training. Test evaluation uses only the last window per engine.
6. **Dropout**: Not specified. Standard value 0.1 in attention and FFN layers.
7. **Parameter initialization**: Not specified. Use PyTorch defaults (Xavier/Glorot for linear layers).
8. **Dimension-wise segmentation detail**: Whether patches are non-overlapping (K disjoint patches as stated) or overlapping. Assume non-overlapping.
