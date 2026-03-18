# Deep Research: JEPA for Time Series & Novel Ideas

## Executive Summary

This document synthesizes research on JEPA for time series, state-of-the-art foundation models, and novel architectural ideas for achieving SOTA on time series forecasting benchmarks.

---

## 1. JEPA for Time Series - Existing Work

### 1.1 TS-JEPA (KTH Stockholm, Sept 2025)
- **Key Innovation**: High masking ratios (>70%), patch tokenization, lightweight transformers
- **Finding**: JEPA achieves "great balance between classification AND forecasting"
- **Paper**: [arxiv.org/abs/2509.25449](https://arxiv.org/abs/2509.25449)

### 1.2 MTS-JEPA (Feb 2026) - HIGHLY RELEVANT
- **Key Innovation**: Multi-resolution prediction + soft codebook bottleneck
- **Why Important**:
  - Decouples transient shocks from long-term trends
  - Codebook captures discrete regime transitions
  - Prevents representation collapse
- **Paper**: [arxiv.org/abs/2602.04643](https://arxiv.org/abs/2602.04643)

### 1.3 LaT-PFN (May 2024)
- **Key Innovation**: Prior-data Fitted Networks + JEPA = zero-shot forecasting
- **Finding**: Trained exclusively on synthetic data, generalizes to real
- **Code**: [github.com/StijnVerdenius/Lat-PFN](https://github.com/StijnVerdenius/Lat-PFN)

### 1.4 C-JEPA (NeurIPS 2024) - CRITICAL
- **Key Innovation**: VICReg regularization prevents JEPA collapse
- **Why Important**: EMA alone doesn't prevent entire collapse
- **Paper**: [NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/file/04a80267ad46fc730011f8760f265054-Paper-Conference.pdf)

---

## 2. Foundation Model Insights

### What Makes SOTA Models Work

| Model | Key Innovation | Weakness |
|-------|----------------|----------|
| **Chronos-2** | First TSFM with univariate + multivariate + covariates | Requires tokenization |
| **Moirai-MoE** | Sparse MoE for diverse patterns, 17% better | MoE routing overhead |
| **Toto** | Student-t mixture (heavy tails), 2T training points | Optimized for observability |
| **iTransformer** | Attention across variates, not time | - |
| **PatchTST** | Patching + channel independence | Ignores cross-variate |

### Key Architectural Lessons

1. **Patching is essential** - all top models use it
2. **RevIN for distribution shift** - reversible instance normalization
3. **Cross-channel modeling matters** - iTransformer approach
4. **Heavy-tailed outputs** - Student-t mixture for industrial data
5. **Multi-resolution** - capture different timescales

---

## 3. Critical Techniques

### 3.1 Preventing Collapse
- **VICReg**: Variance-Invariance-Covariance Regularization
- **Soft Codebook**: Discrete bottleneck acts as regularizer
- **High Masking (>70%)**: Forces learning meaningful representations

### 3.2 Multi-Step Prediction
| Strategy | Pros | Cons |
|----------|------|------|
| Recursive | Single model | Error accumulation |
| Direct | No error propagation | One model per horizon |
| **MIMO (our approach)** | Single pass all horizons | Fixed horizon |
| Latent rollout | Efficient | Requires good latent space |

### 3.3 Graph Neural Networks for Cross-Channel
- **DyGraphformer**: Dynamic graph learning (static graphs don't reflect reality)
- **Key Insight**: Graph structure should be learned, not hand-crafted

### 3.4 State Space Models (Mamba)
- **MambaTS**: Linear complexity, variable-aware scan
- **Benefit**: Much faster than Transformers for long sequences

---

## 4. Novel Ideas for IndustrialJEPA

### Idea 1: Hierarchical Multi-Resolution JEPA + Koopman (HMK-JEPA)
**Concept**: Hierarchy of latent spaces at different temporal resolutions. Koopman operators linearize dynamics at each level.

**Benefits**:
- Interpretable regime identification via Koopman eigenvalues
- Efficient long-horizon rollouts (linear dynamics)
- Multi-scale anomaly precursor detection

### Idea 2: Graph-Structured JEPA with Evolving Topology (GS-JEPA)
**Concept**: JEPA predictor operates on dynamically learned sparse graphs. Predict future states AND future graph topology.

**Why Novel**: Industrial systems have evolving sensor relationships.

### Idea 3: Diffusion-Enhanced JEPA Uncertainty (DE-JEPA)
**Concept**: Diffusion models for distribution of possible future latent states.

**Benefits**:
- Uncertainty quantification for anomaly prediction
- Sample diverse futures
- Conformal calibration on latent distances

### Idea 4: Test-Time Adaptive JEPA (TTA-JEPA)
**Concept**: Online adaptation when deployed on new machines.

**Approach**:
- Frozen pre-trained encoder + predictor
- LoRA-style adapter layers
- Online optimization using self-supervised JEPA loss

### Idea 5: Physics-Informed JEPA (PI-JEPA)
**Concept**: Inject known physical constraints into training.

**Benefits**:
- Predicted states satisfy physical invariants
- Anomaly = violation of physics in latent space

### Idea 6: Neural ODE-JEPA (NODE-JEPA)
**Concept**: Neural ODE predictor for irregular sampling.

**Why Important**: Industrial data has missing values, variable rates.

### Idea 7: Causal JEPA (CI-JEPA)
**Concept**: Learn causal structure in latent space for "what if" questions.

**Benefits**:
- Counterfactual prediction
- Root cause analysis
- Intervention planning

---

## 5. Implementation Recommendations

### For ETTh1 Benchmark (Current Task)

1. **Keep**: Patching, RevIN, EMA target encoder
2. **Add**: VICReg regularization (prevents collapse)
3. **Try**: iTransformer-style cross-variate attention
4. **Consider**: Multi-resolution patches (different sizes concatenated)

### For Industrial World Model (Long-term)

1. **Base**: MTS-JEPA architecture (multi-resolution + soft codebook)
2. **Stability**: VICReg from C-JEPA
3. **Cross-channel**: iTransformer attention or learned graphs
4. **Deployment**: Test-time adaptation for new machines
5. **Uncertainty**: Conformal prediction on latent distances
6. **Efficiency**: Mamba blocks for very long sequences

---

## 6. Key Papers to Read

| Priority | Paper | Why |
|----------|-------|-----|
| 1 | MTS-JEPA | Multi-resolution + codebook, directly relevant |
| 2 | C-JEPA | VICReg for stable training |
| 3 | iTransformer | Cross-variate attention |
| 4 | TS-JEPA | JEPA for time series baseline |
| 5 | Toto | Student-t mixture, factorized attention |

---

## 7. Code Resources

- [TS-JEPA](https://github.com/PierreWANG-dev/ts-jepa) - Time series JEPA implementation
- [iTransformer](https://github.com/thuml/iTransformer) - Cross-variate attention
- [PatchTST](https://github.com/yuqinie98/PatchTST) - Patching reference
- [Toto](https://github.com/DataDog/toto) - Student-t mixture head
- [Time-Series-Library](https://github.com/thuml/Time-Series-Library) - Benchmarking

---

## 8. Summary: Path to SOTA

```
Current Approach (ETTh1)
├── Patching ✓
├── RevIN ✓
├── JEPA predictor ✓
├── EMA target encoder ✓
└── Missing:
    ├── VICReg regularization (prevents collapse)
    ├── Cross-variate attention (captures sensor dependencies)
    └── Multi-resolution (different timescales)

Enhanced Approach
├── Add VICReg loss term
├── Add iTransformer-style attention layer
├── Try soft codebook bottleneck
└── Experiment with multi-resolution patches
```

**Key Insight**: Current SOTA models succeed by combining multiple innovations. A JEPA that integrates VICReg + cross-variate attention + multi-resolution has strong potential to beat existing approaches.
