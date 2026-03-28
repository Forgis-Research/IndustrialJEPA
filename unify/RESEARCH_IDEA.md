# Unify: Cross-Morphological Latent Alignment for Universal Physics Forecasting

## Abstract

A foundation model approach for time-series forecasting that maps heterogeneous mechanical systems (varying sensor counts and types) into a **shared latent trajectory space**. The model learns to forecast in this "idealized" physics space, enabling zero-shot transfer across system morphologies.

---

## 1. Core Hypothesis

> Diverse mechanical systems (simple pendulum → complex CNC mill) are specific projections of the same **Universal Physics Manifold**.

Instead of training isolated models per system, we:
1. Map heterogeneous input spaces into a **fixed-dimension latent trajectory**
2. Train a foundation model to forecast in this latent "physics" space
3. Decode back to system-specific observations

---

## 2. Key Innovations

### 2.1 Input-Dimension Agnostic Architecture

| Problem | Standard SOTA | Our Approach |
|---------|---------------|--------------|
| Variable # of features | Fixed input dim required | **Set-based encoding** (DeepSets, Perceiver-style latent bottlenecks) |
| New sensor added | Retrain from scratch | Encode as additional set element |

### 2.2 Zero-Shot Physics Transfer

The model doesn't just predict numbers—it transfers **dynamic logic**:
- Train on 2-joint and 3-joint systems
- Achieve superior performance on **never-seen 4-joint system**
- Works because the model learns the *recursive nature* of multi-body dynamics

### 2.3 Small Data Superiority

By leveraging shared knowledge across 100 small datasets:
- Outperform dedicated supervised models trained on single datasets
- Effective "physics pre-training" from diverse mechanical instances

---

## 3. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    HETEROGENEOUS INPUTS                         │
│  System A: [x₁, x₂, x₃]     System B: [y₁, y₂]     System C: [z₁...z₁₀] │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              SET-BASED ENCODER (DeepSets/Perceiver)             │
│  • Treats each sensor as a "token" with learned embedding       │
│  • Permutation invariant aggregation                            │
│  • Optional: sensor-type embeddings, positional encodings       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              SHARED LATENT SPACE (Fixed Dimension D)            │
│                    "Universal Physics Manifold"                 │
│           z_t ∈ ℝ^D  for all systems regardless of input dim   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                 LATENT DYNAMICS MODEL                           │
│  • Transformer / SSM / Neural ODE operating in latent space     │
│  • Learns "laws of motion" not "statistical patterns"           │
│  • z_{t+1} = f(z_t, z_{t-1}, ...)                              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              SYSTEM-SPECIFIC DECODER                            │
│  • Lightweight MLP per system (or learned query vectors)        │
│  • Projects latent back to observation space                    │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. Proof-of-Concept: N-Link Pendulum Universe

### 4.1 Synthetic Data Generation

| Component | Details |
|-----------|---------|
| **Physics Engine** | MuJoCo or PyBullet |
| **Systems** | N-link pendulums where N ∈ {1, 2, 3, 5} |
| **Variation** | Randomize mass (m), length (l), friction (b) per instance |
| **Observations** | Joint angles θ, angular velocities θ̇ |

### 4.2 Training Protocol

- **Training set**: N = 1, 2, 3 pendulums (multiple instances each)
- **Validation**: Held-out instances of N = 1, 2, 3
- **Zero-shot test**: N = 5 pendulums (never seen during training)

### 4.3 Success Criteria

| Metric | SOTA Baseline | Our Model |
|--------|---------------|-----------|
| In-distribution (N=1,2,3) | Competitive | Competitive or better |
| Out-of-distribution (N=5) | **Fails** (wrong input shape) | **Works** via latent physics |
| Low-data regime | Limited | Superior (cross-system pretraining) |

---

## 5. Generalization Beyond Mechanics

The framework generalizes to:
> **Any collection of multivariate time series that share underlying generative structure**

Examples:
- **Robotics**: Different robot morphologies (4-leg, 6-leg, wheeled)
- **Manufacturing**: Various CNC machines with different sensor configurations
- **Energy**: Wind turbines with varying instrumentation
- **Healthcare**: Patient monitoring with different device setups

### Key Requirement
Instances must share *something*—underlying physics, causal structure, or generative process.

---

## 6. Research Questions

1. **Latent dimension**: What is the minimal D that captures "universal physics"?
2. **Encoder design**: DeepSets vs Perceiver vs Set Transformer—which works best?
3. **Decoder strategy**: Shared decoder with conditioning vs system-specific decoders?
4. **Training objective**: Reconstruction vs contrastive vs predictive?
5. **Scaling laws**: How does performance scale with # of training systems?

---

## 7. Related Work

- **JEPA** (Joint Embedding Predictive Architecture) — prediction in latent space
- **Perceiver / Perceiver IO** — handling arbitrary input modalities
- **DeepSets** — permutation-invariant set functions
- **Neural ODEs** — continuous-time dynamics modeling
- **Physics-Informed Neural Networks** — embedding physical constraints
- **Meta-learning for dynamics** — MAML, Reptile for quick adaptation

---

## 8. Implementation Roadmap

### Phase 1: Data Generation
- [ ] Set up MuJoCo/PyBullet environment
- [ ] Generate N-link pendulum datasets (N=1,2,3,5)
- [ ] Create data loaders for variable-length inputs

### Phase 2: Architecture
- [ ] Implement set-based encoder
- [ ] Implement latent dynamics model
- [ ] Implement system-specific decoders

### Phase 3: Training
- [ ] Multi-dataset training loop
- [ ] Curriculum learning (simple → complex systems)
- [ ] Evaluation metrics and baselines

### Phase 4: Evaluation
- [ ] In-distribution forecasting accuracy
- [ ] Zero-shot transfer to N=5
- [ ] Comparison vs supervised baselines
- [ ] Ablation studies

---

## 9. Expected Outcomes

1. **Novel architecture** for cross-morphology time series forecasting
2. **Demonstrated zero-shot transfer** across system complexities
3. **Benchmark** on synthetic N-link pendulum domain
4. **Framework** applicable to real industrial systems (C-MAPSS, etc.)
