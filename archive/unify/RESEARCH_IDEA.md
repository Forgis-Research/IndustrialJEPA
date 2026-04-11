# Unify: Cross-Morphological Latent Alignment for Universal Physics Forecasting

> **UNIFY** — *Multiple related datasets are stronger than the sum of their parts?!*

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

## 8. Validation Protocols

### 8.1 Synthetic Sanity Check: N-Link Pendulums

**Setup:**
```
Training Data:    N ∈ {1, 2, 3} pendulums, 1000 trajectories each
Validation Data:  N ∈ {1, 2, 3} held-out trajectories
Zero-Shot Test:   N = 5 pendulum (NEVER seen during training)
```

**Task:** Given 50 timesteps of history, forecast next 50 timesteps of joint angles θ and velocities θ̇.

**Baselines:**

| Model | Description | Expected Result |
|-------|-------------|-----------------|
| **B1: Last-Value** | Predict θ(t+k) = θ(t) | Naive lower bound |
| **B2: Per-N Supervised** | Separate LSTM/Transformer per N | Upper bound for in-distribution |
| **B3: Fixed-Input Transformer** | Standard model, input dim = max(N)*2 | Fails on N=5 (wrong shape) OR requires zero-padding hacks |
| **B4: Our Unified Model** | Set-encoder → latent → dynamics | Should work zero-shot |

**Metrics:**
- **MSE(θ)**: Mean squared error on joint angles
- **MSE(θ̇)**: Mean squared error on angular velocities
- **Physics Violation**: Energy drift (Hamiltonian should be ~conserved)

**Success Criteria:**
```
✓ Criterion 1: On N=1,2,3 (in-distribution)
  Our model ≈ Per-N Supervised (within 10% MSE)

✓ Criterion 2: On N=5 (zero-shot)
  B3 (Fixed-Input) FAILS or degrades severely (>5x MSE vs in-distribution)
  Our model achieves <2x MSE vs our in-distribution performance

✓ Criterion 3: Physics consistency
  Our model's energy drift < B1's energy drift
```

**Ablations:**
- Latent dimension D: {32, 64, 128, 256}
- Encoder type: DeepSets vs Perceiver vs Set Transformer
- With/without physics-informed loss (energy conservation)

---

### 8.2 Real-World Validation: Cross-Robot Forecasting

**Setup:**
```
Training Data:    AURSAD (UR3e, 6-DOF, 20ch) + Voraus-AD (Yu-Cobot, 6-DOF, 66ch)
                  ~4,000 episodes total, 80% train

Validation Data:  Held-out episodes from AURSAD + Voraus-AD (10% each)

Zero-Shot Test:   OXE-KUKA (KUKA iiwa, 7-DOF, 27ch) — NEVER seen during training
```

**Task:** Given 128 timesteps of joint trajectory, forecast next 64 timesteps.

**What We Predict:**
- Joint positions q(t)
- Joint velocities q̇(t)
- (Optional) Joint efforts/torques τ(t)

**Baselines:**

| Model | Training Data | Test Data | Notes |
|-------|---------------|-----------|-------|
| **B1: AURSAD-only** | AURSAD | AURSAD | Per-dataset supervised |
| **B2: Voraus-only** | Voraus-AD | Voraus-AD | Per-dataset supervised |
| **B3: Concat-Padded** | Both (zero-padded to 66ch) | Both | Standard approach with padding |
| **B4: Our Unified** | Both (set-encoded) | Both + KUKA | Our method |

**Metrics:**
- **MAE(q)**: Mean absolute error on joint positions (radians)
- **MAE(q̇)**: Mean absolute error on joint velocities (rad/s)
- **Normalized MSE**: MSE / Var(target) — comparable across datasets
- **Trajectory Correlation**: Pearson r between predicted and true trajectory

**Success Criteria:**

```
✓ Criterion 1: In-Distribution Parity
  On AURSAD test:  Our Unified ≥ 0.95 × AURSAD-only performance
  On Voraus test:  Our Unified ≥ 0.95 × Voraus-only performance

  (Unifying shouldn't HURT in-distribution performance)

✓ Criterion 2: Cross-Robot Transfer (same DOF)
  Train on AURSAD → Test on Voraus (zero-shot)
  Our model: Normalized MSE < 2.0 (meaningful predictions)
  Concat-Padded: Normalized MSE > 3.0 (struggles with sensor mismatch)

✓ Criterion 3: Zero-Shot to New DOF (THE KEY TEST)
  Train on 6-DOF robots → Test on 7-DOF KUKA

  B3 (Concat-Padded): FAILS (input dimension mismatch, or severe degradation)
  Our model: Achieves Normalized MSE < 3.0 on first 6 joints
             Achieves Normalized MSE < 5.0 on 7th joint (extrapolated)

✓ Criterion 4: Data Efficiency
  Train Our Unified on 100% data vs B1/B2 on 20% data
  Our model (100%) > B1/B2 (20%) proves knowledge transfer

  OR: Our model with 50% data ≈ B1/B2 with 100% data
```

**Visualization:**
- Predicted vs actual joint trajectories (overlay plots)
- Latent space visualization (t-SNE/UMAP) — do robots cluster or share structure?
- Attention weights in encoder — which sensors matter?

---

### 8.3 Stretch Goal: Beat Supervised on Small Data

**The Ultimate Test:**
```
Scenario: New robot arrives with only 50 episodes of data

Approach A (Supervised): Train fresh model on 50 episodes
Approach B (Unify):      Fine-tune pre-trained Unify model on 50 episodes

Success: Unify fine-tuned > Supervised from scratch
```

This proves the "foundation model" value — pre-training on diverse robots transfers to new robots with minimal data.

---

### 8.4 Negative Results to Document

If the hypothesis is WRONG, we should find:
- Latent spaces don't align (each robot clusters separately)
- Zero-shot to 7-DOF fails completely (no transfer)
- Unified model hurts in-distribution performance (negative transfer)

These are valuable findings that inform when cross-morphological alignment works vs doesn't.

## 9. Implementation Roadmap

### Phase 1: Synthetic Sanity Check
- [ ] Set up MuJoCo/PyBullet environment
- [ ] Generate N-link pendulum datasets (N=1,2,3,5)
- [ ] Implement set-based encoder + latent dynamics
- [ ] Run validation protocol 8.1
- [ ] **Gate:** Proceed only if Criterion 2 passes (zero-shot works)

### Phase 2: Real-World Robotics
- [ ] Prepare AURSAD + Voraus-AD with unified loader
- [ ] Adapt architecture for real sensor data
- [ ] Run validation protocol 8.2
- [ ] **Gate:** Proceed only if Criterion 1+3 pass

### Phase 3: Scaling & Analysis
- [ ] Add OXE datasets (KUKA, Panda)
- [ ] Latent space analysis (do robots share structure?)
- [ ] Ablation studies (encoder type, latent dim)
- [ ] Data efficiency experiments (protocol 8.3)

### Phase 4: Paper
- [ ] Document negative results (protocol 8.4)
- [ ] Visualizations (trajectories, latent space, attention)
- [ ] Comparison tables vs baselines

---

## 10. Expected Outcomes

1. **Novel architecture** for cross-morphology time series forecasting
2. **Demonstrated zero-shot transfer** across system complexities
3. **Benchmark** on synthetic N-link pendulum domain
4. **Framework** applicable to real industrial systems (C-MAPSS, etc.)
