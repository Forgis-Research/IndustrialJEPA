# Cross-Embodiment Transfer Learning for Industrial Robots

## The Core Challenge

Different machines have different sensors, actuators, and dynamics. A model trained on Machine A must generalize to Machine B without retraining. This is **cross-embodiment transfer**.

### Why It Should Work (The Physics Hypothesis)

Industrial robots share fundamental physics:
- **Dynamics**: F = ma, torque-velocity relationships, friction models
- **Control loops**: PID responses, setpoint tracking behavior
- **Failure modes**: Wear patterns, anomalous friction, misalignment

If a model learns these physics-based representations (not machine-specific quirks), it transfers.

---

## Rapid Validation: Before Training Anything

### 1. Feature Distribution Similarity

**Maximum Mean Discrepancy (MMD)**
```
MMD²(P, Q) = E[k(x,x')] + E[k(y,y')] - 2E[k(x,y)]
```
- Compares distributions in kernel space
- MMD ≈ 0 → distributions are similar → transfer likely
- Use RBF kernel with median heuristic for bandwidth

**Wasserstein Distance (Earth Mover's Distance)**
- Measures "work" to transform one distribution into another
- More interpretable than MMD: W(P,Q) = 5 means "5 units of shift"
- Compute per-channel, then aggregate

**Implementation**:
```python
from scipy.stats import wasserstein_distance
from sklearn.metrics.pairwise import rbf_kernel

def compute_mmd(X, Y, gamma=1.0):
    XX = rbf_kernel(X, X, gamma)
    YY = rbf_kernel(Y, Y, gamma)
    XY = rbf_kernel(X, Y, gamma)
    return XX.mean() + YY.mean() - 2 * XY.mean()
```

### 2. Cross-Correlation Fingerprinting

Each machine has a "fingerprint" of how sensors relate:
- Motor torque ↔ velocity: What's the correlation?
- Position error ↔ effort: How does the controller respond?

**Method**:
1. Compute correlation matrix for each machine
2. Compare matrices (Frobenius norm of difference)
3. Similar fingerprints → similar physics → transfer works

**Visual**: Heatmaps of correlation matrices side-by-side.

### 3. Entropy-Profile Matching (TIMETIC-style)

From [TIMETIC paper](https://arxiv.org/abs/2312.16386):
- Compute permutation entropy of each signal
- Compare entropy profiles between machines
- Similar complexity → similar learnability

**Why it works**: If Machine B has much higher entropy (more chaotic), the simpler patterns learned from A won't apply.

---

## The Signal Alignment Problem

### Current Approach (Padding)
```
AURSAD:  [torque_0..5, force_xyz, ...]  → 13 dims
Voraus:  [voltage_0..5, ...]            → 6 dims
Output:  [signal_0..13, validity_mask]  → padded to max
```

**Problem**: Torque ≠ Voltage. Padding doesn't capture semantics.

### Better Approach: Semantic Signal Embedding

From [CHARM paper](https://arxiv.org/abs/2505.14543):
```
"effort_torque_0" → LLM/SentenceTransformer → embedding → shared space
"effort_voltage_0" → LLM/SentenceTransformer → embedding → shared space
```

The model learns that "torque" and "voltage" are both "effort" signals, just different modalities.

**Implementation sketch**:
```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode([
    "joint torque sensor measuring rotational force in Nm",
    "motor voltage sensor measuring electrical potential in V",
])
# These will be close in embedding space (both effort-related)
```

---

## What Transfers vs. What Doesn't

### Likely to Transfer (Shared Physics)
| Pattern | Why |
|---------|-----|
| Effort-velocity relationships | Newton's laws |
| Anomaly signatures | Friction, wear are universal |
| Control response patterns | PID behavior is similar |
| Temporal dynamics | Inertia, damping are physical |

### Unlikely to Transfer (Machine-Specific)
| Pattern | Why |
|---------|-----|
| Absolute sensor ranges | Different actuators |
| Specific failure thresholds | Different wear characteristics |
| Exact waveform shapes | Different gear ratios, masses |

---

## Validation Protocol

### Phase 1: Statistical Sanity Check (No Training)
1. Load both datasets
2. Compute MMD, Wasserstein for aligned signals
3. Compute correlation fingerprints
4. **Decision gate**: If MMD > threshold, transfer unlikely

### Phase 2: Representation Probe (Minimal Training)
1. Train encoder on source (1-2 epochs)
2. Extract embeddings for both machines
3. Linear probe: Can source embeddings separate target anomalies?
4. **Decision gate**: If probe accuracy < 60%, reconsider approach

### Phase 3: Full Transfer Experiment
Only proceed if Phase 1-2 pass.

---

## Handling Many Machines

### The Scaling Problem
- 10 machines → 45 pairwise transfers to validate
- 100 machines → 4,950 pairs

### Solutions

**1. Hierarchical Clustering**
- Cluster machines by signal similarity (MMD-based)
- Train one model per cluster
- Transfer within clusters is easy

**2. Universal Encoder (One Model)**
- Train on all machines simultaneously
- Use machine ID as conditioning
- Test zero-shot on held-out machines

**3. Signal-Semantic Approach**
- Don't align machines, align signals
- "torque" from any machine maps to same embedding
- Scales to unlimited machines

---

## Quick Experiments to Run

### Experiment 1: Distribution Analysis
```bash
python analysis/cross_machine/01_distribution_analysis.py
# Output: MMD and Wasserstein tables, distribution plots
```

### Experiment 2: Correlation Fingerprints
```bash
python analysis/cross_machine/02_correlation_fingerprints.py
# Output: Correlation heatmaps, fingerprint similarity scores
```

### Experiment 3: Entropy Profiles
```bash
python analysis/cross_machine/03_entropy_profiles.py
# Output: Per-signal entropy, complexity comparison
```

### Experiment 4: Linear Probe
```bash
python analysis/cross_machine/04_linear_probe.py
# Output: Source→Target probe accuracy
```

---

## Key Papers

1. **CHARM** (C3 AI, 2025): Semantic signal embeddings for cross-machine transfer
2. **TIMETIC** (2023): Entropy-based transferability estimation
3. **RT-X** (Google, 2023): Cross-embodiment in robotics via co-training
4. **Open X-Embodiment** (2024): Scaling laws for robot transfer

---

## Our Hypothesis

> JEPA learns physics-based representations that capture effort-dynamics relationships.
> These relationships are invariant across robots performing similar tasks.
> Therefore, a model trained on AURSAD (UR3e, screwdriving) should transfer to
> Voraus (Yu-Cobot, screwdriving) with minimal degradation.

**Validation**: Zero-shot anomaly detection AUC on target ≥ 0.7 × source AUC.
