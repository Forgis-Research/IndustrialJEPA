# LeJEPA Replication Specification

**Paper:** "LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics"
**Authors:** Randall Balestriero, Yann LeCun
**Source:** arXiv:2511.08544 (2025)
**Official code:** https://github.com/rbalestr-lab/lejepa
**Package:** `pip install lejepa`

---

## 1. Paper Digest

### What LeJEPA does

LeJEPA replaces all heuristic collapse-prevention mechanisms in JEPA (EMA, stop-gradient, VICReg) with a single, theoretically grounded regularizer: **SIGReg** (Sketched Isotropic Gaussian Regularization).

The core claim: if the embedding distribution is an isotropic Gaussian N(0, I_D), collapse is impossible by construction. Instead of preventing collapse via structural tricks (teacher-student asymmetry, momentum), you enforce the right distribution via a loss term.

### What is SIGReg?

**SIGReg** enforces that the encoder output distribution matches N(0, I_D) using the Cramér-Wold theorem: two distributions in R^D are identical iff all their 1D projections are identical.

**Algorithm:**
1. Sample M random unit vectors {a_m} from the unit sphere S^{D-1}
2. For each direction, project all embeddings: s_m = a_m^T · z (scalar per sample)
3. Test each 1D projection against N(0,1) using the **Epps-Pulley** characteristic function test
4. Loss = average test statistic across M directions

**Epps-Pulley test:**
```
EP(X_1,...,X_N) = N · ∫ |φ̂_X(t) - φ_{N(0,1)}(t)|² · w(t) dt
```
where φ̂_X(t) = (1/N) Σ exp(i·t·X_j) is the empirical characteristic function and w(t) = exp(-t²/σ²) is a Gaussian weight. Computed via 17-point numerical quadrature.

**Full objective:**
```
L_LeJEPA = (1 - λ) · L_pred + λ · SIGReg(encoder_output)
```
Default λ = 0.05, stable for λ ∈ [0.01, 0.2].

### Key theoretical results

| Theorem | Statement | Implication |
|---------|-----------|-------------|
| Thm 1 (Optimality) | N(0, I_D) uniquely minimizes integrated squared bias for linear/k-NN/kernel probes | Isotropic Gaussian is the provably best embedding distribution |
| Thm 3 (VICReg's dilemma) | Moment-based tests cannot simultaneously have stable gradients AND detect all collapse modes | Explains why VICReg-style variance/covariance terms are insufficient |
| Thm 4 (Gradient bounds) | \|∂EP/∂z_i\| ≤ 4σ²/N, curvature ≤ O(σ³/N) | SIGReg cannot cause gradient explosion — stability by construction |
| Thm 5 (Sketching error) | Approximation error ≤ C · M^{-2α/(D-1)} | M = O(D) random projections suffices |
| Thm 6 (Minibatch bias) | Bias = O(1/N) | Negligible for batch ≥ 128 |

### What collapses does SIGReg prevent?

1. **Complete collapse** — all embeddings → same point (impossible for N(0,I))
2. **Dimensional collapse** — embeddings on a low-rank subspace (impossible for isotropic distribution)
3. **Representational anisotropy** — some dims have much more variance (forbidden by isotropy)

### Key experimental numbers

| Model | Dataset | Epochs | Metric |
|-------|---------|--------|--------|
| ViT-H/14 | ImageNet-1K | 100 | 79.0% linear probe |
| ViT-L/14 | ImageNet-1K | 100 | 75.08% linear probe |
| Galaxy10 (in-domain) | Galaxy10 | 100 | Beats DINOv2 transfer at all label budgets |
| 1D time series | FordA (UCR) | — | 87.7% with 1% labels |
| 1D time series | ECG | — | >91% with 39 labeled examples |

**Critical finding:** Loss-performance Spearman ρ ≈ 0.99 on ImageNet. You can do model selection without running a supervised probe.

---

## 2. Existing Implementation in Our Codebase

**SIGReg is already implemented:**
- `mechanical-jepa/src/models/sigreg.py` (226 lines) — both moments and MMD variants
- `mechanical-jepa/src/models/jepa_v3.py` (169 lines) — MechanicalJEPA V3 with SIGReg instead of EMA
- Archived training script: `mechanical-jepa/archive/train_v3_sigreg.py`

**Status:** Code exists but has NOT been systematically validated against paper results. The V3 model was written but never run through the full V11-V14 experiment pipeline.

---

## 3. Replication Experiments

### Experiment A: Smoke Test — Isotropy Enforcement (15 min)

**Goal:** Verify SIGReg drives an anisotropic distribution toward isotropy.

```python
import torch
# Use our existing implementation
from mechanical_jepa.models.sigreg import sigreg_loss

# Create anisotropic data: PC1 has 10x variance (mimics our encoder's 47.6% PC1)
z = torch.randn(256, 64)
z[:, 0] *= 10.0
z.requires_grad_(True)

opt = torch.optim.Adam([z], lr=1e-2)
for step in range(200):
    loss = sigreg_loss(z, n_projections=512, method='moments')
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 40 == 0:
        _, s, _ = torch.pca_lowrank(z.detach(), q=10)
        ev_pct = (s**2) / (s**2).sum()
        print(f"step {step}: loss={loss.item():.4f}, PC1={ev_pct[0]:.3f}, PC2={ev_pct[1]:.3f}")

# Expected: PC1 drops from ~0.65 toward ~0.15 (uniform = 0.1 for top-10 of 64 dims)
```

**Pass criterion:** PC1 explained variance decreases monotonically; final distribution is approximately uniform across top PCs.

### Experiment B: Compare Our SIGReg vs Official Package (30 min)

**Goal:** Verify our `sigreg.py` produces the same gradients as the official `lejepa` package.

```bash
pip install lejepa
```

```python
import torch, lejepa
from mechanical_jepa.models.sigreg import sigreg_loss

z = torch.randn(128, 256, requires_grad=True)

# Official
ep_test = lejepa.univariate.EppsPulley(num_points=17)
official_fn = lejepa.multivariate.SlicingUnivariateTest(ep_test, num_slices=512)
loss_official = official_fn(z)

# Ours (moments method is NOT the same as EP — just verify the API works)
loss_ours = sigreg_loss(z, n_projections=512, method='moments')

print(f"Official EP loss: {loss_official.item():.4f}")
print(f"Our moments loss: {loss_ours.item():.4f}")
# These will differ (different test statistics), but both should be positive and decrease with isotropy
```

**If our implementation uses moments (mean/var/skew/kurtosis) rather than Epps-Pulley:** consider switching to the official EP-based test for theoretical guarantees. The moments method is VICReg-like and subject to Theorem 3's stability-identifiability dilemma.

### Experiment C: Loss-Performance Correlation on C-MAPSS (1 hour)

**Goal:** Test whether LeJEPA's ρ ≈ 0.99 loss-performance correlation holds on our data.

1. Replace EMA + var_reg with SIGReg in the V2 training loop
2. Pretrain for 50 epochs, saving checkpoints every 5 epochs
3. For each checkpoint: compute (a) SIGReg + prediction loss, (b) frozen RUL probe RMSE on FD001
4. Measure Spearman ρ between the two sequences

**Config:**
- Architecture: V2 (d_model=256, 2L, 4H) — same as our validated baseline
- SIGReg: λ=0.05, M=512 slices, EP test with 17 quadrature points
- Batch: 64 (gradient accumulate to 128 if unstable)
- Data: FD001, 85/15 train/val split, same preprocessing

**Pass criterion:** Spearman ρ ≥ 0.8 between training loss and probe RMSE across epochs. If ρ < 0.5, check: (a) is SIGReg applied to encoder output or predictor output? (should be encoder), (b) is the prediction loss dominating? (try λ=0.1).

### Experiment D: Head-to-Head — EMA vs SIGReg on FD001 (2 hours)

**Goal:** Directly answer whether SIGReg improves our Trajectory JEPA.

| Config | EMA | Collapse prevention | λ |
|--------|-----|--------------------|----|
| V2 baseline | τ=0.99 | variance regularizer | — |
| SIGReg-only | none (same encoder for context & target, stop-grad on target) | SIGReg (EP) | 0.05 |
| EMA + SIGReg | τ=0.99 | SIGReg (EP) | 0.05 |

Run 3 seeds each. Metrics per config:
- Frozen probe RMSE (100% labels)
- E2E RMSE (100% labels)
- PC1 explained variance (should be lower with SIGReg = more isotropic)
- PC1 correlation with RUL (should stay high — we want isotropy without losing signal)
- Epoch of best frozen probe (SIGReg should NOT have epoch-2 pathology)
- Training loss curve stability (SIGReg should be smoother)

**Hypothesis:** SIGReg-only matches or beats V2 baseline on frozen RMSE, with more stable training and lower PC1 dominance. If PC1-RUL correlation drops, SIGReg may spread signal across PCs (good for k-NN/MLP probes, potentially worse for linear probes).

---

## 4. Critical Analysis: SIGReg for Grey Swan Prediction

### Why SIGReg matters for our paper

1. **Pretraining instability.** Our best probe is at epoch 2 for some configs (V1). The variance regularizer `relu(0.1 - pred_var)` is a moment-based heuristic — exactly what Theorem 3 says is insufficient. SIGReg provides continuous, theoretically grounded collapse prevention.

2. **PC1 dominance (47.6%).** Our encoder is anisotropic. SIGReg forces isotropy. The question is whether the dominant PC1 direction is signal (degradation) or an artifact. If signal → SIGReg may hurt linear probes but help k-NN/MLP. If artifact → SIGReg should improve everything.

3. **Memory savings.** Removing the EMA target encoder saves ~1.26M parameters (50% of model). For multi-domain deployment across 5 benchmarks, this matters.

4. **Model selection without labels.** The ρ ≈ 0.99 loss-performance correlation means we could select pretraining hyperparameters without running expensive downstream probes. For a multi-domain benchmark with 15+ dataset/config combinations, this is a massive practical advantage.

5. **Theoretical grounding.** Our paper already cites LeJEPA in related work. If SIGReg works on our data, we upgrade from "we cite their theory" to "we validate their theory on industrial time series" — a stronger contribution.

### How SIGReg changes the architecture diagram

Current (V2):
```
x_past → Context Encoder (f_θ) → h_past → Predictor (g_φ) → ĥ_fut
                    ↕ EMA                                        ↓ L1 Loss
x_future → Target Encoder (f̄_ξ) → h_fut ─── sg[·] ──────────→ ↑
```

With SIGReg (V3):
```
x_past → Encoder (f_θ) → h_past → Predictor (g_φ) → ĥ_fut
              │                                         ↓ L1 Loss
              ├── SIGReg(h_past)                        ↑
              │                                         │
x_future → Encoder (f_θ, no_grad) → h_fut ── sg[·] ──→ ↑
```

Key difference: **single encoder** (no EMA copy), SIGReg applied to encoder output. The target branch uses the same encoder with `torch.no_grad()`.

### What SIGReg does NOT solve

- **The STAR gap** (14.23 vs 12.19 RMSE). SIGReg improves representation quality but the gap is likely architectural (STAR has specialized attention), not representation-quality.
- **FD002 distribution shift.** SIGReg makes embeddings isotropic, but doesn't solve per-condition calibration mismatch.
- **Low-label 5% variance.** SIGReg may improve or worsen this — the isotropic constraint could reduce variance (more stable representations) or increase it (linear probe loses the dominant direction).

### Recommended integration path

1. **Run Experiments A-D above** to validate on our data
2. **If SIGReg matches or beats V2:** Add as a V15 variant, report in architectural ablations alongside full-sequence and cross-sensor
3. **If SIGReg significantly beats V2:** Promote to default, reframe paper's method section to use SIGReg (cite LeJEPA as theoretical foundation)
4. **If SIGReg hurts:** Report as honest negative result in ablations — "isotropy constraint conflicts with linear-probe evaluation when degradation signal is concentrated in a single direction"

### Connection to SFA theory

Our paper's theoretical rationale (Section 5.3) connects trajectory prediction to Slow Feature Analysis. SIGReg adds a complementary theoretical angle: the isotropic Gaussian constraint from LeJEPA's Theorem 1 says the optimal embedding distribution for downstream generalization is N(0, I_D). Our SFA argument says the L1 prediction loss biases toward slow features (degradation). Together:

> **The trajectory-prediction objective selects *which* features to encode (slow-varying degradation), while SIGReg constrains *how* they are encoded (isotropically) — making the representations maximally useful for any downstream probe.**

This is a clean theoretical story for the paper if the experiments support it.

---

## 5. Files and References

| Resource | Path |
|----------|------|
| LeJEPA paper PDF | `paper-replications/LeJEPA/LeJEPA-2025-SIGReg.pdf` |
| Our SIGReg implementation | `mechanical-jepa/src/models/sigreg.py` |
| V3 model (SIGReg-integrated) | `mechanical-jepa/src/models/jepa_v3.py` |
| Archived V3 training script | `mechanical-jepa/archive/train_v3_sigreg.py` |
| Official package | `pip install lejepa` |
| Official repo | https://github.com/rbalestr-lab/lejepa |
| Time series adaptation | https://github.com/driano1221/LeJEPA-TimeSeries |
| BibTeX key | `balestriero2025lejepa` in `paper-neurips/references.bib` |
