# Mechanical-JEPA: Results Summary

**Date**: 2026-03-27 | **Experiments**: 5 (3 seeds each) | **Data**: Open X-Embodiment

---

## One-Line Verdict

**JEPA pretraining transfers dynamics priors across robot embodiments in the few-shot regime (25-39% improvement at 10-shot), but a linear regression model dominates at 50+ shots. JEPA is not the right self-supervised objective for this domain.**

---

## Figures

- **Fig 1** (`fig1_transfer.png`): Cross-embodiment transfer — the key result + the caveat
- **Fig 2** (`fig2_overview.png`): Complete 4-panel overview (pretraining, classification, forecasting, transfer)

---

## Setup

| Component | Detail |
|-----------|--------|
| **Pretraining data** | TOTO (1,003 Franka episodes, 325k timesteps) + DROID (100 eps) |
| **Architecture** | Transformer encoder (d=128, 4 layers, 4 heads) + JEPA predictor, 2.28M params |
| **JEPA objective** | Predict future latent states from context states + actions (action-conditioned) |
| **Masking** | Temporal: 70% context, 30% target. EMA target encoder (momentum=0.996) |
| **Transfer targets** | KUKA iiwa (7-DOF), UR5 (6-DOF), JACO (6-DOF), FANUC (6-DOF) |
| **Data source** | LeRobot/HuggingFace (state + action, no images) |

---

## Results

### Exp 1: Pretraining Converges (val_loss = 0.0086 +/- 0.0019)

Training is stable. No embedding collapse. Loss drops 87% in 50 epochs. Not the bottleneck.

### Exp 2: Embodiment Classification — JEPA Hurts

| Method | Accuracy |
|--------|----------|
| Chance | 20.0% |
| **Pretrained encoder** | **65.1% +/- 1.3%** |
| Random encoder | 79.8% +/- 1.2% |
| Raw features | 81.1% +/- 1.4% |

**Pretrained is 14.7 points worse than random** (p=0.006). JEPA is trained to predict temporal dynamics, so it actively discards the static positional offsets that distinguish robots. This is by design, not a bug — but it means JEPA representations are unsuitable for classification tasks.

### Exp 3: Contact Classification — Trivially Easy (AUROC > 0.99 for all methods)

Not a useful benchmark. Joint positions perfectly predict contact in this dataset (geometric determination in simulation). No evidence for or against transfer.

### Exp 4: In-Domain Forecasting — Linear Wins

| Method | h=1 MSE | h=10 MSE |
|--------|---------|----------|
| **Linear** | **0.00007** | **0.00253** |
| Copy-last | 0.00010 | 0.00349 |
| MLP | 0.00021 | 0.00221 |
| Pretrained (finetuned) | 0.00201 | 0.01644 |
| Scratch transformer | 0.00464 | 0.01840 |

Pretraining helps 2.3x vs scratch transformer. But **linear regression is 29x better** than the pretrained encoder at h=1. Robot joint dynamics in normalized space are nearly linear.

### Exp 5: Cross-Embodiment Few-Shot Transfer — The Key Result

**Transfer ratio (pretrained MSE / scratch MSE) — lower means pretraining helps:**

| Robot | 10-shot | 50-shot | 100-shot |
|-------|---------|---------|----------|
| **KUKA iiwa** | **0.68** | **0.56** | **0.58** |
| **UR5** | **0.61** | **0.66** | 0.84 |
| **JACO** | **0.74** | **0.55** | **0.47** |
| FANUC | 1.03 | 0.91 | 0.94 |

3/4 robots show 25-39% improvement at 10-shot. Benefit persists at 50/100-shot for KUKA and JACO.

**But the elephant in the room**: at 50-shot, linear regression achieves MSE=0.006 vs pretrained encoder=0.385 (64x better). The encoder representations are only useful when you have <50 labeled windows and can't fit a reliable linear model.

---

## Why JEPA Doesn't Work Here

1. **Dynamics are nearly linear.** Normalized robot joint trajectories at 10-20 Hz are smooth and well-approximated by linear extrapolation. JEPA's nonlinear latent prediction offers no advantage over a linear model once you have enough data to fit one.

2. **JEPA discards static features.** The latent prediction objective optimizes for temporal dynamics, stripping away the position-dependent information that matters for classification and even for state-space forecasting.

3. **Channel count is too low.** With 7-8D state, there's no meaningful attention structure to learn (unlike Brain-JEPA's 450 ROIs). The masking/prediction game has limited expressiveness.

4. **Action conditioning is indirect.** All OXE actions are end-effector (Cartesian) deltas, not joint-space commands. The mapping from EE action to joint state change involves the robot's Jacobian, which JEPA must learn implicitly — a hard problem with limited data.

5. **Scale is insufficient.** 1,003 episodes (~325k timesteps) is 74x smaller than Brain-JEPA's 2.3B tokens. Self-supervised learning needs scale to outperform supervised baselines.

---

## What This Means for the Project

JEPA as a self-supervised pretraining objective for robot proprioception is **not promising enough to pursue further**. The results across all our experiments (48 from Phase 1-6 on C-MAPSS/pendulum/ETT + 5 here on OXE) consistently show:

- JEPA pretraining either doesn't help or provides marginal benefit
- Simple baselines (linear, copy-last) dominate in the low-dimensional sensor regime
- Physics-informed attention masking (our earlier work) provides more consistent value

**Recommendation**: Focus the paper on physics-informed channel masking ("When to Mask"), not JEPA. The JEPA negative results across 2 different domains (industrial sensors + robot proprioception) strengthen the paper as honest empirical findings.

---

## Reproducibility

All experiments are in `autoresearch/mechanical_jepa/experiments/run_full_suite.py` — a single self-contained script that runs pretraining + all 4 evaluation phases with configurable seeds. Data downloaded via `datasets/downloaders/download_oxe_hf.py`.
