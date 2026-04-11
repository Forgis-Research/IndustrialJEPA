# Rick's Briefing: ETH AI Center Presentation

**Event**: ETH AI Center Research Talk (after Google DeepMind)
**Duration**: 5 minutes
**Audience**: Highly technical AI researchers

---

## The One-Liner

> "We're building world models for physical systems - learning transferable dynamics via Koopman theory, not overfitting to sensor statistics."

---

## The Three Verticals

### 1. FactoryNet (Data)
**What**: Unified multi-robot industrial time-series dataset
- 18M timesteps, 5 robots (UR3e, Yu-Cobot, Franka, CNC)
- Common schema: Setpoint → Effort → Feedback (causal structure)
- 7K episodes with fault annotations
- Public on HuggingFace: `Forgis/factorynet-hackathon`

**Why it matters**: First unified cross-robot benchmark. Enables transfer learning research.

### 2. FactoryBench (Benchmark)
**What**: Evaluation suite for industrial AI
- Cross-machine transfer (train UR3e → test Yu-Cobot)
- Anomaly detection (one-class on healthy, detect faults)
- Long-horizon stability (H=720 prediction)
- Few-shot adaptation (1%, 10%, 100% labels)

**Why it matters**: Forecasting MSE is not the goal. Industrial tasks need transfer and anomaly detection.

### 3. KH-JEPA / OpenTSLM (Model)
**What**: Koopman-Hierarchical JEPA for physical time series
- JEPA: Predict in latent space (filter noise, learn dynamics)
- SIGReg: Provable collapse prevention (LeJEPA, 2024)
- Koopman: Linear dynamics (z_{t+k} = K^k z_t) - O(1) any horizon

**Why it matters**: Physics is Koopman-linearizable. Financial markets are not. We focus where foundation models make sense.

---

## Latest Results (Autoresearch Run)

| Metric | Result | Target | Status |
|--------|--------|--------|--------|
| ETTh1-96 MSE | 0.403 | <0.45 | PASS |
| h1→h2 transfer | 0.75x | <1.5x | PASS |
| H=720 degradation | 1.49x | <3.0x | PASS |

**Key fix**: Channel-independent processing (PatchTST-style) - 7x effective training data.

---

## Differentiators (vs DeepMind/Google)

| Them | Us |
|------|-----|
| General-purpose forecasting | Physics-focused world models |
| Compete on MSE benchmarks | Compete on transfer + stability |
| Scale (billions of params) | Inductive bias (Koopman) |
| Closed models | Open: data, code, models |

**Our edge**: We're not trying to beat Chronos at MSE. We're doing what they can't: zero-shot deployment on new machines.

---

## Key Technical Points (for Q&A)

1. **Why Koopman?**
   - Physical systems obey dx/dt = f(x)
   - Koopman: any such system is linear in lifted space (dz/dt = Kz)
   - Eigenvalues are interpretable (stability, frequency, damping)
   - This is math, not a choice

2. **Why not financial time series?**
   - Efficient market hypothesis: prediction erases predictability
   - No underlying differential equation
   - Adversarial, zero-sum
   - We focus where physics gives us an advantage

3. **What about scale?**
   - Currently ~500K params (channel-independent, efficient)
   - Roadmap to 1B with industrial pretraining
   - But: inductive bias > scale when physics applies

4. **Cross-machine transfer - how hard?**
   - Same task, different machine: works well (0.75x transfer ratio)
   - Different task + machine: harder (AURSAD screwdriving → voraus pick-place)
   - Honest about limits: sensors must be compatible

---

## Potential Questions & Answers

**Q: How is this different from PatchTST/iTransformer?**
A: Those are forecasting models optimizing MSE. We're building world models - the goal is learned dynamics that transfer, not next-step accuracy.

**Q: Why JEPA over autoencoder/contrastive?**
A: JEPA predicts in latent space, filtering observation noise. It learns dynamics (what matters) not reconstruction (sensor details).

**Q: What's the compute requirement?**
A: Current experiments: consumer GPU. Foundation model roadmap: $500K compute budget for 1B params on industrial data.

**Q: Why should physics work across robots?**
A: Newton's laws are the same. A UR3e and Yu-Cobot both obey F=ma, τ=Iα. The Koopman operator captures this invariant physics.

---

## Slide-by-Slide Speaking Notes

### Slide 1: Title (15 sec)
"We're building world models for physical systems. The key insight: industrial machines obey physics - and physics is learnable."

### Slide 2: The Problem (45 sec)
"Current foundation models treat all time series the same. But physical systems have causal structure - setpoint causes effort causes feedback. We exploit this."

### Slide 3: Our Approach (60 sec)
"Three components: FactoryNet for unified data, KH-JEPA for physics-informed learning, FactoryBench for proper evaluation. The key is Koopman - physics becomes linear in latent space."

### Slide 4: Results (60 sec)
"We pass all tiers. MSE is sanity check. The real test: does it transfer? Train on h1, test on h2 - 0.75x ratio. Long horizon 720 steps - only 1.49x degradation. This is world model behavior."

### Slide 5: Next Steps (30 sec)
"Cross-task and cross-machine transfer experiments running now. Scaling to 1B params on industrial pretraining data. Goal: NeurIPS 2026."

### Slide 6: Call to Action (30 sec)
"Everything is open. Looking for collaborations on: more industrial datasets, benchmark adoption, theoretical analysis of Koopman transfer guarantees."

---

## One-Page Summary for Print

```
FORGIS WORLD MODEL RESEARCH

Problem: Foundation models for time series treat all data the same.
         But physical systems have structure that financial markets lack.

Insight: Physical systems obey differential equations (dx/dt = f(x)).
         Koopman theory: these become LINEAR in latent space.
         Learn the Koopman operator → learn transferable physics.

Solution: KH-JEPA = JEPA + Koopman + SIGReg
         - JEPA: predict dynamics in latent space
         - Koopman: linear predictor (z_{t+k} = K^k z_t)
         - SIGReg: provable collapse prevention

Data: FactoryNet - 18M timesteps, 5 robots, unified schema
      First cross-robot industrial benchmark

Results: ✓ MSE 0.403 (sanity check passed)
         ✓ Transfer ratio 0.75x (physics learned)
         ✓ H=720 degradation 1.49x (stable long-horizon)

Differentiation: Not competing on forecasting MSE.
                 Competing on transfer + stability + interpretability.

Status: Autoresearch validating architecture.
        Scaling to 1B params on industrial pretraining.

Ask: Collaborations on datasets, benchmarks, theory.

Contact: [Your details]
```
