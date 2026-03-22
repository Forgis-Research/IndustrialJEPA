# Lessons Learned

Living document. Update as we learn.

---

## ETTh1 Experiments (March 2026)

### JEPA as Joint Training Objective — No Benefit (on small supervised data)

**What we tried:** Combined JEPA loss + supervised forecasting loss on ETTh1.

**Result:** JEPA + Supervised ≈ Supervised only (0.899 vs 0.900 MSE). No benefit.

**Nuance:** This does NOT mean JEPA is useless. It means:
- On **small, fully-labeled data** (8.6K samples), supervised signal is sufficient
- JEPA adds gradient noise without providing useful representations
- JEPA-only (no supervision) produces garbage (MSE ~1.28) because decoder gets no direct signal

**When JEPA should help:**
- **Pretraining** on large unlabeled data, then fine-tuning with supervision
- **Transfer learning** where representations need to generalize
- **Multi-source training** where JEPA learns shared structure

**Status:** Hypothesis, not yet tested. Next: pretrain JEPA on diverse sources, fine-tune on target.

### Channel-Independence Wins on Single-Dataset Forecasting

**Finding:** CI-Transformer (0.450) beats vanilla transformer (0.899) by 50%.

**Why:** With only 7 channels and 8.6K samples, channel-mixing overfits to spurious correlations.

**Implication:** For single-dataset forecasting, channel-independence is correct. But this means we can't learn cross-channel physics — problematic for transfer.

**The paradox:**
- Channel-independent → good forecasting, no transfer capability
- Channel-dependent → overfits on small data, but could transfer physics
- **Resolution needed:** Structured channel-dependence (role-based architecture)

---

## FactoryNet Experiments (March 2026)

### "Cross-machine transfer" Claims Were Misleading

**What we claimed:** Transfer ratio ~1.07 with channel-independent architecture.

**Reality:** Channel-independent = each sensor predicted in isolation. Near-constant signals (positions, setpoints) are trivially predictable. A persistence baseline would match this.

**Lesson:** "Transfer" requires the model to learn something that generalizes. Per-channel autoregression doesn't count.

### Anomaly Detection — Failed

**Result:** Best AUC 0.53 (random chance).

**What we tried:** Setpoint→effort prediction, various normalization strategies.

**Why it failed:** Anomaly signatures are robot-specific. What looks anomalous on Robot A doesn't transfer to Robot B.

### Episode Normalization for Anomaly — Uncertain

**Previous claim:** "Erases the anomaly signal."

**Reconsideration:** An anomaly might still be a relative outlier even after normalization. The issue may be that anomaly *signatures* (which channels spike, in what pattern) don't transfer, not that normalization erases them.

**Status:** Needs more investigation. Don't assume this is settled.

### Cross-Channel Correlations Don't Transfer (genuine finding)

Different robots have different kinematic coupling. Joint 1's effect on Joint 3 depends on the specific robot's geometry. This cross-channel pattern doesn't generalize.

**Implication:** Either:
1. Learn machine-invariant features (within-joint physics only)
2. Learn topology explicitly (GNN with specified structure)
3. Align by semantic role, not by channel index

---

## Genuine Insights Worth Keeping

1. **FactoryNet data loading works** — iloc-based windowing, episode normalization, source aliases all functional

2. **Shared signal space is achievable** — setpoint_pos, setpoint_vel, effort mapping works across robots

3. **RevIN helps with distribution shift** — but insufficient alone for transfer

4. **Architecture > scale** — Fixing channel-independence gave 50% improvement; scaling up a broken architecture would not

5. **Trivial baselines are strong** — Linear often beats complex models on small data

---

## Open Questions

1. **Can role-based architecture preserve physics without overfitting?**
   - Group channels by component, share weights within component
   - Hypothesis: within-component physics transfers, cross-component topology is machine-specific

2. **Is JEPA useful for pretraining multi-source transfer?**
   - Not yet tested properly
   - Need: pretrain on sources A+B, fine-tune on target C

3. **What's the right granularity for "component"?**
   - Per-joint? Per-actuator? Per-subsystem?
   - May depend on task

4. **Does topology information help?**
   - Specifying "joint1 → joint2 → joint3" as a kinematic chain
   - vs learning it from data

---

## Approaches to Avoid

| Approach | Issue | Status |
|----------|-------|--------|
| Channel-independent for transfer | Can't learn physics | Confirmed |
| JEPA as joint objective on small supervised data | Adds nothing | Confirmed on ETTh1 |
| Scaling up broken architectures | Doesn't fix fundamentals | Confirmed |
| Claiming "transfer" from per-channel prediction | Misleading | Confirmed |
| Episode normalization for anomaly | Uncertain | Needs investigation |
| Setpoint→effort for cross-machine anomaly | Robot-specific signatures | Confirmed failed |
