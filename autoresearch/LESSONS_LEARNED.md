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

## C-MAPSS Experiments (March 2026)

### Physics-Informed Channel Grouping Enables Transfer — CONFIRMED

**What we tried**: Group C-MAPSS sensors by turbofan component (fan, HPC, combustor, turbine, nozzle), process within-component with shared weights, cross-component with attention.

**Result**: Role-Transformer transfers 36% better than CI-Transformer (ratio 4.00 vs 6.23) with 9x lower variance, on FD001→FD002 (1→6 operating conditions).

**Why it works**: Weight sharing within component groups forces the encoder to learn universal sensor dynamics. The transfer mechanism is *compositional* — within-component features remain transferable even though they encode condition information. It's NOT about condition-invariance (t-SNE analysis shows Role-Trans representations cluster MORE by condition, not less).

**Key ablation**: Weight sharing is the critical ingredient (ratio 4.36), not just grouping. Separate encoders per component (4.98) or per sensor (4.69) help less. The shared encoder acts as both inductive bias and regularizer.

**Critical detail**: This result only appears WITHOUT RevIN. RevIN normalizes away the operating-condition information and makes all groupings look equivalent.

### RevIN Hurts on C-MAPSS Transfer

**What we tried**: RevIN (instance normalization) to handle distribution shift between FD001 (1 condition) and FD002 (6 conditions).

**Result**: RevIN makes both in-domain (+21%) and transfer (+34%) worse.

**Why**: The operating-condition variation in C-MAPSS is structured (6 discrete regimes, not random shift). RevIN treats it as random noise to remove, destroying useful conditioning information. Global normalization (fit on train set) is sufficient.

**Lesson**: RevIN helps for distribution shift between datasets, NOT for structured multi-condition data. Don't blindly apply it.

### Grouping Structure Matters More Than Specific Groups

**Result**: Physics grouping gives the best transfer (ratio 3.69), but uniform grouping also helps (3.78-3.86). Random is worse (4.99), and no grouping is catastrophic (7.79-8.96).

**Implication**: The two-level hierarchy (within-component + cross-component) is the main architectural insight. Physics-informed groups add on top, but the structure is the bigger win.

### Role-Trans Zero-Shot ≈ CI-Trans 5% Fine-Tuned

**Finding**: Role-Trans zero-shot on FD002 (45.96) ≈ CI-Trans with 5% FD002 fine-tuning (43.59).

**Implication**: Physics-informed grouping is worth approximately 5% of target-domain labels. This is the practical value proposition.

### Role-Trans Helps Cross-Condition, Not Cross-Fault

**Finding**: Role-Trans is 19-36% better on cross-condition transfer (FD001→FD002, FD001→FD004) but CI-Trans wins on same-condition cross-fault transfer (FD001→FD003, FD002→FD004).

**Why**: Within-component physics is invariant across operating conditions (role-based grouping captures this). But fault-specific cross-channel patterns are memorized by Role-Trans and don't generalize to new fault modes. CI-Trans avoids this overfitting by treating channels independently.

**Implication**: The right architecture depends on the transfer challenge. Role-based for condition shift, channel-independent for fault shift. A NeurIPS paper should present this nuance honestly, not claim blanket superiority.

---

### Per-Condition Normalization Eliminates Architecture Advantage

**Finding**: With KMeans-based operating condition clustering and per-cluster normalization, CI-Trans (31.95) matches Role-Trans (32.63) on FD001→FD002.

**Implication**: Role-Trans' advantage comes from *implicit* condition invariance — within-component features don't change much across conditions. But if you have condition labels (or can cluster), explicit normalization works just as well. This is the honest framing for a paper.

### Shorter Sequences Transfer Better

**Finding**: seq_len=15 gives best transfer ratio (2.85) despite worse in-domain (16.90). seq_len=80 gives ratio 6.82.

**Why**: Longer sequences capture more source-specific patterns (degradation trajectories, condition-specific dynamics) that don't generalize. Shorter windows force the model to learn local physics.

---

### JEPA Pretraining Hurts Transfer — CONFIRMED (C-MAPSS)

**What we tried**: Pretrain Role-Transformer encoder with component-level JEPA on FD001+FD002 unlabeled data (63,950 windows, 30 epochs), then fine-tune for RUL on FD001 only.

**Result**: Transfer ratio 5.46 (JEPA+FT) vs 4.10 (scratch) — 33% WORSE with pretraining.

**Why**: JEPA learns to reconstruct component representations, including condition-specific correlations from FD002. When fine-tuned on FD001 (single condition), the encoder already has multi-condition representations baked in that confuse the RUL head. The JEPA objective optimizes for reconstruction, not invariance.

**Broader lesson**: For transfer learning, the architectural inductive bias (role-based grouping) is more valuable than representation pretraining. JEPA might help if the pretraining objective were explicitly condition-invariant (e.g., contrastive across conditions), but standard JEPA reconstruction does not provide this.

**Status**: JEPA has now failed in 3 configurations:
1. Joint objective on ETTh1 → no benefit
2. JEPA-only on ETTh1 → garbage predictions
3. Pretraining on C-MAPSS → hurts transfer

### MMD Domain Adaptation — Marginal Benefit

**What we tried**: Added MMD loss (λ=0.1) between FD001 and FD002 encoder features during supervised training.

**Result**: Transfer ratio 3.92 vs 4.10 — 4% improvement, lower variance.

**Lesson**: MMD provides modest feature alignment but doesn't solve the fundamental condition shift problem. The main challenge is operating condition distribution shift, not feature distribution mismatch per se.

---

## Approaches to Avoid

| Approach | Issue | Status |
|----------|-------|--------|
| Channel-independent for transfer | Can't learn physics | Confirmed (C-MAPSS) |
| JEPA as joint objective on small supervised data | Adds nothing | Confirmed on ETTh1 |
| JEPA pretraining for transfer | Learns condition-specific features, hurts transfer | Confirmed on C-MAPSS |
| Scaling up broken architectures | Doesn't fix fundamentals | Confirmed |
| Claiming "transfer" from per-channel prediction | Misleading | Confirmed |
| RevIN on structured multi-condition data | Normalizes away useful info | Confirmed on C-MAPSS |
| Patch embeddings on short sequences (≤30) | No benefit over point-wise | Confirmed on C-MAPSS |
| Episode normalization for anomaly | Uncertain | Needs investigation |
| Setpoint→effort for cross-machine anomaly | Robot-specific signatures | Confirmed failed |
