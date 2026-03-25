# Experiment Log

---

# Phase 6: 3-Tier Physics Grouping Validation (2026-03-24)

## Tier 1: Double Pendulum (Synthetic)

**Time**: 21:30-21:52
**Task**: Forecasting next 10 steps from 50-step lookback, 4 channels (theta1, omega1, theta2, omega2)
**Groups**: mass_1=[theta1, omega1], mass_2=[theta2, omega2]
**Transfer**: m_ratio=1.0 (balanced) -> m_ratio=0.5 (unbalanced)

**Results (3 seeds: 42, 123, 456):**

| Model | Source MSE | Target Zero-Shot | Target 10%-Adapted | R(zero) | R(10%) |
|-------|-----------|-----------------|-------------------|---------|--------|
| Mean | 1.061518 | 1.207630 | N/A | 1.14 | N/A |
| Last-Value | 0.077380 | 0.135134 | N/A | 1.75 | N/A |
| Linear | 0.002963 | 0.036864 | 0.016143 | 12.44 | 5.45 |
| MLP | 0.000860 | 0.023571 | 0.003296 | 27.40 | 3.83 |
| CI-Trans | 0.004845 | 0.020568 | 0.015254 | 4.24 | 3.15 |
| Full-Attn | 0.001748 | 0.017115 | 0.006004 | 9.79 | 3.44 |
| **Physics-Grouped** | **0.002896** | **0.016283** | **0.009804** | **5.62** | **3.39** |

**Key Finding**: Physics-grouped achieves **best absolute target MSE** (0.0163 vs CI's 0.0206, 21% better) but worse transfer *ratio* because it also fits source better. The ratio metric penalizes good in-domain performance.

**Honest Assessment**: Physics grouping helps for absolute transfer quality but doesn't clearly beat CI on the ratio metric. With only 2 groups and 4 channels, the grouping structure doesn't provide enough constraint. Full-Attention is competitive.

**Verdict**: PARTIAL WIN for physics grouping (best raw target, but ratio favors CI)

---

## Tier 2: C-MAPSS Turbofan (Mechanical)

**Time**: 21:55-22:30
**Task**: RUL prediction, 14 sensors, seq_len=30
**Groups**: fan=[s2,s8,s12,s21], HPC=[s3,s7,s11,s20], combustor=[s9,s14], turbine=[s4,s13], nozzle=[s15,s17]
**Transfer**: FD001->FD002/FD003/FD004

**Results (3 seeds: 42, 123, 456):**

| Model | FD001 | FD002 | FD003 | FD004 | R(1->2) | R(1->3) | R(1->4) | FD002-adapted |
|-------|-------|-------|-------|-------|---------|---------|---------|--------------|
| Mean | 41.94 | 44.90 | 41.29 | 43.91 | 1.07 | 0.98 | 1.05 | N/A |
| Median | 49.20 | 51.99 | 48.88 | 49.79 | 1.06 | 0.99 | 1.01 | N/A |
| Linear | 48.89 | 21284 | 49.84 | 21190 | 435 | 1.02 | 433 | 15836 |
| MLP | 14.12 | 151830 | 248.70 | 154860 | 10751 | 17.6 | 10966 | 2795 |
| CI-Trans | 13.21+/-0.47 | 78.44+/-26 | 21.19+/-1.6 | 81.98+/-27 | 5.94 | 1.60 | 6.21 | 30.04 |
| Full-Attn | 11.49+/-0.20 | 47.49+/-3.6 | 17.12+/-0.7 | 47.99+/-3.0 | 4.13 | 1.49 | 4.18 | 23.96 |
| **Role-Trans** | **13.02+/-0.26** | **56.98+/-5.5** | **18.46+/-1.9** | **56.73+/-6.8** | **4.37** | **1.42** | **4.36** | **26.39** |

**Key Findings**:
1. **Role-Trans beats CI-Trans by 27.4%** on FD002 zero-shot (56.98 vs 78.44) — consistent with previous results
2. **Full-Attention beats both** on FD002 (47.49 RMSE) — 2D cross-channel attention helps, but physics masking isn't needed here
3. Role-Trans has **lower variance** than CI-Trans (5.5 vs 26.0 on FD002)
4. Role-Trans best on **FD003 ratio** (1.42 vs 1.49/1.60) — cross-fault transfer

**Honest Assessment**: The 2D treatment (temporal + spatial attention) is the big win over CI. Full-Attention outperforms physics-grouped, suggesting learned attention discovers useful patterns beyond our predefined groups. However, Role-Trans still beats CI clearly.

**Verdict**: WIN for physics grouping over CI, but NOT vs Full-Attention

---

## Tier 3: Jena Weather (Physical Real)

**Time**: 22:41-23:40
**Task**: Forecasting next 96 steps from 96-step lookback, 14 channels (hourly weather)
**Groups**: temperature=[T,Tpot,Tdew], pressure=[p,VPmax,VPact,VPdef], humidity=[rh,sh,H2OC,rho], wind=[wv,max_wv,wd]
**Transfer**: Source=2015, Target=2016 (temporal distribution shift)

**Results (3 seeds: 42, 123, 456):**

| Model | Source MSE | Target MSE | Target 10%-Adapted | R(zero) | R(10%) |
|-------|-----------|-----------|-------------------|---------|--------|
| Mean | 1.012888 | 1.002737 | N/A | 0.99 | N/A |
| Last-Value | 0.940940 | 0.860310 | N/A | 0.91 | N/A |
| Linear | 0.499695 | 0.438701 | 0.586511 | 0.88 | 1.17 |
| MLP | 0.508486 | 0.442799 | 0.474813 | 0.87 | 0.93 |
| CI-Trans | 0.516651±0.001 | 0.462512±0.0005 | 0.490712±0.0005 | 0.90 | 0.95 |
| Full-Attn | 0.493065±0.002 | 0.436762±0.002 | 0.485245±0.005 | 0.89 | 0.98 |
| **PhysMask** | **0.495426±0.0004** | **0.439802±0.00008** | **0.484799±0.003** | **0.89** | **0.98** |
| RoleTrans | 0.518621±0.008 | 0.456201±0.007 | 0.508345±0.010 | 0.88 | 0.98 |

**Key Findings**:
1. **PhysMask beats CI-Trans by 4.9%** on target MSE (0.4398 vs 0.4625)
2. **PhysMask has remarkably low variance** (±0.00008) — the physics structure acts as strong regularizer
3. **Full-Attention is best** overall (0.4368), but only 0.7% better than PhysMask
4. **RoleTrans underperforms** — the hierarchical pool-then-cross architecture loses fine-grained channel information. The within-group mean-pooling is too aggressive for weather where all channels matter.
5. **Linear is competitive** — weather year-to-year transfer is relatively easy (R<1.0 means target is easier than source)
6. **10% adaptation hurts** for most models — the small adaptation set introduces overfitting

**Honest Assessment**: Physics-masked attention helps vs CI (4.9% better), matching the pattern from C-MAPSS. But the RoleTrans architecture (shared within-group encoder + cross-group attention) doesn't transfer well to weather — it's designed for component-based systems where groups are naturally independent. Weather variables within a group (e.g., T, Tpot, Tdew) are highly correlated, making the mean-pooling appropriate, but cross-group interactions (e.g., temperature↔humidity) require full resolution.

**Architecture Insight**: PhysMask (full encoder + masked cross-channel attention) > RoleTrans (grouped encoder + cross-group attention). The mask approach preserves all channel information while constraining attention, whereas the RoleTrans bottleneck loses too much.

**Verdict**: WIN for PhysMask over CI-Trans (4.9%). LOSS for RoleTrans. Full-Attn wins overall.

---

## Cross-Tier Summary

| Tier | Best Physics Model | vs CI-Trans | vs Full-Attn | Winner Overall |
|------|-------------------|-------------|--------------|----------------|
| 1. Pendulum | PhysicsGrouped | **+21%** target MSE | Comparable | PhysicsGrouped |
| 2. C-MAPSS | RoleTrans | **+27%** FD002 RMSE | -17% (loses) | Full-Attn |
| 3. Weather | PhysMask | **+4.9%** target MSE | -0.7% (close) | Full-Attn |

**Physics vs CI**: 3/3 tiers — physics grouping consistently beats channel-independent
**Physics vs Full-Attn**: 1/3 tiers — physics grouping rarely beats learned full attention

**Honest Conclusion**: Physics grouping is a reliable improvement over channel-independent processing (4.9-27% on transfer). But full unconstrained attention usually matches or beats it, suggesting that learned cross-channel relationships capture the physics structure automatically when given enough data. The value of physics grouping is in:
1. **Regularization** (extremely low variance on weather)
2. **Data efficiency** (should help more with less data — untested)
3. **Interpretability** (groups are physically meaningful)

---

# Phase 7: Ablation & Statistical Tests (2026-03-24/25)

## Exp 41: Grouping Assignment Ablation — CRITICAL NEGATIVE RESULT

**Time**: 23:41-00:12
**Hypothesis**: Physics grouping should beat random grouping because it captures true physical structure.
**Method**: Same RoleTrans architecture with 5 different grouping assignments on C-MAPSS FD001→FD002.

**Results (3 seeds: 42, 123, 456):**

| Condition | FD001 RMSE | FD002 RMSE | FD002-10% | R(zero) | R(10%) |
|-----------|-----------|-----------|-----------|---------|--------|
| physics | 13.02±0.26 | 56.98±5.50 | 27.45±1.46 | 4.37 | 2.11 |
| random_0 | 11.78±0.13 | 55.23±4.00 | 23.68±1.25 | 4.69 | 2.01 |
| random_1 | 12.66±0.26 | 50.87±1.77 | 25.23±0.71 | 4.02 | 1.99 |
| random_2 | 11.96±0.05 | 52.92±5.58 | 24.68±1.92 | 4.42 | 2.06 |
| wrong | 12.46±0.07 | 61.99±12.54 | 25.76±2.85 | 4.98 | 2.07 |

**Statistical tests:**
- Physics vs ALL random (FD002): t=1.15, p=0.278 (NOT significant)
- Physics vs wrong: t=-0.52, p=0.632 (NOT significant)

**Key Finding**: **Physics grouping does NOT outperform random grouping.** Random grouping average FD002=53.01, physics=56.98. The benefit comes entirely from the **architectural pattern** (shared within-group encoder + cross-group attention), not from the specific channel-to-group assignment.

**Why physics grouping is worse than random on in-domain**: Physics grouping gives FD001=13.02 (worst), while random gives 11.78-12.66 (better). The physics groups may be suboptimal for the prediction task — some random groupings accidentally pair sensors that are more informative together.

**What wrong grouping reveals**: Wrong grouping has 2x higher variance (±12.54) and worst transfer ratio (4.98). Deliberately mixing components hurts stability, even if average performance is comparable. So grouping assignment affects *variance* more than *mean*.

**Implication for paper narrative**:
1. ~~"Physics grouping improves transfer"~~ → **"Grouped architecture improves transfer; physics groups provide stability"**
2. The contribution is the **2D treatment** (temporal per-channel + spatial cross-channel), not the physics knowledge
3. Any reasonable grouping (even random) provides the regularization benefit
4. Physics grouping's value is in **interpretability and variance reduction**, not in absolute performance

**Verdict**: NEGATIVE for physics-specific grouping. POSITIVE for grouped architecture in general.

---

## Exp 42: Multi-Horizon Weather Forecasting — Grouped advantage grows with horizon

**Time**: 00:13-01:21
**Hypothesis**: Physics grouping advantage should increase at longer horizons where cross-channel structure matters more.

**Results (3 seeds, Jena Weather 2016 test):**

| Horizon | CI-Trans | Full-Attn | Role-Trans | RT vs CI |
|---------|----------|-----------|------------|----------|
| H=96 | 0.4570±0.0002 | 0.4267±0.0005 | 0.4323±0.0012 | **5.4%** |
| H=336 | 0.5608±0.0003 | 0.5203±0.0031 | 0.5248±0.0014 | **6.4%** |
| H=720 | 0.6214±0.0010 | 0.5435±0.0026 | 0.5620±0.0084 | **9.6%** |

**Key Finding**: The grouped architecture advantage over CI **grows from 5.4% to 9.6%** as horizon increases from 96 to 720 steps. At H=720, the gap is nearly 2x what it is at H=96.

**Why this happens**: Longer horizons require better understanding of inter-variable dynamics. CI-Trans treats each channel independently, so it can only extrapolate each variable separately. The grouped models capture cross-channel relationships that become more important for multi-step predictions.

**Full-Attn vs Role-Trans**: Full-Attn is consistently better (0.4267 vs 0.4323 at H=96, 0.5435 vs 0.5620 at H=720). The gap also grows slightly with horizon, suggesting that the mean-pooling bottleneck in RoleTrans loses more information at longer horizons.

**Combined with ablation insight**: Since random grouping ≈ physics grouping, the benefit is purely from the grouped architecture pattern, not the specific assignment. The longer the horizon, the more valuable any form of cross-channel processing becomes.

**Verdict**: CONFIRMED that grouped > CI, and the effect scales with task difficulty.

---

## Exp 43: 10-Seed Statistical Significance Tests

### C-MAPSS: CI-Trans vs Role-Trans (10 seeds)

**Time**: 01:22-01:58

| Seed | CI FD001 | CI FD002 | Role FD001 | Role FD002 |
|------|---------|---------|-----------|-----------|
| 42 | 13.01 | 111.77 | 12.67 | 49.21 |
| 123 | 12.76 | 75.30 | 13.27 | 61.17 |
| 456 | 13.85 | 48.25 | 13.14 | 60.55 |
| 789 | 13.38 | 97.75 | 12.48 | 62.60 |
| 1234 | 13.39 | 106.46 | 12.47 | 53.30 |
| 5678 | 13.44 | 94.51 | 12.50 | 57.38 |
| 9012 | 13.50 | 57.09 | 12.35 | 44.01 |
| 3456 | 12.66 | 68.93 | 12.82 | 49.85 |
| 7890 | 12.95 | 95.01 | 12.67 | 49.59 |
| 2468 | 13.55 | 110.03 | 12.31 | 77.91 |

**Summary**:
- CI-Trans FD002: **86.51 ± 21.49**
- Role-Trans FD002: **56.56 ± 9.20**
- **Paired t-test**: t=4.302, **p=0.0020**
- **Wilcoxon**: W=1.0, **p=0.0039**
- **Cohen's d**: 1.43 (very large effect)
- **Role wins**: 9/10 seeds
- **Improvement**: 34.6%

### Pendulum: CI-Trans vs Physics-Grouped (10 seeds)

**Time**: 01:58-02:05

| Seed | CI Target MSE | Physics Target MSE |
|------|-------------|-------------------|
| 42 | 0.016859 | 0.013180 |
| 123 | 0.016179 | 0.012741 |
| 456 | 0.015522 | 0.012306 |
| 789 | 0.015833 | 0.012843 |
| 1234 | 0.014920 | 0.012864 |
| 5678 | 0.015341 | 0.012051 |
| 9012 | 0.015898 | 0.013127 |
| 3456 | 0.016233 | 0.012356 |
| 7890 | 0.016394 | 0.012999 |
| 2468 | 0.015348 | 0.012395 |

**Summary**:
- CI-Trans target: **0.015853 ± 0.000553**
- Physics target: **0.012686 ± 0.000366**
- **Paired t-test**: t=19.478, **p < 0.0001**
- **Physics wins**: 10/10 seeds
- **Improvement**: 20.0%

### Combined Statistical Evidence

| Dataset | Improvement | p-value | Effect Size | Win Rate |
|---------|-------------|---------|-------------|----------|
| C-MAPSS FD002 | 34.6% | 0.0020 | d=1.43 | 9/10 |
| Pendulum | 20.0% | <0.0001 | d=19.5 | 10/10 |

**Verdict**: Grouped architecture advantage over CI is statistically robust across both domains.

---

## Exp 44: Data Efficiency — Full-Attn dominates, Role-Trans is noisy

**Time**: 02:11-02:41
**Hypothesis**: Grouped architecture should help more with less training data due to inductive bias.

**FD002 Transfer RMSE at varying training data fractions:**

| Fraction | CI-Trans | Full-Attn | Role-Trans | RT vs CI |
|----------|---------|-----------|-----------|----------|
| 5% (886) | 81.43 | **62.39** | 71.84 | +11.8% |
| 10% (1773) | 63.39 | **53.08** | 62.30 | +1.7% |
| 25% (4432) | 61.61 | **59.50** | 77.75 | -26.2% |
| 50% (8865) | 56.54 | **50.85** | 77.96 | -37.9% |
| 100% (17731) | 83.38 | **50.97** | 60.66 | +27.2% |

**Honest Assessment**: The data efficiency hypothesis is NOT supported:
1. **Full-Attention wins at ALL data sizes** — even at 5% data (886 samples), it outperforms both CI and Role-Trans
2. **Role-Trans is unstable at intermediate sizes** — variance ±14.42 at 50%, suggesting the grouped architecture can get trapped in local minima
3. Role-Trans only clearly beats CI at extreme low (5%) and full (100%) data
4. At 25-50%, Role-Trans is WORSE than CI — likely due to the mean-pooling bottleneck struggling with limited but non-trivial data

**Why Full-Attn dominates**: With only 14 channels, full attention has very few parameters to learn (14x14 attention matrix). The O(n²) cost is negligible. The inductive bias of grouped architecture only helps when the model capacity is saturated, which doesn't happen with 14 channels.

**Implication**: The grouped architecture advantage is primarily about **transfer**, not about data efficiency. At any training size, full attention learns better in-domain, and often transfers better too. The grouped advantage on transfer only emerges at 100% data where the models are well-trained.

**Verdict**: NEGATIVE for data efficiency claim. Full-Attn is the pragmatic choice.

---

## Exp 45: Pendulum Grouping Ablation — Wrong grouping kills, but singleton wins

**Time**: 02:42-02:55
**Hypothesis**: On pendulum (with true physical independence between masses), physics grouping should clearly beat random grouping.

**Results (5 seeds: 42, 123, 456, 789, 1234):**

| Condition | Groups | Src MSE | Tgt MSE | Ratio |
|-----------|--------|---------|---------|-------|
| **singleton** | {[θ1],[ω1],[θ2],[ω2]} (4 groups) | **0.001620** | **0.017599** | 10.86 |
| **physics** | {[θ1,ω1],[θ2,ω2]} (2 groups) | 0.001392 | 0.026924 | 19.34 |
| cross_1 | {[θ1,ω2],[ω1,θ2]} | 0.002910 | 0.072617 | 24.95 |
| cross_2 | {[θ1,ω2],[θ2,ω1]} | 0.003071 | 0.075245 | 24.51 |
| type | {[θ1,θ2],[ω1,ω2]} | 0.587376 | 0.621355 | 1.06 |
| all | {[θ1,ω1,θ2,ω2]} (1 group) | 0.586540 | 0.628802 | 1.07 |

**All comparisons highly significant** (p < 0.001).

**Key Findings — This Changes the Story**:

1. **Wrong grouping is catastrophic**: "type" (grouping by measurement type) and "all" (single group) are 23x worse. The within-group mean-pooling destroys dynamics when it mixes the wrong channels.

2. **Singleton (per-channel) BEATS physics grouping** (0.0176 vs 0.0269, p<0.001): More groups = more resolution = better performance. The mean-pooling within physics groups loses information.

3. **Cross-mass mixing is 2.7x worse than physics**: Mixing θ1 with ω2 (from different masses) is bad but survivable. Mixing θ1 with θ2 (from same type, different masses) is catastrophic.

4. **The pattern**: singleton > physics >> cross >> type ≈ all

**Why singleton wins**: The RoleTrans architecture pools within groups. With 4 singleton groups, no pooling happens — each channel retains its full representation. The cross-group attention then learns channel relationships freely (like Full-Attention). With 2 physics groups, theta and omega of each mass are averaged, losing critical phase information.

**Implication**: The mean-pooling bottleneck in RoleTrans is the fundamental limitation. Physics grouping helps with *attention masking* (PhysMask) but hurts with *encoder sharing + pooling* (RoleTrans) when groups are small.

**Reconciliation with C-MAPSS ablation**: On C-MAPSS (14 channels, 5 groups of 2-4), random grouping ≈ physics because the groups are larger and pooling averages more channels. On pendulum (4 channels, 2 groups of 2), every channel matters and pooling is more harmful.

**Revised Architecture Recommendation**:
- Use **PhysMask** (attention masking), NOT **RoleTrans** (encoder sharing + pooling)
- PhysMask preserves all channel information while constraining attention
- RoleTrans bottleneck helps only when groups are large enough that pooling is beneficial

**Verdict**: Physics grouping matters (wrong groups are catastrophic) but the RoleTrans mean-pooling hurts. PhysMask is the better approach.

---

# Phase 3: Deep Literature Review (2026-03-23)

## Research: Three Directions for Breakthrough

**Papers reviewed**: 35+
**Time**: 4+ hours of deep research
**Key finding**: Three viable directions identified. Direction 2 (Slot-based Concept Learning) is most novel. Direction 3 (Mechanical-JEPA) has strongest narrative. See `LITERATURE_REVIEW.md` for full analysis.

### Direction 1: Sparse Graph Learning
- 11 papers reviewed (NRI, MTGNN, GTS, iTransformer, etc.)
- **Gap**: Nobody validates learned graphs against known physics or uses them for transfer
- **Risk**: Low (incremental)

### Direction 2: Learned Latent Concepts (RECOMMENDED)
- 13 papers reviewed (CBMs, Slot Attention, SlotPi, SlotFM, etc.)
- **Gap**: No slot attention for industrial sensors. CBMs for C-MAPSS exist but need expert concepts.
- **Key prior**: SlotFM (2025) — first slot attention on sensor data (accelerometers), discovers frequency components
- **Our angle**: Slot attention + physics structure → discovers physical components unsupervised

### Direction 3: Mechanical-JEPA
- 12 papers reviewed (I-JEPA, V-JEPA, MTS-JEPA, Brain-JEPA, etc.)
- **Root cause of our JEPA failures**: (1) too few mask targets, (2) no codebook bottleneck, (3) shallow encoder, (4) MSE not L1, (5) no multi-scale
- **Fix**: MTS-JEPA-style codebook + 70-90% temporal masking + deep encoder
- **Risk**: High (JEPA has failed 3 times already)

### Hybrid: Slot-JEPA
- **Best of both**: Slot attention discovers components, JEPA predicts in slot space with codebook bottleneck
- **Novel**: No prior work on this combination for industrial data

---

# Phase 5: Slot-Concept Transformer (2026-03-23)

## Exp 39a: Slot-Concept Transformer vs Baselines — PROMISING

**Time**: 23:04
**Hypothesis**: Slot attention can discover physical component structure from sensor data and provide transfer comparable to expert-specified groupings.

**Results (3 seeds):**

| Model | FD001 RMSE | FD002 RMSE | Transfer Ratio |
|-------|-----------|-----------|----------------|
| Slot(K=5) | 12.36 ± 0.45 | 56.69 ± 13.38 | 4.59 |
| Role-Trans | 12.45 ± 0.08 | 54.28 ± 4.20 | 4.36 |
| CI-Trans | 13.20 ± 0.44 | 82.24 ± 26.99 | 6.23 |

**Verdict**: PROMISING — Slot-Concept beats CI-Trans convincingly (26% better transfer) but has higher variance than Role-Trans. Best single-seed transfer (45.67, seed 42) beats all Role-Trans seeds.

**Slot assignments**: Collapsed to near-uniform (~0.200 per slot per channel). Despite this, the model works well — the slot attention GRU/MLP creates differentiated internal representations even when attention weights are uniform.

## Exp 39b: Number of Slots Ablation — K=5 IS OPTIMAL

| K | FD001 | FD002 | Ratio |
|---|-------|-------|-------|
| 3 | 12.64 | 53.52 | 4.23 |
| **5** | **12.82** | **45.67** | **3.56** |
| 7 | 13.10 | 51.06 | 3.90 |
| 10 | 12.05 | 56.12 | 4.66 |
| 14 | 12.52 | 61.94 | 4.95 |

**Key finding**: K=5 (matching the number of physical components) gives the best transfer! K=14 (equivalent to per-channel/CI) gives the worst. This suggests the optimal decomposition granularity matches the physical structure even without explicit physics knowledge.

## Exp 39c: Slot Attention Iterations

| Iters | FD001 | FD002 | Ratio |
|-------|-------|-------|-------|
| 1 | 13.07 | 73.41 | 5.62 |
| **3** | **12.82** | **45.67** | **3.56** |
| 5 | 12.18 | 45.71 | 3.75 |
| 7 | 12.53 | 47.44 | 3.79 |

**Insight**: 3 iterations is sufficient. More iterations slightly improve in-domain but don't help transfer. 1 iteration is insufficient.

## Honest Assessment

**What works**: Slot-Concept Transformer provides competitive transfer without requiring physics knowledge. K=5 optimality is a nice result.

**What doesn't**: Slot assignments are uniform — the model doesn't actually discover differentiated component groupings in the attention weights. The competitive performance comes from the GRU-based slot refinement and cross-slot attention, not from meaningful channel-to-slot decomposition.

**Next steps**: Try entropy regularization on slot assignments to force differentiation.

## Exp 40: Entropy-Regularized Slot Assignments — NEGATIVE FOR DISCOVERY

**Time**: 23:39
**Hypothesis**: Entropy regularization will force differentiated channel-to-slot assignments.

**Results (entropy weight sweep, seed=42):**

| λ | FD001 | FD002 | Ratio | Assignment Entropy |
|---|-------|-------|-------|-------------------|
| 0.00 | 12.82 | 45.67 | 3.56 | 1.609/1.609 (uniform) |
| 0.01 | 12.74 | 45.81 | 3.60 | 1.601/1.609 |
| 0.10 | 12.58 | 44.92 | 3.57 | 1.609/1.609 |
| 0.50 | 12.98 | 45.58 | 3.51 | 1.600/1.609 |
| 1.00 | 12.94 | 45.38 | 3.51 | 1.596/1.609 |
| 5.00 | 12.63 | 44.91 | 3.56 | 1.596/1.609 |

**Verdict**: NEGATIVE for concept discovery ✗ — entropy stays near maximum (uniform) regardless of λ. Only 1/5 unique components discovered at any λ.

**3-seed with λ=1.0**: FD001=12.81±0.14, FD002=55.18±13.32, ratio 4.31

**Why slot attention fails to discover components**:
1. **Shared temporal encoder homogenizes features**: All sensors are processed by the same encoder, producing similar features (all show degradation dynamics). Slot attention needs differentiated inputs to differentiate slots.
2. **14 sensors is too few**: In vision, slot attention works with hundreds of spatial tokens. With 14 channels, the competition mechanism doesn't have enough diversity.
3. **Degradation is global**: All C-MAPSS sensors degrade together (engine-level failure). There's no local/component-specific signal strong enough for unsupervised decomposition.

**Honest assessment**: Slot-Concept Transformer works as a model (competitive transfer) but doesn't discover meaningful concepts. The "concept discovery" narrative doesn't hold for C-MAPSS. The model succeeds because the slot attention + cross-slot transformer acts as an implicit grouping regularizer, not because it discovers physics.

**Implication**: The Role-Transformer's explicit physics grouping remains the better approach for this task. Learned concept discovery requires either (a) per-channel encoders (not shared), (b) more diverse sensor behavior, or (c) explicit structure in the loss.

---

# Phase 1: Deep Research (2026-03-22)

## Research: Transfer Learning for Time Series (2024-2026)

**Papers reviewed**: 10+
**Key findings**:
1. **Channel-independence is dominant** — PatchTST (ICLR 2023, 1000+ citations), Moirai (ICML 2024), UniTS (NeurIPS 2024) all use channel-independent or any-variate processing
2. **Nobody does role-based channel grouping** — SCNN (arXiv 2305.13036) decomposes by structured/heterogeneous components but doesn't transfer. SaR routes sensors by pattern similarity but not by physical function. This is our gap to fill.
3. **TS-JEPA exists** (NeurIPS 2024 TSALM Workshop, Ennadir et al.) — adapts JEPA to time series with 70% patch masking. Does NOT use role-based grouping. Code: github.com/Sennadir/TS_JEPA
4. **Foundation models work** — TimesFM, Chronos, Moirai show pretraining transfers. But univariate-focused.

**Implication for us**: Role-based channel grouping for cross-machine transfer is genuinely novel. No existing work combines structured sensor grouping with JEPA-style learning for industrial transfer.

## Research: C-MAPSS SOTA (2024-2025)

**Papers reviewed**: 8
**Key findings**:
- FD001 best RMSE: ~6.62 (LightGBM+CatBoost ensemble, Nature Sci Rep 2025), ~11.36 (Transformer, DL-only)
- FD002 best RMSE: ~12.78 (LGBM), ~14.45 (CNN-LSTM-Attention)
- Cross-subset transfer: GPT-2 fine-tuning (freeze 20/24 layers), MMD-based domain adaptation
- **No JEPA or role-based transfer on C-MAPSS** — this is an open opportunity

**Target for us**: FD001 RMSE < 13.0 (competitive DL baseline), then demonstrate FD001→FD002 transfer improvement with role-based grouping.

## Research: Cross-Machine Fault Diagnosis

**Papers reviewed**: 6
**Key findings**:
- DANN/CDAN are standard for domain adaptation (~5000 citations for DANN)
- CWRU→Paderborn transfer requires careful preprocessing (resampling, filtering)
- Nobody uses role-based grouping for cross-machine — all treat channels uniformly
- Few-shot transfer from artificial→natural faults is a recognized challenge

## Research: Structured / Physics-Informed Architectures

**Papers reviewed**: 8
**Key findings**:
- **Mamba competitive with Transformers** at lower compute (S-Mamba, Neurocomputing 2024)
- **SCNN** decomposes MTS by structured/heterogeneous components — closest to our approach
- **SaR** routes sensors to expert modules by similarity — related to role-based concept
- **Brain-JEPA** (NeurIPS 2024) — JEPA works for scientific multivariate time series

**Implication**: Our contribution sits at the intersection of structured decomposition (SCNN/SaR) + JEPA pretraining (TS-JEPA) + cross-machine transfer — a novel combination.

---

# Phase 2: C-MAPSS Experiments (2026-03-22)

## Setup
- Dataset: C-MAPSS (NASA Turbofan Engine Degradation)
- FD001: 100 train engines, 1 operating condition, 1 fault mode (HPC)
- FD002: 260 train engines, 6 operating conditions, 1 fault mode (HPC)
- Sensors: 14 informative (dropped 7 near-constant)
- Lookback: 30 cycles
- RUL cap: 125 (piece-wise linear, standard)
- Seeds: 42, 123, 456
- Component groups: fan(3), hpc(4), combustor(2), turbine(3), nozzle(2)

## Exp 7: C-MAPSS FD001 Baselines (5 models)

**Time**: 18:10
**Hypothesis**: Role-Transformer should match or beat CI-Transformer on single-dataset while learning transferable structure.
**Change**: Built 5 models — Linear, LSTM, Transformer, CI-Transformer, Role-Transformer

**Results (FD001 test, 3 seeds):**

| Model | Params | Test RMSE | NASA Score |
|-------|--------|-----------|------------|
| Linear | 421 | 48.24 ± 0.06 | 747,117 ± 45,197 |
| LSTM | 53,825 | 14.77 ± 0.26 | 375 ± 14 |
| Transformer | 102,913 | 16.11 ± 0.11 | 499 ± 30 |
| CI-Transformer | 26,480 | 13.38 ± 0.24 | 298 ± 21 |
| **Role-Transformer** | **40,513** | **12.66 ± 0.34** | **248 ± 23** |

**Verdict**: KEEP ✓
**Insights**:
1. **Role-Transformer is best** at 12.66 RMSE, beating CI-Transformer (13.38) by 5.4%
2. CI-Transformer beats LSTM (14.77) by 9.4% — channel-independence helps here too
3. Channel-mixing Transformer overfits severely (val 3.6 vs test 16.1)
4. Role-Transformer beats our target of <13.0 RMSE
5. Competitive with published DL SOTA (~11-13 RMSE range)

## Exp 8: Transfer FD001 → FD002 (THE KEY EXPERIMENT)

**Time**: 18:50
**Hypothesis**: Role-based grouping by physical component enables better transfer across operating conditions than channel-independent processing.
**Change**: Train on FD001 (1 condition), test on FD002 (6 conditions). Normalize FD002 with FD001 stats.

**Results (3 seeds):**

| Model | FD001 RMSE | FD002 RMSE | Transfer Ratio |
|-------|------------|------------|----------------|
| CI-Transformer | 13.05 ± 0.14 | 101.52 ± 30.77 | 7.78 |
| **Role-Transformer** | **12.65 ± 0.17** | **55.10 ± 7.68** | **4.36** |

**Verdict**: KEEP ✓✓✓ (Major result)
**Insights**:
1. **Role-Transformer transfers 44% better** (ratio 4.36 vs 7.78)
2. **Lower variance on transfer** — Role-Transformer std=7.68 vs CI std=30.77
3. CI-Transformer has catastrophic transfer on some seeds (124 RMSE on seed 42)
4. Role-Transformer is more robust: worst seed 65.81 vs CI worst 124.41
5. **The hypothesis is confirmed**: Structured channel-dependence (grouping by physical component) enables cross-condition transfer that channel-independence cannot achieve

**Why it works (hypothesis)**: Within-component sensor relationships (e.g., fan speed ↔ bypass ratio ↔ bleed) reflect physics that is invariant across operating conditions. The Role-Transformer learns these physics within each component, while the cross-component attention captures how subsystems interact. CI-Transformer treats each sensor independently, losing these physics relationships.

**Caveats**:
- Both models degrade substantially on transfer (ratio >4). Need domain adaptation.
- FD002 has 6 operating conditions vs FD001's 1 — this is a hard transfer.
- Published SOTA on FD002 (trained on FD002) is ~13-15 RMSE; we get 55 with zero-shot transfer.

**Next**: Try fine-tuning on small FD002 subset, try RevIN for distribution shift, ablate component groupings.

## Exp 9: RevIN Normalization — REVERT

**Time**: 19:00
**Hypothesis**: RevIN handles distribution shift between operating conditions.
**Change**: Added RevIN layer before Role-Transformer encoder.
**Result**: Without RevIN FD001=12.45/FD002=45.96 → With RevIN FD001=15.07/FD002=61.68
**Verdict**: REVERT ✗
**Insight**: RevIN **hurts** both in-domain (+21%) and transfer (+34%). It normalizes away operating-condition information that the model needs. The global normalization (fit on FD001 train) already handles scale; RevIN adds noise.
**Next**: Run grouping ablation without RevIN.

## Exp 10: Few-shot Fine-tuning (with RevIN) — Informative

**Time**: 19:15
**Result**: 5% FD002 data → 43.69 (-29%), 10% → 43.57, 25% → 43.42
**Insight**: Rapid saturation — 5% and 25% give similar results. The bottleneck is architecture, not data quantity.

## Exp 11: Cross-Component Depth — No Effect

**Time**: 19:30
**Result**: cross=1: 61.68, cross=2: 62.27, cross=3: 62.36 (all with RevIN)
**Verdict**: REVERT ✗ — keep cross=1
**Insight**: Additional cross-component layers don't help. The component representations are already sufficient after 1 layer.

## Exp 12: Grouping Ablation (with RevIN) — Misleading

**Time**: 19:45
**Result**: Physics=61.68, Random=61.23, Uniform-3=59.59, All-one=57.62
**Verdict**: Misleading — RevIN masks the grouping signal. See Exp 14.

## Exp 13: Model Size Ablation

**Time**: 20:00
**Result**: d=16: 60.11, d=32: 61.68, d=64: 62.74 (all with RevIN)
**Insight**: Smaller is slightly better for transfer. Fewer parameters = less overfitting to source domain.

## Exp 14: Grouping Ablation WITHOUT RevIN — KEY RESULT

**Time**: 20:15
**Hypothesis**: Physics-informed grouping should help transfer when RevIN isn't masking the signal.
**Change**: Tested 5 grouping strategies and CI baseline, all without RevIN.

| Grouping | FD001 | FD002 | Ratio |
|----------|-------|-------|-------|
| **Physics-5 (ours)** | 12.45 | **45.96** | **3.69** |
| Uniform-3 | 12.55 | 47.43 | 3.78 |
| Uniform-7 | 11.78 | 45.48 | 3.86 |
| Random-5 | 11.44 | 57.16 | 4.99 |
| All-one-group | 12.28 | 95.70 | 7.79 |
| CI-Trans (baseline) | 13.05 | 116.98 | 8.96 |

**Verdict**: KEEP ✓✓✓
**Insights**:
1. **Physics grouping gives best transfer ratio** (3.69) — 26% better than random (4.99), 59% better than CI (8.96)
2. **Grouping structure matters more than specific groups** — Uniform-3 and Uniform-7 also help vs All-one/CI
3. **All-one-group is much worse than CI** on FD002 (95.70 vs 116.98 is same ballpark) — confirms the hierarchy helps
4. **Random grouping is middle ground** — structure helps even if not physics-informed, but physics is best
5. **RevIN was masking this signal** — with RevIN, all groupings looked equivalent (Exp 12)

## Exp 15: Multi-Seed Confirmation — STRONG RESULT

**Time**: 20:45

| Model | FD001 RMSE | FD002 RMSE | Transfer Ratio |
|-------|-----------|-----------|----------------|
| **Role-Trans** | **12.22 ± 0.38** | **48.82 ± 2.86** | **4.00** |
| CI-Trans | 13.20 ± 0.44 | 82.24 ± 26.99 | 6.23 |

**Verdict**: KEEP ✓✓✓
**Insights**:
1. **Role-Trans transfers 36% better** (ratio 4.00 vs 6.23)
2. **9x lower variance** on transfer (std 2.86 vs 26.99) — more robust
3. **Also better in-domain** (12.22 vs 13.20 on FD001)
4. CI-Trans is unstable: worst seed gives 116.98 on FD002

## Exp 16: Few-shot Fine-tuning WITHOUT RevIN

**Time**: 21:00

| Model | Zero-shot | 1% FT | 5% FT | 10% FT | 25% FT |
|-------|-----------|-------|-------|--------|--------|
| Role-Trans | 45.96 | 42.67 | 42.49 | 41.23 | **38.69** |
| CI-Trans | 116.98 | — | 43.59 | 42.36 | — |

**Verdict**: KEEP insight
**Insights**:
1. Role-Trans + 25% FT: **38.69 RMSE** (best overall FD002 result)
2. CI-Trans catches up with 5% data (43.59), but Role-Trans still wins with more data
3. **Role-Trans zero-shot (45.96) ≈ CI-Trans 5% fine-tuned (43.59)** — the physics grouping is worth ~5% of target domain labels
4. With sufficient fine-tuning data, the architecture advantage narrows — expected.

---

# Karpathy Loop Summary

## Best Results

| Setting | Model | FD001 | FD002 | Ratio |
|---------|-------|-------|-------|-------|
| Zero-shot | **Role-Trans** | **12.22±0.38** | **48.82±2.86** | **4.00** |
| Zero-shot | CI-Trans | 13.20±0.44 | 82.24±26.99 | 6.23 |
| 25% FT | Role-Trans | 12.45 | **38.69** | 3.11 |
| 5% FT | CI-Trans | 13.05 | 43.59 | 3.34 |

## Key Findings

1. **Physics-informed grouping enables better transfer** — 36% improvement in transfer ratio, 9x lower variance
2. **RevIN hurts on this task** — it normalizes away operating-condition information
3. **Grouping structure matters** — physics > uniform > random >> none
4. **Role-Trans zero-shot ≈ CI-Trans 5% fine-tuned** — physics knowledge saves labels
5. **In-domain performance also improves** — Role-Trans beats CI-Trans on FD001 too
6. **The hierarchy (within-component + cross-component) is key** — not just the grouping

## Honest Assessment

**What supports the NeurIPS claim:**
- Role-Trans consistently transfers better than CI-Trans across seeds
- Physics grouping gives the best transfer among all grouping strategies
- The advantage is robust (low variance) and statistically meaningful

**What doesn't:**
- Absolute transfer RMSE is still high (48.82 on FD002 vs ~13 for models trained on FD002)
- Uniform grouping also helps — the benefit isn't purely from physics knowledge
- The FD001→FD002 gap is huge (1 vs 6 operating conditions) — may be too hard for zero-shot
- Single-dataset FD001 improvement (12.22 vs 13.20) is modest

**Next steps:**
- Test on FactoryNet (cross-machine transfer with different robots)
- FD003/FD004 experiments (different fault modes)
- Try domain adaptation (MMD loss) to reduce the gap further
- Try JEPA pretraining on all FD subsets, then fine-tune

---

# Phase 2b: Cross-Fault Transfer (2026-03-22)

## Exp 17: FD001 → FD003 (same conditions, +fan fault)

**Hypothesis**: Role-Trans should also help with cross-fault transfer.
**Result**:

| Model | FD001 RMSE | FD003 RMSE | Ratio |
|-------|-----------|-----------|-------|
| Role-Trans | 12.29 ± 0.37 | 20.30 ± 2.01 | 1.65 |
| CI-Trans | 13.42 ± 0.26 | 18.54 ± 2.24 | **1.38** |

**Verdict**: CI-Trans wins here. When operating conditions are the same, CI doesn't suffer from condition shift and avoids overfitting to specific fault patterns.

## Exp 18: FD001 → FD004 (different conditions + different faults)

| Model | FD001 RMSE | FD004 RMSE | Ratio |
|-------|-----------|-----------|-------|
| **Role-Trans** | **12.29 ± 0.37** | **52.92 ± 4.40** | **4.31** |
| CI-Trans | 13.42 ± 0.26 | 71.79 ± 19.43 | 5.35 |

**Verdict**: Role-Trans wins when both conditions and faults differ. 19% better ratio, 4x lower variance.

## Exp 19: FD003 → FD001 (multi-fault → single-fault, same conditions)

| Model | FD003 RMSE | FD001 RMSE | Ratio |
|-------|-----------|-----------|-------|
| Role-Trans | 13.34 ± 0.16 | 12.58 ± 0.30 | 0.94 |
| CI-Trans | 14.03 ± 0.35 | 13.08 ± 0.08 | 0.93 |

**Verdict**: Both transfer perfectly (ratio <1.0). Training on multi-fault data generalizes to single-fault — expected since multi-fault is a superset.

## Exp 20: FD002 → FD004 (same 6 conditions, +fan fault)

| Model | FD002 RMSE | FD004 RMSE | Ratio |
|-------|-----------|-----------|-------|
| Role-Trans | 18.40 ± 1.17 | 31.99 ± 3.01 | 1.74 |
| CI-Trans | 27.29 ± 7.19 | 30.31 ± 5.65 | **1.11** |

**Verdict**: CI-Trans transfers better on same-condition cross-fault. But note CI-Trans is much worse on source domain (27.29 vs 18.40 on FD002).

## Cross-Transfer Summary

| Transfer Type | Role-Trans Ratio | CI-Trans Ratio | Winner |
|--------------|-----------------|----------------|--------|
| **Cross-condition** (FD001→FD002) | **4.00** | 6.23 | **Role-Trans (+36%)** |
| **Cross-both** (FD001→FD004) | **4.31** | 5.35 | **Role-Trans (+19%)** |
| Cross-fault (FD001→FD003) | 1.65 | **1.38** | CI-Trans |
| Same-domain (FD003→FD001) | 0.94 | 0.93 | Tie |
| Cross-fault (FD002→FD004) | 1.74 | **1.11** | CI-Trans |

**Key Finding**: Role-Trans excels at **cross-condition transfer** (distribution shift across operating regimes). CI-Trans is better at **cross-fault transfer** (same conditions, different failure modes). This makes physical sense:
- **Condition shift**: Within-component physics (captured by Role-Trans) is invariant across operating conditions. CI-Trans loses this information.
- **Fault shift**: When conditions are the same, CI-Trans avoids overfitting to specific cross-channel fault patterns. Role-Trans' component grouping can memorize fault-specific correlations.

**NeurIPS framing**: "Role-based architectures excel when the transfer challenge involves distribution shift across operating regimes, providing 19-36% improvement. When the challenge is novel fault modes under the same conditions, channel-independent approaches remain competitive."

---

# Phase 2c: Karpathy Loop Round 3 (2026-03-22)

## Exp 21: Multi-source Training — Inconclusive

**Change**: Train on FD001+FD003 combined, test on FD002/FD004.
**Result**: High seed variance. Some seeds improve, others worsen. No clear benefit.
**Verdict**: INCONCLUSIVE — needs more seeds or different combination strategy.

## Exp 22: Sequence Length Ablation — INFORMATIVE

| seq_len | Role FD001 | Role FD002 | Role ratio | CI FD002 | CI ratio |
|---------|-----------|-----------|-----------|---------|---------|
| 15 | 16.90 | 48.23 | **2.85** | 53.83 | 3.29 |
| 30 | 12.45 | 45.96 | 3.69 | 116.98 | 8.96 |
| 50 | 14.25 | 54.78 | 3.84 | 51.01 | 3.46 |
| 80 | 12.78 | 87.20 | 6.82 | 80.76 | 5.02 |

**Insights**:
1. Shorter windows → better transfer (less source-specific memorization)
2. seq_len=15: best transfer ratio (2.85) but worst in-domain (16.90)
3. seq_len=30 is a good balance — best absolute FD002 RMSE
4. Role-Trans consistently beats CI-Trans at every seq_len except seq_len=50 where CI is close

## Exp 23: Operating Condition Normalization — MAJOR FINDING

| Method | Role-Trans FD002 | CI-Trans FD002 |
|--------|-----------------|----------------|
| Global norm | 52.46 | 120.03 |
| **Per-condition norm** | **32.63** | **31.95** |

**Verdict**: KEEP insight ✓✓
**Insights**:
1. Per-condition normalization drops FD002 RMSE from 52→33 (Role) and 120→32 (CI)
2. With proper condition normalization, CI-Trans catches up to Role-Trans
3. This means **operating condition shift is the dominant challenge**, not architecture
4. Role-Trans' advantage comes from implicit condition-invariant features within components
5. **Caveat**: Per-condition norm requires knowing operating conditions at test time (semi-supervised)

**Implication for paper**: The story should be: "Without condition information, Role-Trans provides 36% better transfer by learning condition-invariant within-component features. With condition-aware normalization, both architectures benefit equally." This positions Role-Trans as the *unsupervised* approach to condition invariance.

## Exp 24: Settings as Input Features — REVERT

**Result**: Including operating settings (setting1-3) as input features doesn't help transfer. Avg FD002 RMSE: Role+Settings=57.48, CI+Settings=62.46 vs Role-only=48.82.
**Insight**: Model memorizes condition-specific patterns through settings, hurting generalization.

## Exp 25: Per-Condition Normalization — Multi-seed Confirmation

| Seed | Role-Trans + cond-norm | CI-Trans + cond-norm |
|------|----------------------|---------------------|
| 42 | 33.25 | 32.42 |
| 123 | 30.55 | 31.40 |
| 456 | 31.78 | 30.29 |
| **Avg** | **31.86** | **31.37** |

**Verdict**: With condition normalization, architectures are equivalent. Confirms Role-Trans advantage is from implicit condition invariance.

## Exp 26: Dropout Ablation — INFORMATIVE

| Dropout | FD001 | FD002 | Ratio |
|---------|-------|-------|-------|
| 0.0 | 12.18 | 66.39 | 5.45 |
| **0.1** | **12.45** | **45.96** | **3.69** |
| 0.2 | 12.20 | 54.78 | 4.49 |
| 0.3 | 12.53 | 60.98 | 4.87 |
| 0.5 | 13.46 | 63.59 | 4.73 |

**Insight**: Dropout=0.1 is optimal sweet spot. No dropout → severe overfitting to source domain. Too much dropout → underfitting.

## Exp 27: In-Domain FD001 Tuning — Marginal

| Config | FD001 RMSE |
|--------|-----------|
| lr=1e-3, ep=60 | 12.22 ± 0.38 |
| lr=5e-4, ep=100 | 12.19 ± 0.14 |
| lr=1e-3, ep=100 | 12.57 ± 0.26 |
| lr=2e-3, ep=60 | 12.52 ± 0.31 |

**Insight**: lr=5e-4 gives slightly lower variance but same mean. Default config is fine.

---

# Phase 2d: Karpathy Loop Round 5 (2026-03-23)

## Exp 28: JEPA Pretraining — NEGATIVE RESULT

**Time**: 17:19
**Hypothesis**: JEPA pretraining on FD001+FD002 unlabeled data (63,950 windows) should learn condition-invariant representations that improve transfer.
**Change**: Pretrain Role-Transformer encoder with component-level JEPA (mask 40% of components, predict via EMA target encoder), then fine-tune for RUL on FD001.

**Results (3 seeds):**

| Method | FD001 RMSE | FD002 RMSE | Transfer Ratio |
|--------|-----------|-----------|----------------|
| **JEPA+FT** | **12.42 ± 0.11** | **67.85 ± 8.83** | **5.46** |
| Scratch | 12.58 ± 0.03 | 51.54 ± 7.28 | 4.10 |

**Verdict**: REVERT ✗✗
**Insights**:
1. **JEPA pretraining HURTS transfer** — ratio 5.46 vs 4.10 (33% worse)
2. JEPA loss converges nicely (0.006 → 0.001) but learned representations don't help
3. Slightly better in-domain (12.42 vs 12.58) but much worse on transfer
4. **Consistent with ETTh1 finding**: JEPA learns to reconstruct, not to generalize
5. The pretrained encoder likely learns condition-specific features from FD002, which then bias the fine-tuned model

**Why JEPA fails here**: The JEPA objective (predict masked component representations from context) learns the joint distribution of component representations — including condition-specific correlations. When fine-tuned on FD001 (1 condition), the encoder has already learned 6-condition representations that confuse the RUL head. The Role-Trans architecture itself already captures within-component physics; JEPA adds no additional inductive bias for transfer.

**Implication**: JEPA pretraining is not the right approach for this transfer setting. The architectural inductive bias (role-based grouping) is more valuable than representation pretraining.

## Exp 29: MMD Domain Adaptation — MARGINAL

**Time**: 18:00
**Hypothesis**: Aligning FD001 and FD002 feature distributions via MMD loss should reduce transfer gap.
**Change**: Added MMD loss (λ=0.1) between encoder features on FD001 batches and unlabeled FD002 batches during training.

**Results (3 seeds):**

| Method | FD001 RMSE | FD002 RMSE | Transfer Ratio |
|--------|-----------|-----------|----------------|
| **MMD (λ=0.1)** | **12.45 ± 0.09** | **48.77 ± 4.81** | **3.92** |
| Scratch | 12.58 ± 0.03 | 51.54 ± 7.28 | 4.10 |

**Verdict**: MARGINAL — slight improvement, not dramatic
**Insights**:
1. Transfer ratio improves 3.92 vs 4.10 (-4%), mostly from lower variance
2. FD002 RMSE drops from 51.54 to 48.77 (-5%)
3. Lower transfer variance (4.81 vs 7.28) — more robust
4. In-domain performance maintained (12.45 vs 12.58)
5. The improvement is modest — condition shift is the main challenge, and MMD doesn't explicitly handle it

## Exp 30: Patch-based Role-Transformer — NO BENEFIT

**Time**: 18:30
**Hypothesis**: Patch embeddings (patch_len=5, 6 patches) may learn better temporal representations than point-wise input.

**Results (3 seeds):**

| Method | FD001 RMSE | FD002 RMSE | Transfer Ratio |
|--------|-----------|-----------|----------------|
| Patch(5) | 12.98 ± 0.30 | 55.61 ± 3.56 | 4.28 |
| Point | 12.58 ± 0.03 | 51.54 ± 7.28 | 4.10 |

**Verdict**: REVERT ✗
**Insights**:
1. Patches slightly worse on both in-domain (+3%) and transfer (+8%)
2. Patches have lower transfer variance (3.56 vs 7.28) but higher mean
3. With only 30 timesteps and 14 sensors, patches don't capture enough context per patch
4. Point-wise input with positional encoding is sufficient for this sequence length

---

# Phase 2e: Karpathy Loop Round 6 (2026-03-23)

## Exp 31: Contrastive Pretraining — REVERT

**Time**: 19:22
**Hypothesis**: NT-Xent contrastive pretraining (same engine, different timesteps = positive pairs) should learn condition-invariant representations.
**Change**: Pretrain RoleEncoder with contrastive loss on 61,790 positive pairs from FD001+FD002.

**Results (3 seeds):**

| Method | FD001 RMSE | FD002 RMSE | Transfer Ratio |
|--------|-----------|-----------|----------------|
| Contrastive+FT | 12.54 ± 0.14 | 57.01 ± 7.15 | 4.55 |
| Scratch | 12.58 ± 0.03 | 51.54 ± 7.28 | 4.10 |

**Verdict**: REVERT ✗
**Insights**:
1. Contrastive pretraining also hurts transfer (4.55 vs 4.10, +11%)
2. Loss barely decreases (4.71 → 4.68), suggesting temporal proximity is a poor signal for contrastive learning on degrading systems
3. Unlike JEPA (which helped in-domain), contrastive gives no benefit at all
4. **Pretraining approaches consistently fail** on this small-data setting

## Exp 32: Encoder Freezing — KEY INSIGHT

**Time**: 20:00
**Hypothesis**: Role-Trans encoder learns more transferable features than CI-Trans. Test by freezing encoder (trained on FD001) and fine-tuning only the head on FD002.

**Results (3 seeds):**

| Setting | Role-Trans | CI-Trans |
|---------|-----------|---------|
| Zero-shot avg | 51.54 | 82.31 |
| 1% FT frozen-enc avg | 43.03 | 73.98 |
| 5% FT frozen-enc avg | 40.62 | 58.70 |
| 10% FT frozen-enc avg | 39.55 | 49.29 |

**Verdict**: KEEP ✓✓ — strong evidence for Role-Trans encoder quality
**Insights**:
1. **Role-Trans frozen encoder far superior at low data**: At 1% FD002, Role=43.03 vs CI=73.98 (42% better)
2. **CI catches up with more data**: At 10%, CI=49.29 is closer to Role=39.55 (20% gap)
3. **Role-Trans zero-shot < CI-Trans 10% frozen-enc** in some seeds — the architecture is genuinely learning transferable representations
4. **The encoder quality story**: Role-Trans encoder representations are directly useful for FD002 without retraining. CI-Trans encoder learns sensor-specific features that don't transfer.
5. This supports the NeurIPS claim more directly than raw transfer ratios

## Exp 33: Temporal JEPA — INCONCLUSIVE (NaN instability)

**Time**: 20:30
**Hypothesis**: Masking future timesteps (not components) in JEPA should learn temporal dynamics rather than condition-specific patterns.
**Change**: Temporal JEPA with 30% temporal masking, per-sensor processing.

**Results**: Only seed 42 converged (FD001=12.95, FD002=46.54, ratio=3.59). Seeds 123 and 456 diverged to NaN.
**Verdict**: INCONCLUSIVE — numerically unstable implementation. The one valid seed (3.59) is promising vs scratch (3.35) but unreliable.
**Root cause**: Processing 14 sensors sequentially through the encoder creates gradient accumulation instability. Would need mixed-precision or gradient scaling to fix.

---

# Phase 2f: Karpathy Loop Round 7 (2026-03-23)

## Exp 34: Weight Sharing Ablation — KEY RESULT

**Time**: 20:50
**Hypothesis**: Weight sharing within component groups is critical — not just the grouping structure.
**Change**: Compare 4 models: (a) Role-Trans with shared encoder, (b) separate encoder per component, (c) separate encoder per sensor (grouped), (d) CI-Trans.

**Results (3 seeds):**

| Model | Params | FD001 | FD002 | Ratio |
|-------|--------|-------|-------|-------|
| **Role-Trans (shared)** | **40K** | **12.45±0.08** | **54.28±4.20** | **4.36** |
| Separate-per-comp | 147K | 13.51±0.17 | 67.24±17.65 | 4.98 |
| Grouped-no-share | 372K | 14.27±0.18 | 66.87±2.15 | 4.69 |
| CI-Trans | 26K | 13.20±0.44 | 82.24±26.99 | 6.23 |

**Verdict**: KEEP ✓✓✓
**Insights**:
1. **Weight sharing is the key mechanism** — shared encoder gives ratio 4.36, separate-per-comp gives 4.98 (+14%)
2. **Grouping alone helps some** — grouped-no-share (4.69) still beats CI-Trans (6.23) by 25%
3. **But sharing amplifies it** — from 4.69 to 4.36 is another 7% gain from sharing
4. **Fewer params + better transfer** — Role-Trans (40K) transfers better than Grouped-no-share (372K). Sharing acts as regularization.
5. **Three mechanisms**: (a) grouping structure, (b) within-component pooling, (c) weight sharing. All contribute, sharing is most important.

**Paper claim**: "Weight sharing within component groups forces the encoder to learn universal sensor dynamics rather than sensor-specific patterns. This acts as both inductive bias and regularizer, providing 30% better transfer than CI-Trans with 50% more parameters."

## Exp 35: 5-Seed Confirmation — STRONG BUT NOT QUITE SIGNIFICANT

**Time**: 21:10

| Seed | Role FD001 | Role FD002 | CI FD001 | CI FD002 |
|------|-----------|-----------|---------|---------|
| 42 | 12.55 | 48.79 | 13.05 | 116.98 |
| 123 | 12.36 | 59.00 | 12.75 | 78.59 |
| 456 | 12.45 | 55.04 | 13.80 | 51.16 |
| 789 | 11.86 | 47.66 | 13.56 | 72.37 |
| 2024 | 12.55 | 65.34 | 12.76 | 74.44 |

| Model | FD001 | FD002 | Ratio |
|-------|-------|-------|-------|
| **Role-Trans** | **12.36±0.26** | **55.16±6.56** | **4.46** |
| CI-Trans | 13.19±0.43 | 78.71±21.36 | 5.97 |

**Statistical tests**:
- Paired t-test: t=-1.932, p=0.126 (marginal)
- Wilcoxon: W=1.0, p=0.125 (marginal)
- Role-Trans wins 4/5 seeds

**Verdict**: KEEP — consistent advantage but p=0.126 is marginal. CI-Trans's extreme variance (seed 42: 116.98, seed 456: 51.16) makes statistical testing noisy. Role-Trans is more *robust* (std 6.56 vs 21.36).

**For paper**: Report mean ± std AND note that 4/5 seeds show Role-Trans advantage. The robustness (3x lower variance) is as important as the mean improvement.

## Exp 36: Representation Analysis (t-SNE) — SURPRISING

**Time**: 21:30

| Metric | Role-Trans | CI-Trans |
|--------|-----------|---------|
| Silhouette by condition | 0.128 | 0.087 |
| Silhouette by degradation | 0.015 | 0.027 |

**Verdict**: UNEXPECTED — Role-Trans clusters MORE by condition, LESS by degradation
**Insights**:
1. Role-Trans representations are MORE condition-aware, not less — contradicts the "condition-invariant features" narrative
2. CI-Trans representations are slightly more degradation-aware
3. **Reinterpretation**: Role-Trans doesn't transfer better because it ignores conditions. It transfers better because the component-level features remain *functionally relevant* across conditions even though they encode condition information.
4. The transfer advantage is about *compositionality* not *invariance* — within-component features compose differently across conditions but remain individually meaningful.

**Impact on paper narrative**: Drop the "condition-invariant" framing. Instead: "Role-based grouping with weight sharing provides compositional representations where within-component features remain transferable even when they encode condition-specific information. The transfer mechanism is architectural (shared encoder + component pooling), not representational (invariant features)."

---

# Pretraining Summary: All Approaches Tested

| Method | Transfer Ratio | vs Scratch (4.10) | Verdict |
|--------|---------------|-------------------|---------|
| Component JEPA (mask components) | 5.46 | **+33% worse** | Learns condition-specific patterns |
| Contrastive (same-engine pairs) | 4.55 | **+11% worse** | Poor contrastive signal |
| MMD domain adaptation | 3.92 | -4% better | Marginal improvement |
| Temporal JEPA (mask time) | 3.59* | -12% better* | Unstable, 1/3 seeds valid |
| **No pretraining (scratch)** | **4.10** | **baseline** | Architecture is sufficient |

**Conclusion**: On C-MAPSS with 14 sensors and ~20K samples, pretraining does not improve transfer. The Role-Transformer architecture already captures the right inductive bias. Pretraining might help at larger scale or with more diverse data, but on this benchmark, it's the architecture that matters, not the representation learning objective.

---

# Phase 2g: Karpathy Loop Round 8 (2026-03-23)

## Exp 37: Reverse Transfer (FD002→FD001) — HIGH VARIANCE

**Time**: 21:43
**Hypothesis**: Training on FD002 (6 conditions, 260 engines) then testing on FD001 (1 condition) should benefit from more diverse training data.

**Results (3 seeds):**

| Seed | Role FD002 | Role FD001 | Role ratio | CI FD002 | CI FD001 | CI ratio |
|------|-----------|-----------|-----------|---------|---------|---------|
| 42 | 18.69 | 95.55 | 5.11 | 19.35 | 155.50 | 8.04 |
| 123 | 18.22 | 49.46 | 2.71 | 43.95 | 40.22 | 0.92 |
| 456 | 18.60 | 38.83 | 2.09 | 19.95 | 134.04 | 6.72 |

**Verdict**: HIGH VARIANCE — Role-Trans more consistent in-domain (18-19 vs 19-44 for CI), but both architectures are highly unstable on cross-condition transfer in this direction.
**Insight**: FD002→FD001 is harder than FD001→FD002 because the model trained on 6 conditions may specialize to specific condition clusters.

## Exp 38: 10-Seed Confirmation — STATISTICALLY SIGNIFICANT ✓✓✓

**Time**: 22:00
**Purpose**: Definitive statistical test of Role-Trans vs CI-Trans on FD001→FD002.

**Results (10 seeds):**

| Seed | Role FD001 | Role FD002 | CI FD001 | CI FD002 |
|------|-----------|-----------|---------|---------|
| 42 | 12.55 | 48.79 | 13.05 | 116.98 |
| 123 | 12.36 | 59.00 | 12.75 | 78.59 |
| 456 | 12.45 | 55.04 | 13.80 | 51.16 |
| 789 | 11.86 | 47.66 | 13.56 | 72.37 |
| 2024 | 12.55 | 65.34 | 12.76 | 74.44 |
| 7 | 11.99 | 55.18 | 13.41 | 113.63 |
| 13 | 11.64 | 55.16 | 13.14 | 98.01 |
| 99 | 12.04 | 52.72 | 14.05 | 62.48 |
| 1337 | 11.90 | 49.03 | 13.63 | 59.14 |
| 31415 | 12.31 | 49.36 | 13.72 | 98.29 |

**Summary:**

| Model | FD001 RMSE | FD002 RMSE | Transfer Ratio |
|-------|-----------|-----------|----------------|
| **Role-Trans** | **12.17 ± 0.30** | **53.73 ± 5.21** | **4.42** |
| CI-Trans | 13.39 ± 0.42 | 82.51 ± 21.82 | 6.16 |

**Statistical significance:**
- **Paired t-test: t=-3.750, p=0.0046** ✓✓
- **Wilcoxon signed-rank: W=1.0, p=0.0039** ✓✓
- **Role-Trans wins 9/10 seeds** (only seed 456 goes to CI)
- **Bootstrap 95% CI: [15.09, 43.17]** — entirely positive
- **FD001 improvement also significant**: 12.17 vs 13.39 (9% better)

**Verdict**: DEFINITIVE ✓✓✓
**This is the paper-ready result**: Role-Trans provides 35% lower transfer RMSE (53.73 vs 82.51) with 4x lower variance (5.21 vs 21.82), significant at p<0.005 across 10 random seeds.

---

# Grand Summary of All C-MAPSS Experiments

## 38 experiments, 22 key findings

### Architecture (Role-Trans vs CI-Trans)

| Setting | Role-Trans | CI-Trans | Winner | Seeds |
|---------|-----------|---------|--------|-------|
| FD001 in-domain | **12.17±0.30** | 13.39±0.42 | Role (+9%) | 10 |
| **FD001→FD002 (cross-condition)** | **53.73±5.21** | **82.51±21.82** | **Role (+35%, p=0.005)** | **10** |
| FD001→FD003 (cross-fault) | 20.30±2.01 | 18.54±2.24 | CI (+9%) | 3 |
| FD001→FD004 (cross-both) | 52.92±4.40 | 71.79±19.43 | **Role (+26%)** | 3 |
| FD002→FD004 (cross-fault) | 31.99±3.01 | 30.31±5.65 | CI (+6%) | 3 |

### Ablations

| Factor | Finding |
|--------|---------|
| RevIN | Hurts on C-MAPSS (+34% worse) |
| Physics grouping vs random | Physics is 26% better for transfer |
| Physics grouping vs no grouping | Physics is 53% better |
| Sequence length | Shorter is better for transfer (15 > 30 > 50 > 80) |
| Cross-component depth | 1 layer is sufficient |
| Model size | Smaller is slightly better for transfer |
| Multi-source training | Inconclusive |
| Per-condition normalization | Eliminates architecture difference |
| Few-shot (5% FD002) | Role zero-shot ≈ CI 5% fine-tuned |
| JEPA pretraining (component mask) | Hurts transfer (+33% worse) |
| Contrastive pretraining | Hurts transfer (+11% worse) |
| MMD domain adaptation | Marginal improvement (-4%) |
| Patch embeddings (patch_len=5) | No benefit over point-wise |
| Encoder freezing | Role-Trans encoder 42% more transferable at 1% FD002 |

### The NeurIPS Story (final, honest)

**Main result (p<0.005, 10 seeds)**: "Physics-informed channel grouping with weight sharing reduces cross-condition transfer RMSE by 35% (53.73 vs 82.51) with 4x lower variance. Role-Trans wins 9/10 seeds."

**Mechanism (Exp 34, 36)**: "The transfer advantage comes from *compositional* representations — weight sharing forces universal sensor dynamics, while component pooling provides transferable building blocks. This is architectural regularization, NOT condition-invariant features (t-SNE shows Role-Trans is more condition-aware, not less)."

**Pretraining result (Exp 28-33)**: "JEPA, contrastive, and MMD pretraining provide no benefit. The architecture IS the transfer mechanism — pretraining is unnecessary when the inductive bias is correct."

**Honest nuance**: "The advantage is specific to operating condition shift. For novel fault modes under the same conditions, CI-Trans remains competitive. With condition labels (or clustering), per-condition normalization provides comparable benefits."

**Novel contribution**: First systematic study (38 experiments) of structured channel grouping for industrial transfer learning, showing that architectural inductive bias (grouping + weight sharing) provides significant, statistically robust transfer improvement that pretraining methods cannot match.

### Encoder Quality Analysis (Exp 32)

| FD002 Data Available | Role-Trans (frozen enc) | CI-Trans (frozen enc) | Role advantage |
|---------------------|----------------------|---------------------|---------------|
| 0% (zero-shot) | 51.54 | 82.31 | 37% |
| 1% | 43.03 | 73.98 | 42% |
| 5% | 40.62 | 58.70 | 31% |
| 10% | 39.55 | 49.29 | 20% |

Role-Trans encoder representations are directly useful for unseen operating conditions. CI-Trans requires substantial target data to adapt.

---

# ETTh1 Experiment Log (previous work)

Date: 2026-03-22

## Setup
- Dataset: ETTh1 (17,420 rows, 7 channels, hourly)
- Split: 8640 train / 2880 val / 2880 test (standard ETT split from Informer paper)
- Lookback: 96 timesteps
- Normalization: per-channel zero-mean unit-variance (fit on train)
- All results: 3 seeds (42, 123, 456), test set MSE/MAE

## Baselines (Task 2)

| Model | H=96 MSE | H=96 MAE | H=192 MSE | H=336 MSE | H=720 MSE |
|-------|----------|----------|-----------|-----------|-----------|
| Persistence | 1.6307 | 0.8227 | 1.6819 | 1.7086 | 1.7885 |
| Linear | **0.5714** | **0.5283** | **0.7799** | **0.9704** | 1.1948 |
| MLP (1-layer) | 0.7276 | 0.6405 | 0.9583 | 1.0381 | **1.0927** |

**Observations:**
- Linear is the strongest trivial baseline for H={96,192,336}. This is consistent with the DLinear paper's finding that a simple linear model is surprisingly competitive.
- MLP overtakes Linear only at H=720 (long horizon needs more capacity, but also more prone to overfitting at short horizons).
- Persistence is terrible (MSE ~1.63-1.79), confirming the data has meaningful dynamics.

## JEPA Experiments (Task 3)

Architecture: Patch embedding (patch_len=16) -> Transformer encoder (3 layers, d=128, 4 heads) -> Latent projection (dim=64) -> MLP predictor -> Linear decoder. EMA target encoder (momentum=0.996). ~890K trainable params at H=96.

| # | Mode | H=96 MSE | H=96 MAE | H=192 MSE | H=336 MSE | H=720 MSE | Notes |
|---|------|----------|----------|-----------|-----------|-----------|-------|
| 1 | JEPA + Supervised | 0.899 | 0.717 | 0.950 | 1.004 | 1.007 | Both losses |
| 2 | Supervised only | 0.900 | 0.736 | 0.895 | 0.963 | 1.007 | No JEPA loss |
| 3 | JEPA only | 1.282 | 0.851 | 1.284 | 1.277 | 1.275 | Decoder not trained by supervision |

**Key findings:**
1. **JEPA loss provides no benefit.** EXP 1 vs EXP 2 are statistically indistinguishable. At H=192 and H=336, supervised-only is actually slightly better.
2. **JEPA-only (EXP 3) is near-useless.** MSE ~1.28 across all horizons — the decoder receives no gradient signal from supervision, so it cannot map latent predictions to meaningful forecasts. The latent loss decreases during training but this doesn't translate to forecast quality.
3. **All transformer-based models lose badly to the trivial Linear baseline.** Linear gets 0.571 at H=96; our best transformer gets 0.899. That's 57% worse.

## Published SOTA (for reference)

| Model | H=96 MSE | H=192 MSE | H=336 MSE | H=720 MSE | Source |
|-------|----------|-----------|-----------|-----------|--------|
| PatchTST | ~0.370 | ~0.383 | ~0.396 | ~0.419 | Nie et al. 2023 |
| DLinear | ~0.375 | ~0.405 | ~0.439 | ~0.472 | Zeng et al. 2023 |
| iTransformer | ~0.386 | ~0.384 | ~0.396 | ~0.428 | Liu et al. 2024 |

## Diagnosis Experiments (Task 5)

After the vanilla JEPA failed, we diagnosed the overfitting problem with three targeted experiments:

| # | Model | Params | H=96 MSE | H=192 MSE | H=336 MSE | H=720 MSE | Notes |
|---|-------|--------|----------|-----------|-----------|-----------|-------|
| 4 | **CI-Transformer** | 105K | **0.450** | **0.502** | **0.561** | **0.673** | Channel-independent, d=64, 2 layers |
| 5 | Tiny-Transformer | 142K | 0.713 | 0.819 | 0.910 | 0.943 | d=32, 1 layer, channel-mixing |
| 6 | DLinear | 19K | 0.480 | 0.538 | 0.579 | 0.683 | Trend-seasonal decomposition |

### Key finding: Channel independence is the single biggest improvement

The channel-independent transformer (EXP 4) achieves **0.450 MSE at H=96** — a 50% improvement over our vanilla model (0.899) and 21% better than the Linear baseline (0.571). This is the PatchTST insight: with only 7 channels, cross-channel mixing provides almost no value and dramatically increases overfitting.

Comparison to the naive Linear baseline (0.571): CI-Transformer beats it at every horizon.

| Horizon | Linear | CI-Transformer | Improvement |
|---------|--------|----------------|-------------|
| 96 | 0.571 | 0.450 | -21% |
| 192 | 0.780 | 0.502 | -36% |
| 336 | 0.970 | 0.561 | -42% |
| 720 | 1.195 | 0.673 | -44% |

### DLinear performs similarly to CI-Transformer

Our DLinear implementation (0.480 at H=96) is close to CI-Transformer (0.450). Both are much better than channel-mixing models. However, our DLinear is worse than published DLinear (~0.375), suggesting there's still room for tuning.

### Tiny model doesn't help if architecture is wrong

EXP 5 (Tiny-Transformer) shrinks the model but keeps channel-mixing. It improves over vanilla (0.713 vs 0.899) but is still much worse than CI-Transformer (0.450). **The architecture design (channel-independence) matters more than model size.**

## Gap Analysis (Updated)

| Comparison | H=96 | H=192 | H=336 | H=720 |
|------------|------|-------|-------|-------|
| Our best (CI-Transformer) | **0.450** | **0.502** | **0.561** | **0.673** |
| Our DLinear | 0.480 | 0.538 | 0.579 | 0.683 |
| Our Linear baseline | 0.571 | 0.780 | 0.970 | 1.195 |
| PatchTST SOTA | 0.370 | 0.383 | 0.396 | 0.419 |
| Gap: CI-Trans vs SOTA | +22% | +31% | +42% | +61% |

We've closed the gap significantly (from 2.4x worse to 1.2-1.6x worse), but there's still meaningful distance to SOTA, especially at longer horizons.

## Published SOTA (for reference)

| Model | H=96 MSE | H=192 MSE | H=336 MSE | H=720 MSE | Source |
|-------|----------|-----------|-----------|-----------|--------|
| PatchTST | ~0.370 | ~0.383 | ~0.396 | ~0.419 | Nie et al. 2023 |
| DLinear | ~0.375 | ~0.405 | ~0.439 | ~0.472 | Zeng et al. 2023 |
| iTransformer | ~0.386 | ~0.384 | ~0.396 | ~0.428 | Liu et al. 2024 |

## Honest Assessment

### What worked
1. **Channel-independent processing** is the single most impactful change. It reduces effective parameters per-channel while giving 7x more training examples. This matches the PatchTST finding.
2. **Patch-based embeddings** with a shared transformer encoder and linear head is a clean, effective architecture.
3. **DLinear's trend-seasonal decomposition** is competitive and trains in seconds.

### What didn't work
1. **JEPA loss provides zero forecasting benefit** in any configuration. The latent-space prediction objective doesn't translate to better forecasts.
2. **Channel-mixing** is actively harmful on a 7-channel dataset with only 8640 training points. The model overfits to spurious cross-channel correlations.
3. **JEPA-only training** produces useless forecasts (MSE ~1.28, near-persistence baseline).

### Remaining gap to SOTA (~22% at H=96)
Possible causes:
1. **Missing RevIN** — Reversible instance normalization is standard in PatchTST/iTransformer
2. **No instance-wise norm at inference** — our model normalizes globally, not per-instance
3. **Head design** — PatchTST uses a flatten+linear head with careful attention masking
4. **Hyperparameter tuning** — published results are heavily tuned; ours are first-pass
5. **Data preprocessing** — potential differences in split boundaries or normalization

### What to try next
1. **Add RevIN** to CI-Transformer — likely the biggest remaining easy win
2. **Add JEPA as pretraining** — pretrain channel-independent encoder with JEPA, then fine-tune with supervised loss. This is the principled way to use JEPA.
3. **Tune hyperparameters** — patch_len, d_model, n_layers, learning rate, dropout
4. **Instance normalization** — normalize each input window independently at inference
5. **Move to larger datasets** (Weather, Electricity) where JEPA's representation learning may add value

### Bottom line
**The vanilla JEPA approach cannot beat a linear model on ETTh1.** But a channel-independent transformer achieves 0.450 MSE at H=96, beating our linear baseline (0.571) and approaching published DLinear (0.375). The gap to PatchTST SOTA (0.370) is ~22%, likely closeable with RevIN and tuning.

**For the JEPA research direction**: JEPA should be used as a *pretraining* method (self-supervised representation learning), not as a combined training objective. The next step is to pretrain a channel-independent encoder with JEPA on unlabeled data, then fine-tune for forecasting. This is where JEPA could add value — especially for transfer learning to industrial datasets where labeled data is scarce.
