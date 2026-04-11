# Three Research Directions: Rigorous Definition

**Date**: 2026-03-24
**Purpose**: Define SOTA, gaps, validation strategy, and benchmarks for each direction

---

## Direction 1: Online Graph Learning for Physics-Based Channel Grouping

### 1.1 Current SOTA

| Method | Venue | Key Idea | Sparse? | Learns Physics? |
|--------|-------|----------|---------|-----------------|
| **[iTransformer](https://github.com/thuml/iTransformer)** | ICLR 2024 Spotlight | Inverted attention: channels as tokens, time as features | No (dense) | No |
| [MTGNN](https://arxiv.org/abs/2005.11650) | KDD 2020 | Adaptive adjacency via node embeddings A=softmax(E₁E₂ᵀ) | Top-k | No |
| [GTS](https://openreview.net/forum?id=oNvglzl_1r) | ICLR 2021 | Gumbel-softmax for discrete graph sampling | Binary | No |
| [NRI](https://arxiv.org/abs/1802.04687) | ICML 2018 | VAE for interaction graph, GNN decoder | Discrete types | Toy physics only |
| [Cini et al.](https://arxiv.org/abs/2209.11818) | JMLR 2023 | Score-based probabilistic with sparsity budget | Budget-constrained | No |

**Current leader for forecasting**: iTransformer (ICLR 2024 Spotlight)
- Treats each channel as a token, applies attention across channels
- No explicit graph, but implicit dense channel interactions
- Beats PatchTST, DLinear on standard benchmarks

**Current leader for graph discovery**: NRI
- Learns discrete interaction types (spring, no-spring)
- Validated on toy physics (springs, charged particles, Kuramoto)
- NOT validated on real engineering systems

### 1.2 Shortcomings & Genius Ideas to Fix Them

| Shortcoming | Why It Matters | Proposed Fix |
|-------------|----------------|--------------|
| **No physics validation** | Learned graphs may capture noise, not causality | Validate against known component structure (C-MAPSS, robot kinematics) |
| **No transfer** | Graph is tied to specific N sensors | Learn graph *structure* (component patterns) not *instances* |
| **Dense attention** | iTransformer uses O(C²) attention, no sparsity | Combine with sparsity mechanisms (top-k, Gumbel) |
| **Static graphs** | Real systems have mode-dependent coupling | Time-varying or condition-indexed graphs |
| **Channel-independence paradox** | CI methods often beat graph methods | Sparse physics graph should beat both extremes |

**Genius Idea**: **Physics-Validated Sparse Graph Learning**

```
1. Learn sparse graph via Gumbel-softmax or score-based method
2. Regularize toward block-diagonal structure (component groups)
3. Validate: Does learned graph match known physics coupling?
4. Transfer: Train graph on Machine A, apply structure to Machine B
```

Key insight: The learned graph should be **interpretable** and **physically meaningful**. If sensor i and j are in the same turbofan component, they should have high edge probability.

### 1.3 Validation Experiments

| Experiment | Hypothesis | Success Criterion |
|------------|------------|-------------------|
| **Graph ↔ Physics alignment** | Learned edges match component structure | Adjusted Rand Index > 0.7 vs ground-truth grouping |
| **Sparsity vs performance** | Sparse physics graph beats dense attention | Transfer ratio < iTransformer with 80% fewer edges |
| **Graph transfer** | Graph structure transfers across conditions | FD001→FD002 with learned graph < CI baseline |
| **Ablation: fixed vs learned** | Learned graph matches or beats Role-Trans | Ratio ≤ 4.0 (Role-Trans baseline) |

### 1.4 Benchmarks & Data Corpus

| Dataset | Domain | Sensors | Known Structure | Transfer Challenge |
|---------|--------|---------|-----------------|-------------------|
| **C-MAPSS** | Turbofan | 14→21 | 5 components (fan, HPC, combustor, turbine, nozzle) | Cross-condition (FD001→FD002) |
| **CWRU Bearing** | Bearings | 2-4 accel | Inner/outer race, ball, cage | Cross-load, cross-fault |
| **NRI Springs** | Simulated | N particles | Known spring connections | Topology generalization |
| **MuJoCo Ant/Humanoid** | Simulated robot | Joint sensors | Kinematic tree (known) | Sim-to-sim transfer |
| **Tennessee Eastman** | Chemical process | 52 sensors | Process flow diagram | Fault transfer |

**Physics simulation baselines** (from [MuJoCo](https://mujoco.org/)):
- Generate data from known dynamical systems
- Ground-truth graph = physical coupling
- Test: Does learned graph recover true physics?

---

## Direction 2: Learned Latent Concepts (Slot Attention)

### 2.1 Current SOTA

| Method | Venue | Key Idea | Unsupervised? | Time Series? |
|--------|-------|----------|---------------|--------------|
| **[Slot Attention](https://arxiv.org/abs/2006.15055)** | NeurIPS 2020 | Iterative competitive binding to K slots | Yes | No (images) |
| [SlotFormer](https://arxiv.org/abs/2210.05861) | ICLR 2023 | Slot attention + transformer for video prediction | Yes | Video |
| [SlotPi](https://arxiv.org/abs/2412.09600) | KDD 2025 | Slots + Hamiltonian mechanics for physics | Yes | Video (physics) |
| **[SlotFM](https://arxiv.org/abs/2501.xxxxx)** | arXiv 2025 | First slot attention on sensor data (accelerometers) | Yes | Yes! |
| [CBM](https://arxiv.org/abs/2007.04612) | ICML 2020 | Concept Bottleneck Models (supervised) | No | No |
| [CBM for Prognostics](https://arxiv.org/abs/2409.xxxxx) | Info Fusion 2025 | Expert-defined concepts for C-MAPSS RUL | No | Yes |

**Current leader for concept learning**: Slot Attention (Locatello et al.)
- Learns to decompose scene into object slots
- Iterative attention refinement with GRU
- Works on images, video; NOT yet on industrial sensors

**Closest to our goal**: SlotFM (2025)
- Applies slot attention to accelerometer data
- Discovers **frequency components**, not physical components
- Limited to single-sensor, no cross-machine transfer

### 2.2 Shortcomings & Genius Ideas to Fix Them

| Shortcoming | Why It Matters | Proposed Fix |
|-------------|----------------|--------------|
| **Slots collapse to uniform** | Our Exp 39-40: all slots get ~0.2 attention | Per-channel encoders (not shared), or entropy regularization |
| **No physics inductive bias** | Locatello's impossibility: need structure for disentanglement | Use sensor groupings as weak supervision |
| **Image-centric** | Slots designed for spatial tokens, not sensors | Treat each sensor as a "patch" in channel space |
| **No transfer** | Slots are dataset-specific | Slot *structure* (K components) should transfer |
| **Concepts not interpretable** | SlotFM discovers frequency bands, not physical meaning | Align slots to known components via auxiliary loss |

**Genius Idea**: **Physics-Guided Slot Attention**

```
1. Per-channel temporal encoder (NOT shared) → differentiated features
2. Slot attention with K = number of physical components
3. Weak supervision: encourage slot i to attend to component i sensors
4. Predict in slot space (JEPA-style) for dynamics
5. Transfer: slot structure transfers, not slot values
```

Key insight: Slot attention needs **diverse inputs** to differentiate slots. Shared encoder homogenizes features. Per-channel encoders + physics hints should fix this.

### 2.3 Validation Experiments

| Experiment | Hypothesis | Success Criterion |
|------------|------------|-------------------|
| **Slot ↔ Component alignment** | Slots discover physical components | Slot assignment entropy < 1.0 (not uniform) |
| **Per-channel vs shared encoder** | Per-channel enables differentiation | Slot entropy drops significantly |
| **K ablation** | K = num_components is optimal | K=5 beats K=3, K=14 on C-MAPSS |
| **Concept intervention** | Slots are interpretable | Setting slot i to "healthy" reduces predicted RUL |
| **Transfer via slots** | Slot structure transfers | FD001→FD002 with slots ≤ Role-Trans |

### 2.4 Benchmarks & Data Corpus

| Dataset | Domain | Sensors | Components | Concept Ground Truth |
|---------|--------|---------|------------|---------------------|
| **C-MAPSS** | Turbofan | 14 | 5 (fan, HPC, combustor, turbine, nozzle) | Binary degradation per component |
| **CWRU Bearing** | Bearings | 2-4 | 4 (inner race, outer race, ball, cage) | Fault location |
| **FactoryNet** | Robots | 7-13 | Joint-level | Anomaly per joint |
| **MuJoCo** | Simulated | Joint sensors | Body segments | Kinematic tree |
| **PRONOSTIA** | Bearings | 2 (accel, temp) | 1 bearing | RUL stages |

**Interpretability benchmark**:
- Train on healthy + faulty data
- Test: Can we identify which slot represents the faulty component?
- Gold standard: [CBM for Prognostics (EPFL 2025)](https://arxiv.org/abs/2409.xxxxx) — expert concepts

---

## Direction 3: Mechanical-JEPA

### 3.1 Current SOTA

| Method | Venue | Key Idea | Transfer? | Collapse Fix |
|--------|-------|----------|-----------|--------------|
| **[Brain-JEPA](https://arxiv.org/abs/2409.19407)** | NeurIPS 2024 Spotlight | JEPA for fMRI, gradient positioning, spatiotemporal masking | Cross-ethnic | EMA + structured masking |
| [I-JEPA](https://arxiv.org/abs/2301.08243) | CVPR 2023 | Image JEPA with block masking | Frozen eval | EMA only |
| [V-JEPA](https://arxiv.org/abs/2402.05065) | TMLR 2024 | Video JEPA, 90% masking | Frozen eval | EMA + high masking |
| [TS-JEPA](https://github.com/Sennadir/TS_JEPA) | NeurIPS WS 2024 | Time series JEPA, 70% patch masking | No | EMA |
| [MTS-JEPA](https://arxiv.org/abs/2502.xxxxx) | arXiv 2026 | Multivariate TS, codebook bottleneck | Limited | **Codebook!** |
| [C-JEPA](https://arxiv.org/abs/2410.xxxxx) | NeurIPS 2024 | JEPA + VICReg for collapse prevention | No | VICReg + EMA |

**Current leader for brain/bio**: Brain-JEPA
- Gradient positioning (functional coordinates from diffusion maps)
- Spatiotemporal masking (cross-ROI, cross-time, double-cross)
- Achieves SOTA on age prediction, disease diagnosis
- 40k subjects, 300 epochs

**Why our JEPA failed** (4 attempts):
1. Mask ratio too low (40% vs 70-90%)
2. No codebook (collapse prevention)
3. Shallow encoder (2 layers vs 4-6)
4. MSE loss (should be L1)
5. No multi-scale masking
6. Component-level masking learns condition-specific features

### 3.2 Shortcomings & Genius Ideas to Fix Them

| Shortcoming | Why It Matters | Proposed Fix |
|-------------|----------------|--------------|
| **Collapse to trivial solution** | EMA alone insufficient | Codebook bottleneck (MTS-JEPA) or VICReg (C-JEPA) |
| **Condition-specific features** | JEPA learns operating condition, not physics | Cross-condition masking + adversarial invariance |
| **No physics structure** | Standard JEPA ignores sensor relationships | Role-based masking (mask components, not random) |
| **Single scale** | Misses multi-resolution dynamics | Dual-resolution: fine (local) + coarse (trend) |
| **No transfer benchmark** | I-JEPA/V-JEPA don't test cross-domain | Define mechanical transfer benchmark |

**Genius Idea**: **Mechanical-JEPA with Physics Priors**

```
1. Physics-informed masking:
   - Cross-component: mask HPC, predict from fan/turbine
   - Cross-time: mask future, predict from past
   - Double-cross: mask both (hardest)

2. Codebook bottleneck:
   - Quantize latent to K discrete states
   - Prevents collapse, forces categorical dynamics

3. Condition-invariant objective:
   - Adversarial loss: predictor can't distinguish conditions
   - Or: VICReg variance/covariance regularization

4. Multi-scale architecture:
   - Fast encoder: 16-timestep windows
   - Slow encoder: 64-timestep windows
   - Predict at both scales

5. Role-based structure:
   - Encoder respects component groupings
   - Predictor operates in component space
```

### 3.3 Validation Experiments

| Experiment | Hypothesis | Success Criterion |
|------------|------------|-------------------|
| **Codebook vs no-codebook** | Codebook prevents collapse | Latent entropy > 2.0 (not mode collapse) |
| **Masking ratio** | High masking (70-90%) improves transfer | 80% mask beats 40% mask on FD002 |
| **Cross-component masking** | Structured masking learns physics | Cross-component beats random masking |
| **Multi-scale** | Dual-resolution helps | Multi-scale beats single-scale |
| **JEPA vs supervised** | JEPA pretraining helps at scale | Pretrain→finetune beats scratch with 1% labels |
| **Transfer benchmark** | Mechanical-JEPA beats Brain-JEPA baseline | Cross-condition ratio < 4.0 |

### 3.4 Benchmarks & Data Corpus

| Dataset | Domain | Size | Transfer Challenge | Brain-JEPA Analog |
|---------|--------|------|-------------------|-------------------|
| **C-MAPSS** | Turbofan | 700 engines | Cross-condition (1→6 operating modes) | Cross-site |
| **CWRU + NASA + PHM** | Bearings | 1000s samples | Cross-rig, cross-fault | Cross-scanner |
| **MuJoCo Gym** | Simulated robots | Unlimited | Sim-to-sim, morphology transfer | N/A |
| **FactoryNet** | Industrial robots | 5 datasets | Cross-robot (UR3→Franka) | Cross-ethnic |
| **Tennessee Eastman** | Chemical | 52 sensors | Cross-fault mode | Cross-disease |

**Pretraining corpus** (for foundation model):
- Aggregate: C-MAPSS + CWRU + NASA + PHM2012 + Tennessee Eastman
- ~100k samples across domains
- Unified schema: setpoint / effort / feedback
- Target: zero-shot transfer to held-out domain

---

## Cross-Direction Comparison

| Aspect | Dir 1: Sparse Graph | Dir 2: Slot Concepts | Dir 3: Mechanical-JEPA |
|--------|--------------------|--------------------|----------------------|
| **Core innovation** | Physics-validated learned graph | Unsupervised component discovery | JEPA with physics priors |
| **SOTA to beat** | iTransformer (ICLR 2024) | Slot Attention + SlotFM | Brain-JEPA (NeurIPS 2024) |
| **Novelty** | Medium (graph learning exists) | High (slots for sensors is new) | High (JEPA for transfer is new) |
| **Risk** | Low (incremental) | Medium (slots may not discover) | High (JEPA failed 4x) |
| **Data requirement** | Any multivariate TS | Needs component structure | Needs large pretraining corpus |
| **Interpretability** | Graph edges | Slot assignments | Latent dynamics |
| **Transfer mechanism** | Graph structure | Slot structure | Pretrained representations |

---

## Unified Benchmark Suite

### Tier 1: Must-Have (C-MAPSS)
- **In-domain**: FD001 RMSE < 13.0 ✓ (achieved: 12.17)
- **Cross-condition**: FD001→FD002 ratio < 5.0 ✓ (achieved: 4.42)
- **Comparison**: Beat iTransformer, CI-Transformer

### Tier 2: Mechanical Systems
| Dataset | Task | Metric | Baseline to Beat |
|---------|------|--------|------------------|
| CWRU Bearing | Fault diagnosis | Accuracy | >95% (CNN) |
| CWRU→NASA | Cross-rig transfer | Transfer ratio | <2.0 |
| PHM 2012 | RUL prediction | RMSE | Published SOTA |
| Tennessee Eastman | Fault detection | F1 | >90% |

### Tier 3: Simulated Physics (Ground Truth)
| Dataset | Task | Metric | Purpose |
|---------|------|--------|---------|
| NRI Springs | Graph recovery | Edge F1 | Validate graph learning |
| MuJoCo Ant | Dynamics prediction | MSE | Validate JEPA |
| Kuramoto | Interaction inference | ARI | Validate slot discovery |

### Tier 4: Stretch Goals (Foundation Model)
| Dataset | Task | Metric | Purpose |
|---------|------|--------|---------|
| FactoryNet | Cross-robot transfer | Anomaly F1 | Real industrial transfer |
| Weather/Traffic | Zero-shot forecast | MSE | Generalization beyond industrial |
| Aggregate corpus | Pretraining scale | Perplexity | Foundation model quality |

---

## Immediate Next Steps

### Week 1: PoC Experiments
1. **Graph learning**: Implement GTS-style Gumbel-softmax on C-MAPSS, measure graph↔physics alignment
2. **Slot attention fix**: Per-channel encoders instead of shared, measure slot entropy
3. **JEPA fix**: Add codebook bottleneck, increase masking to 80%

### Week 2: Cross-Dataset Validation
1. Download and preprocess: CWRU, NASA Bearing, Tennessee Eastman
2. Define unified schema
3. Run Role-Trans baseline on each
4. Test cross-dataset transfer

### Week 3: Pick Winner
1. Compare PoC results across directions
2. Select most promising direction
3. Scale up experiments
4. Begin paper writing

---

## Physical Concepts for Latent Slots

### Effort Signals → **Energy Concepts**

| Signal Type | Physical Concept | How to Compute |
|-------------|------------------|----------------|
| Torque τ | Rotational energy | E = ∫ τ·ω dt |
| Force F | Work done | W = ∫ F·v dt |
| Current I | Electrical energy | E = ∫ I²R dt |
| Pressure P | Hydraulic work | W = ∫ P·Q dt |

**Slot interpretation**: Each slot represents an *energy flow path* through the system.

### Setpoint Signals → **Control Regime Concepts**

| Signal Type | Physical Concept | How to Compute |
|-------------|------------------|----------------|
| Position x_ref | **Tracking error** | e = x_actual - x_ref |
| Velocity v_ref | **Compliance** | How quickly does effort respond to Δv_ref? |
| Mode indicator | **Operating regime** | Steady-state / transient / ramp / step |
| Trajectory | **Smoothness** | Jerk = d³x/dt³ (rate of acceleration change) |

**Slot interpretation**: Each slot represents a *control mode* or *trajectory phase*.

### Combined Concept: **Impedance**

The relationship between setpoint and effort is **mechanical impedance**:

```
Z(s) = Effort(s) / Motion(s) = Force / Velocity = Torque / Angular velocity
```

**Genius idea for slots**: Learn slots that represent *impedance modes*:
- Slot 1: Free motion (low impedance, effort tracks setpoint easily)
- Slot 2: Contact (high impedance, effort spikes on setpoint change)
- Slot 3: Degradation (impedance drifts over time)

This is physically meaningful AND transferable across machines.

---

## Precise Validation Framework

### Unified Dataset Suite (Same for All 3 Directions)

All experiments use the **same datasets** for fair comparison:

#### Tier 0: Development (Fast Iteration)
| Dataset | Samples | Sensors | Task | Metric |
|---------|---------|---------|------|--------|
| **C-MAPSS FD001** | 100 engines | 14 | RUL | RMSE |
| **C-MAPSS FD002** | 260 engines | 14 | Transfer target | RMSE |

**Why**: Fast (<5 min per experiment), known physics structure, established baselines.

#### Tier 1: Mechanical Systems (Core Validation)

| Dataset | Source | Samples | Sensors | Physics Structure | Download |
|---------|--------|---------|---------|-------------------|----------|
| **C-MAPSS (all)** | [NASA](https://www.kaggle.com/datasets/behrad3d/nasa-cmaps) | 709 engines | 14-21 | 5 turbofan components | ✅ Open |
| **CWRU Bearing** | [Case Western](https://engineering.case.edu/bearingdatacenter) | 4 conditions | 2-4 accel | 4 fault locations | ✅ Open |
| **NASA Bearing** | [NASA PCoE](https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository) | 3 runs | 4 accel | Bearing degradation | ✅ Open |
| **PHM 2012 FEMTO** | [PHM Society](https://github.com/Lucky-Loek/ieee-phm-2012-data-challenge-dataset) | 17 runs | 2 (vib+temp) | Bearing RUL | ✅ Open |
| **MAFAULDA** | [Kaggle](https://www.kaggle.com/datasets/uysimty/mafaulda-machinery-fault-database) | 1951 samples | 8 | Rotating machinery faults | ✅ Open |

**Transfer experiments**:
- C-MAPSS: FD001 → FD002 (1→6 conditions)
- Bearing: CWRU → NASA (cross-rig)
- Bearing: CWRU 0HP → CWRU 3HP (cross-load)

#### Tier 2: Process Industry

| Dataset | Source | Samples | Sensors | Physics | Download |
|---------|--------|---------|---------|---------|----------|
| **Tennessee Eastman** | [Harvard](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/6C3JR1) | 22 faults | 52 | Chemical process flow | ✅ Open |
| **SECOM** | [UCI](https://archive.ics.uci.edu/dataset/179/secom) | 1567 | 590 | Semiconductor | ✅ Open |

#### Tier 3: Simulated Physics (Ground Truth Graphs)

| Dataset | Source | Purpose | Known Structure |
|---------|--------|---------|-----------------|
| **NRI Springs** | [GitHub](https://github.com/ethanfetaya/NRI) | Graph recovery | Spring connections |
| **NRI Charged** | [GitHub](https://github.com/ethanfetaya/NRI) | Interaction types | Charge signs |
| **MuJoCo Ant** | [Gymnasium](https://gymnasium.farama.org/) | Dynamics prediction | Kinematic tree |
| **Kuramoto** | Generate | Oscillator coupling | Coupling matrix |

#### Tier 4: Standard Forecasting (GIFT-Eval Subset)

| Dataset | Domain | Sensors | Frequency | From |
|---------|--------|---------|-----------|------|
| **ETTh1, ETTh2** | Electricity | 7 | Hourly | [GIFT-Eval](https://github.com/SalesforceAIResearch/gift-eval) |
| **ETTm1, ETTm2** | Electricity | 7 | 15-min | GIFT-Eval |
| **Electricity** | Power | 321 | Hourly | GIFT-Eval |
| **Weather** | Climate | 21 | 10-min | GIFT-Eval |
| **Traffic** | Transport | 862 | Hourly | GIFT-Eval |
| **Solar** | Energy | 137 | 10-min | GIFT-Eval |

**Why GIFT-Eval**: Standardized evaluation, leaderboard, comparison to Chronos/Moirai/TimesFM.

---

## Validation Protocol per Direction

### All Directions: Same Metrics, Same Splits

| Tier | Train | Val | Test | Metric |
|------|-------|-----|------|--------|
| 0-2 (Mechanical) | 70% | 10% | 20% | RMSE, Transfer Ratio |
| 3 (Simulated) | Generated | - | Generated | Edge F1, ARI |
| 4 (GIFT-Eval) | Standard | Standard | Standard | MASE, sMAPE |

### Direction 1: Sparse Graph Learning

| Experiment | Dataset | Metric | Baseline | Target |
|------------|---------|--------|----------|--------|
| Graph ↔ physics alignment | C-MAPSS | Adjusted Rand Index | Random: ~0.0 | **>0.5** |
| Sparse vs dense | C-MAPSS | Transfer ratio | iTransformer: ~5.5 | **<4.5** |
| Cross-rig transfer | CWRU→NASA | Accuracy | CI-Trans: ~60% | **>70%** |
| Graph recovery | NRI Springs | Edge F1 | NRI: ~0.95 | **≥0.95** |
| Zero-shot forecast | ETTh1 | MASE | Chronos: ~0.42 | **<0.45** |

### Direction 2: Slot Concepts

| Experiment | Dataset | Metric | Baseline | Target |
|------------|---------|--------|----------|--------|
| Slot ↔ component alignment | C-MAPSS | Assignment entropy | Uniform: 1.61 | **<1.0** |
| Concept intervention | C-MAPSS | RUL change on slot edit | N/A | Interpretable |
| K ablation | C-MAPSS | Transfer ratio | K=14 (CI): 6.2 | K=5: **<4.5** |
| Cross-rig transfer | CWRU→NASA | Accuracy | 60% | **>70%** |
| Concept discovery | Tennessee Eastman | Fault isolation | Expert: 95% | **>85%** |

### Direction 3: Mechanical-JEPA

| Experiment | Dataset | Metric | Baseline | Target |
|------------|---------|--------|----------|--------|
| Collapse prevention | C-MAPSS | Latent entropy | Collapsed: <0.5 | **>2.0** |
| Pretraining helps | C-MAPSS 1% labels | RMSE | Scratch: ~25 | **<20** |
| Cross-condition | FD001→FD002 | Transfer ratio | Role-Trans: 4.4 | **<4.0** |
| Cross-rig | CWRU→NASA | Accuracy | 60% | **>75%** |
| Zero-shot forecast | ETTh1 | MASE | MOMENT: ~0.44 | **<0.44** |
| Foundation scale | Aggregate corpus | Downstream avg | - | SOTA |

---

## Specific Baselines to Beat

### Tier 1-2 (Mechanical)

| Dataset | Task | SOTA Method | SOTA Score | Source |
|---------|------|-------------|------------|--------|
| C-MAPSS FD001 | RUL | LightGBM ensemble | RMSE 6.62 | Nature Sci Rep 2025 |
| C-MAPSS FD001 | RUL (DL only) | Transformer | RMSE 11.36 | Various |
| C-MAPSS FD002 | RUL (trained on FD002) | Similar | RMSE ~13-15 | Various |
| CWRU | Fault diagnosis | 1D-CNN | Accuracy 99%+ | Many papers |
| CWRU→Other | Cross-domain | DANN | Accuracy ~85% | Domain adaptation papers |
| Tennessee Eastman | Fault detection | Various | F1 >95% | Process control literature |

### Tier 4 (GIFT-Eval)

| Dataset | Metric | Chronos | Moirai | TimesFM | Our Target |
|---------|--------|---------|--------|---------|------------|
| ETTh1 | MASE | 0.42 | 0.41 | 0.43 | <0.42 |
| ETTh2 | MASE | 0.38 | 0.37 | 0.39 | <0.38 |
| Electricity | MASE | 0.85 | 0.82 | 0.88 | <0.85 |
| Weather | MASE | 0.65 | 0.63 | 0.67 | <0.65 |

---

## Implementation Checklist

### Data Preparation (Week 1)

```bash
# Download all datasets
data/
├── cmapss/           # FD001-FD004
├── cwru/             # 4 load conditions × 4 fault types
├── nasa_bearing/     # 3 test-to-failure runs
├── phm2012/          # FEMTO bearing
├── mafaulda/         # Rotating machinery
├── tennessee_eastman/# 22 fault scenarios
├── nri/              # Springs, charged (generate)
├── gift_eval/        # ETT, Electricity, Weather, Traffic
└── unified_schema.py # Convert all to setpoint/effort/feedback
```

### Unified Schema

```python
@dataclass
class UnifiedSample:
    setpoint: np.ndarray   # [T, C_setpoint] - commanded values
    effort: np.ndarray     # [T, C_effort] - forces, torques, currents
    feedback: np.ndarray   # [T, C_feedback] - measured positions, etc.
    condition: int         # Operating regime ID
    health: float          # 0=healthy, 1=failed (or RUL)
    machine_id: str        # For transfer experiments
```

### Evaluation Script

```python
def evaluate_all_tiers(model, direction: str):
    results = {}

    # Tier 0-1: Mechanical
    results['cmapss_fd001'] = eval_rul(model, 'FD001')
    results['cmapss_transfer'] = eval_transfer(model, 'FD001', 'FD002')
    results['cwru_accuracy'] = eval_classification(model, 'CWRU')
    results['cwru_transfer'] = eval_transfer(model, 'CWRU', 'NASA')

    # Tier 2: Process
    results['tep_fault'] = eval_fault_detection(model, 'TEP')

    # Tier 3: Simulated (direction-specific)
    if direction == 'graph':
        results['nri_edge_f1'] = eval_graph_recovery(model, 'NRI_Springs')
    elif direction == 'slots':
        results['slot_entropy'] = eval_slot_alignment(model, 'CMAPSS')
    elif direction == 'jepa':
        results['latent_entropy'] = eval_collapse(model)

    # Tier 4: GIFT-Eval
    for dataset in ['ETTh1', 'ETTh2', 'Electricity', 'Weather']:
        results[f'gift_{dataset}'] = eval_gift(model, dataset)

    return results
```

---

## Decision Criteria

After Week 2 PoC experiments, pick the direction where:

1. **Tier 0 works**: C-MAPSS transfer ratio < 4.5
2. **Novel metric succeeds**: Graph ARI > 0.5, OR Slot entropy < 1.0, OR Latent entropy > 2.0
3. **Cross-rig transfers**: CWRU→NASA accuracy > 70%

If multiple directions succeed, pick based on:
- **Novelty**: Slots > JEPA > Graph
- **Risk**: Graph < Slots < JEPA
- **Paper narrative**: JEPA > Slots > Graph

---

## References

- [iTransformer](https://github.com/thuml/iTransformer) - ICLR 2024 Spotlight
- [NRI](https://github.com/ethanfetaya/NRI) - ICML 2018
- [Slot Attention](https://arxiv.org/abs/2006.15055) - NeurIPS 2020
- [Brain-JEPA](https://arxiv.org/abs/2409.19407) - NeurIPS 2024 Spotlight
- [GIFT-Eval](https://github.com/SalesforceAIResearch/gift-eval) - NeurIPS 2024 Workshop
- [MuJoCo](https://mujoco.org/) - Physics simulation
- [C-MAPSS](https://www.kaggle.com/datasets/behrad3d/nasa-cmaps) - NASA turbofan
- [CWRU Bearing](https://engineering.case.edu/bearingdatacenter) - Bearing benchmark
- [PHM Datasets Overview](https://arxiv.org/abs/2403.13694) - Comprehensive survey
- [Tennessee Eastman](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/6C3JR1) - Process benchmark
