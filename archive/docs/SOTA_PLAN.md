# KH-JEPA: Physics-Informed Foundation Model for Industrial Time Series

**Mission**: Build the first Koopman-based foundation model for industrial/physical time series that learns transferable dynamics across machines and domains.

**Compute Budget**: $500k+ (~10,000 H100-hours)

**Target Venue**: NeurIPS 2026 / ICML 2027 (Best Paper candidate)

---

## The Strategic Insight: Industrial vs Financial Time Series

There are fundamentally two types of time series, and we're choosing wisely:

| Aspect | Industrial/Physical | Financial/Market |
|--------|---------------------|------------------|
| **Underlying model** | Physics (differential equations) | Game theory (adversarial) |
| **Koopman-linearizable?** | **YES** — dx/dt = f(x) | NO — no true dynamics |
| **Predictability** | High (most of the time) | Low (efficient market) |
| **Challenge** | Rare anomalies after years | Constant chaos |
| **Value** | Prevent failures, save lives | Beat the market (zero-sum) |
| **Competition** | Less crowded | Everyone wants to be rich |
| **Foundation model justified?** | YES — physics transfers | Questionable |

### Why Koopman is Perfect for Industrial

Koopman operator theory: Any smooth dynamical system governed by differential equations can be represented as **linear dynamics in a lifted (latent) space**.

```
Physical reality:          dx/dt = f(x)      [nonlinear]
Koopman representation:    dz/dt = K·z       [linear!]
```

Industrial systems (motors, robots, pumps, HVAC, vehicles) ARE governed by physics:
- Newton's laws (F = ma)
- Thermodynamics (heat flow)
- Fluid dynamics (pumps, turbines)
- Electrical circuits (motors, drives)

**Key insight**: The Koopman operator K captures the *physics*, not the hardware. A learned K should transfer across machines of similar type because they obey the same physics.

### Why We're NOT Competing with Financial Prediction

Financial time series:
- No underlying physical law
- Adversarial (other traders adapt to your model)
- Efficient market hypothesis suggests prediction is fundamentally limited
- Everyone is trying (and mostly failing)

We sidestep this entirely by focusing on **physical systems where prediction is actually possible**.

---

## The Architecture: KH-JEPA

```
KH-JEPA = JEPA + Koopman Predictor + Hierarchy + VICReg + Cross-Variate
```

### Why Each Component Matters for Industrial

| Component | Physical Justification |
|-----------|----------------------|
| **Koopman Predictor** | Physical systems ARE linearizable — this is the right inductive bias |
| **Hierarchy** | Micro (vibration, electrical), Meso (operations), Macro (wear, degradation) |
| **Cross-Variate** | Sensors are physically coupled (motor current → temperature → vibration) |
| **VICReg** | Prevents collapse, ensures rich latent space for anomaly detection |
| **JEPA** | Predict in latent space = predict physics, not sensor noise |

### Interpretability Bonus

Koopman eigenvalues λ have physical meaning:
- **|λ| < 1**: Stable mode (damping)
- **|λ| > 1**: Unstable mode (potential fault!)
- **arg(λ)**: Oscillation frequency (rotation, vibration)
- **Eigenvalue drift**: System degradation over time

This is exactly what engineers want: interpretable system modes, not black-box predictions.

---

## Training Strategy: Industrial Focus

### Phase 1: Architecture Validation (Week 1-2)
**Goal**: Prove KH-JEPA works on industrial benchmarks

| Benchmark | Domain | Task | Metric | Why Include |
|-----------|--------|------|--------|-------------|
| ETTh1 | Energy/Thermal | Forecasting | MSE < 0.358 | Standard, quasi-physical |
| C-MAPSS | Turbofan engines | RUL prediction | RMSE < 13 | PHM standard, degradation |
| AURSAD | Robot welding | Anomaly detection | AUC > 0.95 | Industrial robotics |
| CWRU | Bearings | Fault classification | Acc > 99% | Classic PHM |

### Phase 2: Industrial Pre-training (Week 3-8)
**Goal**: Train at scale on industrial/physical data

**Data Sources** (prioritize physical systems):

| Dataset | Size | Domain | Include? |
|---------|------|--------|----------|
| Time-300B (energy subset) | ~50B points | Energy, power systems | ✅ Primary |
| Time-300B (manufacturing) | ~30B points | Industrial sensors | ✅ Primary |
| FactoryNet | 18M points | Robotics | ✅ Core |
| C-MAPSS extended | 10M points | Turbofan engines | ✅ Add |
| PHM datasets | 5M points | Bearings, gearboxes | ✅ Add |
| Time-300B (finance) | ~100B points | Markets | ❌ Skip |
| Time-300B (crypto) | ~20B points | Crypto | ❌ Skip |

**Model Scaling**:

| Size | Parameters | Data | Compute | Target |
|------|------------|------|---------|--------|
| Small | 50M | 10B industrial | 500 H100-hrs | Validate |
| Medium | 200M | 50B industrial | 2000 H100-hrs | Competitive |
| Large | 1B | 100B industrial | 5000 H100-hrs | SOTA |

### Phase 3: Industrial Benchmark Domination (Week 9-10)
**Goal**: SOTA on industrial-relevant benchmarks

| Benchmark | Task | Current SOTA | Our Target |
|-----------|------|--------------|------------|
| C-MAPSS FD001 | RUL (RMSE) | ~12 | < 10 |
| C-MAPSS FD003 | RUL (RMSE) | ~13 | < 11 |
| CWRU | Fault class (Acc) | 99.5% | 99.8% |
| PHM 2012 Bearing | RUL | - | Competitive |
| SKAB | Anomaly (F1) | ~0.85 | > 0.90 |
| ETT-full | Forecasting (MSE) | TTT 0.358 | < 0.35 |

**Cross-Machine Transfer** (our hero result):
- Train on AURSAD (UR3e robot)
- Test on voraus-AD (Yu-Cobot) — zero shot
- Target: >60% of supervised performance

### Phase 4: Paper & Release (Week 11-12)

---

## Novel Contributions (Paper Story)

### Primary Contribution
**First Koopman-based Foundation Model for Industrial Time Series**

> "Physical systems obey differential equations. Koopman theory tells us these become linear in a lifted space. We learn that space with JEPA, enabling a foundation model that captures *physics*, not hardware-specific statistics."

### Secondary Contributions

1. **Koopman JEPA Predictor**
   - Linear dynamics: z_{t+k} = K^k · z_t
   - O(1) long-horizon prediction
   - Interpretable eigenvalues = system modes
   - First application of Koopman to JEPA

2. **Cross-Machine Transfer via Physics**
   - Same physics → same Koopman operator
   - Demonstrated on UR3e → Yu-Cobot transfer
   - Foundation model justified by shared dynamics

3. **Hierarchical Multi-Resolution**
   - Micro/Meso/Macro time scales
   - Captures vibration → operations → degradation

4. **Industrial Benchmark Suite**
   - Unified evaluation across PHM, robotics, energy
   - Beyond forecasting: RUL, anomaly detection, fault classification

### Differentiators from Competition

| Model | Focus | Our Advantage |
|-------|-------|---------------|
| Chronos-2 | General forecasting | We're physics-informed (Koopman) |
| TimesFM | General forecasting | We focus on interpretability |
| Time-MoE | General, large scale | We have inductive bias for physical systems |
| Toto | Observability | We target industrial/manufacturing |

---

## Evaluation Framework

### Tier 1: Standard Forecasting
- ETT datasets (MSE, MAE)
- Electricity, Weather, Traffic

### Tier 2: Industrial-Specific
- **RUL Prediction**: C-MAPSS turbofan (RMSE, score)
- **Anomaly Detection**: SKAB, AURSAD, voraus-AD (AUC, F1)
- **Fault Classification**: CWRU bearings, PHM gearbox (Accuracy)

### Tier 3: Transfer Learning
- Cross-machine: UR3e → Yu-Cobot
- Cross-domain: Robots → CNC machines
- Few-shot adaptation: 1%, 10%, 100% labels

### Tier 4: Interpretability
- Koopman eigenvalue analysis
- Mode identification accuracy
- Degradation trend detection lead time

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Koopman doesn't scale | Fall back to MLP predictor (still have JEPA + VICReg) |
| Industrial focus too narrow | Position as "physical systems" (includes energy, transport, HVAC) |
| Can't beat Chronos-2 on general | Don't compete directly — focus on industrial benchmarks |
| Transfer fails | Paper becomes "what's needed for transfer" (still publishable) |

---

## Compute Plan

**Total budget**: $500k ≈ 10,000 H100-hours

| Phase | H100-hours | Cost |
|-------|------------|------|
| Architecture validation | 200 | $10k |
| Industrial data curation | 100 | $5k |
| Small-scale pre-training | 500 | $25k |
| Medium-scale pre-training | 2000 | $100k |
| Large-scale pre-training | 5000 | $250k |
| Benchmark evaluation | 500 | $25k |
| Ablations + analysis | 500 | $25k |
| Buffer | 200 | $10k |

---

## Timeline

```
Week 1-2:   Architecture validation (ETTh1 + C-MAPSS + AURSAD)
Week 3:     Industrial data curation from Time-300B
Week 4-5:   Small-scale pre-training (50M params)
Week 6-8:   Medium/Large-scale pre-training (200M-1B params)
Week 9-10:  Full benchmark evaluation + transfer experiments
Week 11-12: Paper writing + open-source release
```

**Key Milestones**:
- Week 2: KH-JEPA beats baselines on ETTh1 + C-MAPSS
- Week 5: 50M model shows positive transfer signal
- Week 8: 1B model competitive on industrial benchmarks
- Week 10: Cross-machine transfer demonstrated
- Week 12: Paper submitted, code released

---

## The Paper Pitch

**Title Options**:
1. "KH-JEPA: Physics-Informed Foundation Model for Industrial Time Series"
2. "Learning Transferable Dynamics: Koopman Meets JEPA for Industrial AI"
3. "From Hardware Statistics to Physics: Foundation Models for Physical Systems"

**One-sentence summary**:
> We show that combining JEPA with Koopman operator theory yields a foundation model that learns transferable physics rather than hardware-specific statistics, achieving SOTA on industrial benchmarks and demonstrating cross-machine transfer.

**Why reviewers will like it**:
1. **Novel**: First Koopman + JEPA combination
2. **Principled**: Grounded in dynamical systems theory
3. **Practical**: Real industrial applications
4. **Interpretable**: Eigenvalues have physical meaning
5. **Differentiated**: Not competing with general forecasting crowd

---

## Current Status

| Item | Status |
|------|--------|
| KH-JEPA architecture | ✅ Implemented |
| ETTh1 autoresearch setup | ✅ Ready |
| VICReg loss | ✅ Implemented |
| Koopman predictor | ✅ Implemented |
| Cross-variate attention | ✅ Implemented |
| Hierarchical encoder | ✅ Implemented |
| C-MAPSS dataloader | ❌ Not started |
| AURSAD dataloader | ✅ Exists in FactoryNet |
| Industrial data curation | ❌ Not started |
| Time-300B integration | ❌ Not started |

---

## References

### Core Papers
- [I-JEPA](https://arxiv.org/abs/2301.08243) - Original image JEPA
- [C-JEPA](https://proceedings.neurips.cc/paper_files/paper/2024) - VICReg for JEPA
- [Koopman for Deep Learning](https://www.nature.com/articles/s41467-018-07210-0) - Nature Communications

### Industrial Benchmarks
- [C-MAPSS](https://data.nasa.gov/dataset/c-mapss-aircraft-engine-simulator-data) - Turbofan degradation
- [CWRU Bearing](https://engineering.case.edu/bearingdatacenter) - Classic PHM
- [PHM 2012](https://www.phmsociety.org/) - Bearing RUL
- [FactoryNet](https://huggingface.co/datasets/Forgis/factorynet-hackathon) - Robot faults

### Foundation Models
- [Time-300B](https://huggingface.co/datasets/Maple728/Time-300B) - Training data
- [Chronos-2](https://arxiv.org/pdf/2510.15821) - Amazon baseline
- [Time-MoE](https://github.com/Time-MoE/Time-MoE) - ICLR 2025

---

## Decision Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-03-18 | KH-JEPA architecture | Novel + principled + efficient |
| 2026-03-18 | Industrial focus | Koopman fits physics, not finance |
| 2026-03-18 | Skip financial data | Adversarial, no true dynamics |
| 2026-03-18 | $500k compute | Enables foundation model scale |
| 2026-03-18 | Target: physics transfer | Differentiator from general FMs |
