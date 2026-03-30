# Datasets to benchmark ideas and progress

**Goal**: Achieve clear SOTA on forecasting multivariate time series for mechanical systems, ideally via an architectural paradigm shift

---

## Tier 1: Double Pendulum (Synthetic / Controlled)

### Why Perfect
- **Ground truth physics**: Two masses, two rods, known equations
- **Known groups**: Mass 1 sensors, Mass 2 sensors
- **Controllable**: Generate unlimited data, vary parameters
- **Chaotic but predictable short-term**: Chaos kicks in at large angles (>10°), but 10-50 step forecasting is feasible

### Sensors & Groups
```python
GROUPS = {
    "mass_1": ["theta_1", "omega_1"],  # Angle and angular velocity of mass 1
    "mass_2": ["theta_2", "omega_2"],  # Angle and angular velocity of mass 2
}
# Or with Cartesian:
GROUPS = {
    "mass_1": ["x1", "y1", "vx1", "vy1"],
    "mass_2": ["x2", "y2", "vx2", "vy2"],
}
```

### Data Sources
1. **Generate ourselves** (recommended):
   ```python
   # Lagrangian equations, RK4 integration
   # Control: m1, m2, l1, l2, g, initial conditions
   # Generate 1000s of trajectories with varying parameters
   ```

2. **[IBM Double Pendulum Dataset](https://developer.ibm.com/exchanges/data/all/double-pendulum-chaotic/)**:
   - Real video data, 21 runs, ~17,500 frames each
   - Extracted positions via pattern matching
   - High-speed camera, real physics

### Task & Metric
- **Task**: Predict next 10-50 timesteps of (θ1, θ2, ω1, ω2)
- **Metric**: MSE over prediction horizon
- **Transfer**: Train on m1/m2=1.0, test on m1/m2=0.5 (parameter transfer)

### Chaos Concern
> "Double pendulum is chaotic for angles >10° from vertical"

**Solution**:
- Use small-angle regime for initial validation (non-chaotic)
- Then test on full chaotic regime (short-horizon only)
- Compare: Does physics grouping help in chaotic regime where learned models fail?

### Baseline
- Lagrangian Neural Network (LNN) - physics-informed, should be strong
- LSTM - standard
- iTransformer - learned channel attention

---

## Tier 2: Real Mechanical System (KAIST or Bearing)

### Option A: KAIST Multi-Modal (Recommended)
**Source**: [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S2352340923001671)

| Sensor Type | Count | Physical Meaning |
|-------------|-------|------------------|
| Accelerometers | 4 | Vibration (x,y,z + housing) |
| Microphone | 1 | Acoustic emission |
| Thermocouples | 2 | Temperature |
| Current transformers | 3 | Motor current (3-phase) |

```python
GROUPS = {
    "vibration": ["acc_x", "acc_y", "acc_z", "acc_housing"],
    "acoustic": ["mic"],
    "thermal": ["temp_1", "temp_2"],
    "electrical": ["current_a", "current_b", "current_c"],
}
```

**Conditions**: Normal, bearing faults, shaft misalignment, rotor unbalance × 3 torque loads

**Task**: Fault diagnosis or next-step prediction
**Transfer**: Cross-load (train 0 Nm, test 2 Nm) or cross-fault

### Option B: CWRU Bearing (Simpler but fewer groups)
- Only 2-4 accelerometers
- Single "bearing" component
- But very well-established baseline

**Recommendation**: Use KAIST for physics grouping validation (4 groups), CWRU for comparison to published results.

---

## Tier 3: Large Physical System (GIFT-Eval)

### Weather (21 variables, 4 groups)
```python
GROUPS = {
    "temperature": ["T", "Tpot", "Tdew"],
    "pressure": ["p", "VPmax", "VPdef", "VPact"],
    "humidity": ["rh", "sh", "H2OC"],
    "wind": ["wv", "max_wv", "wd"],
    "radiation": ["rain", "raining", "SWDR", "PAR", "max_PAR"],  # if available
}
```

**Source**: [GIFT-Eval](https://github.com/SalesforceAIResearch/gift-eval), Jena Climate from Max Planck Institute

**Task**: Forecast temperature
**Metric**: MSE, MAE (standard GIFT-Eval metrics)
**Baseline**: iTransformer (0.174 MSE), PatchTST, Chronos

### ETT (7 variables, 3 groups) - Optional
Simpler than Weather, also physical (transformer windings).

---

## Transfer Experiments: When?

### At Every Tier (Recommended)

| Tier | Transfer Test | Why |
|------|---------------|-----|
| 1 (Double Pendulum) | Change m1/m2 ratio | Parameter transfer, ground truth |
| 2 (KAIST) | Cross-load or cross-speed | Operating condition transfer |
| 3 (Weather) | ? Different location | Domain transfer |

**Rationale**: Transfer is our core claim. Validate it at every tier, not just at the end.

### Simplified Transfer Protocol
```python
def transfer_experiment(model, tier):
    if tier == 1:  # Double pendulum
        train_data = generate_pendulum(m_ratio=1.0)
        test_data = generate_pendulum(m_ratio=0.5)
    elif tier == 2:  # KAIST
        train_data = load_kaist(load="0Nm")
        test_data = load_kaist(load="2Nm")
    elif tier == 3:  # Weather
        # In-domain only, or find second weather station
        pass

    model.fit(train_data)
    return model.evaluate(test_data)
```

---

## Summary Table

| Tier | System | Vars | Groups | Data | Transfer |
|------|--------|------|--------|------|----------|
| 1 | Double Pendulum | 4-8 | 2 | Generate / IBM | m1/m2 ratio |
| 2 | KAIST Motor | 10 | 4 | Download | Cross-load |
| 3 | Weather | 21 | 4-5 | GIFT-Eval | In-domain |

---

## What Makes This Clean

1. **Progression**: Synthetic → Simple Real → Large Real
2. **Ground truth**: Tier 1 has exact physics equations
3. **Established benchmarks**: Tier 3 has GIFT-Eval leaderboard
4. **Transfer at each tier**: Not an afterthought
5. **Clear groups**: All datasets have physically meaningful groupings

---

## Immediate Actions

### 1. Generate Double Pendulum Data
```python
# 30 minutes of work
import numpy as np
from scipy.integrate import odeint

def double_pendulum(state, t, m1, m2, l1, l2, g):
    # Lagrangian equations of motion
    ...
    return [dtheta1, dtheta2, domega1, domega2]

# Generate training set: m_ratio = 1.0
# Generate test set: m_ratio = 0.5
```

### 2. Download KAIST Dataset
```bash
# From ScienceDirect supplementary materials
wget [KAIST_URL]
```

### 3. Setup GIFT-Eval Weather
```bash
pip install gift-eval
python -c "from gift_eval import download; download('weather')"
```

---

## The Paper Story

> **Title**: "Physics-Informed Channel Grouping for Mechanical Dynamical Systems"
>
> **Abstract**: We show that encoding known physical structure (which sensors measure the same component) into transformer attention outperforms learning channel relationships from data. We validate on three tiers: synthetic double pendulum (ground truth physics), real rotating machinery (KAIST), and large-scale weather forecasting (GIFT-Eval). Physics grouping provides X% better forecasting and Y% better transfer across operating conditions.

---

## Fact Check Summary

| Claim | Verified? | Source |
|-------|-----------|--------|
| Double pendulum datasets exist | ✅ | [IBM DAX](https://developer.ibm.com/exchanges/data/all/double-pendulum-chaotic/) |
| Double pendulum is chaotic >10° | ✅ | Multiple ML papers |
| Short-term prediction feasible | ✅ | LSTM papers show 10-50 step works |
| KAIST has 4 sensor groups | ✅ | [Paper](https://www.sciencedirect.com/science/article/pii/S2352340923001671) |
| Weather has physical groups | ✅ | Standard meteorology |
| GIFT-Eval has Weather dataset | ✅ | [GitHub](https://github.com/SalesforceAIResearch/gift-eval) |
