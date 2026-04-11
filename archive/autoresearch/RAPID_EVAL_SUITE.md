# Rapid Evaluation Suite

**Purpose**: Rock-solid, reproducible benchmark for physics-informed channel grouping.

**Last updated**: 2026-03-25

---

## Tiered Structure

| Tier | System | Type | Channels | Physics Groups | Task | Transfer Test |
|------|--------|------|----------|----------------|------|---------------|
| **1** | Double Pendulum | Synthetic | 4 | 2 (mass_1, mass_2) | Forecasting | m1/m2=1.0 → m1/m2=0.5 |
| **2** | ??? | ??? | ??? | ??? | ??? | ??? |
| **3** | ETT | Real | 7 | 3 (HV, MV, LV) | Forecasting | ETTh1 → ETTh2 |

**⚠️ TIER 2 UNRESOLVED** — See "C-MAPSS Problem" section below.

---

## Tier 1: Double Pendulum (Synthetic)

**Why**: Perfect physics ground truth. We control everything. Fast iteration (<1 min).

### Dataset Generation
```bash
python autoresearch/experiments/generate_pendulum.py
```

### Files
```
data/pendulum/pendulum_source.csv  # m1=m2=1.0, L1=L2=1.0
data/pendulum/pendulum_target.csv  # m1=1.0, m2=0.5, L1=L2=1.0
```

### Columns
```
theta1, omega1, theta2, omega2, time
```
- 10,000 timesteps @ 100Hz
- Initial conditions: θ1=π/4, θ2=π/2, ω1=ω2=0

### Physics Groups
```python
PENDULUM_GROUPS = {
    "mass_1": [0, 1],  # theta1, omega1
    "mass_2": [2, 3],  # theta2, omega2
}
```

### Task
- Input: 30 timesteps
- Output: Predict next 10 timesteps (all 4 channels)
- Metric: MSE

### Baselines to Beat

| Model | Source MSE | Target MSE | Transfer Ratio |
|-------|-----------|-----------|----------------|
| Persistence (y_t+1 = y_t) | ~0.010 | ~0.015 | 1.50 |
| Linear | TBD | TBD | TBD |
| CI-Trans | 0.00178 | 0.00356 | 2.00 |
| Full-Attn | 0.00164 | 0.00346 | 2.11 |
| **PhysMask** | **0.00152** | **0.00296** | **1.95** |

### Success Criteria
- Beat PhysMask by >5%
- p < 0.05 on 5+ seeds
- Transfer ratio < 1.90

---

## Tier 2: ??? (UNRESOLVED)

### The C-MAPSS Problem

C-MAPSS was our Tier 2 candidate, but it has issues:

| Issue | Why It Matters |
|-------|---------------|
| **Synthetic** | It's a NASA simulation, not real sensor data |
| **RUL is hidden** | We predict remaining useful life, not observed sensors |
| **Inconsistent task** | Tier 1 and 3 are forecasting; RUL is regression |
| **Doesn't fit Dir 2/3** | JEPA and Slot-Concept are forecasting methods |

### Options for Tier 2

**Option A: C-MAPSS Sensor Forecasting**
- Keep C-MAPSS but predict sensors, not RUL
- Pro: Known physics groups, we control evaluation
- Con: No published baselines (we define our own)

**Option B: Nonlinear Benchmark (EMPS / Industrial Robot)**
- From nonlinearbenchmark.org
- Pro: REAL mechanical data, published SOTA
- Con: EMPS is SISO (1→1), no channel grouping needed
- Con: Industrial Robot needs investigation

**Option C: Drop Tier 2, Use ETT + Weather**
- Tier 2: ETT (7ch, real, published SOTA)
- Tier 3: Weather (21ch, real, larger scale)
- Pro: Both forecasting, both real, both have baselines
- Con: No mechanical system, no transfer test

**Option D: Find a real industrial dataset**
- Tennessee Eastman: Synthetic chemical process
- HVAC datasets: Real, but 2 channels typically
- SWaT: Real, 51 sensors, but security/anomaly focused

### Current Recommendation

**Use Option A (C-MAPSS Sensor Forecasting) with caveats:**
1. Acknowledge it's simulated
2. Define our own baselines (CI-Trans, Full-Attn, Linear)
3. Focus on transfer (FD001 → FD002) as the key metric
4. Don't claim to beat "published SOTA" (none exists for this task)

---

## Tier 3: ETT (Real Benchmark)

**Why**: Standard benchmark, published SOTA, reproducible, real data.

### Dataset
```
Download: https://github.com/zhouhaoyi/ETDataset
Files: ETT-small/ETTh1.csv, ETTh2.csv
```

### Columns
```
date, HUFL, HULL, MUFL, MULL, LUFL, LULL, OT
```
- HUFL/HULL: High voltage useful/useless load
- MUFL/MULL: Medium voltage useful/useless load
- LUFL/LULL: Low voltage useful/useless load
- OT: Oil temperature (often used as target)

### Split
- Train: 12 months (8640 samples for hourly)
- Val: 4 months
- Test: 4 months

### Physics Groups
```python
ETT_GROUPS = {
    "high_voltage": [0, 1],    # HUFL, HULL
    "medium_voltage": [2, 3],  # MUFL, MULL
    "low_voltage": [4, 5],     # LUFL, LULL
    "output": [6],             # OT
}
```

### Task
- Input: 96 timesteps (lookback)
- Output: Predict 96 timesteps (horizon)
- Target: All 7 channels (multivariate) or OT only (univariate)
- Metric: MSE, MAE

### Baselines to Beat (H=96, Multivariate)

| Model | ETTh1 MSE | ETTh2 MSE | Source |
|-------|----------|----------|--------|
| Linear | 0.386 | 0.559 | Published |
| DLinear | 0.375 | 0.289 | LTSF-Linear |
| PatchTST | 0.370 | 0.274 | PatchTST paper |
| iTransformer | 0.386 | 0.297 | iTransformer paper |
| **TimeXer** | **0.358** | **0.267** | SOTA (NeurIPS 2024) |

### Success Criteria
- Beat DLinear (0.375 on ETTh1)
- Approach TimeXer (0.358)
- Show physics grouping helps (compare PhysMask vs Full-Attn)

### Transfer Test
- Train on ETTh1, test on ETTh2 (different transformer station)
- This IS a valid transfer test (different physical system)

---

## Mandatory Baselines

Every experiment MUST include these baselines:

| Baseline | Implementation | Purpose |
|----------|---------------|---------|
| **Persistence** | y_{t+H} = y_t | Trivial lower bound |
| **Mean** | y_{t+H} = mean(y_train) | Sanity check |
| **Linear** | Linear regression on flattened input | Simple baseline |
| **CI-Trans** | Channel-independent transformer | Current paradigm |
| **Full-Attn** | Full cross-channel attention | Upper bound on mixing |

**Rule**: If your method doesn't beat Linear, don't log the experiment.

---

## Quick Validation Script

```python
# autoresearch/experiments/quick_validate.py

def quick_validate(model_class, model_kwargs):
    """Run <5 min validation before overnight experiments."""

    # Tier 1: Pendulum
    source_mse, target_mse = evaluate_pendulum(model_class, model_kwargs)

    print(f"Pendulum Source MSE: {source_mse:.5f}")
    print(f"Pendulum Target MSE: {target_mse:.5f}")
    print(f"Transfer Ratio: {target_mse/source_mse:.2f}")

    # Sanity checks
    assert source_mse < 0.01, "Source MSE too high (>0.01)"
    assert target_mse < 0.02, "Target MSE too high (>0.02)"
    assert source_mse < linear_baseline, "Doesn't beat Linear!"

    print("✓ Ready for full evaluation")
```

---

## Evaluation Protocol

### For Each Model

1. **Train on source** with 5+ seeds
2. **Evaluate on source** (report mean ± std)
3. **Transfer to target** (no retraining or 10% adaptation)
4. **Evaluate on target** (report mean ± std)
5. **Compute transfer ratio** = target_MSE / source_MSE

### Statistical Rigor

- Minimum 5 seeds for preliminary results
- Minimum 10 seeds for paper-ready results
- Report p-values for comparisons (paired t-test or Wilcoxon)
- Effect size (Cohen's d) for key comparisons

### What to Log

```markdown
## Exp N: [Description]

**Model**: [Architecture]
**Dataset**: Tier [1/2/3]
**Seeds**: [N]

| Metric | Value |
|--------|-------|
| Source MSE | X.XXX ± X.XXX |
| Target MSE | X.XXX ± X.XXX |
| Transfer Ratio | X.XX |
| vs CI-Trans | +/-X.X% (p=X.XXX) |
| vs Full-Attn | +/-X.X% (p=X.XXX) |

**Verdict**: [PROMISING / NEUTRAL / FAILED]
**Insight**: [What we learned]
```

---

## Directory Structure

```
data/
├── pendulum/
│   ├── pendulum_source.csv
│   └── pendulum_target.csv
├── cmapss/           # If using Option A
│   ├── train_FD001.txt
│   └── train_FD002.txt
└── ett/
    ├── ETTh1.csv
    └── ETTh2.csv

autoresearch/
├── experiments/
│   ├── generate_pendulum.py
│   ├── quick_validate.py
│   ├── tier1_pendulum.py
│   ├── tier2_cmapss.py      # Or alternative
│   └── tier3_ett.py
└── RAPID_EVAL_SUITE.md      # This file
```

---

## Changelog

- 2026-03-25: Initial version. Tier 2 unresolved (C-MAPSS problem identified).
