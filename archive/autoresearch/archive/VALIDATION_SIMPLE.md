# Simplified Validation: Physics Groups on Standard Benchmarks

**One idea, three datasets, one metric.**

---

## The Claim

> Physics-informed channel grouping improves forecasting and transfer on systems with known physical structure.

## The Validation

| Dataset | Components | Transfer Test | Baseline | We Beat It? |
|---------|------------|---------------|----------|-------------|
| **C-MAPSS** | 5 (turbofan) | FD001→FD002 | CI-Trans | ✅ Done (p=0.005) |
| **ETT** | 3 (transformer windings) | ETTh1→ETTh2 | iTransformer | ? |
| **Weather** | 4 (temp/pressure/humidity/wind) | ? | iTransformer | ? |

**That's it.** If physics grouping helps on all three, we have a paper.

---

## Dataset Details

### C-MAPSS (Development)
```python
GROUPS = {
    "fan": ["s8", "s12", "s21"],
    "hpc": ["s3", "s7", "s11", "s20"],
    "combustor": ["s2", "s14"],
    "turbine": ["s4", "s9", "s13"],
    "nozzle": ["s15", "s17"],
}
# Task: RUL prediction
# Transfer: 1 operating condition → 6 operating conditions
# Metric: RMSE, Transfer Ratio
```

### ETT (Standard Benchmark)
```python
GROUPS = {
    "high_voltage": ["HUFL", "HULL"],
    "med_voltage": ["MUFL", "MULL"],
    "low_voltage": ["LUFL", "LULL"],
    # OT is target, not input
}
# Task: Forecast OT (oil temperature)
# Transfer: ETTh1 → ETTh2 (different transformer)
# Metric: MSE, MAE (standard)
```

### Weather (Standard Benchmark)
```python
GROUPS = {
    "temperature": ["T (degC)", "Tpot (K)", "Tdew (degC)"],
    "pressure": ["p (mbar)", "VPmax (mbar)", "VPdef (mbar)"],
    "humidity": ["rh (%)", "sh (g/kg)", "H2OC (mmol/mol)"],
    "wind": ["wv (m/s)", "max. wv (m/s)", "wd (deg)"],
}
# Task: Forecast temperature
# Transfer: Different weather stations? Or just in-domain.
# Metric: MSE, MAE (standard)
```

---

## Baselines (From GIFT-Eval Leaderboard)

| Model | ETTh1 (MSE) | ETTh2 (MSE) | Weather (MSE) |
|-------|-------------|-------------|---------------|
| iTransformer | 0.386 | 0.340 | 0.174 |
| PatchTST | 0.370 | 0.332 | 0.177 |
| DLinear | 0.375 | 0.333 | 0.176 |
| Chronos | ~0.42 | ~0.38 | ~0.19 |

**Target**: Beat iTransformer with physics grouping.

---

## Experiment Protocol

### Step 1: C-MAPSS (5 min)
```bash
python train.py --model role_trans --dataset cmapss --groups physics
python train.py --model ci_trans --dataset cmapss  # baseline
# Compare: Transfer ratio
```
✅ Already done: Role-Trans 4.42 vs CI-Trans 6.16

### Step 2: ETT (10 min)
```bash
python train.py --model role_trans --dataset etth1 --groups voltage_levels
python train.py --model itransformer --dataset etth1  # baseline
# Compare: MSE on OT prediction
```

### Step 3: Weather (10 min)
```bash
python train.py --model role_trans --dataset weather --groups physical
python train.py --model itransformer --dataset weather  # baseline
# Compare: MSE on temperature prediction
```

### Step 4: Transfer (Optional, 20 min)
```bash
# ETT transfer
python train.py --model role_trans --train etth1 --test etth2
python train.py --model itransformer --train etth1 --test etth2

# Compare: Does physics grouping help cross-transformer transfer?
```

---

## Success Criteria

| Dataset | Metric | iTransformer | Our Target | Stretch |
|---------|--------|--------------|------------|---------|
| C-MAPSS | Transfer Ratio | 6.16 | <4.5 ✅ | <4.0 |
| ETTh1 | MSE | 0.386 | <0.386 | <0.370 |
| ETTh2 | MSE | 0.340 | <0.340 | <0.332 |
| Weather | MSE | 0.174 | <0.174 | <0.170 |

**Paper threshold**: Beat iTransformer on 2/3 datasets.
**Strong paper**: Beat iTransformer on 3/3 + show transfer improvement.

---

## What We're NOT Doing

- ❌ Electricity (321 consumers, no physics groups)
- ❌ Traffic (human behavior, not physics)
- ❌ M4 (economic data)
- ❌ Complex tiered validation
- ❌ Multiple metrics

**Focus**: One architecture, three physical systems, one question.

---

## Code Structure

```
experiments/
├── train_role_transformer.py    # Our method
├── baselines/
│   ├── itransformer.py          # Main baseline
│   ├── patchtst.py              # Secondary
│   └── dlinear.py               # Sanity check
├── data/
│   ├── cmapss.py                # Loader + groups
│   ├── ett.py                   # Loader + groups
│   └── weather.py               # Loader + groups
└── eval_gift.py                 # GIFT-eval compatible
```

---

## Timeline

| Day | Task | Output |
|-----|------|--------|
| 1 | Implement ETT/Weather loaders with physics groups | Code |
| 1 | Run Role-Trans on ETT | MSE number |
| 1 | Run Role-Trans on Weather | MSE number |
| 2 | Compare to iTransformer | Table |
| 2 | Run transfer experiments | Transfer numbers |
| 3 | Write results section | Draft |

**Total: 3 days to paper-ready validation.**

---

## The Paper Table

| Dataset | Task | iTransformer | PatchTST | **Role-Trans (Ours)** |
|---------|------|--------------|----------|----------------------|
| C-MAPSS | RUL (RMSE) | - | - | **12.17** |
| C-MAPSS | Transfer Ratio | 6.16* | - | **4.42** |
| ETTh1 | Forecast (MSE) | 0.386 | 0.370 | **?** |
| ETTh2 | Forecast (MSE) | 0.340 | 0.332 | **?** |
| Weather | Forecast (MSE) | 0.174 | 0.177 | **?** |

*CI-Trans as iTransformer proxy for C-MAPSS (both channel-attention based)

---

## Key Insight

**iTransformer = dense channel attention (all-to-all)**
**Role-Trans = sparse physics attention (within-component + cross-component)**

If sparse physics structure beats dense learned attention, that's the paper:

> "You don't need to learn channel relationships. Physics tells you which channels are related. Encoding this structure directly outperforms learning it from data."
