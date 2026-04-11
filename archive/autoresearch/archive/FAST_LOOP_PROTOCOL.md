# Fast Loop Protocol: Physical Dynamical Systems Only

**Goal**: Maximum iteration speed while validating on physically meaningful benchmarks.

---

## Philosophy

```
10-minute experiments → insights → iterate → scale up winners
```

**NOT**: Run everything, wait hours, analyze.

---

## Fast Loop Tiers

### Tier 0: Smoke Test (30 seconds)
```python
# Single seed, 10 epochs, tiny subset
python train.py --dataset cmapss_fd001 --subset 0.1 --epochs 10 --seed 42
```

**Pass criterion**: Loss decreases, no NaN, output shape correct.

### Tier 1: Quick Validation (5 minutes)
```python
# Single seed, full data, 30 epochs
python train.py --dataset cmapss_fd001 --epochs 30 --seed 42
```

| Dataset | Samples | Sensors | Time |
|---------|---------|---------|------|
| C-MAPSS FD001 | 17,731 windows | 14 | ~3 min |
| C-MAPSS FD001→FD002 | Same + eval | 14 | ~5 min |

**Pass criterion**: FD001 RMSE < 14, Transfer ratio < 5.0

### Tier 2: Statistical Validation (30 minutes)
```python
# 3 seeds, full training
for seed in [42, 123, 456]:
    python train.py --dataset cmapss_fd001 --epochs 60 --seed $seed
```

**Pass criterion**: Mean ± std, p-value vs baseline

### Tier 3: Cross-Domain (2 hours)
```python
# Add CWRU, check generalization
python train.py --dataset cwru --cross_domain nasa_bearing
```

---

## Physical Dynamical Systems Only

### What Qualifies as "Physical"

A dataset is physical/dynamical if it measures:
1. **Energy flow** (electrical, thermal, mechanical)
2. **Mass/momentum transport** (fluid, traffic flow)
3. **Oscillatory dynamics** (vibration, waves)
4. **Degradation physics** (wear, fatigue)

### GIFT-Eval: Physical Subset

| Dataset | Domain | Physical? | Why |
|---------|--------|-----------|-----|
| **ETTh1, ETTh2** | Electricity | ✅ | Transformer thermal dynamics |
| **ETTm1, ETTm2** | Electricity | ✅ | Same, higher frequency |
| **Electricity** | Power grid | ✅ | Load dynamics, 321 consumers |
| **Solar** | Energy | ✅ | PV generation, weather-coupled |
| **Weather** | Climate | ✅ | Atmospheric physics, 21 vars |
| Traffic | Transport | ⚠️ | Flow dynamics, but human-driven |
| M4_* | Mixed | ❌ | Mostly economic/retail |
| Hospital | Medical | ❌ | Patient counts |
| Restaurant | Demand | ❌ | Human behavior |
| Wiki | Web | ❌ | Page views |

### Minimal Physical Benchmark (GIFT-Eval-Phys)

| Dataset | Sensors | Frequency | Samples | Download |
|---------|---------|-----------|---------|----------|
| ETTh1 | 7 | Hourly | 17,420 | [HuggingFace](https://huggingface.co/datasets/Salesforce/GiftEval) |
| ETTm1 | 7 | 15-min | 69,680 | Same |
| Electricity | 321 | Hourly | 26,304 | Same |
| Weather | 21 | 10-min | 52,696 | Same |
| Solar | 137 | 10-min | 52,560 | Same |

**Total**: 5 datasets, all physical dynamics.

---

## Mechanical Systems Benchmark (Our Core)

Even faster than GIFT-Eval-Phys:

| Dataset | Samples | Sensors | Time per run | Focus |
|---------|---------|---------|--------------|-------|
| **C-MAPSS FD001** | 17.7k | 14 | 3 min | In-domain |
| **C-MAPSS FD002** | 48.8k | 14 | +2 min | Transfer target |
| **CWRU Bearing** | ~2k | 2-4 | 1 min | Fault diagnosis |
| **NRI Springs (synthetic)** | Generate | N | 30 sec | Graph ground truth |

**Total loop time**: <10 minutes for full mechanical benchmark.

---

## Fast Loop Decision Tree

```
Start
  │
  ▼
[Tier 0: Smoke test] ──fail──> Debug code
  │ pass
  ▼
[Tier 1: FD001 quick] ──RMSE>14──> Bad idea, try next
  │ RMSE<14
  ▼
[Tier 1: Transfer quick] ──ratio>5──> Doesn't transfer, iterate
  │ ratio<5
  ▼
[Tier 2: 3-seed stats] ──p>0.1──> Not robust, iterate
  │ p<0.1
  ▼
[Tier 3: CWRU+GIFT-Phys] ──fails──> Doesn't generalize
  │ passes
  ▼
[Full Validation] → Paper
```

---

## Concrete Fast Loop for Each Direction

### Direction 1: Sparse Graph

```bash
# Tier 0 (30s)
python graph_learn.py --smoke_test

# Tier 1 (5min)
python graph_learn.py --dataset fd001 --sparsity 0.3 --seed 42
# Check: Does learned graph match component structure? (ARI > 0.3)

# Tier 1b: Transfer (5min)
python graph_learn.py --train fd001 --test fd002
# Check: Transfer ratio < 5.0?

# If both pass → Tier 2
```

**Key fast metric**: Adjusted Rand Index of learned graph vs physics groups.

### Direction 2: Slot Concepts

```bash
# Tier 0 (30s)
python slot_concept.py --smoke_test

# Tier 1 (5min)
python slot_concept.py --dataset fd001 --n_slots 5 --encoder per_channel
# Check: Slot entropy < 1.2? (uniform = 1.61)

# Tier 1b: Transfer (5min)
python slot_concept.py --train fd001 --test fd002
# Check: Transfer ratio < 5.0?

# If both pass → Tier 2
```

**Key fast metric**: Slot assignment entropy (should drop from uniform 1.61).

### Direction 3: Mechanical-JEPA

```bash
# Tier 0 (30s)
python mech_jepa.py --smoke_test

# Tier 1 (5min)
python mech_jepa.py --dataset fd001 --mask_ratio 0.8 --codebook_size 64
# Check: Latent entropy > 1.5? (collapsed < 0.5)

# Tier 1b: Transfer (5min)
python mech_jepa.py --pretrain fd001 --finetune fd002 --labels 0.05
# Check: Beats scratch baseline?

# If both pass → Tier 2
```

**Key fast metric**: Latent codebook entropy (should stay high, not collapse).

---

## Unified Evaluation Script

```python
#!/usr/bin/env python
"""Fast loop evaluation for all 3 directions."""

import time
from dataclasses import dataclass

@dataclass
class FastLoopResult:
    direction: str
    tier: int
    metric: str
    value: float
    passed: bool
    time_seconds: float

def fast_loop(direction: str, tier: int = 1) -> FastLoopResult:
    start = time.time()

    if direction == "graph":
        # Train graph learner
        model = train_graph_learner(dataset="fd001", epochs=30 if tier >= 1 else 10)
        metric = compute_ari(model.learned_graph, PHYSICS_GROUPS)
        passed = metric > 0.3

    elif direction == "slots":
        # Train slot model
        model = train_slot_concept(dataset="fd001", n_slots=5, encoder="per_channel")
        metric = compute_slot_entropy(model)
        passed = metric < 1.2  # Lower than uniform (1.61)

    elif direction == "jepa":
        # Train JEPA
        model = train_mech_jepa(dataset="fd001", mask_ratio=0.8, codebook_size=64)
        metric = compute_latent_entropy(model)
        passed = metric > 1.5  # Not collapsed

    return FastLoopResult(
        direction=direction,
        tier=tier,
        metric=metric,
        passed=passed,
        time_seconds=time.time() - start
    )

# Run all 3 in parallel
from concurrent.futures import ProcessPoolExecutor
with ProcessPoolExecutor(max_workers=3) as ex:
    results = list(ex.map(fast_loop, ["graph", "slots", "jepa"]))
```

---

## Summary: Fastest Path

| Phase | Time | What |
|-------|------|------|
| **Tier 0** | 2 min | All 3 directions smoke test |
| **Tier 1** | 15 min | All 3 on FD001 + transfer |
| **Decision** | - | Pick winner(s) |
| **Tier 2** | 30 min | 3-seed validation on winner |
| **Tier 3** | 2 hr | CWRU + GIFT-Eval-Phys |

**Total to paper-ready results**: ~3 hours if one direction wins cleanly.

---

## GIFT-Eval-Phys Quick Start

```bash
# Install
pip install gift-eval

# Download physical subset only
python -c "
from gift_eval import download
for ds in ['ETTh1', 'ETTm1', 'Electricity', 'Weather', 'Solar']:
    download(ds)
"

# Evaluate
python eval_gift_phys.py --model my_model --datasets physical
```

**Expected metrics** (to beat):

| Dataset | Chronos | Moirai | TimesFM | Our Target |
|---------|---------|--------|---------|------------|
| ETTh1 | 0.42 | 0.41 | 0.43 | <0.42 |
| Weather | 0.65 | 0.63 | 0.67 | <0.65 |
| Electricity | 0.85 | 0.82 | 0.88 | <0.85 |

---

## Key Insight

**Don't run GIFT-Eval until Tier 2 passes.**

GIFT-Eval is for comparing to foundation models. If we can't beat CI-Transformer on C-MAPSS transfer, we won't beat Chronos on Weather.

The fast loop is:
1. C-MAPSS transfer (5 min) → validates physics transfer
2. Novel metric (5 min) → validates our contribution (graph/slots/codebook)
3. Scale up only winners
