# Overnight Autoresearch Prompt V2: Multi-Benchmark Evaluation

You are an autonomous ML researcher. Tonight's goal: establish a unified evaluation framework across 3 benchmark tiers to validate the **role-based physics-aware architecture** for industrial time series transfer.

## Read First
- `autoresearch/RESEARCH_PLAN.md` — overall vision
- `autoresearch/LESSONS_LEARNED.md` — what failed before
- `autoresearch/EXPERIMENT_LOG.md` — ETTh1 results (channel-independence insight)
- `src/industrialjepa/data/ett.py` — existing ETT dataloader
- `src/industrialjepa/data/factorynet.py` — existing FactoryNet loader

## The Core Hypothesis

**Channel-independent processing wins on single-dataset forecasting but cannot learn transferable physics.**

**Role-based processing** (group channels by physical component, share weights within component) should:
1. Match channel-independent performance on single datasets
2. Enable cross-machine transfer (the key claim)

---

## Architecture to Implement

```python
class RoleBasedWorldModel(nn.Module):
    """
    Level 1: Within-Component Encoder (SHARED weights)
    - Groups channels by physical component (e.g., joint, stage, subsystem)
    - Learns how channels within a component relate (physics)
    - Same weights for all components of same type

    Level 2: Cross-Component GNN (topology-aware)
    - Message passing between components
    - Edges = physical connections
    - Can be learned or specified

    Level 3: Temporal Dynamics (SHARED weights)
    - Predicts next system state from current
    - JEPA-style latent prediction OR direct supervision
    """
```

### Channel Role Mapping

For each dataset, define:
```python
ROLE_MAPPING = {
    "channel_name": {
        "role": "input" | "state" | "output",
        "component": "component_id",
        "physical_type": "position" | "velocity" | "force" | "temperature" | "pressure" | ...
    }
}
```

---

## Benchmark Tier 1: ETT (Sanity Check)

**Goal:** Prove role-based architecture doesn't hurt single-dataset performance.

**Datasets:** ETTh1 (already done), ETTh2, ETTm1, ETTm2

**Setup:**
- Use existing `src/industrialjepa/data/ett.py`
- ETT has 7 channels: HUFL, HULL, MUFL, MULL, LUFL, LULL, OT
- These are load measurements from different transformer zones
- Component grouping: HU (high useful), MU (mid useful), LU (low useful), OT (oil temp)

**Role Mapping for ETT:**
```python
ETT_ROLES = {
    "HUFL": {"component": "high", "type": "load"},
    "HULL": {"component": "high", "type": "load"},
    "MUFL": {"component": "mid", "type": "load"},
    "MULL": {"component": "mid", "type": "load"},
    "LUFL": {"component": "low", "type": "load"},
    "LULL": {"component": "low", "type": "load"},
    "OT": {"component": "thermal", "type": "temperature"},
}
```

**Baselines:** Linear, CI-Transformer (from EXPERIMENT_LOG.md), Role-Transformer (new)

**Metrics:** MSE, MAE for H={96, 192, 336, 720}

**Target:** Match CI-Transformer (0.450 MSE at H=96). Not trying to beat PatchTST here.

---

## Benchmark Tier 2: C-MAPSS (Industrial + Transfer)

**Goal:** Beat SOTA on cross-condition transfer. This is the NeurIPS result.

### Dataset Setup

Download from: https://data.nasa.gov/download/ff5v-kuh6/application%2Fzip
Or use: `wget https://ti.arc.nasa.gov/c/6/` (check exact URL)

**Structure:**
- FD001: 100 engines, 1 operating condition, 1 fault mode
- FD002: 260 engines, 6 operating conditions, 1 fault mode
- FD003: 100 engines, 1 operating condition, 2 fault modes
- FD004: 249 engines, 6 operating conditions, 2 fault modes

**Columns (26 total):**
```
unit_id, cycle, setting1, setting2, setting3, s1, s2, ..., s21
```

- Settings (3): operational_setting_1, operational_setting_2, operational_setting_3 → INPUTS
- Sensors (21): s1-s21 → OUTPUTS (various temperatures, pressures, speeds)

**Sensor Physical Meaning:**
```python
CMAPSS_ROLES = {
    "setting1": {"role": "input", "component": "operating", "type": "altitude"},
    "setting2": {"role": "input", "component": "operating", "type": "mach"},
    "setting3": {"role": "input", "component": "operating", "type": "throttle"},
    "s1": {"role": "output", "component": "fan", "type": "temperature"},
    "s2": {"role": "output", "component": "lpc", "type": "temperature"},
    "s3": {"role": "output", "component": "hpc", "type": "temperature"},
    "s4": {"role": "output", "component": "lpt", "type": "temperature"},
    "s5": {"role": "output", "component": "fan", "type": "pressure"},
    "s6": {"role": "output", "component": "bypass", "type": "pressure"},
    "s7": {"role": "output", "component": "hpc", "type": "pressure"},
    "s8": {"role": "output", "component": "fan", "type": "speed_physical"},
    "s9": {"role": "output", "component": "core", "type": "speed_physical"},
    "s10": {"role": "output", "component": "hpc", "type": "pressure"},
    "s11": {"role": "output", "component": "hpc", "type": "temperature_static"},
    "s12": {"role": "output", "component": "fan", "type": "speed_corrected"},
    "s13": {"role": "output", "component": "core", "type": "speed_corrected"},
    "s14": {"role": "output", "component": "bypass", "type": "ratio"},
    "s15": {"role": "output", "component": "bleed", "type": "pressure"},
    "s16": {"role": "output", "component": "fuel", "type": "flow_ratio"},
    "s17": {"role": "output", "component": "bleed", "type": "flow"},
    "s18": {"role": "output", "component": "coolant", "type": "flow"},  # often constant
    "s19": {"role": "output", "component": "coolant", "type": "flow"},  # often constant
    "s20": {"role": "output", "component": "hpc", "type": "pressure"},
    "s21": {"role": "output", "component": "fan", "type": "pressure"},
}
```

### Component Groups (for within-component attention)
```python
CMAPSS_COMPONENTS = {
    "operating": ["setting1", "setting2", "setting3"],
    "fan": ["s1", "s5", "s8", "s12", "s21"],
    "lpc": ["s2"],
    "hpc": ["s3", "s7", "s10", "s11", "s20"],
    "lpt": ["s4"],
    "bypass": ["s6", "s14"],
    "core": ["s9", "s13"],
    "bleed": ["s15", "s17"],
    "fuel": ["s16"],
    "coolant": ["s18", "s19"],  # often constant, may drop
}
```

### Tasks

**Task A: RUL Prediction (standard)**
- Predict remaining useful life from sensor window
- Metric: RMSE (lower is better)
- Use piece-wise linear RUL (cap at 125 cycles)

**Task B: Next-State Prediction (our framing)**
- Predict sensors at t+H from sensors at t-L:t
- This is the "world model" view
- Transfer: model trained on FD001 should predict FD002 dynamics

### Transfer Experiments

| Train | Test | What It Tests |
|-------|------|---------------|
| FD001 | FD001 | Baseline (no transfer) |
| FD001 | FD002 | Transfer across operating conditions |
| FD001 | FD003 | Transfer across fault modes |
| FD001+FD002+FD003 | FD004 | Multi-source transfer |

### Published SOTA (RUL RMSE)

| Model | FD001 | FD002 | FD003 | FD004 | Year |
|-------|-------|-------|-------|-------|------|
| LSTM | 13.6 | 23.1 | 13.0 | 24.0 | 2018 |
| CNN | 12.4 | 22.4 | 12.5 | 22.3 | 2019 |
| Transformer | 11.8 | 19.2 | 11.4 | 20.1 | 2021 |
| DA-RNN | 12.1 | 18.5 | 11.6 | 18.8 | 2022 |
| TFT | 11.2 | 16.8 | 10.9 | 16.5 | 2023 |

**Target:** Beat TFT on FD001→FD002 transfer (train FD001, test FD002).

### Implementation

Create: `autoresearch/experiments/cmapss_baseline.py`
Create: `src/industrialjepa/data/cmapss.py`

---

## Benchmark Tier 3: FactoryNet (Novel - Establish SOTA)

**Goal:** Define the benchmark, run all baselines, establish SOTA ourselves.

### Dataset

Use existing FactoryNet loader: `src/industrialjepa/data/factorynet.py`

**Sources:**
- AURSAD: 2 robots, 6 joints each, ~15 signals per joint
- Voraus: 1 robot, 6 joints, similar signals

**Shared Signal Space (already implemented):**
```
setpoint_pos, setpoint_vel, actual_pos, actual_vel, effort/torque, current
```

### Component Structure
```python
FACTORYNET_COMPONENTS = {
    "joint1": ["joint1_setpoint_pos", "joint1_actual_pos", "joint1_setpoint_vel", "joint1_actual_vel", "joint1_effort"],
    "joint2": ["joint2_setpoint_pos", ...],
    ...
    "joint6": [...],
}
```

This is PERFECT for role-based architecture:
- 6 components (joints)
- Same roles within each joint (setpoint, actual, effort)
- Weights shared across joints (same physics)
- Topology: serial kinematic chain (joint1 → joint2 → ... → joint6)

### Transfer Setup

| Train | Test | Transfer Type |
|-------|------|---------------|
| AURSAD Robot1 | AURSAD Robot1 | None (baseline) |
| AURSAD Robot1 | AURSAD Robot2 | Same dataset, different machine |
| AURSAD (both) | Voraus | Cross-dataset, different machine |

### Baselines to Run

For each setup, run:
1. **Persistence** — predict last value
2. **Linear** — simple linear model
3. **Channel-Independent Transformer** — our ETTh1 winner
4. **PatchTST** — published strong baseline
5. **Role-Based Transformer** — our new architecture

### Metrics

**Forecasting:**
- MSE, MAE for H={10, 25, 50, 100} timesteps

**Transfer Ratio:**
- `transfer_ratio = MSE(transfer) / MSE(from_scratch)`
- <1.0 means transfer helps

### Establishing SOTA Protocol

Since no published results exist for AURSAD→Voraus transfer:

1. **Document the exact setup** (train/val/test splits, normalization, horizons)
2. **Run 5 strong baselines** with 3 seeds each
3. **Report with confidence intervals**
4. **Create reproducible scripts**
5. **This becomes the reference** for future papers

Create: `autoresearch/experiments/factorynet_transfer.py`
Create: `autoresearch/FACTORYNET_BENCHMARK.md` (detailed protocol)

---

## Engineering Rules

1. **Create a unified evaluation script** that runs all benchmarks with consistent protocol
2. **One architecture, three datasets** — same Role-Based model everywhere
3. **5 seeds minimum** for any transfer result
4. **Log everything** to `autoresearch/MULTI_BENCHMARK_LOG.md`
5. **Commit after each completed benchmark tier**
6. **If something fails, log why and move on** — don't spend >30 min debugging

---

## File Structure When Done

```
autoresearch/
  OVERNIGHT_PROMPT_V2.md          (this file)
  MULTI_BENCHMARK_LOG.md          (NEW - all results)
  FACTORYNET_BENCHMARK.md         (NEW - novel benchmark protocol)
  experiments/
    baselines_etth1.py            (existing)
    jepa_etth1.py                 (existing)
    cmapss_baseline.py            (NEW)
    cmapss_transfer.py            (NEW)
    factorynet_transfer.py        (NEW)
    unified_eval.py               (NEW - runs everything)

src/industrialjepa/
  data/
    ett.py                        (existing)
    factorynet.py                 (existing)
    cmapss.py                     (NEW)
  model/
    role_based.py                 (NEW - the architecture)
```

---

## Success Criteria

| Benchmark | Target |
|-----------|--------|
| ETT (H=96) | Role-Based MSE ≤ 0.50 (match CI-Transformer) |
| C-MAPSS FD001→FD002 | Beat TFT transfer RMSE (<16.8) |
| FactoryNet AURSAD→Voraus | Transfer ratio <1.5 with Role-Based |

---

## Iteration Strategy

If you finish early:

1. **Ablations on C-MAPSS:**
   - Remove within-component attention → does it hurt?
   - Remove cross-component GNN → does topology matter?
   - Vary component groupings

2. **Scale FactoryNet:**
   - Add more signals (gripper, accelerometer if available)
   - Test different kinematic chain topologies

3. **Improve Role-Based architecture:**
   - Try xLSTM as temporal predictor
   - Add RevIN for better normalization
   - Experiment with JEPA pretraining

---

## Output Format

### Multi-Benchmark Log (`autoresearch/MULTI_BENCHMARK_LOG.md`)

```markdown
# Multi-Benchmark Evaluation Log

## Summary Table

| Benchmark | Task | Model | Result | SOTA | Gap |
|-----------|------|-------|--------|------|-----|
| ETTh1 H=96 | Forecast | Role-Based | X.XX | 0.370 | +Y% |
| C-MAPSS FD001→FD002 | RUL Transfer | Role-Based | X.XX | 16.8 | ... |
| FactoryNet AURSAD→Voraus | Forecast Transfer | Role-Based | X.XX | (ours) | N/A |

## Detailed Results

### Tier 1: ETT
[tables]

### Tier 2: C-MAPSS
[tables]

### Tier 3: FactoryNet
[tables]

## Architecture Ablations
[what worked, what didn't]

## Honest Assessment
[transfer story holds up? what's the NeurIPS claim?]
```

---

## When You're Done

1. Update `MULTI_BENCHMARK_LOG.md` with all results
2. If C-MAPSS transfer beats SOTA: this is the headline result
3. If FactoryNet transfer works: this is the novel contribution
4. If both work: strong NeurIPS submission
5. If neither works: document why, propose next steps

**Be brutally honest.** If the role-based architecture doesn't enable transfer, say so clearly.
