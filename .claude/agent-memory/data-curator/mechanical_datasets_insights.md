# Mechanical Vibration Datasets - Key Insights

**Last Updated:** 2026-04-01

## Critical Finding: Action-Conditioning Gap

After analyzing 10 public mechanical vibration datasets, **NONE provide true action-conditioned data** with:
- Commanded control setpoints (target RPM/load)
- Control transitions within episodes
- Continuous degradation labels
- Long temporal episodes with varying actions

## Only 2 Datasets Have Transitions

### MCC5-THU Gearbox ✓
- **BEST for action-conditioning**
- Programmed speed sequences within 60-sec windows
- Example: 0→2500→3000 RPM with timed transitions
- 12 working conditions with time-varying speed/load
- **Limitations:** No timestamps in CSV, discrete faults, no degradation tracking

### Mendeley Varying Speed Bearing ✓
- **GOOD for action-conditioning**
- Speed ramps: increasing, decreasing, up-down, down-up patterns
- Range: 680-2460 RPM
- **Limitations:** Binary faults only, no degradation, multi-modal (must extract vibration)

## Datasets by Use Case

### For Run-to-Failure / RUL Prediction:
1. **IMS/NASA** - 10-min snapshots, timestamps, BUT constant 2000 RPM
2. **XJTU-SY** - 1-min intervals, 15 bearings, BUT 3 fixed conditions
3. **FEMTO** - Continuous monitoring, 17 bearings, BUT 3 fixed conditions

All have excellent temporal structure but NO control variation within episodes.

### For Multi-Condition Learning:
1. **Paderborn** - 4 operating conditions, 32 damage states
2. **PHM 2009** - 5 speeds × 2 loads, 45-dim continuous severity labels
3. **XJTU-SY** - 3 speed/load combinations
4. **FEMTO** - 3 speed/load combinations

All steady-state snapshots, no transitions.

### For Benchmark Testing:
1. **CWRU** - Most cited, but limited: steady-state only, binary faults
2. **MFPT** - Smallest (100 MB), good for quick tests

## Data Quality Issues Discovered

### Missing Timestamps:
- **MCC5-THU:** CSV without time column (must infer from sample count)
- **Paderborn:** Only run number, no temporal info
- **PHM 2009:** No timestamps
- **CWRU/MFPT:** No timestamps

### Measured vs Commanded Values:
- **ALL datasets only provide MEASURED values** (actual RPM/load)
- NONE provide commanded setpoints (target RPM/load)
- Cannot distinguish control signal from system response

### Fault Severity Labels:
- **Binary only:** MFPT, CWRU, OEDI, Mendeley
- **Discrete states:** Paderborn (32 states), MCC5-THU
- **Continuous implicit:** IMS/NASA, XJTU-SY, FEMTO (researchers derive post-hoc)
- **Continuous explicit:** PHM 2009 only (45-dim multi-label)

## Recommended Approach for Industrial-JEPA

### Short-term (Use existing data):
1. **MCC5-THU** - Primary for action-conditioning with speed transitions
2. **Mendeley** - Secondary for speed ramp patterns
3. **XJTU-SY** - For temporal prediction and cross-condition learning
4. **Paderborn/PHM 2009** - For multi-condition fault classification

### Medium-term (Data augmentation):
- Synthetically vary playback speed on run-to-failure datasets
- Add simulated control signals to temporal datasets
- Use operating condition as discrete action token

### Long-term (New data collection):
- Purpose-built dataset with commanded control signals
- Continuous control variations during run-to-failure
- Explicit degradation labels
- OR use simulation environment (e.g., MATLAB Simulink bearing models)

## Dataset Processing Notes

### File Formats:
- **MATLAB .mat:** MFPT, CWRU, OEDI, Paderborn
- **CSV:** XJTU-SY, MCC5-THU, FEMTO
- **Text:** IMS/NASA, PHM 2009
- **Mixed:** Mendeley (CSV/MAT varies by subset)

### Size Planning:
- **Quick tests:** MFPT (100 MB), CWRU (500 MB), OEDI (500 MB)
- **Medium:** PHM 2009 (1 GB), FEMTO (2 GB), MCC5-THU (2 GB)
- **Large:** Mendeley (3 GB), XJTU-SY (5 GB), IMS/NASA (6 GB)
- **Very large:** Paderborn (20 GB - process in batches!)

### Sampling Rates:
- **Low (12-20 kHz):** CWRU, XJTU-SY, MCC5-THU, IMS/NASA
- **Medium (25-48 kHz):** XJTU-SY, MFPT, OEDI
- **High (64 kHz):** Paderborn

## Common Gotchas

1. **CWRU RPM is inferred from load**, not directly measured
2. **XJTU-SY CSV has no timestamp column** - use file sequence + 1-min intervals
3. **MCC5-THU has speed transitions BUT no timestamps** - infer from 12.8 kHz sampling
4. **Paderborn has 20 measurements per condition** - these are REPETITIONS not aging sequence
5. **IMS/NASA has gaps in timestamps** - experiment paused overnight
6. **FEMTO stops when vibration > 20g** - lifetime varies 28 min to 7 hours
7. **Mendeley has 3 subsets** - format varies, needs inspection
8. **OEDI is 1-min segments** - limited temporal context
9. **PHM 2009 has 45-dimensional labels** - complex multi-label format
10. **ALL datasets use measured values only** - no commanded control signals

## Research Gaps Identified

1. **No dataset with commanded control signals** (all measure actual values)
2. **No dataset with continuous degradation labels** (all binary/discrete or implicit)
3. **Only 2 datasets with speed transitions** (MCC5-THU, Mendeley)
4. **No dataset combining transitions + degradation + long episodes**
5. **Limited documentation of control strategies used** in experiments

## Citation Tracking

All 10 datasets have proper citations documented in:
`C:/Users/Jonaspetersen/dev/IndustrialJEPA/mechanical-datasets/datasets_inventory.md`

---

**Next Actions:**
1. Download and inspect MCC5-THU (highest priority for action-conditioning)
2. Download Mendeley subset 1 (verify speed ramp format)
3. Download XJTU-SY (best for temporal prediction with 3 conditions)
4. Develop synthetic augmentation pipeline for adding control signals to other datasets
