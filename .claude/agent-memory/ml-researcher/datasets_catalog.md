---
name: Dataset Catalog and Recommendations
description: All datasets evaluated for IndustrialJEPA, with Tier 2 recommendation (Paderborn) and Brain-JEPA analysis
type: project
---

Complete datasets infrastructure built at `/home/sagemaker-user/IndustrialJEPA/datasets/` on 2026-03-26.

## Rapid Evaluation Suite Tiers

| Tier | Dataset | Status | Key Result |
|---|---|---|---|
| 1 | Double Pendulum (synthetic) | CONFIRMED | PhysMask +7.4% (p=0.0002) |
| 2 | **Paderborn Bearing** | **RECOMMENDED** | Real mech, 4 physics groups, no forecasting SOTA |
| 2-fallback | UCI Hydraulic | Available | 17 sensors, 4 groups, 2205 cycles |
| 3 | ETT | CONFIRMED | PhysMask -1.3% (complex coupling) |

## Tier 2 Decision: Paderborn Bearing
- Real laboratory data (motor drive test rig), 33 bearing conditions
- 7 channels: vibration_1, phase_current_1, phase_current_2, force, speed, torque, temp
- 4 physics groups: vibration / current / mechanical / thermal
- Downloaded sample (K001 healthy, KA01 outer race, KI01 inner race) to `datasets/data/paderborn/`
- No published forecasting SOTA — must define own baselines
- Transfer test: K001 → KA01 (healthy → outer race fault)

## What NOT to Use (and why)
- **CWRU**: 2–4 channels, all vibration — insufficient for attention masking
- **XJTU-SY**: only 2 channels (X, Y) — useless
- **Open X-Embodiment**: camera-centric, 7–20 channels vs 450 needed, TFRecord format
- **C-MAPSS as Tier 2**: synthetic; physics mask ≈ random (Exp 48)

## Brain-JEPA Scale (Direction 3)
Brain-JEPA: 32k patients × 160 timesteps × 450 ROIs.
- OXE is NOT viable (7–20 channels, camera-centric, RLDS/TFRecord format)
- **Best available**: WADI (127 channels × 1.2M timesteps) — requires SUTD data request
- **Next best**: SWaT (51 channels × 946k timesteps) — same request process
- **Alternative**: Electricity UCI (321 channels × 26k timesteps), Traffic (862 × 17k)
- Request form: https://itrust.sutd.edu.sg/testbeds/secure-water-treatment-swat/

## Downloaded Data (in datasets/data/, gitignored)
- `ett/` — ETTh1, ETTh2, ETTm1, ETTm2 (direct CSV from GitHub)
- `hydraulic/` — all 17 sensors + labels (~73 MB UCI download)
- `cwru/` — 4 sample files (normal, IR, ball, OR faults)
- `paderborn/` — K001, KA01, KI01 sample (3 × ~80 .mat files)

## Generated Figures (committed at datasets/analysis/figures/)
- `ett_overview.png` — ETTh1 full series, correlations, physics groups, SOTA comparison
- `hydraulic_overview.png` — 17 sensors, fault distributions, cycle analysis
- `cwru_overview.png` — FFT comparison, fault types, channel count reality check
- `paderborn_overview.png` — vibration+current+mechanical comparison across conditions

**Why:** Tier 2 was the major unresolved bottleneck before 2026-03-26.
**How to apply:** When discussing dataset selection or paper experiments, use Paderborn as Tier 2.
