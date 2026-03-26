# IndustrialJEPA Dataset Catalog

**Last updated**: 2026-03-26

This directory contains the complete dataset infrastructure for the IndustrialJEPA project — a research project on physics-informed attention for multivariate time series forecasting on mechanical/industrial systems.

---

## Directory Structure

```
datasets/
├── README.md                    # This file — master catalog and recommendations
├── catalog/                     # Detailed dataset profiles (one per dataset)
│   ├── cwru_bearing.md
│   ├── paderborn_bearing.md
│   ├── ims_bearing.md
│   ├── xjtu_sy_bearing.md
│   ├── cmapss.md
│   ├── ett.md
│   ├── hydraulic_system.md
│   ├── swat_wadi.md
│   ├── aursad.md
│   ├── voraus_ad.md
│   ├── tennessee_eastman.md
│   ├── monash_archive.md
│   └── open_x_embodiment.md
├── downloaders/                 # Download scripts (one per dataset)
│   ├── download_cwru.py
│   ├── download_paderborn.py
│   ├── download_ett.py
│   ├── download_hydraulic.py
│   ├── download_cmapss.py
│   ├── download_swat.py
│   └── download_open_x.py
├── analysis/                    # EDA scripts
│   ├── analyze_cwru.py
│   ├── analyze_ett.py
│   ├── analyze_hydraulic.py
│   └── figures/                 # Generated PNGs (committed)
│       ├── cwru_overview.png
│       ├── ett_overview.png
│       └── hydraulic_overview.png
└── data/                        # Downloaded raw data (gitignored)
```

---

## Quick Dataset Comparison

| Dataset | Domain | Real? | Channels | Timesteps | Physics Groups | Tier |
|---|---|---|---|---|---|---|
| Double Pendulum | Synthetic | No | 4 | 10,000 | 2 (mass_1, mass_2) | 1 |
| **Paderborn Bearing** | Mechanical | **Yes** | **8** | 256k/file, 33 bearings | **4 (vibration/thermal/current)** | **2 candidate** |
| **Hydraulic System** | Industrial | **Yes** | **17** | 2205×60s | **4 (pressure/flow/thermal/mech)** | **2 fallback** |
| ETT | Power systems | Yes | 7 | 17,420–69,680 | 4 (HV/MV/LV/thermal) | 3 |
| C-MAPSS | Turbofan (sim) | No | 21 (7 useful) | 436–874 units | 4 (temp/pres/speed/flow) | 2-synthetic |
| CWRU Bearing | Mechanical | Yes | 2–4 | 20k–244k per file | 1 (vibration only) | ablation |
| IMS Bearing | Mechanical | Yes | 8 (4–8) | 20k/file × 4k files | 4 (bearing × axis) | ablation |
| XJTU-SY Bearing | Mechanical | Yes | 2 | 32k/file × 15 bearings | 1 (X/Y) | not useful |
| Tennessee Eastman | Chemical (sim) | No | 52 | 500/run × 44 runs | 6 (process stages) | 2-synthetic alt |
| SWaT | Water treatment | Yes | 51 | 946,000 | 6 (P1–P6 stages) | pretraining |
| WADI | Water dist. | Yes | 127 | 1,209,600 | 3 (subsystems) | pretraining |
| AURSAD | Robot (UR3e) | Yes | 20 | ~1500/ep × 4094 ep | 4 (pos/vel/current/cartesian) | current |
| Voraus-AD | Robot (Cobot) | Yes | 66 | ~2000/ep × 2000 ep | 8 (joint modalities) | current |
| Open X-Embodiment | Robotics | Yes | 7–20 | 50–500/traj × 1M traj | varies | not viable |

---

## Rapid Evaluation Suite Recommendations

### Tier 1: Double Pendulum (CONFIRMED)
- Synthetic, 4 channels, 2 physics groups
- Perfect ground truth: physics mask beats full-attention by 7.4% (p=0.0002, Exp 47)

### Tier 2: RECOMMENDATION — Paderborn Bearing

**Primary recommendation: Paderborn Bearing**

Evidence:
- 8 channels with 4 clear physical modalities: vibration (radial + tangential + axial + velocity), thermal (temperature + torque), electrical (2-phase motor current)
- The voltage/current/vibration combination directly tests whether physics grouping helps on multi-modal mechanical data
- Real damage (both artificial EDM and natural progressive wear)
- No registration required for download
- 33 bearing conditions (healthy, outer race, inner race, combined damage)
- 20 measurements per bearing × 4 operating conditions = 80 files per bearing
- Transfer scenarios: healthy→faulty, artificial→natural damage, cross operating condition

**Why Paderborn over CWRU**: CWRU has 2–4 channels of the same modality (vibration only); Paderborn has 4 distinct modalities. Physics-informed grouping is only meaningful when groups represent different physical phenomena.

**The only gap**: No published MSE-based forecasting SOTA exists for Paderborn. This means we must establish our own baselines (persistence, linear, CI-Transformer, Full-Attention). This is acceptable for a paper that is primarily about the architecture, not about matching an external leaderboard.

**Fallback: Hydraulic System (UCI)**
- 17 sensors, 4 physics groups (pressure/flow/thermal/mechanical)
- 2,205 cycles, direct download (no registration)
- Clean labels for 4 fault types
- Downside: Mixed sampling rates (100/10/1 Hz) require careful preprocessing

### Tier 3: ETT (CONFIRMED)
- Real electricity transformer data, 7 channels, 4 groups
- Key result: PhysMask HURTS (-1.3%) because thermal couples to all load groups
- This is the "complex interactions" case in the paper narrative
- Transfer test: ETTh1 → ETTh2

---

## Paper Dataset Recommendations (3–4 strongest)

| Dataset | Role in Paper | SOTA to Beat | Transfer Test |
|---|---|---|---|
| Double Pendulum | "Independent system" — physics mask clearly wins | Internal baselines | m1/m2=1.0 → 0.5 |
| Paderborn Bearing | "Mechanical real data" — multi-modal physics | Internal baselines (no forecasting SOTA) | KA01 → KI01 (outer→inner fault) |
| C-MAPSS | "Correlated system" — physics ≈ random (honest negative) | Internal baselines (sensor forecasting) | FD001 → FD003 |
| ETT | "Complex interactions" — physics hurts | PatchTST (~0.37), iTransformer (~0.39) | ETTh1 → ETTh2 |

This 4-dataset narrative tells a complete scientific story:

1. **When physics masking helps strongly**: Independent groups (pendulum, idealized)
2. **When physics masking helps moderately**: Semi-independent mechanical groups (Paderborn)
3. **When physics masking doesn't help**: Correlated degradation (C-MAPSS)
4. **When physics masking hurts**: Tight cross-group coupling (ETT)

**Paper title fit**: "When to Mask: Physics-Informed Attention Depends on Group Independence"

---

## Brain-JEPA Analog Analysis (Direction 3)

Brain-JEPA scale: **32,000 patients × 160 timesteps × 450 ROIs**

The key dimension is **450 parallel channels** — this enables attention across "channels" (ROIs) to be the core computation. Our physics-informed grouping would operate on these channels.

### Available Datasets by Channel Count

| Dataset | Channels | Timesteps | Total "Tokens" | Access |
|---|---|---|---|---|
| WADI (water dist.) | 127 | 1,209,600 | 153M | Request (1 week) |
| SWaT (water treat.) | 51 | 946,000 | 48M | Request (1 week) |
| Electricity (UCI) | 321 | 26,304 | 8.4M | Direct |
| Traffic (SF) | 862 | 17,544 | 15.1M | Direct |
| Open X-Embodiment | 7–20 | M+ traj | — | Direct (TFRecord) |

### Verdict: Open X-Embodiment is NOT Viable for Brain-JEPA Analog

Critical analysis:
1. **Channel count**: OXE has 7–20 proprioceptive channels per dataset. Brain-JEPA has 450 ROIs. A 20–60x gap means we cannot run the same "many-channel physics attention" experiment at scale.
2. **Modality mismatch**: OXE is primarily a vision dataset (RGB cameras). The proprioceptive signals are secondary. Most published OXE papers use image+action, not pure sensor time series.
3. **Format overhead**: RLDS/TFRecord requires TensorFlow; incompatible with our PyTorch pipeline. Conversion to numpy is ~5x the storage.
4. **Heterogeneous channels**: Different OXE sub-datasets have incompatible sensor spaces. No "universal 20-channel" format exists across the full 60 datasets.

### Best Available Analog: WADI (127 channels, 1.2M timesteps)

WADI gives us:
- **127 channels** (still 3.5x fewer than Brain-JEPA's 450, but closest available)
- **Continuous operation** (16 days at 1 Hz) — perfect for causal sliding windows
- **3 physical subsystems** (chemical treatment, storage, distribution) for physics grouping
- **Published anomaly detection SOTA** for evaluation comparison
- **Requires data request** to SUTD (~1 week turnaround)

### Realistic Brain-JEPA Analog Strategy

Given that no single dataset reaches 450 channels, two options:
1. **Use WADI (127 ch)**: Demonstrate the approach scales to 127 channels; argue this is sufficient for the paper's claims
2. **Stack multiple datasets**: Concatenate Paderborn (8 ch) + Hydraulic (17 ch) + C-MAPSS (21 ch) + AURSAD (20 ch) + ETT (7 ch) = 73 channels total — but channels are physically incoherent across datasets

Recommendation: **WADI is the realistic target**. Request access immediately if Direction 3 (Brain-JEPA) is the priority.

---

## Download Instructions

### Fast (no registration)

```bash
# ETT (all 4 variants, ~25 MB total)
python datasets/downloaders/download_ett.py

# UCI Hydraulic (~73 MB)
python datasets/downloaders/download_hydraulic.py

# CWRU sample (4 files, ~12 MB)
python datasets/downloaders/download_cwru.py --sample

# Paderborn sample (3 bearings, ~480 MB, requires unrar)
python datasets/downloaders/download_paderborn.py --sample

# C-MAPSS (via kaggle API or NASA)
python datasets/downloaders/download_cmapss.py
```

### Requires Registration

```bash
# SWaT/WADI: Submit request at https://itrust.sutd.edu.sg/
python datasets/downloaders/download_swat.py --instructions
```

### Analysis

```bash
# Generate overview figures (run after downloading)
python datasets/analysis/analyze_ett.py
python datasets/analysis/analyze_hydraulic.py
python datasets/analysis/analyze_cwru.py
```

---

## URL Verification Status (2026-03-26)

| Dataset | URL | Status |
|---|---|---|
| ETT | github.com/zhouhaoyi/ETDataset | VERIFIED |
| UCI Hydraulic | archive.ics.uci.edu/dataset/447 | VERIFIED |
| CWRU Bearing | engineering.case.edu/bearingdatacenter | VERIFIED (numeric IDs) |
| Paderborn | groups.uni-paderborn.de/kat/BearingDataCenter/ | VERIFIED (directory listing) |
| C-MAPSS (Kaggle) | kaggle.com/datasets/behrad3d/nasa-cmaps | ACCESSIBLE (requires kaggle API) |
| SWaT | itrust.sutd.edu.sg (data request page) | BEHIND REQUEST WALL |
| AURSAD | zenodo.org/record/4905920 | ACCESSIBLE |
| Voraus-AD | github.com/voraus-io/voraus-AD-dataset | ACCESSIBLE |
| Open X-Embodiment | github.com/google-deepmind/open_x_embodiment | ACCESSIBLE (TFRecord format) |
| XJTU-SY | ieee-dataport.org (requires IEEE account) | BEHIND FREE ACCOUNT |
| Tennessee Eastman | kaggle.com/datasets/averkij/tennessee-eastman-process-simulation-dataset | ACCESSIBLE |

---

## Key Experimental Findings (from 48 experiments)

These findings directly inform dataset selection:

| Finding | Implication for Dataset Choice |
|---|---|
| Pendulum: physics mask +7.4% over full-attn (p<0.001) | Need a real-data analog of "independent groups" |
| C-MAPSS: physics mask ≈ random (p=0.528) | Correlated degradation breaks physics masking |
| ETT: physics mask -1.3% vs full-attn | Tight coupling breaks physics masking |
| Wrong grouping 23x worse than correct (Exp 45) | Physics masking quality depends critically on correct groups |
| All 3 tiers: physics > CI-Transformer (5–34%) | "2D treatment" always helps; masking quality depends on independence |

Paderborn is chosen as Tier 2 precisely because it sits between the pendulum (independent) and C-MAPSS (correlated): the electrical/mechanical/thermal groups in Paderborn are partially independent, predicting a moderate physics mask benefit.
