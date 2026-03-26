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
| Open X-Embodiment | Robotics | Yes | 7–27 | 50–1160/traj × 123k traj | varies | **conditionally viable** |

---

## Research Directions: Evaluation Suites

This section provides **rapid evaluation** (quick dev tests) and **full-scale benchmarks** (paper-ready) for the three IndustrialJEPA research directions.

---

## Direction 1: Sparse Graphs

Physics-informed attention can be viewed as sparse graph attention, where nodes are sensor channels and edges represent physical coupling.

### Rapid Evaluation Suite (Sparse Graphs)

| Dataset | Purpose | Test | Time |
|---|---|---|---|
| Double Pendulum | Unit test graph structure | Physics mask vs random on known 4-node graph | ~10 min |
| Paderborn Bearing (8ch subset) | Real multi-modal sparse graph | 4-group bipartite (vib/thermal/elec/mech) | ~30 min |

### Full-Scale Benchmarks (Sparse Graphs)

| Dataset | Paper Role | Benchmark | Expected Result |
|---|---|---|---|
| **Paderborn Bearing** | Primary mechanical | Graph attention on 8 channels, cross-condition transfer (healthy→faulty) | Physics mask +3-5% |
| **Hydraulic System** | Industrial process | 17-channel causal graph (pressure→flow→thermal), anomaly detection | Physics mask +2-4% |
| **AURSAD** | Robotics graph | 20-channel hierarchical (cmd→current→joint state), fault classification | Physics mask +1-3% |

**Key insight**: Paderborn's 4-group structure (vibration/thermal/electrical/mechanical) is ideal—groups are weakly coupled except at failure modes, creating naturally sparse inter-group edges.

---

## Direction 2: Meta-Feature Prediction

Beyond energy consumption, several high-value meta-features can be predicted from JEPA latent representations for anomaly detection and predictive maintenance.

### High-Value Meta-Features

| Feature | Physical Meaning | Anomaly Signal | Best Dataset |
|---|---|---|---|
| **Effort Variance** | Motor current σ | Stress/friction increase | AURSAD |
| **Coupling Strength** | Cross-channel correlation | Normal: 0.3-0.5, Fault: 0.6-0.9 | Paderborn |
| **Spectral Energy Ratio** | High-freq / low-freq power | Fault → increased HF | Paderborn, CWRU |
| **Phase Delay** | Command→response lag | Mechanical binding | AURSAD |
| **Degradation Rate** | dRUL/dt in latent space | Predictive maintenance | C-MAPSS |
| **Thermal Gradient** | ΔT across components | Overheating precursor | ETT, Paderborn |

### Rapid Evaluation Suite (Meta-Features)

| Test | Dataset | Metric | Baseline |
|---|---|---|---|
| Variance spike detection | AURSAD (current channels) | F1 on fault class | Threshold detector |
| Correlation change | Paderborn (cross-modality) | AUC-ROC | PCA residual |

### Full-Scale Benchmarks (Meta-Features)

| Dataset | Meta-Feature Suite | Paper Claim |
|---|---|---|
| **AURSAD** | Current variance + coupling + phase delay | "Multi-feature anomaly score outperforms single-channel by X%" |
| **Paderborn** | Spectral energy + thermal gradient + coupling | "Cross-modality features detect faults 2-3 timesteps earlier" |
| **C-MAPSS** | Degradation rate in latent space | "JEPA latent RUL estimation competitive with supervised" |

**Novel angle**: Meta-features extracted from JEPA latent space (not raw sensor space) could show better generalization—the latent compresses physics-relevant structure.

---

## Direction 3: Mechanical-JEPA (Cross-Embodiment Robotics)

OXE is **conditionally viable** for Mechanical-JEPA. While channel count (7-27) is far below Brain-JEPA (450 ROIs), OXE enables a distinct contribution: **cross-embodiment proprioceptive transfer learning**.

### Rapid Evaluation Suite (Mechanical-JEPA)

| Phase | Data | Test | Time |
|---|---|---|---|
| Sanity check | TOTO (902 ep, 20 MB) | Overfit single-embodiment prediction | ~1 hr |
| Action correlation | ManiSkill (sim) | Verify state-action coupling (r=0.47) | ~2 hr |
| DOF transfer | Franka→UR5 (zero-shot) | Joint angle MSE degradation | ~1 hr |

### Full-Scale Benchmarks (Mechanical-JEPA)

| Phase | Data | Paper Claim |
|---|---|---|
| **Phase 1: Single-embodiment** | DROID + TOTO (77k Franka episodes) | "JEPA predicts 10-step joint rollouts with MSE < 0.01" |
| **Phase 2: Cross-embodiment** | Franka→{UR5, KUKA, JACO} | "50%+ performance retention with zero-shot transfer" |
| **Phase 3: Action-conditioned** | ManiSkill (best action-state corr) | "Action conditioning improves 20-step rollout stability by X%" |

### Data Requirements (All Downloadable)

| Dataset | Size | Episodes | Role |
|---|---|---|---|
| DROID | ~2 GB | 76,000 | Primary Franka training |
| TOTO | 20 MB | 902 | Long-horizon eval |
| ManiSkill | 1.7 GB | 30,213 | Action-conditioned ablation |
| Berkeley UR5 | ~100 MB | 896 | Transfer target |
| **Total** | ~4 GB | ~108k | Feasible on current machine |

### Critical Limitations to Address in Paper

1. **Channel gap**: OXE proprioception (7-27ch) vs Brain-JEPA (450 ROIs)—can't claim "same architecture scales"
2. **Action space**: All OXE actions are EE-deltas (Cartesian), not joint commands—need learned FK or EE-space formulation
3. **Novel contribution**: Frame as "cross-embodiment transfer via shared latent dynamics" rather than "Brain-JEPA for robots"

### Scale Comparison: OXE vs Brain-JEPA

| Dimension | Brain-JEPA | OXE Mechanical-JEPA | Gap |
|---|---|---|---|
| Subjects/Trajectories | 32,000 | ~123,000 | OXE 4x larger |
| Timesteps per instance | 160 | 50-1,160 (mean 120) | Comparable |
| Channels | 450 ROIs | 7-27 proprioception | OXE 17-64x smaller |
| Total tokens | 2.3B | 31M | Brain-JEPA 74x larger |
| Cross-domain variety | 1 domain (brain) | 7+ embodiments | OXE richer |
| Actions available | NO | YES (per timestep) | OXE unique advantage |

---

## Summary: The 3-Dataset Paper Narrative

For a cohesive paper covering all three directions:

| Direction | Quick Test | Paper Benchmark | Story |
|---|---|---|---|
| **Sparse Graphs** | Paderborn 8ch subset | Full Paderborn + Hydraulic | "Physics grouping creates natural sparse graph structure" |
| **Meta-Features** | AURSAD current variance | AURSAD + C-MAPSS degradation | "Latent meta-features outperform raw sensor thresholds" |
| **Mechanical-JEPA** | TOTO single-robot | DROID→{UR5,KUKA} transfer | "Cross-embodiment transfer via shared joint dynamics" |

The strongest unique contribution is likely **cross-embodiment transfer**—Brain-JEPA can't demonstrate this (single domain), and most robot learning papers don't leverage JEPA-style prediction for transfer.

---

## Physics Masking Story Arc (Paper Narrative)

The 4-dataset narrative tells a complete scientific story about when physics masking helps:

| Dataset | Role in Paper | SOTA to Beat | Transfer Test | Expected Result |
|---|---|---|---|---|
| Double Pendulum | "Independent system" | Internal baselines | m1/m2=1.0 → 0.5 | Physics mask +7.4% (p<0.001) |
| Paderborn Bearing | "Semi-independent mechanical" | Internal baselines | Healthy → Faulty | Physics mask +3-5% |
| C-MAPSS | "Correlated system" | Internal baselines | FD001 → FD003 | Physics mask ≈ random (p=0.528) |
| ETT | "Tightly coupled" | PatchTST (~0.37), iTransformer (~0.39) | ETTh1 → ETTh2 | Physics mask -1.3% (hurts) |

**Paper title fit**: "When to Mask: Physics-Informed Attention Depends on Group Independence"

---

## Additional Dataset Recommendations

### Tier 2: Paderborn Bearing (PRIMARY)

- 8 channels with 4 clear physical modalities: vibration (radial + tangential + axial + velocity), thermal (temperature + torque), electrical (2-phase motor current)
- Real damage (both artificial EDM and natural progressive wear)
- No registration required for download
- 33 bearing conditions (healthy, outer race, inner race, combined damage)
- Transfer scenarios: healthy→faulty, artificial→natural damage, cross operating condition

**Gap**: No published MSE-based forecasting SOTA exists. We establish our own baselines.

### Tier 2 Fallback: Hydraulic System (UCI)

- 17 sensors, 4 physics groups (pressure/flow/thermal/mechanical)
- 2,205 cycles, direct download
- Downside: Mixed sampling rates (100/10/1 Hz) require preprocessing

### Tier 3: ETT

- Real electricity transformer data, 7 channels
- Key result: PhysMask HURTS (-1.3%) because thermal couples to all load groups
- The "complex interactions" case in the paper narrative

---

## High-Channel Brain-JEPA Analog

For experiments requiring many channels (closer to Brain-JEPA's 450 ROIs):

| Dataset | Channels | Timesteps | Access | Notes |
|---|---|---|---|---|
| WADI (water dist.) | 127 | 1,209,600 | Request (1 week) | Best available analog |
| SWaT (water treat.) | 51 | 946,000 | Request (1 week) | |
| Electricity (UCI) | 321 | 26,304 | Direct | |
| Traffic (SF) | 862 | 17,544 | Direct | |

**Recommendation**: WADI (127 channels) is the realistic high-channel target

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
