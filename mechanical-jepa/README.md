# Mechanical-JEPA: Trajectory JEPA for Grey Swan Prediction

Self-supervised encoder for multivariate time series that predicts future sensor trajectories in latent space. A single pretrained encoder supports RUL estimation, anomaly detection, and threshold exceedance via linear probes. See `../paper-neurips/paper.tex` for the full write-up.

## Architecture (V2 = main model)

```
x_{1:t} (past sensors)           x_{t+1:t+k} (future sensors)
        |                                 |
  Context Encoder (causal)          Target Encoder (EMA, bidi)
  2-layer Transformer, d=256       same architecture, no gradient
  0.80M params                     momentum = 0.99
        |                                 |
     h_past                            h_future (stop-gradient)
        |                                 |
     Predictor(h_past, PE(k))  --L1-->  h_future
        |
   Downstream probes:
     h_past -> Linear -> RUL(t)
     ||pred - target||_1 -> anomaly score (zero labels)
```

Defined in `experiments/v11/models.py` (classes `ContextEncoder`, `TargetEncoder`, `Predictor`, `TrajectoryJEPA`).

## Main results (C-MAPSS FD001, 5 seeds)

| Method | Frozen RMSE | E2E RMSE |
|--------|------------|----------|
| **Traj JEPA** | 17.81 +/- 1.7 | **14.23 +/- 0.4** |
| STAR supervised SOTA | -- | 12.19 +/- 0.6 |
| From-scratch (same arch) | -- | 22.99 +/- 2.3 |
| Feature regressor (57 feats) | 19.07 | -- |

Label-efficiency crossover at 5%: frozen JEPA 21.53 beats STAR 24.55.

## Experiment versions

| Version | Focus | Key result |
|---------|-------|------------|
| v11 | V2 Trajectory JEPA on C-MAPSS | E2E 14.23, frozen 17.81 (5 seeds) |
| v12 | Verification gate (shuffle test, health index, diagnostics) | Shuffle +41.5 RMSE, H.I. R^2=0.926 |
| v13 | Label efficiency + STAR replication + cross-fault transfer | 5% crossover, FD001->FD003 transfer |
| v14 | Full-seq target, cross-sensor (iTransformer), bearings | Frozen 15.70 (full-seq), 14.98 (cross-sensor) |
| v15 | SIGReg, SMAP/MSL anomaly, evaluation metrics framework | PA-F1 62.5% SMAP, SIGReg isotropy confirmed |
| v16 | V16a/V16b bidi ablation, cross-machine, label efficiency | Causal > bidi for transfer + low-label |

Each has `RESULTS.md` and `phase*.json` files with all numbers.

## Directory structure

```
mechanical-jepa/
├── experiments/v11-v16/     Per-version experiment scripts + results JSONs
├── evaluation/              Grey swan metrics (non-PA F1, nRMSE, etc.)
├── data/                    Dataset adapters (C-MAPSS, SMAP/MSL, SWaT)
├── notebooks/               Quarto walkthroughs (v11, v12, v14, v15)
├── analysis/                Plots (PCA clocks, sensor correlations)
└── archive/                 Pre-v11 code, old experiments, bearing-era modules
```

## Quick start (on VM with GPU)

```bash
pip install -r requirements.txt

# Training scripts are self-contained per experiment version
python experiments/v15/phase1_sigreg.py    # V15 SIGReg pretraining example
python experiments/v16/phase1_v16b.py      # V16b bidi ablation example

# Model definition (V2 = main) lives in experiments/v11/models.py
```

## Datasets

Active (used in paper):
- **C-MAPSS FD001-FD004** (NASA): 14 sensors, 100-260 engines, turbofan degradation
- **SMAP/MSL** (NASA): 25/55 channels, spacecraft telemetry anomaly detection
- **SWaT** (iTrust): 51 sensors, water treatment (planned)

Archived (bearing-era, in `archive/`):
- FEMTO, XJTU-SY, IMS, CWRU, Paderborn, MFPT, Ottawa, MAFAULDA
