# Mechanical-JEPA: Trajectory JEPA for Grey Swan Prediction

Pretrain once on unlabelled multivariate sensor data. Freeze the encoder. Solve any downstream grey swan task with a linear probe.

See `../paper-neurips/paper.tex` for the full write-up.

## Architecture (V2 = main model)

```
 PRETRAINING (no labels)                    DOWNSTREAM (frozen encoder + linear probe)
 ========================                   =========================================

 x_{1:t}              x_{t+1:t+k}          x_{1:t}
 (past sensors)       (future sensors)      (past sensors)
       |                    |                     |
 Context Encoder      Target Encoder        Context Encoder (frozen)
 (causal, 2L, d=256)  (EMA copy, bidi)           |
       |                    |               h_past (frozen)
    h_past             h_future                   |
       |                    |              Linear probe (trained)
    Predictor ----L1---> h_future                 |
                                           RUL / TTE / anomaly score
```

- **RUL probe**: h_past -> Linear -> cycles to failure
- **Anomaly probe**: ||predictor(h_past) - h_future||_1 as anomaly score (zero labels needed)
- **TTE probe**: h_past -> Linear -> time until sensor threshold breach

Model: `experiments/v11/models.py` (`ContextEncoder`, `TargetEncoder`, `Predictor`, `TrajectoryJEPA`). Total: 0.99M parameters.

## Main results

### C-MAPSS FD001 (turbofan RUL, 5 seeds)

| Method | Frozen RMSE | E2E RMSE |
|--------|------------|----------|
| **Traj JEPA (ours)** | 17.81 +/- 1.7 | **14.23 +/- 0.4** |
| STAR supervised SOTA | -- | 12.19 +/- 0.6 |
| From-scratch (no pretraining) | -- | 22.99 +/- 2.3 |
| 57-feature Ridge regressor | 19.07 | -- |

At 5% labels: frozen JEPA (21.53) beats supervised STAR (24.55).
Pretraining contribution: +16.9 RMSE at 10% labels (from-scratch ablation).

### SMAP anomaly detection (zero labels)

PA-F1 = 62.5% using only the prediction error (vs MTS-JEPA 33.6%).

## Experiment versions

| Version | Focus | Key result |
|---------|-------|------------|
| v11 | V2 Trajectory JEPA on C-MAPSS | E2E 14.23, frozen 17.81 |
| v12 | Verification (shuffle test, health index) | Shuffle +41.5 RMSE, H.I. R^2=0.926 |
| v13 | Label efficiency + cross-fault transfer | 5% crossover vs STAR |
| v14 | Full-seq target, cross-sensor | Frozen 15.70, cross-sensor 14.98 |
| v15 | SIGReg, SMAP/MSL anomaly, metrics | PA-F1 62.5% SMAP |
| v16 | Bidi ablation, cross-machine, label eff. | Causal > bidi for generalization |

Each version has `RESULTS.md` and `phase*.json` with all numbers.

## Directory structure

```
mechanical-jepa/
├── experiments/v11-v16/   Experiment scripts + results JSONs
├── evaluation/            Grey swan metrics (non-PA F1, nRMSE, etc.)
├── data/                  Dataset adapters (C-MAPSS, SMAP/MSL, SWaT)
├── notebooks/             Quarto walkthroughs (v11, v12, v14, v15)
└── analysis/              Plots (PCA degradation clocks, sensor correlations)
```

## Quick start (on VM with GPU)

```bash
pip install -r requirements.txt
python experiments/v15/phase1_sigreg.py   # example: V15 SIGReg pretraining
python experiments/v16/phase1_v16b.py     # example: V16b bidi ablation
```

## Datasets

- **C-MAPSS FD001-FD004** (NASA): 14 sensors, 100-260 engines, turbofan degradation
- **SMAP/MSL** (NASA): 25/55 channels, spacecraft telemetry anomaly detection
- **SWaT** (iTrust): 51 sensors, water treatment (planned)
