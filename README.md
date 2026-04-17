# IndustrialJEPA

Self-supervised pretraining for grey swan prediction in multivariate time series.

**The idea**: pretrain a single JEPA encoder on unlabelled sensor data by predicting future trajectories in latent space. Freeze the encoder. Then solve any grey swan task -- remaining useful life, anomaly detection, threshold exceedance -- by training a simple linear probe on a small labelled set. The frozen encoder matches or beats supervised SOTA across tasks, with the advantage growing as labels get scarcer.

**Target venue**: NeurIPS 2026. Paper: `paper-neurips/paper.tex`.

## Repository layout

```
IndustrialJEPA/
├── mechanical-jepa/           Core research codebase
│   ├── experiments/           v11 (main model) through v16 (ablations)
│   ├── evaluation/            Grey swan metrics (non-PA F1, nRMSE, etc.)
│   ├── data/                  Dataset adapters (C-MAPSS, SMAP/MSL, SWaT)
│   ├── notebooks/             Quarto walkthroughs (v11, v12, v14, v15)
│   └── analysis/              Plots and analysis scripts
├── paper-neurips/             NeurIPS 2026 paper
│   ├── paper.tex              Main manuscript
│   ├── references.bib         Bibliography
│   ├── figures/               Publication figures (PDF)
│   └── figure-pipeline/       TikZ figure design bible + compile/validate tooling
├── paper-replications/        Baseline replications (STAR, MTS-JEPA, DCSSL, etc.)
└── archive/                   Frozen: pre-pivot code, bearing era, early drafts
```

## How it works

1. **Pretrain** on unlabelled multivariate sensor data (e.g. C-MAPSS turbofan, SMAP spacecraft telemetry). The encoder learns to predict future latent trajectories -- no failure labels needed.
2. **Freeze** the encoder. The learned representations capture degradation dynamics, sensor drift, and anomaly precursors as a byproduct of the prediction objective.
3. **Probe** for any downstream grey swan task with a linear layer on the frozen representations:
   - **RUL**: linear probe on h_past -> cycles to failure
   - **Anomaly detection**: prediction error ||pred - target||_1 is directly an anomaly score (zero labels)
   - **Threshold exceedance**: linear probe on h_past -> time until sensor breach

## Key results

### C-MAPSS FD001 (turbofan RUL, 5 seeds)

| Method | Frozen RMSE | E2E RMSE |
|--------|------------|----------|
| **Traj JEPA (ours)** | 17.81 +/- 1.7 | **14.23 +/- 0.4** |
| STAR supervised SOTA | -- | 12.19 +/- 0.6 |
| From-scratch (same arch, no pretraining) | -- | 22.99 +/- 2.3 |

- **Pretraining contribution**: +8.8 RMSE at 100% labels, +16.9 at 10% (from-scratch ablation)
- **Label-efficiency crossover**: at 5% labels, frozen JEPA (21.53) beats supervised STAR (24.55) with 3x lower variance

### SMAP anomaly detection (zero labels)

Prediction error as anomaly score: **PA-F1 = 62.5%** (vs MTS-JEPA 33.6%, TS2Vec 28.1%). No anomaly labels used.

All numbers backed by JSONs in `mechanical-jepa/experiments/v{11-16}/`.

## Experiments guide

| Directory | Purpose | Key result |
|-----------|---------|------------|
| `experiments/v11/` | Main model: V2 Trajectory JEPA | E2E 14.23, frozen 17.81 |
| `experiments/v12/` | Verification (shuffle test, health index) | Shuffle +41.5, H.I. R^2=0.926 |
| `experiments/v13/` | Label efficiency + cross-fault transfer | 5% crossover vs STAR |
| `experiments/v14/` | Full-seq target, cross-sensor | Frozen 15.70, cross-sensor 14.98 |
| `experiments/v15/` | SIGReg, SMAP/MSL anomaly, metrics | PA-F1 62.5% SMAP |
| `experiments/v16/` | Bidi ablation, cross-machine, label eff. | Causal > bidi for generalization |

## Building the paper

```bash
cd paper-neurips
pdflatex paper.tex && bibtex paper && pdflatex paper.tex && pdflatex paper.tex
```

## Running experiments

```bash
cd mechanical-jepa
pip install -r requirements.txt
python experiments/v11/models.py  # model definition (V2 = main)
# Training scripts: experiments/v{N}/phase*.py
```

## Archive

`archive/` contains frozen material from earlier research phases (robotics era, bearing datasets, old experiments v8-v10, early paper drafts, legacy modules). Nothing under `archive/` is imported by active code.
