# FAM-JEPA: Forecast-Anything Model for Multivariate Time Series Events

Pretrain once on unlabelled multivariate sensor data. Freeze the encoder. Finetune only the predictor for each event type.

See `../paper-neurips/paper.tex` for the full write-up.

## Architecture (V17, 2.37M params)

```
 PRETRAINING (no labels)                    DOWNSTREAM (freeze encoder, finetune predictor)
 ========================                   =============================================

 x_{≤t}               x_{t+1:t+Δt}         x_{≤t}
 (past sensors)        (future sensors)      (past sensors)
       |                    |                     |
 Context Encoder       Target Encoder        Context Encoder (FROZEN)
 (causal, 2L, d=256)   (EMA copy)                |
       |                    |               h_past (frozen)
    h_past             h_future                   |
       |                    |              Predictor (FINETUNED, 790K params)
    Predictor ----L1---> h_future           ĥ_{t+Δt} for each horizon Δt
                                                  |
                                           σ(w · ĥ_{t+Δt} + b) → p(t, Δt)
                                           pos-weighted BCE loss
```

## Evaluation

- **Primary**: AUPRC pooled over all (t, Δt) cells of probability surface
- **Secondary**: AUROC pooled over same cells
- **Legacy**: RMSE (C-MAPSS), PA-F1 (anomaly) — derived from stored surfaces
- All surfaces stored as `.npz` for recomputation

## Directory structure

```
fam-jepa/
├── data/              Dataset loaders (config.py for paths)
│   ├── config.py      Central path config (INDUSTRIAL_JEPA_DATA env var)
│   ├── smap_msl.py    SMAP + MSL spacecraft telemetry
│   ├── psm.py         PSM server metrics
│   ├── smd.py         SMD server machine dataset
│   ├── mba.py         MBA cardiac ECG arrhythmia
│   └── swat.py        SWaT water treatment (stub)
├── evaluation/        Metrics + losses
│   ├── surface_metrics.py   evaluate_probability_surface() — PRIMARY
│   ├── losses.py            weighted_bce_loss(), build_label_surface()
│   ├── grey_swan_metrics.py Legacy metrics (RMSE, PA-F1, etc.)
│   └── linear_probe.py     Logistic probe + Mahalanobis
├── experiments/       v11-v21 sessions
│   ├── RESULTS.md     Master results table
│   └── v21/           Current: AUPRC + BCE framework
├── notebooks/         Quarto analysis
└── checkpoints/       Pretrained weights (.pt)
```

## Datasets

- **C-MAPSS FD001-003** (NASA): turbofan degradation, 14 sensors
- **SMAP / MSL** (NASA): spacecraft telemetry anomaly, 25/55 channels
- **PSM** (eBay): server metrics anomaly, 25 channels
- **SMD**: server machine anomaly, 38 channels
- **MBA** (MIT-BIH): cardiac ECG arrhythmia, 2 channels
