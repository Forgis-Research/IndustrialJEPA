# IndustrialJEPA — FAM (Forecast-Anything Model)

## What this is

A NeurIPS 2026 paper: self-supervised event prediction in multivariate time series via causal JEPA. Two contributions: (1) predictor finetuning as the downstream abstraction, (2) one architecture across N datasets and M domains.

## Repo structure

```
paper-neurips/          # THE paper (paper.tex) + figures + references
fam-jepa/
  data/                 # Dataset loaders (one file per dataset)
    smap_msl.py         # SMAP + MSL spacecraft telemetry
    psm.py              # PSM server metrics
    smd.py              # SMD server machine dataset
    swat.py             # SWaT (stub, needs data registration)
  evaluation/           # Evaluation code (used by ALL experiments)
    grey_swan_metrics.py  # evaluate_event_prediction() — the ONE eval function
    linear_probe.py       # Logistic probe + Mahalanobis scoring
  experiments/
    RESULTS.md          # Master results table (update after every session)
    v11-v19/            # Past experiment sessions (scripts + results JSONs)
    v20/                # Current session (SESSION_PROMPT.md + PLAN.md)
  notebooks/            # Quarto analysis notebooks (separate from experiments)
  checkpoints/          # Pretrained model weights (.pt files)
paper-replications/     # Replication code for baselines (STAR, MTS-JEPA, etc.)
archive/                # Old code, old paper drafts, old data pipelines
```

## Key conventions

- **Evaluation**: Every experiment calls `evaluate_event_prediction()`. No ad-hoc metrics.
- **Reporting**: `mean ± std (Ns, 95% CI [lo, hi])`. Always decompose F1 into P + R.
- **Paper file**: `paper-neurips/paper.tex` (the v20 clean draft).
- **Overnight sessions**: `experiments/vNN/SESSION_PROMPT.md` is pasted into Claude Code on the GPU VM.
- **Results persistence**: `experiments/RESULTS.md` is the single source of truth. Updated after every phase.
- **Notebooks vs experiments**: Scripts that run on the VM go in `experiments/vNN/`. Analysis that renders to HTML goes in `notebooks/`.

## Architecture (1.26M params)

- Context encoder: causal Transformer (d=256, L=2, 4 heads) → h_past
- Predictor: 2-layer MLP (198K params) → h_hat_fut conditioned on horizon k
- Target encoder: EMA copy (or SIGReg variant)
- Pretraining: L1 loss on predicted vs actual future representations
- Downstream: freeze encoder, finetune predictor + linear head (198K params)

## Datasets (per-dataset pretrain, shared architecture)

C-MAPSS FD001-004, SMAP, MSL, PSM, SMD, MBA (ECG), Paderborn (bearings).
Data loading: `fam-jepa/experiments/v11/data_utils.py` (C-MAPSS), `fam-jepa/data/*.py` (others).
