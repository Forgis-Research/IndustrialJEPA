# IndustrialJEPA — FAM (Forecast-Anything Model)

## What this is

A NeurIPS 2026 paper: self-supervised event prediction in multivariate time series via causal JEPA. Two contributions: (1) predictor finetuning as the downstream abstraction, (2) one architecture across N datasets and M domains.

## Repo structure

```
paper-neurips/          # THE paper (paper.tex) + figures + references
fam-jepa/
  data/                 # Dataset loaders (one file per dataset)
    config.py           # Central path configuration (env var override)
    smap_msl.py         # SMAP + MSL spacecraft telemetry
    psm.py              # PSM server metrics
    smd.py              # SMD server machine dataset
    mba.py              # MBA cardiac ECG arrhythmia
    swat.py             # SWaT (stub, needs data registration)
  evaluation/           # Evaluation code (used by ALL experiments)
    surface_metrics.py    # evaluate_probability_surface() — PRIMARY eval function
    losses.py             # weighted_bce_loss(), build_label_surface()
    grey_swan_metrics.py  # Legacy metrics (RMSE, PA-F1, anomaly_metrics)
    linear_probe.py       # Logistic probe + Mahalanobis scoring
  experiments/
    RESULTS.md          # Master results table (update after every session)
    v11-v20/            # Past experiment sessions (scripts + results JSONs)
    v21/                # Current session (SESSION_PROMPT.md + PLAN.md)
  notebooks/            # Quarto analysis notebooks (separate from experiments)
  checkpoints/          # Pretrained model weights (.pt files)
paper-replications/     # Replication code for baselines (STAR, MTS-JEPA, etc.)
archive/                # Old code, old paper drafts, old data pipelines
```

## Key conventions

- **Primary metric**: AUPRC pooled over probability surface p(t, Δt). Use `evaluation.surface_metrics.evaluate_probability_surface()`.
- **Training loss**: positive-weighted BCE on per-horizon sigmoid head.
- **Legacy metrics**: RMSE (C-MAPSS), PA-F1 (anomaly) — derived from stored surfaces, for literature comparability only.
- **Reporting**: `mean ± std (Ns, 95% CI [lo, hi])`. Always decompose F1 into P + R.
- **Paper file**: `paper-neurips/paper.tex`.
- **Overnight sessions**: `experiments/vNN/SESSION_PROMPT.md` is pasted into Claude Code on the GPU VM.
- **Results persistence**: `experiments/RESULTS.md` is the single source of truth. Updated after every phase.
- **Surface storage**: every experiment stores p_surface + y_surface as .npz for metric recomputation.
- **Notebooks vs experiments**: Scripts that run on the VM go in `experiments/vNN/`. Analysis that renders to HTML goes in `notebooks/`.

## Architecture (2.37M params, V17)

- Context encoder: causal Transformer (d=256, L=2, 4 heads) → h_past
- Predictor: 2-layer MLP (790K params) → ĥ_{t+Δt} conditioned on horizon Δt
- Target encoder: EMA copy (or SIGReg variant)
- Pretraining: L1 loss on predicted vs actual future representations
- Downstream: freeze encoder, finetune predictor + per-horizon sigmoid head (790K + 257 params)

## Datasets (per-dataset pretrain, shared architecture)

C-MAPSS FD001-003, SMAP, MSL, PSM, SMD, MBA (ECG).
Data loading: `fam-jepa/data/*.py` (all datasets), `fam-jepa/data/config.py` (paths).
