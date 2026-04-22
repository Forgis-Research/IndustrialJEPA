# IndustrialJEPA — FAM (Forecast-Anything Model)

Self-supervised event prediction in multivariate time series via causal JEPA.

**The idea**: pretrain a single JEPA encoder on unlabelled sensor data by predicting future representations in latent space. Freeze the encoder. Finetune only the predictor for each event type — the encoder learns dynamics, the predictor is steered toward the event. Evaluate through a unified probability surface p(t, Δt) scored by AUPRC.

**Target venue**: NeurIPS 2026. Paper: `paper-neurips/paper.tex`.

## Repository layout

```
IndustrialJEPA/
├── fam-jepa/                  Core research codebase
│   ├── data/                  Dataset loaders + config.py (8 datasets)
│   ├── evaluation/            surface_metrics.py (AUPRC), losses.py (BCE), legacy metrics
│   ├── experiments/           v11-v21 sessions (scripts + results JSONs)
│   ├── notebooks/             Quarto analysis notebooks
│   └── checkpoints/           Pretrained model weights
├── paper-neurips/             NeurIPS 2026 paper
│   ├── paper.tex              Main manuscript
│   ├── references.bib         Bibliography
│   └── figures/               Publication figures (PDF)
├── paper-replications/        Baseline replications (STAR, MTS-JEPA, etc.)
└── archive/                   Frozen: pre-pivot code, early drafts
```

## How it works

1. **Pretrain** on unlabelled multivariate sensor data (C-MAPSS, SMAP, PSM, etc.). The encoder learns to predict future latent representations — no event labels needed.
2. **Freeze** the encoder. Finetune only the predictor (790K params) with per-horizon sigmoid + positive-weighted BCE.
3. **Evaluate** through probability surface p(t, Δt): AUPRC pooled over all cells (primary), AUROC (secondary), legacy metrics derived from stored surfaces.

## Building the paper

```bash
cd paper-neurips
pdflatex paper.tex && bibtex paper && pdflatex paper.tex && pdflatex paper.tex
```
