# IndustrialJEPA

Self-supervised representation learning for grey swan prediction in industrial time series. A single JEPA encoder, pretrained on sensor trajectories via latent future prediction, supports RUL estimation, anomaly detection, and threshold exceedance through trivial linear probes.

**Target venue**: NeurIPS 2026. Paper: `paper-neurips/paper.tex`.

## Repository layout

```
IndustrialJEPA/
├── mechanical-jepa/           Core research codebase
│   ├── src/                   Shared modules (data loaders, models)
│   ├── experiments/           v11 (main model) through v16 (ablations)
│   ├── evaluation/            Grey swan evaluation metrics
│   ├── pretraining/           JEPA pretraining loop
│   ├── downstream/            RUL probe training
│   ├── data/                  Dataset adapters (C-MAPSS, SMAP/MSL, SWaT)
│   ├── notebooks/             Quarto walkthroughs (v11-v15)
│   ├── analysis/              Plots and analysis scripts
│   └── archive/               Pre-v11 code and old experiments
├── paper-neurips/             NeurIPS 2026 paper
│   ├── paper.tex              Main manuscript
│   ├── references.bib         Bibliography
│   ├── figures/               Publication figures (PDF)
│   ├── figure-pipeline/       TikZ figure design bible + compile/validate tooling
│   ├── review_history.md      Reviewer iteration scores
│   └── open_questions.md      Unresolved items
├── paper-replications/        Baseline replications
│   ├── star/                  STAR (Fan et al. 2024) - supervised RUL SOTA
│   ├── mts-jepa/              MTS-JEPA (2026) - SSL anomaly detection
│   ├── dcssl/                 DCSSL (Shen et al. 2026) - SSL RUL
│   ├── cnn-gru-mha/           Yu et al. 2024 - transfer learning
│   ├── when-will-it-fail/     Park et al. 2025 (ICML) - A2P
│   ├── Brain-JEPA/            Reference code
│   └── LeJEPA/                Reference code
├── .claude/                   Agent definitions and memory
└── archive/                   Pre-pivot work (robotics era, dataset curation, early drafts)
```

## Key results (V2 = main model, C-MAPSS FD001)

| Method | Frozen RMSE | E2E RMSE | Seeds |
|--------|------------|----------|-------|
| Traj JEPA E2E | 17.81 +/- 1.7 | **14.23 +/- 0.4** | 5 |
| STAR (supervised SOTA) | -- | 12.19 +/- 0.6 | 5 |
| From-scratch (same arch) | -- | 22.99 +/- 2.3 | 5 |

**Label-efficiency crossover**: at 5% labels, frozen JEPA (21.53) beats supervised STAR (24.55).

**Anomaly detection**: prediction error as zero-label anomaly score: SMAP PA-F1 = 62.5% (vs MTS-JEPA 33.6%).

All numbers backed by JSONs in `mechanical-jepa/experiments/v{11-16}/`.

## Experiments guide

| Directory | Purpose | Key files |
|-----------|---------|-----------|
| `experiments/v11/` | Main model: V2 Trajectory JEPA | `models.py`, `RESULTS.md` |
| `experiments/v12/` | Verification gate (diagnostics, shuffle test, health index) | `RESULTS.md` |
| `experiments/v13/` | Label efficiency + cross-fault transfer | `RESULTS.md` |
| `experiments/v14/` | Full-seq target, cross-sensor, bearings | `RESULTS.md` |
| `experiments/v15/` | SIGReg, SMAP/MSL anomaly, metrics framework | `RESULTS.md`, `phase*.json` |
| `experiments/v16/` | V16a/V16b bidi ablation, label efficiency, cross-machine | `RESULTS.md`, `phase*.json` |

## Building the paper

```bash
cd paper-neurips
pdflatex paper.tex
bibtex paper
pdflatex paper.tex
pdflatex paper.tex
```

Figures are pre-compiled PDFs in `figures/`. To rebuild a TikZ figure:
```bash
cd figure-pipeline
bash compile_figure.sh fig_<name>.tex   # compiles + quality checks
python validate_figure.py fig_<name>.pdf # automated validation
cp fig_<name>.pdf ../figures/
```

## Running experiments on the VM

```bash
# On SageMaker or similar GPU instance
cd mechanical-jepa
pip install -r requirements.txt
python experiments/v11/models.py  # verify model loads

# Overnight session (autonomous Claude Code)
# Paste contents of paper-neurips/SESSION_PROMPT_V16.md as opening prompt
```

## Archive

`archive/` contains frozen material from earlier research phases:
- `pre-paper/` -- autoresearch, dataset curation, mechanical-datasets, dcssl-replication
- `paper-neurips-drafts/` -- early paper scaffolding (v8 bearing era), literature reviews
- Root `archive/` -- robotics-era code (AURSAD, voraus, world-model)

Nothing under `archive/` is imported by active code.
