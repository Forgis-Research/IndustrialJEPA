# Paper Replications

Self-contained replications of baseline papers for the IndustrialJEPA NeurIPS submission. Each subfolder targets a single paper and contains its spec, code, results, and notes.

## Replications

| Folder | Paper | Venue | Status |
|--------|-------|:-----:|:------:|
| [`star/`](star/) | Fan et al. "STAR: A Simple Transformer with Adaptive Residuals for RUL Prediction" | ICASSP 2024 | FD001-FD003 done |
| [`mts-jepa/`](mts-jepa/) | He et al. "MTS-JEPA: Multi-Resolution Joint-Embedding Predictive Architecture for Time-Series Anomaly Prediction" | arXiv 2026 | done (SMAP/MSL) |
| [`dcssl/`](dcssl/) | Shen et al. "A novel dual-dimensional contrastive SSL framework for rolling bearing RUL prediction" | Sci. Rep. 2026 | done |
| [`cnn-gru-mha/`](cnn-gru-mha/) | Yu et al. "RUL of Rolling Bearings via Transfer Learning with CNN-GRU-MHA" | Appl. Sci. 2024 | done |
| [`when-will-it-fail/`](when-will-it-fail/) | Park et al. "When Will It Fail?: Anomaly to Prompt for Forecasting Future Anomalies" | ICML 2025 | done |

`star/` is the primary supervised baseline for C-MAPSS. `mts-jepa/` is the primary SSL anomaly detection baseline for SMAP/MSL.

## Convention

Each replication folder contains:
- `REPLICATION_SPEC.md` — what we are trying to match and why
- `results/` — structured JSON + markdown tables
- the paper's PDF at the root (where available)
