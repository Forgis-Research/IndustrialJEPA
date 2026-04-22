# V21 Plan — Probability Surface + AUPRC Results

## Goal
Fill every red placeholder in paper.tex Tab 1 with real AUPRC numbers.
Store probability surfaces (.npz) so any metric can be recomputed.

## Key changes from v20
- **Metric**: AUPRC pooled over (t, Δt) replaces per-window F1w
- **Loss**: pos-weighted BCE replaces MSE
- **Head**: per-horizon sigmoid replaces 16d→1 linear
- **Storage**: .npz surfaces for every run
- **Priority**: anomaly datasets FIRST (historically flakier, highest value)

## Phase budget (~10-12h on A10G)

| Phase | Est. time | What | Deliverable |
|-------|-----------|------|-------------|
| 0 | 1.5h | Infrastructure: BCE head, surface eval, surface→legacy | `v21/pred_ft_utils.py` |
| 1 | 2-3h | **SMAP, MSL, PSM, SMD, MBA** (3 seeds, Mahal + pred-FT) | Tab 1 rows 4-8 |
| 2 | 2.5h | C-MAPSS FD001/002/003 (5 seeds, pred-FT + E2E) | Tab 1 rows 1-3 |
| 3 | 1h | Fill paper.tex tables, abstract, summary paragraph | Commit |
| 4 | 1.5h | Label efficiency with AUPRC (FD001, 5 seeds) | Tab label_efficiency |
| 5 | 0.75h | Chronos baseline with AUPRC | Tab chronos |
| 6 | 1h | Appendix figures, Quarto notebook | Quality |

## Critical path
Phase 0 → Phase 1 (anomaly) → Phase 2 (C-MAPSS) → Phase 3 (paper)
Phase 4-6 are stretch.

## Prior checkpoint availability

| Dataset | Checkpoint | Location | Epochs |
|---------|-----------|----------|--------|
| SMAP | v17 | `checkpoints/` or `v17/ckpts/` | 150 |
| MSL | v17 | same | 150 |
| PSM | v19 | `v19/ckpts/` | 50 |
| SMD | v19 | `v19/ckpts/` | 50 |
| MBA | v19 | `v19/ckpts/` | 50 |
| FD001 | v17 | `v17/ckpts/` | 200 |
| FD001 SIGReg | v20 | `v20/ckpts/` | 200 |

If any checkpoint is missing, re-pretrain: SMAP/MSL ~45min, others ~20min.
