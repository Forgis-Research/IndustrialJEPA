# V21 Plan — Probability Surface + AUPRC Results

## Goal
Fill every red placeholder in paper.tex Tab 1 with real AUPRC numbers.
Store probability surfaces (.npz) so any metric can be recomputed.

## Key changes from v20
- **Metric**: AUPRC pooled over (t, Δt) replaces per-window F1w
- **Loss**: pos-weighted BCE replaces MSE
- **Head**: per-horizon sigmoid replaces 16d→1 linear
- **Storage**: .npz surfaces for every run

## Phase budget (~10-12h on A10G)

| Phase | Time | What | Deliverable |
|-------|------|------|-------------|
| 0 | 1.5h | Infrastructure: BCE head, surface eval, surface→legacy | `v21/pred_ft_utils.py` |
| 1 | 2.5h | C-MAPSS FD001/002/003 (5 seeds, pred-FT + E2E + probe) | Tab 1 rows 1-3 |
| 2 | 3.0h | SMAP, MSL, PSM, SMD, MBA (3 seeds each) | Tab 1 rows 4-8 |
| 3 | 1.0h | Fill paper.tex tables, update abstract | Commit |
| 4 | 1.5h | Label efficiency with AUPRC (FD001, 5 seeds) | Tab label_efficiency |
| 5 | 1.0h | Chronos baseline with AUPRC | Tab chronos |
| 6 | 1.0h | Appendix figures, Quarto notebook | Quality |

## Critical path
Phase 0 → Phase 1 → Phase 3 (paper table).
Phase 2 can run in parallel with Phase 1 if using different checkpoints.
Phase 4-6 are stretch.
