---
name: NeurIPS paper v18 honest pass - SMAP multi-seed + MSL failure
description: Paper revision 2026-04-20 integrating V18 Phase 4f multi-seed SMAP results and honest MSL failure; headline shifted from 0.73 to 0.666 +/- 0.048 (3 seeds); MSL reported as PA-F1 0.00 with three-hypothesis failure discussion
type: project
---

V18 Phase 4f results integrated into paper.tex:

- SMAP Mahalanobis 3-seed mean: PA-F1 0.666 +/- 0.048 (seed 42 = 0.733 upper-tail outlier, seed 123 = 0.626, seed 456 = 0.639)
- MSL Mahalanobis (seed 42): PA-F1 0.000, non-PA F1 0.000, AUC-PR 0.083 (near random baseline 0.105) - fails completely
- MSL pretraining loss did not fully converge (~0.02 vs SMAP's 0.015 at 50 epochs)

**Why:** We now report the 3-seed SMAP mean as the honest headline (seed 42 was upper-tail) and the MSL failure as a known limitation with three failure hypotheses (undertraining, anomaly-structure mismatch, geometry mismatch). The \todo{MSL} marker has been removed from the benchmark table.

**How to apply:** When reviewing further paper edits, remember:
- The "one encoder, two probes" framing now explicitly caveats "on SMAP only"
- JEPA pretraining contribution over random-init baseline is +0.08 PA-F1 on the 3-seed mean (was +0.145 when reporting seed 42 alone)
- MSL Mahalanobis = 0.0 is kept in Table 2 as the honest number with footnote $\P$; do not refactor away
- Contribution #4 now has three sub-claims: (i) random-init decomposition, (ii) multi-seed SMAP result, (iii) honest MSL failure that delineates method's scope
- Sections edited: Abstract, Contribution #4, \cref{tab:benchmark} (SMAP cell + MSL cell + footnote), \cref{tab:anomaly} (added 3-seed row), \cref{tab:randinit} (added 3-seed row and renamed decomposition row), Headline paragraph, Decomposition paragraph (numeric updates), new MSL failure paragraph, Honest framing paragraph, What this means paragraph, Limitations item #5, Conclusion paragraph 2 and Future work
