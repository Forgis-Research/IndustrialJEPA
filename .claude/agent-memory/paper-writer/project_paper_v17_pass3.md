---
name: NeurIPS paper v17 pass 3 (2026-04-20)
description: Third-pass revision of paper-neurips/paper.tex addressing round-2 reviewer blockers and integrating Phase 4c/d/e results
type: project
---

Third-pass revision of /home/sagemaker-user/IndustrialJEPA/paper-neurips/paper.tex completed 2026-04-20 in response to round-2 reviewer scores of 5/10 and 4/10.

**Why:** Two reviewers flagged (a) number discrepancies between abstract/body/figures (+16.9 vs +16.0; +8.8 vs +7.9), (b) SMAP 73.3 duplicated in E2E and Frozen rows when only Frozen has Mahalanobis, (c) MSL 43.3 unsupported, (d) overclaim of Mahalanobis-on-representation as novel methodology, (e) overclaim of FAM matching STAR without paired significance test, (f) missing Mahalanobis canonical citations.

**How to apply:** The reframe is now that scoring geometry contributes more than the encoder (random-init baseline reaches PA-F1 0.588 under Mahalanobis, pretraining adds +0.145). The FAM-vs-STAR crossover is not significant under Welch's unpaired t at any budget; the robust claim is lower seed variance (sigma 0.9 vs 6.4 at 5%). These honest positions should be retained in all subsequent revisions.

**Key numbers to keep consistent across revisions:**
- From-scratch ablation: +7.9 at 100% labels (22.99 - 15.08), +16.0 at 10% (35.59 - 19.62)
- Random-init Mahalanobis SMAP: PA-F1 0.588 +/- 0.008 (3 seeds)
- Pretrained Mahalanobis SMAP: PA-F1 0.733 (1 seed, +0.145 over random-init)
- PCA-k sweep: 5/10/20/50/100 -> 0.734/0.733/0.767/0.796/0.809
- Bootstrap at k=10: 0.736 +/- 0.026
- Lead-time: 99.9% continuation / 0.1% true lead-time
- Welch's unpaired t FAM vs STAR: 100% delta=+2.89 p<0.001; 20% +0.11 p=0.95; 10% +0.91 p=0.55; 5% -3.00 p=0.36, 95% CI [-8.6, +2.9]
- Mahalanobis references added to bib: lee2018mahalanobis, sehwag2021ssd, ren2021simple

**Remaining TODO markers in paper (11 total):** SWaT eval (x2), MSL Mahalanobis, FAM 50% rerun (x2), fig regen, NASA scoring col, figure regen again, honest-protocol ablation rerun, paired STAR test, FD003/FD004 std rerun, TS2Vec/PatchTST baselines.
