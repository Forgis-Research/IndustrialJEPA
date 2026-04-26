# FAM (ours) vs. MTS-JEPA: Head-to-Head Comparison (V18)

**Date:** 2026-04-20
**Purpose:** Rigorous, updated head-to-head for NeurIPS. Supersedes `v14/mtsjepa_comparison.md` with v17/v18 numbers and honest framing.

---

## 1. Comparison Table

| Dimension | FAM (ours, v17/v18) | MTS-JEPA (He et al. 2026) | Winner |
|:----------|:--------------------|:--------------------------|:-------|
| **Primary task** | RUL regression (C-MAPSS), threshold-exceedance TTE, anomaly as auxiliary | Anomaly prediction on next window | Different |
| **C-MAPSS FD001 frozen RMSE (honest probe)** | **15.53 ± 1.68** (v17, 3 seeds) / 15.73 ± 0.14 (v2 honest, 5 seeds) | N/A — not evaluated on C-MAPSS | Ours (only entrant) |
| **C-MAPSS FD001 E2E RMSE @ 100% labels** | **~15.08 ± 0.12** (v18 phase 1b, 5 seeds, honest protocol) / 13.80 (v11, old protocol) | N/A | Ours |
| **C-MAPSS FD001 E2E RMSE @ 5% labels** | **~21.55 ± 1.55** (v18, 5 seeds) | N/A | Ours |
| **C-MAPSS F1 @ k=30** | **0.919 ± 0.024** (v18, honest frozen) | N/A | Ours |
| **SMAP non-PA F1** | 0.038 (v17, L1 score, honest) | ~0.33 (paper, PA-F1 mislabeled as F1 in some tables) | MTS-JEPA |
| **SMAP PA-F1** | 0.219 (v17, pred-error score) / 0.625 (old v15 claim; retracted as bug) | 0.336 (paper) | MTS-JEPA |
| **MSL PA-F1** | 0.433 (v15) | 0.336 (paper) | Ours (on one metric) |
| **Model size** | 1.26M params total (0.58M encoder) | ~5–8M (paper d=256, L=6); our replication 1.47M at d=128 L=3 | Ours (4–6× smaller) |
| **Loss terms** | 1 (L1 on normalized h) + 1 var-reg; optional SIGReg (1 more) | 7+ weighted terms (KL_fine, MSE_fine, KL_coarse, L_emb, L_com, L_ent_sample, L_ent_batch, L_rec) | Ours (simpler) |
| **Horizon** | Stochastic k ~ LogU[1, 150] (v17); previously U[5, 30] (v2) | Fixed: one window ahead (T_t=100) | Ours (more flexible) |
| **Collapse prevention** | EMA momentum 0.99 → SIGReg curriculum (v17); EMA+L1 suffices otherwise | Soft codebook + dual entropy + EMA on both encoder and codebook | Tie (both work) |
| **Training stability (our observation)** | Stable across 5 seeds, std < 1.7 at 100% | Fragile at small batch; our replication had KL divergence, best epoch 1–7 | Ours |
| **Training cost** | ~45 min A10G, 200 epochs (FD001) | Requires batch ≥ 128 for stability; paper doesn't report wall-clock | Ours (documented) |
| **Multi-subset (FD002/3/4)** | v14 reports on FD003 (18.39 frozen, 13.67 E2E) and FD004 (28.08 frozen, 25.27 E2E) | N/A | Ours |
| **Label efficiency** | Evaluated at 100/50/20/10/5% | Fixed 60/20/20 split — no label sweep | Ours |
| **Lead-time / continuation split** | Reported for SMAP replication (v13/v14) | Not reported in paper | Ours (methodology) |
| **Anomaly from prediction error** | Direct (the score = L1 between pred and EMA target) | NOT exploited (their anomaly comes from MLP on codebook) | Different (ours is more minimal) |

---

## 2. Honest Framing for the Paper

The original claim in `paper-neurips/paper.tex` abstract/contributions that FAM achieves 62.5% SMAP F1 vs MTS-JEPA 33.6% is a **V15 measurement that was not reproducible under v17 bug fixes** (argument order in `anomaly_metrics`, EMA target). The correct statement is:

 - FAM targets prognostics (RUL, threshold-exceedance) with strong numbers on C-MAPSS FD001-FD004.
 - FAM's prediction-error anomaly score underperforms on SMAP (non-PA F1 0.038, PA-F1 0.219) because SMAP anomalies are recurrent patterns a well-trained JEPA learns to predict. The prediction-error score anti-correlates with labels on this benchmark (gap = -0.61 in favor of normals).
 - MTS-JEPA's anomaly detection is stronger on SMAP (~0.33 PA-F1) because its codebook-based scoring captures regime shifts, not raw predictability.
 - **The two methods address different failure modes**: MTS-JEPA for recurrent anomaly regimes; FAM for slow-onset prognostic degradation. This is not an either-or comparison.

---

## 3. What Reviewers Would Catch

Paper currently overclaims by a factor of ~30x on SMAP non-PA F1 and by ~2x on PA-F1. The FD001 E2E claim of 13.80 was from an **old probe protocol**; under the honest protocol, v18 E2E lands at ~15.1 at 100% labels, matching or slightly improving over the honest v2 frozen baseline (15.73). The genuine improvements from v17's architectural changes (LogU k, fixed-window target, curriculum SIGReg) are real but small (+0.4 RMSE).

**Recommended paper revisions** (see Phase 3 reviewer-writer loop):
1. Abstract: replace "F1 62.5% on SMAP" with honest framing (PA-F1 0.22 on SMAP, 0.43 on MSL; prognostics is the strength)
2. Contributions: rewrite #4 to frame anomaly as auxiliary/negative result
3. Add Phase 0 honest re-probe of v2: 15.73 vs previously-reported 17.81 (explains the 17.81 → 15.38 story)
4. Add Phase 1 honest E2E: 15.08 vs v11's 13.80 — note the protocol difference
5. Add v14 multi-subset numbers explicitly
6. Add MTS-JEPA comparison table (this file) to experiments section

---

## 4. References to Existing Experiments

 - `mechanical-jepa/experiments/v14/mtsjepa_comparison.md` — original detailed architectural diff (kept as-is; this v18 doc replaces the numeric claims only)
 - `mechanical-jepa/experiments/v14/RESULTS.md` — FD001/003/004 cross-subset numbers
 - `mechanical-jepa/experiments/v17/RESULTS.md` — honest v17 numbers (frozen 15.38 ± 1.08)
 - `mechanical-jepa/experiments/v18/phase0_honest_reprobe.json` — v2 honest re-probe
 - `mechanical-jepa/experiments/v18/phase1a_frozen_multi_k.json` — multi-k F1 on v17 ckpts
 - `mechanical-jepa/experiments/v18/phase1b_e2e_results.json` — v17 E2E label sweep (in progress)
 - `paper-replications/mts-jepa/` — our local replication attempt, seed variance and lead-time analysis

---

## 5. Key Takeaway for Positioning

FAM's story is not "we beat MTS-JEPA at everything." The honest story is:

> **One encoder, multiple tasks, prognostics-first.** FAM is a simple (1 loss, 1.26M params, 45 min training) self-supervised encoder whose primary strength is C-MAPSS-style prognostics: frozen at 5% labels beats supervised STAR, and achieves 15-cycle RMSE on FD001. SMAP anomaly detection is an auxiliary task where FAM underperforms dedicated anomaly methods like MTS-JEPA, highlighting that the prediction-error-as-anomaly-score assumption breaks for recurrent anomalies. We report this honestly as a diagnostic test of the JEPA framework, not as a competitive SMAP result.
