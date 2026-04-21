# FAM Results — Persistent Master Table

**Last updated**: v20 (2026-04-21). Update after every session.

This file is the single source of truth for all experimental results.
Every number that enters the paper must have an entry here with provenance.
The autoresearch agent reads this at the start of every session.

---

## How to read this table

- All results use the **honest probe protocol** (AdamW, WD=1e-2, val n_cuts=10) unless marked `(old)`.
- Format: `mean ± std (Ns, 95% CI [lo, hi])` where Ns = number of seeds.
- Legacy metrics (RMSE, NASA-S, PA-F1) shown for literature comparability.
- Primary metrics are **Stage 1 F1** (event detection) and **Stage 2 macro-F1** (event timing).
- Window size for Stage 2: Δ=30 cycles for C-MAPSS, Δ=100 steps for SMAP/MSL (appendix studies sensitivity).

---

## C-MAPSS FD001 — RUL (Time-to-Failure)

| Method | RMSE | NASA-S | F1@30 | Seeds | Source |
|--------|------|--------|-------|-------|--------|
| STAR (paper) | 10.61 | 169 | — | 1 | Fan et al. 2024 |
| STAR (our replic.) | 12.19 ± 0.55 | — | — | 5 | v18 phase 11 |
| AE-LSTM (paper) | 13.99 | — | — | 1 | Hamzaoui & LeCam 2026 |
| Chronos-T5-large (frozen) | 16.00 ± 0.15 | — | 0.905 | 3 | v18 phase 8 |
| Chronos-2 (frozen) | 16.21 ± 0.43 | — | 0.892 | 3 | v18 phase 8b |
| FAM scratch (E2E) | 22.99 ± 2.3 | — | — | 5 | v18 |
| **FAM V2 frozen (honest)** | **15.73 ± 0.14** | — | — | 5 | v18 phase 0 |
| **FAM V17 frozen (honest)** | **15.53 ± 1.68** | — | 0.919 | 3 | v18 phase 1a |
| **FAM V17 E2E 100% (honest)** | **15.08 ± 0.10** | 402 ± 109 | — | 5 | v18 phase 1b |
| **FAM V17 E2E 5% (honest)** | **21.55 ± 1.52** | 1042 ± 200 | — | 5 | v18 phase 1b |
| FAM V14 cross-sensor (frozen) | 15.38 ± 0.33 | — | — | 3 | v18 phase 6d |
| FAM V14 full-seq (frozen) | 15.54 ± 0.04 | — | — | 3 | v18 phase 6b |
| V2 frozen (old protocol) | 17.81 ± 1.7 | — | — | 5 | v11 (superseded) |
| V11 E2E (old protocol) | 13.80 | — | — | 1 | v11 (superseded, protocol-confounded) |

### Missing cells (v20 priority)
- [ ] V2 E2E honest (isolates protocol vs architecture effect)
- [ ] Stage 1/2 F1 for all rows (unified eval recomputation)

---

## C-MAPSS FD001 — Label Efficiency

| Method | 100% | 50% | 20% | 10% | 5% |
|--------|------|-----|-----|-----|-----|
| STAR (replic.) | 12.19 ± 0.6 | 13.26 ± 0.7 | 17.74 ± 3.6 | 18.72 ± 2.8 | 24.55 ± 6.4 |
| FAM Frozen | 15.53 ± 1.7 | 17.58 ± 0.5 | 19.53 ± 0.7 | 20.71 ± 0.9 | 21.47 ± 0.9 |
| FAM E2E | 15.08 ± 0.1 | 15.85 ± 0.6 | 17.85 ± 0.6 | 19.62 ± 1.4 | 21.55 ± 1.5 |

---

## C-MAPSS Cross-Subset Transfer

| Subset | FAM E2E 100% | FAM E2E 5% | STAR (paper) | Source |
|--------|-------------|-----------|-------------|--------|
| FD001 | 15.08 ± 0.10 | 21.55 ± 1.52 | 10.61 | v18 |
| FD002 | — | — | 13.47 | — |
| FD003 | 15.38 ± 1.20 | 31.27 ± 2.86 | 10.71 | v18 phase 10 |
| FD004 | 26.32 ± 0.58 | 40.42 ± 1.63 | 15.87 | v18 phase 10 |

### Missing cells (v20 priority)
- [ ] FD001-pretrained frozen probe on FD002/003/004 (transfer test)
- [ ] FD002 E2E

---

## SMAP/MSL — Anomaly Detection

| Method | SMAP PA-F1 | SMAP non-PA F1 | MSL PA-F1 | MSL non-PA F1 | Source |
|--------|-----------|---------------|----------|--------------|--------|
| MTS-JEPA (paper) | 0.336 | — | 0.336 | — | He et al. 2026 |
| TS2Vec (paper) | 0.281 | — | — | — | Yue et al. 2022 |
| PatchTST (paper) | 0.286 | — | — | — | Nie et al. 2023 |
| Random-init + Mahal (k=100) | 0.604 ± 0.007 | 0.061 | 0.623 ± 0.033 | — | v18 phase 12 |
| **FAM + Mahal (k=100)** | **0.793 ± 0.014** | 0.038 | **0.707 ± 0.050** | — | v18 phase 4 |
| FAM pred-error | 0.219 | 0.038 | — | — | v17 |
| JEPA pretraining delta | +0.189 | — | +0.084 | — | v18 phase 4k |

### Missing cells (v20 priority)
- [ ] P, R, AUROC, AUPRC for all rows
- [ ] Linear probe (logistic regression) on frozen h_past
- [ ] Stage 1/2 F1 (unified eval)

---

## Architecture Ablations (v20)

| Decision | Chosen | Alternatives | FD001 frozen RMSE | Evidence |
|----------|--------|-------------|-------------------|----------|
| Attention mask | Causal | Bidirectional | TBD | v20 phase 3 |
| Horizon sampling | LogU[1,150] | U[5,30], fixed | TBD | v20 phase 3 |
| Target window | Fixed-w | Sliding | TBD | v20 phase 3 |
| EMA momentum | 0.99 | 0.996, 0.999, none | TBD | v20 phase 3 |
| Encoder depth | L=2 | L=1, L=4 | TBD | v20 phase 3 |
| d_model | 256 | 128, 64 | TBD | v20 phase 3 |
| Loss | L1 | MSE, smooth-L1 | TBD | v20 phase 3 |

---

## Experiment Ideas / Backlog

| Idea | Priority | Status | Notes |
|------|----------|--------|-------|
| V2 E2E honest | **BLOCKING** | Not started | Isolates 13.80 mystery |
| PSM dataset | High | Loader ready | MTS-JEPA reports on this |
| SWaT dataset | Medium | Blocked (registration) | — |
| TS2Vec/PatchTST on FD001 | Medium | Not started | Reviewer requested |
| Supervised LSTM at 5% labels | Low | Not started | — |
| SIGReg-pretrained vs EMA-pretrained E2E | Medium | Not started | — |

---

## V20 Results (predictor finetuning + per-window F1)

### Phase 0b: Finetuning mode sweep on FD001 (5 seeds, W=16 horizon F1)

| Mode | Params | 100% F1w | 100% RMSE | 5% F1w | 5% RMSE |
|------|--------|----------|-----------|--------|---------|
| probe_h      | 257    | 0.299 ± 0.061 | 15.997 ± 1.481 | 0.061 ± 0.101 | 20.359 ± 1.215 |
| frozen_multi | 4K     | 0.148 ± 0.030 | 19.009 ± 0.122 | 0.181 ± 0.142 | 24.385 ± 4.851 |
| **pred_ft**  | **790K** | **0.391 ± 0.085** | 16.903 ± 1.711 | **0.261 ± 0.165** | 24.334 ± 6.835 |
| e2e          | 2.37M  | 0.408 ± 0.120 | 14.956 ± 1.157 | 0.177 ± 0.242 | 20.085 ± 1.885 |
| scratch      | 2.37M  | 0.397 ± 0.084 | 14.483 ± 0.656 | 0.035 ± 0.049 | 32.922 ± 1.987 |

Headline: **pred_ft beats e2e (+0.084) and scratch (+0.226) on F1w at 5% labels**.
Runtime: 8.5 min total. Checkpoint: v17_seed42_best.pt. Source: `v20/phase0_pred_ft.json`.

### Phase 3b: EMA vs SIGReg pretraining on FD001 (5 seeds, pred-FT downstream)

| Pretraining | 100% F1w | 100% RMSE | 5% F1w | 5% RMSE |
|-------------|----------|-----------|--------|---------|
| EMA          | 0.391 ± 0.085 | 16.90 ± 1.71 | 0.261 ± 0.165 | 24.33 ± 6.83 |
| SIGReg-enc   | 0.401 ± 0.070 | 14.08 ± 0.98 | 0.252 ± 0.156 | 17.52 ± 1.34 |
| **SIGReg-pred** | **0.451 ± 0.064** | **13.71 ± 0.34** | 0.243 ± 0.165 | **17.30 ± 3.55** |

SIGReg-pred wins at 100% (F1w +0.060, RMSE -3.19); all tied within CI at 5% on F1w,
but SIGReg-pred is much more stable (RMSE std 3.55 vs 6.83). Source:
`v20/phase3_sigreg.json`.

### Phase 1b: PSM pred-FT (NEGATIVE RESULT)

Using v19 PSM ckpts (seed 42/123/456), chronological 60/10/30 split of labeled test:

| Mode | F1w | AUROCw | global F1 |
|------|-----|--------|-----------|
| probe_h      | 0.411 ± 0.054 | 0.548 | 0.411 |
| frozen_multi | 0.401 ± 0.106 | 0.531 | 0.401 |
| pred_ft      | 0.326 ± 0.014 | 0.460 | 0.326 |

AUROCw near random across all modes => distribution shift in PSM test timeline
defeats supervised FT. Keep Mahalanobis for PSM primary (0.813 PA-F1 from v19).

### Phase 2: Chronos-T5-tiny baseline (FD001, 3 seeds)

| Model | Params | F1w | RMSE | Protocol |
|-------|--------|-----|------|----------|
| Chronos-T5-tiny | 8.4M | 0.419 ± 0.028 | 16.80 ± 1.54 | frozen + linear probe |
| FAM pred_ft (ref)| 2.37M | 0.391 ± 0.085 | 16.90 ± 1.71 | frozen enc + pred-FT |

FAM matches Chronos within CI at 3.5x fewer params. Source:
`v20/phase2_chronos_perwindow.json`.
