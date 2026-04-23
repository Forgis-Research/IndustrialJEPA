# FAM Results — Persistent Master Table

**Last updated**: v22 (2026-04-22). Update after every session.

This file is the single source of truth for all experimental results.
Every number that enters the paper must have an entry here with provenance.

---

## How to read this table

- **Primary metric**: AUPRC pooled over probability surface p(t, Δt). One number per dataset.
- **Secondary metric**: AUROC pooled over same surface.
- **Legacy metrics**: RMSE (C-MAPSS), PA-F1 (anomaly datasets) — for literature comparability only.
- Format: `mean ± std (Ns, 95% CI [lo, hi])` where Ns = number of seeds.
- v20 results use per-window F1w (old primary). v21+ results use AUPRC (new primary).
- Probability surfaces stored as `.npz` for v21+ runs — any metric recomputable.

---

## Main Benchmark Table (Paper Tab 1)

**v22 update**: anomaly rows replaced with pred-FT numbers (frozen encoder,
BCE on per-horizon logits) using intra-entity chronological splits for
SMAP/MSL/SMD and chronological splits with window-size gap for PSM/MBA.
C-MAPSS rows unchanged from v21. See v22 phase 1 for details.

| Dataset | Domain | AUPRC ↑ | AUROC ↑ | F1-best (non-PA) | Legacy | SOTA legacy | Source |
|---------|--------|---------|---------|-------------------|--------|-------------|--------|
| C-MAPSS FD001 | Turbofan | 0.945±0.016 | 0.987±0.004 | 0.872±0.016 | RMSE 17.1±4.6 | RMSE 10.61 (STAR) | v21 phase 2 |
| C-MAPSS FD002 | Turbofan | 0.955±0.009 | 0.988±0.003 | 0.870±0.038 | RMSE 12.4±1.3 | RMSE 13.47 (STAR) | v21 phase 2 |
| C-MAPSS FD003 | Turbofan | 0.932±0.010 | 0.984±0.002 | 0.828±0.041 | RMSE 16.2±1.9 | RMSE 10.71 (STAR) | v21 phase 2 |
| SMAP | Spacecraft | **0.290±0.042** | 0.433±0.049 | **0.440±0.003** | F1 0.440 | PA-F1 0.336 (MTS-JEPA) | v22 phase 1 |
| MSL | Spacecraft | **0.237±0.077** | 0.506±0.057 | **0.330±0.022** | F1 0.330 | PA-F1 0.336 (MTS-JEPA) | v22 phase 1 |
| PSM | Server | **0.417±0.113** | 0.478±0.097 | **0.519±0.006** | F1 0.519 | PA-F1 0.616 (MTS-JEPA) | v22 phase 1 |
| SMD | Server | **0.196±0.025** | 0.655±0.039 | **0.262±0.030** | F1 0.262 | PA-F1 0.925 (AT) | v22 phase 1 |
| MBA | Cardiac | **0.784±0.024** | 0.751±0.041 | **0.725±0.024** | F1 0.725 | — | v22 phase 1 |

---

## C-MAPSS FD001 — Finetuning Mode (v21, AUPRC metric)

Same V17 backbone, per-horizon EventHead + pos-weighted BCE,
monotonicity enforced (violation rate = 0).

| Mode | Params | 100% AUPRC | 100% RMSE | 5% AUPRC | 5% RMSE | Seeds | Source |
|------|--------|-------------|-----------|----------|---------|-------|--------|
| probe_h | 2.6K | 0.928±0.023 | 22.52±4.00 | 0.866±0.021 | 23.33±2.57 | 3 | v21 phase 2 |
| **pred_ft** | **790K** | **0.945±0.016** | 17.06±4.64 | **0.808±0.062** | 23.66±4.94 | 3 | v21 phase 2 |
| e2e | 2.37M | **0.962±0.006** | **14.65±1.76** | 0.726±0.204 | 19.70±1.53 | 3 | v21 phase 2 |
| scratch | 2.37M | 0.938±0.010 | 19.65±4.34 | 0.646±0.166 | 41.32±7.54 | 3 | v21 phase 2 |

---

## C-MAPSS FD001 — Finetuning Mode (v20, F1w metric, legacy)

These use the v20 per-window F1w metric. v21 will rerun with AUPRC.

| Mode | Params | 100% F1w | 100% RMSE | 5% F1w | 5% RMSE | Seeds | Source |
|------|--------|----------|-----------|--------|---------|-------|--------|
| probe_h | 257 | 0.299±0.061 | 15.997±1.481 | 0.061±0.101 | 20.359±1.215 | 5 | v20 phase 0 |
| frozen_multi | 4K | 0.148±0.030 | 19.009±0.122 | 0.181±0.142 | 24.385±4.851 | 5 | v20 phase 0 |
| **pred_ft** | **790K** | **0.391±0.085** | 16.903±1.711 | **0.261±0.165** | 24.334±6.835 | 5 | v20 phase 0 |
| e2e | 2.37M | 0.408±0.120 | 14.956±1.157 | 0.177±0.242 | 20.085±1.885 | 5 | v20 phase 0 |
| scratch | 2.37M | 0.397±0.084 | 14.483±0.656 | 0.035±0.049 | 32.922±1.987 | 5 | v20 phase 0 |

**Headline**: pred_ft beats e2e (+0.084) and scratch (+0.226) on F1w at 5% labels.

---

## C-MAPSS FD001 — Label Efficiency (v21, AUPRC, 5 seeds)

Pred-FT vs E2E at 5 budgets under the v21 AUPRC protocol. Paired t-test
(two-sided) and Wilcoxon one-sided across matched seeds.

| Budget | Pred-FT AUPRC | E2E AUPRC | Δ | paired t(4) | paired p | Wilcoxon p |
|--------|----------------|-----------|-----|-------------|----------|-------------|
| 100% | 0.940±0.012 | **0.952±0.012** | -0.012 | -4.30 | **0.013** | 0.062 |
| 50%  | 0.924±0.012 | 0.938±0.012    | -0.014 | -2.27 | 0.086    | 0.125 |
| 20%  | 0.912±0.009 | 0.917±0.013    | -0.005 | -0.59 | 0.585    | 0.438 |
| 10%  | **0.897±0.013** | 0.871±0.051  | +0.026 | +1.33 | 0.253    | 0.438 |
| 5%   | **0.870±0.030** | 0.855±0.037  | +0.016 | +1.19 | 0.300    | 0.312 |

**10-seed follow-up (v21 phase 4b/4c) on the two low-label cells:**

| Budget | N | Pred-FT AUPRC | E2E AUPRC | Δ | paired t(9) | paired p | Wilcoxon p |
|--------|----|---------------|-----------|-----|-------------|----------|-------------|
| 10% | 10 | 0.897±0.011 | 0.863±0.051 | +0.035 | +2.22 | **0.054** | 0.131 |
| 5%  | 10 | 0.881±0.024 | 0.855±0.034 | +0.025 | +2.10 | **0.065** | 0.084 |

**Observation.** At $N{=}10$ under AUPRC the crossover is marginally
significant at both 10\% and 5\% labels (p $\approx 0.05$--0.07). The
v20 F1w-at-fixed-threshold crossover at 10\% labels ($p=0.023$) is
clearer because F1w at a fixed threshold drops to 0 on seeds where E2E
hasn't calibrated scale, whereas AUPRC is threshold-free and integrates
over all operating points so E2E's ranking remains recoverable. The
direction and magnitude of pred-FT's low-label advantage is preserved;
the F1w-to-AUPRC change moves p from 0.02 to 0.05.

---

## C-MAPSS — Label Efficiency (v20, F1w metric, legacy)

| Subset | Budget | N | Pred-FT F1w | E2E F1w | paired p | collapses p/e | Source |
|--------|--------|---|-------------|---------|----------|---------------|--------|
| FD001 | 100% | 5 | 0.391±0.085 | 0.408±0.119 | 0.84 | 0/0 | v20 phase 0 |
| FD001 | 50% | 5 | 0.391±0.083 | 0.356±0.074 | 0.27 | 0/0 | v20 phase 5 |
| FD001 | 20% | 10 | 0.276±0.184 | 0.312±0.102 | 0.57 | 2/0 | v20 phase 9 |
| **FD001** | **10%** | **10** | **0.291±0.120** | 0.133±0.164 | **0.023** | 0/4 | v20 phase 8 |
| FD001 | 5% | 10 | 0.229±0.141 | 0.124±0.201 | 0.114 | 1/7 | v20 phase 7 |
| FD002 | 100% | 5 | 0.315±0.097 | **0.426±0.046** | 0.038 | 0/0 | v20 phase 11 |
| FD002 | 10% | 5 | 0.258±0.059 | 0.259±0.048 | 0.98 | 0/0 | v20 phase 11 |
| FD002 | 5% | 5 | 0.289±0.120 | 0.216±0.133 | 0.070 | 0/0 | v20 phase 11 |
| FD003 (200ep) | 100% | 5 | **0.270±0.059** | 0.123±0.166 | 0.043 | 0/2 | v20 phase 10 |
| FD003 (200ep) | 5% | 5 | 0.062±0.082 | 0.009±0.021 | 0.25 | 1/4 | v20 phase 10 |
| FD003 (100ep) | 100% | 5 | 0.146±0.089 | 0.246±0.124 | 0.13 | 0/0 | v20 phase 6 |
| FD003 (100ep) | 5% | 5 | 0.115±0.160 | 0.016±0.025 | 0.26 | 1/4 | v20 phase 6 |

---

## Pretraining Ablation: EMA vs SIGReg (v20, FD001, pred-FT)

| Pretraining | 100% F1w | 100% RMSE | 5% F1w | 5% RMSE | Seeds | Source |
|-------------|----------|-----------|--------|---------|-------|--------|
| EMA (default) | 0.391±0.085 | 16.90±1.71 | 0.261±0.165 | 24.33±6.83 | 5 | v20 phase 3 |
| SIGReg-enc | 0.401±0.070 | 14.08±0.98 | 0.252±0.156 | 17.52±1.34 | 5 | v20 phase 3 |
| **SIGReg-pred** | **0.451±0.064** | **13.71±0.34** | 0.243±0.165 | **17.30±3.55** | 5 | v20 phase 3 |

---

## Chronos Comparison (v21, FD001, AUPRC)

Same embedding-then-linear-K-head BCE protocol for Chronos as for
FAM probe\_h.

| Model | Params | Downstream | AUPRC | AUROC | RMSE | Seeds | Source |
|-------|--------|------------|-------|-------|------|-------|--------|
| Chronos-T5-tiny | 8.4M | frozen + linear head | 0.901±0.002 | 0.980±0.001 | 25.58±3.71 | 3 | v21 phase 5 |
| FAM probe_h    | 2.37M | frozen + linear head | 0.928±0.023 | 0.985±0.004 | 22.52±4.00 | 3 | v21 phase 2 |
| **FAM pred_ft** | 2.37M | frozen enc + pred-FT | **0.945±0.016** | 0.987±0.004 | 17.06±4.64 | 3 | v21 phase 2 |
| **FAM E2E**     | 2.37M | full finetune        | **0.962±0.006** | **0.993±0.002** | **14.65±1.76** | 3 | v21 phase 2 |

## Chronos Comparison (v20, FD001, F1w legacy)

| Model | Params | F1w | RMSE | Seeds | Source |
|-------|--------|-----|------|-------|--------|
| Chronos-T5-tiny | 8.4M | 0.419±0.028 | 16.80±1.54 | 3 | v20 phase 2 |
| FAM pred_ft | 2.37M | 0.391±0.085 | 16.90±1.71 | 5 | v20 phase 0 |

---

## SMAP/MSL — Anomaly (v18/v19, Mahalanobis)

| Method | SMAP PA-F1 | SMAP non-PA F1 | MSL PA-F1 | MSL non-PA F1 | Source |
|--------|-----------|---------------|----------|--------------|--------|
| MTS-JEPA (paper) | 0.336 | — | 0.336 | — | He et al. 2026 |
| Random-init + Mahal (k=100) | 0.604±0.007 | 0.061 | 0.623±0.033 | — | v18 phase 12 |
| **FAM + Mahal (k=100)** | **0.793±0.014** | 0.038 | **0.707±0.050** | — | v18 phase 4 |

---

## PSM/SMD/MBA — Anomaly (v19, Mahalanobis)

| Dataset | PA-F1 | non-PA F1 | Seeds | Source |
|---------|-------|-----------|-------|--------|
| PSM | 0.813±0.048 | 0.085 | 3 | v19 |
| SMD | 0.252±0.017 | 0.144 | 3 | v19 |
| MBA | 0.551±0.054 | 0.251 | 3 | v19 |

---

## STAR Comparison (v18 replication)

| Subset | STAR (paper) | STAR (our replic.) | FAM E2E 100% | Source |
|--------|-------------|-------------------|-------------|--------|
| FD001 | 10.61 | 12.19±0.55 (5s) | 15.08±0.10 | v18 |
| FD003 | 10.71 | — | 15.38±1.20 | v18 phase 10 |
| FD004 | 15.87 | — | 26.32±0.58 | v18 phase 10 |

---

## v21 Targets

v21 re-evaluates everything with:
1. **Per-horizon sigmoid + pos-weighted BCE** (replaces MSE)
2. **AUPRC pooled over p(t,Δt)** (replaces F1w)
3. **Stored probability surfaces** (.npz) for metric recomputation

| Priority | What | Fills |
|----------|------|-------|
| Phase 1 | C-MAPSS FD001/002/003 pred-FT + E2E (5 seeds) | Tab 1 rows 1-3 |
| Phase 2 | SMAP/MSL/PSM/SMD/MBA (3 seeds) | Tab 1 rows 4-8 |
| Phase 4 | Label efficiency with AUPRC | Paper Tab label_efficiency |
| Phase 5 | Chronos with AUPRC | Paper Tab chronos |

---

## v22 — Anomaly Pred-FT (Entity Splits) + Cross-Channel Encoder Variants

### Anomaly Pred-FT (v22 phase 1, 3 seeds per dataset)

Frozen encoder (v17/v18/v19 ckpts), BCE on per-horizon logits, intra-entity
chronological split (SMAP/MSL/SMD) or chronological split with window-size
gap (PSM/MBA).  Surfaces stored to `v22/surfaces/*_pred_ft_seed*.npz`.

| Dataset | Split type | Pred-FT AUPRC | Pred-FT AUROC | F1-best (non-PA) | Mahal AUPRC (v21) | Δ AUPRC |
|---------|-----------|---------------|----------------|-------------------|--------------------|---------|
| SMAP    | entity (55) | 0.290±0.042 | 0.433±0.049 | 0.440±0.003 | 0.192±0.007 | **+0.098** |
| MSL     | entity (24) | 0.237±0.077 | 0.506±0.057 | 0.330±0.022 | 0.203±0.029 | +0.034 |
| SMD     | entity (28) | 0.196±0.025 | 0.655±0.039 | 0.262±0.030 | 0.091±0.010 | **+0.105** |
| PSM     | stream      | 0.417±0.113 | 0.478±0.097 | 0.519±0.006 | 0.413±0.035 | +0.004 |
| MBA     | stream      | 0.784±0.024 | 0.751±0.041 | 0.725±0.024 | 0.663±0.078 | **+0.121** |

Pred-FT improves AUPRC over Mahalanobis calibration on all five datasets.
AUROC is honest-chance on SMAP/MSL/PSM (≤ 0.51) but strong on SMD/MBA
(0.66, 0.75).  Root cause: the public benchmarks concentrate anomaly
segments in the late region of each entity, so an intra-entity
chronological split still produces a large train→test anomaly-rate shift
(SMAP: 2.1% anom in ft_train vs 28.2% in ft_test — 13× shift; MSL: 6.3%
vs 17.7%; SMD: 2.8% vs 6.1%; PSM single stream: 17.7% vs 39.9%).  Pred-FT
with aggressive pos-weight overfits the rare-anomaly region and ranks
poorly on the anom-dense tail.  SMD/MBA are cleaner because their shifts
are milder.

### Encoder Variant Pretraining (v22 phase 4-5, FD001)

Fair-comparison protocol: fixed past=100, LogUniform k ∈ [1,150], L1 loss
on L2-normalized predictions + variance regularizer (λ=0.04), AdamW
lr=3e-4 with cosine schedule, batch 64, early stop (patience=5, max 50ep).

| Variant   | Params   | Best pretrain L (mean 3 seeds) | Final epoch |
|-----------|----------|---------------------------------|-------------|
| baseline  | 2.37M    | 0.0068 | 11 |
| variantA  | 2.40M    | 0.0057 | 21 |
| variantB  | 3.04M    | 0.0069 | 12 |

Variant A converges to a lower pretraining loss.  Variant B does not.

### Encoder Variant Comparison (v22 phase 6, FD001 pred-FT, 100% labels, 3 seeds)

Same downstream head, same data, same training budget — only the frozen
encoder differs.  Legacy RMSE from surface → `surface_to_rul_expected`.

| Variant   | AUPRC           | AUROC         | RMSE (expected) | Surface mono_v |
|-----------|-----------------|----------------|------------------|----------------|
| baseline  | **0.951±0.010** | 0.990±0.002  | 17.89±2.87       | 0.000 |
| variantA  | 0.936±0.004     | 0.987±0.001  | 17.23±1.49       | 0.000 |
| variantB  | 0.939±0.013     | 0.987±0.003  | 18.38±1.83       | 0.000 |

**No variant beats baseline on FD001.**  Baseline leads on AUPRC (+0.015
over A, +0.012 over B); variant A has the tightest RMSE variance but does
not improve the mean.  C-MAPSS channels (14 thermo-mechanical sensors)
are evidently not benefiting from explicit cross-channel attention; the
causal temporal transformer already captures the relevant structure.

### Cross-Channel Variants on SMAP Anomaly (v22 phase 7, 3 seeds)

Despite no winner on FD001, we ran variants A and B on SMAP anomaly
pred-FT because the cross-channel hypothesis is most plausible on
spacecraft telemetry (25 heterogeneous telemetry + command channels).
Pretrain on SMAP train stream (fixed past=100, LogUniform k, L1 loss,
early stop patience=5), then pred-FT with entity splits.

| Encoder   | SMAP AUPRC      | SMAP AUROC      | SMAP F1 (non-PA) | ΔAUPRC vs baseline |
|-----------|-----------------|------------------|-------------------|---------------------|
| baseline  | 0.290±0.042     | 0.433±0.049      | 0.440±0.003       | -                   |
| variantA  | 0.347±0.025     | 0.539±0.042      | 0.443±0.007       | **+0.057**          |
| variantB  | **0.384±0.098** | **0.600±0.075**  | **0.477±0.038**   | **+0.094**          |

**Both variants beat baseline on SMAP — and variant B crosses AUROC 0.5
(0.600 vs baseline 0.433).**  This is the first SMAP pred-FT result with
consistently above-chance ranking.

### Cross-Channel Variants Across All Anomaly Datasets (v22 phase 7b, 3 seeds)

Phase 7 extension: we pretrain variants A/B and run pred-FT on the
remaining four anomaly datasets under identical protocol.  All numbers
mean ± std over 3 seeds; AUPRC/AUROC/F1 from the stored probability
surfaces (non-PA F1 at best threshold).

| Dataset | Channels | Encoder     | AUPRC           | AUROC           | F1 (non-PA)     | ΔAUPRC vs baseline |
|---------|----------|-------------|-----------------|------------------|------------------|---------------------|
| SMAP    | 25       | baseline    | 0.290±0.042     | 0.433±0.049      | 0.440±0.003      | -                   |
| SMAP    | 25       | variantA    | 0.347±0.025     | 0.539±0.042      | 0.443±0.007      | +0.057              |
| SMAP    | 25       | **variantB**| **0.384±0.098** | **0.600±0.075**  | **0.477±0.038**  | **+0.094**          |
| MSL     | 55       | baseline    | 0.237±0.077     | 0.506±0.057      | 0.330±0.022      | -                   |
| MSL     | 55       | variantA    | 0.193±0.029     | 0.466±0.056      | 0.330±0.023      | -0.044              |
| MSL     | 55       | variantB    | 0.195±0.039     | 0.454±0.065      | 0.327±0.018      | -0.042              |
| SMD     | 38       | baseline    | 0.196±0.025     | 0.655±0.039      | 0.262±0.030      | -                   |
| SMD     | 38       | variantA    | 0.219±0.080     | 0.688±0.033      | 0.266±0.058      | +0.023              |
| SMD     | 38       | **variantB**| **0.281±0.029** | **0.706±0.028**  | **0.301±0.049**  | **+0.085**          |
| PSM     | 25       | baseline    | 0.417±0.113     | 0.478±0.097      | 0.519±0.006      | -                   |
| PSM     | 25       | **variantA**| **0.534±0.078** | **0.595±0.087**  | **0.531±0.024**  | **+0.117**          |
| PSM     | 25       | variantB    | 0.370±0.067     | 0.431±0.085      | 0.515±0.000      | -0.047              |
| MBA     | 2        | baseline    | 0.784±0.024     | 0.751±0.041      | 0.725±0.024      | -                   |
| MBA     | 2        | variantA    | 0.757±0.002     | 0.708±0.000      | 0.694±0.002      | -0.027              |
| MBA     | 2        | variantB    | 0.761±0.004     | 0.710±0.006      | 0.694±0.003      | -0.023              |

**Cross-channel variants win on 3/5 multi-channel anomaly datasets
(SMAP, SMD, PSM)**, tie on MSL (within the noisy baseline std=0.077),
and lose on MBA (2 channels — cross-channel attention is trivial with so
few channels).  On C-MAPSS FD001 (14 correlated engine sensors) baseline
was also best.

Taking all 6 pred-FT datasets together (FD001 + 5 anomaly):

- **Baseline wins** on FD001 (correlated engine sensors), MBA (2 channels),
  MSL (within noise).
- **Variant B wins** on SMAP, SMD (large AUPRC + AUROC gains).
- **Variant A wins** on PSM (large AUPRC + AUROC gain).

No single cross-channel architecture dominates every anomaly dataset.
The two variants capture different structure: variant A (pure
iTransformer on fixed T=100) works best when temporal dynamics are
less important than cross-channel coupling; variant B (parallel temporal
+ cross-channel streams) works best when temporal dynamics still matter
but cross-channel signal is additive.

**Implication for the paper**: the "one architecture across all datasets"
framing of the main benchmark table still holds with the baseline encoder.
The cross-channel variant wins here are dataset-specific and are reported
as a v22 ablation for reviewers.  Promoting variant B into Tab 1 on a
per-dataset basis ("select encoder per dataset") is possible but
requires a narrative shift that is a decision to make after reviewing
this notebook.

### IMPORTANT: Phase 7 variant wins are a protocol artifact on SMAP (v22 phase 7d)

Follow-up: we pretrained **baseline from scratch** on SMAP with the same
fixed-past-window=100 protocol used for variantB, then ran pred-FT with
10 matched seeds for both.  Result:

| Encoder                           | SMAP AUPRC (10s) | SMAP AUROC (10s) | F1 (non-PA) |
|-----------------------------------|--------------------|--------------------|--------------|
| v17 baseline (variable-past)      | 0.290 ± 0.042 (3s) | 0.433 ± 0.049 (3s) | 0.440        |
| **baseline (fixed-past pretrain)**| **0.382 ± 0.050**  | **0.615 ± 0.038**  | **0.487 ± 0.026** |
| variantB (fixed-past pretrain)    | 0.373 ± 0.056      | 0.583 ± 0.052      | 0.462 ± 0.025 |

**Paired t-test (N=10 seeds, variantB - baseline):**

  AUPRC: Δ = -0.009 ± 0.079,  t(9) = -0.38,  p = 0.716  (not significant)
  AUROC: Δ = -0.032 ± 0.078,  t(9) = -1.29,  p = 0.228  (not significant)

**Under matched protocol, cross-channel attention (variantB) confers no
advantage on SMAP.**  The apparent variantB advantage in phase 7 was
entirely an artifact of the pretraining protocol change - specifically,
switching from variable-length past (v17, min_past=10) to fixed
past=100.  Fresh fixed-window pretraining lifts SMAP AUPRC from 0.290
(v17) to 0.382 (+0.092) regardless of whether cross-channel attention
is added.

This is an important correction to the v22 phase 7 narrative.  The
causal temporal transformer is sufficient for SMAP once pretraining
exposes it to the inference-time context length.  See phase 7e below
for SMD/PSM/MSL, which we rerun under matched protocol to check
whether their variant wins also disappear.
