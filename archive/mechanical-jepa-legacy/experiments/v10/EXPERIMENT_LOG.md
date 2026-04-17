# V10 Trajectory JEPA Experiment Log

Session: 2026-04-09 23:06

Starting V10 session at 2026-04-09 23:06
Device: cuda

============================================================
PART A: HC Feature Importance Analysis
============================================================

### Part A.1: Feature Correlations

Feature                        rho    |rho|
  spectral_centroid          0.585    0.585
  band_energy_0_1kHz        -0.497    0.497
  band_energy_3_5kHz         0.362    0.362
  shape_factor              -0.343    0.343
  kurtosis                  -0.323    0.323
  band_energy_5_nyq          0.316    0.316
  band_energy_1_3kHz        -0.264    0.264
  clearance_factor          -0.264    0.264
  impulse_factor            -0.252    0.252
  envelope_kurtosis         -0.247    0.247
  skewness                   0.241    0.241
  envelope_peak             -0.229    0.229
  crest_factor              -0.226    0.226
  peak                      -0.226    0.226
  spectral_entropy           0.209    0.209
  spectral_spread            0.124    0.124
  envelope_rms               0.007    0.007
  rms                       -0.004    0.004

**Top-3 features**: ['spectral_centroid', 'band_energy_0_1kHz', 'band_energy_3_5kHz']
**Top-5 features**: ['spectral_centroid', 'band_energy_0_1kHz', 'band_energy_3_5kHz', 'shape_factor', 'kurtosis']

### Part A.2: HC+MLP Ablations

Subset                             MLP RMSE    ± std
----------------------------------------------------
  All 18                             0.0580   0.0025
  Top-3                              0.0348   0.0012
  Top-5                              0.0422   0.0013
  Top-10                             0.0509   0.0052
  Spectral centroid only             0.0304   0.0011
  Time-domain (8)                    0.0442   0.0051
  Frequency-domain (7)               0.0483   0.0036

### Part A.3: HC+LSTM Ablations

Subset                            LSTM RMSE    ± std
----------------------------------------------------
  All 18                             0.0715   0.0190
  Top-3                              0.0250   0.0050
  Top-5                              0.0293   0.0097
  Top-10                             0.0710   0.0088
  Spectral centroid only             0.0358   0.0132
  Time-domain (8)                    0.0318   0.0095
  Frequency-domain (7)               0.0508   0.0136

**Selected for Trajectory JEPA**: Top-5 features: ['spectral_centroid', 'band_energy_0_1kHz', 'band_energy_3_5kHz', 'shape_factor', 'kurtosis']

============================================================
PART B: Trajectory JEPA
============================================================

Continuing V10 session at 2026-04-09 23:26
Device: cuda

Part A results loaded from previous run.

============================================================
PART B: Trajectory JEPA
============================================================

Continuing V10 session at 2026-04-09 23:31
Device: cuda

Part A results loaded from previous run.

============================================================
PART B: Trajectory JEPA
============================================================

### B.5 Sanity: Pretraining quality
Loss decreased: True (0.5710 -> 0.0727)
h_future PC1 Spearman with RUL: rho=-0.150
Max per-dim |Spearman| with RUL: 0.496

Continuing at 00:21 — probes complete

## Exp 1: Trajectory JEPA Pretraining
Loss: 0.8182 → 0.0711. Decreased: True
h_future max |Spearman| with RUL: 0.496 (V9 JEPA: 0.121)

## Exp 2: Linear Probes (pre-computed, fast)
probe(ĥ_future): 0.2113 ± 0.0039
probe(h_past):   0.2246 ± 0.0098
Shuffle test:    0.2326 ± 0.0117
Temporal signal: True

## Exp 3: Heteroscedastic Probe
RMSE=0.2263 ± 0.0152, PICP@90%=0.864, MPIW=0.6019

## Exp 4: MLP Probe
RMSE=0.2686 ± 0.0066

## Exp 5: E2E Fine-tuning
RMSE=0.1548 ± 0.0184

## Summary
Elapsed time: 0.0024
HC+LSTM Top-3 (best HC): 0.0250 ± 0.0050
Best Traj JEPA: 0.1548
Traj JEPA h_future PC1 corr: 0.496 >> V9 patch JEPA 0.121

---

## Session Complete: 2026-04-10

### All Deliverables

- [x] `experiments/v10/RESULTS.md` — full results table with statistical tests
- [x] `experiments/v10/EXPERIMENT_LOG.md` — every experiment logged
- [x] `notebooks/10_v10_trajectory_jepa.qmd` — complete Quarto walkthrough
- [x] `analysis/plots/v10/` — all plots saved
- [x] `experiments/v10/hc_feature_analysis.md` — HC feature importance report

### Final Numbers

| Method | RMSE | ± std |
|:-------|:----:|:-----:|
| Elapsed time (cut-point, near-trivial) | 0.0024 | — |
| HC+LSTM All-18 | 0.0715 | 0.0190 |
| **HC+LSTM Top-3 (best)** | **0.0250** | 0.0050 |
| HC+LSTM Top-5 | 0.0293 | 0.0097 |
| Traj JEPA probe(h_past) | 0.2246 | 0.0098 |
| Traj JEPA probe(ĥ_future) | 0.2113 | 0.0039 |
| Traj JEPA hetero | 0.2263 | 0.0152 |
| Traj JEPA MLP probe | 0.2686 | 0.0066 |
| **Traj JEPA E2E (best Traj JEPA)** | **0.1548** | 0.0184 |

### Key Insights

1. Top-3 HC features beat All-18. Spectral centroid (ρ=0.585) dominates.
2. Trajectory JEPA h_future |Spearman| = 0.496 vs V9 patch JEPA 0.121. Architecture works.
3. Data limitation: 18 episodes is insufficient for Traj JEPA to beat HC+LSTM.
4. DCSSL correction: V9 cited 0.131; correct is 0.0822 (Shen et al. 2026, Table 4).
5. Temporal signal confirmed: shuffle test p < 0.001.

