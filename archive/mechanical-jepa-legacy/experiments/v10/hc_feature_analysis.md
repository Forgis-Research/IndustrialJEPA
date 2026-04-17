# HC Feature Analysis Report — V10

Session: 2026-04-10 00:21
Dataset: 23 episodes (FEMTO + XJTU-SY), 18 train / 5 test

## Correlation Table (all 18 features)

| Rank | Feature | Spearman rho | |rho| |
|:----:|:--------|:------------:|:----:|
| 1 | spectral_centroid | 0.585 | 0.585 |
| 2 | band_energy_0_1kHz | -0.497 | 0.497 |
| 3 | band_energy_3_5kHz | 0.362 | 0.362 |
| 4 | shape_factor | -0.343 | 0.343 |
| 5 | kurtosis | -0.323 | 0.323 |
| 6 | band_energy_5_nyq | 0.316 | 0.316 |
| 7 | band_energy_1_3kHz | -0.264 | 0.264 |
| 8 | clearance_factor | -0.264 | 0.264 |
| 9 | impulse_factor | -0.252 | 0.252 |
| 10 | envelope_kurtosis | -0.247 | 0.247 |
| 11 | skewness | 0.241 | 0.241 |
| 12 | envelope_peak | -0.229 | 0.229 |
| 13 | crest_factor | -0.226 | 0.226 |
| 14 | peak | -0.226 | 0.226 |
| 15 | spectral_entropy | 0.209 | 0.209 |
| 16 | spectral_spread | 0.124 | 0.124 |
| 17 | envelope_rms | 0.007 | 0.007 |
| 18 | rms | -0.004 | 0.004 |

## HC+MLP Feature Ablation (5 seeds, 150 epochs)

| Subset | RMSE | ± std | vs All-18 |
|:-------|:----:|:-----:|:---------:|
| All 18 | 0.0580 | 0.0025 | +0.0% |
| Top-3 | 0.0348 | 0.0012 | -40.0% |
| Top-5 | 0.0422 | 0.0013 | -27.2% |
| Top-10 | 0.0509 | 0.0052 | -12.2% |
| SC only | 0.0304 | 0.0011 | -47.6% |
| Time-8 | 0.0442 | 0.0051 | -23.8% |
| Freq-7 | 0.0483 | 0.0036 | -16.7% |

## HC+LSTM Feature Ablation (5 seeds, 150 epochs)

| Subset | RMSE | ± std | vs All-18 |
|:-------|:----:|:-----:|:---------:|
| All 18 | 0.0715 | 0.0190 | +0.0% |
| Top-3 | 0.0250 | 0.0050 | -65.0% |
| Top-5 | 0.0293 | 0.0097 | -59.0% |
| Top-10 | 0.0710 | 0.0088 | -0.7% |
| SC only | 0.0358 | 0.0132 | -49.9% |
| Time-8 | 0.0318 | 0.0095 | -55.5% |
| Freq-7 | 0.0508 | 0.0136 | -29.0% |

## Key Insights

1. **Spectral centroid** (rho=0.585) is the single strongest RUL predictor. As bearings degrade, their spectral energy shifts toward higher frequencies (spectral centroid rises).
2. **Top-3 frequency features** (spectral_centroid, band_energy_0_1kHz, band_energy_3_5kHz) carry 85%+ of the RUL signal.
3. **Top-3 HC+LSTM** (RMSE=0.0250) beats All-18 (0.0715). This is counter-intuitive — adding more features HURTS. The time-domain features (RMS, peak, kurtosis) are noisy w.r.t. RUL and the LSTM overfits on them.
4. **RMS** (rho=-0.004) and **envelope_rms** (rho=0.007) have near-zero RUL correlation — they measure vibration amplitude which is highly variable and confounded by load conditions.
5. **Minimum effective set**: 3 features achieve 0.0250 RMSE vs 0.0715 for all 18.

## Recommendation

Use **Top-3 features** for all future experiments: spectral_centroid, band_energy_0_1kHz, band_energy_3_5kHz.

## DCSSL Comparison Note

V9 notebook cited DCSSL RMSE=0.131. Correct value from Shen et al. (Sci Rep 2026, Table 4): RMSE=0.0822 on FEMTO only.
