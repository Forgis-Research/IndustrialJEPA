# V33 Ablation Table: Cross-Channel Attention in FAM

h-AUROC, mean +/- std (n=3 seeds, 95% CI in parentheses). **Bold** = winner per row among {Baseline v33, Ch-Drop, ST-JEPA}.

| Dataset | v30 Ref | Baseline v33 | Ch-Drop (best rate) | ST-JEPA (best mask) |
|---------|---------|-------------|---------------------|---------------------|
| PSM     | 0.562   | 0.5545 +/- 0.0290 (CI: 0.48, 0.63) | **0.5678 +/- 0.0037 (CI: 0.56, 0.58), rate=0.5** | 0.4787 +/- 0.0117 (CI: 0.45, 0.51), mask=0.6 |
| SMAP    | 0.598   | 0.5324 +/- 0.0966 (CI: 0.29, 0.77) | **0.5767 +/- 0.0359 (CI: 0.49, 0.67), rate=0.0** | 0.4892 +/- 0.0146 (CI: 0.45, 0.53), mask=0.0 |
| FD001   | 0.786   | 0.7208 +/- 0.0560 (CI: 0.58, 0.86) | **0.7322 +/- 0.0165 (CI: 0.69, 0.77), rate=0.1** | 0.4940 +/- 0.0324 (CI: 0.41, 0.57), mask=0.0 |

## Statistical Tests (paired t-test, n=3)

| Comparison | t | p (two-sided) | Cohen's d | Significant? |
|-----------|---|--------------|-----------|--------------|
| PSM: ch-drop vs baseline | 0.896 | 0.465 | +0.46 | No |
| PSM: ST-JEPA vs baseline | -5.289 | **0.034** | -2.62 | Yes (regression) |
| SMAP: ch-drop vs baseline | 0.681 | 0.566 | +0.46 | No |
| SMAP: ST-JEPA vs baseline | -0.775 | 0.520 | -0.45 | No |
| FD001: ch-drop vs baseline | 0.284 | 0.803 | +0.20 | No |
| FD001: ST-JEPA vs baseline | -7.056 | **0.020** | -4.05 | Yes (regression) |

## Notes

- Ch-drop "wins" every row numerically but NO win is statistically significant (all p > 0.1).
- SMAP best ch-drop rate = 0.0 (no dropout at all). This is a re-run of baseline, not an improvement.
- ST-JEPA significantly HURTS PSM (p=0.034) and FD001 (p=0.020). Both cases show representational collapse: h_std dropped to ~0.003, val_loss to ~0.005 within epoch 1 of pretraining.
- Baseline v33 underperforms v30 reference on SMAP (-0.066) and FD001 (-0.065) due to protocol differences (matched max_context=512, n_cuts=40 vs per-dataset v30 tuning). PSM baseline within tolerance (-0.007).
- With n=3, d > 2.0 is needed for p < 0.05. Channel-dropout effects (d ~ 0.2-0.5) require n >= 16 seeds to reach significance.
