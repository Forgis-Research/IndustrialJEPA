# A2P Replication Specification

**Paper:** "When Will It Fail?: Anomaly to Prompt for Forecasting Future Anomalies in Time Series"
**Authors:** Min-Yeong Park, Won-Jeong Lee, Seong Tae Kim, Gyeong-Moon Park
**Venue:** ICML 2025, PMLR 267
**Affiliation:** Kyung Hee University / Korea University
**arXiv:** 2506.23596v1 (30 Jun 2025)
**Official code:** https://github.com/KU-VGI/AP
**Local PDF:** `when-will-it-fail-2025.pdf`

---

## Task: Anomaly Prediction (AP)

Given input signal `X_in ∈ R^(L_in × C)`, produce binary anomaly labels
`O ∈ R^(L_out)` over the **future (unarrived) window** of length `L_out`.
This is distinct from:

- **Forecasting**: predict `X_hat_out` only, no anomaly labels
- **Anomaly detection (AD)**: detect anomalies in the **observed** signal
- **AP (this paper)**: predict **when** abnormal events will occur in the future

---

## Method: A2P = AAF + SAP

Two-stage framework with shared transformer backbone `theta`:

### Stage 1 - Pretraining
1. **Anomaly-Aware Forecasting Network (AAF)**: cross-attention module that learns the relationship between anomalies in a prior signal `X_in^z` (with injected anomalies from 5 types: seasonal, global, trend, contextual, shapelet - following Darban et al. 2025) and their following anomalies in `X_out^z`. Trained with BCE-style loss against a binary anomaly probability target `y_out^z`.

2. **Anomaly Prompt Pool (APP)**: `M` learnable (key, prompt) pairs `(k_m, p_m)` where `p_m ∈ R^(L_z × D)`. A three-layer transformer feature extractor `f_ftr` with a [CLS] token produces a query from the reconstructed input. Top-N prompts (cosine similarity) are attached to the input embedding. Trained with Divergence loss `L_D` = `-KL(A(X_in^p_tilde) || A(X_in_tilde)) - lambda_k * gamma(f_ftr(X_in^r), k_m)` to push pseudo-normal and anomaly-infused embeddings apart and pull selected keys toward their normal queries.

3. **Forecasting loss** `L_F`: `0.5 * (||X_hat_out - X_out||^2 + ||X_hat_out^z - X_out^z||^2)` pretrains `theta_F`.

### Stage 2 - Main training (AAF and APP frozen)
- **Anomaly-aware forecasting loss**: `L_AF = g(X_in, X_hat_out) dot ||X_hat_out - X_out||^2` where `g(.)` is the frozen AAF producing anomaly probability (up-weights abnormal timesteps).
- **Reconstruction loss**: `L_R = 0.5 * (||X_in - X_in^p,r||^2 + ||X_in - X_in^r||^2)` where `X_in^p` uses APP-attached anomaly prompts. Forces the model to de-anomalize prompt-infused inputs back to normal.

### Total objective
`L_Total = λ_AAF L_AAF + λ_D L_D + λ_F L_F` (pretrain) + `λ_R L_R + λ_AF L_AF` (main). All λ = 1.

### Test time
Forward `X_in` through shared backbone `theta` → forecast `X_hat_out`.
Reconstruct `X_hat_out^r = theta(X_hat_out)`. Compute anomaly score
from `(X_hat_out, X_hat_out^r)` following Xu et al. 2022 (AnomalyTransformer scheme).

---

## Datasets

| Dataset   | Domain             | Dimensions | Notes |
|-----------|--------------------|:----------:|-------|
| MBA       | ECG (MIT-BIH SVDB) | 2          | Moody & Mark 2001. Supraventricular arrhythmia. |
| Exathlon  | Spark telemetry    | 19 (x 8)   | Jacob et al. 2020. |
| SMD       | Server machine     | 38         | Su et al. 2019a. 5-week internet company. |
| WADI      | Water distribution | 123        | Ahmed et al. 2017. |

Standard TS-AD public benchmarks. **Do not re-download if already present** in `C:/Users/Jonaspetersen/dev/IndustrialJEPA/mechanical-datasets/` or common cache dirs; the Anomaly-Transformer / TSAD-Eval toolboxes distribute these.

---

## Target Results (Table 1, F1 score, L_in = 100)

Paper reports F1-with-tolerance (no point adjustment). A2P rows:

| L_out | MBA | Exathlon | SMD | WADI | Avg F1 |
|:-----:|:---:|:--------:|:---:|:----:|:------:|
| 100   | 67.55 +/- 5.62 | 18.64 +/- 0.16 | 36.29 +/- 0.18 | 64.91 +/- 0.47 | **46.84** |
| 200   | 74.63 +/- 5.92 | 28.71 +/- 0.54 | 42.36 +/- 0.80 | 66.65 +/- 1.93 | **53.08** |
| 400   | 69.35 +/- 7.15 | 43.57 +/- 1.10 | 48.10 +/- 2.55 | 74.57 +/- 6.37 | **58.89** |

Baselines: all combinations of forecasters (PatchTST, MICN, GPT2, iTransformer, FITS) x AD (AnomalyTransformer, DCDetector, CAD). Best baseline avg F1 values to beat: 41.55 (L=100), 41.38 (L=200), 41.18 (L=400).

### Ablations to reproduce

- **Table 2**: AAF/SAP on-off (4 cells, L_in=L_out=100)
- **Table 3**: L_D and L_F on-off
- **Table 4**: Shared backbone on-off (51.53 -> 67.55 MBA)
- **Table 5**: Anomaly probability in L_AF (MBA F1 64.20 -> 67.55)
- **Table 6**: MSE forecasting on MBA (A2P 0.788 / 0.864 / 0.930 vs PatchTST 1.174 / 1.261 / 1.272)

---

## Validation Protocol

- Seeds: 3 (match paper)
- Metric: F1 with tolerance `t` (no point adjustment). Paper does **not** use PA.
- Threshold: "percentage of anomalies in test data" protocol from Shen et al. 2020a
- L_in = 100; L_out in {100, 200, 400}
- Optimizer / LR / batch size: match official repo or infer from appendix A.2

---

## Success Criteria

Our replication is "close enough" when A2P avg F1 on all 4 datasets at L_out = 100 is within **3 points** of paper (i.e., avg F1 >= 43.8). Stretch goal: within **1 point**.

A partial-success path: if dataset acquisition is blocked, demonstrate the method works on **MBA alone** (smallest, 2D, public, easy) and document the gap.
