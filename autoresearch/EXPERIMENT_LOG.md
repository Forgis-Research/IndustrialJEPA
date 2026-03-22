# ETTh1 Experiment Log

Date: 2026-03-22

## Setup
- Dataset: ETTh1 (17,420 rows, 7 channels, hourly)
- Split: 8640 train / 2880 val / 2880 test (standard ETT split from Informer paper)
- Lookback: 96 timesteps
- Normalization: per-channel zero-mean unit-variance (fit on train)
- All results: 3 seeds (42, 123, 456), test set MSE/MAE

## Baselines (Task 2)

| Model | H=96 MSE | H=96 MAE | H=192 MSE | H=336 MSE | H=720 MSE |
|-------|----------|----------|-----------|-----------|-----------|
| Persistence | 1.6307 | 0.8227 | 1.6819 | 1.7086 | 1.7885 |
| Linear | **0.5714** | **0.5283** | **0.7799** | **0.9704** | 1.1948 |
| MLP (1-layer) | 0.7276 | 0.6405 | 0.9583 | 1.0381 | **1.0927** |

**Observations:**
- Linear is the strongest trivial baseline for H={96,192,336}. This is consistent with the DLinear paper's finding that a simple linear model is surprisingly competitive.
- MLP overtakes Linear only at H=720 (long horizon needs more capacity, but also more prone to overfitting at short horizons).
- Persistence is terrible (MSE ~1.63-1.79), confirming the data has meaningful dynamics.

## JEPA Experiments (Task 3)

Architecture: Patch embedding (patch_len=16) -> Transformer encoder (3 layers, d=128, 4 heads) -> Latent projection (dim=64) -> MLP predictor -> Linear decoder. EMA target encoder (momentum=0.996). ~890K trainable params at H=96.

| # | Mode | H=96 MSE | H=96 MAE | H=192 MSE | H=336 MSE | H=720 MSE | Notes |
|---|------|----------|----------|-----------|-----------|-----------|-------|
| 1 | JEPA + Supervised | 0.899 | 0.717 | 0.950 | 1.004 | 1.007 | Both losses |
| 2 | Supervised only | 0.900 | 0.736 | 0.895 | 0.963 | 1.007 | No JEPA loss |
| 3 | JEPA only | 1.282 | 0.851 | 1.284 | 1.277 | 1.275 | Decoder not trained by supervision |

**Key findings:**
1. **JEPA loss provides no benefit.** EXP 1 vs EXP 2 are statistically indistinguishable. At H=192 and H=336, supervised-only is actually slightly better.
2. **JEPA-only (EXP 3) is near-useless.** MSE ~1.28 across all horizons — the decoder receives no gradient signal from supervision, so it cannot map latent predictions to meaningful forecasts. The latent loss decreases during training but this doesn't translate to forecast quality.
3. **All transformer-based models lose badly to the trivial Linear baseline.** Linear gets 0.571 at H=96; our best transformer gets 0.899. That's 57% worse.

## Published SOTA (for reference)

| Model | H=96 MSE | H=192 MSE | H=336 MSE | H=720 MSE | Source |
|-------|----------|-----------|-----------|-----------|--------|
| PatchTST | ~0.370 | ~0.383 | ~0.396 | ~0.419 | Nie et al. 2023 |
| DLinear | ~0.375 | ~0.405 | ~0.439 | ~0.472 | Zeng et al. 2023 |
| iTransformer | ~0.386 | ~0.384 | ~0.396 | ~0.428 | Liu et al. 2024 |

## Gap Analysis

| Comparison | H=96 | H=192 | H=336 | H=720 |
|------------|------|-------|-------|-------|
| Our best (Supervised) | 0.900 | 0.895 | 0.963 | 1.007 |
| Our Linear baseline | 0.571 | 0.780 | 0.970 | 1.195 |
| PatchTST SOTA | 0.370 | 0.383 | 0.396 | 0.419 |
| Gap: Ours vs Linear | +57% | +15% | -1% | -16% |
| Gap: Ours vs SOTA | +143% | +134% | +143% | +140% |

Our transformer-based models are **2.4x worse than published SOTA** across all horizons.

## Honest Assessment

### What went wrong
1. **The transformer model overfits massively.** Training loss drops to ~0.15 but val MSE stays at ~0.9-1.0. Early stopping kicks in at epoch 11-20. The model memorizes training patterns but doesn't generalize.

2. **The architecture is wrong for this task.** Published results show that for ETTh1:
   - PatchTST works because it uses channel-independent processing + supervised patching with proper attention masking
   - DLinear works because it's simple enough not to overfit
   - Our "flatten patches -> MLP predictor -> MLP decoder" loses too much structure

3. **JEPA adds nothing here.** The JEPA loss in latent space is orthogonal to the actual forecasting objective. The encoder/decoder bottleneck means the JEPA loss just regularizes the latent space, but not in a way that helps forecasting. The EMA target is essentially a slowly-updating copy, not providing meaningfully different signal.

4. **Our Linear baseline (0.571) is already worse than published DLinear (0.375).** This suggests our data split or normalization might differ slightly from standard implementations, or our linear model architecture (flatten all channels) is suboptimal vs DLinear's decomposition approach.

### What we learned
1. **JEPA is not naturally suited for direct forecasting.** JEPA was designed for self-supervised *representation learning*, not end-to-end supervised forecasting. Using it as a forecasting model without a large-scale pretraining phase misses the point.

2. **The predictor architecture matters enormously.** Flattening patch embeddings into an MLP predictor destroys temporal structure. Published models preserve temporal/channel structure throughout.

3. **Overfitting is the dominant failure mode on ETTh1.** With only 8640 training points and 7 channels, a 890K parameter model is overparameterized. The linear model (96*7 = 672 input features) has far fewer effective parameters.

### What to try next (priority order)
1. **Channel-independent mode.** Process each channel separately through the encoder (like PatchTST). This is 7x more samples with 7x fewer parameters per sample.
2. **Reduce model size drastically.** Try d_model=32, 1-2 layers, latent_dim=16. Match capacity to data size.
3. **Add proper regularization.** Dropout on patches, weight decay, data augmentation (window slicing, masking).
4. **Rethink JEPA for forecasting.** Consider: pretrain with JEPA (self-supervised), then fine-tune with supervised loss. Don't combine them simultaneously.
5. **Match DLinear's decomposition.** Add trend-seasonal decomposition before the model.
6. **Use RevIN.** Reversible instance normalization handles distribution shift between train/test.

### Bottom line
**The vanilla JEPA approach cannot beat a linear model on ETTh1.** This is not surprising — JEPA was designed for visual representation learning with large-scale data, not small-scale supervised forecasting. The path forward is either (a) make JEPA work as a *pretraining* method with fine-tuning, or (b) abandon JEPA for forecasting and focus on where its representation learning strengths matter (anomaly detection, transfer learning). The linear baseline at 0.571 is the number to beat before trying anything fancier.
