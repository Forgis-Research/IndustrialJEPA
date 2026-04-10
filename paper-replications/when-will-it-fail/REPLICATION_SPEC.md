# A2P Replication Specification

**Paper:** "When Will It Fail?: Anomaly to Prompt for Forecasting Future Anomalies in Time Series"
**Authors:** Min-Yeong Park, Won-Jeong Lee, Seong Tae Kim, Gyeong-Moon Park
**Venue:** ICML 2025 (Proceedings of the 42nd International Conference on Machine Learning, pp. 48086-48103)
**arXiv:** 2506.23596
**Official code:** https://github.com/KU-VGI/AP

---

## Task: Anomaly Prediction (AP)

Given an observed input window X_in of length L_in and C channels, predict binary anomaly labels over the future unobserved window of length L_out. This is distinct from:
- **Anomaly Detection (AD):** detect anomalies in the observed signal
- **Forecasting (F):** predict future signal values

AP is evaluated without point adjustment (unlike most AD work), which makes the metric harder.

---

## Target Numbers (Table 1 from paper)

F1 (tolerance t) averaged over 3 seeds. L_in = L_out for each experiment.

| Dataset | Channels | L_out=100 | L_out=200 | L_out=400 |
|---------|----------|-----------|-----------|-----------|
| MBA | 2 | 67.55 +/- 5.62 | 72.40 +/- 5.41 | 71.34 +/- 3.92 |
| Exathlon | 19 | 34.18 +/- 1.32 | 45.27 +/- 5.38 | 55.65 +/- 5.74 |
| SMD | 38 | 52.07 +/- 1.88 | 54.64 +/- 2.36 | 56.40 +/- 1.57 |
| WADI | 123 | 29.87 +/- 2.03 | 35.97 +/- 2.13 | 48.96 +/- 3.38 |
| **Avg** | | **46.84** | **53.08** | **58.89** |

Key baselines (L_out=100 avg F1):
- Best single baseline: ~41 F1
- A2P beats best baseline by ~6 F1 points on average

---

## Method: A2P Architecture

Two-stage framework with shared PatchTST transformer backbone.

### Stage 1: Pretraining (AAF + APP)

**Anomaly-Aware Forecasting (AAF) - AAFN.py:**
- Cross-attention module where query=predicted window, key=injected-anomaly input
- Learns to predict anomaly probability in the future given anomaly patterns in the past
- 5 synthetic anomaly types (global, contextual, seasonal, trend, shapelet) from Darban et al. 2025
- FE_model (encoder+decoder autoencoder) identifies high-error timesteps for adaptive injection
- Loss: MSE between AAFN output and binary anomaly labels

**Synthetic Anomaly Prompting (SAP) / Anomaly Prompt Pool (APP):**
- M learnable (key, prompt) pairs in the pool (default M=10, prompt_num=3, top_k=3)
- CLS-token query from 3-layer transformer selects top-K prompts by cosine similarity
- Divergence loss: KL divergence pushes prompt-infused embeddings away from clean embeddings
- Signal-adaptive: injection focuses on high-reconstruction-error timesteps

**Shared backbone training:**
- Forecasting loss on both clean and anomaly-injected inputs
- Backbone shared between F model (PatchTST) and AD model (AnomalyTransformer)

### Stage 2: Main Training (AAF+APP frozen)

- **L_AF:** forecasting loss weighted by anomaly probability from AAFN
- **L_R:** reconstruction loss on predicted window (forces backbone to undo anomalies)
- Combined: L = recon_coeff * L_AD + af_coeff * L_AF

### Test Time Inference

1. Forward X_in through F model -> X_hat_out (predicted future values)
2. Run X_hat_out through AD model (AnomalyTransformer) -> anomaly score per timestep
3. Threshold anomaly score at percentile = (1 - anomaly_ratio) to get binary predictions
4. Threshold set using combined train+test energy scores (no separate validation)

---

## Architecture Details

| Component | Implementation |
|-----------|---------------|
| Backbone F | PatchTST (patch_len=10, stride=8, e_layers=3, d_model=256) |
| Backbone AD | AnomalyTransformer (win_size=L_out, same d_model, n_heads=8) |
| FE model | Conv autoencoder (FE.py) |
| AAFN | Single-layer MultiheadAttention + Linear proj |
| Prompt pool | Learnable embeddings, cosine nearest-neighbor lookup |
| Shared layers | QKV projections at layers 0,1,2 shared between F and AD |

---

## Datasets

| Name | Channels | Train size | Test size | Anomaly % | Source |
|------|----------|------------|-----------|-----------|--------|
| MBA | 2 | 7,680 | 7,680 | ~3.1% | MIT-BIH Arrhythmia Database (PhysioNet) |
| SMD | 38 | 708,405 | 708,420 | 4.16% | Server Machine Dataset |
| Exathlon | 19 | varies (8 subsets) | varies | ~5% | Exathlon benchmark |
| WADI | 123 | large | large | ~5% | Water Distribution (iTrust SUTD) |

Local paths:
- MBA: `/mnt/sagemaker-nvme/ad_datasets/MBA/`
- SMD: `/mnt/sagemaker-nvme/ad_datasets/SMD/`

---

## Evaluation Protocol

**F1 with tolerance t (no point adjustment):**
- Tolerance window: the paper uses tolerance=50 by default
- Threshold: percentile(combined_train+test_energy, 100 - anormly_ratio)
- anormly_ratio: set to the actual anomaly percentage in the test set per dataset
- MBA: anormly_ratio ~1.0 (paper default)
- SMD: anormly_ratio ~4.16

**Key distinction from PA (point adjustment):**
- Standard AD papers use PA: if any prediction hits within an anomaly segment, the whole segment counts
- A2P explicitly does NOT use PA - raw F1 only
- This makes numbers lower but more honest

---

## Default Hyperparameters (from run.sh)

```
AD_model=AT
model=PatchTST
joint_epochs=5
cross_attn_epochs=5
d_model=256
contrastive_loss_coeff=1.0
forecast_loss_coeff=1.0
cross_attn_loss_coeff=1.0
recon_loss_coeff=1.0
af_loss_coeff=1.0
prompt_num=3, pool_size=10, top_k=3
cross_attn_nhead=1
noise_step=100
batch_size=8
lr=0.0001
```

---

## Success Criteria

- Phase 1 done: MBA F1 within 5 points of paper (>= 62.55) for at least 1 seed
- Phase 2 done: SMD F1 within 10 points of paper (>= 42.07) for at least 1 seed
- Full replication: avg F1 within 5 points of paper across MBA + SMD

---

## Replication Command (MBA, L_out=100)

```bash
cd /home/sagemaker-user/IndustrialJEPA/paper-replications/when-will-it-fail/AP
python3 run.py \
  --random_seed 0 \
  --root_path /mnt/sagemaker-nvme/ad_datasets/MBA \
  --dataset MBA \
  --model_id F+AD_100_100 \
  --seq_len 100 --pred_len 100 --win_size 100 \
  --step 100 --noise_step 100 \
  --joint_epochs 5 \
  --share \
  --AD_model AT \
  --d_model 256 \
  --noise_injection \
  --pretrain_noise \
  --contrastive_loss \
  --forecast_loss \
  --cross_attn --cross_attn_epochs 5 \
  --cross_attn_nheads 1 \
  --ftr_idx 0
```

---

## Files

```
when-will-it-fail/
  AP/                    - official code (git clone KU-VGI/AP)
  results/               - per-run JSON files
    all_results.json     - aggregated results matching dcssl schema
    RESULTS_TABLE.md     - paper vs ours comparison
    improvements/        - improvement probe results
  figures/               - matplotlib figures
  notebooks/             - Quarto .qmd files
  REPLICATION_SPEC.md    - this file
  EXPERIMENT_LOG.md      - chronological experiment log
  RECON_NOTES.md         - code archaeology notes
  IMPROVEMENT_IDEAS.md   - NeurIPS improvement ideas
  SESSION_SUMMARY.md     - final session summary
```
