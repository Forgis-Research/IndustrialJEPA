# A2P Code Recon Notes

**Date:** 2026-04-10
**Repo:** https://github.com/KU-VGI/AP (cloned to `AP/`)

---

## File-to-Section Mapping

| File | Paper Section |
|------|--------------|
| `AAFN.py` | Sec 3.2 - Anomaly-Aware Forecasting Network (AAF). Also contains `train_cross_attn()`. |
| `FE.py` | Sec 3.2 - Feature extractor autoencoder (FE_model). Used by inject_amplify_learnable to find high-error timesteps for adaptive injection. |
| `shared_model.py` | Sec 3.1 - SharedModel wraps PatchTST (F) + AnomalyTransformer (AD) with shared QKV layers. |
| `AD_models/AT_model/AnomalyTransformer.py` | Sec 3.3 + Xu et al. 2022 - AnomalyTransformer backbone for AD |
| `AD_models/AT_model/prompt.py` | Sec 3.3 - Anomaly Prompt Pool (APP) implementation |
| `F_models/PatchTST.py` | Sec 3.1 - PatchTST forecasting backbone |
| `solvers/joint_solver.py` | Sec 3 - Full training loop: train_noise_and_cross() = Stage 1, train() = Stage 2, test() = evaluation |
| `utils/injection.py` | Sec 3.2 - 5 anomaly injection types + signal-adaptive injection with FE |
| `config/parser.py` | All hyperparameters + `set_data_config()` which sets channel counts per dataset |
| `run.py` | Main entry point, runs seeds loop, calls run_single_dataset() |
| `vus/` | VUS-ROC/PR metrics (bundled, from https://github.com/TheDatumOrg/VUS) |

---

## Hyperparameters: Config vs Hardcoded

### Config file / CLI (configurable):
- `joint_epochs` (default 5 in run.sh) - Stage 1 pretraining epochs
- `cross_attn_epochs` (default 5) - AAFN pretraining epochs
- `d_model` (default 256)
- `pool_size` (default 10) - prompt pool size M
- `prompt_num` (default 3) - prompts prepended per input
- `top_k` (default 3) - top-K prompts selected from pool
- `seq_len`, `pred_len`, `win_size` - window sizes
- `anormly_ratio` (default 1.0) - anomaly percentage for thresholding
- `noise_injection`, `cross_attn`, `share`, `forecast_loss`, `contrastive_loss` - flags

### Hardcoded:
- FE training: 10 epochs hardcoded in `train_ftr_extractor()` (joint_solver.py:255)
- AAFN learning rate: 5e-5 hardcoded in `train_cross_attn()` (AAFN.py:17)
- Temperature for AD anomaly score: 50 hardcoded in `calc_series_prior_loss_test()` (base.py:38)
- Contrastive loss in pretraining: loss = cross_attn_loss1 + cross_attn_loss2 only (cross_attn_loss3, cross_attn_loss4 tracked but not used in backward pass)
- Main training LR decay: TST scheduler (every epoch)

---

## Dataset Format Expected

All datasets loaded as `.npy` files:
```
<root_path>/<DATASET>_train.npy  - shape (N_train, C)
<root_path>/<DATASET>_test.npy   - shape (N_test, C)
<root_path>/<DATASET>_test_label.npy - shape (N_test,) binary 0/1
```

Exception: PSM uses CSV. Exathlon has sub-dataset variants (exathlon_1, exathlon_2, ..., exathlon_10).

Standard scaler fit on train, applied to both train and test.

---

## F1-with-Tolerance Implementation (Critical)

From `solvers/base.py::detection_adjustment()`:
```python
# adj_tolerance = 50 (default)
for i in range(len(gt)):
    if gt[i] == 1 and AT_detected[i] == 1:
        # Mark window gt[i-50:i+50] as detected if those are anomaly timesteps
        for j in range(i, i-tol, -1):  # backward 50 steps
            if gt[j] == 0: break
            AT_detected[j] = 1
        for j in range(i, i+tol):      # forward 50 steps
            if gt[j] == 0: break
            AT_detected[j] = 1
```

This is a RESTRICTED tolerance adjustment: only expands a TRUE POSITIVE to neighboring anomaly timesteps within a window. It does NOT flip true negatives.

**Compare to standard PA (point adjustment):**
- PA: if any prediction in anomaly segment is 1, mark WHOLE segment as 1
- A2P tolerance: only expands detections that already hit the anomaly, within window size 50

This is more conservative than PA but less conservative than raw F1. The paper claims this avoids the "free ride" problem of PA while being robust to minor timing errors in prediction.

---

## Threshold Setting

```python
# From base.py::get_threshold()
combined_energy = np.concatenate([train_energy, test_energy], axis=0)
thresh = np.percentile(combined_energy, 100 - anormly_ratio)
```

- Uses BOTH train and test energy to set threshold
- `anormly_ratio` = expected anomaly % (default 1.0)
- This is a percentile threshold that flags top-1% of energy as anomalous
- IMPORTANT: uses test data to set threshold -> this is technically information leakage, but standard in AD literature

---

## Bugs / Issues Found

### 1. AAFN losses 3 and 4 are not backpropagated
In `AAFN.py::train_cross_attn()`:
```python
loss = cross_attn_loss1 + cross_attn_loss2  # only these two
loss.backward()
```
`cross_attn_loss3` (N-N) and `cross_attn_loss4` (AN-N) are tracked but not included in backward. This may be intentional (supervised only on injected pairs) but is undocumented.

### 2. train_loss list never appended in Stage 2
In `joint_solver.py::train()`:
```python
train_loss = []
for i, batch in enumerate(self.train_loader):
    # ... training loop ...
    # train_loss.append(total_loss.item()) <- MISSING
train_loss = np.average(train_loss)  # This averages empty list -> NaN, but error not caught
```
Loss is printed per-batch but the epoch average is broken. Does not affect model - only logging.

### 3. Random seed in run.py
The run.py uses `random_seeds = [20462]` hardcoded for single-seed runs, then calls `fix_seed()`. The `--random_seed` flag passed as arg overrides within `run_seeds()` loop. For 3-seed experiments, need to modify `random_seeds` list in run.py or pass multiple seeds.

### 4. MBA dataset format mismatch
The paper says "2-dimensional ECG" (2 channels). The original MIT-BIH database has 2 ECG channels per recording. Our MBA prep from TranAD xlsx format preserves this correctly.

However, the paper's reported 3.x% anomaly ratio on MBA suggests labels cover arrhythmias (V + F beat types from TranAD). This matches our conversion: 3.12% anomaly in test set.

### 5. anormly_ratio parameter
In run.sh, `anormly_ratio` is set to 1.0 (default). This means the threshold is set at the 99th percentile of combined energy. For MBA with 3.12% anomaly rate, this is slightly miscalibrated. Setting `anormly_ratio=3.12` might improve MBA results.

---

## Datasets Notes

### MBA (MIT-BIH Arrhythmia)
- Source: PhysioNet/MIT-BIH via TranAD repo xlsx files
- 2 channels: ECG1, ECG2
- 7,680 samples train, 7,680 samples test
- Anomaly types: V (ventricular ectopic), F (fusion beats) - 24 anomalous beats in test
- Anomaly ratio: 3.12% (240/7680 with window=5 around each anomalous beat)
- Local path: `/mnt/sagemaker-nvme/ad_datasets/MBA/`

### SMD (Server Machine Dataset)
- Source: HuggingFace thuml/Time-Series-Library
- 38 channels, 708K timesteps train + test
- Anomaly ratio: 4.16%
- Local path: `/mnt/sagemaker-nvme/ad_datasets/SMD/`

---

## Architecture Cost (Paper Fig 7)
- MBA: ~2.3M params, ~10 GFLOPs, ~30 min train
- WADI: ~3.6M params, ~25 GFLOPs, ~1h train

---

## Run Command for Full Replication

```bash
cd /home/sagemaker-user/IndustrialJEPA/paper-replications/when-will-it-fail/AP

# MBA, L_out=100 (paper default)
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
