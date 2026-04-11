# Phase 0 Recon Notes: A2P Official Repository

**Repo:** https://github.com/KU-VGI/AP  
**Read date:** 2026-04-10  
**Reader:** ml-researcher agent

---

## Repository Layout

```
AP/
├── AD_models/AT_model/       # AnomalyTransformer AD branch
├── F_models/                 # PatchTST / MICN / GPT2 / iTransformer forecasting branch
├── config/parser.py          # All CLI arguments and set_data_config()
├── data_provider/
│   ├── data_factory.py       # Dataset routing and normalisation
│   └── data_loader.py        # F_AD_Dataset - windowed dataset class
├── layers/                   # PatchTST backbone, Transformer layers, RevIN, Embed
├── solvers/
│   ├── base.py               # Base_Solver: threshold, detection_adjustment, get_scores
│   └── joint_solver.py       # Solver: full train/test pipeline
├── utils/
│   ├── injection.py          # Anomaly injection (5 types: global, contextual, trend, shapelet, seasonal)
│   ├── metrics.py            # Regression metrics (MAE/MSE/RMSE - NOT F1)
│   ├── tools.py              # EarlyStopping, StandardScaler, adjust_lr
│   └── utils.py              # fix_seed, my_kl_loss
├── vus/                      # VUS-ROC / VUS-PR metric library
├── AAFN.py                   # Anomaly-Aware Forecasting Network + train_cross_attn()
├── FE.py                     # Feature extractor (3-layer Transformer with [CLS] token)
├── shared_model.py           # SharedModel: wraps AD+F branches with shared theta layers
├── run.py                    # Entry point
└── run.sh                    # Reference launch script
```

---

## Paper Section to Code Mapping

| Paper section | Code file | Key class / function |
|---------------|-----------|----------------------|
| Section 3.2 (Shared backbone) | `shared_model.py` | `SharedModel` - shares QKV projections across AT and PatchTST via `--share` flag |
| Section 3.3 (AAF) | `AAFN.py` | `AAFN` class + `train_cross_attn()` pretraining loop |
| Section 3.3 (APP) | `solvers/joint_solver.py` + `shared_model.py` | Prompt pool in `SharedModel.calc_losses()`, KL divergence loss |
| Section 3.3 (Anomaly injection) | `utils/injection.py` | `inject_amplify_learnable()` - 5 types, learnable scales |
| Section 3.3 (Feature extractor f_ftr) | `FE.py` | `FE_model` (Transformer with CLS token) |
| Section 3.4 (Main training L_AF) | `solvers/joint_solver.py` | `train()` - AAFN output gates MSE loss |
| Section 3.4 (L_R reconstruction) | `shared_model.py` | `calc_losses()` - prompt-infused vs clean reconstruction |
| Section 3.6 (Test time) | `solvers/joint_solver.py` | `test()` -> `test_from_predicted()` |
| Section 4 (F1-with-tolerance metric) | `solvers/base.py` | `detection_adjustment()` + `get_scores()` |

---

## Key Hyperparameters

All from `config/parser.py` and `run.sh`:

| Parameter | Default | Notes |
|-----------|---------|-------|
| `seq_len` | 100 | L_in |
| `pred_len` | 100 / 200 / 400 | L_out |
| `win_size` | 100 | AD window (= seq_len) |
| `step` | pred_len | stride between test windows |
| `d_model` | 256 | transformer hidden dim |
| `joint_epochs` | 5 | main training epochs (paper: 5) |
| `cross_attn_epochs` | 5 | AAF pretraining epochs |
| `noise_step` | 100 | injection cadence |
| `anormly_ratio` | from `set_data_config()` | dataset-specific anomaly % for threshold |
| `pool_size` | 10 | M = number of prompts in APP |
| `top_k` | 3 | N = top-N prompts selected |
| `prompt_num` | 3 | prompt token length L_z |
| Loss coefficients | all 1.0 | lambda_AAF = lambda_D = lambda_F = lambda_R = lambda_AF = 1.0 |
| lr | 0.0001 | Adam |
| batch_size | 8 | from parser default |
| patience | 20 | early stopping |

**Data path:** `/local_datasets/AD_datasets/MBA` - hardcoded in run.sh, must be overridden with `--root_path`.

---

## Dataset Format

From `data_provider/data_factory.py`:

- Files expected: `{dataset}_train.npy`, `{dataset}_test.npy`, `{dataset}_test_label.npy`
- PSM exception: `.csv` format
- Sub-datasets (exathlon, NAB, UCR): `{dataset}_labels.npy`
- Normalisation: `StandardScaler` is imported but appears applied outside the Dataset class; `reorganize()` in `data_factory.py` applies it on load
- `anormly_ratio` is set by `set_data_config(args)` based on dataset name - this controls the threshold percentile (100 - ratio)

**CRITICAL:** The `anormly_ratio` parameter is what implements the "percentage of anomalies in test data" threshold protocol from the paper. This is NOT learned - it is set to the known anomaly prevalence of each dataset.

---

## F1-with-Tolerance Implementation

This is the most important metric detail. From `base.py`:

```python
def detection_adjustment(self, AT_detected_ori, gt, tol):
    for i in range(len(gt)):
        if gt[i] == 1 and AT_detected[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, i-tol, -1):  # backward window
                if gt[j] == 0: break
                AT_detected[j] = 1          # fill back
            for j in range(i, i+tol):       # forward window
                if gt[j] == 0: break
                AT_detected[j] = 1          # fill forward
        elif gt[i] == 0:
            anomaly_state = False
```

**Interpretation:** Once the model correctly hits at least one timestep within an anomaly segment, ALL timesteps within `tol` steps of that hit (that are also ground-truth anomaly) get credited as correct predictions. This is materially different from point adjustment (PA) in that:
- It does NOT credit the entire anomaly segment - it only fills within `tol` steps of the actual hit.
- It requires the model to correctly identify at least one timestep (PA requires only 1 hit anywhere in the entire anomaly segment).

**However,** the paper argues this is stricter than PA. In practice, if `tol` is large enough and anomaly segments are contiguous, it approaches PA behaviour. The tolerance `tol` value is set via `--tolerance` arg (default appears to be the window size or pred_len - exact value in parser not confirmed from the summary).

---

## Bugs and Mismatches Found

### Bug 1: Single seed in run.py
`run.py` line: `random_seeds = [20462]` - **hardcoded single seed**. The paper reports results over 3 seeds (mean ± std). The only way to run 3 seeds is to modify `random_seeds` manually or call `run.py` three times with different `--random_seed` values. This is not documented in the README.

### Bug / Opacity 2: F1 metric identity
The code computes four distinct F1 variants per run:
- `f1_gt` - AD on ground-truth future signal (oracle upper bound)
- `f1_gt_adj` - same, with detection_adjustment
- `f1_pred` - AD on predicted future signal (the actual AP task)
- `f1_pred_adj` - same, with detection_adjustment

The paper's Table 1 numbers correspond to `f1_pred` (or `f1_pred_adj`?) - this is ambiguous from the code summary. The output line `F1: {f1_pred_list}` and `AD F1: {f1_pred_adj_list}` suggests the primary metric is `f1_pred` and the adjusted version is secondary. **This requires confirmation by running the code** - if the paper uses `f1_pred_adj` (which has detection_adjustment applied), that is structurally similar to PA despite their claim.

### Bug / Design 3: Threshold on train+test combined energy
`get_threshold()` concatenates train_energy and test_energy, then takes the `100 - anormly_ratio` percentile. Using test data in threshold calibration is a mild form of data leakage - the threshold sees the test distribution before evaluation. This is the standard protocol in AnomalyTransformer and is disclosed, but it means the threshold is not strictly train-only.

### Mismatch 4: Loss coefficient notation
Paper Eq. 8 shows `L_Total = lambda_AAF * L_AAF + lambda_D * L_D + lambda_F * L_F` (pretraining) and `lambda_R * L_R + lambda_AF * L_AF` (main). CLI flags are `--contrastive_loss_coeff`, `--forecast_loss_coeff`, `--cross_attn_loss_coeff`, `--recon_loss_coeff`, `--af_loss_coeff`. The mapping:
- `cross_attn_loss_coeff` = lambda_AAF
- `contrastive_loss_coeff` = lambda_D
- `forecast_loss_coeff` = lambda_F
- `recon_loss_coeff` = lambda_R
- `af_loss_coeff` = lambda_AF

All default to 1.0, consistent with paper.

### Mismatch 5: Epochs are very low
`joint_epochs=5` and `cross_attn_epochs=5`. Ten total training epochs seems very low for a transformer on multivariate TS data. The paper states "training takes at most 1 hour on WADI" which is consistent with very few epochs if each epoch over WADI's 123D signal at batch_size=8 is expensive. This means the model is highly sensitive to learning rate and batch size choices.

### Observation 6: No data download script
The repo contains no `download_data.sh` or similar. Data must be sourced externally. Standard TSAD community `.npy` files are available from the AnomalyTransformer repo (https://github.com/thuml/Anomaly-Transformer) which also provides MBA, WADI, SMD, MSL, etc. in the expected `.npy` format.

---

## Environment Notes

- PyTorch 2.0.0 + CUDA (multi-version CUDA libs in requirements.txt)
- `dgl==1.1.3` is listed but may not be directly used by A2P (could be a leftover dependency)
- Python environment is reproducible via `environment.yml`
- `vus/` directory implements VUS-ROC/VUS-PR metrics (Paparrizos et al. 2022 VLDB) - these are reported alongside F1 but are NOT the primary metric in Table 1

---

## Run Command (exact, for MBA replication)

```bash
# Clone
git clone https://github.com/KU-VGI/AP.git AP/

# Install
cd AP && pip install -r requirements.txt

# Download data (from AnomalyTransformer repo or PhysioNet for MBA)
# Place as: /path/to/data/MBA_train.npy, MBA_test.npy, MBA_test_label.npy

# Run MBA, L_out=100, 3 seeds
for seed in 0 1 2; do
  python -u run.py \
    --random_seed $seed \
    --root_path /path/to/data \
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
    --cross_attn_nheads 1
done
```

**Note:** `run.py` hardcodes `random_seeds = [20462]` - to run multiple seeds, either edit this list or pass `--random_seed` and call the script repeatedly.

---

## Data Availability Summary

| Dataset | Source | Format | Status |
|---------|--------|--------|--------|
| MBA | AnomalyTransformer repo (GitHub) or PhysioNet MIT-BIH SVDB | `.npy` | Publicly available |
| SMD | AnomalyTransformer repo | `.npy` | Publicly available |
| WADI | iTrust Singapore (request form) | CSV -> `.npy` | Gated - requires registration |
| Exathlon | LDAP Zurich / Jacob et al. 2020 arXiv | `.npy` | Publicly available |

AnomalyTransformer dataset pack (thuml) includes MBA, SMD, MSL, SMAP, PSM, SWAT - likely also the pre-processed versions A2P expects.
