---
name: IndustrialJEPA Project Context
description: Complete project state including V6 (CWRU→Paderborn JEPA transfer) and V7 baseline establishment on Forgis/Mechanical-Components HF dataset
type: project
---

IndustrialJEPA is a research project on self-supervised learning (JEPA) for industrial time series.

## Mechanical-JEPA V6 (Completed 2026-04-04)

### CORRECTED V6 Results (JSON-backed, 3-seed, fixed Paderborn API bug)

| Method | CWRU F1 | Paderborn F1 | Transfer Gain | Source |
|--------|---------|-------------|---------------|--------|
| CNN Supervised | 1.000 ± 0.000 | 0.987 ± 0.005 | +0.457 ± 0.020 | transfer_baselines_v6_final.json |
| **JEPA V2 (ours)** | 0.773 ± 0.018 | **0.900 ± 0.008** | **+0.371 ± 0.026** | transfer_baselines_v6_final.json |
| Transformer Supervised | 0.969 ± 0.026 | 0.673 ± 0.063 | +0.144 ± 0.044 | transfer_baselines_v6_final.json |
| Random Init | ~0.412 | 0.529 ± 0.024 | 0.000 | transfer_baselines_v6_final.json |

**KEY FINDING: JEPA@N=10 (0.735) > Transformer@N=all (0.689)** — p=0.034, d=0.92.
This is the primary publishable result using local CWRU (134 samples) and local Paderborn datasets.

### JEPA V2 Architecture (jepa_v2.py)
- Encoder: 4-layer Transformer, d=512, 4 heads, sinusoidal PE
- Input: (B, 3, 4096) → 16 patches of 256 samples
- Mask ratio: 0.625, EMA momentum=0.996, L1 loss + variance reg (lambda=0.1)
- 5 critical components verified by ablation (sine PE, high mask ratio, L1, var reg, EMA)
- Best checkpoint: `mechanical-jepa/checkpoints/jepa_v2_20260401_003619.pt`

### Local Datasets (V6)
- CWRU: `mechanical-jepa/data/bearings/` (134 samples, 4 classes, 12kHz)
- Paderborn: `datasets/data/paderborn/` (K001/KA01/KI01, 64kHz → resample to 20kHz)
- Use `create_paderborn_loaders` (NOT PaderbornDataset constructor directly)

---

## V7: Baseline Establishment (Completed 2026-04-07)

### Dataset: Forgis/Mechanical-Components (HuggingFace)
- ~12,000 samples, 9.5 GB, 16 sources, 5 component types
- **USE PARQUET DIRECTLY** — `load_dataset()` fails due to fsspec glob bug
- Local cache: `/tmp/hf_cache/bearings/` (download with hf_hub_download)
- Signal structure: `row['signal']` is array-of-arrays (n_channels, n_samples)
- Signal: `np.array(row['signal'])[0]` gives channel 0

### Key signal properties (for processing)
- FEMTO: 2560 samples at 25600Hz = 0.1s (very short)
- XJTU-SY: 32768 samples at 25600Hz = 1.28s
- CWRU (HF): 120k-485k samples at 12000Hz = 10-40s
- Paderborn (HF): 256k samples at 64000Hz = 4s
- MAFAULDA: 25600 samples at 50000Hz = 0.512s (short)
- rul_percent = 1 - episode_position exactly (definitional, not predictive)

### V7 Baseline Results (4 task families)

**Task 1: Cross-Domain Fault Classification**
- Setup: Train CWRU+MAFAULDA+SEU → Test Ottawa+Paderborn
- Best: Random Forest F1=0.193 ± 0.021
- Without MAFAULDA: RF F1=0.216 (MAFAULDA hurts — noisy source)
- In-domain: CWRU=0.725, Paderborn=0.872, Ottawa=0.828
- JEPA target: F1 > 0.30 cross-domain

**Task 2: Anomaly Detection (FEMTO)**
- Best: Kurtosis threshold AUROC=0.779 (beats IsolationForest=0.710)
- CNN Autoencoder: 0.414 (poor on 0.1s FEMTO signals)
- JEPA target: AUROC > 0.85

**Task 3: HI Forecasting (FEMTO RMS)**
- H=1: Last-value RMSE=0.351, Random Forest=0.311 (best)
- ARIMA completely fails on non-stationary degradation (RMSE=1.82)
- Kurtosis much harder than RMS to forecast
- JEPA target: RMSE < 0.25 at H=1

**Task 4: RUL Estimation (FEMTO+XJTU-SY)**
- Best real baseline: XGBoost RMSE=0.212
- Constant mean: 0.290 (strong trivial since RUL~uniform on [0,1])
- JEPA target: RMSE < 0.17

### Key Architecture Insight
- rul_percent = 1 - episode_position is definitional → oracle cheat (RMSE=0)
- MAFAULDA: all fault types have nearly identical kurtosis (2.6-2.9) → not good for classification training
- Features for anomaly detection: kurtosis alone > complex ML (physically motivated)
- Deep models don't beat feature-based with limited data (40 CWRU + 140 SEU train samples)

### Code Structure (V7 baselines)
- Baselines dir: `mechanical-jepa/baselines/`
- Data loading: `data_utils.py` (uses local `/tmp/hf_cache/bearings/`)
- Features: `features.py` (18 features: time-domain, freq-domain, envelope)
- Classification: `run_classification_baselines.py`
- Anomaly: `run_anomaly_baselines.py`
- Forecasting: `run_forecasting_baselines.py`
- In-domain: `run_within_source_clf.py`
- Results: `baselines/results/*.json`
- Notebook: `notebooks/07_baseline_establishment.ipynb`
- Figures: `notebooks/plots/fig_baseline_*.{pdf,png}`

**Why:** V7 establishes the bar that JEPA-V7 needs to beat for publication. Cross-domain classification is the primary publishable target.
**How to apply:** For any new JEPA model evaluation, compare against RF cross-domain F1=0.216 (no MAFAULDA) and Kurtosis AUROC=0.779.
