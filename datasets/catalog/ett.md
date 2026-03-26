# ETT Dataset (Electricity Transformer Temperature)

## Executive Summary
- **Domain**: Power Systems / Electrical Infrastructure
- **Task**: Multivariate time series forecasting (long-term)
- **Size**: 70,080 timesteps × 7 channels (ETTh: hourly; ETTm: 15-minute)
- **Sampling Rate**: 1 sample/hour (ETTh1/h2) or 1 sample/15min (ETTm1/m2)
- **Real vs Synthetic**: Real — from Chinese electricity distribution stations, 2016–2018
- **License**: Public domain (open research use)
- **Download URL**: https://github.com/zhouhaoyi/ETDataset (CSV files in repo)
- **Published SOTA**: Extensive — standard benchmark for long-term forecasting papers

## Detailed Description

ETT (Electricity Transformer Temperature) was introduced with the Informer paper (AAAI 2021) and has become the most widely-used benchmark in long-term time series forecasting. The dataset contains oil temperature and load measurements from electricity transformers.

### Dataset Variants
| Variant | Timesteps | Sampling | Total Duration |
|---|---|---|---|
| ETTh1 | 17,420 | 1/hour | ~2 years |
| ETTh2 | 17,420 | 1/hour | ~2 years |
| ETTm1 | 69,680 | 1/15 min | ~2 years |
| ETTm2 | 69,680 | 1/15 min | ~2 years |

### Standard Train/Val/Test Split
- ETTh: 12 months / 4 months / 4 months
- ETTm: 12 months / 4 months / 4 months

## Features
| Feature Name | Type | Unit | Description | Physics Group |
|---|---|---|---|---|
| HUFL | load | kW | High UseFul Load | HV power |
| HULL | load | kW | High UseLess Load | HV power |
| MUFL | load | kW | Middle UseFul Load | MV power |
| MULL | load | kW | Middle UseLess Load | MV power |
| LUFL | load | kW | Low UseFul Load | LV power |
| LULL | load | kW | Low UseLess Load | LV power |
| OT | temperature | °C | Oil Temperature (target) | thermal |

### Physics Groups
```python
ETT_GROUPS = {
    "HV_power":  [0, 1],    # HUFL, HULL
    "MV_power":  [2, 3],    # MUFL, MULL
    "LV_power":  [4, 5],    # LUFL, LULL
    "thermal":   [6],       # OT
}
```

## Published Benchmarks / SOTA (ETTh1 Multivariate Forecasting, MSE)
| Method | H=96 | H=192 | H=336 | H=720 | Paper | Year |
|---|---|---|---|---|---|---|
| Informer | 0.865 | 1.083 | 1.099 | 1.068 | Zhou et al. | 2021 |
| Autoformer | 0.449 | 0.500 | 0.521 | 0.514 | Wu et al. | 2021 |
| PatchTST | 0.370 | 0.413 | 0.422 | 0.447 | Nie et al. | 2022 |
| DLinear | 0.386 | 0.437 | 0.481 | 0.456 | Zeng et al. | 2023 |
| iTransformer | 0.386 | 0.441 | 0.487 | 0.503 | Liu et al. | 2023 |
| TimesNet | 0.384 | 0.436 | 0.491 | 0.521 | Wu et al. | 2023 |
| TimesFM | ~0.36 | ~0.40 | ~0.43 | ~0.45 | Google | 2024 |
| Moirai | ~0.35 | ~0.39 | ~0.42 | ~0.44 | Salesforce | 2024 |

Note: MSE values computed on normalized data. Lower = better. Results vary slightly by implementation.

## Relevance to IndustrialJEPA

### Current Usage
Used as Tier 3 in the Rapid Evaluation Suite. Key experimental result: PhysMask performs **worse** than Full-Attention (-1.3%), showing that cross-group interactions (thermal ↔ load) are important for this system. This is a clean negative result for the paper.

### Physics Grouping Assessment
**Moderate** — the HV/MV/LV grouping has physical meaning (voltage levels in the transformer), but OT (oil temperature) is strongly coupled to all load groups. Cross-group attention is informationally necessary here, which explains why physics masking hurts.

### Verdict for Tier 3
**Perfect** — large published SOTA, real data, clear negative result for physics masking. Keep as Tier 3. Transfer test: ETTh1 → ETTh2.

### Forecasting Task
- Input length: 96 (standard), 336, 512, or 720 timesteps
- Prediction horizons: 96, 192, 336, 720 timesteps
- Standard metric: MSE + MAE on the full multivariate output

## Download Notes
- CSV files directly in GitHub repo (no download script needed)
- URL: https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv
- 4 files: ETTh1.csv, ETTh2.csv, ETTm1.csv, ETTm2.csv
- Also available in Autoformer/Time-Series-Library Google Drive bundle
- Downloader: `datasets/downloaders/download_ett.py`
