# Monash Time Series Forecasting Archive

## Executive Summary
- **Domain**: Multi-domain benchmark archive
- **Task**: Time series forecasting (univariate + multivariate)
- **Size**: 58 datasets, ranging from ~100 to 1M+ time series
- **Sampling Rate**: Varies (hourly, daily, weekly, monthly, yearly)
- **Real vs Synthetic**: Real (mostly)
- **License**: Creative Commons (most datasets)
- **Download URL**: https://forecastingdata.org/ and https://zenodo.org/communities/monash-forecasting-archive
- **Published SOTA**: Extensive — used in N-BEATS, N-HiTS, TimesNet comparisons

## Detailed Description

The Monash Time Series Forecasting Archive is a curated collection of 58 diverse time series datasets in a unified .tsf format, assembled by Rakshitha Godahewa et al. (2021). It covers domains from energy to weather to financial data. This is the standard archive used to benchmark global forecasting models.

### Relevant Datasets Within Monash Archive
| Dataset | Series | Length | Channels | Domain | Sampling |
|---|---|---|---|---|---|
| M4 (Hourly) | 414 | 700–960 | 1 | Economic | Hourly |
| M4 (Daily) | 4,227 | 93–9,919 | 1 | Economic | Daily |
| Electricity (UCI) | 321 | 26,304 | 1 | Energy | Hourly |
| Solar Energy | 137 | 52,560 | 1 | Energy | 10-min |
| Wind Farms | 339 | 283,968 | 1 | Energy | 4-sec |
| Hospital | 767 | 84 | 1 | Healthcare | Monthly |
| Tourism | 1,311 | 11–333 | 1 | Economics | Monthly |
| Weather (Jena) | 21 | 52,696 | 1/21 | Climate | 10-min |

### Key Multivariate Datasets (Most Relevant)
| Dataset | Total Timesteps | Channels | Notes |
|---|---|---|---|
| ETTh1/h2 | 17,420 | 7 | Part of this archive |
| Electricity | 26,304 | 321 | UCI Electricity Load |
| Traffic | 17,544 | 862 | SF Bay road occupancy |
| Weather (Jena) | 52,696 | 21 | Climate station sensors |
| Exchange Rate | 7,588 | 8 | Currency exchange rates |

## Relevance to IndustrialJEPA

### Standard Benchmarks (Time Series Library datasets)
These 5 datasets form the de-facto standard for long-term forecasting comparison:

| Dataset | Channels | Timesteps | Domain | SOTA MSE (H=96) |
|---|---|---|---|---|
| ETTh1 | 7 | 17,420 | Power | ~0.35 (Moirai/TimesFM) |
| ETTm1 | 7 | 69,680 | Power | ~0.30 |
| Weather | 21 | 52,696 | Climate | ~0.15 |
| Electricity (ECL) | 321 | 26,304 | Energy | ~0.14 |
| Traffic | 862 | 17,544 | Transport | ~0.40 |

### Key Insight: Physics Grouping in Standard Benchmarks
| Dataset | Groups | Independence? | Expected PhysMask Benefit |
|---|---|---|---|
| ETTh1 | HV/MV/LV + thermal | No — thermal couples all | Negative (confirmed Exp 46) |
| Weather | 21 sensors, loose coupling | Partial | Negative (confirmed Exp 46) |
| Electricity | 321 households | Yes — largely independent | Positive |
| Traffic | 862 road sensors | Yes — spatially local | Positive |

Traffic and Electricity could be interesting Tier 3+ additions where independence assumption holds.

### Scale for Brain-JEPA Direction
Electricity (321 channels × 26k timesteps) and Traffic (862 channels × 17k timesteps) are the closest standard benchmarks to Brain-JEPA scale (450 channels), but still an order of magnitude smaller in both channel count and samples.

## Download Notes
- All datasets available at: https://zenodo.org/communities/monash-forecasting-archive
- .tsf format; Python reader: `from aeon.datasets import load_forecasting`
- ETT, Weather, Electricity, Traffic: also in Autoformer Google Drive bundle
- Downloader: `datasets/downloaders/download_ett.py` handles ETT; standard benchmarks via Time-Series-Library bundle
