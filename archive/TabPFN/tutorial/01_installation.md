# TabPFN-TS Installation Guide

## Quick Install (Recommended)

```bash
# Create a fresh environment (optional but recommended)
conda create -n tabpfn python=3.11
conda activate tabpfn

# Install TabPFN-TS
pip install tabpfn-time-series
```

## Full Install (With All Extensions)

```bash
# Full TabPFN ecosystem
pip install tabpfn

# Additional dependencies for experiments
pip install pandas numpy matplotlib scikit-learn
pip install statsmodels  # For ARIMA baselines
pip install prophet      # For Prophet baselines (optional)
```

## Hardware Requirements

| Setup | GPU Memory | CPU Only Feasible? |
|-------|------------|-------------------|
| Small datasets (<1000 samples) | None required | Yes |
| Medium datasets (1k-10k) | ~4GB | Slow but works |
| Large datasets (10k-50k) | ~8GB | Not recommended |

**Note**: TabPFN-TS defaults to using the TabPFN cloud client, which offloads computation. For local inference, you need a GPU.

## Verify Installation

```python
# Quick verification
import pandas as pd
import numpy as np
from tabpfn_time_series import TabPFNTSPipeline, TabPFNMode

# Generate toy data
np.random.seed(42)
y = np.sin(np.linspace(0, 4*np.pi, 100)) + np.random.normal(0, 0.1, 100)

# Prepare data (DataFrame with timestamp and target columns)
context_df = pd.DataFrame({
    'item_id': ['test'] * len(y),
    'timestamp': pd.date_range('2024-01-01', periods=len(y), freq='s'),
    'target': y
})

# Create pipeline (uses cloud API - no GPU needed)
pipeline = TabPFNTSPipeline(tabpfn_mode=TabPFNMode.CLIENT)

# Forecast
predictions_df = pipeline.predict_df(context_df=context_df, prediction_length=10)

print(f"Input shape: {y.shape}")
print(f"Prediction DataFrame shape: {predictions_df.shape}")
print(f"Columns: {predictions_df.columns.tolist()}")
print(f"First 3 median predictions: {predictions_df['0.5'].values[:3]}")
```

Expected output:
```
Input shape: (100,)
Prediction DataFrame shape: (10, 9)
Columns: ['0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9']
First 3 median predictions: [x.xxx, x.xxx, x.xxx]
```

## Troubleshooting

### "No module named 'tabpfn_ts'"
```bash
pip install --upgrade tabpfn-time-series
```

### GPU not detected
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
```

### Using cloud inference (no local GPU)
TabPFN-TS uses the TabPFN client by default, which sends data to Prior Labs' cloud for inference. To disable telemetry:

```bash
export TABPFN_DISABLE_TELEMETRY=1
```

## Project-Specific Setup

For this IndustrialJEPA project:

```bash
# From project root
cd TabPFN

# Install in development mode
pip install -e .

# Or just ensure dependencies are available
pip install tabpfn-time-series pandas numpy matplotlib scikit-learn
```

## Next Steps

1. **02_quickstart.ipynb** - First forecasts with synthetic data
2. **03_covariates.ipynb** - Adding external features
3. **04_mechanical.ipynb** - Apply to real mechanical system data
