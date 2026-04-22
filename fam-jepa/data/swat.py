"""
SWaT (Secure Water Treatment) Dataset Adapter (V15).

SWaT requires registration at:
  https://itrust.sutd.edu.sg/itrust-labs_datasets/dataset_info/

Dataset stats (when available):
  - 51 sensors
  - 495K normal + 177K attack timesteps
  - 41 labeled attack events

Status: Data NOT available (requires registration + approval).
This adapter provides the interface for when data is obtained.

To use:
  1. Register at the iTrust URL above
  2. Download SWaT_Dataset_Attack_v0.csv and SWaT_Dataset_Normal_v0.csv
  3. Place in datasets/data/swat/
  4. Set SWAT_DATA_DIR below
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Optional, Tuple

try:
    from .config import SWAT_DIR
    SWAT_DATA_DIR = SWAT_DIR
except ImportError:
    SWAT_DATA_DIR = Path('/home/sagemaker-user/IndustrialJEPA/datasets/data/swat')


def check_swat_available() -> bool:
    """Check if SWaT data files are present."""
    normal_file = SWAT_DATA_DIR / 'SWaT_Dataset_Normal_v0.csv'
    attack_file = SWAT_DATA_DIR / 'SWaT_Dataset_Attack_v0.csv'
    return normal_file.exists() and attack_file.exists()


def load_swat(normalize: bool = True) -> Optional[dict]:
    """
    Load SWaT dataset.

    Returns None if data not available.
    Otherwise returns dict with train/test/labels arrays.
    """
    if not check_swat_available():
        print("SWaT data not available. Requires registration at:")
        print("  https://itrust.sutd.edu.sg/itrust-labs_datasets/dataset_info/")
        return None

    import pandas as pd

    # Normal data (for training)
    normal_df = pd.read_csv(SWAT_DATA_DIR / 'SWaT_Dataset_Normal_v0.csv',
                             skiprows=1, header=0)
    # Attack data (for testing)
    attack_df = pd.read_csv(SWAT_DATA_DIR / 'SWaT_Dataset_Attack_v0.csv',
                             skiprows=1, header=0)

    # Select 51 sensor/actuator columns (drop Timestamp and Normal/Attack label)
    sensor_cols = [c for c in normal_df.columns
                   if c not in ['Timestamp', ' Normal/Attack']]
    # Keep only numeric columns
    sensor_cols = [c for c in sensor_cols
                   if normal_df[c].dtype in [np.float64, np.int64, 'float64', 'int64']]

    train = normal_df[sensor_cols].values.astype(np.float32)
    test = attack_df[sensor_cols].values.astype(np.float32)
    labels = (attack_df[' Normal/Attack'].str.strip() == 'Attack').astype(np.int32).values

    if normalize:
        mu = train.mean(axis=0, keepdims=True)
        std = train.std(axis=0, keepdims=True) + 1e-6
        train = (train - mu) / std
        test = (test - mu) / std

    return {
        'train': train,
        'test': test,
        'labels': labels,
        'n_channels': train.shape[1],
        'name': 'SWaT',
        'anomaly_rate': float(labels.mean()),
    }


if __name__ == '__main__':
    if check_swat_available():
        data = load_swat()
        print(f"SWaT: train={data['train'].shape}, test={data['test'].shape}")
        print(f"Anomaly rate: {data['anomaly_rate']:.3f}")
    else:
        print("SWaT data not available (requires registration).")
        print("Skipping SWaT experiments for V15.")
        print("Add to V16 plan: obtain SWaT data and run Phase 5c.")
