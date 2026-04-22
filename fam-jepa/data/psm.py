"""
PSM (Pooled Server Metrics) Dataset Adapter.

eBay server machine dataset used in MTS-JEPA, Anomaly Transformer, etc.
  - 25 channels (after dropping near-constant + timestamp)
  - ~132K train / ~87K test timesteps
  - ~27% anomaly rate in test

NOTE: PSM is a genuinely single continuous stream (one server pool).
There are no independent entities to split by.  For predictor finetuning,
use a chronological split of the test stream with a gap of >=window_size
timesteps between train/val/test to prevent temporal leakage.

Data source: RANSynCoders repo (eBay/RANSynCoders).
Download: run `paper-replications/mts-jepa/download_datasets.py` first,
  or place train.npy/test.npy/test_labels.npy in PSM_DATA_DIR.

Standard window size: 100 timesteps (matching SMAP/MSL convention).
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Optional, Tuple

try:
    from .config import PSM_DIR
    PSM_DATA_DIR = PSM_DIR
except ImportError:
    PSM_DATA_DIR = Path('/home/sagemaker-user/IndustrialJEPA/paper-replications/mts-jepa/data/PSM')

WINDOW_SIZE = 100
STRIDE = 1
TRAIN_STRIDE = 10


def check_psm_available() -> bool:
    """Check if PSM data files are present (either .npy or .csv)."""
    return (PSM_DATA_DIR / 'train.npy').exists() or (PSM_DATA_DIR / 'train.csv').exists()


def load_psm(normalize: bool = True, drop_constant: bool = True,
             constant_threshold: float = 1e-8) -> dict:
    """
    Load PSM dataset. Returns dict with train/test arrays and labels.

    If only CSVs exist (not yet converted to .npy), converts and caches them.
    """
    npy_train = PSM_DATA_DIR / 'train.npy'
    npy_test = PSM_DATA_DIR / 'test.npy'
    npy_labels = PSM_DATA_DIR / 'test_labels.npy'

    if npy_train.exists():
        train = np.load(npy_train).astype(np.float32)
        test = np.load(npy_test).astype(np.float32)
        labels = np.load(npy_labels).astype(np.int32)
    else:
        # Convert from CSV
        import pandas as pd
        train_df = pd.read_csv(PSM_DATA_DIR / 'train.csv')
        test_df = pd.read_csv(PSM_DATA_DIR / 'test.csv')
        label_df = pd.read_csv(PSM_DATA_DIR / 'test_label.csv')

        # Drop timestamp column
        for df in [train_df, test_df]:
            for col in list(df.columns):
                if 'time' in col.lower():
                    df.drop(columns=[col], inplace=True)
                    break

        train = np.nan_to_num(train_df.values, nan=0.0).astype(np.float32)
        test = np.nan_to_num(test_df.values, nan=0.0).astype(np.float32)
        labels = label_df.iloc[:, -1].values.astype(np.int32)

        # Cache as .npy for next time
        np.save(npy_train, train)
        np.save(npy_test, test)
        np.save(npy_labels, labels)

    # Align test/labels length (PSM test and labels can differ by 1)
    min_len = min(len(test), len(labels))
    test = test[:min_len]
    labels = labels[:min_len]

    # Drop near-constant channels
    if drop_constant:
        var = train.var(axis=0)
        keep = var > constant_threshold
        if keep.sum() < train.shape[1]:
            print(f"  PSM: dropping {train.shape[1] - keep.sum()} constant channels "
                  f"({train.shape[1]} -> {keep.sum()})")
            train = train[:, keep]
            test = test[:, keep]

    mu, std = None, None
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
        'name': 'PSM',
        'mu': mu,
        'std': std,
        'anomaly_rate': float(labels.mean()),
    }


# Reuse SlidingWindowDataset and AnomalyPretrainDataset from smap_msl
# Import at function level to avoid circular deps
def get_psm_dataloader(n_samples: int = 50000, batch_size: int = 64,
                       seed: int = 42) -> Tuple[DataLoader, dict]:
    """Load PSM and return DataLoader for pretraining."""
    from .smap_msl import AnomalyPretrainDataset, collate_anomaly_pretrain
    data = load_psm()
    ds = AnomalyPretrainDataset(data['train'], n_samples=n_samples, seed=seed)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True,
                        collate_fn=collate_anomaly_pretrain, num_workers=0)
    return loader, data


if __name__ == '__main__':
    print("Testing PSM data adapter...")
    if not check_psm_available():
        print("PSM data not found. Run paper-replications/mts-jepa/download_datasets.py first.")
    else:
        data = load_psm()
        print(f"PSM: train={data['train'].shape}, test={data['test'].shape}")
        print(f"Channels: {data['n_channels']}, Anomaly rate: {data['anomaly_rate']:.3f}")
        print("Adapter test PASSED")
