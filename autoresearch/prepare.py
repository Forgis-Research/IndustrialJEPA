#!/usr/bin/env python3
"""
Data preparation for ETT benchmark experiments.
This file is NOT modified by the agent.

Downloads ETTh1 dataset and prepares it for multi-step forecasting.

Usage:
    python prepare.py
"""

import os
import sys
from pathlib import Path
import urllib.request

import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader

# ============================================================================
# ETT DATASET
# ============================================================================

ETT_URL = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv"


def download_ett(data_dir: Path = Path("data")):
    """Download ETTh1 dataset if not present."""
    data_dir.mkdir(exist_ok=True)
    filepath = data_dir / "ETTh1.csv"

    if not filepath.exists():
        print(f"Downloading ETTh1 from {ETT_URL}...")
        urllib.request.urlretrieve(ETT_URL, filepath)
        print(f"Saved to {filepath}")
    else:
        print(f"ETTh1 already exists at {filepath}")

    return filepath


class ETTDataset(Dataset):
    """ETT dataset for multi-step forecasting."""

    def __init__(self, data: np.ndarray, seq_len: int = 96, pred_len: int = 96):
        """
        Args:
            data: (T, num_features) normalized time series
            seq_len: input sequence length (context)
            pred_len: prediction horizon
        """
        self.data = torch.FloatTensor(data)
        self.seq_len = seq_len
        self.pred_len = pred_len

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_len]  # (seq_len, num_features)
        y = self.data[idx + self.seq_len:idx + self.seq_len + self.pred_len]  # (pred_len, num_features)
        return {'x': x, 'y': y}


def prepare_data(
    seq_len: int = 96,
    pred_len: int = 96,
    batch_size: int = 32,
):
    """Prepare ETTh1 dataloaders for multi-step forecasting.

    Standard ETT split:
    - Train: 12 months (8640 hours)
    - Val: 4 months (2880 hours)
    - Test: 4 months (2880 hours)
    """

    print(f"Preparing ETTh1 data...")
    print(f"  Sequence length (input): {seq_len}")
    print(f"  Prediction length (output): {pred_len}")
    print(f"  Batch size: {batch_size}")

    # Download if needed
    filepath = download_ett()

    # Load data
    df = pd.read_csv(filepath)

    # Drop date column, keep only numerical features
    # Columns: date, HUFL, HULL, MUFL, MULL, LUFL, LULL, OT
    data = df.drop('date', axis=1).values.astype(np.float32)

    # Standard ETT split points
    train_end = 12 * 30 * 24  # 12 months
    val_end = train_end + 4 * 30 * 24  # +4 months

    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]

    # Normalize using train statistics (important!)
    train_mean = train_data.mean(axis=0)
    train_std = train_data.std(axis=0)

    train_data = (train_data - train_mean) / (train_std + 1e-8)
    val_data = (val_data - train_mean) / (train_std + 1e-8)
    test_data = (test_data - train_mean) / (train_std + 1e-8)

    print(f"Data shapes:")
    print(f"  Train: {train_data.shape}")
    print(f"  Val: {val_data.shape}")
    print(f"  Test: {test_data.shape}")
    print(f"  Features: {data.shape[1]}")

    # Create datasets
    train_dataset = ETTDataset(train_data, seq_len, pred_len)
    val_dataset = ETTDataset(val_data, seq_len, pred_len)
    test_dataset = ETTDataset(test_data, seq_len, pred_len)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    info = {
        'num_features': data.shape[1],
        'seq_len': seq_len,
        'pred_len': pred_len,
        'train_size': len(train_dataset),
        'val_size': len(val_dataset),
        'test_size': len(test_dataset),
        'train_mean': train_mean,
        'train_std': train_std,
    }

    print(f"Dataset sizes:")
    print(f"  Train samples: {info['train_size']}")
    print(f"  Val samples: {info['val_size']}")
    print(f"  Test samples: {info['test_size']}")

    # Save config for train.py
    torch.save({
        'info': info,
        'seq_len': seq_len,
        'pred_len': pred_len,
    }, 'data_info.pt')

    print("✓ Data prepared. Saved info to data_info.pt")

    return train_loader, val_loader, test_loader, info


def get_dataloaders(batch_size: int = 32, seq_len: int = None, pred_len: int = None):
    """Load prepared dataloaders (call after prepare_data or with custom lengths)."""

    # Load saved config
    data_info = torch.load('data_info.pt', weights_only=False)
    saved_seq_len = data_info['seq_len']
    saved_pred_len = data_info['pred_len']

    # Use saved or override
    seq_len = seq_len or saved_seq_len
    pred_len = pred_len or saved_pred_len

    # Re-prepare if lengths changed
    if seq_len != saved_seq_len or pred_len != saved_pred_len:
        return prepare_data(seq_len, pred_len, batch_size)

    # Load data
    filepath = Path("data/ETTh1.csv")
    df = pd.read_csv(filepath)
    data = df.drop('date', axis=1).values.astype(np.float32)

    # Standard ETT split
    train_end = 12 * 30 * 24
    val_end = train_end + 4 * 30 * 24

    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]

    # Normalize
    train_mean = train_data.mean(axis=0)
    train_std = train_data.std(axis=0)

    train_data = (train_data - train_mean) / (train_std + 1e-8)
    val_data = (val_data - train_mean) / (train_std + 1e-8)
    test_data = (test_data - train_mean) / (train_std + 1e-8)

    # Create datasets
    train_dataset = ETTDataset(train_data, seq_len, pred_len)
    val_dataset = ETTDataset(val_data, seq_len, pred_len)
    test_dataset = ETTDataset(test_data, seq_len, pred_len)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    info = {
        'num_features': data.shape[1],
        'seq_len': seq_len,
        'pred_len': pred_len,
        'train_size': len(train_dataset),
        'val_size': len(val_dataset),
        'test_size': len(test_dataset),
        'train_mean': train_mean,
        'train_std': train_std,
    }

    return train_loader, val_loader, test_loader, info


if __name__ == "__main__":
    prepare_data()
