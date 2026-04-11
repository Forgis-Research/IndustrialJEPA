# SPDX-FileCopyrightText: 2026 Industrial JEPA Authors
# SPDX-License-Identifier: MIT

"""ETTh1 Dataset for time series forecasting benchmarks."""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path


class ETTh1Dataset(Dataset):
    """
    ETTh1 dataset for long-term time series forecasting.

    Standard split: 12 months train / 4 months val / 4 months test
    (12*30*24=8640 train, 2880 val, 2880 test, but standard is 60/20/20 of rows)

    Standard ETT split (from Informer paper):
      train: [0, 12*30*24) = [0, 8640)
      val:   [12*30*24 - L, 12*30*24 + 4*30*24) = [8640-L, 11520)
      test:  [12*30*24 + 4*30*24 - L, end) = [11520-L, end)
    """

    def __init__(
        self,
        csv_path: str = "data/ETTh1.csv",
        split: str = "train",
        lookback: int = 96,
        horizon: int = 96,
        scale: bool = True,
        train_mean: np.ndarray = None,
        train_std: np.ndarray = None,
    ):
        self.lookback = lookback
        self.horizon = horizon
        self.split = split

        df = pd.read_csv(csv_path)
        # Drop date column, keep 7 channels
        data = df.iloc[:, 1:].values.astype(np.float32)  # (17420, 7)

        # Standard ETT split boundaries
        n = len(data)  # 17420
        train_end = 12 * 30 * 24  # 8640
        val_end = train_end + 4 * 30 * 24  # 11520

        # Compute normalization from train set
        if train_mean is None or train_std is None:
            self.train_mean = data[:train_end].mean(axis=0)
            self.train_std = data[:train_end].std(axis=0)
            self.train_std[self.train_std < 1e-8] = 1.0
        else:
            self.train_mean = train_mean
            self.train_std = train_std

        if scale:
            data = (data - self.train_mean) / self.train_std

        # Select split (with lookback border for val/test)
        if split == "train":
            self.data = data[:train_end]
        elif split == "val":
            self.data = data[train_end - lookback:val_end]
        elif split == "test":
            self.data = data[val_end - lookback:]
        else:
            raise ValueError(f"Unknown split: {split}")

        self.n_samples = len(self.data) - lookback - horizon + 1
        assert self.n_samples > 0, f"Not enough data for {split}: {len(self.data)} rows, need {lookback + horizon}"

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.lookback]            # (lookback, 7)
        y = self.data[idx + self.lookback:idx + self.lookback + self.horizon]  # (horizon, 7)
        return torch.from_numpy(x), torch.from_numpy(y)

    @property
    def n_channels(self):
        return self.data.shape[1]


def get_etth1_loaders(
    csv_path: str = "data/ETTh1.csv",
    lookback: int = 96,
    horizon: int = 96,
    batch_size: int = 32,
    num_workers: int = 0,
):
    """Create train/val/test dataloaders for ETTh1."""
    train_ds = ETTh1Dataset(csv_path, "train", lookback, horizon)
    val_ds = ETTh1Dataset(csv_path, "val", lookback, horizon,
                          train_mean=train_ds.train_mean, train_std=train_ds.train_std)
    test_ds = ETTh1Dataset(csv_path, "test", lookback, horizon,
                           train_mean=train_ds.train_mean, train_std=train_ds.train_std)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader, train_ds


if __name__ == "__main__":
    # Sanity check
    for horizon in [96, 192, 336, 720]:
        train_loader, val_loader, test_loader, train_ds = get_etth1_loaders(
            csv_path="data/ETTh1.csv", horizon=horizon
        )
        print(f"\nH={horizon}:")
        print(f"  Train: {len(train_loader.dataset)} samples, {len(train_loader)} batches")
        print(f"  Val:   {len(val_loader.dataset)} samples, {len(val_loader)} batches")
        print(f"  Test:  {len(test_loader.dataset)} samples, {len(test_loader)} batches")

        x, y = next(iter(train_loader))
        print(f"  x shape: {x.shape}, y shape: {y.shape}")
        print(f"  x mean: {x.mean():.4f}, x std: {x.std():.4f}")
