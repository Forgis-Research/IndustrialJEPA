"""
Data utilities for MTS-JEPA replication.
Implements: dataset loading, constant channel removal, RevIN, multi-scale views,
non-overlapping window pairs, and downstream 6:2:2 splits.
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# Expected effective dimensions after constant channel removal (Table 5)
EXPECTED_DIMS = {
    "MSL": 33,
    "SMAP": 24,
    "SWaT": 40,
    "PSM": 25,
}


def load_dataset(name):
    """
    Load a dataset by name. Returns (train_data, test_data, test_labels).

    train_data: (N_train, V) float32
    test_data: (N_test, V) float32
    test_labels: (N_test,) int32 — point-level binary anomaly labels
    """
    ds_dir = os.path.join(DATA_DIR, name)

    train = np.load(os.path.join(ds_dir, "train.npy")).astype(np.float32)
    test = np.load(os.path.join(ds_dir, "test.npy")).astype(np.float32)
    labels = np.load(os.path.join(ds_dir, "test_labels.npy")).astype(np.int32)

    # Ensure 2D
    if train.ndim == 1:
        train = train.reshape(-1, 1)
    if test.ndim == 1:
        test = test.reshape(-1, 1)

    # Truncate labels to match test length
    labels = labels[:len(test)]

    print(f"Loaded {name}: train {train.shape}, test {test.shape}, "
          f"anomaly rate {labels.mean():.4f}")

    return train, test, labels


def remove_constant_channels(train_data, test_data, eps=1e-8):
    """
    Remove channels with zero (or near-zero) variance on training data.
    Returns (train_filtered, test_filtered, mask).
    """
    variances = np.var(train_data, axis=0)
    mask = variances > eps

    n_original = train_data.shape[1]
    n_kept = mask.sum()
    print(f"  Constant channel removal: {n_original} -> {n_kept} "
          f"(removed {n_original - n_kept})")

    return train_data[:, mask], test_data[:, mask], mask


def make_non_overlapping_windows(data, window_length=100):
    """
    Partition time series into non-overlapping windows of length T_w.

    data: (T, V) array
    Returns: (N_windows, T_w, V) array
    """
    T, V = data.shape
    n_windows = T // window_length
    # Trim to exact multiple
    data = data[:n_windows * window_length]
    windows = data.reshape(n_windows, window_length, V)
    return windows


def make_window_labels(labels, window_length=100):
    """
    Create window-level labels: y = 1 if ANY point in window is anomalous.

    labels: (T,) point-level binary labels
    Returns: (N_windows,) window-level labels
    """
    n_windows = len(labels) // window_length
    labels = labels[:n_windows * window_length]
    labels = labels.reshape(n_windows, window_length)
    return (labels.max(axis=1) > 0).astype(np.int32)


class RevIN(torch.nn.Module):
    """
    Reversible Instance Normalization (Kim et al., ICLR 2022).
    Per-window, per-channel normalization.
    """
    def __init__(self, num_features, eps=1e-5, affine=True):
        super().__init__()
        self.eps = eps
        self.affine = affine
        self.num_features = num_features
        if affine:
            self.gamma = torch.nn.Parameter(torch.ones(1, 1, num_features))
            self.beta = torch.nn.Parameter(torch.zeros(1, 1, num_features))
        self._mean = None
        self._std = None

    def forward(self, x):
        """
        Normalize. x: (B, T, V)
        """
        self._mean = x.mean(dim=1, keepdim=True)  # (B, 1, V)
        self._std = x.std(dim=1, keepdim=True) + self.eps  # (B, 1, V)
        x_norm = (x - self._mean) / self._std
        if self.affine:
            x_norm = x_norm * self.gamma + self.beta
        return x_norm

    def inverse(self, x):
        """
        Denormalize using cached statistics. x: (B, T, V)
        """
        if self.affine:
            x = (x - self.beta) / self.gamma
        return x * self._std + self._mean


def create_views(window, n_patches=5, patch_length=20):
    """
    Create multi-scale views from a normalized window.

    window: (B, T_w, V) normalized window, T_w = n_patches * patch_length

    Returns:
        fine_view: (B, P, L, V) — P patches of length L
        coarse_view: (B, 1, L, V) — average every P consecutive time points
    """
    B, T, V = window.shape
    assert T == n_patches * patch_length, f"T={T} != {n_patches}*{patch_length}"

    # Fine view: split into P non-overlapping patches
    fine_view = window.reshape(B, n_patches, patch_length, V)

    # Coarse view: average every n_patches consecutive time steps
    # Reshape to (B, patch_length, n_patches, V), average over the n_patches dim
    # This gives (B, patch_length, V) — a single downsampled window
    window_reshaped = window.reshape(B, patch_length, n_patches, V)
    coarse_view = window_reshaped.mean(dim=2)  # (B, patch_length, V)
    coarse_view = coarse_view.unsqueeze(1)  # (B, 1, L, V)

    return fine_view, coarse_view


class PretrainDataset(Dataset):
    """
    Pre-training dataset: non-overlapping context-target window pairs.

    Constructs (X_t, X_{t+1}) pairs where X_t is context and X_{t+1} is target.
    """
    def __init__(self, data, window_length=100):
        """
        data: (T, V) continuous multivariate time series
        """
        windows = make_non_overlapping_windows(data, window_length)
        # Create consecutive pairs
        self.contexts = windows[:-1]  # X_t
        self.targets = windows[1:]    # X_{t+1}

    def __len__(self):
        return len(self.contexts)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.contexts[idx], dtype=torch.float32),
            torch.tensor(self.targets[idx], dtype=torch.float32),
        )


class DownstreamDataset(Dataset):
    """
    Downstream dataset: context windows with window-level labels.
    """
    def __init__(self, data, labels, window_length=100):
        """
        data: (T, V) time series
        labels: (T,) point-level binary labels
        """
        self.windows = make_non_overlapping_windows(data, window_length)
        self.labels = make_window_labels(labels, window_length)

        # Create context-label pairs: context X_t predicts label of X_{t+1}
        self.contexts = self.windows[:-1]
        self.target_labels = self.labels[1:]

    def __len__(self):
        return len(self.contexts)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.contexts[idx], dtype=torch.float32),
            torch.tensor(self.target_labels[idx], dtype=torch.float32),
        )


def prepare_data(dataset_name, window_length=100, batch_size=64,
                 pretrain_val_ratio=0.1, seed=42):
    """
    Full data preparation pipeline.

    Returns dict with:
        - train_data, test_data: raw numpy arrays (after channel removal)
        - n_vars: number of effective variables
        - pretrain_train_loader, pretrain_val_loader
        - downstream splits (6:2:2 of test set)
    """
    rng = np.random.RandomState(seed)

    # Load raw data
    train_data, test_data, test_labels = load_dataset(dataset_name)

    # Remove constant channels
    train_data, test_data, channel_mask = remove_constant_channels(train_data, test_data)
    n_vars = train_data.shape[1]

    # Check against expected dimensions
    if dataset_name in EXPECTED_DIMS:
        expected = EXPECTED_DIMS[dataset_name]
        if n_vars != expected:
            print(f"  WARNING: Expected {expected} dims for {dataset_name}, got {n_vars}")

    # Pre-training: split training data 9:1 chronologically
    pretrain_split = int(len(train_data) * (1 - pretrain_val_ratio))
    pretrain_train = PretrainDataset(train_data[:pretrain_split], window_length)
    pretrain_val = PretrainDataset(train_data[pretrain_split:], window_length)

    pretrain_train_loader = DataLoader(
        pretrain_train, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=True, drop_last=True
    )
    pretrain_val_loader = DataLoader(
        pretrain_val, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=True
    )

    # Downstream: 6:2:2 chronological split of test data
    test_windows = make_non_overlapping_windows(test_data, window_length)
    test_window_labels = make_window_labels(test_labels, window_length)

    # Context-label pairs: X_t -> y_{t+1}
    n_pairs = len(test_windows) - 1
    contexts = test_windows[:-1]
    target_labels = test_window_labels[1:]

    split1 = int(n_pairs * 0.6)
    split2 = int(n_pairs * 0.8)

    ds_train = (contexts[:split1], target_labels[:split1])
    ds_val = (contexts[split1:split2], target_labels[split1:split2])
    ds_test = (contexts[split2:], target_labels[split2:])

    print(f"  Downstream splits: train={split1}, val={split2-split1}, test={n_pairs-split2}")
    print(f"  Anomaly rates: train={target_labels[:split1].mean():.3f}, "
          f"val={target_labels[split1:split2].mean():.3f}, "
          f"test={target_labels[split2:].mean():.3f}")

    return {
        "dataset_name": dataset_name,
        "n_vars": n_vars,
        "channel_mask": channel_mask,
        "train_data": train_data,
        "test_data": test_data,
        "test_labels": test_labels,
        "pretrain_train_loader": pretrain_train_loader,
        "pretrain_val_loader": pretrain_val_loader,
        "downstream_train": ds_train,
        "downstream_val": ds_val,
        "downstream_test": ds_test,
        "window_length": window_length,
    }
