"""
Bearing fault dataset for Mechanical-JEPA.

Loads CWRU and IMS data with proper train/test split BY BEARING (not by window).
This prevents data leakage from correlated windows of the same bearing.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from scipy.io import loadmat
import torch
from torch.utils.data import Dataset, DataLoader


# =============================================================================
# Signal Loading Functions
# =============================================================================

def load_cwru_signal(data_dir: Path, file_id: str) -> Optional[np.ndarray]:
    """
    Load CWRU bearing signal.

    Returns: (n_samples, n_channels) array or None if not found
    """
    mat_file = data_dir / 'raw' / 'cwru' / f'{file_id}.mat'
    if not mat_file.exists():
        return None

    try:
        data = loadmat(str(mat_file), squeeze_me=True)
        channels = []

        # CWRU has DE (drive end), FE (fan end), BA (base) channels
        for suffix in ['_DE_time', '_FE_time', '_BA_time']:
            for key in data.keys():
                if suffix in key:
                    channels.append(data[key])
                    break

        if not channels:
            return None

        # Stack channels, handle different lengths by truncating to min
        min_len = min(len(c) for c in channels)
        signal = np.stack([c[:min_len] for c in channels], axis=1)
        return signal.astype(np.float32)

    except Exception as e:
        print(f"Error loading {file_id}: {e}")
        return None


def load_ims_signal(filepath: Path) -> Optional[np.ndarray]:
    """
    Load IMS bearing signal.

    Returns: (n_samples, n_channels) array or None if not found
    """
    try:
        data = np.loadtxt(filepath, dtype=np.float32)
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        return data
    except Exception as e:
        return None


# =============================================================================
# Dataset Class
# =============================================================================

class BearingDataset(Dataset):
    """
    Bearing fault dataset with sliding windows.

    Key design choices:
    - Split by bearing_id, not by window (prevents data leakage)
    - Z-score normalization per channel
    - Returns (n_channels, window_size) tensors for transformer input
    - Pads/truncates to fixed channel count for batching
    """

    def __init__(
        self,
        data_dir: Path,
        bearing_ids: List[str],
        episodes_df: pd.DataFrame,
        window_size: int = 4096,
        stride: int = 2048,
        normalize: bool = True,
        max_windows_per_bearing: Optional[int] = None,
        n_channels: int = 3,  # Fixed channel count for batching
    ):
        """
        Args:
            data_dir: Path to data/bearings directory
            bearing_ids: List of bearing IDs to include
            episodes_df: DataFrame with episode metadata
            window_size: Samples per window
            stride: Stride between windows
            normalize: Whether to z-score normalize
            max_windows_per_bearing: Limit windows per bearing (for balancing)
            n_channels: Fixed number of channels (pads/truncates to this)
        """
        self.data_dir = Path(data_dir)
        self.window_size = window_size
        self.stride = stride
        self.normalize = normalize
        self.n_channels = n_channels

        # Filter episodes to selected bearings
        self.episodes = episodes_df[episodes_df['bearing_id'].isin(bearing_ids)].copy()

        # Build window index
        self.windows = []  # List of (bearing_id, dataset, start_idx, fault_label)
        self._signals = {}  # Cache loaded signals

        for _, row in self.episodes.iterrows():
            bearing_id = row['bearing_id']
            dataset = row['dataset']
            n_samples = row['n_samples']
            fault_label = row['fault_label']

            # Calculate window positions
            n_windows = (n_samples - window_size) // stride + 1
            if max_windows_per_bearing:
                n_windows = min(n_windows, max_windows_per_bearing)

            for i in range(n_windows):
                start_idx = i * stride
                self.windows.append({
                    'bearing_id': bearing_id,
                    'dataset': dataset,
                    'start_idx': start_idx,
                    'fault_label': fault_label,
                    'measurement_id': row.get('measurement_id', bearing_id),
                })

        print(f"BearingDataset: {len(self.windows)} windows from {len(bearing_ids)} bearings")

    def _load_signal(self, bearing_id: str, dataset: str, measurement_id: str) -> Optional[np.ndarray]:
        """Load and cache signal."""
        cache_key = f"{dataset}_{measurement_id}"

        if cache_key not in self._signals:
            if dataset == 'cwru':
                signal = load_cwru_signal(self.data_dir, bearing_id)
            elif dataset == 'ims':
                # IMS: bearing_id is like "ims_1st_test", measurement_id is timestamp
                test_name = bearing_id.replace('ims_', '')
                filepath = self.data_dir / 'raw' / 'ims' / test_name / measurement_id
                signal = load_ims_signal(filepath)
            else:
                signal = None

            self._signals[cache_key] = signal

        return self._signals[cache_key]

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        """
        Returns:
            signal: (n_channels, window_size) tensor
            fault_label: int
            bearing_id: str
        """
        window_info = self.windows[idx]
        bearing_id = window_info['bearing_id']
        dataset = window_info['dataset']
        start_idx = window_info['start_idx']
        fault_label = window_info['fault_label']
        measurement_id = window_info['measurement_id']

        # Load signal
        signal = self._load_signal(bearing_id, dataset, measurement_id)

        if signal is None:
            # Return zeros if signal not found (shouldn't happen)
            return torch.zeros(self.n_channels, self.window_size), fault_label, bearing_id

        # Extract window
        end_idx = start_idx + self.window_size
        if end_idx > len(signal):
            end_idx = len(signal)
            start_idx = end_idx - self.window_size

        window = signal[start_idx:end_idx].T  # (actual_channels, window_size)

        # Pad or truncate to fixed channel count
        actual_channels = window.shape[0]
        if actual_channels < self.n_channels:
            # Pad with zeros
            padding = np.zeros((self.n_channels - actual_channels, window.shape[1]), dtype=np.float32)
            window = np.vstack([window, padding])
        elif actual_channels > self.n_channels:
            # Truncate to first n_channels
            window = window[:self.n_channels]

        # Normalize per channel
        if self.normalize:
            mean = window.mean(axis=1, keepdims=True)
            std = window.std(axis=1, keepdims=True) + 1e-8
            window = (window - mean) / std

        return torch.tensor(window, dtype=torch.float32), fault_label, bearing_id


# =============================================================================
# Data Loading Utilities
# =============================================================================

def create_dataloaders(
    data_dir: Path,
    batch_size: int = 64,
    window_size: int = 4096,
    stride: int = 2048,
    test_ratio: float = 0.2,
    seed: int = 42,
    num_workers: int = 0,
    dataset_filter: Optional[str] = None,
    n_channels: int = 3,  # Fixed channel count for batching
    stratified: bool = True,  # Stratify by fault type
) -> Tuple[DataLoader, DataLoader, Dict]:
    """
    Create train and test dataloaders with proper split by bearing.

    Args:
        data_dir: Path to data/bearings directory
        batch_size: Batch size
        window_size: Window size in samples
        stride: Stride between windows
        test_ratio: Fraction of bearings for test set
        seed: Random seed for reproducibility
        num_workers: DataLoader workers
        dataset_filter: 'cwru', 'ims', or None for all
        stratified: If True, stratify split by fault type

    Returns:
        train_loader, test_loader, info_dict
    """
    data_dir = Path(data_dir)

    # Load episodes metadata
    episodes_df = pd.read_parquet(data_dir / 'bearing_episodes.parquet')

    # Filter by dataset if specified
    if dataset_filter:
        episodes_df = episodes_df[episodes_df['dataset'] == dataset_filter]

    np.random.seed(seed)

    if stratified:
        # Stratified split: ensure each fault type appears in both train and test
        train_bearings = []
        test_bearings = []

        # Group bearings by fault type
        for fault_type in episodes_df['fault_type'].unique():
            fault_bearings = episodes_df[episodes_df['fault_type'] == fault_type]['bearing_id'].unique().tolist()
            np.random.shuffle(fault_bearings)

            n_test = max(1, int(len(fault_bearings) * test_ratio))
            test_bearings.extend(fault_bearings[:n_test])
            train_bearings.extend(fault_bearings[n_test:])

    else:
        # Simple random split
        bearings = episodes_df['bearing_id'].unique().tolist()
        np.random.shuffle(bearings)

        n_test = max(1, int(len(bearings) * test_ratio))
        test_bearings = bearings[:n_test]
        train_bearings = bearings[n_test:]

    print(f"Train bearings ({len(train_bearings)}): {train_bearings[:5]}...")
    print(f"Test bearings ({len(test_bearings)}): {test_bearings}")

    # Create datasets
    train_dataset = BearingDataset(
        data_dir=data_dir,
        bearing_ids=train_bearings,
        episodes_df=episodes_df,
        window_size=window_size,
        stride=stride,
        n_channels=n_channels,
    )

    test_dataset = BearingDataset(
        data_dir=data_dir,
        bearing_ids=test_bearings,
        episodes_df=episodes_df,
        window_size=window_size,
        stride=stride,
        n_channels=n_channels,
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # Collect info
    info = {
        'train_bearings': train_bearings,
        'test_bearings': test_bearings,
        'train_windows': len(train_dataset),
        'test_windows': len(test_dataset),
        'n_channels': n_channels,  # Fixed channel count
        'window_size': window_size,
        'fault_labels': {
            'healthy': 0,
            'outer_race': 1,
            'inner_race': 2,
            'ball': 3,
        },
    }

    return train_loader, test_loader, info


# =============================================================================
# Test
# =============================================================================

if __name__ == '__main__':
    # Test data loading
    data_dir = Path(__file__).parent.parent.parent / 'data' / 'bearings'

    train_loader, test_loader, info = create_dataloaders(
        data_dir=data_dir,
        batch_size=32,
        window_size=4096,
        dataset_filter='cwru',  # Start with CWRU only for testing
    )

    print(f"\nInfo: {info}")

    # Test batch
    for batch in train_loader:
        signals, labels, bearing_ids = batch
        print(f"\nBatch shape: {signals.shape}")
        print(f"Labels: {labels[:5]}")
        print(f"Bearings: {bearing_ids[:5]}")
        break
