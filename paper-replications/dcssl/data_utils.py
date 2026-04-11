"""
FEMTO/PRONOSTIA Bearing Dataset Utilities for DCSSL Replication.

Handles:
- Loading CSV snapshots from each bearing directory
- RUL label construction (piecewise linear with FPT detection)
- Dataset classes for pretraining and fine-tuning
- Standard feature extraction (optional)
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict, Optional
import warnings


# =====================================================================
# FEMTO Data Loading
# =====================================================================

# Bearing lifecycle lengths (number of snapshots to end-of-life)
# These are fixed from the FEMTO paper/competition
BEARING_EOL_FILES = {
    # Training bearings (full life)
    "Bearing1_1": 2803, "Bearing1_2": 871,
    "Bearing2_1": 911,  "Bearing2_2": 797,
    "Bearing3_1": 515,  "Bearing3_2": 1637,
    # Test bearings (truncated - need to know actual truncation)
    # These are dynamically detected from available files
}

# Operating conditions
CONDITION_INFO = {
    1: {"speed_rpm": 1800, "load_N": 4000, "train": ["Bearing1_1", "Bearing1_2"],
        "test": ["Bearing1_3", "Bearing1_4", "Bearing1_5", "Bearing1_6", "Bearing1_7"]},
    2: {"speed_rpm": 1650, "load_N": 4200, "train": ["Bearing2_1", "Bearing2_2"],
        "test": ["Bearing2_3", "Bearing2_4", "Bearing2_5", "Bearing2_6", "Bearing2_7"]},
    3: {"speed_rpm": 1500, "load_N": 5000, "train": ["Bearing3_1", "Bearing3_2"],
        "test": ["Bearing3_3"]},
}


def find_bearing_dir(data_root: Path, bearing_name: str,
                     prefer_full_test: bool = True) -> Optional[Path]:
    """
    Find a bearing directory under data_root.

    Priority order:
    1. Full_Test_Set (run-to-failure, for test bearings) if prefer_full_test=True
    2. Learning_set (for training bearings)
    3. Any matching directory
    """
    # Try preferred directories first
    search_priority = []
    if prefer_full_test:
        search_priority = [
            data_root / "Full_Test_Set" / bearing_name,
            data_root / "Learning_set" / bearing_name,
            data_root / "Test_set" / bearing_name,
        ]
    else:
        search_priority = [
            data_root / "Learning_set" / bearing_name,
            data_root / "Test_set" / bearing_name,
            data_root / "Full_Test_Set" / bearing_name,
        ]

    for candidate in search_priority:
        if candidate.exists():
            return candidate

    # Fallback: walk entire tree
    for root, dirs, files in os.walk(data_root):
        root_path = Path(root)
        if root_path.name.lower() == bearing_name.lower():
            return root_path

    return None


def load_bearing_snapshots(data_root: Path, bearing_name: str) -> np.ndarray:
    """
    Load all vibration snapshots for a bearing.

    Returns:
        Array of shape (n_snapshots, 2560, 2) — [time, samples, channels]
        Channel 0 = horizontal, Channel 1 = vertical
    """
    bearing_dir = find_bearing_dir(data_root, bearing_name)
    if bearing_dir is None:
        raise FileNotFoundError(f"Bearing {bearing_name} not found under {data_root}")

    # Find acc (vibration) CSV files; filter out temperature files
    csv_files = sorted([f for f in bearing_dir.glob("*.csv")
                        if "acc" in f.name.lower() or "vibration" in f.name.lower()])
    if not csv_files:
        # Fallback: all CSV files that aren't temperature
        csv_files = sorted([f for f in bearing_dir.glob("*.csv")
                            if "temp" not in f.name.lower()])

    if not csv_files:
        raise FileNotFoundError(f"No vibration CSV files found in {bearing_dir}")

    snapshots = []
    for f in csv_files:
        try:
            # FEMTO format: each row has columns [h, m, s, us, horizontal_acc, vertical_acc]
            # Delimiter varies: Learning_set uses comma, Full_Test_Set uses semicolon
            # Try comma first, then semicolon
            try:
                data = np.loadtxt(str(f), delimiter=',')
            except ValueError:
                data = np.loadtxt(str(f), delimiter=';')
            if data.ndim == 1:
                data = data.reshape(-1, 1)

            if data.shape[1] >= 6:
                # Full format with timestamp columns
                vib = data[:, 4:6]  # columns 4 & 5 are the accelerations
            elif data.shape[1] == 2:
                vib = data
            elif data.shape[1] >= 2:
                vib = data[:, -2:]
            else:
                continue

            # Should be exactly 2560 samples
            if len(vib) < 100:  # skip tiny files
                continue
            if len(vib) != 2560:
                # Pad or truncate to 2560
                if len(vib) < 2560:
                    vib = np.pad(vib, ((0, 2560 - len(vib)), (0, 0)), mode='edge')
                else:
                    vib = vib[:2560]

            snapshots.append(vib.astype(np.float32))
        except Exception as e:
            warnings.warn(f"Could not load {f}: {e}")
            continue

    if not snapshots:
        raise ValueError(f"No valid snapshots loaded for {bearing_name}")

    return np.stack(snapshots, axis=0)  # (n_snapshots, 2560, 2)


def compute_rms(snapshots: np.ndarray) -> np.ndarray:
    """Compute RMS of each snapshot across both channels."""
    # snapshots: (n, 2560, 2)
    return np.sqrt(np.mean(snapshots ** 2, axis=(1, 2)))


def compute_kurtosis(snapshots: np.ndarray) -> np.ndarray:
    """Compute kurtosis of each snapshot (mean over channels)."""
    from scipy.stats import kurtosis as scipy_kurtosis
    n = len(snapshots)
    kurt = np.zeros(n)
    for i in range(n):
        k_h = scipy_kurtosis(snapshots[i, :, 0], fisher=False)
        k_v = scipy_kurtosis(snapshots[i, :, 1], fisher=False)
        kurt[i] = (k_h + k_v) / 2
    return kurt


def detect_fpt(snapshots: np.ndarray, method: str = "rms", threshold_std: float = 3.0,
               window: int = 100) -> int:
    """
    Detect First Prediction Time (FPT) — when degradation begins.

    Uses a threshold: FPT = first sustained crossing where RMS/kurtosis exceeds
    mean + threshold_std * std of the "healthy" early portion.

    Args:
        snapshots: (n_snapshots, 2560, 2)
        method: "rms" or "kurtosis"
        threshold_std: number of std deviations for threshold
        window: number of early snapshots to estimate healthy baseline (default=100)
            Use a large window to avoid spurious early crossings.

    Returns:
        FPT index (integer snapshot index where degradation starts)
    """
    if method == "rms":
        signal = compute_rms(snapshots)
    else:
        signal = compute_kurtosis(snapshots)

    n = len(signal)
    # Baseline from early portion — use at least 50 points
    baseline_window = max(50, min(window, n // 5))
    baseline = signal[:baseline_window]
    mu = baseline.mean()
    sigma = baseline.std()

    # If sigma is very small, use a percentage-based threshold instead
    if sigma < 1e-4 or mu / sigma > 100:
        threshold = mu * 1.2  # 20% above healthy mean
    else:
        threshold = mu + threshold_std * sigma

    # Find first sustained crossing (consecutive crossings over a small window)
    # to avoid single noisy spikes being flagged as FPT
    sustain = 3  # require at least 3 consecutive crossings
    for i in range(baseline_window, n - sustain + 1):
        if all(signal[i + k] > threshold for k in range(sustain)):
            return i

    # If no crossing found, FPT = 0 (assume entire life is relevant)
    return 0


def compute_rul_labels(n_snapshots: int, fpt: int) -> np.ndarray:
    """
    Compute piecewise-linear RUL labels normalized to [0, 1].

    - From snapshot 0 to fpt: RUL = 1.0 (healthy)
    - From fpt to n_snapshots-1: linear decay from 1.0 to 0.0
    """
    rul = np.ones(n_snapshots, dtype=np.float32)
    if fpt < n_snapshots - 1:
        decay_length = n_snapshots - 1 - fpt
        for i in range(fpt, n_snapshots):
            rul[i] = 1.0 - (i - fpt) / decay_length
    return rul


def load_bearing_with_rul(
    data_root: Path,
    bearing_name: str,
    fpt_method: str = "rms",
    fpt_threshold: float = 3.0,
    normalize_snapshots: bool = True,
) -> Dict:
    """
    Load bearing data with RUL labels.

    Returns dict with:
        - 'snapshots': (n, 2560, 2) float32
        - 'rul': (n,) float32
        - 'fpt': int
        - 'n_snapshots': int
        - 'rms': (n,) float32
    """
    snapshots = load_bearing_snapshots(data_root, bearing_name)
    n = len(snapshots)

    # Normalize snapshots by global std
    if normalize_snapshots:
        std = snapshots.std()
        if std > 0:
            snapshots = snapshots / std

    # Detect FPT
    fpt = detect_fpt(snapshots, method=fpt_method, threshold_std=fpt_threshold)

    # Compute RUL
    rul = compute_rul_labels(n, fpt)
    rms = compute_rms(snapshots)

    return {
        "bearing_name": bearing_name,
        "snapshots": snapshots,
        "rul": rul,
        "fpt": fpt,
        "n_snapshots": n,
        "rms": rms,
    }


# =====================================================================
# Feature Extraction (optional, for feature-based models)
# =====================================================================

def extract_features_from_snapshot(snapshot: np.ndarray) -> np.ndarray:
    """
    Extract 18 statistical/spectral features from a 2560x2 snapshot.

    For each channel: RMS, peak, kurtosis, skewness, crest factor,
    shape factor, impulse factor, variance, mean_abs, peak-to-peak,
    spectral_entropy, spectral_centroid (12 × 2 channels = 24 total,
    but we take the mean across channels for many features = 18).
    """
    from scipy.stats import kurtosis as sp_kurtosis, skew as sp_skew
    from scipy.fft import fft

    feats = []
    for ch in range(2):  # horizontal, vertical
        x = snapshot[:, ch]
        n = len(x)

        rms = np.sqrt(np.mean(x ** 2))
        peak = np.max(np.abs(x))
        mean_abs = np.mean(np.abs(x))
        variance = np.var(x)
        ptp = np.ptp(x)

        if rms > 0:
            crest = peak / rms
            shape = rms / mean_abs if mean_abs > 0 else 0.0
            impulse = peak / mean_abs if mean_abs > 0 else 0.0
        else:
            crest = shape = impulse = 0.0

        kurt = float(sp_kurtosis(x, fisher=False))
        skewness = float(sp_skew(x))

        # Spectral features
        X = np.abs(fft(x))[:n // 2]
        freqs = np.arange(n // 2)
        if X.sum() > 0:
            spectral_centroid = (freqs * X).sum() / X.sum()
            probs = X / X.sum()
            spectral_entropy = -np.sum(probs * np.log(probs + 1e-10))
        else:
            spectral_centroid = spectral_entropy = 0.0

        feats.extend([rms, peak, mean_abs, variance, ptp, crest, shape, impulse,
                       kurt, skewness, spectral_centroid, spectral_entropy])

    return np.array(feats, dtype=np.float32)  # 24-dim


def extract_features_from_bearing(snapshots: np.ndarray) -> np.ndarray:
    """Extract features from all snapshots. Returns (n, 24)."""
    return np.array([extract_features_from_snapshot(s) for s in snapshots])


# =====================================================================
# PyTorch Datasets
# =====================================================================

class FEMTOPretrainDataset(Dataset):
    """
    Dataset for self-supervised pretraining.

    Each item is a pair of augmented views of the same snapshot
    along with its position in the bearing sequence (for temporal loss).
    """

    def __init__(
        self,
        bearing_data_list: List[Dict],
        augmentations: Optional[object] = None,
        crop_length: int = 1024,  # sub-sequence length for random crop
    ):
        self.crop_length = crop_length
        self.augmentations = augmentations

        # Flatten all snapshots with bearing index and time position
        self.items = []
        for b_idx, bdata in enumerate(bearing_data_list):
            n = bdata["n_snapshots"]
            for t in range(n):
                self.items.append({
                    "bearing_idx": b_idx,
                    "time_idx": t,
                    "n_snapshots": n,
                    "rul": bdata["rul"][t],
                    "snapshot": bdata["snapshots"][t],  # (2560, 2)
                })

    def __len__(self):
        return len(self.items)

    def _augment(self, snapshot: np.ndarray) -> np.ndarray:
        """Apply random crop + optional masking to a snapshot."""
        L = len(snapshot)
        # Random crop
        if self.crop_length < L:
            start = np.random.randint(0, L - self.crop_length)
            crop = snapshot[start: start + self.crop_length].copy()
        else:
            crop = snapshot.copy()

        # Random amplitude scaling (slight jitter)
        scale = np.random.uniform(0.9, 1.1)
        crop = crop * scale

        # Random segment masking (10-20% of the signal)
        mask_len = int(len(crop) * np.random.uniform(0.05, 0.15))
        if mask_len > 0:
            mask_start = np.random.randint(0, len(crop) - mask_len)
            crop[mask_start: mask_start + mask_len] = 0.0

        return crop.astype(np.float32)

    def __getitem__(self, idx):
        item = self.items[idx]
        snapshot = item["snapshot"]

        view1 = self._augment(snapshot)
        view2 = self._augment(snapshot)

        # Transpose to (channels, time) for Conv1D
        view1 = view1.T  # (2, crop_length)
        view2 = view2.T

        return {
            "view1": torch.FloatTensor(view1),
            "view2": torch.FloatTensor(view2),
            "bearing_idx": item["bearing_idx"],
            "time_idx": item["time_idx"],
            "n_snapshots": item["n_snapshots"],
            "rul": torch.FloatTensor([item["rul"]]),
        }


class FEMTORULDataset(Dataset):
    """
    Dataset for RUL regression (fine-tuning or training prediction head).

    Each item is a snapshot with its RUL label.
    """

    def __init__(
        self,
        bearing_data_list: List[Dict],
        augment: bool = False,
        crop_length: int = 2560,
    ):
        self.augment = augment
        self.crop_length = min(crop_length, 2560)

        self.snapshots = []
        self.rul_labels = []
        self.bearing_indices = []

        for b_idx, bdata in enumerate(bearing_data_list):
            n = bdata["n_snapshots"]
            for t in range(n):
                self.snapshots.append(bdata["snapshots"][t])  # (2560, 2)
                self.rul_labels.append(bdata["rul"][t])
                self.bearing_indices.append(b_idx)

        self.snapshots = np.array(self.snapshots, dtype=np.float32)
        self.rul_labels = np.array(self.rul_labels, dtype=np.float32)

        # Time indices per bearing (for elapsed time feature)
        self.time_indices = []
        self.n_snapshots_per_item = []
        for b_idx, bdata in enumerate(bearing_data_list):
            n = bdata["n_snapshots"]
            for t in range(n):
                self.time_indices.append(t)
                self.n_snapshots_per_item.append(n)
        self.time_indices = np.array(self.time_indices, dtype=np.float32)
        self.n_snapshots_per_item = np.array(self.n_snapshots_per_item, dtype=np.float32)
        # Normalized elapsed time [0, 1]
        self.elapsed_time = self.time_indices / self.n_snapshots_per_item

    def __len__(self):
        return len(self.rul_labels)

    def __getitem__(self, idx):
        snap = self.snapshots[idx].copy()  # (2560, 2)

        if self.augment:
            # Light augmentation during fine-tuning
            snap = snap * np.random.uniform(0.95, 1.05)

        # Truncate to crop_length if needed
        if self.crop_length < 2560:
            snap = snap[:self.crop_length]

        # Transpose to (channels, time)
        snap = snap.T  # (2, crop_length)

        return {
            "x": torch.FloatTensor(snap),
            "rul": torch.FloatTensor([self.rul_labels[idx]]),
            "bearing_idx": self.bearing_indices[idx],
            "elapsed_time": torch.FloatTensor([self.elapsed_time[idx]]),
        }


# =====================================================================
# Loading helpers
# =====================================================================

def load_condition_data(
    data_root: Path,
    condition: int,
    verbose: bool = True,
    fpt_threshold: float = 3.0,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Load all train and test bearings for a given condition.

    Returns:
        train_data: list of bearing dicts
        test_data: list of bearing dicts
    """
    info = CONDITION_INFO[condition]
    train_data = []
    test_data = []

    for bearing_name in info["train"]:
        if verbose:
            print(f"  Loading {bearing_name}...", end=" ", flush=True)
        try:
            bdata = load_bearing_with_rul(data_root, bearing_name, fpt_threshold=fpt_threshold)
            train_data.append(bdata)
            if verbose:
                print(f"OK ({bdata['n_snapshots']} snapshots, FPT={bdata['fpt']})")
        except Exception as e:
            if verbose:
                print(f"FAILED: {e}")

    for bearing_name in info["test"]:
        if verbose:
            print(f"  Loading {bearing_name}...", end=" ", flush=True)
        try:
            bdata = load_bearing_with_rul(data_root, bearing_name, fpt_threshold=fpt_threshold)
            test_data.append(bdata)
            if verbose:
                print(f"OK ({bdata['n_snapshots']} snapshots, FPT={bdata['fpt']})")
        except Exception as e:
            if verbose:
                print(f"FAILED: {e}")

    return train_data, test_data


def get_data_stats(data_list: List[Dict]) -> Dict:
    """Get summary statistics of a list of bearing data."""
    total_snapshots = sum(d["n_snapshots"] for d in data_list)
    avg_fpt_ratio = np.mean([d["fpt"] / d["n_snapshots"] for d in data_list])
    return {
        "n_bearings": len(data_list),
        "total_snapshots": total_snapshots,
        "avg_fpt_ratio": avg_fpt_ratio,
        "bearing_names": [d["bearing_name"] for d in data_list],
    }
