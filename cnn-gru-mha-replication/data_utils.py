"""
Data utilities for CNN-GRU-MHA replication.

Reuses FEMTO CSV loading from dcssl-replication/data_utils.py.
Adds:
  - DWT denoising (sym8, level=3, keep approximation only)
  - Min-max normalization per snapshot to [0, 1]
  - Horizontal channel selection (channel 0)
  - Linear RUL labels: Y_i = (N - i) / N
  - XJTU-SY data loading (if available)
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings

import pywt

# Reuse FEMTO CSV loading from dcssl-replication
# Use importlib to avoid module name collision with this file
import importlib.util as _ilu

_DCSSL_DIR = Path(__file__).parent.parent / "dcssl-replication"
_spec = _ilu.spec_from_file_location(
    "dcssl_data_utils", str(_DCSSL_DIR / "data_utils.py")
)
_dcssl_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_dcssl_mod)

load_bearing_snapshots = _dcssl_mod.load_bearing_snapshots
find_bearing_dir = _dcssl_mod.find_bearing_dir


# =====================================================================
# Constants
# =====================================================================

FEMTO_DATA_ROOT = Path("/mnt/sagemaker-nvme/femto_data/10. FEMTO Bearing")

# Paper Table 2 transfer setup
FEMTO_TRANSFERS = {
    "test1": {
        "source": "Bearing1_3",
        "targets": ["Bearing2_3", "Bearing2_4", "Bearing3_1", "Bearing3_3"],
    },
    "test2": {
        "source": "Bearing2_3",
        "targets": ["Bearing1_3", "Bearing1_4", "Bearing3_3", "Bearing3_3"],
    },
    "test3": {
        "source": "Bearing3_2",
        "targets": ["Bearing1_3", "Bearing1_4", "Bearing2_3", "Bearing2_4"],
    },
}

# XJTU-SY transfer setup (Table 5)
XJTU_TRANSFERS = {
    "exp1": {"source": "Bearing1_3", "targets": ["Bearing2_3", "Bearing3_2"]},
    "exp2": {"source": "Bearing2_3", "targets": ["Bearing1_3", "Bearing3_2"]},
}

# Paper target RMSE values (Table 4)
PAPER_FEMTO_TARGETS = {
    ("Bearing1_3", "Bearing2_3"): 0.0463,
    ("Bearing1_3", "Bearing2_4"): 0.0449,
    ("Bearing1_3", "Bearing3_1"): 0.0427,
    ("Bearing1_3", "Bearing3_3"): 0.0461,
    ("Bearing2_3", "Bearing1_3"): 0.0458,
    ("Bearing2_3", "Bearing1_4"): 0.0426,
    ("Bearing2_3", "Bearing3_3"): 0.0416,
    ("Bearing3_2", "Bearing1_3"): 0.0382,
    ("Bearing3_2", "Bearing1_4"): 0.0397,
    ("Bearing3_2", "Bearing2_3"): 0.0413,
    ("Bearing3_2", "Bearing2_4"): 0.0418,
}
PAPER_FEMTO_AVG = 0.0443


# =====================================================================
# Preprocessing
# =====================================================================

def dwt_denoise(signal: np.ndarray, wavelet: str = "sym8", level: int = 3) -> np.ndarray:
    """
    DWT denoising: decompose, zero detail coefficients, reconstruct.

    Args:
        signal: 1D array of shape (2560,)
        wavelet: wavelet family (default sym8 per paper)
        level: decomposition levels (default 3 per paper)

    Returns:
        Denoised signal of same shape.
    """
    coeffs = pywt.wavedec(signal, wavelet=wavelet, level=level)
    # Keep only approximation coefficients (index 0), zero detail coefficients
    denoised_coeffs = [coeffs[0]] + [np.zeros_like(c) for c in coeffs[1:]]
    denoised = pywt.waverec(denoised_coeffs, wavelet=wavelet)
    # Trim/pad to original length
    if len(denoised) > len(signal):
        denoised = denoised[:len(signal)]
    elif len(denoised) < len(signal):
        denoised = np.pad(denoised, (0, len(signal) - len(denoised)), mode="edge")
    return denoised.astype(np.float32)


def minmax_normalize(signal: np.ndarray) -> np.ndarray:
    """
    Min-max normalize a signal to [0, 1].

    Args:
        signal: 1D array

    Returns:
        Normalized array in [0, 1].
    """
    s_min = signal.min()
    s_max = signal.max()
    denom = s_max - s_min
    if denom < 1e-8:
        return np.zeros_like(signal)
    return ((signal - s_min) / denom).astype(np.float32)


def preprocess_snapshot(snapshot: np.ndarray, channel: int = 0) -> np.ndarray:
    """
    Full preprocessing pipeline per snapshot:
      1. Select horizontal channel
      2. DWT denoising (sym8, level=3)
      3. Min-max normalization to [0, 1]

    Args:
        snapshot: (2560, 2) raw snapshot
        channel: channel index (0=horizontal, 1=vertical); paper uses horizontal only

    Returns:
        Preprocessed 1D signal of shape (2560,)
    """
    sig = snapshot[:, channel].copy().astype(np.float32)
    sig = dwt_denoise(sig)
    sig = minmax_normalize(sig)
    return sig


def compute_linear_rul(n_snapshots: int) -> np.ndarray:
    """
    Compute linear RUL labels: Y_i = (N - i) / N

    Goes from ~1.0 at i=0 to 0.0 at i=N-1 (when i=N, Y=0).
    Note: Y_0 = N/N = 1.0, Y_{N-1} = 1/N ≈ 0, Y_N = 0.

    Per the paper: Y_i = (N - i) / N where N = total snapshots.

    Returns:
        Array of shape (n_snapshots,) with linear RUL labels.
    """
    i = np.arange(n_snapshots)
    return ((n_snapshots - i) / n_snapshots).astype(np.float32)


# =====================================================================
# Bearing data loading
# =====================================================================

def load_bearing_for_cnn_gru(
    data_root: Path,
    bearing_name: str,
    channel: int = 0,
    verbose: bool = False,
) -> Dict:
    """
    Load and preprocess a bearing for CNN-GRU-MHA training.

    Steps:
      1. Load raw CSV snapshots via dcssl-replication loader
      2. Preprocess each snapshot: DWT + minmax
      3. Assign linear RUL labels

    Returns dict:
      - 'bearing_name': str
      - 'snapshots': (N, 2560) float32 — preprocessed horizontal channel
      - 'rul': (N,) float32 — linear RUL labels
      - 'n_snapshots': int
    """
    raw = load_bearing_snapshots(data_root, bearing_name)  # (N, 2560, 2)
    n = len(raw)

    if verbose:
        print(f"  {bearing_name}: {n} snapshots, preprocessing...")

    # Preprocess each snapshot
    processed = np.zeros((n, 2560), dtype=np.float32)
    for i in range(n):
        processed[i] = preprocess_snapshot(raw[i], channel=channel)

    rul = compute_linear_rul(n)

    return {
        "bearing_name": bearing_name,
        "snapshots": processed,  # (N, 2560)
        "rul": rul,
        "n_snapshots": n,
    }


def get_transfer_split(
    bdata: Dict,
    split_ratio: float = 0.5,
    random_split: bool = True,
    seed: int = None,
) -> Tuple[Dict, Dict]:
    """
    Split a target bearing 1:1 into fine-tune half and eval half.

    Args:
        bdata: bearing data dict
        split_ratio: fraction for fine-tune (default 0.5 = 1:1 split)
        random_split: if True, use random split (both halves cover full life);
                      if False, chronological split (first half = FT, second half = eval)
        seed: random seed for reproducible random split

    Returns:
        (finetune_data, eval_data) as dicts with same structure
    """
    n = bdata["n_snapshots"]

    if random_split:
        # Random split: both halves cover full RUL range
        rng = np.random.default_rng(seed if seed is not None else 42)
        all_idx = np.arange(n)
        rng.shuffle(all_idx)
        split_idx = int(n * split_ratio)
        ft_idx = np.sort(all_idx[:split_idx])
        eval_idx = np.sort(all_idx[split_idx:])
    else:
        # Chronological: first half = FT, second half = eval
        split_idx = int(n * split_ratio)
        ft_idx = np.arange(split_idx)
        eval_idx = np.arange(split_idx, n)

    finetune = {
        "bearing_name": bdata["bearing_name"] + "_finetune",
        "snapshots": bdata["snapshots"][ft_idx],
        "rul": bdata["rul"][ft_idx],
        "n_snapshots": len(ft_idx),
    }
    eval_data = {
        "bearing_name": bdata["bearing_name"] + "_eval",
        "snapshots": bdata["snapshots"][eval_idx],
        "rul": bdata["rul"][eval_idx],
        "n_snapshots": len(eval_idx),
    }
    return finetune, eval_data


# =====================================================================
# XJTU-SY loading (if available)
# =====================================================================

XJTU_CANDIDATE_ROOTS = [
    Path("/mnt/sagemaker-nvme/xjtu_data"),
    Path("/home/sagemaker-user/IndustrialJEPA/data/xjtu"),
    Path("/tmp/xjtu"),
]


def find_xjtu_root() -> Optional[Path]:
    """Find XJTU-SY data root if it exists."""
    for candidate in XJTU_CANDIDATE_ROOTS:
        if candidate.exists():
            return candidate
    return None


def load_xjtu_bearing(data_root: Path, bearing_name: str, channel: int = 0) -> Optional[Dict]:
    """
    Load XJTU-SY bearing data.

    XJTU-SY format: CSV files per snapshot with columns [time, horiz, vert]
    or similar to FEMTO format.

    Returns None if not found.
    """
    bearing_dir = None
    for root, dirs, files in os.walk(str(data_root)):
        if Path(root).name == bearing_name:
            bearing_dir = Path(root)
            break

    if bearing_dir is None:
        warnings.warn(f"XJTU bearing {bearing_name} not found under {data_root}")
        return None

    csv_files = sorted(bearing_dir.glob("*.csv"))
    if not csv_files:
        return None

    snapshots = []
    for f in csv_files:
        try:
            data = pd.read_csv(str(f), header=None).values.astype(np.float32)
            if data.shape[1] >= 2:
                sig = data[:, channel] if data.shape[1] > channel else data[:, 0]
                # Pad/truncate to 32768 (XJTU standard) then crop to 2560
                if len(sig) > 2560:
                    sig = sig[:2560]
                elif len(sig) < 2560:
                    sig = np.pad(sig, (0, 2560 - len(sig)), mode="edge")
                snapshots.append(sig.astype(np.float32))
        except Exception as e:
            warnings.warn(f"Could not load {f}: {e}")
            continue

    if not snapshots:
        return None

    raw_snaps = np.stack(snapshots)  # (N, 2560)
    n = len(raw_snaps)

    # Preprocess: DWT + minmax
    processed = np.zeros_like(raw_snaps)
    for i in range(n):
        processed[i] = minmax_normalize(dwt_denoise(raw_snaps[i]))

    rul = compute_linear_rul(n)

    return {
        "bearing_name": bearing_name,
        "snapshots": processed,
        "rul": rul,
        "n_snapshots": n,
    }
