"""
MBA (MIT-BIH Arrhythmia) Dataset Adapter.

2-channel ECG at ~360 Hz from the MIT-BIH Arrhythmia Database.
  - 2 channels (MLII + V5 leads)
  - ~7680 train / ~7680 test timesteps
  - Arrhythmia labels expanded to +-20 sample windows (TranAD protocol)

NOTE: MBA is a genuinely single continuous ECG recording (one patient).
There are no independent entities to split by.  For predictor finetuning,
use a chronological split of the test stream with a gap of >=window_size
timesteps between train/val/test to prevent temporal leakage.
Very small dataset — label efficiency may be limited.

Data source: TranAD repo (https://github.com/imperial-qore/TranAD).
Place train.xlsx, test.xlsx, labels.xlsx in MBA_DATA_DIR.

Normalization: min-max [0, 1] per channel (TranAD protocol).
"""

import numpy as np
from pathlib import Path
from typing import Optional, Tuple

try:
    from .config import MBA_DIR
    MBA_DATA_DIR = MBA_DIR
except ImportError:
    MBA_DATA_DIR = Path('/home/sagemaker-user/IndustrialJEPA/paper-replications/mts-jepa/data/tranad_repo/data/MBA')


def check_mba_available() -> bool:
    """Check if MBA data files are present."""
    return ((MBA_DATA_DIR / 'train.xlsx').exists() and
            (MBA_DATA_DIR / 'test.xlsx').exists() and
            (MBA_DATA_DIR / 'labels.xlsx').exists())


def _minmax_normalize(data: np.ndarray, mn=None, mx=None):
    """Min-max normalize to [0, 1]. Returns (normalized, min, max)."""
    if mn is None:
        mn = data.min(axis=0)
    if mx is None:
        mx = data.max(axis=0)
    return (data - mn) / (mx - mn + 1e-8), mn, mx


def load_mba(normalize: bool = True, label_window: int = 20) -> Optional[dict]:
    """
    Load MBA dataset.

    Returns None if data not available.
    Otherwise returns dict with train/test/labels arrays.

    Args:
        normalize: apply min-max normalization (TranAD protocol)
        label_window: expand each arrhythmia annotation to +-window samples
    """
    if not check_mba_available():
        print("MBA data not available. Download from TranAD repo:")
        print("  https://github.com/imperial-qore/TranAD")
        return None

    import pandas as pd

    labels_df = pd.read_excel(MBA_DATA_DIR / 'labels.xlsx')
    train_df = pd.read_excel(MBA_DATA_DIR / 'train.xlsx')
    test_df = pd.read_excel(MBA_DATA_DIR / 'test.xlsx')

    # Drop header-artifact row and sample index column
    train = train_df.values[1:, 1:].astype(np.float32)
    test = test_df.values[1:, 1:].astype(np.float32)

    # Build binary labels: expand each annotation to +-label_window samples
    label_indices = labels_df.values[:, 1].astype(int)
    labels = np.zeros(test.shape[0], dtype=np.int32)
    for offset in range(-label_window, label_window + 1):
        idx = label_indices + offset
        idx = idx[(idx >= 0) & (idx < test.shape[0])]
        labels[idx] = 1

    mu, std = None, None
    mn, mx = None, None
    if normalize:
        train, mn, mx = _minmax_normalize(train)
        test, _, _ = _minmax_normalize(test, mn, mx)

    return {
        'train': train,
        'test': test,
        'labels': labels,
        'n_channels': train.shape[1],
        'name': 'MBA',
        'mu': mu,
        'std': std,
        'mn': mn,
        'mx': mx,
        'anomaly_rate': float(labels.mean()),
        'normalization': 'minmax',
    }


if __name__ == '__main__':
    print("Testing MBA data adapter...")
    if not check_mba_available():
        print("MBA data not available (requires TranAD repo download).")
    else:
        data = load_mba()
        print(f"MBA: train={data['train'].shape}, test={data['test'].shape}")
        print(f"Channels: {data['n_channels']}, Anomaly rate: {data['anomaly_rate']:.3f}")
        print("Adapter test PASSED")
