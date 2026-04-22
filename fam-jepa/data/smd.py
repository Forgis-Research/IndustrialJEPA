"""
SMD (Server Machine Dataset) Adapter.

Collected from a large Internet company's server machines.
  - 38 channels (CPU, memory, network, disk metrics)
  - 28 machines, each with separate train/test
  - ~4% anomaly rate (varies per machine)

IMPORTANT: the concatenated train.npy / test.npy files glue 28 independent
machines end-to-end along the time axis.  Any chronological split of the
concatenated array crosses machine boundaries → distribution shift.

For predictor finetuning, use split_smd_entities().  This performs an
INTRA-machine chronological split: for each machine, the first 60% of its
test stream goes to ft_train, next 10% to ft_val, last 30% to ft_test.
Every machine appears in all three splits.  A gap of window_size timesteps
is inserted at each boundary to prevent temporal leakage.

Data source: OmniAnomaly / Anomaly Transformer repos.
Standard window size: 100 timesteps.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Optional, Tuple, List, Dict

try:
    from .config import SMD_DIR
    SMD_DATA_DIR = SMD_DIR
except ImportError:
    SMD_DATA_DIR = Path('/home/sagemaker-user/IndustrialJEPA/paper-replications/mts-jepa/data/SMD')

WINDOW_SIZE = 100
TRAIN_STRIDE = 10


def check_smd_available() -> bool:
    """Check if SMD data files are present."""
    return ((SMD_DATA_DIR / 'train.npy').exists() or
            any(SMD_DATA_DIR.glob('machine-*')))


# ---------------------------------------------------------------------------
# Per-entity (per-machine) loader
# ---------------------------------------------------------------------------

def load_smd_entities(normalize: bool = True, drop_constant: bool = True,
                      constant_threshold: float = 1e-8) -> List[Dict]:
    """Load SMD as a list of independent machine entities.

    Each machine has its own train (normal) and test (with anomalies) arrays.
    """
    machine_dirs = sorted(SMD_DATA_DIR.glob('machine-*'))
    if not machine_dirs:
        raise FileNotFoundError(
            f"No machine-* dirs in {SMD_DATA_DIR}. "
            f"SMD per-machine files are required for entity-level splits."
        )

    # Compute global stats from all machines' train data
    all_train = []
    for md in machine_dirs:
        all_train.append(np.load(md / 'train.npy').astype(np.float32))
    concat_train = np.concatenate(all_train, axis=0)

    # Identify near-constant channels
    keep_mask = None
    if drop_constant:
        var = concat_train.var(axis=0)
        keep_mask = var > constant_threshold

    if normalize:
        src = concat_train[:, keep_mask] if keep_mask is not None else concat_train
        mu = src.mean(axis=0, keepdims=True)
        std = src.std(axis=0, keepdims=True) + 1e-6
    else:
        mu, std = None, None
    del concat_train, all_train

    entities = []
    for md in machine_dirs:
        train = np.load(md / 'train.npy').astype(np.float32)
        test = np.load(md / 'test.npy').astype(np.float32)
        labels = np.load(md / 'test_labels.npy').astype(np.int32)

        min_len = min(len(test), len(labels))
        test = test[:min_len]
        labels = labels[:min_len]

        if keep_mask is not None:
            train = train[:, keep_mask]
            test = test[:, keep_mask]

        if normalize:
            train = (train - mu) / std
            test = (test - mu) / std

        entities.append({
            'entity_id': md.name,
            'train': train,
            'test': test,
            'labels': labels,
            'has_anomaly': bool(labels.any()),
        })

    return entities


def split_smd_entities(ratios: Tuple[float, float, float] = (0.6, 0.1, 0.3),
                       normalize: bool = True) -> Dict[str, List[Dict]]:
    """Load SMD machines and split intra-machine for pred-FT.

    Every machine appears in all three splits (ft_train / ft_val / ft_test).
    Split is chronological within each machine's test stream, with a gap of
    WINDOW_SIZE timesteps at each boundary to prevent temporal leakage.
    """
    from .smap_msl import _intra_entity_split, WINDOW_SIZE
    entities = load_smd_entities(normalize=normalize)
    return _intra_entity_split(entities, ratios, window_size=WINDOW_SIZE)


# ---------------------------------------------------------------------------
# Concatenated loaders (for pretraining and Mahalanobis baseline)
# ---------------------------------------------------------------------------

def _load_combined(data_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load combined .npy files if they exist."""
    train = np.load(data_dir / 'train.npy').astype(np.float32)
    test = np.load(data_dir / 'test.npy').astype(np.float32)
    labels = np.load(data_dir / 'test_labels.npy').astype(np.int32)
    return train, test, labels


def _load_per_machine(data_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load per-machine files and concatenate."""
    machine_dirs = sorted(data_dir.glob('machine-*'))
    if not machine_dirs:
        raise FileNotFoundError(f"No machine-* dirs in {data_dir}")

    trains, tests, labels_list = [], [], []
    for md in machine_dirs:
        trains.append(np.load(md / 'train.npy').astype(np.float32))
        tests.append(np.load(md / 'test.npy').astype(np.float32))
        labels_list.append(np.load(md / 'test_labels.npy').astype(np.int32))

    return (np.concatenate(trains, axis=0),
            np.concatenate(tests, axis=0),
            np.concatenate(labels_list, axis=0))


def load_smd(normalize: bool = True, drop_constant: bool = True,
             constant_threshold: float = 1e-8) -> dict:
    """Load SMD as concatenated arrays (for pretraining / Mahalanobis).

    WARNING: the returned test array concatenates 28 independent machines.
    Do NOT use a chronological split of this array for finetuning — use
    split_smd_entities() instead.
    """
    if (SMD_DATA_DIR / 'train.npy').exists():
        train, test, labels = _load_combined(SMD_DATA_DIR)
    else:
        train, test, labels = _load_per_machine(SMD_DATA_DIR)

    # Align lengths
    min_len = min(len(test), len(labels))
    test = test[:min_len]
    labels = labels[:min_len]

    # Drop near-constant channels
    if drop_constant:
        var = train.var(axis=0)
        keep = var > constant_threshold
        if keep.sum() < train.shape[1]:
            print(f"  SMD: dropping {train.shape[1] - keep.sum()} constant channels "
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
        'name': 'SMD',
        'mu': mu,
        'std': std,
        'anomaly_rate': float(labels.mean()),
    }


def get_smd_dataloader(n_samples: int = 50000, batch_size: int = 64,
                       seed: int = 42) -> Tuple[DataLoader, dict]:
    """Load SMD and return DataLoader for pretraining."""
    from .smap_msl import AnomalyPretrainDataset, collate_anomaly_pretrain
    data = load_smd()
    ds = AnomalyPretrainDataset(data['train'], n_samples=n_samples, seed=seed)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True,
                        collate_fn=collate_anomaly_pretrain, num_workers=0)
    return loader, data


if __name__ == '__main__':
    print("Testing SMD data adapter...")
    if not check_smd_available():
        print("SMD data not found. Download from OmniAnomaly/Anomaly-Transformer repo.")
    else:
        data = load_smd()
        print(f"SMD: train={data['train'].shape}, test={data['test'].shape}")
        print(f"Channels: {data['n_channels']}, Anomaly rate: {data['anomaly_rate']:.3f}")
        print("Adapter test PASSED")
