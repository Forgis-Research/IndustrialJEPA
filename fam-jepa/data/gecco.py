"""GECCO 2018 Industrial Challenge: Drinking Water Quality Anomaly Prediction.

Single-stream environmental time series from Thueringer Fernwasserversorgung
(German waterworks). 9 sensor channels sampled at ~1 minute, 122K timesteps,
with binary event labels marking contamination anomalies.

Domain: environmental IoT / public-health infrastructure. Distinct from every
existing FAM benchmark. SOTA: F1 ~0.71 (Muharemi+19, J. Intell. Fuzzy Syst.),
AUROC ~0.88 in TAB (Qiu+25, PVLDB 2025).

File layout on disk:
  datasets/data/gecco/gecco2018.csv   (13.6 MB, 122K rows + header)

Columns (after dropping index + timestamp):
  Tp, Cl, pH, Redox, Leit, Trueb, Cl_2, Fm, Fm_2, EVENT
  9 sensor channels + 1 label column (EVENT in {TRUE, FALSE} -> {1, 0}).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

try:
    from .config import _ROOT as CONFIG_ROOT
    GECCO_DIR = CONFIG_ROOT / 'datasets' / 'data' / 'gecco'
except Exception:
    GECCO_DIR = Path('/home/sagemaker-user/IndustrialJEPA/datasets/data/gecco')

SENSOR_COLS = ['Tp', 'Cl', 'pH', 'Redox', 'Leit', 'Trueb', 'Cl_2', 'Fm', 'Fm_2']
LABEL_COL = 'EVENT'
N_CHANNELS = len(SENSOR_COLS)


def _normalize(x: np.ndarray, stats: Dict[str, np.ndarray]) -> np.ndarray:
    return (x - stats['mean']) / stats['std']


def load_gecco(normalize: bool = True,
               split_ratios: tuple = (0.5, 0.25, 0.25),
               gap: int = 200,
               csv_name: str = 'gecco2018.csv') -> Dict:
    """Events are clustered in time (3 segments at ~12%, ~35%, ~70-95%
    of the stream). A vanilla 70/15/15 split puts val in a quiet gap with
    zero events. (0.5, 0.25, 0.25) gives each split at least 0.2% prevalence;
    see the docstring of this module for the cluster timeline.
    """
    """Load GECCO as a single chronological stream with chronological splits.

    Returns dict:
      pretrain_stream   : {0: (N_pre, 9)} - normal-only portion of train
      ft_train, ft_val, ft_test : lists with one entity dict each
      n_channels, name, mu, std, anomaly_rate
    """
    path = GECCO_DIR / csv_name
    if not path.exists():
        raise FileNotFoundError(
            f'GECCO dataset not found at {path}. '
            f"Run: wget -O {path} "
            f"'https://zenodo.org/records/3884398/files/1_gecco2018_water_quality.csv?download=1'")

    df = pd.read_csv(path)
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    X = df[SENSOR_COLS].to_numpy(dtype=np.float32)
    # Some rows have NaN in lab columns; ffill then 0-fill
    X = pd.DataFrame(X, columns=SENSOR_COLS).ffill().fillna(0.0).to_numpy(dtype=np.float32)
    y = df[LABEL_COL].astype(bool).to_numpy().astype(np.int32)

    T = len(X)
    t1 = int(split_ratios[0] * T)
    t2 = int((split_ratios[0] + split_ratios[1]) * T)

    # Compute normalization stats from the train portion (no leakage)
    mu = X[:t1].mean(axis=0)
    std = X[:t1].std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    if normalize:
        X = (X - mu) / std

    # Pretrain stream: only NORMAL rows in the training portion (so pretrain
    # never sees contamination dynamics).
    normal_train_mask = y[:t1] == 0
    pretrain_stream = X[:t1][normal_train_mask].astype(np.float32)

    ft_train = [{'entity_id': 'GECCO',
                 'test': X[:t1],
                 'labels': y[:t1]}]
    ft_val = [{'entity_id': 'GECCO',
               'test':  X[t1 + gap:t2],
               'labels': y[t1 + gap:t2]}]
    ft_test = [{'entity_id': 'GECCO',
                'test':  X[t2 + gap:],
                'labels': y[t2 + gap:]}]

    return {
        'pretrain_stream': {0: pretrain_stream},
        'ft_train': ft_train,
        'ft_val': ft_val,
        'ft_test': ft_test,
        'n_channels': N_CHANNELS,
        'name': 'GECCO2018',
        'mu': mu.astype(np.float32),
        'std': std.astype(np.float32),
        'anomaly_rate': float(y.mean()),
        'T_total': int(T),
    }


if __name__ == '__main__':
    d = load_gecco()
    print(f"n_channels: {d['n_channels']}")
    print(f"T_total: {d['T_total']}")
    print(f"anomaly_rate: {d['anomaly_rate']:.4f}")
    print(f"pretrain: {d['pretrain_stream'][0].shape}")
    print(f"ft_train: {d['ft_train'][0]['test'].shape}  "
          f"anom_rate {d['ft_train'][0]['labels'].mean():.4f}")
    print(f"ft_val:   {d['ft_val'][0]['test'].shape}  "
          f"anom_rate {d['ft_val'][0]['labels'].mean():.4f}")
    print(f"ft_test:  {d['ft_test'][0]['test'].shape}  "
          f"anom_rate {d['ft_test'][0]['labels'].mean():.4f}")
