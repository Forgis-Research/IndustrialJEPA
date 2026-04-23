"""BATADAL: Battle of the Attack Detection Algorithms, water distribution.

Taormina et al. 2018 (Journal of Water Resources Planning and Management).
Simulated SCADA data from the C-Town water distribution network (EPANET).
43 channels at 1 hour sampling. Fresh cybersecurity-ICS domain.

File layout on disk:
  datasets/data/batadal/BATADAL_dataset03.csv   (train: 8761 h normal, 2014)
  datasets/data/batadal/BATADAL_dataset04.csv   (test:  4177 h with 7 attacks)

In dataset04, ATT_FLAG uses the competition coding: 1 = attack hour,
-999 = unknown (not revealed to participants). Published methods treat -999
as 0 (normal). We follow that convention.

The pretrain stream is the full 1-year dataset03 (all normal). Pred-FT uses
a chronological 60/10/30 split of dataset04 with a gap.

SOTA: AUC 0.972 (Nguyen+24 hybrid RF+XGB+LSTM); the original Taormina
benchmark evaluated F1 / S_TTD.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

try:
    from .config import _ROOT as CONFIG_ROOT
    BATADAL_DIR = CONFIG_ROOT / 'datasets' / 'data' / 'batadal'
except Exception:
    BATADAL_DIR = Path('/home/sagemaker-user/IndustrialJEPA/datasets/data/batadal')

# 43 SCADA channels: 7 tank levels, 12 flows, 13 pressures, 11 pump/valve states
LEVEL_COLS = [f'L_T{i}' for i in range(1, 8)]
FLOW_COLS  = [f'F_PU{i}' for i in range(1, 12)] + ['F_V2']
STATE_COLS = [f'S_PU{i}' for i in range(1, 12)] + ['S_V2']
PRESS_COLS = ['P_J280', 'P_J269', 'P_J300', 'P_J256', 'P_J289', 'P_J415',
              'P_J302', 'P_J306', 'P_J307', 'P_J317', 'P_J14', 'P_J422']
SENSOR_COLS = LEVEL_COLS + FLOW_COLS + STATE_COLS + PRESS_COLS
LABEL_COL = 'ATT_FLAG'
N_CHANNELS = len(SENSOR_COLS)   # 43


def _load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Strip whitespace from column names (BATADAL CSVs have leading spaces)
    df.columns = [c.strip() for c in df.columns]
    return df


def load_batadal(normalize: bool = True,
                 split_ratios: tuple = (0.6, 0.1, 0.3),
                 gap: int = 48) -> Dict:
    """Return BATADAL as a pretrain stream + three chronological FT splits."""
    train_path = BATADAL_DIR / 'BATADAL_dataset03.csv'
    test_path  = BATADAL_DIR / 'BATADAL_dataset04.csv'
    if not train_path.exists():
        raise FileNotFoundError(train_path)
    if not test_path.exists():
        raise FileNotFoundError(test_path)

    df_tr = _load_csv(train_path)
    df_te = _load_csv(test_path)

    # Sanity: check all sensor columns present
    missing = [c for c in SENSOR_COLS if c not in df_tr.columns]
    if missing:
        raise RuntimeError(f'missing columns in {train_path.name}: {missing[:5]}')

    X_pre = df_tr[SENSOR_COLS].to_numpy(dtype=np.float32)
    X_te  = df_te[SENSOR_COLS].to_numpy(dtype=np.float32)
    y_te  = df_te[LABEL_COL].to_numpy(dtype=np.int32)
    # Convention: -999 -> 0 (unknown treated as normal, standard in the literature)
    y_te = np.where(y_te == 1, 1, 0).astype(np.int32)

    mu = X_pre.mean(axis=0)
    std = X_pre.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    if normalize:
        X_pre = (X_pre - mu) / std
        X_te  = (X_te - mu) / std

    T_te = len(X_te)
    t1 = int(split_ratios[0] * T_te)
    t2 = int((split_ratios[0] + split_ratios[1]) * T_te)

    ft_train = [{'entity_id': 'BATADAL',
                 'test': X_te[:t1], 'labels': y_te[:t1]}]
    ft_val   = [{'entity_id': 'BATADAL',
                 'test': X_te[t1 + gap:t2], 'labels': y_te[t1 + gap:t2]}]
    ft_test  = [{'entity_id': 'BATADAL',
                 'test': X_te[t2 + gap:], 'labels': y_te[t2 + gap:]}]

    return {
        'pretrain_stream': {0: X_pre.astype(np.float32)},
        'ft_train': ft_train,
        'ft_val': ft_val,
        'ft_test': ft_test,
        'n_channels': N_CHANNELS,
        'name': 'BATADAL',
        'mu': mu.astype(np.float32),
        'std': std.astype(np.float32),
        'anomaly_rate': float(y_te.mean()),
        'T_pretrain': int(len(X_pre)),
        'T_test': int(T_te),
    }


if __name__ == '__main__':
    d = load_batadal()
    print(f"n_channels: {d['n_channels']}")
    print(f"T_pretrain: {d['T_pretrain']}")
    print(f"T_test:     {d['T_test']}")
    print(f"anomaly_rate (overall dataset04): {d['anomaly_rate']:.4f}")
    print(f"pretrain_stream[0]: {d['pretrain_stream'][0].shape}")
    for split in ['ft_train', 'ft_val', 'ft_test']:
        arr = d[split][0]
        print(f"{split}: x={arr['test'].shape}  y_prev={arr['labels'].mean():.4f}")
