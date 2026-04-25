"""SKAB (Skoltech Anomaly Benchmark) — hydraulic test rig anomalies.

Source: https://github.com/waico/SKAB (cloned to datasets/data/skab/).

8 sensor channels at 1 Hz from a hydraulic test rig:
  Accelerometer1RMS, Accelerometer2RMS, Current, Pressure, Temperature,
  Thermocouple, Voltage, Volume Flow RateRMS.

Layout on disk:
  data/skab/data/anomaly-free/anomaly-free.csv   — long normal stream (pretrain)
  data/skab/data/valve1/{0..15}.csv              — valve-1 fault experiments
  data/skab/data/valve2/{0..3}.csv               — valve-2 fault experiments
  data/skab/data/other/{0..13}.csv               — other faults
Each labeled CSV has columns sensor*8 + ``anomaly`` (binary) + ``changepoint``.

Protocol: chronological 60/20/20 split BY EXPERIMENT (each experiment is one
~20-min run with a fault injected). Pretrain stream = anomaly-free.csv only,
so the encoder never sees fault dynamics during self-supervised pretraining.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

try:
    from .config import _ROOT as CONFIG_ROOT
    SKAB_DIR = CONFIG_ROOT / 'datasets' / 'data' / 'skab' / 'data'
except Exception:
    SKAB_DIR = Path('/home/sagemaker-user/IndustrialJEPA/datasets/data/skab/data')

SENSOR_COLS = ['Accelerometer1RMS', 'Accelerometer2RMS', 'Current', 'Pressure',
               'Temperature', 'Thermocouple', 'Voltage', 'Volume Flow RateRMS']
N_CHANNELS = len(SENSOR_COLS)
LABEL_COL = 'anomaly'


def _load_experiment(path: Path) -> tuple:
    df = pd.read_csv(path, sep=';')
    X = df[SENSOR_COLS].to_numpy(dtype=np.float32)
    y = df[LABEL_COL].astype(int).to_numpy().astype(np.int32)
    return X, y


def load_skab(normalize: bool = False, gap: int = 200) -> Dict:
    """Load all labeled SKAB experiments, chronological 60/20/20 split.

    Pretrain stream is the long anomaly-free.csv (~9405 timesteps). FT
    train/val/test pull from the labeled experiments by experiment index
    (deterministic).
    """
    pretrain_path = SKAB_DIR / 'anomaly-free' / 'anomaly-free.csv'
    if not pretrain_path.exists():
        raise FileNotFoundError(
            f'SKAB anomaly-free.csv missing at {pretrain_path}. '
            f'Run: git clone https://github.com/waico/SKAB.git {SKAB_DIR.parent}')
    pre_df = pd.read_csv(pretrain_path, sep=';')
    pre_X = pre_df[SENSOR_COLS].to_numpy(dtype=np.float32)

    # Compute pretrain-stream stats for normalization (no leakage)
    mu = pre_X.mean(axis=0)
    std = pre_X.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    if normalize:
        pre_X = (pre_X - mu) / std

    # Labeled experiments — sort deterministically by (subdir, file_index)
    exps: List[tuple] = []
    for sub in ['valve1', 'valve2', 'other']:
        d = SKAB_DIR / sub
        if not d.exists():
            continue
        files = sorted(d.glob('*.csv'), key=lambda p: (sub, int(p.stem)))
        for fp in files:
            X, y = _load_experiment(fp)
            if normalize:
                X = (X - mu) / std
            exps.append((f'{sub}_{fp.stem}', X.astype(np.float32), y))

    n = len(exps)
    n_train = int(0.6 * n)
    n_val = int(0.2 * n)
    train_exps = exps[:n_train]
    val_exps = exps[n_train:n_train + n_val]
    test_exps = exps[n_train + n_val:]

    def to_entities(es):
        return [{'entity_id': name, 'test': X, 'labels': y}
                for (name, X, y) in es]

    return {
        'pretrain_stream': {0: pre_X.astype(np.float32)},
        'ft_train': to_entities(train_exps),
        'ft_val': to_entities(val_exps),
        'ft_test': to_entities(test_exps),
        'n_channels': N_CHANNELS,
        'name': 'SKAB',
        'mu': mu.astype(np.float32),
        'std': std.astype(np.float32),
        'anomaly_rate': float(np.mean(
            [y.mean() for (_, _, y) in exps] if exps else [0.0])),
        'n_experiments': n,
    }


if __name__ == '__main__':
    d = load_skab()
    print(f"n_experiments: {d['n_experiments']} "
          f"(train {len(d['ft_train'])}, val {len(d['ft_val'])}, "
          f"test {len(d['ft_test'])})")
    print(f"pretrain: {d['pretrain_stream'][0].shape}")
    print(f"anomaly_rate (mean over labeled exps): {d['anomaly_rate']:.4f}")
    for k in ('ft_train', 'ft_val', 'ft_test'):
        total = sum(e['test'].shape[0] for e in d[k])
        pos = sum(int(e['labels'].sum()) for e in d[k])
        print(f"{k}: {len(d[k])} exps, {total} steps, {pos} pos "
              f"({100*pos/max(total,1):.2f}%)")
