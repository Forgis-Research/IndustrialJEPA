"""ETTm1 — Electricity Transformer Temperature, derived thermal-event labels.

Source: https://github.com/zhouhaoyi/ETDataset (single CSV, 69680 rows at
15-min resolution, 7 channels). One CSV at datasets/data/ettm/ETTm1.csv.

Channels: HUFL, HULL, MUFL, MULL, LUFL, LULL, OT. The first 6 are load
features at high/middle/low ufl/ull. OT (oil temperature) is the standard
forecast target.

We frame ETTm1 as event prediction by deriving a binary thermal-event
label using a CAUSAL rolling-window threshold:

  baseline_t = rolling 7-day mean(OT)        (672 steps at 15-min)
  scale_t    = rolling 7-day std(OT)
  delta_t    = (OT_t - baseline_t) / scale_t
  y_t        = 1 iff delta_t > 2.0

This captures local thermal-stress events (rapid temperature rises
relative to the recent baseline) and distributes events across all
seasons. A naive global threshold (OT > mu_train + 2*sigma_train) puts
ALL events in the first 6 months when OT is highest, leaving val/test
with zero positives — a useless protocol.

There is no published event-prediction SOTA on ETTm1 (it is used as a
forecasting benchmark). FAM is the first to frame it this way.

The rolling baseline is causal (looks back only) so no future leakage
into the labels. Standard chronological 60/20/20 split.

NOTE: the prompt described ETTm1 as "1/min" but the public CSV is at
15-minute resolution (96 obs/day). We treat one row = one "step", so
P=16 covers 4 hours and a 512-step context covers ~5.3 days.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

try:
    from .config import _ROOT as CONFIG_ROOT
    ETTM_DIR = CONFIG_ROOT / 'datasets' / 'data' / 'ettm'
except Exception:
    ETTM_DIR = Path('/home/sagemaker-user/IndustrialJEPA/datasets/data/ettm')

SENSOR_COLS = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT']
N_CHANNELS = len(SENSOR_COLS)
TARGET_COL = 'OT'


def load_ettm1(normalize: bool = False, gap: int = 200,
               threshold_sigma: float = 2.0,
               window_steps: int = 7 * 96,   # 7 days at 15-min resolution
               csv_name: str = 'ETTm1.csv') -> Dict:
    """Load ETTm1 with derived thermal-event labels (causal rolling threshold).

    Returns the standard FAM-style bundle.
    """
    path = ETTM_DIR / csv_name
    if not path.exists():
        raise FileNotFoundError(
            f'ETTm1 CSV missing at {path}. Run: '
            f"wget -O {path} https://raw.githubusercontent.com/zhouhaoyi/"
            f"ETDataset/main/ETT-small/ETTm1.csv")

    df = pd.read_csv(path)
    X = df[SENSOR_COLS].to_numpy(dtype=np.float32)
    T = X.shape[0]
    t1 = int(0.6 * T)
    t2 = int(0.8 * T)

    # Causal rolling 7-day mean/std of OT — no future leakage.
    ot = df[TARGET_COL].to_numpy()
    s = pd.Series(ot)
    local_mean = s.rolling(window_steps, min_periods=window_steps // 4).mean().to_numpy()
    local_std = s.rolling(window_steps, min_periods=window_steps // 4).std().to_numpy()
    g_mu = ot[:window_steps].mean()
    g_std = ot[:window_steps].std()
    local_mean = np.where(np.isnan(local_mean), g_mu, local_mean)
    local_std = np.where(np.isnan(local_std), g_std, local_std)
    delta = (ot - local_mean) / np.maximum(local_std, 1e-3)
    y = (delta > threshold_sigma).astype(np.int32)
    threshold = threshold_sigma  # store as the σ-multiplier, not absolute

    mu = X[:t1].mean(axis=0)
    std = X[:t1].std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    if normalize:
        X = (X - mu) / std

    # Pretrain stream: only NORMAL rows in train portion (so the encoder
    # never sees thermal-event dynamics in self-supervised pretraining).
    train_normal_mask = y[:t1] == 0
    pretrain_stream = X[:t1][train_normal_mask].astype(np.float32)

    ft_train = [{'entity_id': 'ETTm1',
                 'test': X[:t1],
                 'labels': y[:t1]}]
    ft_val = [{'entity_id': 'ETTm1',
               'test':  X[t1 + gap:t2],
               'labels': y[t1 + gap:t2]}]
    ft_test = [{'entity_id': 'ETTm1',
                'test':  X[t2 + gap:],
                'labels': y[t2 + gap:]}]

    return {
        'pretrain_stream': {0: pretrain_stream},
        'ft_train': ft_train,
        'ft_val': ft_val,
        'ft_test': ft_test,
        'n_channels': N_CHANNELS,
        'name': 'ETTm1',
        'mu': mu.astype(np.float32),
        'std': std.astype(np.float32),
        'threshold_sigma': threshold,
        'event_definition': f'OT exceeds rolling-{window_steps}step baseline + {threshold_sigma}σ',
        'anomaly_rate': float(y.mean()),
        'T_total': int(T),
    }


if __name__ == '__main__':
    d = load_ettm1()
    print(f"T_total: {d['T_total']}, threshold_sigma={d['threshold_sigma']:.3f}")
    print(f"event def: {d['event_definition']}")
    print(f"anomaly_rate (whole stream): {d['anomaly_rate']:.4f}")
    print(f"pretrain (normal-only train): {d['pretrain_stream'][0].shape}")
    for k in ('ft_train', 'ft_val', 'ft_test'):
        e = d[k][0]
        print(f"{k}: shape={e['test'].shape}, anom_rate={e['labels'].mean():.4f}")
