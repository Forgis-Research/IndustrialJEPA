"""
Unified data loading for Mechanical-JEPA baselines.

Uses locally cached parquet files for speed. Reads row-by-row to minimize peak memory.
Pre-processing:
- Feature extraction: native signal at native SR (no resampling needed)
- Deep models: resample to 12800 Hz, use 8192-sample window
"""

import numpy as np
import pandas as pd
import json
import os
from scipy.signal import resample_poly
from math import gcd
from typing import Optional, List, Tuple

CACHE_DIR = '/tmp/hf_cache/bearings'
TOKEN = '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'
HF_BASE = 'hf://datasets/Forgis/Mechanical-Components'
STORAGE_OPTS = {'token': TOKEN}

TARGET_SR = 12800
DEEP_WINDOW_LEN = 8192  # 0.64s at 12800 Hz

SOURCE_SR = {
    'cwru': 12000, 'mfpt': 48828, 'ims': 20480, 'xjtu_sy': 25600,
    'paderborn': 64000, 'femto': 25600, 'mafaulda': 50000,
    'seu': 5120, 'mcc5_thu': 12800, 'ottawa_bearing': 42000,
    'vbl_va001': 20000, 'sca_pulpmill': 25600, 'mendeley_bearing': 50000,
}


def local_path(filename: str) -> str:
    """Get local cached path for a parquet file."""
    basename = os.path.basename(filename)
    return os.path.join(CACHE_DIR, basename)


def load_parquet(filename: str) -> pd.DataFrame:
    """Load from local cache; fall back to HF if not cached."""
    local = local_path(filename)
    if os.path.exists(local):
        return pd.read_parquet(local)
    else:
        return pd.read_parquet(f'{HF_BASE}/{filename}', storage_options=STORAGE_OPTS)


def get_sr(row: pd.Series) -> int:
    """Get sampling rate from row metadata."""
    meta = row.get('extra_metadata', None)
    if meta is not None:
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except Exception:
                meta = {}
        if isinstance(meta, dict) and 'sampling_rate_hz' in meta:
            return int(meta['sampling_rate_hz'])
    source = str(row.get('source_id', '')).lower()
    return SOURCE_SR.get(source, TARGET_SR)


def get_ch0(row: pd.Series) -> Optional[np.ndarray]:
    """Extract channel 0 as float32."""
    try:
        sig = np.array(row['signal'])
        ch = np.array(sig[0], dtype=np.float32)
        return ch if len(ch) >= 64 else None
    except Exception:
        return None


def norm(x: np.ndarray) -> Optional[np.ndarray]:
    """Zero-mean, unit-std. Returns None if degenerate."""
    std = x.std()
    if std < 1e-10:
        return None
    return ((x - x.mean()) / std).astype(np.float32)


def proc_native(row: pd.Series) -> Optional[np.ndarray]:
    """Native signal for feature extraction."""
    ch = get_ch0(row)
    if ch is None:
        return None
    return norm(ch)


def proc_deep(row: pd.Series, win_len: int = DEEP_WINDOW_LEN) -> Optional[np.ndarray]:
    """Resample + fixed window for deep models."""
    ch = get_ch0(row)
    if ch is None:
        return None
    sr = get_sr(row)
    if sr != TARGET_SR:
        g = gcd(sr, TARGET_SR)
        ch = resample_poly(ch, TARGET_SR // g, sr // g).astype(np.float32)
    if len(ch) < win_len:
        return None
    start = (len(ch) - win_len) // 2
    return norm(ch[start:start + win_len])


# ============================================================
# TASK 1: CLASSIFICATION
# ============================================================

def load_classification_data(verbose: bool = True) -> Tuple:
    """
    Load classification data (native signals, variable length).
    Train: CWRU + MAFAULDA + SEU
    Test: Ottawa + Paderborn (from shard 4)
    Returns: X_train (list), y_train, X_test, y_test, src_train, src_test
    """
    train_sigs, train_labels, train_srcs = [], [], []
    test_sigs, test_labels, test_srcs = [], [], []

    def _ingest(df, sigs_list, labels_list, srcs_list, filter_src=None):
        if filter_src:
            if isinstance(filter_src, str):
                df = df[df['source_id'] == filter_src]
            else:
                df = df[df['source_id'].isin(filter_src)]
        n = 0
        for _, row in df.iterrows():
            w = proc_native(row)
            if w is not None:
                sigs_list.append(w)
                labels_list.append(str(row['fault_type']))
                srcs_list.append(str(row['source_id']))
                n += 1
        return n

    # TRAIN
    df = load_parquet('bearings/extra_cwru_mfpt.parquet')
    n = _ingest(df, train_sigs, train_labels, train_srcs, ['cwru', 'mfpt'])
    if verbose: print(f"CWRU+MFPT: {n}")
    del df

    for i in range(8):
        try:
            df = load_parquet(f'bearings/mafaulda_{i:03d}.parquet')
            _ingest(df, train_sigs, train_labels, train_srcs)
            del df
        except Exception:
            break
    if verbose: print(f"MAFAULDA: {sum(1 for s in train_srcs if s == 'mafaulda')}")

    df = load_parquet('bearings/seu_bearings.parquet')
    n = _ingest(df, train_sigs, train_labels, train_srcs)
    if verbose: print(f"SEU: {n}")
    del df

    # TEST: Ottawa
    df = load_parquet('bearings/ottawa_bearings.parquet')
    n = _ingest(df, test_sigs, test_labels, test_srcs)
    if verbose: print(f"Ottawa: {n}")
    del df

    # TEST: Paderborn (in shard 4 — large file, process carefully)
    df = load_parquet('bearings/train-00004-of-00005.parquet')
    n = _ingest(df, test_sigs, test_labels, test_srcs, 'paderborn')
    if verbose: print(f"Paderborn: {n}")
    del df

    if verbose:
        print(f"Total train: {len(train_sigs)}, test: {len(test_sigs)}")

    return (train_sigs, np.array(train_labels), test_sigs, np.array(test_labels),
            np.array(train_srcs), np.array(test_srcs))


def load_classification_data_deep(verbose: bool = True) -> Tuple:
    """
    Load fixed-length windows for deep models.
    Train: CWRU + SEU (long enough signals)
    Test: Paderborn (also long enough)
    """
    train_wins, train_labels, train_srcs = [], [], []
    test_wins, test_labels, test_srcs = [], [], []

    def _ingest_deep(df, wins_list, labels_list, srcs_list, filter_src=None):
        if filter_src:
            if isinstance(filter_src, str):
                df = df[df['source_id'] == filter_src]
            else:
                df = df[df['source_id'].isin(filter_src)]
        n = 0
        for _, row in df.iterrows():
            w = proc_deep(row)
            if w is not None:
                wins_list.append(w)
                labels_list.append(str(row['fault_type']))
                srcs_list.append(str(row['source_id']))
                n += 1
        return n

    # CWRU (12kHz, 20-40s → resampled to 12800Hz, long signals)
    df = load_parquet('bearings/extra_cwru_mfpt.parquet')
    n = _ingest_deep(df, train_wins, train_labels, train_srcs, ['cwru'])
    if verbose: print(f"CWRU deep: {n}")
    del df

    # SEU (5120Hz, 5s → 12800Hz * 5s = 64000 samples → take 8192)
    df = load_parquet('bearings/seu_bearings.parquet')
    n = _ingest_deep(df, train_wins, train_labels, train_srcs)
    if verbose: print(f"SEU deep: {n}")
    del df

    # TEST: Paderborn (64kHz, 4s → 12800Hz * 4s = 51200 → take 8192)
    df = load_parquet('bearings/train-00004-of-00005.parquet')
    n = _ingest_deep(df, test_wins, test_labels, test_srcs, 'paderborn')
    if verbose: print(f"Paderborn deep: {n}")
    del df

    if not train_wins or not test_wins:
        return (np.array([]), np.array([]), np.array([]), np.array([]),
                np.array([]), np.array([]))

    X_tr = np.array(train_wins, dtype=np.float32)
    X_te = np.array(test_wins, dtype=np.float32)
    if verbose:
        print(f"Deep shapes: train={X_tr.shape}, test={X_te.shape}")

    return (X_tr, np.array(train_labels), X_te, np.array(test_labels),
            np.array(train_srcs), np.array(test_srcs))


# ============================================================
# TASK 2: ANOMALY DETECTION
# ============================================================

def load_anomaly_data(source: str = 'femto', verbose: bool = True) -> Tuple:
    """
    One-class anomaly detection: train on healthy, test on mix.
    Returns: X_train (list), X_test (list), y_test (0=healthy, 1=anomaly)
    """
    if source == 'femto':
        dfs = []
        for i in range(4):
            df = load_parquet(f'bearings/train-{i:05d}-of-00005.parquet')
            dfs.append(df[df['source_id'] == 'femto'])
            del df
        df = pd.concat(dfs, ignore_index=True)

    elif source == 'cwru':
        df = load_parquet('bearings/extra_cwru_mfpt.parquet')
        df = df[df['source_id'] == 'cwru'].reset_index(drop=True)

    elif source == 'mafaulda':
        dfs = []
        for i in range(8):
            try:
                dfs.append(load_parquet(f'bearings/mafaulda_{i:03d}.parquet'))
            except Exception:
                break
        df = pd.concat(dfs, ignore_index=True)
    else:
        raise ValueError(f"Unknown source: {source}")

    healthy = df[df['health_state'] == 'healthy'].reset_index(drop=True)
    faulty = df[df['health_state'].isin(['degrading', 'faulty'])].reset_index(drop=True)
    n_train = max(1, int(0.8 * len(healthy)))

    if verbose:
        print(f"  {source}: {len(healthy)} healthy, {len(faulty)} anomalous")
        print(f"  Train: {n_train}, Test: {len(healthy)-n_train+len(faulty)}")

    X_train = [proc_native(row) for _, row in healthy.iloc[:n_train].iterrows()]
    X_train = [x for x in X_train if x is not None]

    X_test, y_test = [], []
    for _, row in healthy.iloc[n_train:].iterrows():
        w = proc_native(row)
        if w is not None:
            X_test.append(w)
            y_test.append(0)
    for _, row in faulty.iterrows():
        w = proc_native(row)
        if w is not None:
            X_test.append(w)
            y_test.append(1)

    del df
    if verbose:
        print(f"  Valid: train={len(X_train)}, test={len(X_test)}, anomaly_rate={sum(y_test)/len(y_test):.2%}")

    return X_train, X_test, np.array(y_test)


# ============================================================
# TASK 3 & 4: RUL
# ============================================================

def load_rul_data(sources: List[str] = ['femto', 'xjtu_sy'], verbose: bool = True) -> List[dict]:
    """
    Load run-to-failure data. Returns list of dicts with signal, rul_percent, episode_id, etc.
    """
    all_rows = []

    if 'femto' in sources:
        for i in range(4):
            df = load_parquet(f'bearings/train-{i:05d}-of-00005.parquet')
            df = df[(df['source_id'] == 'femto') & (df['rul_percent'].notna())]
            for _, row in df.iterrows():
                ch = get_ch0(row)
                if ch is not None and len(ch) >= 64:
                    all_rows.append({
                        'signal': ch, 'sr': get_sr(row),
                        'rul_percent': float(row['rul_percent']),
                        'episode_id': str(row['episode_id']),
                        'episode_position': float(row['episode_position']),
                        'source': 'femto',
                    })
            del df

    if 'xjtu_sy' in sources:
        # XJTU-SY is in shard 3 (647 rows) and shard 4 (723 rows but also has paderborn)
        for shard_idx in [3]:
            df = load_parquet(f'bearings/train-{shard_idx:05d}-of-00005.parquet')
            df = df[(df['source_id'] == 'xjtu_sy') & (df['rul_percent'].notna())]
            for _, row in df.iterrows():
                ch = get_ch0(row)
                if ch is not None and len(ch) >= 64:
                    all_rows.append({
                        'signal': ch, 'sr': get_sr(row),
                        'rul_percent': float(row['rul_percent']),
                        'episode_id': str(row['episode_id']),
                        'episode_position': float(row['episode_position']),
                        'source': 'xjtu_sy',
                    })
            del df

    if verbose:
        from collections import Counter
        srcs = Counter(r['source'] for r in all_rows)
        eps = len(set(r['episode_id'] for r in all_rows))
        print(f"RUL data: {len(all_rows)} samples, sources={dict(srcs)}, episodes={eps}")

    return all_rows
