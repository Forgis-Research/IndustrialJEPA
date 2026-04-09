"""
V8 Data Pipeline: Load, resample to 12800 Hz, extract 1024-sample windows.
Handles pretraining (all bearing sources) and RUL evaluation (FEMTO + XJTU-SY).

Key decisions:
- Single channel (primary vibration accelerometer per source)
- 1024-sample windows at 12,800 Hz (0.08s)
- Non-overlapping windows for pretraining, per-snapshot for RUL
- Episode-based split for RUL: 75% train / 25% test
- Instance normalization per window
"""

import numpy as np
import pandas as pd
import json
import os
import sys
from scipy.signal import resample_poly
from math import gcd
from typing import Optional, List, Dict, Tuple
from collections import defaultdict

CACHE_DIR = '/tmp/hf_cache/bearings'
TARGET_SR = 12800
WINDOW_LEN = 1024  # 0.08s at 12800 Hz

SOURCE_SR = {
    'cwru': 12000,
    'mfpt': 48828,
    'ims': 20480,
    'xjtu_sy': 25600,
    'paderborn': 64000,
    'femto': 25600,
    'mafaulda': 50000,
    'ottawa_bearing': 42000,
    'vbl_va001': 20000,
    'sca_pulpmill': 8192,  # Will upsample
    'mendeley_bearing': 50000,
}

# Sources to use for pretraining (exclude gearbox sources)
PRETRAIN_SOURCES = [
    'cwru', 'mfpt', 'femto', 'xjtu_sy', 'ims',
    'paderborn', 'ottawa_bearing', 'mafaulda', 'vbl_va001', 'sca_pulpmill'
]


def load_parquet(filename: str) -> pd.DataFrame:
    """Load from local cache."""
    local = os.path.join(CACHE_DIR, os.path.basename(filename))
    if os.path.exists(local):
        return pd.read_parquet(local)
    raise FileNotFoundError(f"Not cached: {filename}")


def get_sr(row: pd.Series) -> int:
    """Get native sampling rate from row metadata."""
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


def resample_to_target(ch: np.ndarray, native_sr: int) -> np.ndarray:
    """Resample signal to TARGET_SR using polyphase resampling."""
    if native_sr == TARGET_SR:
        return ch
    g = gcd(native_sr, TARGET_SR)
    up = TARGET_SR // g
    down = native_sr // g
    return resample_poly(ch, up, down).astype(np.float32)


def instance_norm(x: np.ndarray) -> Optional[np.ndarray]:
    """Zero-mean, unit-std normalization."""
    std = x.std()
    if std < 1e-10:
        return None
    return ((x - x.mean()) / std).astype(np.float32)


def extract_windows(signal: np.ndarray, win_len: int = WINDOW_LEN,
                    max_windows: int = 20) -> List[np.ndarray]:
    """
    Extract non-overlapping windows from a signal.
    For very long signals, cap at max_windows to control memory.
    """
    n = len(signal)
    if n < win_len:
        return []
    n_wins = n // win_len
    n_wins = min(n_wins, max_windows)
    windows = []
    for i in range(n_wins):
        w = signal[i * win_len:(i + 1) * win_len]
        w_norm = instance_norm(w)
        if w_norm is not None:
            windows.append(w_norm)
    return windows


# ============================================================
# PRETRAINING DATA LOADING
# ============================================================

def load_pretrain_windows(verbose: bool = True) -> Tuple[np.ndarray, List[str]]:
    """
    Load all bearing windows for JEPA pretraining.
    Returns: (windows, source_ids) where windows is (N, 1024) float32.
    """
    all_windows = []
    all_sources = []
    stats = {}

    # ---- CWRU + MFPT ----
    df = load_parquet('extra_cwru_mfpt.parquet')
    for src in ['cwru', 'mfpt']:
        sub = df[df['source_id'] == src]
        n_windows = 0
        for _, row in sub.iterrows():
            ch = get_ch0(row)
            if ch is None:
                continue
            sr = get_sr(row)
            ch = resample_to_target(ch, sr)
            wins = extract_windows(ch, max_windows=15)
            all_windows.extend(wins)
            all_sources.extend([src] * len(wins))
            n_windows += len(wins)
        stats[src] = n_windows
        if verbose:
            print(f"  {src}: {len(sub)} signals → {n_windows} windows")
    del df

    # ---- IMS ----
    try:
        df = load_parquet('extra_ims.parquet')
        n_windows = 0
        for _, row in df.iterrows():
            ch = get_ch0(row)
            if ch is None:
                continue
            sr = get_sr(row)
            ch = resample_to_target(ch, sr)
            wins = extract_windows(ch, max_windows=10)
            all_windows.extend(wins)
            all_sources.extend(['ims'] * len(wins))
            n_windows += len(wins)
        stats['ims'] = n_windows
        if verbose:
            print(f"  ims: {len(df)} signals → {n_windows} windows")
        del df
    except Exception as e:
        if verbose:
            print(f"  ims: skipped ({e})")

    # ---- MAFAULDA ----
    n_windows = 0
    n_signals = 0
    for i in range(8):
        try:
            df = load_parquet(f'mafaulda_{i:03d}.parquet')
            for _, row in df.iterrows():
                ch = get_ch0(row)
                if ch is None:
                    continue
                sr = get_sr(row)
                ch = resample_to_target(ch, sr)
                wins = extract_windows(ch, max_windows=10)
                all_windows.extend(wins)
                all_sources.extend(['mafaulda'] * len(wins))
                n_windows += len(wins)
                n_signals += 1
            del df
        except Exception:
            break
    stats['mafaulda'] = n_windows
    if verbose:
        print(f"  mafaulda: {n_signals} signals → {n_windows} windows")

    # ---- Ottawa ----
    try:
        df = load_parquet('ottawa_bearings.parquet')
        n_windows = 0
        for _, row in df.iterrows():
            ch = get_ch0(row)
            if ch is None:
                continue
            sr = get_sr(row)
            ch = resample_to_target(ch, sr)
            wins = extract_windows(ch, max_windows=10)
            all_windows.extend(wins)
            all_sources.extend(['ottawa_bearing'] * len(wins))
            n_windows += len(wins)
        stats['ottawa_bearing'] = n_windows
        if verbose:
            print(f"  ottawa: {len(df)} signals → {n_windows} windows")
        del df
    except Exception as e:
        if verbose:
            print(f"  ottawa: skipped ({e})")

    # ---- SEU (bonus source, not in plan but available) ----
    # Skip - not in PRETRAIN_SOURCES

    # ---- Main shards: FEMTO, XJTU-SY, Paderborn, VBL, SCA ----
    shard_sources = {
        'femto': list(range(4)),
        'xjtu_sy': [3],
        'paderborn': [4],
        'vbl_va001': list(range(3)),
        'sca_pulpmill': list(range(3)),
    }

    shard_cache = {}  # avoid loading same shard twice

    for src, shards in shard_sources.items():
        n_windows = 0
        n_signals = 0
        for shard_idx in shards:
            shard_key = f'train-{shard_idx:05d}-of-00005.parquet'
            if shard_key not in shard_cache:
                shard_cache[shard_key] = load_parquet(shard_key)
            df = shard_cache[shard_key]
            sub = df[df['source_id'] == src]
            for _, row in sub.iterrows():
                ch = get_ch0(row)
                if ch is None:
                    continue
                sr = get_sr(row)
                ch = resample_to_target(ch, sr)
                # For FEMTO: snapshots are already short (~1280 samples at 25.6kHz → 640 at 12.8kHz)
                # After resampling, FEMTO signals are ~640 samples — too short for 1024 window
                # Use the full signal and handle short ones
                if len(ch) >= WINDOW_LEN:
                    wins = extract_windows(ch, max_windows=10)
                elif len(ch) >= 512:
                    # Pad to window length
                    padded = np.pad(ch, (0, WINDOW_LEN - len(ch)), mode='wrap')
                    w = instance_norm(padded)
                    wins = [w] if w is not None else []
                else:
                    wins = []
                all_windows.extend(wins)
                all_sources.extend([src] * len(wins))
                n_windows += len(wins)
                n_signals += 1
        stats[src] = n_windows
        if verbose:
            print(f"  {src}: {n_signals} signals → {n_windows} windows")

    shard_cache.clear()

    if not all_windows:
        raise RuntimeError("No windows loaded!")

    X = np.stack(all_windows, axis=0)  # (N, 1024)
    if verbose:
        print(f"\nPretraining total: {X.shape[0]} windows")
        for src, n in sorted(stats.items()):
            print(f"  {src}: {n}")

    return X, all_sources


# ============================================================
# RUL DATA LOADING (Episode-aware)
# ============================================================

def load_rul_episodes(sources: List[str] = ['femto', 'xjtu_sy'],
                      verbose: bool = True) -> Dict[str, List[dict]]:
    """
    Load run-to-failure data, organized by episode.

    For each episode, returns list of snapshots sorted by episode_position,
    with computed elapsed_time_seconds.

    Snapshot intervals:
    - FEMTO: 10 seconds
    - XJTU-SY: 60 seconds

    Returns: Dict[episode_id → list of snapshot dicts]
    """
    SNAPSHOT_INTERVAL = {
        'femto': 10.0,    # seconds between snapshots
        'xjtu_sy': 60.0,  # seconds between snapshots
    }

    episodes = defaultdict(list)

    if 'femto' in sources:
        for shard_idx in range(4):
            df = load_parquet(f'train-{shard_idx:05d}-of-00005.parquet')
            sub = df[(df['source_id'] == 'femto') & (df['rul_percent'].notna())]
            for _, row in sub.iterrows():
                ch = get_ch0(row)
                if ch is None:
                    continue
                sr = get_sr(row)
                ch_resampled = resample_to_target(ch, sr)
                # Handle short signals
                if len(ch_resampled) >= WINDOW_LEN:
                    window = ch_resampled[:WINDOW_LEN]
                elif len(ch_resampled) >= 256:
                    window = np.pad(ch_resampled, (0, WINDOW_LEN - len(ch_resampled)), mode='wrap')
                else:
                    continue
                w_norm = instance_norm(window)
                if w_norm is None:
                    continue

                ep_id = str(row['episode_id'])
                ep_pos = float(row['episode_position'])
                rul = float(row['rul_percent'])

                episodes[ep_id].append({
                    'window': w_norm,
                    'rul_percent': rul,
                    'episode_id': ep_id,
                    'episode_position': ep_pos,
                    'source': 'femto',
                    'snapshot_interval': SNAPSHOT_INTERVAL['femto'],
                })
            del df

    if 'xjtu_sy' in sources:
        for shard_idx in [3]:
            df = load_parquet(f'train-{shard_idx:05d}-of-00005.parquet')
            sub = df[(df['source_id'] == 'xjtu_sy') & (df['rul_percent'].notna())]
            for _, row in sub.iterrows():
                ch = get_ch0(row)
                if ch is None:
                    continue
                sr = get_sr(row)
                ch_resampled = resample_to_target(ch, sr)
                if len(ch_resampled) >= WINDOW_LEN:
                    window = ch_resampled[:WINDOW_LEN]
                elif len(ch_resampled) >= 256:
                    window = np.pad(ch_resampled, (0, WINDOW_LEN - len(ch_resampled)), mode='wrap')
                else:
                    continue
                w_norm = instance_norm(window)
                if w_norm is None:
                    continue

                ep_id = str(row['episode_id'])
                ep_pos = float(row['episode_position'])
                rul = float(row['rul_percent'])

                episodes[ep_id].append({
                    'window': w_norm,
                    'rul_percent': rul,
                    'episode_id': ep_id,
                    'episode_position': ep_pos,
                    'source': 'xjtu_sy',
                    'snapshot_interval': SNAPSHOT_INTERVAL['xjtu_sy'],
                })
            del df

    # Sort each episode by position and compute elapsed_time_seconds
    for ep_id, snapshots in episodes.items():
        snapshots.sort(key=lambda s: s['episode_position'])
        interval = snapshots[0]['snapshot_interval']
        n = len(snapshots)
        for i, s in enumerate(snapshots):
            # Normalize position to [0,1]
            s['episode_position_norm'] = i / max(n - 1, 1)
            s['elapsed_time_seconds'] = i * interval
            s['delta_t'] = interval  # constant for these datasets
            s['lifetime_seconds'] = n * interval

    if verbose:
        print(f"\nRUL episodes loaded:")
        src_eps = defaultdict(list)
        for ep_id, snaps in episodes.items():
            src_eps[snaps[0]['source']].append(len(snaps))
        for src, lengths in sorted(src_eps.items()):
            interval = SNAPSHOT_INTERVAL.get(src, 0)
            lifetimes_s = [l * interval for l in lengths]
            print(f"  {src}: {len(lengths)} episodes, "
                  f"snapshots: {min(lengths)}-{max(lengths)} "
                  f"(mean={np.mean(lengths):.0f}), "
                  f"lifetime: {min(lifetimes_s)/3600:.1f}h-{max(lifetimes_s)/3600:.1f}h "
                  f"(mean={np.mean(lifetimes_s)/3600:.1f}h, std={np.std(lifetimes_s)/3600:.2f}h)")

    return dict(episodes)


def episode_train_test_split(episodes: Dict[str, List[dict]],
                             test_ratio: float = 0.25,
                             seed: int = 42,
                             verbose: bool = True):
    """
    Episode-based 75/25 train/test split.
    Split separately by source to ensure balance.
    """
    rng = np.random.RandomState(seed)

    # Group by source
    by_source = defaultdict(list)
    for ep_id, snaps in episodes.items():
        by_source[snaps[0]['source']].append(ep_id)

    train_eps = []
    test_eps = []

    for src, ep_ids in sorted(by_source.items()):
        ep_ids = sorted(ep_ids)  # deterministic order
        rng.shuffle(ep_ids)
        n_test = max(1, int(len(ep_ids) * test_ratio))
        n_test = min(n_test, len(ep_ids) - 1)  # keep at least 1 train
        test_eps.extend(ep_ids[:n_test])
        train_eps.extend(ep_ids[n_test:])
        if verbose:
            print(f"  {src}: {len(ep_ids)} total → {len(ep_ids) - n_test} train / {n_test} test")

    if verbose:
        print(f"  Total: {len(train_eps)} train, {len(test_eps)} test episodes")

    return train_eps, test_eps


def compute_piecewise_rul(episodes: Dict[str, List[dict]],
                           onset_kurtosis_threshold: float = 2.0) -> Dict[str, List[float]]:
    """
    Compute piecewise-linear RUL labels.
    Onset detection: when kurtosis exceeds threshold (proxy for degradation start).

    Returns: Dict[episode_id → list of rul_piecewise values]
    """
    from scipy.stats import kurtosis as scipy_kurtosis

    piecewise_rul = {}
    for ep_id, snapshots in episodes.items():
        n = len(snapshots)
        T = n - 1  # total index

        # Find degradation onset
        onset_idx = None
        for i, s in enumerate(snapshots):
            w = s['window']
            kurt = float(scipy_kurtosis(w, fisher=True))
            if kurt > onset_kurtosis_threshold:
                onset_idx = i
                break

        if onset_idx is None or onset_idx == T:
            onset_idx = 0  # default: linear from start

        # Piecewise linear: clamped to 1.0 until onset, then linear decline
        ruls = []
        for i in range(n):
            if i < onset_idx:
                ruls.append(1.0)
            else:
                remaining = (T - i) / max(T - onset_idx, 1)
                ruls.append(max(0.0, remaining))
        piecewise_rul[ep_id] = ruls

    return piecewise_rul


def compute_envelope_rms_per_snapshot(snapshots: List[dict]) -> np.ndarray:
    """Compute envelope RMS for each snapshot (health indicator)."""
    from scipy.signal import hilbert
    env_rms = []
    for s in snapshots:
        w = s['window']
        try:
            env = np.abs(hilbert(w))
            env_rms.append(float(np.sqrt(np.mean(env ** 2))))
        except Exception:
            env_rms.append(0.0)
    return np.array(env_rms, dtype=np.float32)


def compute_handcrafted_features_per_snapshot(snapshots: List[dict]) -> np.ndarray:
    """Extract 18 handcrafted features for each snapshot."""
    sys.path.insert(0, '/home/sagemaker-user/IndustrialJEPA/mechanical-jepa')
    from baselines.features import extract_features
    feats = []
    for s in snapshots:
        f = extract_features(s['window'], sr=TARGET_SR)
        feats.append(f)
    return np.array(feats, dtype=np.float32)  # (n_snapshots, 18)


if __name__ == '__main__':
    print("=== V8 Data Pipeline Test ===\n")

    print("--- Loading pretraining data ---")
    X, sources = load_pretrain_windows(verbose=True)
    print(f"\nPretraining array: {X.shape}, dtype={X.dtype}")
    print(f"Signal range: [{X.min():.3f}, {X.max():.3f}]")

    print("\n--- Loading RUL episodes ---")
    episodes = load_rul_episodes(['femto', 'xjtu_sy'], verbose=True)

    print("\n--- Episode train/test split ---")
    train_eps, test_eps = episode_train_test_split(episodes, seed=42)

    print("\n--- Lifetime variance check ---")
    all_lifetimes = []
    for ep_id, snaps in episodes.items():
        all_lifetimes.append(snaps[0]['lifetime_seconds'])
    all_lifetimes = np.array(all_lifetimes)
    print(f"Episode lifetimes: min={all_lifetimes.min()/3600:.2f}h, "
          f"max={all_lifetimes.max()/3600:.2f}h, "
          f"mean={all_lifetimes.mean()/3600:.2f}h, "
          f"std={all_lifetimes.std()/3600:.2f}h")
    print(f"CV (std/mean) = {all_lifetimes.std()/all_lifetimes.mean():.3f}")
    print("High CV means elapsed-time-only baseline will struggle.")

    print("\n--- Piecewise RUL labels ---")
    piecewise = compute_piecewise_rul(episodes)
    ep_sample = list(episodes.keys())[0]
    linear_rul = [s['rul_percent'] for s in episodes[ep_sample]]
    pw_rul = piecewise[ep_sample]
    print(f"Episode {ep_sample} ({len(linear_rul)} snapshots):")
    print(f"  Linear RUL:    [{linear_rul[0]:.3f}, ..., {linear_rul[len(linear_rul)//2]:.3f}, ..., {linear_rul[-1]:.3f}]")
    print(f"  Piecewise RUL: [{pw_rul[0]:.3f}, ..., {pw_rul[len(pw_rul)//2]:.3f}, ..., {pw_rul[-1]:.3f}]")

    print("\n=== Data Pipeline OK ===")
