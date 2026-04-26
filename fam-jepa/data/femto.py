"""
FEMTO/PRONOSTIA Bearing RUL Dataset Loader.

PHM IEEE 2012 Data Challenge: run-to-failure bearing vibration signals.
- 6 training bearings (full run-to-failure)
- 11 test bearings (partial + remaining useful life annotations)
- 2 channels: horizontal (H) + vertical (V) acceleration at 25.6 kHz
- 2560 samples per snapshot (snapshot every 10s for ~0.1s)
- 3 operating conditions (speed/load)

For FAM pretraining we use summary statistics per snapshot to create a
coarser time series: each snapshot becomes a feature vector (8 features:
RMS_H, RMS_V, peak_H, peak_V, kurtosis_H, kurtosis_V, crest_H, crest_V).
We have ~6k snapshots for the largest training bearing (Bearing1_1),
giving a pretrain sequence of ~6k timesteps in feature space.

For event prediction: the event is bearing failure. Since FEMTO provides
run-to-failure training data, labels[t] = 1 only at t = T-1 (last snapshot).
Test labels include RUL annotations which we convert to precursor windows.

Data source: nested zip at:
  archive/pre-paper/datasets/data/femto/femto_bearing.zip
  -> 10. FEMTO Bearing/FEMTOBearingDataSet.zip
  -> Training_set.zip, Test_set.zip, Validation_Set.zip

Usage:
    from fam_jepa.data.femto import load_femto, check_femto_available
    bundle = load_femto()  # returns None if data not available
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from scipy.stats import kurtosis as _scipy_kurtosis
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# Operating condition mapping: Bearing{cond}_{trial} -> condition
COND_MAP = {
    '1': 'Cond1_1800rpm_4000N',
    '2': 'Cond2_1650rpm_4200N',
    '3': 'Cond3_1500rpm_5000N',
}

# Training bearings (full run-to-failure)
TRAIN_BEARINGS = ['Bearing1_1', 'Bearing1_2', 'Bearing2_1', 'Bearing2_2', 'Bearing3_1', 'Bearing3_2']

# Test bearings (partial, with RUL annotations)
TEST_BEARINGS = ['Bearing1_3', 'Bearing1_4', 'Bearing1_5', 'Bearing1_6', 'Bearing1_7',
                 'Bearing2_3', 'Bearing2_4', 'Bearing2_5', 'Bearing2_6', 'Bearing2_7',
                 'Bearing3_3']

FEMTO_ZIP = Path('/home/sagemaker-user/IndustrialJEPA/archive/pre-paper/datasets/data/femto/femto_bearing.zip')
FEMTO_INNER_PATH = '10. FEMTO Bearing/FEMTOBearingDataSet.zip'

HORIZONS = [1, 5, 10, 20, 50, 100, 150, 200]  # snapshot-level horizons

# Subsample factor: reduce 2560-sample snapshots to a shorter time series
# by computing features (8 per snapshot) rather than using raw samples.
# This makes pretrain feasible at FAM's patch_size=16, max_context=512.
SNAPSHOT_FEATURES = 8  # n_channels for FAM


def check_femto_available() -> bool:
    """Check if FEMTO zip is present."""
    return FEMTO_ZIP.exists()


def _kurtosis(x: np.ndarray) -> float:
    """Fisher kurtosis (kurtosis - 3). Uses scipy if available."""
    if HAS_SCIPY:
        return float(_scipy_kurtosis(x, fisher=True))
    # Fallback: compute manually
    m4 = np.mean((x - np.mean(x)) ** 4)
    std = np.std(x)
    if std < 1e-10:
        return 0.0
    return float(m4 / std ** 4) - 3.0


def _snapshot_to_features(snapshot: np.ndarray) -> np.ndarray:
    """
    Reduce a (2560, 2) vibration snapshot to 8 summary features.

    Features per channel (4 each, 8 total):
    - RMS amplitude
    - Peak amplitude
    - Kurtosis (Fisher)
    - Crest factor (peak / RMS)
    """
    assert snapshot.shape == (2560, 2), f"Expected (2560, 2), got {snapshot.shape}"
    feats = []
    for ch in range(2):
        x = snapshot[:, ch]
        rms = float(np.sqrt(np.mean(x ** 2)))
        peak = float(np.max(np.abs(x)))
        kurt = _kurtosis(x)
        crest = peak / max(rms, 1e-10)
        feats.extend([rms, peak, kurt, crest])
    return np.array(feats, dtype=np.float32)  # (8,)


def _read_csv_snapshot(data: bytes) -> np.ndarray:
    """Parse a FEMTO acc_*.csv file -> (2560, 2) array."""
    lines = data.decode('utf-8', errors='replace').strip().split('\n')
    rows = []
    for line in lines:
        parts = line.split(',')
        if len(parts) >= 6:
            try:
                h = float(parts[4])
                v = float(parts[5])
                rows.append([h, v])
            except ValueError:
                continue
    arr = np.array(rows, dtype=np.float32)
    # Handle truncation/overflow
    if len(arr) > 2560:
        arr = arr[:2560]
    elif len(arr) < 2560:
        # Pad with zeros
        pad = np.zeros((2560 - len(arr), 2), dtype=np.float32)
        arr = np.concatenate([arr, pad], axis=0)
    return arr


def _load_bearing_features(subzip: zipfile.ZipFile,
                             bearing_name: str,
                             folder: str = 'Learning_set') -> Optional[np.ndarray]:
    """
    Load all snapshots for a bearing and return feature matrix (T, 8).

    subzip: the Training_set.zip or Test_set.zip ZipFile object
    bearing_name: e.g. 'Bearing1_1'
    folder: 'Learning_set' or 'Test_set'
    """
    # Get sorted list of acc_*.csv files for this bearing
    prefix = f"{folder}/{bearing_name}/"
    file_list = sorted([
        n for n in subzip.namelist()
        if n.startswith(prefix) and n.endswith('.csv') and 'acc_' in n
    ])

    if not file_list:
        print(f"  WARNING: no files found for {bearing_name} in {folder}")
        return None

    features = []
    for fname in file_list:
        try:
            data = subzip.read(fname)
            snapshot = _read_csv_snapshot(data)
            feat = _snapshot_to_features(snapshot)
            features.append(feat)
        except Exception as e:
            print(f"  WARNING: failed to read {fname}: {e}")
            continue

    if not features:
        return None

    return np.array(features, dtype=np.float32)  # (T, 8)


def _normalize_features(train_feats: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Compute global mean/std over all training bearing features."""
    all_data = np.concatenate(train_feats, axis=0)  # (N_total, 8)
    mu = all_data.mean(axis=0)
    std = all_data.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return mu, std


def load_femto(
    pretrain_fraction: float = 0.7,
    ft_fraction: float = 0.2,
    val_fraction: float = 0.1,
    min_snapshots: int = 100,
    seed: int = 42,
    verbose: bool = True,
) -> Optional[Dict]:
    """
    Load FEMTO bearing dataset as FAM-compatible bundle.

    Returns None if data not available.

    Protocol:
    - Training bearings (6 bearings, full run-to-failure):
      - Used for pretraining AND finetuning (train/val split chronologically)
      - Pretrain: raw feature sequences from all 6 training bearings
      - FT train: first pretrain_fraction of each training bearing
      - FT val: next ft_fraction of each training bearing
      - Test: last test_fraction of each training bearing

    - Test bearings (11 bearings, partial):
      - Used only for test evaluation
      - Labels: binary sequence where labels[t] = 1 at last snapshot (event)

    Note: FEMTO is a pure lifecycle dataset (all bearings fail eventually).
    The event label strategy follows C-MAPSS: labels[T-1] = 1 only.
    A more sophisticated strategy is to label the last failure_window timesteps;
    we use failure_window = 50 snapshots (= ~500s before failure).

    Args:
        pretrain_fraction: fraction of training bearing data for pretraining
        ft_fraction: fraction of training bearing data for finetuning
        val_fraction: fraction of training bearing data for validation
        min_snapshots: minimum snapshots to include a bearing
        seed: random seed (not used for chronological splits)
        verbose: print progress

    Returns:
        Bundle dict with keys: pretrain_seqs, ft_train, ft_val, ft_test,
        n_channels, horizons, name.
    """
    if not check_femto_available():
        print("FEMTO data not available. Expected:", FEMTO_ZIP)
        return None

    if verbose:
        print("Loading FEMTO from nested zip...", flush=True)

    try:
        with zipfile.ZipFile(FEMTO_ZIP) as outer_z:
            inner_zdata = outer_z.read(FEMTO_INNER_PATH)
            inner_z = zipfile.ZipFile(io.BytesIO(inner_zdata))

            # Load training set
            train_zdata = inner_z.read('Training_set.zip')
            train_z = zipfile.ZipFile(io.BytesIO(train_zdata))

            # Load test set
            test_zdata = inner_z.read('Test_set.zip')
            test_z = zipfile.ZipFile(io.BytesIO(test_zdata))
    except Exception as e:
        print(f"Error opening FEMTO zip: {e}")
        return None

    # ---------------------------------------------------------------
    # Load training bearing features
    # ---------------------------------------------------------------
    train_bearing_feats: Dict[str, np.ndarray] = {}
    for bname in TRAIN_BEARINGS:
        if verbose:
            print(f"  Loading training bearing {bname}...", flush=True, end='')
        feats = _load_bearing_features(train_z, bname, folder='Learning_set')
        if feats is not None and len(feats) >= min_snapshots:
            train_bearing_feats[bname] = feats
            if verbose:
                print(f" {len(feats)} snapshots")
        else:
            if verbose:
                print(" SKIP (too short or failed)")

    if not train_bearing_feats:
        print("ERROR: no training bearings loaded")
        return None

    # ---------------------------------------------------------------
    # Global z-score normalization (computed on training bearings)
    # ---------------------------------------------------------------
    mu, std = _normalize_features(list(train_bearing_feats.values()))
    if verbose:
        print(f"  Normalization: mu={mu.round(3)}, std={std.round(3)}", flush=True)

    # Normalize all training bearing features
    for bname in train_bearing_feats:
        train_bearing_feats[bname] = (train_bearing_feats[bname] - mu) / std

    # ---------------------------------------------------------------
    # Build pretrain_seqs (all training bearings)
    # ---------------------------------------------------------------
    pretrain_seqs = {i: feats for i, (bname, feats) in
                     enumerate(train_bearing_feats.items())}

    # ---------------------------------------------------------------
    # Build ft_train, ft_val from training bearings (chronological split)
    # ---------------------------------------------------------------
    FAILURE_WINDOW = 50  # last 50 snapshots are labeled as precursor window

    def make_entity_record(feats: np.ndarray, bearing_name: str) -> Dict:
        T = len(feats)
        labels = np.zeros(T, dtype=np.int32)
        # Mark the last FAILURE_WINDOW snapshots as event precursors
        labels[max(0, T - FAILURE_WINDOW):] = 1
        return {
            'entity_id': bearing_name,
            'test': feats,  # shape (T, 8)
            'labels': labels,
        }

    ft_train_entities = []
    ft_val_entities = []
    ft_test_entities = []

    for bname, feats in train_bearing_feats.items():
        T = len(feats)
        # Chronological split: 70%/20%/10% with no gap (bearings are short)
        t1 = int(pretrain_fraction * T)
        t2 = int((pretrain_fraction + ft_fraction) * T)
        gap = 10  # small gap between splits

        if t1 < min_snapshots or (T - t2) < 10:
            # Too short: use entire sequence for each
            ft_train_entities.append(make_entity_record(feats, bname + '_train'))
            ft_val_entities.append(make_entity_record(feats[t1:t2], bname + '_val'))
            ft_test_entities.append(make_entity_record(feats[t2:], bname + '_test'))
        else:
            ft_train_entities.append(make_entity_record(feats[:t1], bname + '_train'))
            ft_val_entities.append(make_entity_record(feats[t1 + gap:t2], bname + '_val'))
            ft_test_entities.append(make_entity_record(feats[t2 + gap:], bname + '_test'))

    # ---------------------------------------------------------------
    # Load test bearings for additional test evaluation
    # ---------------------------------------------------------------
    if verbose:
        print("  Loading test bearings...", flush=True)

    test_bearing_feats: Dict[str, np.ndarray] = {}
    for bname in TEST_BEARINGS:
        feats = _load_bearing_features(test_z, bname, folder='Test_set')
        if feats is not None and len(feats) >= 10:
            # Normalize using training mean/std
            test_bearing_feats[bname] = (feats - mu) / std
            if verbose:
                print(f"    {bname}: {len(feats)} snapshots")

    # Add test bearings to ft_test
    for bname, feats in test_bearing_feats.items():
        T = len(feats)
        labels = np.zeros(T, dtype=np.int32)
        # Last FAILURE_WINDOW snapshots are labeled as precursor (end-of-life)
        # For test bearings this is conservative since they may be truncated
        labels[max(0, T - FAILURE_WINDOW):] = 1
        ft_test_entities.append({
            'entity_id': bname,
            'test': feats,
            'labels': labels,
        })

    if verbose:
        n_train_w = sum(len(e['test']) for e in ft_train_entities)
        n_test_w = sum(len(e['test']) for e in ft_test_entities)
        print(f"  ft_train: {len(ft_train_entities)} entities, {n_train_w} snapshots", flush=True)
        print(f"  ft_val: {len(ft_val_entities)} entities", flush=True)
        print(f"  ft_test: {len(ft_test_entities)} entities, {n_test_w} snapshots", flush=True)
        print(f"  pretrain_seqs: {len(pretrain_seqs)} sequences", flush=True)
        print(f"  n_channels: {SNAPSHOT_FEATURES}", flush=True)

    return {
        'pretrain_seqs': pretrain_seqs,
        'ft_train': ft_train_entities,
        'ft_val': ft_val_entities,
        'ft_test': ft_test_entities,
        'n_channels': SNAPSHOT_FEATURES,
        'horizons': HORIZONS,
        'name': 'FEMTO',
        'normalization': {'mu': mu.tolist(), 'std': std.tolist()},
        'failure_window': FAILURE_WINDOW,
    }


if __name__ == '__main__':
    print("Testing FEMTO loader...")
    bundle = load_femto(verbose=True)
    if bundle is None:
        print("FEMTO data not available!")
    else:
        print("\nBundle keys:", list(bundle.keys()))
        print(f"n_channels: {bundle['n_channels']}")
        print(f"horizons: {bundle['horizons']}")
        print(f"pretrain_seqs: {len(bundle['pretrain_seqs'])} sequences")
        for i, seq in bundle['pretrain_seqs'].items():
            print(f"  seq {i}: shape {seq.shape}")
        print(f"\nft_train: {len(bundle['ft_train'])} entities")
        for e in bundle['ft_train'][:2]:
            T = len(e['test'])
            n_pos = e['labels'].sum()
            print(f"  {e['entity_id']}: T={T}, positive_labels={n_pos}")
