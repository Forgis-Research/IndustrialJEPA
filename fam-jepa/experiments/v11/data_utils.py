"""
V11 Data Utilities: C-MAPSS Turbofan Engine Dataset
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import torch
from torch.utils.data import Dataset

# ============================================================
# Constants
# ============================================================
CMAPSS_DATA_DIR = "/home/sagemaker-user/IndustrialJEPA/datasets/data/cmapss/6. Turbofan Engine Degradation Simulation Data Set"

# 1-indexed sensor numbers to keep (dropping s1,5,6,10,16,18,19 as near-constant)
SELECTED_SENSORS = [2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21]
N_SENSORS = len(SELECTED_SENSORS)  # 14

# Column names for CMAPSS files (space separated)
COL_NAMES = (
    ['engine_id', 'cycle'] +
    ['op1', 'op2', 'op3'] +
    [f's{i}' for i in range(1, 22)]
)
N_COLS = len(COL_NAMES)  # 26

RUL_CAP = 125


def load_raw(subset: str) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Load raw CMAPSS train/test/RUL files for a subset (e.g., 'FD001')."""
    base = CMAPSS_DATA_DIR
    train_path = os.path.join(base, f"train_{subset}.txt")
    test_path = os.path.join(base, f"test_{subset}.txt")
    rul_path = os.path.join(base, f"RUL_{subset}.txt")

    train_df = pd.read_csv(train_path, sep=' ', header=None, names=COL_NAMES,
                           index_col=False)
    # Drop trailing NaN columns if any
    train_df = train_df.dropna(axis=1, how='all')
    test_df = pd.read_csv(test_path, sep=' ', header=None, names=COL_NAMES,
                          index_col=False)
    test_df = test_df.dropna(axis=1, how='all')
    rul_arr = pd.read_csv(rul_path, header=None).values.flatten()
    return train_df, test_df, rul_arr


def compute_rul_labels(n_cycles: int, rul_max: int = RUL_CAP) -> np.ndarray:
    """Compute piecewise-linear capped RUL labels for a single engine."""
    rul = np.arange(n_cycles, 0, -1, dtype=np.float32)
    return np.minimum(rul, rul_max)


def get_sensor_cols() -> List[str]:
    """Return column names for selected sensors."""
    return [f's{i}' for i in SELECTED_SENSORS]


def get_op_cols() -> List[str]:
    return ['op1', 'op2', 'op3']


def fit_normalizer(train_df: pd.DataFrame,
                   per_condition: bool = False
                   ) -> Dict:
    """
    Compute normalization stats from training data.
    - per_condition=False: global min-max per sensor (for FD001/FD003)
    - per_condition=True: per-op-condition min-max per sensor (for FD002/FD004)
    Returns a dict with keys (condition_label or 'global') -> {sensor: (min, max)}
    """
    sensor_cols = get_sensor_cols()
    stats = {}

    if not per_condition:
        s = {}
        for col in sensor_cols:
            s[col] = (float(train_df[col].min()), float(train_df[col].max()))
        stats['global'] = s
    else:
        # Cluster op conditions by rounding op1 to nearest 0.5 (FD002/004 have 6 conditions)
        from sklearn.cluster import KMeans
        op_cols = get_op_cols()
        op_data = train_df[op_cols].values
        # Use KMeans with 6 clusters
        kmeans = KMeans(n_clusters=6, random_state=42, n_init=10)
        train_df = train_df.copy()
        train_df['_cond'] = kmeans.fit_predict(op_data)
        stats['_kmeans'] = kmeans
        for cond in range(6):
            mask = train_df['_cond'] == cond
            s = {}
            for col in sensor_cols:
                s[col] = (float(train_df.loc[mask, col].min()),
                          float(train_df.loc[mask, col].max()))
            stats[cond] = s
    return stats


def normalize_row(row_sensors: np.ndarray,
                  stats: Dict,
                  condition: Optional[int] = None) -> np.ndarray:
    """Normalize a single row of sensor values using precomputed stats."""
    sensor_cols = get_sensor_cols()
    if condition is not None and condition in stats:
        s = stats[condition]
    else:
        s = stats['global']
    result = np.zeros_like(row_sensors, dtype=np.float32)
    for i, col in enumerate(sensor_cols):
        mn, mx = s[col]
        if mx > mn:
            result[i] = (row_sensors[i] - mn) / (mx - mn)
        else:
            result[i] = 0.0
    return result


def build_engine_sequences(df: pd.DataFrame,
                            stats: Dict,
                            per_condition: bool = False
                            ) -> Dict[int, np.ndarray]:
    """
    Build normalized sensor sequences per engine.
    Returns dict: engine_id -> array of shape (T, N_SENSORS)
    """
    sensor_cols = get_sensor_cols()
    op_cols = get_op_cols()
    sequences = {}

    kmeans = stats.get('_kmeans', None)

    for eid, grp in df.groupby('engine_id'):
        grp = grp.sort_values('cycle')
        sensor_vals = grp[sensor_cols].values.astype(np.float32)

        if per_condition and kmeans is not None:
            op_vals = grp[op_cols].values
            conditions = kmeans.predict(op_vals)
            rows = []
            for i in range(len(sensor_vals)):
                rows.append(normalize_row(sensor_vals[i], stats, conditions[i]))
            sequences[int(eid)] = np.stack(rows)
        else:
            rows = [normalize_row(sensor_vals[i], stats) for i in range(len(sensor_vals))]
            sequences[int(eid)] = np.stack(rows)

    return sequences


def load_cmapss_subset(subset: str = 'FD001') -> Dict:
    """
    Load and preprocess one CMAPSS subset.
    Returns dict with:
      - train_engines: {id: (T, 14) array}
      - val_engines: {id: (T, 14) array}
      - test_engines: {id: (T, 14) array}
      - test_rul: array of shape (N_test,) - ground truth RUL at last cycle
      - train_ids, val_ids
      - stats: normalization stats
      - per_condition: bool
    """
    train_df, test_df, rul_arr = load_raw(subset)
    per_condition = subset in ('FD002', 'FD004')

    # Fit normalizer on train data only
    stats = fit_normalizer(train_df, per_condition=per_condition)

    # Build sequences
    all_train_seqs = build_engine_sequences(train_df, stats, per_condition)
    all_test_seqs = build_engine_sequences(test_df, stats, per_condition)

    # Train/val split: 85%/15% by engine_id, seed=42
    all_ids = sorted(all_train_seqs.keys())
    rng = np.random.default_rng(42)
    n_val = max(1, int(0.15 * len(all_ids)))
    val_ids = set(rng.choice(all_ids, size=n_val, replace=False).tolist())
    train_ids = [i for i in all_ids if i not in val_ids]

    train_engines = {i: all_train_seqs[i] for i in train_ids}
    val_engines = {i: all_train_seqs[i] for i in val_ids}
    test_engines = {i: all_test_seqs[i] for i in sorted(all_test_seqs.keys())}

    return {
        'train_engines': train_engines,
        'val_engines': val_engines,
        'test_engines': test_engines,
        'test_rul': rul_arr.astype(np.float32),
        'train_ids': train_ids,
        'val_ids': list(val_ids),
        'stats': stats,
        'per_condition': per_condition,
        'subset': subset,
        'raw_train_df': train_df,
        'raw_test_df': test_df,
    }


def sanity_check_fd001(train_df: pd.DataFrame, test_df: pd.DataFrame,
                        rul_arr: np.ndarray):
    """Assert known FD001 shapes and statistics."""
    assert train_df.shape == (20631, 26), f"train shape: {train_df.shape}"
    assert test_df.shape == (13096, 26), f"test shape: {test_df.shape}"
    assert rul_arr.shape == (100,), f"RUL shape: {rul_arr.shape}"
    n_train_eng = train_df['engine_id'].nunique()
    n_test_eng = test_df['engine_id'].nunique()
    assert n_train_eng == 100, f"train engines: {n_train_eng}"
    assert n_test_eng == 100, f"test engines: {n_test_eng}"
    print("FD001 sanity checks PASSED")


# ============================================================
# Datasets
# ============================================================

class CMAPSSPretrainDataset(Dataset):
    """
    Pretraining dataset - NO RUL labels used.
    Each item: (past_sensors, future_sensors, k, t)
    where past = x[0:t], future = x[t:t+k]
    """

    def __init__(self,
                 engines: Dict[int, np.ndarray],
                 n_cuts_per_engine: int = 20,
                 min_past: int = 10,
                 min_horizon: int = 5,
                 max_horizon: int = 30,
                 seed: int = 42):
        self.min_past = min_past
        self.min_horizon = min_horizon
        self.max_horizon = max_horizon
        self.rng = np.random.default_rng(seed)

        # Enumerate all (engine_id, cut_t, k) triples
        self.items = []
        for eid, seq in engines.items():
            T = len(seq)
            for _ in range(n_cuts_per_engine):
                # Sample k first
                k = int(self.rng.integers(min_horizon, max_horizon + 1))
                # Sample t: need at least min_past past, at least k future
                t_min = min_past
                t_max = T - k
                if t_min > t_max:
                    continue
                t = int(self.rng.integers(t_min, t_max + 1))
                self.items.append((seq, t, k))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        seq, t, k = self.items[idx]
        past = torch.from_numpy(seq[:t])   # (t, 14)
        future = torch.from_numpy(seq[t:t + k])  # (k, 14)
        return past, future, k, t


class CMAPSSFinetuneDataset(Dataset):
    """
    Fine-tuning dataset - uses RUL labels.
    Each item: (past_sensors up to cycle t, rul_normalized)
    """

    def __init__(self,
                 engines: Dict[int, np.ndarray],
                 rul_cap: int = RUL_CAP,
                 n_cuts_per_engine: int = 5,
                 seed: int = 42,
                 use_last_only: bool = False):
        self.rul_cap = rul_cap
        rng = np.random.default_rng(seed)
        self.items = []

        for eid, seq in engines.items():
            T = len(seq)
            rul_labels = compute_rul_labels(T, rul_cap)  # (T,)

            if use_last_only:
                t = T
                rul_t = rul_labels[-1]  # RUL at last cycle
                past = torch.from_numpy(seq[:t])
                self.items.append((past, float(rul_t) / rul_cap))
            else:
                # Sample multiple cut points per engine
                cuts = sorted(rng.integers(10, T, size=min(n_cuts_per_engine, T - 10)).tolist())
                for t in cuts:
                    rul_t = rul_labels[t - 1]  # 0-indexed
                    past = torch.from_numpy(seq[:t])
                    self.items.append((past, float(rul_t) / rul_cap))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        past, rul_norm = self.items[idx]
        return past, torch.tensor(rul_norm, dtype=torch.float32)


class CMAPSSTestDataset(Dataset):
    """
    Test dataset - last window per test engine.
    If engine has fewer than window_length cycles, left-pad with zeros.
    """

    def __init__(self,
                 engines: Dict[int, np.ndarray],
                 test_rul: np.ndarray,
                 rul_cap: int = RUL_CAP,
                 window_length: int = 30):
        self.items = []
        eng_ids = sorted(engines.keys())
        for i, eid in enumerate(eng_ids):
            seq = engines[eid]
            T = len(seq)
            # Use full sequence as "past" (last-window evaluation)
            past = torch.from_numpy(seq)
            rul_gt = float(test_rul[i])  # ground truth RUL at last cycle
            # Normalize RUL to [0,1] for loss computation, but store raw for RMSE
            self.items.append((past, rul_gt))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        past, rul_gt = self.items[idx]
        return past, torch.tensor(rul_gt, dtype=torch.float32)


# ============================================================
# Collate functions for variable-length sequences
# ============================================================

def collate_pretrain(batch):
    """Pad variable-length past sequences for pretraining."""
    past_list, future_list, k_list, t_list = zip(*batch)
    # Pad past
    max_t = max(p.shape[0] for p in past_list)
    B = len(past_list)
    S = past_list[0].shape[1]
    past_padded = torch.zeros(B, max_t, S)
    past_mask = torch.zeros(B, max_t, dtype=torch.bool)  # True = padding
    for i, p in enumerate(past_list):
        T = p.shape[0]
        past_padded[i, :T] = p
        past_mask[i, T:] = True

    # Pad future (for target encoder)
    max_k = max(f.shape[0] for f in future_list)
    future_padded = torch.zeros(B, max_k, S)
    future_mask = torch.zeros(B, max_k, dtype=torch.bool)
    for i, f in enumerate(future_list):
        K = f.shape[0]
        future_padded[i, :K] = f
        future_mask[i, K:] = True

    k_tensor = torch.tensor(k_list, dtype=torch.long)
    t_tensor = torch.tensor(t_list, dtype=torch.long)
    return past_padded, past_mask, future_padded, future_mask, k_tensor, t_tensor


def collate_finetune(batch):
    """Pad variable-length sequences for finetuning."""
    past_list, rul_list = zip(*batch)
    max_t = max(p.shape[0] for p in past_list)
    B = len(past_list)
    S = past_list[0].shape[1]
    past_padded = torch.zeros(B, max_t, S)
    past_mask = torch.zeros(B, max_t, dtype=torch.bool)
    for i, p in enumerate(past_list):
        T = p.shape[0]
        past_padded[i, :T] = p
        past_mask[i, T:] = True
    rul_tensor = torch.stack(rul_list)
    return past_padded, past_mask, rul_tensor


def collate_test(batch):
    """Collate test items - same as finetune but rul_gt is raw (not normalized)."""
    past_list, rul_list = zip(*batch)
    max_t = max(p.shape[0] for p in past_list)
    B = len(past_list)
    S = past_list[0].shape[1]
    past_padded = torch.zeros(B, max_t, S)
    past_mask = torch.zeros(B, max_t, dtype=torch.bool)
    for i, p in enumerate(past_list):
        T = p.shape[0]
        past_padded[i, :T] = p
        past_mask[i, T:] = True
    rul_tensor = torch.stack(rul_list)
    return past_padded, past_mask, rul_tensor
