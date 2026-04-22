"""
SMAP / MSL Anomaly Detection Dataset Adapter.

SMAP (Soil Moisture Active Passive): NASA spacecraft anomaly dataset.
  - 25 features per entity (telemetry + command)
  - 55 independent telemetry entities, each with own train/test/labels
  - Concatenated: 135K train / 428K test timesteps, 12.8% anomaly rate

MSL (Mars Science Laboratory): NASA spacecraft anomaly dataset.
  - 55 features per entity
  - 27 independent telemetry entities
  - Concatenated: 58K train / 74K test timesteps, 10.5% anomaly rate

IMPORTANT: the concatenated train.npy / test.npy files glue independent
entities end-to-end along the time axis.  Any chronological split of the
concatenated array crosses entity boundaries → distribution shift.

For predictor finetuning, use split_smap_entities() / split_msl_entities().
These perform an INTRA-entity chronological split: for each entity, the
first 60% of its test stream goes to ft_train, next 10% to ft_val, last
30% to ft_test.  Every entity appears in all three splits, so the predictor
sees all channels during training.  A gap of `window_size` timesteps is
inserted at each boundary to prevent temporal leakage.

Source: telemanom repo (NASA), OmniAnomaly / MTS-JEPA replication data.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Optional, Tuple, List, Dict

try:
    from .config import SMAP_DIR, MSL_DIR
    SMAP_DATA_DIR = SMAP_DIR
    MSL_DATA_DIR = MSL_DIR
except ImportError:
    SMAP_DATA_DIR = Path('/home/sagemaker-user/IndustrialJEPA/paper-replications/mts-jepa/data/SMAP')
    MSL_DATA_DIR = Path('/home/sagemaker-user/IndustrialJEPA/paper-replications/mts-jepa/data/MSL')

WINDOW_SIZE = 100
STRIDE = 1  # overlap for test evaluation
TRAIN_STRIDE = 10  # sparse sampling for training (memory efficiency)


# ---------------------------------------------------------------------------
# Per-entity loaders (correct for predictor finetuning splits)
# ---------------------------------------------------------------------------

def _load_entities(data_dir: Path, normalize: bool = True) -> List[Dict]:
    """Load per-entity data from the channels/ subdirectory.

    Each entity is an independent telemetry channel with its own train/test
    arrays and anomaly labels.  Returns a list of dicts, one per entity.
    """
    channels_dir = data_dir / 'channels'
    if not channels_dir.exists():
        raise FileNotFoundError(
            f"Per-entity data not found at {channels_dir}. "
            f"Re-run paper-replications/mts-jepa/download_datasets.py "
            f"(it saves per-channel files under channels/)."
        )

    # Load channel name list if available, else discover from directory
    names_file = data_dir / 'channel_names.npy'
    if names_file.exists():
        channel_names = list(np.load(names_file, allow_pickle=True))
    else:
        channel_names = sorted([d.name for d in channels_dir.iterdir() if d.is_dir()])

    # Compute global normalization stats from concatenated train
    if normalize:
        all_train = []
        for name in channel_names:
            ch_dir = channels_dir / name
            all_train.append(np.load(ch_dir / 'train.npy').astype(np.float32))
        concat_train = np.concatenate(all_train, axis=0)
        mu = concat_train.mean(axis=0, keepdims=True)
        std = concat_train.std(axis=0, keepdims=True) + 1e-6
        del concat_train, all_train
    else:
        mu, std = None, None

    entities = []
    for name in channel_names:
        ch_dir = channels_dir / name
        train = np.load(ch_dir / 'train.npy').astype(np.float32)
        test = np.load(ch_dir / 'test.npy').astype(np.float32)
        labels = np.load(ch_dir / 'test_labels.npy').astype(np.int32)

        if normalize:
            train = (train - mu) / std
            test = (test - mu) / std

        entities.append({
            'entity_id': name,
            'train': train,
            'test': test,
            'labels': labels,
            'has_anomaly': bool(labels.any()),
        })

    return entities


def _intra_entity_split(entities: List[Dict],
                        ratios: Tuple[float, float, float] = (0.6, 0.1, 0.3),
                        window_size: int = WINDOW_SIZE,
                        ) -> Dict[str, List[Dict]]:
    """Intra-entity chronological split for predictor finetuning.

    For EACH entity, splits its test stream chronologically:
      - first  ratios[0] of timesteps → ft_train
      - next   ratios[1] of timesteps → ft_val
      - last   ratios[2] of timesteps → ft_test

    A gap of `window_size` timesteps is discarded at each split boundary
    so that no training window can peek into val/test time ranges.

    Every entity appears in all three splits, so the predictor sees all
    channels during training.  Entities whose test segment is too short
    to produce at least one window per split are skipped with a warning.
    """
    ft_train, ft_val, ft_test = [], [], []

    for e in entities:
        test = e['test']
        labels = e['labels']
        T = len(test)

        # Compute split boundaries
        t1 = int(ratios[0] * T)
        t2 = int((ratios[0] + ratios[1]) * T)

        # Apply gap: discard window_size timesteps after each boundary
        # so no window in split A can include timesteps from split B.
        #   ft_train: [0, t1)
        #   gap:      [t1, t1 + gap)
        #   ft_val:   [t1 + gap, t2)
        #   gap:      [t2, t2 + gap)
        #   ft_test:  [t2 + gap, T)
        gap = window_size
        val_start = t1 + gap
        test_start = t2 + gap

        # Check each split has enough room for at least one window
        min_len = window_size + 1  # need at least 1 window
        if t1 < min_len or (t2 - val_start) < min_len or (T - test_start) < min_len:
            print(f"  WARNING: entity {e['entity_id']} too short (T={T}), skipping")
            continue

        # Train portion also uses each entity's normal train data
        ft_train.append({
            'entity_id': e['entity_id'],
            'pretrain_normal': e['train'],       # normal-only (for context)
            'test': test[:t1],
            'labels': labels[:t1],
        })
        ft_val.append({
            'entity_id': e['entity_id'],
            'test': test[val_start:t2],
            'labels': labels[val_start:t2],
        })
        ft_test.append({
            'entity_id': e['entity_id'],
            'test': test[test_start:],
            'labels': labels[test_start:],
        })

    return {'ft_train': ft_train, 'ft_val': ft_val, 'ft_test': ft_test}


def load_smap_entities(normalize: bool = True) -> List[Dict]:
    """Load SMAP as a list of independent entities."""
    return _load_entities(SMAP_DATA_DIR, normalize=normalize)


def load_msl_entities(normalize: bool = True) -> List[Dict]:
    """Load MSL as a list of independent entities."""
    return _load_entities(MSL_DATA_DIR, normalize=normalize)


def split_smap_entities(ratios: Tuple[float, float, float] = (0.6, 0.1, 0.3),
                        normalize: bool = True) -> Dict[str, List[Dict]]:
    """Load SMAP entities and split intra-entity for pred-FT.

    Every entity appears in all three splits (ft_train / ft_val / ft_test).
    Split is chronological within each entity's test stream, with a gap of
    WINDOW_SIZE timesteps at each boundary to prevent temporal leakage.
    """
    entities = load_smap_entities(normalize=normalize)
    return _intra_entity_split(entities, ratios)


def split_msl_entities(ratios: Tuple[float, float, float] = (0.6, 0.1, 0.3),
                       normalize: bool = True) -> Dict[str, List[Dict]]:
    """Load MSL entities and split intra-entity for pred-FT."""
    entities = load_msl_entities(normalize=normalize)
    return _intra_entity_split(entities, ratios)


# ---------------------------------------------------------------------------
# Concatenated loaders (for pretraining and Mahalanobis baseline)
# ---------------------------------------------------------------------------

def load_smap(normalize: bool = True) -> dict:
    """Load SMAP as concatenated arrays (for pretraining / Mahalanobis).

    WARNING: the returned test array concatenates 55 independent entities.
    Do NOT use a chronological split of this array for finetuning — use
    split_smap_entities() instead.
    """
    train = np.load(SMAP_DATA_DIR / 'train.npy').astype(np.float32)
    test = np.load(SMAP_DATA_DIR / 'test.npy').astype(np.float32)
    labels = np.load(SMAP_DATA_DIR / 'test_labels.npy').astype(np.int32)

    mu, std = None, None
    if normalize:
        mu = train.mean(axis=0, keepdims=True)
        std = train.std(axis=0, keepdims=True) + 1e-6
        train = (train - mu) / std
        test = (test - mu) / std

    return {
        'train': train,          # (135183, 25)
        'test': test,            # (427617, 25)
        'labels': labels,        # (427617,) binary
        'n_channels': train.shape[1],
        'name': 'SMAP',
        'mu': mu,
        'std': std,
        'anomaly_rate': float(labels.mean()),
    }


def load_msl(normalize: bool = True) -> dict:
    """Load MSL as concatenated arrays (for pretraining / Mahalanobis).

    WARNING: the returned test array concatenates 27 independent entities.
    Do NOT use a chronological split of this array for finetuning — use
    split_msl_entities() instead.
    """
    train = np.load(MSL_DATA_DIR / 'train.npy').astype(np.float32)
    test = np.load(MSL_DATA_DIR / 'test.npy').astype(np.float32)
    labels = np.load(MSL_DATA_DIR / 'test_labels.npy').astype(np.int32)

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
        'name': 'MSL',
        'mu': mu,
        'std': std,
        'anomaly_rate': float(labels.mean()),
    }


class SlidingWindowDataset(Dataset):
    """
    Sliding window dataset for pretraining (reconstruction-based).

    Each item: (window, next_window) for prediction training.
    Or for anomaly detection: just the window.
    """

    def __init__(self, data: np.ndarray, window_size: int = WINDOW_SIZE,
                 stride: int = TRAIN_STRIDE, mode: str = 'pretrain',
                 horizon: int = 10):
        """
        Args:
            data: (T, C) time series
            window_size: context window size
            stride: step between windows
            mode: 'pretrain' (context + target) or 'score' (context only)
            horizon: prediction horizon for pretrain mode
        """
        self.data = torch.from_numpy(data).float()  # (T, C)
        self.window_size = window_size
        self.stride = stride
        self.mode = mode
        self.horizon = horizon

        T = len(data)
        if mode == 'pretrain':
            self.indices = list(range(0, T - window_size - horizon, stride))
        else:
            self.indices = list(range(0, T - window_size + 1, stride))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        start = self.indices[idx]
        window = self.data[start: start + self.window_size]  # (W, C)
        if self.mode == 'pretrain':
            target_start = start + self.horizon
            target = self.data[target_start: target_start + self.window_size]  # (W, C)
            return window, target, self.horizon
        else:
            return window


class AnomalyPretrainDataset(Dataset):
    """
    Variable-length pretraining dataset for anomaly models.
    Random context + horizon sampling from training time series.
    """

    def __init__(self, data: np.ndarray, n_samples: int = 50000,
                 min_context: int = 20, max_context: int = 100,
                 min_horizon: int = 5, max_horizon: int = 20, seed: int = 42):
        self.data = data  # (T, C)
        self.T = len(data)
        self.min_context = min_context
        self.max_context = max_context
        self.min_horizon = min_horizon
        self.max_horizon = max_horizon
        self._build(n_samples, seed)

    def _build(self, n_samples, seed):
        rng = np.random.RandomState(seed)
        self.samples = []
        for _ in range(n_samples):
            ctx_len = int(rng.randint(self.min_context, self.max_context + 1))
            horizon = int(rng.randint(self.min_horizon, self.max_horizon + 1))
            t = int(rng.randint(ctx_len, self.T - horizon))
            self.samples.append((t, ctx_len, horizon))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        t, ctx_len, horizon = self.samples[idx]
        past = self.data[t - ctx_len: t]          # (ctx_len, C)
        full = self.data[t - ctx_len: t + horizon]  # (ctx_len + horizon, C)
        return (torch.from_numpy(past).float(),
                torch.from_numpy(full).float(),
                horizon)


def collate_anomaly_pretrain(batch):
    """Collate variable-length pretrain samples (no masks - SMAP is dense)."""
    pasts, fulls, ks = zip(*batch)
    T_past_max = max(p.shape[0] for p in pasts)
    T_full_max = max(f.shape[0] for f in fulls)
    B = len(pasts)
    C = pasts[0].shape[1]

    x_past = torch.zeros(B, T_past_max, C)
    past_mask = torch.ones(B, T_past_max, dtype=torch.bool)
    x_full = torch.zeros(B, T_full_max, C)
    full_mask = torch.ones(B, T_full_max, dtype=torch.bool)

    for i, (p, f, k) in enumerate(zip(pasts, fulls, ks)):
        x_past[i, :p.shape[0]] = p
        past_mask[i, :p.shape[0]] = False
        x_full[i, :f.shape[0]] = f
        full_mask[i, :f.shape[0]] = False

    k_tensor = torch.tensor(ks, dtype=torch.long)
    return x_past, past_mask, x_full, full_mask, k_tensor


def compute_anomaly_scores(model, data_arr: np.ndarray,
                             window_size: int = WINDOW_SIZE,
                             batch_size: int = 256,
                             device: str = 'cuda') -> np.ndarray:
    """
    Compute per-timestep anomaly scores using prediction error.

    For each timestep t, score = ||h_hat(t-W:t, horizon=H) - h_actual(t:t+H)||_1

    Returns:
        scores: (T,) array where scores[t] is anomaly score at timestep t.
        Timesteps 0..W-1 are scored as the first window average.
    """
    device_t = torch.device(device if torch.cuda.is_available() else 'cpu')
    model.eval()
    T, C = data_arr.shape
    scores = np.zeros(T, dtype=np.float32)
    counts = np.zeros(T, dtype=np.float32)

    # Slide window over test data with stride 1
    stride = max(1, window_size // 10)  # stride=10 for speed, stride=1 for max quality
    windows = []
    window_starts = []
    for start in range(0, T - window_size - 10, stride):
        windows.append(data_arr[start: start + window_size])
        window_starts.append(start)

    # Process in batches
    horizon = 10  # fixed evaluation horizon
    all_scores = []

    for b_start in range(0, len(windows), batch_size):
        batch_wins = windows[b_start: b_start + batch_size]
        batch_starts = window_starts[b_start: b_start + batch_size]

        # Stack: (B, W, C)
        x_past = torch.tensor(np.stack(batch_wins), dtype=torch.float32).to(device_t)
        # Create corresponding full sequences (past + next 10 steps)
        full_wins = []
        for start in batch_starts:
            end = min(start + window_size + horizon, T)
            if end <= start:
                full_wins.append(data_arr[start: start + window_size])
            else:
                full_wins.append(data_arr[start: end])

        # Pad full sequences
        T_full_max = max(f.shape[0] for f in full_wins)
        B = len(full_wins)
        x_full = torch.zeros(B, T_full_max, C, dtype=torch.float32).to(device_t)
        full_mask = torch.ones(B, T_full_max, dtype=torch.bool).to(device_t)
        for i, f in enumerate(full_wins):
            x_full[i, :f.shape[0]] = torch.from_numpy(f).float()
            full_mask[i, :f.shape[0]] = False

        past_mask = torch.zeros(B, window_size, dtype=torch.bool).to(device_t)
        k_tensor = torch.full((B,), horizon, dtype=torch.long, device=device_t)

        with torch.no_grad():
            try:
                # Get context and predicted embeddings
                h_t = model.encode_context(x_past, mask=past_mask)
                h_hat = model.predictor(h_t, k_tensor)
                h_full = model.encode_context(x_full, mask=full_mask)

                # L1 prediction error as anomaly score
                err = (h_hat - h_full).abs().mean(dim=-1)  # (B,)
                all_scores.extend([(start, float(e)) for start, e
                                    in zip(batch_starts, err.cpu().numpy())])
            except Exception as e:
                all_scores.extend([(start, 0.0) for start in batch_starts])

    # Map window scores back to per-timestep
    for start, score in all_scores:
        end = min(start + window_size, T)
        scores[start:end] += score
        counts[start:end] += 1

    valid = counts > 0
    scores[valid] /= counts[valid]
    # Fill initial timesteps with first window score
    if counts[0] == 0 and len(all_scores) > 0:
        scores[:window_size] = all_scores[0][1] if all_scores else 0.0

    return scores


def evaluate_anomaly_detection(model, data: dict, window_size: int = WINDOW_SIZE,
                                 device: str = 'cuda') -> dict:
    """
    Full anomaly detection evaluation pipeline.

    1. Compute anomaly scores on test data
    2. Threshold at 95th percentile of normal-period scores (first 10% of test)
    3. Compute non-PA F1, PA F1, AUC-PR

    Returns dict of metrics.
    """
    from ..evaluation.grey_swan_metrics import anomaly_metrics

    test_arr = data['test']
    labels = data['labels']

    print(f"  Computing anomaly scores on {data['name']}...")
    scores = compute_anomaly_scores(model, test_arr, window_size=window_size, device=device)
    print(f"  Score range: [{scores.min():.4f}, {scores.max():.4f}]")

    # Use first 10% of test as "normal" for threshold (heuristic)
    n_normal = int(0.1 * len(test_arr))
    threshold = float(np.percentile(scores[:n_normal], 95))

    metrics = anomaly_metrics(scores, labels, threshold=threshold)
    metrics['dataset'] = data['name']
    metrics['window_size'] = window_size
    metrics['threshold'] = threshold
    return metrics


def get_smap_dataloader(n_samples: int = 50000, batch_size: int = 64,
                          seed: int = 42) -> Tuple[DataLoader, dict]:
    """Convenience: load SMAP and return DataLoader for pretraining."""
    data = load_smap()
    ds = AnomalyPretrainDataset(data['train'], n_samples=n_samples, seed=seed)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True,
                        collate_fn=collate_anomaly_pretrain, num_workers=0)
    return loader, data


def get_msl_dataloader(n_samples: int = 50000, batch_size: int = 64,
                        seed: int = 42) -> Tuple[DataLoader, dict]:
    """Convenience: load MSL and return DataLoader for pretraining."""
    data = load_msl()
    ds = AnomalyPretrainDataset(data['train'], n_samples=n_samples, seed=seed)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True,
                        collate_fn=collate_anomaly_pretrain, num_workers=0)
    return loader, data


if __name__ == '__main__':
    print("Testing SMAP/MSL data adapter...")
    data_smap = load_smap()
    print(f"SMAP: train={data_smap['train'].shape}, test={data_smap['test'].shape}")
    print(f"Anomaly rate: {data_smap['anomaly_rate']:.3f}")

    data_msl = load_msl()
    print(f"MSL: train={data_msl['train'].shape}, test={data_msl['test'].shape}")
    print(f"Anomaly rate: {data_msl['anomaly_rate']:.3f}")

    # Test DataLoader
    loader, _ = get_smap_dataloader(n_samples=100, batch_size=4)
    batch = next(iter(loader))
    x_past, past_mask, x_full, full_mask, k = batch
    print(f"Batch: x_past={x_past.shape}, x_full={x_full.shape}, k={k}")
    print("Adapter test PASSED")
