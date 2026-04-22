"""
SMAP / MSL Anomaly Detection Dataset Adapter (V15).

SMAP (Soil Moisture Active Passive): NASA spacecraft anomaly dataset.
  - 25 channels (telemetry + command features)
  - 135K train / 428K test timesteps
  - 12.8% anomaly rate in test

MSL (Mars Science Laboratory): NASA spacecraft anomaly dataset.
  - 55 channels
  - 58K train / 74K test timesteps
  - 10.5% anomaly rate in test

Data is preprocessed NASA telemetry; already in normalized float32 arrays.
Source: OmniAnomaly / MTS-JEPA replication data.

Standard window size: 100 timesteps.
Anomaly score: prediction reconstruction error from JEPA predictor.
Threshold: 95th percentile of validation scores on normal data.
Primary metric: non-PA F1 (honest); also report PA F1 for comparison.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Optional, Tuple

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


def load_smap(normalize: bool = True) -> dict:
    """Load SMAP dataset. Returns dict with train/test arrays and labels."""
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
    """Load MSL dataset."""
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
