"""
Phase 4: RUL Baselines — all methods on the same episode split.

Methods:
  1. Constant mean (trivial)
  2. Elapsed time only (linear regression)
  3. Handcrafted + MLP
  4. Handcrafted + LSTM
  5. Envelope RMS + LSTM
  6. JEPA + MLP (frozen encoder)
  7. JEPA + LSTM (main method)
  8. End-to-end CNN-LSTM
  9. CNN-GRU-MHA (published SOTA replica)
  10. Transformer (encoder-decoder)
  11. Linear elapsed-time (with elapsed_time_seconds)

Both RUL label conventions: linear and piecewise.
Cross-dataset transfer: FEMTO→XJTU-SY and XJTU-SY→FEMTO.
"""

import os
import sys
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
from scipy.stats import spearmanr

sys.path.insert(0, '/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/v8')
sys.path.insert(0, '/home/sagemaker-user/IndustrialJEPA/mechanical-jepa')

from data_pipeline import (
    load_rul_episodes, episode_train_test_split, compute_piecewise_rul,
    compute_envelope_rms_per_snapshot, compute_handcrafted_features_per_snapshot,
)
from jepa_v8 import MechanicalJEPAV8
from rul_model import (
    RULMLP, RULLSTM, HandcraftedMLP, HandcraftedLSTM,
    EnvelopeRMSLSTM, EndToEndCNNLSTM, CNNGRUMHAEncoder, TransformerRUL,
)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CHECKPOINT_DIR = '/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/v8/checkpoints'
RESULTS_DIR = '/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/v8/results'

SEEDS = [42, 123, 456]
N_EPOCHS_LSTM = 50
N_EPOCHS_MLP = 30
BATCH_SIZE = 32
LR = 1e-3


# ============================================================
# METRICS
# ============================================================

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """Compute all RUL metrics."""
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = float(1 - ss_res / (ss_tot + 1e-10))
    spearman, _ = spearmanr(y_true, y_pred)

    # PHM 2012 asymmetric score
    errors = y_pred - y_true
    phm_score = float(np.mean(np.where(errors <= 0,
                                        np.exp(-errors / 0.13) - 1,
                                        np.exp(errors / 0.10) - 1)))

    return {
        'rmse': rmse, 'mae': mae, 'r2': float(r2),
        'spearman': float(spearman) if not np.isnan(spearman) else 0.0,
        'phm_score': phm_score,
    }


def compute_monotonicity(rul_sequence: np.ndarray) -> float:
    """Monotonicity: fraction of consecutive pairs where RUL decreases (as expected)."""
    diffs = np.diff(rul_sequence)
    return float(np.mean(diffs <= 0))


# ============================================================
# EPISODE NORMALIZATION
# ============================================================

def normalize_elapsed_time(t: float, max_t: float) -> float:
    """Normalize elapsed time to [0, 1] using expected max lifetime."""
    return min(t / (max_t + 1e-10), 1.0)


def compute_max_lifetime(train_episodes: List[str],
                          episodes: Dict) -> float:
    """Max lifetime in train set (seconds). Used for normalization."""
    lifetimes = [episodes[ep][0]['lifetime_seconds'] for ep in train_episodes]
    return max(lifetimes)


# ============================================================
# DATASET CLASS
# ============================================================

class EpisodeDataset(Dataset):
    """Dataset that provides complete episodes for LSTM training."""
    def __init__(
        self,
        episode_ids: List[str],
        episodes: Dict,
        rul_labels: Dict,   # ep_id → list of float
        jepa_embeddings: Optional[Dict] = None,  # ep_id → (T, 256)
        handcrafted_feats: Optional[Dict] = None,  # ep_id → (T, 18)
        env_rms: Optional[Dict] = None,  # ep_id → (T,)
        max_lifetime: float = 7200.0,
        mode: str = 'jepa',  # 'jepa' | 'handcrafted' | 'env_rms' | 'raw'
        include_signals: bool = False,
    ):
        self.episode_ids = episode_ids
        self.episodes = episodes
        self.rul_labels = rul_labels
        self.jepa_embeddings = jepa_embeddings
        self.handcrafted_feats = handcrafted_feats
        self.env_rms = env_rms
        self.max_lifetime = max_lifetime
        self.mode = mode
        self.include_signals = include_signals

    def __len__(self):
        return len(self.episode_ids)

    def __getitem__(self, idx):
        ep_id = self.episode_ids[idx]
        snaps = self.episodes[ep_id]
        T = len(snaps)
        rul = self.rul_labels[ep_id]  # list of float

        # Elapsed time and delta_t
        elapsed = np.array([s['elapsed_time_seconds'] / self.max_lifetime
                             for s in snaps], dtype=np.float32)
        delta_t = np.array([s['delta_t'] / 3600.0  # normalize to hours
                             for s in snaps], dtype=np.float32)

        rul_arr = np.array(rul, dtype=np.float32)

        result = {
            'elapsed_time': torch.tensor(elapsed).unsqueeze(-1),  # (T, 1)
            'delta_t': torch.tensor(delta_t).unsqueeze(-1),        # (T, 1)
            'rul': torch.tensor(rul_arr).unsqueeze(-1),            # (T, 1)
            'episode_id': ep_id,
            'T': T,
        }

        if self.mode == 'jepa' and self.jepa_embeddings is not None:
            result['z'] = torch.tensor(self.jepa_embeddings[ep_id])  # (T, D)
        elif self.mode == 'handcrafted' and self.handcrafted_feats is not None:
            result['feats'] = torch.tensor(self.handcrafted_feats[ep_id])  # (T, 18)
        elif self.mode == 'env_rms' and self.env_rms is not None:
            result['env_rms'] = torch.tensor(self.env_rms[ep_id]).unsqueeze(-1)  # (T, 1)
        elif self.mode == 'raw':
            signals = np.stack([s['window'] for s in snaps])  # (T, 1024)
            result['signals'] = torch.tensor(signals).unsqueeze(1)  # (T, 1, 1024)

        return result


def collate_episodes(batch):
    """Collate episodes of variable length by padding."""
    max_T = max(item['T'] for item in batch)
    B = len(batch)

    result = {'T': [item['T'] for item in batch], 'episode_id': [item['episode_id'] for item in batch]}

    for key in ['elapsed_time', 'delta_t', 'rul']:
        tensors = []
        for item in batch:
            t = item[key]
            T = item['T']
            if T < max_T:
                # Pad with last value
                pad_size = max_T - T
                t = torch.cat([t, t[-1:].expand(pad_size, *t.shape[1:])], dim=0)
            tensors.append(t)
        result[key] = torch.stack(tensors)

    for key in ['z', 'feats', 'env_rms', 'signals']:
        if key in batch[0]:
            tensors = []
            for item in batch:
                t = item[key]
                T = item['T']
                if T < max_T:
                    pad_size = max_T - T
                    t = torch.cat([t, t[-1:].expand(pad_size, *t.shape[1:])], dim=0)
                tensors.append(t)
            result[key] = torch.stack(tensors)

    return result


# ============================================================
# PRECOMPUTE JEPA EMBEDDINGS
# ============================================================

def precompute_jepa_embeddings(
    episodes: Dict,
    encoder_path: str,
    batch_size: int = 64,
) -> Dict[str, np.ndarray]:
    """
    Run frozen JEPA encoder over all episode snapshots.
    Returns: Dict[ep_id → (T, 256) embeddings]
    """
    print("Precomputing JEPA embeddings...")
    ckpt = torch.load(encoder_path, map_location=DEVICE)
    model = MechanicalJEPAV8().to(DEVICE)
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    embeddings = {}
    all_eps = list(episodes.keys())

    for ep_id in all_eps:
        snaps = episodes[ep_id]
        signals = np.stack([s['window'] for s in snaps])  # (T, 1024)
        T = len(signals)

        ep_embeds = []
        for i in range(0, T, batch_size):
            batch = torch.tensor(signals[i:i+batch_size]).unsqueeze(1).to(DEVICE)  # (B, 1, 1024)
            with torch.no_grad():
                z = model.get_embeddings(batch)  # (B, 256)
            ep_embeds.append(z.cpu().numpy())
        embeddings[ep_id] = np.concatenate(ep_embeds, axis=0)  # (T, 256)

    print(f"  Embeddings computed for {len(embeddings)} episodes")
    return embeddings


def precompute_random_embeddings(episodes: Dict) -> Dict[str, np.ndarray]:
    """Random encoder baseline (untrained JEPA)."""
    torch.manual_seed(42)
    model = MechanicalJEPAV8().to(DEVICE)
    model.eval()
    embeddings = {}
    for ep_id, snaps in episodes.items():
        signals = np.stack([s['window'] for s in snaps])
        T = len(signals)
        ep_embeds = []
        for i in range(0, T, 64):
            batch = torch.tensor(signals[i:i+64]).unsqueeze(1).to(DEVICE)
            with torch.no_grad():
                z = model.get_embeddings(batch)
            ep_embeds.append(z.cpu().numpy())
        embeddings[ep_id] = np.concatenate(ep_embeds, axis=0)
    return embeddings


# ============================================================
# TRAINING UTILITIES
# ============================================================

def train_lstm_model(
    model: nn.Module,
    train_ids: List[str],
    episodes: Dict,
    rul_labels: Dict,
    mode: str,
    max_lifetime: float,
    seed: int = 42,
    n_epochs: int = N_EPOCHS_LSTM,
    jepa_embeddings: Optional[Dict] = None,
    handcrafted_feats: Optional[Dict] = None,
    env_rms: Optional[Dict] = None,
) -> nn.Module:
    """Train LSTM-style model on training episodes."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    dataset = EpisodeDataset(
        train_ids, episodes, rul_labels,
        jepa_embeddings=jepa_embeddings,
        handcrafted_feats=handcrafted_feats,
        env_rms=env_rms,
        max_lifetime=max_lifetime,
        mode=mode,
        include_signals=(mode == 'raw'),
    )
    loader = DataLoader(dataset, batch_size=min(BATCH_SIZE, len(train_ids)),
                         shuffle=True, collate_fn=collate_episodes)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    model.train()
    for epoch in range(n_epochs):
        epoch_loss = 0
        n_batches = 0
        for batch in loader:
            B = len(batch['T'])
            elapsed = batch['elapsed_time'].to(DEVICE)  # (B, T, 1)
            delta_t = batch['delta_t'].to(DEVICE)
            rul = batch['rul'].to(DEVICE)

            if mode == 'jepa':
                z = batch['z'].to(DEVICE)
                pred = model(z, delta_t, elapsed)
            elif mode == 'handcrafted':
                feats = batch['feats'].to(DEVICE)
                pred = model(feats, delta_t, elapsed)
            elif mode == 'env_rms':
                env = batch['env_rms'].to(DEVICE)
                pred = model(env, delta_t, elapsed)
            elif mode == 'raw':
                sigs = batch['signals'].to(DEVICE)
                pred = model(sigs, delta_t, elapsed)
            else:
                raise ValueError(f"Unknown mode: {mode}")

            # Mask padded positions
            T_lens = batch['T']
            max_T = rul.shape[1]
            mask = torch.zeros(B, max_T, 1, device=DEVICE)
            for i, t in enumerate(T_lens):
                mask[i, :t] = 1.0

            loss = (F.mse_loss(pred * mask, rul * mask, reduction='sum') /
                    (mask.sum() + 1e-10))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()

    return model


def train_mlp_model(
    model: nn.Module,
    train_ids: List[str],
    episodes: Dict,
    rul_labels: Dict,
    max_lifetime: float,
    seed: int = 42,
    n_epochs: int = N_EPOCHS_MLP,
    jepa_embeddings: Optional[Dict] = None,
    handcrafted_feats: Optional[Dict] = None,
) -> nn.Module:
    """Train MLP model (single-snapshot)."""
    torch.manual_seed(seed)

    # Build flat dataset
    X_list, E_list, Y_list = [], [], []
    for ep_id in train_ids:
        snaps = episodes[ep_id]
        rul = rul_labels[ep_id]
        T = len(snaps)
        for i, (s, r) in enumerate(zip(snaps, rul)):
            elapsed = s['elapsed_time_seconds'] / max_lifetime
            if jepa_embeddings is not None:
                z = jepa_embeddings[ep_id][i]
                X_list.append(z)
            elif handcrafted_feats is not None:
                X_list.append(handcrafted_feats[ep_id][i])
            E_list.append([elapsed])
            Y_list.append([r])

    X = torch.tensor(np.array(X_list, dtype=np.float32)).to(DEVICE)
    E = torch.tensor(np.array(E_list, dtype=np.float32)).to(DEVICE)
    Y = torch.tensor(np.array(Y_list, dtype=np.float32)).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    bs = BATCH_SIZE
    N = len(X)

    model.train()
    for epoch in range(n_epochs):
        perm = torch.randperm(N)
        for i in range(0, N, bs):
            idx = perm[i:i+bs]
            if jepa_embeddings is not None:
                pred = model(X[idx], E[idx])
            else:
                pred = model(X[idx], E[idx])
            loss = F.mse_loss(pred, Y[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return model


# ============================================================
# EVALUATION
# ============================================================

@torch.no_grad()
def evaluate_lstm_model(
    model: nn.Module,
    test_ids: List[str],
    episodes: Dict,
    rul_labels: Dict,
    mode: str,
    max_lifetime: float,
    jepa_embeddings: Optional[Dict] = None,
    handcrafted_feats: Optional[Dict] = None,
    env_rms: Optional[Dict] = None,
) -> Dict:
    """Evaluate LSTM model on test episodes."""
    model.eval()
    all_true, all_pred = [], []
    per_episode = {}

    for ep_id in test_ids:
        snaps = episodes[ep_id]
        T = len(snaps)
        rul = rul_labels[ep_id]

        elapsed = torch.tensor([[s['elapsed_time_seconds'] / max_lifetime]
                                  for s in snaps], dtype=torch.float32).unsqueeze(0).to(DEVICE)  # (1, T, 1)
        delta_t = torch.tensor([[s['delta_t'] / 3600.0]
                                  for s in snaps], dtype=torch.float32).unsqueeze(0).to(DEVICE)

        if mode == 'jepa':
            z = torch.tensor(jepa_embeddings[ep_id], dtype=torch.float32).unsqueeze(0).to(DEVICE)
            pred = model(z, delta_t, elapsed)
        elif mode == 'handcrafted':
            feats = torch.tensor(handcrafted_feats[ep_id], dtype=torch.float32).unsqueeze(0).to(DEVICE)
            pred = model(feats, delta_t, elapsed)
        elif mode == 'env_rms':
            env = torch.tensor(env_rms[ep_id], dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(DEVICE)
            pred = model(env, delta_t, elapsed)
        elif mode == 'raw':
            signals = torch.tensor(np.stack([s['window'] for s in snaps]),
                                    dtype=torch.float32).unsqueeze(0).unsqueeze(2).to(DEVICE)
            pred = model(signals, delta_t, elapsed)

        pred_np = pred.squeeze().cpu().numpy()
        true_np = np.array(rul, dtype=np.float32)

        if pred_np.ndim == 0:
            pred_np = pred_np.reshape(1)
        if true_np.ndim == 0:
            true_np = true_np.reshape(1)

        all_true.extend(true_np.tolist())
        all_pred.extend(pred_np.tolist())
        per_episode[ep_id] = compute_metrics(true_np, pred_np)
        per_episode[ep_id]['monotonicity'] = compute_monotonicity(pred_np)

    all_true = np.array(all_true)
    all_pred = np.array(all_pred)
    metrics = compute_metrics(all_true, all_pred)
    metrics['per_episode'] = per_episode
    return metrics


@torch.no_grad()
def evaluate_mlp_model(
    model: nn.Module,
    test_ids: List[str],
    episodes: Dict,
    rul_labels: Dict,
    max_lifetime: float,
    jepa_embeddings: Optional[Dict] = None,
    handcrafted_feats: Optional[Dict] = None,
) -> Dict:
    model.eval()
    all_true, all_pred = [], []
    per_episode = {}

    for ep_id in test_ids:
        snaps = episodes[ep_id]
        T = len(snaps)
        rul = rul_labels[ep_id]

        if jepa_embeddings is not None:
            X = torch.tensor(jepa_embeddings[ep_id], dtype=torch.float32).to(DEVICE)
        else:
            X = torch.tensor(handcrafted_feats[ep_id], dtype=torch.float32).to(DEVICE)

        elapsed = torch.tensor([[s['elapsed_time_seconds'] / max_lifetime]
                                  for s in snaps], dtype=torch.float32).to(DEVICE)
        pred = model(X, elapsed)
        pred_np = pred.squeeze(-1).cpu().numpy()
        true_np = np.array(rul, dtype=np.float32)

        all_true.extend(true_np.tolist())
        all_pred.extend(pred_np.tolist())
        per_episode[ep_id] = compute_metrics(true_np, pred_np)
        per_episode[ep_id]['monotonicity'] = compute_monotonicity(pred_np)

    return {**compute_metrics(np.array(all_true), np.array(all_pred)),
            'per_episode': per_episode}


# ============================================================
# BASELINES 1 & 2: TRIVIAL
# ============================================================

def baseline_constant_mean(test_ids, episodes, rul_labels):
    """Always predict mean RUL of training set."""
    all_true = []
    for ep_id in test_ids:
        all_true.extend(rul_labels[ep_id])
    mean_pred = np.mean(all_true)

    all_true_flat, all_pred_flat = [], []
    per_episode = {}
    for ep_id in test_ids:
        true_arr = np.array(rul_labels[ep_id])
        pred_arr = np.full_like(true_arr, mean_pred)
        all_true_flat.extend(true_arr.tolist())
        all_pred_flat.extend(pred_arr.tolist())
        per_episode[ep_id] = compute_metrics(true_arr, pred_arr)
    return {**compute_metrics(np.array(all_true_flat), np.array(all_pred_flat)),
            'per_episode': per_episode, 'mean_pred': mean_pred}


def baseline_elapsed_time_linear(train_ids, test_ids, episodes, rul_labels, max_lifetime):
    """Linear regression on elapsed_time_normalized → RUL%."""
    from sklearn.linear_model import LinearRegression

    # Build train data
    X_train, Y_train = [], []
    for ep_id in train_ids:
        snaps = episodes[ep_id]
        rul = rul_labels[ep_id]
        for s, r in zip(snaps, rul):
            t = s['elapsed_time_seconds'] / max_lifetime
            X_train.append([t])
            Y_train.append(r)

    lr_model = LinearRegression()
    lr_model.fit(np.array(X_train), np.array(Y_train))

    # Evaluate
    all_true_flat, all_pred_flat = [], []
    per_episode = {}
    for ep_id in test_ids:
        snaps = episodes[ep_id]
        rul = rul_labels[ep_id]
        X_test = [[s['elapsed_time_seconds'] / max_lifetime] for s in snaps]
        pred = lr_model.predict(np.array(X_test)).clip(0, 1)
        true_arr = np.array(rul)
        all_true_flat.extend(true_arr.tolist())
        all_pred_flat.extend(pred.tolist())
        per_episode[ep_id] = compute_metrics(true_arr, pred)
    return {**compute_metrics(np.array(all_true_flat), np.array(all_pred_flat)),
            'per_episode': per_episode}


# ============================================================
# MAIN: RUN ALL BASELINES
# ============================================================

def run_all_baselines(
    label_type: str = 'linear',  # 'linear' | 'piecewise'
    encoder_path: Optional[str] = None,
    seeds: List[int] = SEEDS,
) -> Dict:
    """
    Run all baselines. Returns results dict.
    """
    print(f"\n{'='*60}")
    print(f"Running baselines: label_type={label_type}")
    print(f"{'='*60}")

    # --- Load data ---
    print("\n--- Loading RUL episodes ---")
    episodes = load_rul_episodes(['femto', 'xjtu_sy'], verbose=True)

    # --- Train/test split (fixed across all methods and seeds) ---
    train_ids, test_ids = episode_train_test_split(episodes, test_ratio=0.25, seed=42)
    print(f"\nSplit: {len(train_ids)} train, {len(test_ids)} test")

    # --- Compute RUL labels ---
    if label_type == 'linear':
        rul_labels = {ep_id: [s['rul_percent'] for s in snaps]
                      for ep_id, snaps in episodes.items()}
    else:
        rul_labels = compute_piecewise_rul(episodes)

    # --- Max lifetime for normalization ---
    max_lifetime = compute_max_lifetime(train_ids, episodes)
    print(f"Max lifetime in train: {max_lifetime:.0f}s ({max_lifetime/3600:.2f}h)")

    # --- Precompute features ---
    print("\n--- Precomputing features ---")
    t0 = time.time()
    handcrafted_feats = {}
    env_rms_feats = {}
    for ep_id, snaps in episodes.items():
        handcrafted_feats[ep_id] = compute_handcrafted_features_per_snapshot(snaps)
        env_rms_feats[ep_id] = compute_envelope_rms_per_snapshot(snaps)
    print(f"  Handcrafted features: {time.time()-t0:.1f}s")

    # --- Precompute JEPA embeddings ---
    jepa_embeddings = None
    random_embeddings = None
    if encoder_path and os.path.exists(encoder_path):
        print("\n--- Precomputing JEPA embeddings ---")
        jepa_embeddings = precompute_jepa_embeddings(episodes, encoder_path)
        print("--- Precomputing random embeddings ---")
        random_embeddings = precompute_random_embeddings(episodes)
    else:
        print(f"\nWARNING: encoder not found at {encoder_path}, skipping JEPA methods")

    # Normalize env_rms per source (fit on train set)
    all_env_train = np.concatenate([env_rms_feats[ep] for ep in train_ids])
    env_mean = all_env_train.mean()
    env_std = all_env_train.std() + 1e-10
    for ep_id in env_rms_feats:
        env_rms_feats[ep_id] = (env_rms_feats[ep_id] - env_mean) / env_std

    # Normalize handcrafted features (fit on train set)
    all_feats_train = np.concatenate([handcrafted_feats[ep] for ep in train_ids])
    feat_mean = all_feats_train.mean(0)
    feat_std = all_feats_train.std(0) + 1e-10
    for ep_id in handcrafted_feats:
        handcrafted_feats[ep_id] = ((handcrafted_feats[ep_id] - feat_mean) / feat_std)

    # Normalize JEPA embeddings if available
    if jepa_embeddings:
        all_embeds_train = np.concatenate([jepa_embeddings[ep] for ep in train_ids])
        emb_mean = all_embeds_train.mean(0)
        emb_std = all_embeds_train.std(0) + 1e-10
        for ep_id in jepa_embeddings:
            jepa_embeddings[ep_id] = ((jepa_embeddings[ep_id] - emb_mean) / emb_std).astype(np.float32)

    # ============================================================
    # RUN BASELINES
    # ============================================================
    results = {}

    # --- Baseline 1: Constant mean ---
    print("\n[1/11] Constant mean...")
    # Use training labels to compute mean
    train_rul_labels = {ep: rul_labels[ep] for ep in train_ids}
    b1 = baseline_constant_mean(test_ids, episodes, rul_labels)
    results['constant_mean'] = {k: v for k, v in b1.items() if k != 'per_episode'}
    results['constant_mean']['per_episode'] = b1['per_episode']
    print(f"  RMSE: {b1['rmse']:.4f}")

    # --- Baseline 2: Elapsed time linear ---
    print("[2/11] Elapsed time linear regression...")
    b2 = baseline_elapsed_time_linear(train_ids, test_ids, episodes, rul_labels, max_lifetime)
    results['elapsed_time_linear'] = {k: v for k, v in b2.items() if k != 'per_episode'}
    results['elapsed_time_linear']['per_episode'] = b2['per_episode']
    print(f"  RMSE: {b2['rmse']:.4f}")
    time_only_rmse = b2['rmse']

    # --- Baseline 3: Handcrafted + MLP ---
    print("[3/11] Handcrafted + MLP...")
    b3_results = []
    for seed in seeds:
        m = HandcraftedMLP(n_features=18).to(DEVICE)
        m = train_mlp_model(m, train_ids, episodes, rul_labels, max_lifetime,
                             seed=seed, n_epochs=N_EPOCHS_MLP,
                             handcrafted_feats=handcrafted_feats)
        r = evaluate_mlp_model(m, test_ids, episodes, rul_labels, max_lifetime,
                                 handcrafted_feats=handcrafted_feats)
        b3_results.append(r)
    results['handcrafted_mlp'] = aggregate_seed_results(b3_results)
    print(f"  RMSE: {results['handcrafted_mlp']['rmse_mean']:.4f} ± {results['handcrafted_mlp']['rmse_std']:.4f}")

    # --- Baseline 4: Handcrafted + LSTM ---
    print("[4/11] Handcrafted + LSTM...")
    b4_results = []
    for seed in seeds:
        m = HandcraftedLSTM(n_features=18).to(DEVICE)
        m = train_lstm_model(m, train_ids, episodes, rul_labels, 'handcrafted',
                              max_lifetime, seed=seed, n_epochs=N_EPOCHS_LSTM,
                              handcrafted_feats=handcrafted_feats)
        r = evaluate_lstm_model(m, test_ids, episodes, rul_labels, 'handcrafted',
                                  max_lifetime, handcrafted_feats=handcrafted_feats)
        b4_results.append(r)
    results['handcrafted_lstm'] = aggregate_seed_results(b4_results)
    print(f"  RMSE: {results['handcrafted_lstm']['rmse_mean']:.4f} ± {results['handcrafted_lstm']['rmse_std']:.4f}")

    # --- Baseline 5: Envelope RMS + LSTM ---
    print("[5/11] Envelope RMS + LSTM...")
    b5_results = []
    for seed in seeds:
        m = EnvelopeRMSLSTM().to(DEVICE)
        m = train_lstm_model(m, train_ids, episodes, rul_labels, 'env_rms',
                              max_lifetime, seed=seed, n_epochs=N_EPOCHS_LSTM,
                              env_rms=env_rms_feats)
        r = evaluate_lstm_model(m, test_ids, episodes, rul_labels, 'env_rms',
                                  max_lifetime, env_rms=env_rms_feats)
        b5_results.append(r)
    results['envelope_rms_lstm'] = aggregate_seed_results(b5_results)
    print(f"  RMSE: {results['envelope_rms_lstm']['rmse_mean']:.4f} ± {results['envelope_rms_lstm']['rmse_std']:.4f}")

    # --- Baselines 6, 7: JEPA + MLP/LSTM ---
    if jepa_embeddings:
        print("[6/11] JEPA + MLP...")
        b6_results = []
        for seed in seeds:
            m = RULMLP(embed_dim=256).to(DEVICE)
            m = train_mlp_model(m, train_ids, episodes, rul_labels, max_lifetime,
                                 seed=seed, n_epochs=N_EPOCHS_MLP,
                                 jepa_embeddings=jepa_embeddings)
            r = evaluate_mlp_model(m, test_ids, episodes, rul_labels, max_lifetime,
                                     jepa_embeddings=jepa_embeddings)
            b6_results.append(r)
        results['jepa_mlp'] = aggregate_seed_results(b6_results)
        print(f"  RMSE: {results['jepa_mlp']['rmse_mean']:.4f} ± {results['jepa_mlp']['rmse_std']:.4f}")

        print("[7/11] JEPA + LSTM (main method)...")
        b7_results = []
        for seed in seeds:
            m = RULLSTM(embed_dim=256).to(DEVICE)
            m = train_lstm_model(m, train_ids, episodes, rul_labels, 'jepa',
                                  max_lifetime, seed=seed, n_epochs=N_EPOCHS_LSTM,
                                  jepa_embeddings=jepa_embeddings)
            r = evaluate_lstm_model(m, test_ids, episodes, rul_labels, 'jepa',
                                      max_lifetime, jepa_embeddings=jepa_embeddings)
            b7_results.append(r)
        results['jepa_lstm'] = aggregate_seed_results(b7_results)
        print(f"  RMSE: {results['jepa_lstm']['rmse_mean']:.4f} ± {results['jepa_lstm']['rmse_std']:.4f}")
    else:
        results['jepa_mlp'] = {'note': 'encoder not available'}
        results['jepa_lstm'] = {'note': 'encoder not available'}

    # --- Baseline 8: End-to-end CNN-LSTM ---
    print("[8/11] End-to-end CNN-LSTM...")
    b8_results = []
    for seed in seeds:
        m = EndToEndCNNLSTM().to(DEVICE)
        m = train_lstm_model(m, train_ids, episodes, rul_labels, 'raw',
                              max_lifetime, seed=seed, n_epochs=N_EPOCHS_LSTM)
        r = evaluate_lstm_model(m, test_ids, episodes, rul_labels, 'raw',
                                  max_lifetime)
        b8_results.append(r)
    results['cnn_lstm_e2e'] = aggregate_seed_results(b8_results)
    print(f"  RMSE: {results['cnn_lstm_e2e']['rmse_mean']:.4f} ± {results['cnn_lstm_e2e']['rmse_std']:.4f}")

    # --- Baseline 9: CNN-GRU-MHA (SOTA) ---
    print("[9/11] CNN-GRU-MHA (published SOTA)...")
    b9_results = []
    for seed in seeds:
        m = CNNGRUMHAEncoder().to(DEVICE)
        m = train_lstm_model(m, train_ids, episodes, rul_labels, 'raw',
                              max_lifetime, seed=seed, n_epochs=N_EPOCHS_LSTM)
        r = evaluate_lstm_model(m, test_ids, episodes, rul_labels, 'raw',
                                  max_lifetime)
        b9_results.append(r)
    results['cnn_gru_mha'] = aggregate_seed_results(b9_results)
    print(f"  RMSE: {results['cnn_gru_mha']['rmse_mean']:.4f} ± {results['cnn_gru_mha']['rmse_std']:.4f}")

    # --- Baseline 10: Transformer encoder-decoder ---
    print("[10/11] Transformer (handcrafted features)...")
    b10_results = []
    for seed in seeds:
        m = TransformerRUL(n_features=18).to(DEVICE)
        m = train_lstm_model(m, train_ids, episodes, rul_labels, 'handcrafted',
                              max_lifetime, seed=seed, n_epochs=N_EPOCHS_LSTM,
                              handcrafted_feats=handcrafted_feats)
        r = evaluate_lstm_model(m, test_ids, episodes, rul_labels, 'handcrafted',
                                  max_lifetime, handcrafted_feats=handcrafted_feats)
        b10_results.append(r)
    results['transformer_handcrafted'] = aggregate_seed_results(b10_results)
    print(f"  RMSE: {results['transformer_handcrafted']['rmse_mean']:.4f} ± {results['transformer_handcrafted']['rmse_std']:.4f}")

    # --- (Optional) Random JEPA embeddings to show training matters ---
    if random_embeddings:
        print("[11/11] Random JEPA + LSTM (ablation)...")
        b11_results = []
        for seed in seeds:
            m = RULLSTM(embed_dim=256).to(DEVICE)
            m = train_lstm_model(m, train_ids, episodes, rul_labels, 'jepa',
                                  max_lifetime, seed=seed, n_epochs=N_EPOCHS_LSTM,
                                  jepa_embeddings=random_embeddings)
            r = evaluate_lstm_model(m, test_ids, episodes, rul_labels, 'jepa',
                                      max_lifetime, jepa_embeddings=random_embeddings)
            b11_results.append(r)
        results['random_jepa_lstm'] = aggregate_seed_results(b11_results)
        print(f"  RMSE: {results['random_jepa_lstm']['rmse_mean']:.4f} ± {results['random_jepa_lstm']['rmse_std']:.4f}")

    # --- Summary ---
    print(f"\n{'='*60}")
    print(f"RESULTS SUMMARY (label_type={label_type})")
    print(f"{'='*60}")
    print(f"{'Method':<30} {'RMSE':>8} {'±':>4} {'MAE':>8} {'R²':>8} {'vs Time-Only':>12}")
    print('-' * 75)
    for method, r in results.items():
        if 'rmse_mean' in r:
            rmse = r['rmse_mean']
            std = r['rmse_std']
            mae = r.get('mae_mean', float('nan'))
            r2 = r.get('r2_mean', float('nan'))
            vs_base = (time_only_rmse - rmse) / time_only_rmse * 100
            print(f"{method:<30} {rmse:8.4f} {std:6.4f} {mae:8.4f} {r2:8.4f} {vs_base:+11.1f}%")
        elif 'rmse' in r:
            rmse = r['rmse']
            mae = r.get('mae', float('nan'))
            r2 = r.get('r2', float('nan'))
            vs_base = (time_only_rmse - rmse) / time_only_rmse * 100
            print(f"{method:<30} {rmse:8.4f} {'':6} {mae:8.4f} {r2:8.4f} {vs_base:+11.1f}%")
    print(f"Time-only RMSE: {time_only_rmse:.4f}")

    # Save results
    out = {
        'label_type': label_type,
        'time_only_rmse': time_only_rmse,
        'train_episodes': train_ids,
        'test_episodes': test_ids,
        'max_lifetime': max_lifetime,
        'results': {k: {kk: vv for kk, vv in v.items() if kk != 'per_episode'}
                    for k, v in results.items() if isinstance(v, dict)},
    }
    fname = os.path.join(RESULTS_DIR, f'rul_baselines_{label_type}.json')
    with open(fname, 'w') as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nResults saved: {fname}")

    return results


def aggregate_seed_results(seed_results: List[Dict]) -> Dict:
    """Aggregate metrics across seeds."""
    agg = {}
    metrics_to_agg = ['rmse', 'mae', 'r2', 'spearman', 'phm_score']
    for m in metrics_to_agg:
        vals = [r[m] for r in seed_results if m in r]
        if vals:
            agg[f'{m}_mean'] = float(np.mean(vals))
            agg[f'{m}_std'] = float(np.std(vals))
    # Keep per-episode from first seed
    if 'per_episode' in seed_results[0]:
        agg['per_episode'] = seed_results[0]['per_episode']
    return agg


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder', type=str,
                         default='/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/v8/checkpoints/jepa_v8_best.pt',
                         help='Path to trained JEPA checkpoint')
    parser.add_argument('--label-type', type=str, default='linear',
                         choices=['linear', 'piecewise'])
    args = parser.parse_args()

    results = run_all_baselines(
        label_type=args.label_type,
        encoder_path=args.encoder,
    )
