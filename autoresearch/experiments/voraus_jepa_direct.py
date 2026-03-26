#!/usr/bin/env python3
"""
Exp 52 (v3): Mechanical-JEPA on Voraus-AD — Direct parquet loading.

Bypasses FactoryNetDataset to avoid OOM: streams parquet files one at a time,
extracts windows per file, concatenates into a bounded numpy array.

Selects 10 anomaly files + 10 normal files (of 21+39 available) for a
balanced, memory-feasible subset of the Voraus-AD dataset.
"""

import sys
import time
import json
import copy
import gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import glob

PROJECT_ROOT = Path('/home/sagemaker-user/IndustrialJEPA')
sys.path.insert(0, str(PROJECT_ROOT))

import logging
logging.getLogger('industrialjepa').setLevel(logging.WARNING)
logging.getLogger('datasets').setLevel(logging.WARNING)
import warnings
warnings.filterwarnings('ignore')

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEEDS = [42, 123, 456]

JEPA_CONFIG = {
    'window_size': 64,
    'stride': 32,
    'mask_ratio': 0.30,
    'd_model': 64,
    'n_heads': 4,
    'n_layers': 3,
    'predictor_layers': 2,
    'ema_decay': 0.996,
    'lr': 3e-4,
    'weight_decay': 0.01,
    'epochs_pretrain': 30,
    'batch_size': 256,
}

# Parquet file directory (already cached from HF)
PARQUET_DIR = Path('/home/sagemaker-user/.cache/huggingface/hub/'
                   'datasets--Forgis--FactoryNet_Dataset/snapshots/'
                   '2a26a097cbefc3da1ff16a50b041f1870e145510/data/raw')

# Voraus setpoint and effort columns (from file inspection)
SETPOINT_COLS = [f'setpoint_pos_{i}' for i in range(6)] + [f'setpoint_vel_{i}' for i in range(6)]
EFFORT_COLS = [f'effort_current_{i}' for i in range(6)]


# ============================================================================
# Model (same as aursad_jepa_v2)
# ============================================================================

class PatchEmbed(nn.Module):
    def __init__(self, n_channels, patch_len, d_model):
        super().__init__()
        self.patch_len = patch_len
        self.proj = nn.Linear(n_channels * patch_len, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        B, T, C = x.shape
        T_trunc = (T // self.patch_len) * self.patch_len
        x = x[:, :T_trunc]
        n_patches = T_trunc // self.patch_len
        x = x.reshape(B, n_patches, self.patch_len * C)
        return self.norm(self.proj(x))


class TransformerEncoder(nn.Module):
    def __init__(self, d_model, n_heads, n_layers, max_seq=256, dropout=0.1):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.randn(1, max_seq, d_model) * 0.02)
        el = nn.TransformerEncoderLayer(d_model, n_heads, d_model*4, dropout,
                                         batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(el, n_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        B, T, D = x.shape
        x = x + self.pos_embed[:, :T]
        return self.norm(self.transformer(x))


class MechanicalJEPA(nn.Module):
    def __init__(self, n_channels, window_size, d_model=64, n_heads=4, n_layers=3,
                 predictor_layers=2, patch_len=8, mask_ratio=0.30, ema_decay=0.996):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.ema_decay = ema_decay
        self.d_model = d_model
        self.patch_len = patch_len
        self.n_patches = window_size // patch_len

        self.patch_embed = PatchEmbed(n_channels, patch_len, d_model)
        self.encoder = TransformerEncoder(d_model, n_heads, n_layers)
        self.target_encoder = copy.deepcopy(self.encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False

        self.mask_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        pred_layers = []
        for _ in range(predictor_layers):
            pred_layers.extend([nn.Linear(d_model, d_model), nn.LayerNorm(d_model), nn.GELU()])
        pred_layers.append(nn.Linear(d_model, d_model))
        self.predictor = nn.Sequential(*pred_layers)
        self.pred_pos_embed = nn.Parameter(torch.randn(1, self.n_patches, d_model) * 0.02)

    def create_block_mask(self, n_patches, mask_ratio):
        n_mask = max(1, int(n_patches * mask_ratio))
        start = torch.randint(0, n_patches - n_mask + 1, (1,)).item()
        mask = torch.zeros(n_patches, dtype=torch.bool)
        mask[start:start + n_mask] = True
        return mask

    def forward(self, x):
        B, T, C = x.shape
        patches = self.patch_embed(x)
        n_patches = patches.size(1)

        mask = self.create_block_mask(n_patches, self.mask_ratio)
        context_idx = (~mask).nonzero().squeeze(-1)
        masked_idx = mask.nonzero().squeeze(-1)

        with torch.no_grad():
            z_target = self.target_encoder(patches)

        context_patches = patches[:, context_idx]
        z_context = self.encoder(context_patches)

        n_mask = len(masked_idx)
        mask_tokens = self.mask_token.expand(B, n_mask, -1)
        mask_tokens = mask_tokens + self.pred_pos_embed[:, masked_idx]
        ctx_mean = z_context.mean(dim=1, keepdim=True).expand(-1, n_mask, -1)
        z_pred = self.predictor(ctx_mean + mask_tokens)

        z_target_masked = z_target[:, masked_idx]
        loss = F.mse_loss(z_pred, z_target_masked.detach())
        return loss

    @torch.no_grad()
    def encode(self, x):
        patches = self.patch_embed(x)
        z = self.encoder(patches)
        return z.mean(dim=1)

    def ema_update(self):
        with torch.no_grad():
            for p_enc, p_tgt in zip(self.encoder.parameters(),
                                     self.target_encoder.parameters()):
                p_tgt.data = self.ema_decay * p_tgt.data + (1 - self.ema_decay) * p_enc.data


# ============================================================================
# Direct Parquet Loading
# ============================================================================

def get_available_columns(parquet_path):
    """Get column list from one parquet file without loading all data."""
    df = pd.read_parquet(parquet_path, columns=['episode_id'])
    cols_df = pd.read_parquet(parquet_path).columns.tolist()
    del df
    gc.collect()
    return cols_df


def discover_columns(parquet_path):
    """Discover setpoint and effort columns from first parquet file."""
    df = pd.read_parquet(parquet_path)
    cols = list(df.columns)
    del df
    gc.collect()

    # Find available setpoint cols
    sp_cols = [c for c in cols if c.startswith('setpoint_pos_') or c.startswith('setpoint_vel_')]
    # Find available effort cols - try current first, then torque
    ef_cols = [c for c in cols if c.startswith('effort_current_')]
    if not ef_cols:
        ef_cols = [c for c in cols if c.startswith('effort_torque_')]
    if not ef_cols:
        ef_cols = [c for c in cols if c.startswith('effort_')]

    print(f"  Discovered setpoint cols ({len(sp_cols)}): {sp_cols[:4]}...")
    print(f"  Discovered effort cols ({len(ef_cols)}): {ef_cols}")
    return sorted(sp_cols), sorted(ef_cols)


def extract_windows_from_parquet(parquet_path, sp_cols, ef_cols, window_size=64, stride=32,
                                  max_windows_per_file=3000):
    """Load one parquet file and extract sliding windows.

    Returns:
        windows: np.array (N, T, C) float32
        labels: np.array (N,) int32 - 1=anomaly, 0=normal
        is_anomaly_file: bool
    """
    all_cols = sp_cols + ef_cols + ['episode_id', 'ctx_anomaly_label']
    available_cols = [c for c in all_cols]

    df = pd.read_parquet(parquet_path, columns=available_cols)

    # Convert float64 to float32 immediately
    for c in sp_cols + ef_cols:
        if c in df.columns:
            df[c] = df[c].astype(np.float32)
        else:
            df[c] = 0.0

    n_ch = len(sp_cols) + len(ef_cols)
    is_anomaly_file = 'Anomaly' in df['ctx_anomaly_label'].values if 'ctx_anomaly_label' in df.columns else False

    # Process per episode to get contiguous windows
    windows_list = []
    labels_list = []

    episode_ids = df['episode_id'].unique() if 'episode_id' in df.columns else ['all']
    np.random.shuffle(episode_ids)  # randomize episode order

    windows_so_far = 0
    for ep_id in episode_ids:
        if windows_so_far >= max_windows_per_file:
            break

        if 'episode_id' in df.columns:
            ep_df = df[df['episode_id'] == ep_id]
        else:
            ep_df = df

        # Episode-level normalization (subtract mean, divide by std)
        ep_data = ep_df[sp_cols + ef_cols].values.astype(np.float32)
        mu = ep_data.mean(axis=0, keepdims=True)
        sigma = ep_data.std(axis=0, keepdims=True) + 1e-6
        ep_data = (ep_data - mu) / sigma

        # Get episode label
        ep_label = 0
        if 'ctx_anomaly_label' in ep_df.columns:
            mode_label = ep_df['ctx_anomaly_label'].mode().iloc[0] if len(ep_df) > 0 else 'Normal'
            ep_label = 1 if 'Anomaly' in str(mode_label) else 0

        # Sliding window extraction
        T = len(ep_data)
        if T < window_size:
            continue

        for start in range(0, T - window_size + 1, stride):
            w = ep_data[start:start + window_size]
            windows_list.append(w)
            labels_list.append(ep_label)
            windows_so_far += 1
            if windows_so_far >= max_windows_per_file:
                break

    del df
    gc.collect()

    if not windows_list:
        return np.zeros((0, window_size, n_ch), dtype=np.float32), np.zeros(0, dtype=np.int32)

    windows = np.stack(windows_list, axis=0).astype(np.float32)
    labels = np.array(labels_list, dtype=np.int32)
    return windows, labels


def load_voraus_balanced(window_size=64, stride=32,
                          n_anomaly_files=10, n_normal_files=12):
    """Load Voraus-AD by streaming parquets one at a time.

    Selects n_anomaly_files from the anomaly set and n_normal_files from the
    normal set. Splits 80/20 train/test respecting file-level stratification.
    """
    all_parquets = sorted(glob.glob(str(PARQUET_DIR / 'voraus_*.parquet')))

    # Separate anomaly vs normal files (files 001-021 = anomaly, 022-060 = normal)
    anomaly_parquets = [f for f in all_parquets if int(Path(f).stem.split('_')[1]) <= 21]
    normal_parquets = [f for f in all_parquets if int(Path(f).stem.split('_')[1]) > 21]

    print(f"  Available: {len(anomaly_parquets)} anomaly files, {len(normal_parquets)} normal files")

    # Discover columns from first file
    sp_cols, ef_cols = discover_columns(anomaly_parquets[0])
    n_ch = len(sp_cols) + len(ef_cols)
    print(f"  Channels: {n_ch} ({len(sp_cols)} setpoint + {len(ef_cols)} effort)")

    # Select files (spread across the range for diversity)
    np.random.seed(42)
    sel_anomaly = np.random.choice(len(anomaly_parquets), min(n_anomaly_files, len(anomaly_parquets)), replace=False)
    sel_anomaly = [anomaly_parquets[i] for i in sorted(sel_anomaly)]

    sel_normal = np.random.choice(len(normal_parquets), min(n_normal_files, len(normal_parquets)), replace=False)
    sel_normal = [normal_parquets[i] for i in sorted(sel_normal)]

    print(f"  Selected: {len(sel_anomaly)} anomaly + {len(sel_normal)} normal files")

    # Split files into train/test (file-level split to avoid data leakage)
    n_anomaly_train = max(1, int(len(sel_anomaly) * 0.8))
    n_normal_train = max(1, int(len(sel_normal) * 0.8))

    train_files = sel_anomaly[:n_anomaly_train] + sel_normal[:n_normal_train]
    test_files = sel_anomaly[n_anomaly_train:] + sel_normal[n_normal_train:]

    print(f"  Train files: {len(train_files)}, Test files: {len(test_files)}")

    # Stream each file and collect windows
    def collect_windows(file_list, desc, max_per_file=3000):
        all_X, all_y = [], []
        for f in tqdm(file_list, desc=f"  Loading {desc}"):
            X, y = extract_windows_from_parquet(f, sp_cols, ef_cols, window_size, stride,
                                                 max_windows_per_file=max_per_file)
            if len(X) > 0:
                all_X.append(X)
                all_y.append(y)
        if not all_X:
            return np.zeros((0, window_size, n_ch), np.float32), np.zeros(0, np.int32)
        return np.concatenate(all_X, axis=0), np.concatenate(all_y, axis=0)

    train_X_np, train_y_np = collect_windows(train_files, "train")
    test_X_np, test_y_np = collect_windows(test_files, "test")

    print(f"  Train: {train_X_np.shape}, anomaly_rate={train_y_np.mean():.3f}")
    print(f"  Test:  {test_X_np.shape}, anomaly_rate={test_y_np.mean():.3f}")

    train_X = torch.FloatTensor(train_X_np)
    train_y = torch.LongTensor(train_y_np)
    test_X = torch.FloatTensor(test_X_np)
    test_y = torch.LongTensor(test_y_np)

    return train_X, train_y, test_X, test_y, n_ch


# ============================================================================
# JEPA Training and Eval (identical to aursad_jepa_v2)
# ============================================================================

def pretrain_jepa(model, train_X, config, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

    dataset = TensorDataset(train_X)
    loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True,
                        pin_memory=True, num_workers=0)

    optimizer = torch.optim.AdamW(
        [p for n, p in model.named_parameters() if 'target_encoder' not in n],
        lr=config['lr'], weight_decay=config['weight_decay']
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['epochs_pretrain'])

    losses = []
    for epoch in range(config['epochs_pretrain']):
        model.train()
        epoch_loss = 0
        n = 0
        for (batch_X,) in loader:
            batch_X = batch_X.to(DEVICE, non_blocking=True)
            optimizer.zero_grad()
            loss = model(batch_X)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            model.ema_update()
            epoch_loss += loss.item()
            n += 1

        epoch_loss /= max(n, 1)
        losses.append(epoch_loss)
        scheduler.step()

        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}/{config['epochs_pretrain']}: loss={epoch_loss:.4f}")

    return losses


def eval_linear_probe(model, train_X, train_y, test_X, test_y, batch_size=512):
    model.eval()

    def get_reps(X):
        reps = []
        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                bx = X[i:i+batch_size].to(DEVICE, non_blocking=True)
                z = model.encode(bx)
                reps.append(z.cpu().numpy())
        return np.vstack(reps)

    train_reps = get_reps(train_X)
    test_reps = get_reps(test_X)

    scaler = StandardScaler()
    train_reps_s = scaler.fit_transform(train_reps)
    test_reps_s = scaler.transform(test_reps)

    train_y_np = train_y.numpy()
    test_y_np = test_y.numpy()

    if len(np.unique(train_y_np)) < 2 or len(np.unique(test_y_np)) < 2:
        print("    Warning: single class in split, AUROC=0.5")
        return {'auroc': 0.5, 'f1': 0.0}

    clf = LogisticRegression(max_iter=500, C=1.0, class_weight='balanced')
    clf.fit(train_reps_s, train_y_np)

    probs = clf.predict_proba(test_reps_s)[:, 1]
    preds = clf.predict(test_reps_s)

    try:
        auroc = roc_auc_score(test_y_np, probs)
    except Exception:
        auroc = 0.5

    try:
        f1 = f1_score(test_y_np, preds, zero_division=0)
    except Exception:
        f1 = 0.0

    return {'auroc': float(auroc), 'f1': float(f1)}


# ============================================================================
# Main
# ============================================================================

def main():
    print(f"\n{'='*60}")
    print(f"EXP 52: Mechanical-JEPA on VORAUS-AD (Direct Parquet)")
    print(f"Device: {DEVICE}")
    print(f"{'='*60}")

    t0 = time.time()
    config = JEPA_CONFIG

    # Load data
    print("\n[Data] Streaming Voraus-AD parquets...")
    train_X, train_y, test_X, test_y, n_channels = load_voraus_balanced(
        window_size=config['window_size'],
        stride=config['stride'],
        n_anomaly_files=10,
        n_normal_files=12,
    )

    print(f"\nData ready in {time.time()-t0:.1f}s")
    print(f"  Train: {train_X.shape}, anomaly_rate={train_y.float().mean():.3f}")
    print(f"  Test:  {test_X.shape}, anomaly_rate={test_y.float().mean():.3f}")

    all_results = []

    for seed in SEEDS:
        print(f"\n  --- SEED {seed} ---")
        torch.manual_seed(seed)
        np.random.seed(seed)

        # JEPA pretrained
        model_jepa = MechanicalJEPA(
            n_channels=n_channels,
            window_size=config['window_size'],
            d_model=config['d_model'],
            n_heads=config['n_heads'],
            n_layers=config['n_layers'],
            predictor_layers=config['predictor_layers'],
            patch_len=8,
            mask_ratio=config['mask_ratio'],
            ema_decay=config['ema_decay'],
        ).to(DEVICE)

        n_params = sum(p.numel() for p in model_jepa.parameters())
        if seed == SEEDS[0]:
            print(f"  Model: {n_params:,} params, n_channels={n_channels}")

        t_pretrain = time.time()
        losses = pretrain_jepa(model_jepa, train_X, config, seed=seed)
        print(f"  Pretrain: {time.time()-t_pretrain:.1f}s, loss {losses[0]:.4f}->{losses[-1]:.4f}")

        probe_jepa = eval_linear_probe(model_jepa, train_X, train_y, test_X, test_y)
        print(f"  JEPA probe: AUROC={probe_jepa['auroc']:.4f}, F1={probe_jepa['f1']:.4f}")

        # Random init baseline
        model_random = MechanicalJEPA(
            n_channels=n_channels,
            window_size=config['window_size'],
            d_model=config['d_model'],
            n_heads=config['n_heads'],
            n_layers=config['n_layers'],
            predictor_layers=config['predictor_layers'],
            patch_len=8,
            mask_ratio=config['mask_ratio'],
            ema_decay=config['ema_decay'],
        ).to(DEVICE)

        probe_random = eval_linear_probe(model_random, train_X, train_y, test_X, test_y)
        print(f"  Random probe: AUROC={probe_random['auroc']:.4f}, F1={probe_random['f1']:.4f}")

        all_results.append({
            'seed': seed,
            'n_params': n_params,
            'n_channels': n_channels,
            'loss_start': float(losses[0]),
            'loss_end': float(losses[-1]),
            'jepa_auroc': float(probe_jepa['auroc']),
            'jepa_f1': float(probe_jepa['f1']),
            'random_auroc': float(probe_random['auroc']),
            'random_f1': float(probe_random['f1']),
            'delta': float(probe_jepa['auroc'] - probe_random['auroc']),
        })

    jepa_aurocs = [r['jepa_auroc'] for r in all_results]
    random_aurocs = [r['random_auroc'] for r in all_results]
    deltas = [r['delta'] for r in all_results]

    summary = {
        'dataset': 'voraus',
        'exp_num': 52,
        'n_channels': n_channels,
        'window_size': config['window_size'],
        'epochs_pretrain': config['epochs_pretrain'],
        'n_params': all_results[0]['n_params'],
        'n_anomaly_files': 10,
        'n_normal_files': 12,
        'n_train_windows': int(train_X.shape[0]),
        'n_test_windows': int(test_X.shape[0]),
        'jepa_auroc_mean': float(np.mean(jepa_aurocs)),
        'jepa_auroc_std': float(np.std(jepa_aurocs)),
        'random_auroc_mean': float(np.mean(random_aurocs)),
        'random_auroc_std': float(np.std(random_aurocs)),
        'delta_mean': float(np.mean(deltas)),
        'delta_std': float(np.std(deltas)),
        'verdict': 'JEPA_BETTER' if np.mean(deltas) > 0.01 else 'NO_BENEFIT',
        'total_time': float(time.time() - t0),
        'seeds': all_results,
    }

    print(f"\n{'='*60}")
    print(f"RESULTS: VORAUS-AD")
    print(f"{'='*60}")
    print(f"JEPA:   AUROC = {summary['jepa_auroc_mean']:.4f} ± {summary['jepa_auroc_std']:.4f}")
    print(f"Random: AUROC = {summary['random_auroc_mean']:.4f} ± {summary['random_auroc_std']:.4f}")
    print(f"Delta:  {summary['delta_mean']:+.4f} ± {summary['delta_std']:.4f}")
    print(f"Verdict: {summary['verdict']}")
    print(f"Time: {summary['total_time']:.1f}s")

    out_path = PROJECT_ROOT / "datasets" / "data" / "voraus_jepa_results.json"
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nVoraus results saved: {out_path}")

    return summary


if __name__ == '__main__':
    main()
