"""
Phase 2a: FD002 Condition Token

Keep 14 sensor channels with per-condition KMeans normalization.
Prepend a learnable condition embedding (6-way, looked up from KMeans
cluster ID) to each sequence. This lets the encoder know which operating
regime it's in without overloading the sensor channels.

Pretrain on FD002, fine-tune frozen + E2E at 100%, 5 seeds.
Target: frozen FD002 RMSE < 20 (vs current 26.33).

Output: experiments/v13/fd002_condition_token_results.json
"""

import sys
import json
import time
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from sklearn.cluster import KMeans

V11_DIR = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/experiments/v11')
V13_DIR = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/experiments/v13')
sys.path.insert(0, str(V11_DIR))

from data_utils import (
    load_raw, get_sensor_cols, get_op_cols, fit_normalizer,
    N_SENSORS, RUL_CAP, compute_rul_labels
)
from models import (
    TrajectoryJEPA, ContextEncoder, TargetEncoder, Predictor,
    RULProbe, sinusoidal_pe, trajectory_jepa_loss, SensorProjection,
    TransformerEncoder
)

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEEDS = [42, 123, 456, 789, 1024]
N_CONDITIONS = 6

print(f"Phase 2a: FD002 Condition Token")
print(f"Device: {DEVICE}")
t0_global = time.time()


# ============================================================
# Data preparation for FD002 with condition tokens
# ============================================================

def load_fd002_with_conditions():
    """Load FD002 with per-condition normalization and condition labels."""
    train_df, test_df, rul_arr = load_raw('FD002')
    sensor_cols = get_sensor_cols()
    op_cols = get_op_cols()

    # Fit KMeans on training op conditions
    op_train = train_df[op_cols].values.astype(np.float32)
    kmeans = KMeans(n_clusters=N_CONDITIONS, random_state=42, n_init=10)
    kmeans.fit(op_train)

    # Per-condition normalization
    train_df = train_df.copy()
    train_df['_cond'] = kmeans.predict(op_train)

    # Compute per-condition min-max stats
    cond_stats = {}
    for c in range(N_CONDITIONS):
        mask = train_df['_cond'] == c
        s = {}
        for col in sensor_cols:
            s[col] = (float(train_df.loc[mask, col].min()),
                      float(train_df.loc[mask, col].max()))
        cond_stats[c] = s

    def normalize_df(df, kmeans, cond_stats):
        """Normalize a dataframe using per-condition stats."""
        df = df.copy()
        op_vals = df[op_cols].values.astype(np.float32)
        conditions = kmeans.predict(op_vals)
        df['_cond'] = conditions

        sensor_vals = df[sensor_cols].values.astype(np.float32)
        normalized = np.zeros_like(sensor_vals)
        for i in range(len(sensor_vals)):
            c = conditions[i]
            s = cond_stats[c]
            for j, col in enumerate(sensor_cols):
                mn, mx = s[col]
                if mx > mn:
                    normalized[i, j] = (sensor_vals[i, j] - mn) / (mx - mn)
        return df, normalized, conditions

    train_df_n, train_sensors, train_conds = normalize_df(train_df, kmeans, cond_stats)
    test_df_n, test_sensors, test_conds = normalize_df(test_df, kmeans, cond_stats)

    # Build engine sequences with condition labels
    def build_sequences(df, sensors, conditions):
        seqs = {}
        conds_per_engine = {}
        for eid, grp in df.groupby('engine_id'):
            idx = grp.index.values
            seqs[int(eid)] = sensors[idx]
            conds_per_engine[int(eid)] = conditions[idx]
        return seqs, conds_per_engine

    train_seqs, train_cond_seqs = build_sequences(train_df_n, train_sensors, train_conds)
    test_seqs, test_cond_seqs = build_sequences(test_df_n, test_sensors, test_conds)

    # Train/val split
    all_ids = sorted(train_seqs.keys())
    rng = np.random.default_rng(42)
    n_val = max(1, int(0.15 * len(all_ids)))
    val_ids = set(rng.choice(all_ids, size=n_val, replace=False).tolist())
    train_ids = [i for i in all_ids if i not in val_ids]

    return {
        'train_engines': {i: train_seqs[i] for i in train_ids},
        'train_conds': {i: train_cond_seqs[i] for i in train_ids},
        'val_engines': {i: train_seqs[i] for i in val_ids},
        'val_conds': {i: train_cond_seqs[i] for i in val_ids},
        'test_engines': {i: test_seqs[i] for i in sorted(test_seqs.keys())},
        'test_conds': {i: test_cond_seqs[i] for i in sorted(test_cond_seqs.keys())},
        'test_rul': rul_arr.astype(np.float32),
        'kmeans': kmeans,
    }


# ============================================================
# Condition-aware encoder: prepend condition embedding
# ============================================================

class ConditionContextEncoder(nn.Module):
    """Context encoder with learnable condition token prepended."""

    def __init__(self, n_sensors=14, d_model=256, n_heads=4, n_layers=2,
                 d_ff=512, dropout=0.1, n_conditions=6):
        super().__init__()
        self.d_model = d_model
        self.proj = SensorProjection(n_sensors, 1, d_model)
        self.transformer = TransformerEncoder(d_model, n_heads, n_layers, d_ff, dropout)
        # Learnable condition embeddings
        self.condition_embed = nn.Embedding(n_conditions, d_model)

    def forward(self, x, conditions, key_padding_mask=None):
        """
        x: (B, T, S) sensor data
        conditions: (B,) condition IDs (int)
        returns: h_past (B, d_model)
        """
        B, T, S = x.shape
        tokens = self.proj(x)  # (B, T, d_model)

        # Add positional encoding
        positions = torch.arange(T, device=x.device)
        pe = sinusoidal_pe(positions, self.d_model)
        tokens = tokens + pe.unsqueeze(0)

        # Prepend condition token
        cond_token = self.condition_embed(conditions).unsqueeze(1)  # (B, 1, d_model)
        tokens = torch.cat([cond_token, tokens], dim=1)  # (B, T+1, d_model)

        # Adjust padding mask
        if key_padding_mask is not None:
            cond_pad = torch.zeros(B, 1, dtype=torch.bool, device=x.device)
            key_padding_mask = torch.cat([cond_pad, key_padding_mask], dim=1)

        out = self.transformer(tokens, key_padding_mask=key_padding_mask, causal=True)

        # Extract last non-padded position
        if key_padding_mask is not None:
            valid = (~key_padding_mask).long()
            last_idx = (valid * torch.arange(out.shape[1], device=x.device).unsqueeze(0)).argmax(dim=1)
            h_past = out[torch.arange(B, device=x.device), last_idx]
        else:
            h_past = out[:, -1]

        return h_past


# ============================================================
# Datasets
# ============================================================

class FD002PretrainDataset(Dataset):
    def __init__(self, engines, cond_seqs, n_cuts=20, min_past=10,
                 min_k=5, max_k=30, seed=42):
        self.items = []
        rng = np.random.default_rng(seed)
        for eid, seq in engines.items():
            conds = cond_seqs[eid]
            T = len(seq)
            for _ in range(n_cuts):
                k = int(rng.integers(min_k, max_k+1))
                t_max = T - k
                if min_past > t_max:
                    continue
                t = int(rng.integers(min_past, t_max+1))
                # Use mode condition of the past
                mode_cond = int(np.bincount(conds[:t].astype(int)).argmax())
                self.items.append((seq[:t], seq[t:t+k], k, t, mode_cond))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        past, future, k, t, cond = self.items[idx]
        return torch.from_numpy(past), torch.from_numpy(future), k, t, cond


class FD002FinetuneDataset(Dataset):
    def __init__(self, engines, cond_seqs, n_cuts=5, seed=42, use_last_only=False):
        self.items = []
        rng = np.random.default_rng(seed)
        for eid, seq in engines.items():
            conds = cond_seqs[eid]
            T = len(seq)
            rul = compute_rul_labels(T, RUL_CAP)
            if use_last_only:
                mode_cond = int(np.bincount(conds.astype(int)).argmax())
                self.items.append((torch.from_numpy(seq), float(rul[-1])/RUL_CAP, mode_cond))
            else:
                cuts = sorted(rng.integers(10, T, size=min(n_cuts, T-10)).tolist())
                for t in cuts:
                    mode_cond = int(np.bincount(conds[:t].astype(int)).argmax())
                    self.items.append((torch.from_numpy(seq[:t]), float(rul[t-1])/RUL_CAP, mode_cond))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        past, rul_norm, cond = self.items[idx]
        return past, torch.tensor(rul_norm, dtype=torch.float32), cond


class FD002TestDataset(Dataset):
    def __init__(self, engines, cond_seqs, test_rul):
        self.items = []
        eng_ids = sorted(engines.keys())
        for i, eid in enumerate(eng_ids):
            seq = engines[eid]
            conds = cond_seqs[eid]
            mode_cond = int(np.bincount(conds.astype(int)).argmax())
            self.items.append((torch.from_numpy(seq), float(test_rul[i]), mode_cond))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        past, rul_gt, cond = self.items[idx]
        return past, torch.tensor(rul_gt, dtype=torch.float32), cond


def collate_fd002(batch):
    if len(batch[0]) == 5:  # pretrain
        past_list, future_list, k_list, t_list, cond_list = zip(*batch)
        B = len(past_list)
        S = past_list[0].shape[1]
        max_t = max(p.shape[0] for p in past_list)
        past_padded = torch.zeros(B, max_t, S)
        past_mask = torch.zeros(B, max_t, dtype=torch.bool)
        for i, p in enumerate(past_list):
            T = p.shape[0]
            past_padded[i, :T] = p
            past_mask[i, T:] = True
        max_k = max(f.shape[0] for f in future_list)
        future_padded = torch.zeros(B, max_k, S)
        future_mask = torch.zeros(B, max_k, dtype=torch.bool)
        for i, f in enumerate(future_list):
            K = f.shape[0]
            future_padded[i, :K] = f
            future_mask[i, K:] = True
        k_tensor = torch.tensor(k_list, dtype=torch.long)
        cond_tensor = torch.tensor(cond_list, dtype=torch.long)
        return past_padded, past_mask, future_padded, future_mask, k_tensor, cond_tensor
    else:  # finetune/test (past, rul, cond)
        past_list, rul_list, cond_list = zip(*batch)
        B = len(past_list)
        S = past_list[0].shape[1]
        max_t = max(p.shape[0] for p in past_list)
        past_padded = torch.zeros(B, max_t, S)
        past_mask = torch.zeros(B, max_t, dtype=torch.bool)
        for i, p in enumerate(past_list):
            T = p.shape[0]
            past_padded[i, :T] = p
            past_mask[i, T:] = True
        rul_tensor = torch.stack(rul_list)
        cond_tensor = torch.tensor(cond_list, dtype=torch.long)
        return past_padded, past_mask, rul_tensor, cond_tensor


# ============================================================
# Training
# ============================================================

print("\nLoading FD002 data...")
data = load_fd002_with_conditions()
print(f"Train engines: {len(data['train_engines'])}, Val: {len(data['val_engines'])}, Test: {len(data['test_engines'])}")

# Pretrain
print(f"\n{'='*60}")
print("PRETRAINING condition-aware encoder on FD002")
print(f"{'='*60}")

encoder = ConditionContextEncoder(
    n_sensors=N_SENSORS, d_model=256, n_heads=4, n_layers=2,
    d_ff=512, dropout=0.1, n_conditions=N_CONDITIONS
).to(DEVICE)

# Simple predictor
predictor = Predictor(d_model=256, d_hidden=256).to(DEVICE)

# Target encoder (standard, no condition token)
from models import TargetEncoder
target_enc = TargetEncoder(
    n_sensors=N_SENSORS, patch_length=1, d_model=256, n_heads=4,
    n_layers=2, d_ff=512, dropout=0.1
).to(DEVICE)

# Initialize target from encoder (matching keys)
for p in target_enc.parameters():
    p.requires_grad = False

all_params = list(encoder.parameters()) + list(predictor.parameters())
optimizer = torch.optim.AdamW(all_params, lr=3e-4, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 200)

best_val_loss = float('inf')
best_state = None

for epoch in range(1, 201):
    train_ds = FD002PretrainDataset(
        data['train_engines'], data['train_conds'],
        n_cuts=20, seed=epoch
    )
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True,
                              collate_fn=collate_fd002)

    encoder.train()
    predictor.train()
    total_loss = 0
    n = 0

    for batch in train_loader:
        past, past_mask, future, future_mask, k, conds = [x.to(DEVICE) for x in batch]
        optimizer.zero_grad()

        h_past = encoder(past, conds, past_mask)

        with torch.no_grad():
            h_future = target_enc(future, future_mask)

        pred = predictor(h_past, k)
        loss, pred_l, var_l = trajectory_jepa_loss(pred, h_future)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(all_params, 1.0)
        optimizer.step()

        # EMA update target encoder from context encoder's transformer
        m = 0.996
        ctx_params = dict(encoder.transformer.named_parameters())
        tgt_params = dict(target_enc.transformer.named_parameters())
        with torch.no_grad():
            for key in tgt_params:
                if key in ctx_params:
                    tgt_params[key].data.mul_(m).add_(ctx_params[key].data, alpha=1-m)

        total_loss += loss.item() * past.shape[0]
        n += past.shape[0]

    scheduler.step()
    avg_loss = total_loss / n

    if epoch % 20 == 0 or epoch == 1:
        print(f"Ep {epoch:3d} | loss={avg_loss:.4f}")
        if avg_loss < best_val_loss:
            best_val_loss = avg_loss
            best_state = {
                'encoder': copy.deepcopy(encoder.state_dict()),
                'predictor': copy.deepcopy(predictor.state_dict()),
            }

if best_state is not None:
    encoder.load_state_dict(best_state['encoder'])
    print(f"\nPretraining done. Best loss: {best_val_loss:.4f}")

# ============================================================
# Fine-tune: frozen + E2E
# ============================================================
print(f"\n{'='*60}")
print("FINE-TUNING on FD002")
print(f"{'='*60}")

frozen_rmses = []
e2e_rmses = []

for seed in SEEDS:
    print(f"\n--- seed={seed} ---")

    # === Frozen ===
    enc_fr = copy.deepcopy(encoder).to(DEVICE)
    enc_fr.eval()
    for p in enc_fr.parameters():
        p.requires_grad = False
    probe_fr = RULProbe(256).to(DEVICE)
    opt_fr = torch.optim.Adam(probe_fr.parameters(), lr=1e-3)

    torch.manual_seed(seed)
    np.random.seed(seed)
    train_ds = FD002FinetuneDataset(data['train_engines'], data['train_conds'], seed=seed)
    val_ds = FD002FinetuneDataset(data['val_engines'], data['val_conds'], use_last_only=True)
    test_ds = FD002TestDataset(data['test_engines'], data['test_conds'], data['test_rul'])
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, collate_fn=collate_fd002)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, collate_fn=collate_fd002)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, collate_fn=collate_fd002)

    best_val, best_ps, ni = float('inf'), None, 0
    for ep in range(100):
        probe_fr.train()
        for past, mask, rul, conds in train_loader:
            past, mask, rul, conds = past.to(DEVICE), mask.to(DEVICE), rul.to(DEVICE), conds.to(DEVICE)
            opt_fr.zero_grad()
            with torch.no_grad():
                h = enc_fr(past, conds, mask)
            loss = F.mse_loss(probe_fr(h), rul)
            loss.backward()
            opt_fr.step()
        probe_fr.eval()
        pv, tv = [], []
        with torch.no_grad():
            for past, mask, rul, conds in val_loader:
                past, mask, conds = past.to(DEVICE), mask.to(DEVICE), conds.to(DEVICE)
                pv.append(probe_fr(enc_fr(past, conds, mask)).cpu().numpy())
                tv.append(rul.numpy())
        vr = float(np.sqrt(np.mean((np.concatenate(pv)*RUL_CAP - np.concatenate(tv)*RUL_CAP)**2)))
        if vr < best_val:
            best_val = vr; best_ps = copy.deepcopy(probe_fr.state_dict()); ni = 0
        else:
            ni += 1
            if ni >= 20: break

    probe_fr.load_state_dict(best_ps)
    probe_fr.eval()
    pt, tt = [], []
    with torch.no_grad():
        for past, mask, rul_gt, conds in test_loader:
            past, mask, conds = past.to(DEVICE), mask.to(DEVICE), conds.to(DEVICE)
            pt.append(probe_fr(enc_fr(past, conds, mask)).cpu().numpy() * RUL_CAP)
            tt.append(rul_gt.numpy())
    rmse_fr = float(np.sqrt(np.mean((np.concatenate(pt) - np.concatenate(tt))**2)))
    frozen_rmses.append(rmse_fr)
    print(f"  Frozen: {rmse_fr:.3f}")

    # === E2E ===
    enc_e2e = copy.deepcopy(encoder).to(DEVICE)
    for p in enc_e2e.parameters():
        p.requires_grad = True
    probe_e2e = RULProbe(256).to(DEVICE)
    opt_e2e = torch.optim.Adam(list(enc_e2e.parameters()) + list(probe_e2e.parameters()), lr=1e-4)

    torch.manual_seed(seed)
    np.random.seed(seed)
    best_val, best_ps, best_es, ni = float('inf'), None, None, 0
    for ep in range(100):
        enc_e2e.train(); probe_e2e.train()
        for past, mask, rul, conds in train_loader:
            past, mask, rul, conds = past.to(DEVICE), mask.to(DEVICE), rul.to(DEVICE), conds.to(DEVICE)
            opt_e2e.zero_grad()
            h = enc_e2e(past, conds, mask)
            loss = F.mse_loss(probe_e2e(h), rul)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(enc_e2e.parameters()) + list(probe_e2e.parameters()), 1.0)
            opt_e2e.step()
        enc_e2e.eval(); probe_e2e.eval()
        pv, tv = [], []
        with torch.no_grad():
            for past, mask, rul, conds in val_loader:
                past, mask, conds = past.to(DEVICE), mask.to(DEVICE), conds.to(DEVICE)
                pv.append(probe_e2e(enc_e2e(past, conds, mask)).cpu().numpy())
                tv.append(rul.numpy())
        vr = float(np.sqrt(np.mean((np.concatenate(pv)*RUL_CAP - np.concatenate(tv)*RUL_CAP)**2)))
        if vr < best_val:
            best_val = vr
            best_ps = copy.deepcopy(probe_e2e.state_dict())
            best_es = copy.deepcopy(enc_e2e.state_dict())
            ni = 0
        else:
            ni += 1
            if ni >= 20: break

    probe_e2e.load_state_dict(best_ps)
    enc_e2e.load_state_dict(best_es)
    enc_e2e.eval(); probe_e2e.eval()
    pt, tt = [], []
    with torch.no_grad():
        for past, mask, rul_gt, conds in test_loader:
            past, mask, conds = past.to(DEVICE), mask.to(DEVICE), conds.to(DEVICE)
            pt.append(probe_e2e(enc_e2e(past, conds, mask)).cpu().numpy() * RUL_CAP)
            tt.append(rul_gt.numpy())
    rmse_e2e = float(np.sqrt(np.mean((np.concatenate(pt) - np.concatenate(tt))**2)))
    e2e_rmses.append(rmse_e2e)
    print(f"  E2E:    {rmse_e2e:.3f}")

# Results
fr_mean = float(np.mean(frozen_rmses))
fr_std = float(np.std(frozen_rmses))
e2e_mean = float(np.mean(e2e_rmses))
e2e_std = float(np.std(e2e_rmses))

print(f"\n{'='*60}")
print(f"FD002 CONDITION TOKEN RESULTS")
print(f"{'='*60}")
print(f"Frozen: {fr_mean:.3f} +/- {fr_std:.3f} (target: < 20, baseline: 26.33)")
print(f"E2E:    {e2e_mean:.3f} +/- {e2e_std:.3f}")

results = {
    'frozen': {'mean': fr_mean, 'std': fr_std, 'all': [float(x) for x in frozen_rmses], 'baseline': 26.33, 'target': 20.0},
    'e2e': {'mean': e2e_mean, 'std': e2e_std, 'all': [float(x) for x in e2e_rmses]},
    'target_met': fr_mean < 20.0,
    'wall_time_s': float(time.time() - t0_global),
}

out_path = V13_DIR / 'fd002_condition_token_results.json'
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {out_path}")
print(f"Total wall time: {time.time()-t0_global:.1f}s")
