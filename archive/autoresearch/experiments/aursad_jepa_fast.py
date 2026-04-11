#!/usr/bin/env python3
"""
Exp 51 (Fast): Mechanical-JEPA on AURSAD — Efficient Implementation

Uses DataLoader directly instead of per-sample iteration to avoid
the O(N_windows) Python overhead.

Key change: Use torch DataLoader directly with FactoryNetDataset,
collate batches efficiently rather than materializing all windows upfront.
"""

import sys
import time
import json
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from industrialjepa.data.factorynet import FactoryNetDataset, FactoryNetConfig

import logging
logging.getLogger('industrialjepa').setLevel(logging.WARNING)
logging.getLogger('datasets').setLevel(logging.WARNING)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEEDS = [42, 123, 456]

JEPA_CONFIG = {
    'window_size': 64,
    'mask_ratio': 0.30,
    'd_model': 64,
    'n_heads': 4,
    'n_layers': 3,
    'predictor_layers': 2,
    'ema_decay': 0.996,
    'lr': 3e-4,
    'weight_decay': 0.01,
    'epochs_pretrain': 30,
    'batch_size': 128,
}


# ============================================================================
# Efficient Model (same as aursad_jepa.py)
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
        predictor_list = []
        for i in range(predictor_layers):
            predictor_list.extend([nn.Linear(d_model, d_model), nn.LayerNorm(d_model), nn.GELU()])
        predictor_list.append(nn.Linear(d_model, d_model))
        self.predictor = nn.Sequential(*predictor_list)
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
        pred_input = ctx_mean + mask_tokens
        z_pred = self.predictor(pred_input)

        z_target_masked = z_target[:, masked_idx]
        loss = F.mse_loss(z_pred, z_target_masked.detach())
        return loss, z_context

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
# Efficient Data Loading
# ============================================================================

def load_factorynet(data_source, window_size=64, batch_size=128):
    """Load FactoryNet dataset, return DataLoaders (no materialization)."""
    print(f"[Data] Loading {data_source} from HuggingFace...")

    config = FactoryNetConfig(
        dataset_name="Forgis/FactoryNet_Dataset",
        config_name="normalized",
        data_source=data_source,
        window_size=window_size,
        stride=window_size // 2,
        normalize=True,
        norm_mode="episode",
        max_episodes=None,
        train_healthy_only=False,
    )

    train_ds = FactoryNetDataset(config, split='train')
    shared = train_ds.get_shared_data()
    val_ds = FactoryNetDataset(config, split='val', shared_data=shared)
    test_ds = FactoryNetDataset(config, split='test', shared_data=shared)

    # Determine channel count from first sample
    sp, ef, meta = train_ds[0]
    n_channels = sp.shape[1] + ef.shape[1]

    print(f"  Train: {len(train_ds)} windows, Val: {len(val_ds)}, Test: {len(test_ds)}")
    print(f"  Channels: {n_channels} (setpoint={sp.shape[1]}, effort={ef.shape[1]})")

    def collate_fn(batch):
        sps, efs, metas = zip(*batch)
        x = torch.cat([torch.stack(sps), torch.stack(efs)], dim=-1)
        labels = torch.tensor([1 if m['is_anomaly'] else 0 for m in metas])
        return x, labels

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            collate_fn=collate_fn, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             collate_fn=collate_fn, num_workers=0)

    return train_loader, val_loader, test_loader, n_channels


# ============================================================================
# Training and Evaluation
# ============================================================================

def pretrain_jepa(model, train_loader, config, seed=42):
    """Pretrain JEPA using DataLoader (efficient, no materialization)."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    optimizer = torch.optim.AdamW(
        list(model.encoder.parameters()) +
        list(model.predictor.parameters()) +
        list(model.patch_embed.parameters()),
        lr=config['lr'], weight_decay=config['weight_decay']
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['epochs_pretrain'])

    losses = []
    for epoch in range(config['epochs_pretrain']):
        model.train()
        epoch_loss = 0
        n = 0
        for batch_X, _ in train_loader:
            batch_X = batch_X.to(DEVICE)
            optimizer.zero_grad()
            loss, _ = model(batch_X)
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


def get_representations(model, loader):
    """Extract representations and labels from a DataLoader."""
    model.eval()
    reps, labels = [], []
    with torch.no_grad():
        for batch_X, batch_y in loader:
            batch_X = batch_X.to(DEVICE)
            z = model.encode(batch_X)
            reps.append(z.cpu().numpy())
            labels.append(batch_y.numpy())
    return np.vstack(reps), np.concatenate(labels)


def eval_linear_probe(model, train_loader, test_loader):
    """Linear probe: anomaly detection AUROC."""
    train_reps, train_y = get_representations(model, train_loader)
    test_reps, test_y = get_representations(model, test_loader)

    scaler = StandardScaler()
    train_reps_s = scaler.fit_transform(train_reps)
    test_reps_s = scaler.transform(test_reps)

    if len(np.unique(train_y)) < 2:
        return {'auroc': 0.5, 'f1': 0.0}

    clf = LogisticRegression(max_iter=500, C=1.0, class_weight='balanced')
    clf.fit(train_reps_s, train_y)

    probs = clf.predict_proba(test_reps_s)[:, 1]
    preds = clf.predict(test_reps_s)

    try:
        auroc = roc_auc_score(test_y, probs)
    except Exception:
        auroc = 0.5

    try:
        f1 = f1_score(test_y, preds, zero_division=0)
    except Exception:
        f1 = 0.0

    return {'auroc': auroc, 'f1': f1}


# ============================================================================
# Main
# ============================================================================

def run_dataset(data_source, exp_num):
    """Run JEPA pretraining + evaluation on one dataset."""
    print(f"\n{'='*60}")
    print(f"EXP {exp_num}: Mechanical-JEPA on {data_source}")
    print(f"Device: {DEVICE}")
    print(f"{'='*60}")

    t0 = time.time()
    config = JEPA_CONFIG

    train_loader, val_loader, test_loader, n_channels = load_factorynet(
        data_source=data_source,
        window_size=config['window_size'],
        batch_size=config['batch_size'],
    )

    print(f"\nData loaded in {time.time()-t0:.1f}s")

    all_results = []

    for seed in SEEDS:
        print(f"\n  SEED {seed}")
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
        print(f"  Model: {n_params:,} params")

        t_pretrain = time.time()
        losses = pretrain_jepa(model_jepa, train_loader, config, seed=seed)
        print(f"  Pretrain done: {time.time()-t_pretrain:.1f}s, loss {losses[0]:.4f}->{losses[-1]:.4f}")

        probe_jepa = eval_linear_probe(model_jepa, train_loader, test_loader)
        print(f"  JEPA: AUROC={probe_jepa['auroc']:.4f}, F1={probe_jepa['f1']:.4f}")

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

        probe_random = eval_linear_probe(model_random, train_loader, test_loader)
        print(f"  Random: AUROC={probe_random['auroc']:.4f}, F1={probe_random['f1']:.4f}")

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

    # Aggregate
    jepa_aurocs = [r['jepa_auroc'] for r in all_results]
    random_aurocs = [r['random_auroc'] for r in all_results]
    deltas = [r['delta'] for r in all_results]

    summary = {
        'dataset': data_source,
        'exp_num': exp_num,
        'n_channels': n_channels,
        'window_size': config['window_size'],
        'epochs_pretrain': config['epochs_pretrain'],
        'n_params': all_results[0]['n_params'],
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
    print(f"RESULTS: {data_source}")
    print(f"{'='*60}")
    print(f"JEPA:   AUROC = {summary['jepa_auroc_mean']:.4f} ± {summary['jepa_auroc_std']:.4f}")
    print(f"Random: AUROC = {summary['random_auroc_mean']:.4f} ± {summary['random_auroc_std']:.4f}")
    print(f"Delta:  {summary['delta_mean']:+.4f} ± {summary['delta_std']:.4f}")
    print(f"Verdict: {summary['verdict']}")
    print(f"Time: {summary['total_time']:.1f}s")

    return summary


def main():
    t0 = time.time()

    # Run AURSAD
    aursad_summary = run_dataset('aursad', exp_num=51)
    aursad_out = PROJECT_ROOT / "datasets" / "data" / "aursad_jepa_results.json"
    with open(aursad_out, 'w') as f:
        json.dump(aursad_summary, f, indent=2)
    print(f"AURSAD results saved to {aursad_out}")

    # Run Voraus-AD
    voraus_summary = run_dataset('voraus', exp_num=52)
    voraus_out = PROJECT_ROOT / "datasets" / "data" / "voraus_jepa_results.json"
    with open(voraus_out, 'w') as f:
        json.dump(voraus_summary, f, indent=2)
    print(f"Voraus results saved to {voraus_out}")

    # Cross-dataset comparison
    print(f"\n{'='*60}")
    print("CROSS-DATASET JEPA COMPARISON")
    print(f"{'='*60}")
    print(f"{'Dataset':<15} {'JEPA AUROC':>12} {'Random AUROC':>14} {'Delta':>8} {'Verdict':<15}")
    print("-" * 65)
    for s in [aursad_summary, voraus_summary]:
        print(f"{s['dataset']:<15} {s['jepa_auroc_mean']:>12.4f} "
              f"{s['random_auroc_mean']:>14.4f} "
              f"{s['delta_mean']:>+8.4f} "
              f"{s['verdict']:<15}")

    print(f"\nTotal time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
