#!/usr/bin/env python3
"""
Exp 51: Mechanical-JEPA Pretraining on AURSAD

JEPA self-supervised pretraining on AURSAD screwdriving data.
Architecture:
  x_visible → Encoder → z_visible
  z_visible → Predictor(mask_tokens) → z_hat_masked
  x_full → Target_Encoder(EMA) → z_target
  Loss = ||z_hat_masked - z_target_masked||^2  (in latent space, NOT reconstruction)

Then evaluate:
  1. Linear probe: anomaly detection (normal vs anomaly)
  2. Forecasting: next-step prediction fine-tuned from JEPA encoder
  3. Comparison: JEPA pretrained vs random init vs from-scratch

Dataset: AURSAD (UR3e screwdriving, 6 DOF current signals, 4094 episodes)
Loaded via FactoryNetDataset (HuggingFace Forgis/FactoryNet_Dataset)
"""

import sys
import time
import json
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from industrialjepa.data.factorynet import FactoryNetDataset, FactoryNetConfig

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEEDS = [42, 123, 456]

# JEPA config
JEPA_CONFIG = {
    'window_size': 64,
    'mask_ratio': 0.30,   # 30% of timesteps masked
    'context_ratio': 0.70,
    'd_model': 64,
    'n_heads': 4,
    'n_layers': 3,
    'predictor_layers': 2,
    'ema_decay': 0.996,
    'lr': 3e-4,
    'weight_decay': 0.01,
    'epochs_pretrain': 30,
    'batch_size': 64,
}

# ============================================================================
# Model
# ============================================================================

class PatchEmbed(nn.Module):
    """Patch embedding for multivariate time series."""
    def __init__(self, n_channels, patch_len, d_model):
        super().__init__()
        self.patch_len = patch_len
        self.proj = nn.Linear(n_channels * patch_len, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: (B, T, C) -> patch -> (B, n_patches, d_model)
        B, T, C = x.shape
        # Truncate to multiple of patch_len
        T_trunc = (T // self.patch_len) * self.patch_len
        x = x[:, :T_trunc]
        # Reshape to patches
        n_patches = T_trunc // self.patch_len
        x = x.reshape(B, n_patches, self.patch_len * C)
        return self.norm(self.proj(x))


class TransformerEncoder(nn.Module):
    """Transformer encoder."""
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
    """Mechanical-JEPA for multivariate time series.

    Input: (B, T, C) — setpoint or setpoint+effort concatenated
    Masking: temporal blocks (contiguous patches)
    Loss: L2 in latent space (NOT reconstruction)
    """
    def __init__(self, n_channels, window_size, d_model=64, n_heads=4, n_layers=3,
                 predictor_layers=2, patch_len=8, mask_ratio=0.30, ema_decay=0.996):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.ema_decay = ema_decay
        self.d_model = d_model
        self.patch_len = patch_len
        self.n_patches = window_size // patch_len

        # Patch embedding
        self.patch_embed = PatchEmbed(n_channels, patch_len, d_model)

        # Online encoder
        self.encoder = TransformerEncoder(d_model, n_heads, n_layers)

        # Target encoder (EMA copy of encoder)
        self.target_encoder = copy.deepcopy(self.encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False

        # Predictor: lightweight MLP (operates on context tokens)
        # Predicts target tokens from context tokens + mask tokens
        self.mask_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        predictor_layers_list = []
        for i in range(predictor_layers):
            predictor_layers_list.extend([
                nn.Linear(d_model, d_model),
                nn.LayerNorm(d_model),
                nn.GELU(),
            ])
        predictor_layers_list.append(nn.Linear(d_model, d_model))
        self.predictor = nn.Sequential(*predictor_layers_list)

        # Prediction positional embed (for masked positions)
        self.pred_pos_embed = nn.Parameter(torch.randn(1, self.n_patches, d_model) * 0.02)

    def create_block_mask(self, n_patches, mask_ratio, rng=None):
        """Create contiguous block mask."""
        n_mask = max(1, int(n_patches * mask_ratio))
        # Random start of masked block
        if rng is not None:
            start = rng.randint(0, n_patches - n_mask + 1)
        else:
            start = torch.randint(0, n_patches - n_mask + 1, (1,)).item()
        mask = torch.zeros(n_patches, dtype=torch.bool)
        mask[start:start + n_mask] = True
        return mask

    def forward(self, x):
        """
        x: (B, T, C) — input time series window
        Returns loss and dict with representations
        """
        B, T, C = x.shape

        # 1. Patch embedding
        patches = self.patch_embed(x)  # (B, n_patches, d_model)
        n_patches = patches.size(1)

        # 2. Create block mask (same for all in batch)
        mask = self.create_block_mask(n_patches, self.mask_ratio)
        context_idx = (~mask).nonzero().squeeze(-1)   # visible patches
        masked_idx = mask.nonzero().squeeze(-1)         # masked patches

        # 3. Get target representations (no gradient)
        with torch.no_grad():
            z_target = self.target_encoder(patches)     # (B, n_patches, d_model)

        # 4. Encode visible patches only
        context_patches = patches[:, context_idx]       # (B, n_context, d_model)
        z_context = self.encoder(context_patches)       # (B, n_context, d_model)

        # 5. Predict masked patches
        # Build prediction input: context + mask tokens at masked positions
        n_mask = len(masked_idx)
        mask_tokens = self.mask_token.expand(B, n_mask, -1)     # (B, n_mask, d_model)
        mask_tokens = mask_tokens + self.pred_pos_embed[:, masked_idx]

        # Simple: predict from mean of context + positional info
        ctx_mean = z_context.mean(dim=1, keepdim=True).expand(-1, n_mask, -1)  # (B, n_mask, d_model)
        pred_input = ctx_mean + mask_tokens
        z_pred = self.predictor(pred_input)             # (B, n_mask, d_model)

        # 6. JEPA loss: L2 between predicted and target (masked positions only)
        z_target_masked = z_target[:, masked_idx]       # (B, n_mask, d_model)
        loss = F.mse_loss(z_pred, z_target_masked.detach())

        return loss, {
            'z_context': z_context,
            'z_target': z_target,
            'z_pred': z_pred,
            'mask_ratio_actual': n_mask / n_patches,
        }

    @torch.no_grad()
    def encode(self, x):
        """Get representations for evaluation."""
        patches = self.patch_embed(x)
        z = self.encoder(patches)
        return z.mean(dim=1)  # (B, d_model) — mean-pooled

    def ema_update(self):
        """Update target encoder with EMA."""
        with torch.no_grad():
            for p_enc, p_tgt in zip(self.encoder.parameters(),
                                     self.target_encoder.parameters()):
                p_tgt.data = self.ema_decay * p_tgt.data + (1 - self.ema_decay) * p_enc.data


# ============================================================================
# Data Loading
# ============================================================================

def load_aursad_data(max_episodes=None, window_size=64):
    """Load AURSAD via FactoryNetDataset."""
    print("[Data] Loading AURSAD from HuggingFace...")
    import logging
    logging.getLogger('industrialjepa').setLevel(logging.WARNING)
    logging.getLogger('datasets').setLevel(logging.WARNING)

    config = FactoryNetConfig(
        dataset_name="Forgis/FactoryNet_Dataset",
        config_name="normalized",
        data_source="aursad",
        window_size=window_size,
        stride=window_size // 2,
        normalize=True,
        norm_mode="episode",
        max_episodes=max_episodes,
        train_healthy_only=False,  # We need anomalies for evaluation
    )

    # Load train dataset (will download all parquets, cached after first run)
    train_ds = FactoryNetDataset(config, split='train')
    val_ds = FactoryNetDataset(config, split='val', shared_data=train_ds.get_shared_data())
    test_ds = FactoryNetDataset(config, split='test', shared_data=train_ds.get_shared_data())

    # Sample structure to determine n_channels
    sample = train_ds[0]
    setpoint, effort, meta = sample
    n_setpoint = setpoint.shape[1]
    n_effort = effort.shape[1]
    n_channels = n_setpoint + n_effort  # Use setpoint + effort concatenated

    print(f"  Train: {len(train_ds)} windows")
    print(f"  Val: {len(val_ds)} windows")
    print(f"  Test: {len(test_ds)} windows")
    print(f"  Channels: setpoint={n_setpoint}, effort={n_effort}, total={n_channels}")

    # Extract labels (anomaly flag)
    def extract_arrays(ds):
        setpoints, efforts, labels = [], [], []
        for i in range(len(ds)):
            sp, ef, m = ds[i]
            setpoints.append(sp)
            efforts.append(ef)
            labels.append(1 if m['is_anomaly'] else 0)
        return (torch.stack(setpoints),
                torch.stack(efforts),
                torch.tensor(labels, dtype=torch.long))

    print("  Extracting arrays...")
    train_sp, train_ef, train_labels = extract_arrays(train_ds)
    val_sp, val_ef, val_labels = extract_arrays(val_ds)
    test_sp, test_ef, test_labels = extract_arrays(test_ds)

    # Concatenate setpoint + effort
    train_X = torch.cat([train_sp, train_ef], dim=-1)
    val_X = torch.cat([val_sp, val_ef], dim=-1)
    test_X = torch.cat([test_sp, test_ef], dim=-1)

    print(f"  Train X shape: {train_X.shape}, labels: {train_labels.shape}")
    print(f"  Anomaly rate: train={train_labels.float().mean():.3f}, test={test_labels.float().mean():.3f}")

    return {
        'train': (train_X, train_labels),
        'val': (val_X, val_labels),
        'test': (test_X, test_labels),
        'n_channels': n_channels,
    }


# ============================================================================
# Training Functions
# ============================================================================

def pretrain_jepa(model, train_X, config, seed=42):
    """Pretrain JEPA on unlabeled data."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    # For pretraining: use all data, ignore labels
    dataset = TensorDataset(train_X)
    loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)

    optimizer = torch.optim.AdamW(
        list(model.encoder.parameters()) +
        list(model.predictor.parameters()) +
        list(model.patch_embed.parameters()),
        lr=config['lr'],
        weight_decay=config['weight_decay']
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['epochs_pretrain'])

    losses = []
    for epoch in range(config['epochs_pretrain']):
        model.train()
        epoch_loss = 0
        n = 0
        for (batch_X,) in loader:
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


def eval_linear_probe(model, train_X, train_y, test_X, test_y):
    """Linear probe on frozen JEPA representations."""
    model.eval()
    with torch.no_grad():
        # Process in batches for memory efficiency
        def get_reps(X):
            reps = []
            for i in range(0, len(X), 256):
                batch = X[i:i+256].to(DEVICE)
                reps.append(model.encode(batch).cpu().numpy())
            return np.vstack(reps)

        train_reps = get_reps(train_X)
        test_reps = get_reps(test_X)

    scaler = StandardScaler()
    train_reps_s = scaler.fit_transform(train_reps)
    test_reps_s = scaler.transform(test_reps)

    train_y_np = train_y.numpy()
    test_y_np = test_y.numpy()

    if len(np.unique(train_y_np)) < 2:
        print("    Warning: only one class in train set, skipping linear probe")
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

    return {'auroc': auroc, 'f1': f1}


def eval_forecasting(model_init_fn, train_X, test_X, frozen_encoder=None,
                     n_epochs=20, lr=1e-3, batch_size=64):
    """Evaluate next-step forecasting.

    If frozen_encoder is provided: freeze encoder, train only head.
    Otherwise: train from scratch.
    """
    n_channels = train_X.shape[-1]
    window_size = train_X.shape[1]

    # Build forecasting head on top of JEPA encoder
    class ForecastHead(nn.Module):
        def __init__(self, encoder, d_model, n_channels):
            super().__init__()
            self.encoder = encoder
            self.head = nn.Linear(d_model, n_channels)

        def forward(self, x):
            z = self.encoder.encode(x)   # (B, d_model)
            return self.head(z)           # (B, n_channels)

    model = model_init_fn().to(DEVICE)
    head = ForecastHead(model, model.d_model, n_channels).to(DEVICE)

    if frozen_encoder is not None:
        # Load pretrained weights and freeze
        head.encoder.load_state_dict(frozen_encoder.state_dict())
        for p in head.encoder.parameters():
            p.requires_grad = False
        params = head.head.parameters()
    else:
        params = head.parameters()

    optimizer = torch.optim.AdamW(params, lr=lr)

    # Build forecasting pairs: X[:-1] -> X[-1]
    # Use last timestep as target (next-step prediction)
    # For simplicity: predict mean of last quarter
    input_X = train_X[:, :-1]  # (N, T-1, C)
    # Pad to original size
    pad = torch.zeros(input_X.size(0), 1, n_channels)
    input_X_padded = torch.cat([input_X, pad], dim=1)  # (N, T, C)
    target_Y = train_X[:, -1]  # (N, C)

    dataset = TensorDataset(input_X_padded, target_Y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    head.train()
    for epoch in range(n_epochs):
        for (bx, by) in loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            optimizer.zero_grad()
            pred = head(bx)
            loss = F.mse_loss(pred, by)
            loss.backward()
            optimizer.step()

    # Test
    head.eval()
    test_input = test_X[:, :-1]
    pad_test = torch.zeros(test_input.size(0), 1, n_channels)
    test_input_padded = torch.cat([test_input, pad_test], dim=1)
    test_target = test_X[:, -1]

    with torch.no_grad():
        test_mse = 0
        n = 0
        for i in range(0, len(test_input_padded), batch_size):
            bx = test_input_padded[i:i+batch_size].to(DEVICE)
            by = test_target[i:i+batch_size].to(DEVICE)
            pred = head(bx)
            test_mse += F.mse_loss(pred, by).item()
            n += 1
        test_mse /= max(n, 1)

    return test_mse


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 60)
    print("EXP 51: Mechanical-JEPA on AURSAD")
    print(f"Device: {DEVICE}")
    print("=" * 60)

    t0 = time.time()
    config = JEPA_CONFIG

    # Load data
    data = load_aursad_data(max_episodes=None, window_size=config['window_size'])
    n_channels = data['n_channels']
    train_X, train_y = data['train']
    val_X, val_y = data['val']
    test_X, test_y = data['test']

    print(f"\nData loaded in {time.time()-t0:.1f}s")

    all_results = []

    for seed in SEEDS:
        print(f"\n{'='*50}")
        print(f"SEED {seed}")
        print(f"{'='*50}")
        torch.manual_seed(seed)
        np.random.seed(seed)

        # ---- Method 1: JEPA pretrained ----
        print("\n[1] JEPA Pretraining...")
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
        print(f"  Model params: {n_params:,}")

        t_pretrain = time.time()
        losses = pretrain_jepa(model_jepa, train_X, config, seed=seed)
        print(f"  Pretraining done in {time.time()-t_pretrain:.1f}s")
        print(f"  Loss: {losses[0]:.4f} -> {losses[-1]:.4f}")

        # Linear probe from JEPA
        print("\n  [1a] Linear probe (anomaly detection)...")
        probe_results = eval_linear_probe(model_jepa, train_X, train_y, test_X, test_y)
        print(f"  AUROC={probe_results['auroc']:.4f}, F1={probe_results['f1']:.4f}")

        # ---- Method 2: Random init (no pretraining) ----
        print("\n[2] Random init (no pretraining)...")
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
        print(f"  AUROC={probe_random['auroc']:.4f}, F1={probe_random['f1']:.4f}")

        seed_result = {
            'seed': seed,
            'n_params': n_params,
            'n_channels': n_channels,
            'pretrain_loss_start': float(losses[0]),
            'pretrain_loss_end': float(losses[-1]),
            'jepa_probe_auroc': float(probe_results['auroc']),
            'jepa_probe_f1': float(probe_results['f1']),
            'random_probe_auroc': float(probe_random['auroc']),
            'random_probe_f1': float(probe_random['f1']),
            'jepa_vs_random_delta': float(probe_results['auroc'] - probe_random['auroc']),
        }
        all_results.append(seed_result)

        print(f"\n  JEPA vs Random: {probe_results['auroc']:.4f} vs {probe_random['auroc']:.4f} "
              f"(delta={seed_result['jepa_vs_random_delta']:+.4f})")

    # Aggregate across seeds
    jepa_aurocs = [r['jepa_probe_auroc'] for r in all_results]
    random_aurocs = [r['random_probe_auroc'] for r in all_results]
    deltas = [r['jepa_vs_random_delta'] for r in all_results]

    summary = {
        'dataset': 'AURSAD',
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
        'seeds': all_results,
    }

    print("\n" + "=" * 60)
    print("AURSAD JEPA RESULTS")
    print("=" * 60)
    print(f"JEPA pretrained:  AUROC = {summary['jepa_auroc_mean']:.4f} ± {summary['jepa_auroc_std']:.4f}")
    print(f"Random init:      AUROC = {summary['random_auroc_mean']:.4f} ± {summary['random_auroc_std']:.4f}")
    print(f"Delta:            {summary['delta_mean']:+.4f} ± {summary['delta_std']:.4f}")
    print(f"Verdict:          {summary['verdict']}")
    print(f"Total time: {time.time()-t0:.1f}s")

    # Save
    out_path = PROJECT_ROOT / "datasets" / "data" / "aursad_jepa_results.json"
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {out_path}")

    return summary


if __name__ == "__main__":
    main()
