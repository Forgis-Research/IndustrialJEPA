"""
V13 Experiment 4: Deeper Architecture (V4: n_layers=4 vs V2 n_layers=2).

Hypothesis: V2 uses 2 Transformer layers in the context encoder (1.26M params).
STAR uses a more complex architecture (~3.67M params total). Scaling depth from 2->4
layers might give the encoder more representational capacity to capture the full
degradation trajectory, closing some of the 2.3 RMSE gap.

Architecture variants:
- V2 (baseline): n_layers=2, d_model=256, n_heads=4, d_ff=512 (1.26M context encoder)
- V4 (deeper): n_layers=4, d_model=256, n_heads=4, d_ff=512 (~2.3M context encoder)
- V5 (wider): n_layers=2, d_model=384, n_heads=4, d_ff=768 (~2.7M context encoder)

Plan:
1. Pretrain V4 and V5 from scratch (200 epochs, probe-based early stopping)
2. Fine-tune E2E at 100% labels (5 seeds)
3. Compare to V2 baseline (14.23 +/- 0.39)

Kill criterion: If V4 frozen probe RMSE > V2 frozen (17.81), stop - deeper is not better.

Output: experiments/v13/arch_results.json
"""

import os
import sys
import time
import json
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

BASE = '/home/sagemaker-user/IndustrialJEPA/mechanical-jepa'
V11_DIR = os.path.join(BASE, 'experiments/v11')
V13_DIR = os.path.join(BASE, 'experiments/v13')
sys.path.insert(0, V11_DIR)

from data_utils import (
    load_cmapss_subset, CMAPSSFinetuneDataset, CMAPSSTestDataset,
    CMAPSSPretrainDataset, collate_pretrain, collate_finetune, collate_test,
    N_SENSORS, RUL_CAP
)
from models import TrajectoryJEPA, trajectory_jepa_loss, count_parameters
from torch.utils.data import DataLoader

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEEDS = [42, 123, 456, 789, 1024]
BUDGET = 1.0
LR_FINETUNE = 1e-4
N_FINETUNE_EPOCHS = 150
PATIENCE_FT = 20
N_CUTS_FINETUNE = 5  # Start with same as baseline for fair comparison

print(f"Device: {DEVICE}")
print("="*60)

# ============================================================
# Architecture configs to test
# ============================================================
ARCH_CONFIGS = {
    'v2_baseline': dict(n_layers=2, d_model=256, n_heads=4, d_ff=512),
    'v4_deep':     dict(n_layers=4, d_model=256, n_heads=4, d_ff=512),
    # 'v5_wide':   dict(n_layers=2, d_model=384, n_heads=6, d_ff=768),  # optional
}


class LinearProbe(nn.Module):
    def __init__(self, d_model=256):
        super().__init__()
        self.linear = nn.Linear(d_model, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, h):
        return self.sigmoid(self.linear(h)).squeeze(-1)


def eval_probe_rmse_quick(enc_model, train_eng, val_eng, d_model, n_probe_epochs=50):
    """Quick probe evaluation for early stopping."""
    enc_model.eval()
    probe = LinearProbe(d_model).to(DEVICE)
    optim_probe = torch.optim.Adam(probe.parameters(), lr=1e-3)
    train_ds = CMAPSSFinetuneDataset(train_eng, n_cuts_per_engine=3)
    val_ds = CMAPSSFinetuneDataset(val_eng, use_last_only=False, n_cuts_per_engine=10)
    tr_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate_finetune)
    va_loader = DataLoader(val_ds, batch_size=32, shuffle=False, collate_fn=collate_finetune)
    for ep in range(n_probe_epochs):
        probe.train()
        for past, mask, rul in tr_loader:
            past, mask, rul = past.to(DEVICE), mask.to(DEVICE), rul.to(DEVICE)
            with torch.no_grad():
                h = enc_model.context_encoder(past, mask)
            pred = probe(h)
            loss = F.mse_loss(pred.squeeze(), rul.float())
            optim_probe.zero_grad(); loss.backward(); optim_probe.step()
    probe.eval()
    preds, targets = [], []
    with torch.no_grad():
        for past, mask, rul in va_loader:
            past, mask = past.to(DEVICE), mask.to(DEVICE)
            h = enc_model.context_encoder(past, mask)
            preds.append(probe(h).cpu().numpy())
            targets.append(rul.numpy())
    preds = np.concatenate(preds) * RUL_CAP
    targets = np.concatenate(targets) * RUL_CAP
    return float(np.sqrt(np.mean((preds - targets)**2)))


def pretrain_architecture(arch_name, arch_cfg, data):
    """Pretrain a model with given architecture config."""
    print(f"\n{'='*60}")
    print(f"Pretraining {arch_name}: {arch_cfg}")
    print(f"{'='*60}")

    model = TrajectoryJEPA(
        n_sensors=N_SENSORS, patch_length=1,
        d_model=arch_cfg['d_model'], n_heads=arch_cfg['n_heads'],
        n_layers=arch_cfg['n_layers'], d_ff=arch_cfg['d_ff'],
        dropout=0.1, ema_momentum=0.99, predictor_hidden=256,
    )
    print(f"Model params: {count_parameters(model):,}")
    model = model.to(DEVICE)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=3e-4, weight_decay=0.01
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 200)

    N_EPOCHS = 200
    BATCH_SIZE = 4
    N_CUTS = 30
    LAMBDA_VAR = 0.01
    PROBE_EVERY = 5
    patience_probe = 10

    ckpt_path = os.path.join(V13_DIR, f'best_pretrain_{arch_name}.pt')
    best_probe_rmse = float('inf')
    best_state = None
    no_improve = 0
    history = {'loss': [], 'probe_rmse': [], 'probe_epochs': []}

    t0 = time.time()
    for epoch in range(1, N_EPOCHS + 1):
        train_ds = CMAPSSPretrainDataset(
            data['train_engines'], n_cuts_per_engine=N_CUTS,
            min_past=10, min_horizon=5, max_horizon=30, seed=epoch
        )
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                                  collate_fn=collate_pretrain, num_workers=0)
        model.train()
        total_loss, n = 0.0, 0
        for past, past_mask, future, future_mask, k, t in train_loader:
            past, past_mask = past.to(DEVICE), past_mask.to(DEVICE)
            future, future_mask = future.to(DEVICE), future_mask.to(DEVICE)
            k = k.to(DEVICE)
            optimizer.zero_grad()
            pred_future, h_future, h_past = model.forward_pretrain(
                past, past_mask, future, future_mask, k
            )
            loss, pred_l, var_l = trajectory_jepa_loss(pred_future, h_future, LAMBDA_VAR)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            model.update_ema()
            B = past.shape[0]
            total_loss += loss.item() * B
            n += B
        history['loss'].append(total_loss / n)
        scheduler.step()

        if epoch % PROBE_EVERY == 0 or epoch == 1:
            probe_rmse = eval_probe_rmse_quick(model, data['train_engines'], data['val_engines'],
                                                arch_cfg['d_model'])
            history['probe_rmse'].append(probe_rmse)
            history['probe_epochs'].append(epoch)
            print(f"  Ep {epoch:3d} | loss={total_loss/n:.4f} | probe_RMSE={probe_rmse:.2f} "
                  f"(best={best_probe_rmse:.2f}, no_improve={no_improve})")

            if probe_rmse < best_probe_rmse:
                best_probe_rmse = probe_rmse
                best_state = copy.deepcopy(model.state_dict())
                torch.save(best_state, ckpt_path)
                no_improve = 0
            else:
                no_improve += 1

            if no_improve >= patience_probe:
                print(f"  Early stopping at epoch {epoch}")
                break
        elif epoch % 20 == 0:
            print(f"  Ep {epoch:3d} | loss={total_loss/n:.4f}")

    elapsed = (time.time() - t0) / 60
    print(f"\nPretraining {arch_name} complete in {elapsed:.1f} min")
    print(f"Best probe RMSE: {best_probe_rmse:.2f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, ckpt_path, best_probe_rmse, history


def eval_test_rmse(model_ft, probe, te_loader):
    """Compute test RMSE in raw cycles."""
    model_ft.eval(); probe.eval()
    preds, trues = [], []
    with torch.no_grad():
        for past, mask, rul in te_loader:
            past, mask = past.to(DEVICE), mask.to(DEVICE)
            h_past = model_ft.context_encoder(past, mask)
            pred_norm = probe(h_past)
            pred_raw = pred_norm.cpu().numpy().flatten() * RUL_CAP
            preds.extend(pred_raw.tolist())
            trues.extend(rul.numpy().flatten().tolist())
    return float(np.sqrt(np.mean((np.array(preds) - np.array(trues))**2)))


def run_finetune(ckpt_path, arch_cfg, train_eng, val_eng, test_eng, test_rul, seed):
    """E2E fine-tuning from pretrained checkpoint."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    model_ft = TrajectoryJEPA(
        n_sensors=N_SENSORS, patch_length=1,
        d_model=arch_cfg['d_model'], n_heads=arch_cfg['n_heads'],
        n_layers=arch_cfg['n_layers'], d_ff=arch_cfg['d_ff'],
    )
    model_ft.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
    model_ft = model_ft.to(DEVICE)
    probe = LinearProbe(d_model=arch_cfg['d_model']).to(DEVICE)

    train_ds = CMAPSSFinetuneDataset(train_eng, n_cuts_per_engine=N_CUTS_FINETUNE, seed=seed)
    val_ds = CMAPSSFinetuneDataset(val_eng, use_last_only=True)
    test_ds = CMAPSSTestDataset(test_eng, test_rul)

    batch_size = 32
    tr = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_finetune)
    va = DataLoader(val_ds, batch_size=16, shuffle=False, collate_fn=collate_finetune)
    te = DataLoader(test_ds, batch_size=16, shuffle=False, collate_fn=collate_test)

    # E2E fine-tuning
    for p in model_ft.context_encoder.parameters():
        p.requires_grad = True
    optim = torch.optim.Adam(
        list(model_ft.context_encoder.parameters()) + list(probe.parameters()),
        lr=LR_FINETUNE, weight_decay=0.0
    )

    best_val = float('inf')
    best_encoder_state = None
    best_probe_state = None
    no_impr = 0

    for ep in range(N_FINETUNE_EPOCHS):
        model_ft.train(); probe.train()
        for past, mask, rul in tr:
            past, mask, rul = past.to(DEVICE), mask.to(DEVICE), rul.to(DEVICE)
            h_past = model_ft.context_encoder(past, mask)
            pred = probe(h_past)
            loss = F.mse_loss(pred.squeeze(), rul.float())
            optim.zero_grad(); loss.backward(); optim.step()

        model_ft.eval(); probe.eval()
        val_losses = []
        with torch.no_grad():
            for past, mask, rul in va:
                past, mask, rul = past.to(DEVICE), mask.to(DEVICE), rul.to(DEVICE)
                h_past = model_ft.context_encoder(past, mask)
                pred = probe(h_past)
                val_losses.append(F.mse_loss(pred.squeeze(), rul.float()).item())
        val_rmse = float(np.sqrt(np.mean(val_losses))) * RUL_CAP

        if val_rmse < best_val:
            best_val = val_rmse
            best_encoder_state = {k: v.clone() for k, v in model_ft.context_encoder.state_dict().items()}
            best_probe_state = {k: v.clone() for k, v in probe.state_dict().items()}
            no_impr = 0
        else:
            no_impr += 1
            if no_impr >= PATIENCE_FT:
                break

    model_ft.context_encoder.load_state_dict(best_encoder_state)
    probe.load_state_dict(best_probe_state)
    test_rmse = eval_test_rmse(model_ft, probe, te)
    return test_rmse, best_val, ep + 1


# Load data
data = load_cmapss_subset('FD001')
print(f"Train engines: {len(data['train_engines'])}, Val: {len(data['val_engines'])}")

results = {}

for arch_name, arch_cfg in ARCH_CONFIGS.items():
    print(f"\n{'#'*60}")
    print(f"ARCHITECTURE: {arch_name}")
    print(f"{'#'*60}")

    # Check if we have a V2 baseline checkpoint to skip pretraining
    if arch_name == 'v2_baseline':
        v2_ckpt = os.path.join(V11_DIR, 'best_pretrain_L1_v2.pt')
        if os.path.exists(v2_ckpt):
            print(f"  Using existing V2 checkpoint: {v2_ckpt}")
            ckpt_path = v2_ckpt
            best_probe_rmse = 19.22  # Known from V12 diagnostics
        else:
            print("  V2 checkpoint not found, pretraining...")
            _, ckpt_path, best_probe_rmse, _ = pretrain_architecture(arch_name, arch_cfg, data)
    else:
        _, ckpt_path, best_probe_rmse, _ = pretrain_architecture(arch_name, arch_cfg, data)

    print(f"\nFine-tuning {arch_name} (E2E, 5 seeds)...")
    seed_rmses = []
    for seed in SEEDS:
        t0 = time.time()
        test_rmse, val_rmse, n_epochs = run_finetune(
            ckpt_path, arch_cfg, data['train_engines'], data['val_engines'],
            data['test_engines'], data['test_rul'], seed
        )
        elapsed = time.time() - t0
        seed_rmses.append(test_rmse)
        print(f"  seed={seed}: test={test_rmse:.3f}, val={val_rmse:.3f}, "
              f"ep={n_epochs}, t={elapsed:.0f}s")

    mean_rmse = float(np.mean(seed_rmses))
    std_rmse = float(np.std(seed_rmses))
    print(f"\n  {arch_name}: {mean_rmse:.3f} +/- {std_rmse:.3f}")

    results[arch_name] = {
        'arch': arch_cfg,
        'best_probe_rmse': best_probe_rmse,
        'mean_rmse': mean_rmse,
        'std_rmse': std_rmse,
        'per_seed_rmse': [float(r) for r in seed_rmses],
    }

    with open(os.path.join(V13_DIR, 'arch_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Saved intermediate results")

# Final summary
print("\n" + "="*60)
print("FINAL SUMMARY")
print("="*60)
print(f"{'Architecture':>20} | {'Pretrain RMSE':>14} | {'E2E RMSE':>10} +/- {'Std':>6}")
print("-"*60)
for arch_name, r in results.items():
    print(f"{arch_name:>20} | {r['best_probe_rmse']:>14.2f} | "
          f"{r['mean_rmse']:>10.3f} +/- {r['std_rmse']:>4.3f}")

print(f"\nV12 E2E reference (V2 baseline): 14.23 +/- 0.39")
print(f"STAR reference: 12.19 +/- 0.55 (FD001, 5 seeds)")
