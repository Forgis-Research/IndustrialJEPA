"""
V13 Experiment 2: Non-linear probe head variants.

Hypothesis: The linear RULProbe (Linear->Sigmoid) is too simple.
A non-linear MLP head may better capture the non-linear mapping from
JEPA latent space to RUL.

Current probe: Linear(256, 1) -> Sigmoid (linear)
Variants:
- mlp_small: Linear(256, 64) -> ReLU -> Linear(64, 1) -> Sigmoid
- mlp_large: Linear(256, 128) -> ReLU -> Dropout(0.1) -> Linear(128, 32) -> ReLU -> Linear(32, 1) -> Sigmoid
- mlp_bn: Linear(256, 64) -> BN -> ReLU -> Linear(64, 1) -> Sigmoid
- linear_baseline: same as RULProbe (replicate Exp 1 e2e_baseline)

Runs E2E fine-tuning (encoder unfrozen) for all variants.
Goal: Determine if probe nonlinearity closes any of the ~2 RMSE gap.

Output: experiments/v13/probe_variant_results.json
"""

import os
import sys
import time
import json
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
    collate_finetune, collate_test, N_SENSORS, RUL_CAP
)
from models import TrajectoryJEPA
from torch.utils.data import DataLoader

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
D_MODEL = 256
CKPT = os.path.join(V11_DIR, 'best_pretrain_L1_v2.pt')
SEEDS = [42, 123, 456, 789, 1024]
BUDGET = 1.0
N_EPOCHS = 150
LR = 1e-4

print(f"Device: {DEVICE}")
print(f"Checkpoint: {CKPT}")
print(f"D_MODEL: {D_MODEL}, Budget: {int(BUDGET*100)}%, Seeds: {len(SEEDS)}")
print("="*60)


class LinearProbe(nn.Module):
    """Linear probe (same as original RULProbe)."""
    def __init__(self, d_model=256):
        super().__init__()
        self.linear = nn.Linear(d_model, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, h):
        return self.sigmoid(self.linear(h)).squeeze(-1)


class MLPSmall(nn.Module):
    """Two-layer MLP probe."""
    def __init__(self, d_model=256, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),
        )
    def forward(self, h):
        return self.net(h).squeeze(-1)


class MLPLarge(nn.Module):
    """Three-layer MLP probe with dropout."""
    def __init__(self, d_model=256, h1=128, h2=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, h1),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, 1),
            nn.Sigmoid(),
        )
    def forward(self, h):
        return self.net(h).squeeze(-1)


class MLPWithBN(nn.Module):
    """Two-layer MLP with batch normalization."""
    def __init__(self, d_model=256, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),
        )
    def forward(self, h):
        return self.net(h).squeeze(-1)


PROBE_CLASSES = {
    'linear_baseline': LinearProbe,
    'mlp_small': MLPSmall,
    'mlp_large': MLPLarge,
    'mlp_bn': MLPWithBN,
}


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
    rmse = float(np.sqrt(np.mean((np.array(preds) - np.array(trues))**2)))
    return rmse


def run_finetune(train_eng, val_eng, test_eng, test_rul, probe_cls, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

    model_ft = TrajectoryJEPA(
        n_sensors=N_SENSORS, patch_length=1, d_model=D_MODEL,
        n_heads=4, n_layers=2, d_ff=512
    )
    model_ft.load_state_dict(torch.load(CKPT, map_location='cpu'))
    model_ft = model_ft.to(DEVICE)
    probe = probe_cls(d_model=D_MODEL).to(DEVICE)

    train_ds = CMAPSSFinetuneDataset(train_eng, n_cuts_per_engine=5, seed=seed)
    val_ds = CMAPSSFinetuneDataset(val_eng, use_last_only=True)
    test_ds = CMAPSSTestDataset(test_eng, test_rul)
    tr = DataLoader(train_ds, batch_size=16, shuffle=True, collate_fn=collate_finetune)
    va = DataLoader(val_ds, batch_size=16, shuffle=False, collate_fn=collate_finetune)
    te = DataLoader(test_ds, batch_size=16, shuffle=False, collate_fn=collate_test)

    # E2E fine-tuning
    for p in model_ft.context_encoder.parameters():
        p.requires_grad = True
    optim = torch.optim.Adam(
        list(model_ft.context_encoder.parameters()) + list(probe.parameters()),
        lr=LR, weight_decay=0.0
    )

    best_val = float('inf')
    best_encoder_state = None
    best_probe_state = None
    no_impr = 0
    patience = 20

    for ep in range(N_EPOCHS):
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
        val_loss = float(np.mean(val_losses))
        val_rmse = float(np.sqrt(val_loss)) * RUL_CAP

        if val_rmse < best_val:
            best_val = val_rmse
            best_encoder_state = {k: v.clone() for k, v in model_ft.context_encoder.state_dict().items()}
            best_probe_state = {k: v.clone() for k, v in probe.state_dict().items()}
            no_impr = 0
        else:
            no_impr += 1
            if no_impr >= patience:
                break

    model_ft.context_encoder.load_state_dict(best_encoder_state)
    probe.load_state_dict(best_probe_state)
    test_rmse = eval_test_rmse(model_ft, probe, te)
    return test_rmse, best_val, ep + 1


# Load data
data = load_cmapss_subset('FD001')
train_eng_all = data['train_engines']
val_eng = data['val_engines']
test_eng = data['test_engines']
test_rul = data['test_rul']
train_eng = train_eng_all

print(f"Train engines: {len(train_eng_all)}, Val: {len(val_eng)}")
print()

results = {}
for probe_name, ProbeClass in PROBE_CLASSES.items():
    print(f"\n{'='*60}")
    print(f"Probe: {probe_name}")
    print(f"{'='*60}")

    seed_rmses = []
    for seed in SEEDS:
        t0 = time.time()
        test_rmse, val_rmse, n_epochs = run_finetune(
            train_eng, val_eng, test_eng, test_rul, ProbeClass, seed
        )
        elapsed = time.time() - t0
        seed_rmses.append(test_rmse)
        print(f"  seed={seed}: test={test_rmse:.3f}, val={val_rmse:.3f}, ep={n_epochs}, t={elapsed:.0f}s")

    mean_rmse = float(np.mean(seed_rmses))
    std_rmse = float(np.std(seed_rmses))
    print(f"\n  {probe_name}: {mean_rmse:.3f} +/- {std_rmse:.3f}")

    results[probe_name] = {
        'mean_rmse': mean_rmse,
        'std_rmse': std_rmse,
        'per_seed_rmse': [float(r) for r in seed_rmses],
        'probe_class': probe_name,
    }

    with open(os.path.join(V13_DIR, 'probe_variant_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Saved intermediate results")

# Final summary
print("\n" + "="*60)
print("FINAL SUMMARY")
print("="*60)
print(f"{'Probe':>20} | {'RMSE Mean':>10} +/- {'Std':>6}")
print("-"*45)
ref = 14.23
for name, r in sorted(results.items(), key=lambda x: x[1]['mean_rmse']):
    delta = r['mean_rmse'] - ref
    print(f"{name:>20} | {r['mean_rmse']:>10.3f} +/- {r['std_rmse']:>4.3f}  (delta vs V12 ref: {delta:+.3f})")

print(f"\nV12 reference: E2E (linear probe) = {ref}")
print(f"STAR reference: 12.19 +/- 0.55 (FD001, 5 seeds)")
