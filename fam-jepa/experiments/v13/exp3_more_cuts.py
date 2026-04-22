"""
V13 Experiment 3: More window cuts per engine during fine-tuning.

Hypothesis: STAR uses 15,000 training windows (176/engine, stride=1 from ALL positions).
JEPA uses only 5 cuts/engine = 425 windows. This 35x difference in training data
is likely a major contributor to the 2.3 RMSE gap.

We test increasing n_cuts_per_engine from 5 to [10, 20, 50, 176 (STAR-equiv)].

Key question: Does more training data during fine-tuning close the gap?

Output: experiments/v13/cuts_results.json
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
PATIENCE = 20

# Cuts to test: 5 (baseline), 10, 20, 50, 176 (STAR-equivalent)
N_CUTS_LIST = [5, 10, 20, 50, 176]

print(f"Device: {DEVICE}")
print(f"Checkpoint: {CKPT}")
print(f"D_MODEL: {D_MODEL}, Budget: {int(BUDGET*100)}%, Seeds: {len(SEEDS)}")
print(f"N_CUTS to test: {N_CUTS_LIST}")
print("="*60)


class LinearProbe(nn.Module):
    """Linear probe (same as original RULProbe)."""
    def __init__(self, d_model=256):
        super().__init__()
        self.linear = nn.Linear(d_model, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, h):
        return self.sigmoid(self.linear(h)).squeeze(-1)


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


def run_finetune(train_eng, val_eng, test_eng, test_rul, n_cuts, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

    model_ft = TrajectoryJEPA(
        n_sensors=N_SENSORS, patch_length=1, d_model=D_MODEL,
        n_heads=4, n_layers=2, d_ff=512
    )
    model_ft.load_state_dict(torch.load(CKPT, map_location='cpu'))
    model_ft = model_ft.to(DEVICE)
    probe = LinearProbe(d_model=D_MODEL).to(DEVICE)

    train_ds = CMAPSSFinetuneDataset(train_eng, n_cuts_per_engine=n_cuts, seed=seed)
    val_ds = CMAPSSFinetuneDataset(val_eng, use_last_only=True)
    test_ds = CMAPSSTestDataset(test_eng, test_rul)

    # Adjust batch size for larger datasets
    batch_size = min(64, len(train_ds) // 10 + 1)
    batch_size = max(16, batch_size)

    tr = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_finetune)
    va = DataLoader(val_ds, batch_size=16, shuffle=False, collate_fn=collate_finetune)
    te = DataLoader(test_ds, batch_size=16, shuffle=False, collate_fn=collate_test)

    # E2E fine-tuning (encoder unfrozen)
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
            if no_impr >= PATIENCE:
                break

    model_ft.context_encoder.load_state_dict(best_encoder_state)
    probe.load_state_dict(best_probe_state)
    test_rmse = eval_test_rmse(model_ft, probe, te)
    return test_rmse, best_val, ep + 1, len(train_ds)


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
for n_cuts in N_CUTS_LIST:
    print(f"\n{'='*60}")
    print(f"N_CUTS_PER_ENGINE: {n_cuts} (est. {n_cuts * len(train_eng)} train windows)")
    print(f"{'='*60}")

    seed_rmses = []
    for seed in SEEDS:
        t0 = time.time()
        test_rmse, val_rmse, n_epochs, n_train = run_finetune(
            train_eng, val_eng, test_eng, test_rul, n_cuts, seed
        )
        elapsed = time.time() - t0
        seed_rmses.append(test_rmse)
        print(f"  seed={seed}: test={test_rmse:.3f}, val={val_rmse:.3f}, "
              f"ep={n_epochs}, n_train={n_train}, t={elapsed:.0f}s")

    mean_rmse = float(np.mean(seed_rmses))
    std_rmse = float(np.std(seed_rmses))
    print(f"\n  n_cuts={n_cuts}: {mean_rmse:.3f} +/- {std_rmse:.3f}")

    results[str(n_cuts)] = {
        'n_cuts': n_cuts,
        'mean_rmse': mean_rmse,
        'std_rmse': std_rmse,
        'per_seed_rmse': [float(r) for r in seed_rmses],
    }

    with open(os.path.join(V13_DIR, 'cuts_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Saved intermediate results")


# Final summary
print("\n" + "="*60)
print("FINAL SUMMARY")
print("="*60)
print(f"{'N_Cuts':>8} | {'RMSE Mean':>10} +/- {'Std':>6} | {'Delta vs 5':>12}")
print("-"*50)
ref = results.get('5', {}).get('mean_rmse', 14.48)
for n_cuts_str, r in sorted(results.items(), key=lambda x: int(x[0])):
    delta = r['mean_rmse'] - ref
    print(f"{n_cuts_str:>8} | {r['mean_rmse']:>10.3f} +/- {r['std_rmse']:>4.3f}  "
          f"({delta:+.3f})")

print(f"\nV12 E2E reference (5 cuts): 14.23 +/- 0.39")
print(f"STAR reference: 12.19 +/- 0.55 (FD001, 5 seeds)")
print(f"STAR training windows: 15000")
