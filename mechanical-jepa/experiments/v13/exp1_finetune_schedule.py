"""
V13 Experiment 1: Fine-tuning schedule variants to close the JEPA-STAR gap.

Hypothesis: E2E fine-tuning at LR=1e-4 from scratch is suboptimal.
Better schedules might help:
- Warmup-freeze: freeze encoder for first 20 epochs, then unfreeze
- Lower LR: LR=5e-5 for E2E (more conservative)
- Weight decay: L2 reg to reduce overfitting at low labels

Goal: Close some of the 2 RMSE gap between JEPA E2E (14.23) and STAR (12.19).

Output: experiments/v13/finetune_schedule_results.json
"""

import os
import sys
import time
import json
import numpy as np
import torch
import torch.nn.functional as F

BASE = '/home/sagemaker-user/IndustrialJEPA/mechanical-jepa'
V11_DIR = os.path.join(BASE, 'experiments/v11')
V13_DIR = os.path.join(BASE, 'experiments/v13')
sys.path.insert(0, V11_DIR)

from data_utils import (
    load_cmapss_subset, CMAPSSFinetuneDataset, CMAPSSTestDataset,
    collate_finetune, collate_test, N_SENSORS, RUL_CAP
)
from models import TrajectoryJEPA, RULProbe
from torch.utils.data import DataLoader

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
D_MODEL = 256
CKPT = os.path.join(V11_DIR, 'best_pretrain_L1_v2.pt')
SEEDS = [42, 123, 456, 789, 1024]
BUDGET = 1.0  # 100% labels for this experiment
N_EPOCHS = 150  # allow extra epochs for warmup schedule

print(f"Device: {DEVICE}")
print(f"Checkpoint: {CKPT}")
print(f"Budget: {int(BUDGET*100)}% labels, {len(SEEDS)} seeds")
print("="*60)

data = load_cmapss_subset('FD001')


def eval_test_rmse(model_ft, probe, te_loader):
    """Compute test RMSE in raw cycles. Probe outputs normalized [0,1]; test RUL is in raw cycles."""
    model_ft.eval(); probe.eval()
    preds, trues = [], []
    with torch.no_grad():
        for past, mask, rul in te_loader:
            past, mask = past.to(DEVICE), mask.to(DEVICE)
            h_past = model_ft.context_encoder(past, mask)
            pred_norm = probe(h_past)
            # Scale back to raw cycles (probe predicts normalized RUL in [0,1])
            pred_raw = pred_norm.cpu().numpy().flatten() * RUL_CAP
            preds.extend(pred_raw.tolist())
            trues.extend(rul.numpy().flatten().tolist())
    rmse = float(np.sqrt(np.mean((np.array(preds) - np.array(trues))**2)))
    return rmse


def run_finetune(train_eng, val_eng, test_eng, test_rul, mode, seed, lr=1e-4, wd=0.0, warmup_epochs=0):
    """
    mode: 'frozen' | 'e2e' | 'warmup_freeze' | 'e2e_low_lr' | 'e2e_wd'
    warmup_epochs: freeze encoder for first N epochs (for warmup_freeze mode)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    model_ft = TrajectoryJEPA(
        n_sensors=N_SENSORS, patch_length=1, d_model=D_MODEL,
        n_heads=4, n_layers=2, d_ff=512
    )
    model_ft.load_state_dict(torch.load(CKPT, map_location='cpu'))
    model_ft = model_ft.to(DEVICE)
    probe = RULProbe(D_MODEL).to(DEVICE)

    train_ds = CMAPSSFinetuneDataset(train_eng, n_cuts_per_engine=5, seed=seed)
    val_ds = CMAPSSFinetuneDataset(val_eng, use_last_only=True)
    test_ds = CMAPSSTestDataset(test_eng, test_rul)
    tr = DataLoader(train_ds, batch_size=16, shuffle=True, collate_fn=collate_finetune)
    va = DataLoader(val_ds, batch_size=16, shuffle=False, collate_fn=collate_finetune)
    te = DataLoader(test_ds, batch_size=16, shuffle=False, collate_fn=collate_test)

    best_val = float('inf')
    best_encoder_state = None
    best_probe_state = None
    no_impr = 0
    patience = 20

    if mode == 'frozen':
        for p in model_ft.parameters():
            p.requires_grad = False
        optim = torch.optim.Adam(probe.parameters(), lr=1e-3)
    elif mode in ('e2e', 'e2e_low_lr', 'e2e_wd', 'warmup_freeze'):
        # Start frozen or unfrozen depending on mode
        if mode == 'warmup_freeze':
            # Start frozen
            for p in model_ft.parameters():
                p.requires_grad = False
            optim = torch.optim.Adam(probe.parameters(), lr=1e-3, weight_decay=wd)
        else:
            for p in model_ft.context_encoder.parameters():
                p.requires_grad = True
            optim = torch.optim.Adam(
                list(model_ft.context_encoder.parameters()) + list(probe.parameters()),
                lr=lr, weight_decay=wd
            )

    for ep in range(N_EPOCHS):
        # Warmup-freeze transition
        if mode == 'warmup_freeze' and ep == warmup_epochs:
            # Unfreeze encoder with lower LR
            for p in model_ft.context_encoder.parameters():
                p.requires_grad = True
            # Add encoder params to optimizer with lower LR
            optim = torch.optim.Adam([
                {'params': probe.parameters(), 'lr': lr},
                {'params': model_ft.context_encoder.parameters(), 'lr': lr * 0.1},
            ], weight_decay=wd)
            print(f"  [warmup complete at ep={ep}] unfroze encoder with LR={lr*0.1:.1e}")

        # Train
        if mode == 'frozen' or (mode == 'warmup_freeze' and ep < warmup_epochs):
            model_ft.eval()
        else:
            model_ft.train()
        probe.train()

        for past, mask, rul in tr:
            past, mask, rul = past.to(DEVICE), mask.to(DEVICE), rul.to(DEVICE)
            h_past = model_ft.context_encoder(past, mask)
            pred = probe(h_past)
            loss = F.mse_loss(pred.squeeze(), rul.float())
            optim.zero_grad(); loss.backward(); optim.step()

        # Validate
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

    # Load best
    model_ft.context_encoder.load_state_dict(best_encoder_state)
    probe.load_state_dict(best_probe_state)

    test_rmse = eval_test_rmse(model_ft, probe, te)
    return test_rmse, best_val, ep + 1


# Load data and prepare engines - load_cmapss_subset already does train/val split
train_eng_all = data['train_engines']
val_eng = data['val_engines']
test_eng = data['test_engines']
test_rul = data['test_rul']

print(f"Train engines (all): {len(train_eng_all)}")
print(f"Val engines: {len(val_eng)}")

# Run budget 100% experiments
train_eng = train_eng_all  # 100% budget

VARIANTS = {
    'e2e_baseline': dict(mode='e2e', lr=1e-4, wd=0.0, warmup_epochs=0),
    'e2e_low_lr': dict(mode='e2e_low_lr', lr=5e-5, wd=0.0, warmup_epochs=0),
    'e2e_wd': dict(mode='e2e_wd', lr=1e-4, wd=1e-4, warmup_epochs=0),
    'warmup_freeze': dict(mode='warmup_freeze', lr=1e-4, wd=0.0, warmup_epochs=20),
}

results = {}
for variant_name, variant_cfg in VARIANTS.items():
    print(f"\n{'='*60}")
    print(f"Variant: {variant_name}")
    print(f"Config: {variant_cfg}")
    print(f"{'='*60}")

    seed_rmses = []
    for seed in SEEDS:
        t0 = time.time()
        test_rmse, val_rmse, n_epochs = run_finetune(
            train_eng, val_eng, test_eng, test_rul,
            seed=seed, **variant_cfg
        )
        elapsed = time.time() - t0
        seed_rmses.append(test_rmse)
        print(f"  seed={seed}: test={test_rmse:.3f}, val={val_rmse:.3f}, ep={n_epochs}, t={elapsed:.0f}s")

    mean_rmse = float(np.mean(seed_rmses))
    std_rmse = float(np.std(seed_rmses))
    print(f"\n  {variant_name}: {mean_rmse:.3f} +/- {std_rmse:.3f}")

    results[variant_name] = {
        'mean_rmse': mean_rmse,
        'std_rmse': std_rmse,
        'per_seed_rmse': [float(r) for r in seed_rmses],
        'config': variant_cfg,
    }

    # Intermediate save
    with open(os.path.join(V13_DIR, 'finetune_schedule_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Saved intermediate results")

# Final summary
print("\n" + "="*60)
print("FINAL SUMMARY")
print("="*60)
print(f"{'Variant':>20} | {'RMSE Mean':>10} +/- {'Std':>6}")
print("-"*45)
baseline = results.get('e2e_baseline', {}).get('mean_rmse', 14.23)
for name, r in sorted(results.items(), key=lambda x: x[1]['mean_rmse']):
    delta = r['mean_rmse'] - baseline
    print(f"{name:>20} | {r['mean_rmse']:>10.3f} +/- {r['std_rmse']:>4.3f}  (delta vs baseline: {delta:+.3f})")

print(f"\nV12 reference: E2E baseline = 14.23 +/- 0.39")
print(f"STAR reference: 12.19 +/- 0.55 (FD001, 5 seeds)")
