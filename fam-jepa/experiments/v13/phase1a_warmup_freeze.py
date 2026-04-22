"""
Phase 1a: Warmup-Freeze Fine-Tuning

Freeze encoder for first 20 epochs (probe-only warmup), then unfreeze
for E2E with standard LR=1e-4. Rationale: letting the probe converge first
prevents early gradient noise from wrecking pretrained encoder weights.

5 seeds, 100% labels, FD001. Compare vs standard E2E (14.23).

NOTE: Prior v13 session tested warmup_freeze with lr=1e-5 for encoder post-unfreeze.
This run uses lr=1e-4 (standard) which is the version specified in the prompt.

Output: experiments/v13/warmup_freeze_results.json
"""

import sys
import json
import time
import copy
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader

V11_DIR = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/experiments/v11')
V13_DIR = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/experiments/v13')
sys.path.insert(0, str(V11_DIR))

from data_utils import (
    load_cmapss_subset, N_SENSORS, RUL_CAP,
    CMAPSSFinetuneDataset, CMAPSSTestDataset, collate_finetune, collate_test
)
from models import TrajectoryJEPA, RULProbe

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PRETRAIN_CKPT = V11_DIR / 'best_pretrain_L1_v2.pt'
SEEDS = [42, 123, 456, 789, 1024]
WARMUP_EPOCHS = 20
TOTAL_EPOCHS = 100
PATIENCE = 20
BATCH_SIZE = 16

print(f"Phase 1a: Warmup-Freeze Fine-Tuning")
print(f"Device: {DEVICE}")
print(f"Warmup epochs (frozen): {WARMUP_EPOCHS}")
print(f"Total epochs: {TOTAL_EPOCHS}")
t0_global = time.time()

# Load data
data = load_cmapss_subset('FD001')
train_engines = data['train_engines']
val_engines = data['val_engines']
test_engines = data['test_engines']
test_rul = data['test_rul']


def run_warmup_freeze(seed):
    """
    Phase 1: Freeze encoder, train probe for WARMUP_EPOCHS
    Phase 2: Unfreeze encoder, train all with LR=1e-4 (standard E2E)
    """
    model = TrajectoryJEPA(
        n_sensors=N_SENSORS, patch_length=1, d_model=256, n_heads=4, n_layers=2,
        d_ff=512, dropout=0.1, ema_momentum=0.996, predictor_hidden=256
    ).to(DEVICE)
    model.load_state_dict(torch.load(str(PRETRAIN_CKPT), map_location=DEVICE))
    probe = RULProbe(256).to(DEVICE)

    torch.manual_seed(seed)
    np.random.seed(seed)

    train_ds = CMAPSSFinetuneDataset(train_engines, n_cuts_per_engine=5, seed=seed)
    val_ds = CMAPSSFinetuneDataset(val_engines, use_last_only=True)
    test_ds = CMAPSSTestDataset(test_engines, test_rul)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_finetune)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_finetune)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_test)

    best_val_rmse = float('inf')
    best_probe_state = None
    best_encoder_state = None
    no_improve = 0
    phase = 'frozen'

    # Start with frozen encoder, probe-only optimizer
    for p in model.parameters():
        p.requires_grad = False
    optimizer = torch.optim.Adam(probe.parameters(), lr=1e-3)

    for epoch in range(1, TOTAL_EPOCHS + 1):
        # Switch phase at WARMUP_EPOCHS
        if epoch == WARMUP_EPOCHS + 1:
            phase = 'e2e'
            # Unfreeze context encoder
            for p in model.context_encoder.parameters():
                p.requires_grad = True
            # New optimizer with all params, standard E2E LR
            optimizer = torch.optim.Adam(
                list(model.context_encoder.parameters()) +
                list(model.predictor.parameters()) +
                list(probe.parameters()),
                lr=1e-4  # Standard E2E LR (key difference from prior run that used 1e-5)
            )
            no_improve = 0  # Reset patience for E2E phase

        if phase == 'frozen':
            model.eval()
        else:
            model.train()
        probe.train()

        for past, mask, rul in train_loader:
            past, mask, rul = past.to(DEVICE), mask.to(DEVICE), rul.to(DEVICE)
            optimizer.zero_grad()
            if phase == 'frozen':
                with torch.no_grad():
                    h = model.encode_past(past, mask)
            else:
                h = model.encode_past(past, mask)
            pred = probe(h)
            loss = F.mse_loss(pred, rul)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in list(model.parameters()) + list(probe.parameters()) if p.requires_grad], 1.0
            )
            optimizer.step()

        # Validation
        model.eval()
        probe.eval()
        preds, targets = [], []
        with torch.no_grad():
            for past, mask, rul in val_loader:
                past, mask = past.to(DEVICE), mask.to(DEVICE)
                h = model.encode_past(past, mask)
                preds.append(probe(h).cpu().numpy())
                targets.append(rul.numpy())
        val_rmse = float(np.sqrt(np.mean((np.concatenate(preds)*RUL_CAP - np.concatenate(targets)*RUL_CAP)**2)))

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_probe_state = copy.deepcopy(probe.state_dict())
            best_encoder_state = copy.deepcopy(model.context_encoder.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE and phase == 'e2e':
                break

    # Restore best
    probe.load_state_dict(best_probe_state)
    model.context_encoder.load_state_dict(best_encoder_state)
    model.eval()
    probe.eval()

    # Test
    preds, targets = [], []
    with torch.no_grad():
        for past, mask, rul_gt in test_loader:
            past, mask = past.to(DEVICE), mask.to(DEVICE)
            h = model.encode_past(past, mask)
            preds.append(probe(h).cpu().numpy() * RUL_CAP)
            targets.append(rul_gt.numpy())
    test_rmse = float(np.sqrt(np.mean((np.concatenate(preds) - np.concatenate(targets))**2)))
    return test_rmse, best_val_rmse, epoch


# Also run standard E2E baseline for direct comparison
def run_standard_e2e(seed):
    """Standard E2E (no warmup freeze) for direct comparison."""
    model = TrajectoryJEPA(
        n_sensors=N_SENSORS, patch_length=1, d_model=256, n_heads=4, n_layers=2,
        d_ff=512, dropout=0.1, ema_momentum=0.996, predictor_hidden=256
    ).to(DEVICE)
    model.load_state_dict(torch.load(str(PRETRAIN_CKPT), map_location=DEVICE))
    probe = RULProbe(256).to(DEVICE)

    for p in model.context_encoder.parameters():
        p.requires_grad = True
    optimizer = torch.optim.Adam(
        list(model.context_encoder.parameters()) +
        list(model.predictor.parameters()) +
        list(probe.parameters()),
        lr=1e-4
    )

    torch.manual_seed(seed)
    np.random.seed(seed)

    train_ds = CMAPSSFinetuneDataset(train_engines, n_cuts_per_engine=5, seed=seed)
    val_ds = CMAPSSFinetuneDataset(val_engines, use_last_only=True)
    test_ds = CMAPSSTestDataset(test_engines, test_rul)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_finetune)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_finetune)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_test)

    best_val, best_ps, best_es, ni = float('inf'), None, None, 0
    for epoch in range(1, TOTAL_EPOCHS + 1):
        model.train(); probe.train()
        for past, mask, rul in train_loader:
            past, mask, rul = past.to(DEVICE), mask.to(DEVICE), rul.to(DEVICE)
            optimizer.zero_grad()
            h = model.encode_past(past, mask)
            loss = F.mse_loss(probe(h), rul)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(probe.parameters()), 1.0)
            optimizer.step()
        model.eval(); probe.eval()
        preds, targets = [], []
        with torch.no_grad():
            for past, mask, rul in val_loader:
                past, mask = past.to(DEVICE), mask.to(DEVICE)
                preds.append(probe(model.encode_past(past, mask)).cpu().numpy())
                targets.append(rul.numpy())
        vr = float(np.sqrt(np.mean((np.concatenate(preds)*RUL_CAP - np.concatenate(targets)*RUL_CAP)**2)))
        if vr < best_val:
            best_val = vr; best_ps = copy.deepcopy(probe.state_dict()); best_es = copy.deepcopy(model.context_encoder.state_dict()); ni = 0
        else:
            ni += 1
            if ni >= PATIENCE: break
    probe.load_state_dict(best_ps); model.context_encoder.load_state_dict(best_es)
    model.eval(); probe.eval()
    preds, targets = [], []
    with torch.no_grad():
        for past, mask, rul_gt in test_loader:
            past, mask = past.to(DEVICE), mask.to(DEVICE)
            preds.append(probe(model.encode_past(past, mask)).cpu().numpy() * RUL_CAP)
            targets.append(rul_gt.numpy())
    return float(np.sqrt(np.mean((np.concatenate(preds) - np.concatenate(targets))**2)))


results = {'warmup_freeze': [], 'standard_e2e': []}

for seed in SEEDS:
    print(f"\n--- seed={seed} ---")
    t0 = time.time()

    # W&B
    run = None
    if HAS_WANDB:
        try:
            run = wandb.init(
                project="industrialjepa", name=f"v13-phase1a-warmup-s{seed}",
                tags=["v13-phase1a-warmup-freeze"],
                config={'phase': '1a', 'seed': seed, 'warmup_epochs': WARMUP_EPOCHS},
                reinit=True,
            )
        except Exception:
            pass

    rmse_wf, val_wf, epochs_wf = run_warmup_freeze(seed)
    rmse_e2e = run_standard_e2e(seed)

    print(f"  Warmup-freeze: RMSE={rmse_wf:.3f} (val={val_wf:.2f}, epochs={epochs_wf})")
    print(f"  Standard E2E:  RMSE={rmse_e2e:.3f}")
    print(f"  Delta: {rmse_wf - rmse_e2e:+.3f}")

    if run is not None:
        try:
            wandb.log({'warmup_rmse': rmse_wf, 'standard_rmse': rmse_e2e, 'delta': rmse_wf - rmse_e2e})
            wandb.finish()
        except Exception: pass

    results['warmup_freeze'].append({'seed': seed, 'test_rmse': rmse_wf, 'val_rmse': val_wf, 'epochs': epochs_wf})
    results['standard_e2e'].append({'seed': seed, 'test_rmse': rmse_e2e})

# Summary
wf_rmses = [r['test_rmse'] for r in results['warmup_freeze']]
e2e_rmses = [r['test_rmse'] for r in results['standard_e2e']]

results['summary'] = {
    'warmup_freeze_mean': float(np.mean(wf_rmses)),
    'warmup_freeze_std': float(np.std(wf_rmses)),
    'standard_e2e_mean': float(np.mean(e2e_rmses)),
    'standard_e2e_std': float(np.std(e2e_rmses)),
    'delta_mean': float(np.mean(wf_rmses) - np.mean(e2e_rmses)),
    'improved': float(np.mean(wf_rmses)) < float(np.mean(e2e_rmses)),
    'wall_time_total_s': time.time() - t0_global,
}

print(f"\n{'='*60}")
print(f"WARMUP-FREEZE RESULTS")
print(f"{'='*60}")
print(f"Warmup-freeze: {results['summary']['warmup_freeze_mean']:.3f} +/- {results['summary']['warmup_freeze_std']:.3f}")
print(f"Standard E2E:  {results['summary']['standard_e2e_mean']:.3f} +/- {results['summary']['standard_e2e_std']:.3f}")
print(f"Delta: {results['summary']['delta_mean']:+.3f}")
print(f"{'IMPROVED' if results['summary']['improved'] else 'NO IMPROVEMENT'}")

out_path = V13_DIR / 'warmup_freeze_results.json'
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {out_path}")
print(f"Total wall time: {time.time()-t0_global:.1f}s")
