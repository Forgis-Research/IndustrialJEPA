"""
Phase 0c: From-Scratch Ablation

THE MOST IMPORTANT EXPERIMENT IN V13.

Same V2 transformer encoder (d=256, L=2, same param count) + same linear probe
+ same E2E protocol (LR=1e-4, AdamW, patience=20) - but initialized from
random weights instead of the pretrained checkpoint.

Run at 4 label budgets: 100%, 20%, 10%, 5%. 5 seeds each.

Interpretation:
  delta > 3 RMSE -> pretraining does real work under E2E (strong SSL claim)
  delta 1-3 RMSE -> helps modestly (paper leads with frozen/H.I., not E2E)
  delta < 1 RMSE -> negligible (E2E is supervised learning in a transformer)

If the delta grows as labels decrease, that's the pitch: "pretraining matters
most when labels are scarce."

Output: experiments/v13/from_scratch_ablation.json
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
from train_utils import subsample_engines

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PRETRAIN_CKPT = V11_DIR / 'best_pretrain_L1_v2.pt'

SEEDS = [42, 123, 456, 789, 1024]
LABEL_BUDGETS = [1.0, 0.20, 0.10, 0.05]
N_EPOCHS = 100
PATIENCE = 20
BATCH_SIZE = 16

print(f"Phase 0c: From-Scratch Ablation")
print(f"Device: {DEVICE}")
print(f"Budgets: {LABEL_BUDGETS}")
print(f"Seeds: {SEEDS}")
t0_global = time.time()

# Load data
data = load_cmapss_subset('FD001')
all_train_engines = data['train_engines']
val_engines = data['val_engines']
test_engines = data['test_engines']
test_rul = data['test_rul']


def run_e2e_finetune(use_pretrained: bool, train_engines_subset, seed):
    """
    Run E2E fine-tuning from pretrained or random init.
    Returns test RMSE.
    """
    model = TrajectoryJEPA(
        n_sensors=N_SENSORS, patch_length=1, d_model=256, n_heads=4, n_layers=2,
        d_ff=512, dropout=0.1, ema_momentum=0.996, predictor_hidden=256
    ).to(DEVICE)

    if use_pretrained:
        model.load_state_dict(torch.load(str(PRETRAIN_CKPT), map_location=DEVICE))

    probe = RULProbe(256).to(DEVICE)

    # E2E: unfreeze context encoder
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

    train_ds = CMAPSSFinetuneDataset(train_engines_subset, n_cuts_per_engine=5, seed=seed)
    val_ds = CMAPSSFinetuneDataset(val_engines, use_last_only=True)
    test_ds = CMAPSSTestDataset(test_engines, test_rul)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_finetune)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_finetune)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_test)

    best_val_rmse = float('inf')
    best_probe_state = None
    best_encoder_state = None
    no_improve = 0

    for epoch in range(1, N_EPOCHS + 1):
        model.train()
        probe.train()

        for past, mask, rul in train_loader:
            past, mask, rul = past.to(DEVICE), mask.to(DEVICE), rul.to(DEVICE)
            optimizer.zero_grad()
            h = model.encode_past(past, mask)
            pred = probe(h)
            loss = F.mse_loss(pred, rul)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(model.context_encoder.parameters()) + list(probe.parameters()), 1.0
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
                pred = probe(h)
                preds.append(pred.cpu().numpy())
                targets.append(rul.numpy())
        preds = np.concatenate(preds) * RUL_CAP
        targets = np.concatenate(targets) * RUL_CAP
        val_rmse = float(np.sqrt(np.mean((preds - targets) ** 2)))

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_probe_state = copy.deepcopy(probe.state_dict())
            best_encoder_state = copy.deepcopy(model.context_encoder.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                break

    # Load best
    probe.load_state_dict(best_probe_state)
    model.context_encoder.load_state_dict(best_encoder_state)
    model.eval()
    probe.eval()

    # Test RMSE
    preds, targets = [], []
    with torch.no_grad():
        for past, mask, rul_gt in test_loader:
            past, mask = past.to(DEVICE), mask.to(DEVICE)
            h = model.encode_past(past, mask)
            pred = probe(h)
            preds.append(pred.cpu().numpy() * RUL_CAP)
            targets.append(rul_gt.numpy())
    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    test_rmse = float(np.sqrt(np.mean((preds - targets) ** 2)))

    return test_rmse, best_val_rmse


# Also run frozen probe for reference
def run_frozen_probe(train_engines_subset, seed):
    """Run frozen probe (no E2E) for reference."""
    model = TrajectoryJEPA(
        n_sensors=N_SENSORS, patch_length=1, d_model=256, n_heads=4, n_layers=2,
        d_ff=512, dropout=0.1, ema_momentum=0.996, predictor_hidden=256
    ).to(DEVICE)
    model.load_state_dict(torch.load(str(PRETRAIN_CKPT), map_location=DEVICE))
    model.eval()

    for p in model.parameters():
        p.requires_grad = False

    probe = RULProbe(256).to(DEVICE)
    optimizer = torch.optim.Adam(probe.parameters(), lr=1e-3)

    torch.manual_seed(seed)
    np.random.seed(seed)

    train_ds = CMAPSSFinetuneDataset(train_engines_subset, n_cuts_per_engine=5, seed=seed)
    val_ds = CMAPSSFinetuneDataset(val_engines, use_last_only=True)
    test_ds = CMAPSSTestDataset(test_engines, test_rul)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_finetune)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_finetune)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_test)

    best_val_rmse = float('inf')
    best_state = None
    no_improve = 0

    for epoch in range(1, N_EPOCHS + 1):
        probe.train()
        for past, mask, rul in train_loader:
            past, mask, rul = past.to(DEVICE), mask.to(DEVICE), rul.to(DEVICE)
            optimizer.zero_grad()
            with torch.no_grad():
                h = model.encode_past(past, mask)
            pred = probe(h)
            loss = F.mse_loss(pred, rul)
            loss.backward()
            optimizer.step()

        probe.eval()
        preds, targets = [], []
        with torch.no_grad():
            for past, mask, rul in val_loader:
                past, mask = past.to(DEVICE), mask.to(DEVICE)
                h = model.encode_past(past, mask)
                pv = probe(h)
                preds.append(pv.cpu().numpy())
                targets.append(rul.numpy())
        val_rmse = float(np.sqrt(np.mean((np.concatenate(preds)*RUL_CAP - np.concatenate(targets)*RUL_CAP)**2)))

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_state = copy.deepcopy(probe.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                break

    probe.load_state_dict(best_state)
    probe.eval()

    preds, targets = [], []
    with torch.no_grad():
        for past, mask, rul_gt in test_loader:
            past, mask = past.to(DEVICE), mask.to(DEVICE)
            h = model.encode_past(past, mask)
            pred = probe(h)
            preds.append(pred.cpu().numpy() * RUL_CAP)
            targets.append(rul_gt.numpy())
    test_rmse = float(np.sqrt(np.mean((np.concatenate(preds) - np.concatenate(targets))**2)))

    return test_rmse, best_val_rmse


all_results = {}

for budget in LABEL_BUDGETS:
    budget_key = f"{int(budget*100)}pct"
    print(f"\n{'='*60}")
    print(f"Label Budget: {int(budget*100)}%")
    print(f"{'='*60}")

    pretrained_rmses = []
    scratch_rmses = []
    frozen_rmses = []
    seed_details = []

    for seed in SEEDS:
        print(f"\n--- budget={int(budget*100)}%, seed={seed} ---")
        t0 = time.time()

        # Subsample engines
        if budget < 1.0:
            train_sub = subsample_engines(all_train_engines, budget, seed=seed)
        else:
            train_sub = all_train_engines

        n_engines = len(train_sub)
        print(f"  Engines: {n_engines}")

        # W&B
        run = None
        if HAS_WANDB:
            try:
                run = wandb.init(
                    project="industrialjepa",
                    name=f"v13-phase0c-scratch-{budget_key}-s{seed}",
                    tags=["v13-phase0c-from-scratch"],
                    config={
                        'phase': '0c',
                        'budget': budget,
                        'seed': seed,
                        'n_engines': n_engines,
                    },
                    reinit=True,
                )
            except Exception:
                pass

        # 1. Pretrained E2E
        print(f"  Running pretrained E2E...")
        rmse_pre, val_pre = run_e2e_finetune(use_pretrained=True, train_engines_subset=train_sub, seed=seed)
        print(f"    Pretrained E2E RMSE: {rmse_pre:.3f}")

        # 2. Random init E2E (from scratch)
        print(f"  Running from-scratch E2E...")
        rmse_scratch, val_scratch = run_e2e_finetune(use_pretrained=False, train_engines_subset=train_sub, seed=seed)
        print(f"    From-scratch E2E RMSE: {rmse_scratch:.3f}")

        # 3. Frozen probe (pretrained)
        print(f"  Running frozen probe...")
        rmse_frozen, val_frozen = run_frozen_probe(train_engines_subset=train_sub, seed=seed)
        print(f"    Frozen probe RMSE: {rmse_frozen:.3f}")

        delta = rmse_scratch - rmse_pre
        print(f"  Delta (scratch - pretrained): {delta:+.3f} RMSE")

        wall_time = time.time() - t0

        if run is not None:
            try:
                wandb.log({
                    'pretrained_e2e_rmse': rmse_pre,
                    'scratch_e2e_rmse': rmse_scratch,
                    'frozen_rmse': rmse_frozen,
                    'delta_scratch_minus_pretrained': delta,
                    'wall_time_s': wall_time,
                })
                wandb.finish()
            except Exception:
                pass

        pretrained_rmses.append(rmse_pre)
        scratch_rmses.append(rmse_scratch)
        frozen_rmses.append(rmse_frozen)
        seed_details.append({
            'seed': seed,
            'pretrained_e2e_rmse': rmse_pre,
            'scratch_e2e_rmse': rmse_scratch,
            'frozen_rmse': rmse_frozen,
            'delta': delta,
            'val_pre': val_pre,
            'val_scratch': val_scratch,
            'val_frozen': val_frozen,
            'n_engines': n_engines,
            'wall_time_s': wall_time,
        })

    pre_mean = float(np.mean(pretrained_rmses))
    pre_std = float(np.std(pretrained_rmses))
    scratch_mean = float(np.mean(scratch_rmses))
    scratch_std = float(np.std(scratch_rmses))
    frozen_mean = float(np.mean(frozen_rmses))
    frozen_std = float(np.std(frozen_rmses))
    delta_mean = scratch_mean - pre_mean

    print(f"\n  {budget_key} SUMMARY:")
    print(f"    Pretrained E2E: {pre_mean:.3f} +/- {pre_std:.3f}")
    print(f"    From-scratch E2E: {scratch_mean:.3f} +/- {scratch_std:.3f}")
    print(f"    Frozen probe: {frozen_mean:.3f} +/- {frozen_std:.3f}")
    print(f"    Delta (scratch - pretrained): {delta_mean:+.3f}")

    if delta_mean > 3:
        interp = "STRONG: Pretraining does real work under E2E"
    elif delta_mean > 1:
        interp = "MODERATE: Pretraining helps modestly"
    else:
        interp = "WEAK: Pretraining negligible under E2E"
    print(f"    Interpretation: {interp}")

    all_results[budget_key] = {
        'budget': budget,
        'pretrained_e2e': {'mean': pre_mean, 'std': pre_std, 'all': pretrained_rmses},
        'scratch_e2e': {'mean': scratch_mean, 'std': scratch_std, 'all': scratch_rmses},
        'frozen_probe': {'mean': frozen_mean, 'std': frozen_std, 'all': frozen_rmses},
        'delta_mean': delta_mean,
        'interpretation': interp,
        'per_seed': seed_details,
    }

    # Save intermediate
    with open(V13_DIR / 'from_scratch_ablation.json', 'w') as f:
        json.dump(all_results, f, indent=2)

# Final summary
print(f"\n{'='*60}")
print("FROM-SCRATCH ABLATION SUMMARY")
print(f"{'='*60}")
print(f"{'Budget':<10} {'Pretrained':>12} {'Scratch':>12} {'Frozen':>12} {'Delta':>8}")
for key, res in all_results.items():
    print(f"{key:<10} {res['pretrained_e2e']['mean']:>12.3f} {res['scratch_e2e']['mean']:>12.3f} "
          f"{res['frozen_probe']['mean']:>12.3f} {res['delta_mean']:>+8.3f}")

# Check if delta grows with lower labels
deltas = [(res['budget'], res['delta_mean']) for res in all_results.values()]
deltas.sort(key=lambda x: x[0], reverse=True)
if len(deltas) >= 2 and deltas[-1][1] > deltas[0][1]:
    pitch = "CONFIRMED: Delta grows as labels decrease. 'Pretraining matters most when labels are scarce.'"
else:
    pitch = "NOT CONFIRMED: Delta does not clearly grow with fewer labels."
print(f"\nLabel-scarcity pitch: {pitch}")

all_results['summary'] = {
    'pitch': pitch,
    'deltas_by_budget': deltas,
    'wall_time_total_s': time.time() - t0_global,
}

out_path = V13_DIR / 'from_scratch_ablation.json'
with open(out_path, 'w') as f:
    json.dump(all_results, f, indent=2)
print(f"\nFinal results saved to {out_path}")
print(f"Total wall time: {time.time()-t0_global:.1f}s")
