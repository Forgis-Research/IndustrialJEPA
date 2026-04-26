"""
Phase 1c: Longer Prediction Horizon

Pretrain from scratch with k in [5, 50] (vs current [5, 30]).
Predicting further ahead forces the encoder to learn slower dynamics.
Requires new pretraining run (200 epochs) + full fine-tuning sweep.

Kill criterion: if probe RMSE after new pretraining doesn't improve vs
current 19.22 (V11 frozen RMSE ~17.81), horizon isn't the bottleneck.

Output: experiments/v13/longer_horizon_results.json
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
    CMAPSSPretrainDataset, CMAPSSFinetuneDataset, CMAPSSTestDataset,
    collate_pretrain, collate_finetune, collate_test
)
from models import TrajectoryJEPA, RULProbe, trajectory_jepa_loss
from train_utils import (
    pretrain_one_epoch, linear_probe_rmse, finetune, DEVICE
)

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

SEEDS = [42, 123, 456, 789, 1024]
PRETRAIN_EPOCHS = 200
PRETRAIN_LR = 3e-4
PRETRAIN_WD = 0.01
PRETRAIN_BATCH = 8
PROBE_EVERY = 10

print(f"Phase 1c: Longer Prediction Horizon")
print(f"Device: {DEVICE}")
t0_global = time.time()

# Load data
data = load_cmapss_subset('FD001')
train_engines = data['train_engines']
val_engines = data['val_engines']
test_engines = data['test_engines']
test_rul = data['test_rul']

# ============================================================
# Pretrain with k in [5, 50] (vs baseline [5, 30])
# ============================================================
print(f"\n{'='*60}")
print(f"PRETRAINING with max_horizon=50 (vs baseline max_horizon=30)")
print(f"{'='*60}")

model = TrajectoryJEPA(
    n_sensors=N_SENSORS, patch_length=1, d_model=256, n_heads=4, n_layers=2,
    d_ff=512, dropout=0.1, ema_momentum=0.996, predictor_hidden=256
).to(DEVICE)

optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=PRETRAIN_LR, weight_decay=PRETRAIN_WD
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, PRETRAIN_EPOCHS)

best_probe_rmse = float('inf')
best_state = None
history = {'loss': [], 'pred_loss': [], 'var_loss': [], 'probe_rmse': [], 'probe_epochs': []}

# W&B for pretraining
run = None
if HAS_WANDB:
    try:
        run = wandb.init(
            project="industrialjepa",
            name="v13-phase1c-pretrain-horizon50",
            tags=["v13-phase1c-pretrain"],
            config={
                'phase': '1c',
                'type': 'pretrain',
                'max_horizon': 50,
                'min_horizon': 5,
                'n_epochs': PRETRAIN_EPOCHS,
                'd_model': 256,
                'n_layers': 2,
            },
            reinit=True,
        )
    except Exception as e:
        print(f"W&B init failed: {e}")

for epoch in range(1, PRETRAIN_EPOCHS + 1):
    # Rebuild dataset each epoch with different random cuts
    train_ds = CMAPSSPretrainDataset(
        train_engines, n_cuts_per_engine=20,
        min_past=10, min_horizon=5, max_horizon=50,  # KEY CHANGE: max_horizon=50
        seed=epoch
    )
    train_loader = DataLoader(train_ds, batch_size=PRETRAIN_BATCH, shuffle=True,
                              collate_fn=collate_pretrain, num_workers=0)

    metrics = pretrain_one_epoch(model, train_loader, optimizer, lambda_var=0.01)
    history['loss'].append(metrics['loss'])
    history['pred_loss'].append(metrics['pred_loss'])
    history['var_loss'].append(metrics['var_loss'])
    scheduler.step()

    if epoch % PROBE_EVERY == 0 or epoch == 1:
        probe_rmse = linear_probe_rmse(model, train_engines, val_engines)
        history['probe_rmse'].append(probe_rmse)
        history['probe_epochs'].append(epoch)

        if probe_rmse < best_probe_rmse:
            best_probe_rmse = probe_rmse
            best_state = copy.deepcopy(model.state_dict())

        if run is not None:
            try:
                wandb.log({
                    'epoch': epoch,
                    'loss': metrics['loss'],
                    'pred_loss': metrics['pred_loss'],
                    'var_loss': metrics['var_loss'],
                    'probe_rmse': probe_rmse,
                    'best_probe_rmse': best_probe_rmse,
                })
            except Exception:
                pass

        print(f"Ep {epoch:3d} | loss={metrics['loss']:.4f} pred={metrics['pred_loss']:.4f} "
              f"var={metrics['var_loss']:.4f} | probe={probe_rmse:.2f} (best={best_probe_rmse:.2f})")

if run is not None:
    try:
        wandb.finish()
    except Exception:
        pass

# Save checkpoint
if best_state is not None:
    model.load_state_dict(best_state)
    ckpt_path = V13_DIR / 'best_pretrain_horizon50.pt'
    torch.save(best_state, str(ckpt_path))
    print(f"\nSaved best pretrained model to {ckpt_path}")
    print(f"Best probe RMSE during pretraining: {best_probe_rmse:.2f}")

pretrain_time = time.time() - t0_global
print(f"Pretraining wall time: {pretrain_time:.1f}s")

# Kill criterion check
BASELINE_FROZEN_RMSE = 17.81  # V11 frozen RMSE
if best_probe_rmse > BASELINE_FROZEN_RMSE + 2:
    print(f"\n*** KILL CRITERION: probe RMSE {best_probe_rmse:.2f} >> baseline {BASELINE_FROZEN_RMSE}")
    print(f"*** Longer horizon does NOT improve pretraining. Stopping early.")
    results = {
        'killed': True,
        'reason': f"Probe RMSE {best_probe_rmse:.2f} worse than baseline {BASELINE_FROZEN_RMSE}",
        'best_probe_rmse': best_probe_rmse,
        'baseline_frozen_rmse': BASELINE_FROZEN_RMSE,
        'pretrain_time_s': pretrain_time,
        'pretrain_history': {k: v for k, v in history.items() if k != 'embeddings'},
    }
    with open(V13_DIR / 'longer_horizon_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved.")
    sys.exit(0)

# ============================================================
# Fine-tune: frozen + E2E at 100%
# ============================================================
print(f"\n{'='*60}")
print(f"FINE-TUNING with horizon-50 pretrained model")
print(f"{'='*60}")

frozen_rmses = []
e2e_rmses = []

for seed in SEEDS:
    print(f"\n--- seed={seed} ---")

    run = None
    if HAS_WANDB:
        try:
            run = wandb.init(
                project="industrialjepa",
                name=f"v13-phase1c-finetune-s{seed}",
                tags=["v13-phase1c-finetune"],
                config={'phase': '1c', 'type': 'finetune', 'seed': seed, 'max_horizon': 50},
                reinit=True,
            )
        except Exception:
            pass

    # Load the horizon-50 pretrained model
    model_frozen = TrajectoryJEPA(
        n_sensors=N_SENSORS, patch_length=1, d_model=256, n_heads=4, n_layers=2,
        d_ff=512, dropout=0.1, ema_momentum=0.996, predictor_hidden=256
    ).to(DEVICE)
    model_frozen.load_state_dict(best_state)

    # Frozen probe
    res_frozen = finetune(
        model_frozen, train_engines, val_engines, test_engines, test_rul,
        mode='frozen', seed=seed, verbose=False
    )
    frozen_rmses.append(res_frozen['test_rmse'])
    print(f"  Frozen: {res_frozen['test_rmse']:.3f}")

    # E2E
    model_e2e = TrajectoryJEPA(
        n_sensors=N_SENSORS, patch_length=1, d_model=256, n_heads=4, n_layers=2,
        d_ff=512, dropout=0.1, ema_momentum=0.996, predictor_hidden=256
    ).to(DEVICE)
    model_e2e.load_state_dict(best_state)

    res_e2e = finetune(
        model_e2e, train_engines, val_engines, test_engines, test_rul,
        mode='e2e', seed=seed, verbose=False
    )
    e2e_rmses.append(res_e2e['test_rmse'])
    print(f"  E2E:    {res_e2e['test_rmse']:.3f}")

    if run is not None:
        try:
            wandb.log({'frozen_rmse': res_frozen['test_rmse'], 'e2e_rmse': res_e2e['test_rmse']})
            wandb.finish()
        except Exception:
            pass

# Results
frozen_mean = float(np.mean(frozen_rmses))
frozen_std = float(np.std(frozen_rmses))
e2e_mean = float(np.mean(e2e_rmses))
e2e_std = float(np.std(e2e_rmses))

print(f"\n{'='*60}")
print(f"LONGER HORIZON RESULTS (max_horizon=50 vs baseline max_horizon=30)")
print(f"{'='*60}")
print(f"Horizon-50 Frozen: {frozen_mean:.3f} +/- {frozen_std:.3f} (baseline: ~17.81)")
print(f"Horizon-50 E2E:    {e2e_mean:.3f} +/- {e2e_std:.3f} (baseline: ~14.23)")
print(f"Best probe RMSE during pretraining: {best_probe_rmse:.2f}")

improved = e2e_mean < 14.23 - 0.5  # Need > 0.5 RMSE improvement
print(f"\n{'IMPROVED' if improved else 'NO IMPROVEMENT'}")

results = {
    'killed': False,
    'max_horizon': 50,
    'baseline_max_horizon': 30,
    'best_probe_rmse': best_probe_rmse,
    'frozen': {
        'mean': frozen_mean, 'std': frozen_std, 'all': frozen_rmses,
        'baseline': 17.81,
    },
    'e2e': {
        'mean': e2e_mean, 'std': e2e_std, 'all': e2e_rmses,
        'baseline': 14.23,
    },
    'improved': improved,
    'pretrain_history': {k: v for k, v in history.items()},
    'pretrain_time_s': pretrain_time,
    'total_time_s': time.time() - t0_global,
}

out_path = V13_DIR / 'longer_horizon_results.json'
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {out_path}")
print(f"Total wall time: {time.time()-t0_global:.1f}s")
