"""
Phase 1d: Deeper Architecture (V4: d=256, L=4)

Scale from 2 to 4 transformer layers (~3.5M params).
Requires new pretraining + fine-tuning. Compare frozen and E2E at 100%.

Kill criterion: if V4 frozen doesn't improve over V2 frozen (17.81),
depth doesn't help (consistent with v11 finding that width > depth).

Output: experiments/v13/deeper_architecture_results.json
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

# V4 architecture: deeper (4 layers vs 2)
V4_N_LAYERS = 4
V4_D_FF = 512  # Keep same width

print(f"Phase 1d: Deeper Architecture (V4: d=256, L={V4_N_LAYERS})")
print(f"Device: {DEVICE}")
t0_global = time.time()

# Load data
data = load_cmapss_subset('FD001')
train_engines = data['train_engines']
val_engines = data['val_engines']
test_engines = data['test_engines']
test_rul = data['test_rul']

# Count params
model_v4 = TrajectoryJEPA(
    n_sensors=N_SENSORS, patch_length=1, d_model=256, n_heads=4,
    n_layers=V4_N_LAYERS, d_ff=V4_D_FF, dropout=0.1,
    ema_momentum=0.996, predictor_hidden=256
).to(DEVICE)
n_params = sum(p.numel() for p in model_v4.parameters() if p.requires_grad)
print(f"V4 parameters: {n_params:,}")

# V2 reference for comparison
model_v2_ref = TrajectoryJEPA(
    n_sensors=N_SENSORS, patch_length=1, d_model=256, n_heads=4,
    n_layers=2, d_ff=512, dropout=0.1, ema_momentum=0.996, predictor_hidden=256
)
n_params_v2 = sum(p.numel() for p in model_v2_ref.parameters() if p.requires_grad)
print(f"V2 parameters: {n_params_v2:,}")
print(f"Param ratio V4/V2: {n_params/n_params_v2:.2f}x")
del model_v2_ref

# ============================================================
# Pretrain V4
# ============================================================
print(f"\n{'='*60}")
print(f"PRETRAINING V4 (d=256, L={V4_N_LAYERS})")
print(f"{'='*60}")

optimizer = torch.optim.AdamW(
    [p for p in model_v4.parameters() if p.requires_grad],
    lr=PRETRAIN_LR, weight_decay=PRETRAIN_WD
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, PRETRAIN_EPOCHS)

best_probe_rmse = float('inf')
best_state = None
history = {'loss': [], 'pred_loss': [], 'var_loss': [], 'probe_rmse': [], 'probe_epochs': []}

run = None
if HAS_WANDB:
    try:
        run = wandb.init(
            project="industrialjepa",
            name="v13-phase1d-pretrain-v4-L4",
            tags=["v13-phase1d-pretrain"],
            config={
                'phase': '1d',
                'type': 'pretrain',
                'n_layers': V4_N_LAYERS,
                'd_model': 256,
                'n_params': n_params,
                'n_epochs': PRETRAIN_EPOCHS,
            },
            reinit=True,
        )
    except Exception as e:
        print(f"W&B init failed: {e}")

for epoch in range(1, PRETRAIN_EPOCHS + 1):
    train_ds = CMAPSSPretrainDataset(
        train_engines, n_cuts_per_engine=20,
        min_past=10, min_horizon=5, max_horizon=30,  # Same horizon as V2
        seed=epoch
    )
    train_loader = DataLoader(train_ds, batch_size=PRETRAIN_BATCH, shuffle=True,
                              collate_fn=collate_pretrain, num_workers=0)

    metrics = pretrain_one_epoch(model_v4, train_loader, optimizer, lambda_var=0.01)
    history['loss'].append(metrics['loss'])
    history['pred_loss'].append(metrics['pred_loss'])
    history['var_loss'].append(metrics['var_loss'])
    scheduler.step()

    if epoch % PROBE_EVERY == 0 or epoch == 1:
        probe_rmse = linear_probe_rmse(model_v4, train_engines, val_engines)
        history['probe_rmse'].append(probe_rmse)
        history['probe_epochs'].append(epoch)

        if probe_rmse < best_probe_rmse:
            best_probe_rmse = probe_rmse
            best_state = copy.deepcopy(model_v4.state_dict())

        if run is not None:
            try:
                wandb.log({
                    'epoch': epoch,
                    'loss': metrics['loss'],
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

if best_state is not None:
    model_v4.load_state_dict(best_state)
    ckpt_path = V13_DIR / 'best_pretrain_v4_L4.pt'
    torch.save(best_state, str(ckpt_path))
    print(f"\nSaved V4 checkpoint: {ckpt_path}")
    print(f"Best probe RMSE: {best_probe_rmse:.2f}")

pretrain_time = time.time() - t0_global

# Kill criterion
BASELINE_FROZEN_RMSE = 17.81
if best_probe_rmse > BASELINE_FROZEN_RMSE + 2:
    print(f"\n*** KILL CRITERION: V4 probe RMSE {best_probe_rmse:.2f} >> V2 baseline {BASELINE_FROZEN_RMSE}")
    print(f"*** Depth doesn't help. Consistent with v11 finding.")
    results = {
        'killed': True,
        'reason': f"V4 probe RMSE {best_probe_rmse:.2f} worse than V2 baseline {BASELINE_FROZEN_RMSE}",
        'best_probe_rmse': best_probe_rmse,
        'n_params_v4': n_params,
        'pretrain_time_s': pretrain_time,
    }
    with open(V13_DIR / 'deeper_architecture_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    sys.exit(0)

# ============================================================
# Fine-tune V4: frozen + E2E at 100%
# ============================================================
print(f"\n{'='*60}")
print(f"FINE-TUNING V4")
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
                name=f"v13-phase1d-finetune-v4-s{seed}",
                tags=["v13-phase1d-finetune"],
                config={'phase': '1d', 'type': 'finetune', 'seed': seed, 'n_layers': V4_N_LAYERS},
                reinit=True,
            )
        except Exception:
            pass

    # Frozen
    model_fr = TrajectoryJEPA(
        n_sensors=N_SENSORS, patch_length=1, d_model=256, n_heads=4,
        n_layers=V4_N_LAYERS, d_ff=V4_D_FF, dropout=0.1,
        ema_momentum=0.996, predictor_hidden=256
    ).to(DEVICE)
    model_fr.load_state_dict(best_state)
    res_fr = finetune(model_fr, train_engines, val_engines, test_engines, test_rul,
                       mode='frozen', seed=seed, verbose=False)
    frozen_rmses.append(res_fr['test_rmse'])

    # E2E
    model_e2e = TrajectoryJEPA(
        n_sensors=N_SENSORS, patch_length=1, d_model=256, n_heads=4,
        n_layers=V4_N_LAYERS, d_ff=V4_D_FF, dropout=0.1,
        ema_momentum=0.996, predictor_hidden=256
    ).to(DEVICE)
    model_e2e.load_state_dict(best_state)
    res_e2e = finetune(model_e2e, train_engines, val_engines, test_engines, test_rul,
                        mode='e2e', seed=seed, verbose=False)
    e2e_rmses.append(res_e2e['test_rmse'])

    print(f"  Frozen: {res_fr['test_rmse']:.3f}, E2E: {res_e2e['test_rmse']:.3f}")

    if run is not None:
        try:
            wandb.log({'frozen_rmse': res_fr['test_rmse'], 'e2e_rmse': res_e2e['test_rmse']})
            wandb.finish()
        except Exception:
            pass

# Results
fr_mean = float(np.mean(frozen_rmses))
fr_std = float(np.std(frozen_rmses))
e2e_mean = float(np.mean(e2e_rmses))
e2e_std = float(np.std(e2e_rmses))

print(f"\n{'='*60}")
print(f"V4 (d=256, L=4) RESULTS")
print(f"{'='*60}")
print(f"V4 Frozen: {fr_mean:.3f} +/- {fr_std:.3f} (V2 baseline: ~17.81)")
print(f"V4 E2E:    {e2e_mean:.3f} +/- {e2e_std:.3f} (V2 baseline: ~14.23)")
print(f"V4 params: {n_params:,}")

improved = e2e_mean < 14.23 - 0.5
print(f"\n{'IMPROVED' if improved else 'NO IMPROVEMENT'}")

results = {
    'killed': False,
    'n_layers': V4_N_LAYERS,
    'd_model': 256,
    'n_params_v4': n_params,
    'best_probe_rmse': best_probe_rmse,
    'frozen': {
        'mean': fr_mean, 'std': fr_std, 'all': frozen_rmses,
        'baseline_v2': 17.81,
    },
    'e2e': {
        'mean': e2e_mean, 'std': e2e_std, 'all': e2e_rmses,
        'baseline_v2': 14.23,
    },
    'improved': improved,
    'pretrain_history': history,
    'pretrain_time_s': pretrain_time,
    'total_time_s': time.time() - t0_global,
}

out_path = V13_DIR / 'deeper_architecture_results.json'
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {out_path}")
print(f"Total wall time: {time.time()-t0_global:.1f}s")
