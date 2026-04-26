"""
Phase 0b: STAR FD004 Sweep

Complete the 5-seed FD004 sweep with FD004-specific hyperparams.
Config: bs=64, w=64, scales=4, dm=64, nh=4

FD004 is the hardest subset (6 operating conditions, 2 fault modes).
Paper target: RMSE=15.87, Score=1449.

Output: experiments/v13/star_fd004_results.json
"""

import sys
import json
import time
import numpy as np
import torch
from pathlib import Path

STAR_DIR = Path('/home/sagemaker-user/IndustrialJEPA/paper-replications/star')
V13_DIR = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/experiments/v13')
sys.path.insert(0, str(STAR_DIR))

from data_utils import prepare_data
from models import build_model, count_parameters
from train_utils import train, evaluate, RUL_CAP

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

SEEDS = [42, 123, 456, 789, 1024]
SUBSET = 'FD004'
MAX_EPOCHS = 200
PATIENCE = 20

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Phase 0b: STAR FD004 Sweep")
print(f"Device: {DEVICE}")
t0_global = time.time()

PAPER_TARGET = {'rmse': 15.87, 'score': 1449}

all_rmse = []
all_score = []
seed_results = []

for seed in SEEDS:
    print(f"\n--- FD004 seed={seed} ---")
    t0 = time.time()

    torch.manual_seed(seed)
    np.random.seed(seed)

    # FD004-specific config
    data = prepare_data(
        SUBSET,
        window_length=64,
        batch_size=64,
        val_fraction=0.15,
        seed=seed,
        rul_cap=int(RUL_CAP),
    )

    print(f"  Train engines: {data['n_train_engines']}, windows: {data['n_train_windows']}")

    # FD004 uses d_model=256 per CONFIGS in run_experiments.py
    model = build_model(SUBSET, DEVICE)
    n_params = count_parameters(model)
    print(f"  Model: {n_params:,} params")

    # W&B
    run = None
    if HAS_WANDB:
        try:
            run = wandb.init(
                project="industrialjepa",
                name=f"v13-phase0b-star-fd004-s{seed}",
                tags=["v13-phase0b-star-fd004"],
                config={
                    'phase': '0b',
                    'method': 'STAR',
                    'subset': SUBSET,
                    'seed': seed,
                    'n_params': n_params,
                    'window_length': 64,
                    'n_scales': 4,
                    'd_model': 256,
                    'n_heads': 4,
                },
                reinit=True,
            )
        except Exception as e:
            print(f"  W&B init failed: {e}")

    train_info = train(
        model=model,
        train_loader=data['train_loader'],
        val_loader=data['val_loader'],
        lr=0.0002,
        max_epochs=MAX_EPOCHS,
        patience=PATIENCE,
        device=DEVICE,
        verbose=True,
    )

    test_rmse, test_score, preds, trues = evaluate(model, data['test_loader'], DEVICE)
    wall_time = time.time() - t0

    print(f"  RESULT: RMSE={test_rmse:.3f}, Score={test_score:.1f} "
          f"(paper: RMSE={PAPER_TARGET['rmse']}, Score={PAPER_TARGET['score']})")
    print(f"  Wall time: {wall_time:.0f}s")

    if run is not None:
        try:
            wandb.log({
                'test_rmse': test_rmse,
                'test_score': test_score,
                'best_val_rmse': train_info['best_val_rmse'],
                'best_epoch': train_info['best_epoch'],
                'wall_time_s': wall_time,
            })
            wandb.finish()
        except Exception:
            pass

    all_rmse.append(test_rmse)
    all_score.append(test_score)
    seed_results.append({
        'seed': seed,
        'test_rmse': test_rmse,
        'test_score': test_score,
        'best_val_rmse': train_info['best_val_rmse'],
        'best_epoch': train_info['best_epoch'],
        'epochs_run': train_info['epochs_run'],
        'n_params': n_params,
        'wall_time_s': wall_time,
        'preds': preds.tolist(),
        'trues': trues.tolist(),
    })

    # Save intermediate
    intermediate = {
        'subset': SUBSET,
        'seeds_completed': [r['seed'] for r in seed_results],
        'per_seed': seed_results,
        'all_rmse': all_rmse,
    }
    with open(V13_DIR / 'star_fd004_results.json', 'w') as f:
        json.dump(intermediate, f, indent=2)

rmse_mean = float(np.mean(all_rmse))
rmse_std = float(np.std(all_rmse))
score_mean = float(np.mean(all_score))
score_std = float(np.std(all_score))

print(f"\n{'='*60}")
print(f"FD004 SUMMARY (5 seeds):")
print(f"  RMSE: {rmse_mean:.3f} +/- {rmse_std:.3f} (paper: {PAPER_TARGET['rmse']})")
print(f"  Score: {score_mean:.1f} +/- {score_std:.1f} (paper: {PAPER_TARGET['score']})")
rmse_gap = 100 * (rmse_mean - PAPER_TARGET['rmse']) / PAPER_TARGET['rmse']
print(f"  Gap vs paper: {rmse_gap:+.1f}%")
print(f"{'='*60}")

results = {
    'subset': SUBSET,
    'rmse_mean': rmse_mean,
    'rmse_std': rmse_std,
    'score_mean': score_mean,
    'score_std': score_std,
    'paper_rmse': PAPER_TARGET['rmse'],
    'paper_score': PAPER_TARGET['score'],
    'rmse_gap_pct': rmse_gap,
    'per_seed': seed_results,
    'all_rmse': all_rmse,
    'all_score': all_score,
    'wall_time_total_s': time.time() - t0_global,
}

out_path = V13_DIR / 'star_fd004_results.json'
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {out_path}")
print(f"Total wall time: {time.time()-t0_global:.1f}s")
