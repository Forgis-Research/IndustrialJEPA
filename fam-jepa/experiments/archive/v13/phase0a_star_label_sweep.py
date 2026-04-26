"""
Phase 0a: STAR Label-Efficiency Sweep

Run STAR on FD001 with reduced label budgets: 50%, 20%, 10%, 5%.
100% already done (RMSE=12.19).

Kill criterion: STAR@20% <= 14 RMSE -> label-efficiency pitch is dead.
If STAR@20% > 16 -> pitch is strong. Between 14-16: survives but weakened.

Output: experiments/v13/star_label_efficiency.json
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

from data_utils import load_raw, fit_normalizer, compute_rul_labels, make_windows, CMAPSSDataset, RUL_CAP, N_SENSORS, SENSOR_COLS, OP_COLS
from models import build_model, count_parameters
from train_utils import train, evaluate

# Try wandb
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

import psutil

SEEDS = [42, 123, 456, 789, 1024]
LABEL_BUDGETS = [0.50, 0.20, 0.10, 0.05]
MAX_EPOCHS = 200
PATIENCE = 20
SUBSET = 'FD001'

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Phase 0a: STAR Label-Efficiency Sweep")
print(f"Device: {DEVICE}")
print(f"Budgets: {LABEL_BUDGETS}")
print(f"Seeds: {SEEDS}")
t0_global = time.time()


def prepare_data_with_budget(subset, budget, seed, window_length=32, batch_size=32):
    """Prepare data with a subset of training engines."""
    from torch.utils.data import DataLoader

    train_df_raw, test_df_raw, rul_test = load_raw(subset)

    # All unique engine IDs
    all_engines = train_df_raw['engine_id'].unique()
    rng = np.random.default_rng(seed)

    # Val split first (15%)
    n_val = max(1, int(len(all_engines) * 0.15))
    val_engines = rng.choice(all_engines, size=n_val, replace=False)
    remaining = np.array([e for e in all_engines if e not in val_engines])

    # Now subsample training engines to the budget
    n_train_budget = max(1, int(len(remaining) * budget))
    train_engines = rng.choice(remaining, size=n_train_budget, replace=False)

    train_df = train_df_raw[train_df_raw['engine_id'].isin(train_engines)].reset_index(drop=True)
    val_df = train_df_raw[train_df_raw['engine_id'].isin(val_engines)].reset_index(drop=True)

    # Fit normalizer on ALL training data (not subsampled) for fair normalization
    stats = fit_normalizer(train_df_raw[train_df_raw['engine_id'].isin(remaining)].reset_index(drop=True))

    X_train, y_train = make_windows(train_df, window_length, stats, is_train=True, rul_cap=RUL_CAP)
    X_val, y_val = make_windows(val_df, window_length, stats, is_train=True, rul_cap=RUL_CAP)
    X_test, _ = make_windows(test_df_raw, window_length, stats, is_train=False, rul_cap=RUL_CAP)
    y_test = np.minimum(rul_test, RUL_CAP).astype(np.float32)

    train_ds = CMAPSSDataset(X_train, y_train)
    val_ds = CMAPSSDataset(X_val, y_val)
    test_ds = CMAPSSDataset(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    return {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'n_train_engines': len(train_engines),
        'n_train_windows': len(X_train),
    }


all_results = {}

for budget in LABEL_BUDGETS:
    budget_key = f"{int(budget*100)}pct"
    print(f"\n{'='*60}")
    print(f"STAR Label Budget: {int(budget*100)}%")
    print(f"{'='*60}")

    seed_rmses = []
    seed_scores = []
    seed_results = []

    for seed in SEEDS:
        print(f"\n--- budget={int(budget*100)}%, seed={seed} ---")
        t0 = time.time()

        torch.manual_seed(seed)
        np.random.seed(seed)

        data = prepare_data_with_budget(SUBSET, budget, seed)
        print(f"  Train engines: {data['n_train_engines']}, windows: {data['n_train_windows']}")

        model = build_model(SUBSET, DEVICE)
        n_params = count_parameters(model)

        # W&B logging
        run = None
        if HAS_WANDB:
            try:
                run = wandb.init(
                    project="industrialjepa",
                    name=f"v13-phase0a-star-{budget_key}-s{seed}",
                    tags=[f"v13-phase0a-star-label-sweep"],
                    config={
                        'phase': '0a',
                        'method': 'STAR',
                        'subset': SUBSET,
                        'budget': budget,
                        'seed': seed,
                        'n_params': n_params,
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

        print(f"  RESULT: RMSE={test_rmse:.3f}, Score={test_score:.1f}, time={wall_time:.0f}s")

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

        seed_rmses.append(test_rmse)
        seed_scores.append(test_score)
        seed_results.append({
            'seed': seed,
            'test_rmse': test_rmse,
            'test_score': test_score,
            'best_val_rmse': train_info['best_val_rmse'],
            'best_epoch': train_info['best_epoch'],
            'epochs_run': train_info['epochs_run'],
            'n_train_engines': data['n_train_engines'],
            'n_train_windows': data['n_train_windows'],
            'wall_time_s': wall_time,
        })

    rmse_mean = float(np.mean(seed_rmses))
    rmse_std = float(np.std(seed_rmses))
    score_mean = float(np.mean(seed_scores))
    score_std = float(np.std(seed_scores))

    print(f"\n  {budget_key} SUMMARY: RMSE={rmse_mean:.3f} +/- {rmse_std:.3f}, Score={score_mean:.1f} +/- {score_std:.1f}")

    all_results[budget_key] = {
        'budget': budget,
        'rmse_mean': rmse_mean,
        'rmse_std': rmse_std,
        'score_mean': score_mean,
        'score_std': score_std,
        'per_seed': seed_results,
        'all_rmse': seed_rmses,
    }

    # Save intermediate results
    out_path = V13_DIR / 'star_label_efficiency.json'
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)

# Final summary
print(f"\n{'='*60}")
print("STAR LABEL EFFICIENCY SUMMARY")
print(f"{'='*60}")
print(f"{'Budget':<10} {'RMSE':>10} {'Std':>8}")
print(f"100%       12.186    0.553   (from prior run)")
for key, res in all_results.items():
    print(f"{key:<10} {res['rmse_mean']:>10.3f} {res['rmse_std']:>8.3f}")

# Kill criterion check
if '20pct' in all_results:
    star_20 = all_results['20pct']['rmse_mean']
    if star_20 <= 14:
        verdict = "DEAD: STAR@20% <= 14 RMSE. Label-efficiency pitch is killed."
    elif star_20 > 16:
        verdict = "STRONG: STAR@20% > 16 RMSE. Label-efficiency pitch survives."
    else:
        verdict = f"WEAKENED: STAR@20% = {star_20:.2f} (between 14-16). Pitch survives but weakened."
    print(f"\nKILL CRITERION: {verdict}")
    all_results['kill_criterion'] = {
        'star_20pct_rmse': star_20,
        'verdict': verdict,
    }

all_results['wall_time_total_s'] = time.time() - t0_global
all_results['reference_100pct'] = {'rmse_mean': 12.186, 'rmse_std': 0.553}

out_path = V13_DIR / 'star_label_efficiency.json'
with open(out_path, 'w') as f:
    json.dump(all_results, f, indent=2)
print(f"\nFinal results saved to {out_path}")
print(f"Total wall time: {time.time()-t0_global:.1f}s")
