"""
Phase 2: STAR label-efficiency sweep on FD001
Label budgets: 100%, 50%, 20%, 10%, 5%
5 seeds per budget
Output: experiments/v12/star_label_efficiency.json
"""

import json
import sys
import time
import copy
from pathlib import Path

import numpy as np
import torch

# Add STAR replication path
STAR_DIR = Path('/home/sagemaker-user/IndustrialJEPA/paper-replications/star')
sys.path.insert(0, str(STAR_DIR))

from data_utils import prepare_data, load_raw, fit_normalizer, make_windows, CMAPSSDataset
from models import build_model, count_parameters
from train_utils import train, evaluate
from torch.utils.data import DataLoader

V12_DIR = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/experiments/v12')
V12_DIR.mkdir(exist_ok=True)

SEEDS = [42, 123, 456, 789, 1024]
BUDGETS = [1.0, 0.5, 0.2, 0.1, 0.05]
SUBSET = 'FD001'
RUL_CAP = 125
VAL_FRACTION = 0.15
MAX_EPOCHS = 200
PATIENCE = 20

CONFIGS = {
    "FD001": dict(lr=0.0002, batch_size=32, window_length=32, n_scales=3, d_model=128, n_heads=1),
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")


def prepare_data_with_budget(subset, window_length, batch_size, val_fraction, seed, rul_cap, budget):
    """Like prepare_data but subsamples training engines to 'budget' fraction."""
    from data_utils import (load_raw, fit_normalizer, make_windows, CMAPSSDataset,
                             compute_rul_labels)

    train_df_raw, test_df_raw, rul_test = load_raw(subset)

    # Train/val split by engine (same as STAR, seed-based)
    all_engines = train_df_raw["engine_id"].unique()
    rng = np.random.default_rng(seed)
    n_val = max(1, int(len(all_engines) * val_fraction))
    val_engines = rng.choice(all_engines, size=n_val, replace=False)
    train_engines_all = np.array([e for e in all_engines if e not in val_engines])

    # Subsample training engines by budget
    n_train = max(1, int(len(train_engines_all) * budget))
    rng2 = np.random.default_rng(seed + 10000)
    train_engines_sub = rng2.choice(train_engines_all, size=n_train, replace=False)

    train_df_full = train_df_raw[train_df_raw["engine_id"].isin(train_engines_all)].reset_index(drop=True)
    train_df = train_df_raw[train_df_raw["engine_id"].isin(train_engines_sub)].reset_index(drop=True)
    val_df = train_df_raw[train_df_raw["engine_id"].isin(val_engines)].reset_index(drop=True)

    # Fit normalizer on FULL training engines (not subsampled) - important for fairness
    stats = fit_normalizer(train_df_full)

    X_train, y_train = make_windows(train_df, window_length, stats, is_train=True, rul_cap=rul_cap)
    X_val, y_val = make_windows(val_df, window_length, stats, is_train=True, rul_cap=rul_cap)
    X_test, _ = make_windows(test_df_raw, window_length, stats, is_train=False, rul_cap=rul_cap)

    y_test = np.minimum(rul_test, rul_cap).astype(np.float32)

    train_ds = CMAPSSDataset(X_train, y_train)
    val_ds = CMAPSSDataset(X_val, y_val)
    test_ds = CMAPSSDataset(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "n_train_engines": len(train_engines_sub),
        "n_val_engines": len(val_engines),
        "n_test_engines": len(test_df_raw["engine_id"].unique()),
        "n_train_windows": len(X_train),
    }


results_by_budget = {}
cfg = CONFIGS[SUBSET]

for budget in BUDGETS:
    budget_pct = int(budget * 100)
    print(f"\n{'='*60}")
    print(f"Budget: {budget_pct}%")
    print(f"{'='*60}")

    seed_rmses = []
    seed_results = []

    for seed in SEEDS:
        print(f"\n--- seed={seed} ---")
        t0 = time.time()

        torch.manual_seed(seed)
        np.random.seed(seed)

        data = prepare_data_with_budget(
            SUBSET, cfg["window_length"], cfg["batch_size"],
            VAL_FRACTION, seed, RUL_CAP, budget
        )
        print(f"  Train engines: {data['n_train_engines']}, windows: {data['n_train_windows']}")

        model = build_model(SUBSET, device)

        train_info = train(
            model=model,
            train_loader=data["train_loader"],
            val_loader=data["val_loader"],
            lr=cfg["lr"],
            max_epochs=MAX_EPOCHS,
            patience=PATIENCE,
            device=device,
            checkpoint_path=None,
            verbose=True,
        )

        test_rmse, test_score, preds, trues = evaluate(model, data["test_loader"], device)
        elapsed = time.time() - t0
        print(f"  RMSE={test_rmse:.3f} | epochs={train_info['epochs_run']} | time={elapsed:.0f}s")

        seed_rmses.append(float(test_rmse))
        seed_results.append({
            "seed": seed,
            "test_rmse": float(test_rmse),
            "test_score": float(test_score),
            "best_val_rmse": float(train_info["best_val_rmse"]),
            "epochs_run": train_info["epochs_run"],
            "n_train_engines": data["n_train_engines"],
        })

    mean_rmse = float(np.mean(seed_rmses))
    std_rmse = float(np.std(seed_rmses))
    print(f"\n  Budget {budget_pct}%: RMSE={mean_rmse:.3f} +/- {std_rmse:.3f}")

    results_by_budget[str(budget_pct)] = {
        "budget_fraction": budget,
        "budget_pct": budget_pct,
        "mean_rmse": mean_rmse,
        "std_rmse": std_rmse,
        "per_seed": seed_results,
    }

    # Save intermediate results
    out = {
        "subset": SUBSET,
        "description": "STAR label-efficiency sweep FD001",
        "budgets_pct": BUDGETS,
        "results": results_by_budget,
    }
    with open(V12_DIR / "star_label_efficiency.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"  Intermediate results saved.")

print("\n\nFINAL SUMMARY:")
print(f"{'Budget':>8} | {'RMSE Mean':>10} | {'RMSE Std':>10}")
print("-" * 35)
for k, v in results_by_budget.items():
    print(f"{k:>7}% | {v['mean_rmse']:>10.3f} | {v['std_rmse']:>10.3f}")

print(f"\nResults saved to {V12_DIR / 'star_label_efficiency.json'}")
