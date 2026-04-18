"""Restart Phase 1 for seeds 123, 456 only (seed 42 already done)."""
import sys, json, time, os
from pathlib import Path
import numpy as np

V17 = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/experiments/v17')
sys.path.insert(0, str(V17))

import phase1_v17_baseline as P
from phase1_v17_baseline import pretrain_one_seed, load_cmapss_subset

RESTART_SEEDS = [123, 456]

def main():
    # Load existing seed 42 results
    existing = V17 / 'phase1_results_seed42.json'
    prior_per_seed = []
    if existing.exists():
        with open(existing) as f:
            r = json.load(f)
        prior_per_seed = r.get('per_seed', [])
        print(f"Loaded {len(prior_per_seed)} prior seeds from {existing.name}",
              flush=True)

    data = load_cmapss_subset('FD001')
    all_results = []  # fresh runs
    t0 = time.time()
    for seed in RESTART_SEEDS:
        r = pretrain_one_seed(seed, data, n_epochs=P.N_EPOCHS)
        all_results.append(r)

        # ---- merge with prior_per_seed and save ----
        merged_per_seed = prior_per_seed + [
            {k: v for k, v in rr.items() if k != 'history'}
            for rr in all_results
        ]
        out = {
            'config': 'v17_baseline',
            'w': P.W_WIN, 'K_max': P.K_MAX, 'k_eval_f1': P.K_EVAL_F1,
            'd_model': P.D_MODEL, 'n_layers': P.N_LAYERS, 'n_epochs': P.N_EPOCHS,
            'batch_size': P.BATCH_SIZE, 'lr': P.LR, 'ema_momentum': P.EMA_MOMENTUM,
            'seeds_done': [s.get('seed') for s in merged_per_seed],
            'per_seed': merged_per_seed,
            'v2_baseline_rmse': 17.81,
        }
        with open(V17 / 'phase1_results.json', 'w') as f:
            json.dump(out, f, indent=2, default=float)

    # ---- Final aggregate including seed 42 ----
    # Pull best-val & final from prior per-seed (seed 42)
    all_per_seed = list(prior_per_seed) + [
        {k: v for k, v in rr.items() if k != 'history'}
        for rr in all_results
    ]

    rmses_val = [r['best_val_rmse'] for r in all_per_seed]
    rmses_test = [r['final']['test_rmse'] for r in all_per_seed]
    f1s_test = [r['final']['test_f1'] for r in all_per_seed]
    aucpr_test = [r['final']['test_auc_pr'] for r in all_per_seed]

    summary = {
        'config': 'v17_baseline',
        'seeds': [s.get('seed') for s in all_per_seed],
        'w': P.W_WIN, 'K_max': P.K_MAX, 'k_eval_f1': P.K_EVAL_F1,
        'probe_val_rmse_per_seed': rmses_val,
        'probe_val_rmse_mean': float(np.mean(rmses_val)),
        'probe_val_rmse_std': float(np.std(rmses_val)),
        'test_rmse_per_seed': rmses_test,
        'test_rmse_mean': float(np.mean(rmses_test)),
        'test_rmse_std': float(np.std(rmses_test)),
        'f1_at_k30_per_seed': f1s_test,
        'f1_at_k30_mean': float(np.mean(f1s_test)),
        'auc_pr_at_k30_mean': float(np.mean(aucpr_test)),
        'v2_baseline_rmse': 17.81,
        'histories': [r.get('history', {}) for r in all_results],
        'runtime_hours': (time.time() - t0) / 3600,
    }
    with open(V17 / 'phase1_results.json', 'w') as f:
        json.dump(summary, f, indent=2, default=float)

    print("\n===== Phase 1 aggregated SUMMARY =====")
    print(f"Seeds             : {summary['seeds']}")
    print(f"Probe val RMSE    : {np.mean(rmses_val):.2f} +/- {np.std(rmses_val):.2f}")
    print(f"Test RMSE         : {np.mean(rmses_test):.2f} +/- {np.std(rmses_test):.2f}")
    print(f"Test F1@k=30      : {np.mean(f1s_test):.3f}")
    print(f"Test AUC-PR       : {np.mean(aucpr_test):.3f}")
    print(f"V2 ref (n=5)      : 17.81")


if __name__ == '__main__':
    main()
