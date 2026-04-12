"""
MTS-JEPA Replication: Full experiment runner.
Pre-trains on all datasets with 5 seeds, evaluates downstream, logs results.
"""
import os
import sys
import json
import time
import argparse
import numpy as np
import torch
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_utils import prepare_data
from train_utils import run_single_experiment, DEFAULT_LOSS_CONFIG

# ============================================================================
# Configuration
# ============================================================================

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
EXPERIMENT_LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "EXPERIMENT_LOG.md")

SEEDS = [42, 123, 456, 789, 1024]

# Medium configuration — balances paper fidelity with A10G compute budget
# Paper uses d=256, 6 layers, K=128 on multi-GPU; we use d=128, 3 layers, K=64 on single A10G
# Key architectural elements preserved: codebook, multi-resolution, EMA, dual predictors
TRAIN_CONFIG = {
    'lr': 5e-4,
    'weight_decay': 1e-5,
    'n_epochs': 100,
    'patience': 10,
    'patience_start': 90,  # Force at least 90 epochs of training (loss may increase)
    'max_grad_norm': 0.5,
    'd_model': 128,
    'd_out': 128,
    'n_codes': 64,
    'tau': 0.1,
    'patch_length': 20,
    'n_patches': 5,
    'n_encoder_layers': 3,
    'n_heads': 4,
    'dropout': 0.1,
    'ema_rho': 0.996,
    'loss_config': DEFAULT_LOSS_CONFIG,
}

# Paper Table 1 targets — updated from actual REPLICATION_SPEC.md
PAPER_TARGETS = {
    'MSL': {'f1': 33.58, 'auc': 66.08, 'precision': 35.87, 'recall': 40.80},
    'SMAP': {'f1': 33.64, 'auc': 65.41, 'precision': 24.24, 'recall': 56.02},
    'PSM': {'f1': 61.61, 'auc': 77.85, 'precision': 55.01, 'recall': 72.00},
    'SWaT': {'f1': 72.89, 'auc': 84.95, 'precision': 98.00, 'recall': 58.05},
}


def log_experiment(dataset, seed, n_params, pretrain_info, downstream_results,
                   wall_time, notes=""):
    """Append experiment entry to EXPERIMENT_LOG.md."""
    timestamp = datetime.now().isoformat()
    paper = PAPER_TARGETS.get(dataset, {})

    with open(EXPERIMENT_LOG, "a") as f:
        f.write(f"\n## {dataset} | seed={seed} | {timestamp}\n\n")
        f.write(f"- **Parameters**: {n_params:,}\n")
        f.write(f"- **Best epoch**: {pretrain_info.get('best_epoch', 'N/A')}\n")
        f.write(f"- **Val loss**: {pretrain_info.get('val_loss', 'N/A'):.4f}\n")
        f.write(f"- **Codebook utilization**: {pretrain_info.get('codebook_utilization', 0):.3f}\n")
        f.write(f"- **Codebook perplexity**: {pretrain_info.get('codebook_perplexity', 0):.1f}\n")
        f.write(f"- **Wall time**: {wall_time:.0f}s ({wall_time/60:.1f}min)\n")

        f1 = downstream_results.get('f1', 0)
        auc = downstream_results.get('auc', 0)
        paper_f1 = paper.get('f1', 0)
        paper_auc = paper.get('auc', 0)

        f.write(f"- **F1**: {f1:.2f} (paper: {paper_f1:.2f}, "
                f"delta: {((f1 - paper_f1) / paper_f1 * 100) if paper_f1 else 0:+.1f}%)\n")
        f.write(f"- **AUC**: {auc:.2f} (paper: {paper_auc:.2f}, "
                f"delta: {((auc - paper_auc) / paper_auc * 100) if paper_auc else 0:+.1f}%)\n")
        f.write(f"- **Precision**: {downstream_results.get('precision', 0):.2f}\n")
        f.write(f"- **Recall**: {downstream_results.get('recall', 0):.2f}\n")

        if notes:
            f.write(f"- **Notes**: {notes}\n")
        f.write("\n")


def run_dataset(dataset_name, seeds=None, device='cuda', config=None, batch_size=64):
    """Run all seeds for a single dataset."""
    if seeds is None:
        seeds = SEEDS
    if config is None:
        config = TRAIN_CONFIG.copy()

    print(f"\n{'#'*60}")
    print(f"# Dataset: {dataset_name}")
    print(f"# Seeds: {seeds}")
    print(f"{'#'*60}")

    # Prepare data
    data_dict = prepare_data(dataset_name, window_length=100, batch_size=batch_size)

    per_seed_results = []

    for seed in seeds:
        ckpt_dir = os.path.join(CHECKPOINT_DIR, dataset_name, f"seed{seed}")

        start = time.time()
        result = run_single_experiment(
            dataset_name, seed, data_dict,
            device=device, config=config,
            checkpoint_dir=ckpt_dir, verbose=True,
        )
        wall_time = time.time() - start

        # Log
        log_experiment(
            dataset_name, seed, result['n_params'],
            result['pretrain'], result['downstream'],
            wall_time,
        )

        # Save per-seed result
        result['wall_time'] = wall_time
        per_seed_results.append(result)

        # Save incrementally
        result_path = os.path.join(RESULTS_DIR, f"{dataset_name}_seed{seed}.json")
        os.makedirs(RESULTS_DIR, exist_ok=True)
        with open(result_path, 'w') as f:
            json.dump(result, f, indent=2, default=str)

    # Aggregate results
    metrics = ['f1', 'auc', 'precision', 'recall']
    agg = {}
    for m in metrics:
        values = [r['downstream'].get(m, 0) for r in per_seed_results]
        agg[f'{m}_mean'] = np.mean(values)
        agg[f'{m}_std'] = np.std(values)

    # Compute gap vs paper
    paper = PAPER_TARGETS.get(dataset_name, {})
    for m in ['f1', 'auc']:
        paper_val = paper.get(m, 0)
        if paper_val > 0:
            gap = abs(agg[f'{m}_mean'] - paper_val) / paper_val * 100
            agg[f'{m}_gap_pct'] = gap
            if gap < 5:
                agg[f'{m}_status'] = 'EXACT'
            elif gap < 15:
                agg[f'{m}_status'] = 'GOOD'
            elif gap < 25:
                agg[f'{m}_status'] = 'MARGINAL'
            else:
                agg[f'{m}_status'] = 'FAILED'

    # Save aggregate
    agg_result = {
        'dataset': dataset_name,
        'n_seeds': len(seeds),
        'aggregate': agg,
        'paper_targets': paper,
        'per_seed': per_seed_results,
    }
    agg_path = os.path.join(RESULTS_DIR, f"{dataset_name}_aggregate.json")
    with open(agg_path, 'w') as f:
        json.dump(agg_result, f, indent=2, default=str)

    # Print summary
    print(f"\n{'='*60}")
    print(f"AGGREGATE: {dataset_name}")
    print(f"{'='*60}")
    for m in metrics:
        mean = agg[f'{m}_mean']
        std = agg[f'{m}_std']
        paper_val = paper.get(m, 0)
        status = agg.get(f'{m}_status', '')
        print(f"  {m:>10s}: {mean:.2f} ± {std:.2f}  (paper: {paper_val:.2f}) {status}")

    return agg_result


def main():
    parser = argparse.ArgumentParser(description="MTS-JEPA Replication Experiments")
    parser.add_argument('--datasets', nargs='+', default=['PSM', 'MSL', 'SMAP'],
                        help='Datasets to run')
    parser.add_argument('--seeds', nargs='+', type=int, default=None,
                        help='Seeds to use (default: all 5)')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--fast', action='store_true',
                        help='Use smaller model for faster iteration')
    args = parser.parse_args()

    # Initialize experiment log
    if not os.path.exists(EXPERIMENT_LOG):
        with open(EXPERIMENT_LOG, 'w') as f:
            f.write("# MTS-JEPA Replication: Experiment Log\n\n")
            f.write(f"Started: {datetime.now().isoformat()}\n\n")

    config = TRAIN_CONFIG.copy()

    if args.fast:
        config.update({
            'd_model': 64,
            'd_out': 64,
            'n_codes': 32,
            'n_encoder_layers': 2,
            'n_heads': 4,
            'n_epochs': 30,
            'patience_start': 15,
        })
        print("*** FAST MODE: Reduced model for quick iteration ***\n")

    all_results = {}
    for dataset in args.datasets:
        try:
            result = run_dataset(
                dataset, seeds=args.seeds, device=args.device,
                config=config, batch_size=args.batch_size,
            )
            all_results[dataset] = result
        except Exception as e:
            print(f"\nERROR on {dataset}: {e}")
            import traceback
            traceback.print_exc()

    # Final summary
    print(f"\n{'#'*60}")
    print("# FINAL REPLICATION SUMMARY")
    print(f"{'#'*60}\n")

    for ds, result in all_results.items():
        agg = result['aggregate']
        paper = result['paper_targets']
        f1_status = agg.get('f1_status', 'N/A')
        auc_status = agg.get('auc_status', 'N/A')
        print(f"{ds:>6s}: F1={agg['f1_mean']:.2f}±{agg['f1_std']:.2f} "
              f"(paper {paper.get('f1', 0):.2f}) [{f1_status}] | "
              f"AUC={agg['auc_mean']:.2f}±{agg['auc_std']:.2f} "
              f"(paper {paper.get('auc', 0):.2f}) [{auc_status}]")


if __name__ == '__main__':
    main()
