"""
MTS-JEPA Ablation Studies (Table 3 Replication).

All ablations:
1. w/o KL Regularization
2. w/o Reconstruction Decoder
3. w/o Predictive Objective
4. w/o Codebook Loss
5. w/o Codebook Module
6. w/o Downsampling (coarse branch)
"""
import os
import sys
import json
import copy
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_utils import prepare_data
from train_utils import (
    run_single_experiment, DEFAULT_LOSS_CONFIG, MTSJEPA,
)
from models import MTSJEPANoCodebook

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "ablations")


def make_ablation_config(ablation_name, base_config=None):
    """Create modified config for each ablation."""
    if base_config is None:
        base_config = {
            'lr': 5e-4, 'weight_decay': 1e-5, 'n_epochs': 50,
            'patience': 10, 'patience_start': 25, 'max_grad_norm': 0.5,
            'd_model': 128, 'd_out': 128, 'n_codes': 64, 'tau': 0.1,
            'patch_length': 20, 'n_patches': 5, 'n_encoder_layers': 3,
            'n_heads': 4, 'dropout': 0.1, 'ema_rho': 0.996,
            'loss_config': copy.deepcopy(DEFAULT_LOSS_CONFIG),
        }

    config = copy.deepcopy(base_config)

    if ablation_name == 'no_kl':
        # Remove KL terms, keep only MSE prediction
        config['loss_config']['kl_scale'] = 0.0

    elif ablation_name == 'no_reconstruction':
        # Remove reconstruction decoder and loss
        config['loss_config']['lambda_r_start'] = 0.0
        config['loss_config']['lambda_r_end'] = 0.0

    elif ablation_name == 'no_prediction':
        # Remove predictive objective entirely
        config['loss_config']['lambda_f'] = 0.0
        config['loss_config']['lambda_c'] = 0.0

    elif ablation_name == 'no_codebook_loss':
        # Keep codebook module but remove auxiliary losses
        config['loss_config']['lambda_emb'] = 0.0
        config['loss_config']['lambda_com'] = 0.0
        config['loss_config']['lambda_ent_sample'] = 0.0
        config['loss_config']['lambda_ent_batch'] = 0.0

    elif ablation_name == 'no_codebook':
        # Remove codebook entirely — flag for special model construction
        config['_no_codebook'] = True

    elif ablation_name == 'no_downsampling':
        # Remove coarse branch, only fine predictor
        config['loss_config']['lambda_c'] = 0.0

    elif ablation_name == 'full':
        pass  # Baseline — no changes

    return config


def run_ablation(ablation_name, dataset_name, seed=42, device='cuda'):
    """Run a single ablation experiment."""
    config = make_ablation_config(ablation_name)

    print(f"\n--- Ablation: {ablation_name} on {dataset_name} (seed={seed}) ---")

    data_dict = prepare_data(dataset_name, window_length=100, batch_size=32)

    ckpt_dir = os.path.join(RESULTS_DIR, dataset_name, ablation_name, f"seed{seed}")

    result = run_single_experiment(
        dataset_name, seed, data_dict,
        device=device, config=config,
        checkpoint_dir=ckpt_dir, verbose=True,
    )

    result['ablation'] = ablation_name

    # Save
    os.makedirs(RESULTS_DIR, exist_ok=True)
    result_path = os.path.join(RESULTS_DIR, f"{dataset_name}_{ablation_name}_seed{seed}.json")
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2, default=str)

    return result


def run_all_ablations(dataset_names=None, seed=42, device='cuda'):
    """Run all ablations on specified datasets."""
    if dataset_names is None:
        dataset_names = ['PSM', 'MSL']

    ablation_names = [
        'full',
        'no_kl',
        'no_reconstruction',
        'no_prediction',
        'no_codebook_loss',
        'no_codebook',
        'no_downsampling',
    ]

    results = {}
    for ds in dataset_names:
        results[ds] = {}
        for abl in ablation_names:
            try:
                result = run_ablation(abl, ds, seed, device)
                results[ds][abl] = result
            except Exception as e:
                print(f"ERROR: {abl} on {ds}: {e}")
                import traceback
                traceback.print_exc()

    # Print ablation table
    print(f"\n{'='*80}")
    print("ABLATION RESULTS")
    print(f"{'='*80}")
    print(f"{'Ablation':<25s} | {'Dataset':<6s} | {'F1':>6s} | {'AUC':>6s} | {'Notes'}")
    print("-" * 80)
    for ds in dataset_names:
        for abl in ablation_names:
            if abl in results[ds]:
                r = results[ds][abl]
                d = r.get('downstream', {})
                print(f"{abl:<25s} | {ds:<6s} | {d.get('f1', 0):6.2f} | "
                      f"{d.get('auc', 0):6.2f} |")
        print()

    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', default=['PSM', 'MSL'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    run_all_ablations(args.datasets, args.seed, args.device)
