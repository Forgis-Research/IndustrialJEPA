"""
Phase 5: Comprehensive evaluation of RUL methods.

- Per-dataset breakdown (FEMTO, XJTU-SY)
- Cross-dataset transfer evaluation
- FEMTO nRMSE (published SOTA comparison)
- Episode-level analysis
"""

import os
import sys
import json
import numpy as np
from typing import Dict, List, Optional
from collections import defaultdict

sys.path.insert(0, '/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/v8')
from data_pipeline import load_rul_episodes, episode_train_test_split, compute_piecewise_rul
from rul_baselines import (
    compute_metrics, compute_monotonicity, compute_max_lifetime,
    baseline_constant_mean, baseline_elapsed_time_linear,
    aggregate_seed_results, precompute_jepa_embeddings, precompute_random_embeddings,
    compute_handcrafted_features_per_snapshot, compute_envelope_rms_per_snapshot,
    train_lstm_model, evaluate_lstm_model, train_mlp_model, evaluate_mlp_model,
    SEEDS, N_EPOCHS_LSTM, N_EPOCHS_MLP
)

import torch
import torch.nn as nn

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RESULTS_DIR = '/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/v8/results'
CHECKPOINT_DIR = '/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/v8/checkpoints'


def compute_femto_nrmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    FEMTO/PHM2012 normalized RMSE:
    nRMSE = RMSE / (rul_max - rul_min)
    With rul_max=1.0 and rul_min=0.0 for RUL% formulation, nRMSE = RMSE.
    But many papers use time-based formulation; we note both.
    """
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    nrmse = rmse / (y_true.max() - y_true.min() + 1e-10)
    return nrmse


def run_cross_dataset_evaluation(
    encoder_path: Optional[str] = None,
    label_type: str = 'linear',
) -> Dict:
    """
    Evaluate cross-dataset transfer:
    - Train on FEMTO, test on XJTU-SY
    - Train on XJTU-SY, test on FEMTO
    - Within-dataset (FEMTO only, XJTU-SY only)
    """
    print(f"\n{'='*60}")
    print(f"Cross-Dataset Evaluation (label_type={label_type})")
    print(f"{'='*60}")

    # Load all episodes
    all_episodes = load_rul_episodes(['femto', 'xjtu_sy'], verbose=True)

    # Separate by source
    femto_eps = {ep: snaps for ep, snaps in all_episodes.items()
                  if snaps[0]['source'] == 'femto'}
    xjtu_eps = {ep: snaps for ep, snaps in all_episodes.items()
                 if snaps[0]['source'] == 'xjtu_sy'}

    # RUL labels
    if label_type == 'linear':
        all_rul = {ep: [s['rul_percent'] for s in snaps]
                   for ep, snaps in all_episodes.items()}
    else:
        all_rul = compute_piecewise_rul(all_episodes)

    # Precompute features
    print("\n--- Precomputing features for cross-dataset eval ---")
    hc_feats = {ep: compute_handcrafted_features_per_snapshot(snaps)
                for ep, snaps in all_episodes.items()}
    env_rms = {ep: compute_envelope_rms_per_snapshot(snaps)
               for ep, snaps in all_episodes.items()}

    # JEPA embeddings
    jepa_embs = None
    if encoder_path and os.path.exists(encoder_path):
        jepa_embs = precompute_jepa_embeddings(all_episodes, encoder_path)

    from rul_model import RULLSTM, HandcraftedLSTM

    # Cross-dataset configurations
    configs = [
        {
            'name': 'femto_to_femto',
            'train_src': 'femto', 'test_src': 'femto',
            'train_eps': list(femto_eps.keys())[:12],
            'test_eps': list(femto_eps.keys())[12:],
        },
        {
            'name': 'xjtu_to_xjtu',
            'train_src': 'xjtu_sy', 'test_src': 'xjtu_sy',
            'train_eps': list(xjtu_eps.keys())[:5],
            'test_eps': list(xjtu_eps.keys())[5:],
        },
        {
            'name': 'femto_to_xjtu',
            'train_src': 'femto', 'test_src': 'xjtu_sy',
            'train_eps': list(femto_eps.keys()),
            'test_eps': list(xjtu_eps.keys()),
        },
        {
            'name': 'xjtu_to_femto',
            'train_src': 'xjtu_sy', 'test_src': 'femto',
            'train_eps': list(xjtu_eps.keys()),
            'test_eps': list(femto_eps.keys()),
        },
    ]

    cross_results = {}

    for config in configs:
        name = config['name']
        train_ids = config['train_eps']
        test_ids = config['test_eps']

        if not train_ids or not test_ids:
            print(f"  {name}: skipped (insufficient episodes)")
            continue

        print(f"\n  {name}: train on {len(train_ids)}, test on {len(test_ids)}")

        max_lt = compute_max_lifetime(train_ids, all_episodes)

        # Normalize features on train
        hc_train = np.concatenate([hc_feats[ep] for ep in train_ids])
        hc_mean = hc_train.mean(0)
        hc_std = hc_train.std(0) + 1e-10
        hc_norm = {ep: ((hc_feats[ep] - hc_mean) / hc_std).astype(np.float32)
                   for ep in all_episodes}

        env_train = np.concatenate([env_rms[ep] for ep in train_ids])
        env_mean = env_train.mean()
        env_std = env_train.std() + 1e-10
        env_norm = {ep: ((env_rms[ep] - env_mean) / env_std).astype(np.float32)
                    for ep in all_episodes}

        if jepa_embs:
            emb_train = np.concatenate([jepa_embs[ep] for ep in train_ids])
            emb_mean = emb_train.mean(0)
            emb_std = emb_train.std(0) + 1e-10
            jepa_norm = {ep: ((jepa_embs[ep] - emb_mean) / emb_std).astype(np.float32)
                         for ep in all_episodes}

        config_results = {}

        # Method: elapsed time only
        b = baseline_elapsed_time_linear(train_ids, test_ids, all_episodes, all_rul, max_lt)
        config_results['elapsed_time'] = {k: v for k, v in b.items() if k != 'per_episode'}
        print(f"    Elapsed-time RMSE: {b['rmse']:.4f}")

        # Method: handcrafted + LSTM
        hc_seed_results = []
        for seed in SEEDS:
            m = HandcraftedLSTM(n_features=18).to(DEVICE)
            m = train_lstm_model(m, train_ids, all_episodes, all_rul, 'handcrafted',
                                  max_lt, seed=seed, n_epochs=N_EPOCHS_LSTM,
                                  handcrafted_feats=hc_norm)
            r = evaluate_lstm_model(m, test_ids, all_episodes, all_rul, 'handcrafted',
                                      max_lt, handcrafted_feats=hc_norm)
            hc_seed_results.append(r)
        config_results['handcrafted_lstm'] = aggregate_seed_results(hc_seed_results)
        print(f"    HC+LSTM RMSE: {config_results['handcrafted_lstm']['rmse_mean']:.4f}")

        # Method: JEPA + LSTM
        if jepa_embs:
            jepa_seed_results = []
            for seed in SEEDS:
                m = RULLSTM(embed_dim=256).to(DEVICE)
                m = train_lstm_model(m, train_ids, all_episodes, all_rul, 'jepa',
                                      max_lt, seed=seed, n_epochs=N_EPOCHS_LSTM,
                                      jepa_embeddings=jepa_norm)
                r = evaluate_lstm_model(m, test_ids, all_episodes, all_rul, 'jepa',
                                          max_lt, jepa_embeddings=jepa_norm)
                jepa_seed_results.append(r)
            config_results['jepa_lstm'] = aggregate_seed_results(jepa_seed_results)
            print(f"    JEPA+LSTM RMSE: {config_results['jepa_lstm']['rmse_mean']:.4f}")

        cross_results[name] = config_results

    # Save
    fname = os.path.join(RESULTS_DIR, f'cross_dataset_{label_type}.json')
    with open(fname, 'w') as f:
        json.dump(cross_results, f, indent=2, default=str)
    print(f"\nCross-dataset results saved: {fname}")

    return cross_results


def print_summary_table(results_path: str, label_type: str = 'linear'):
    """Print formatted summary table from saved results."""
    with open(results_path) as f:
        data = json.load(f)

    results = data.get('results', {})
    time_rmse = data.get('time_only_rmse', float('nan'))

    print(f"\n{'='*80}")
    print(f"RUL PREDICTION RESULTS (label_type={label_type})")
    print(f"{'='*80}")
    print(f"{'Method':<30} {'RMSE':>8} {'±std':>6} {'MAE':>8} {'R²':>6} {'Spearman':>10} {'vs Time-Only':>12}")
    print('-' * 90)

    for method, r in results.items():
        if isinstance(r, dict) and 'rmse_mean' in r:
            rmse = r['rmse_mean']
            std = r['rmse_std']
            mae = r.get('mae_mean', float('nan'))
            r2 = r.get('r2_mean', float('nan'))
            spear = r.get('spearman_mean', float('nan'))
            vs = (time_rmse - rmse) / time_rmse * 100
            print(f"{method:<30} {rmse:8.4f} {std:6.4f} {mae:8.4f} {r2:6.3f} {spear:10.3f} {vs:+11.1f}%")
        elif isinstance(r, dict) and 'rmse' in r:
            rmse = r['rmse']
            mae = r.get('mae', float('nan'))
            r2 = r.get('r2', float('nan'))
            spear = r.get('spearman', float('nan'))
            vs = (time_rmse - rmse) / time_rmse * 100
            print(f"{method:<30} {rmse:8.4f} {'':6} {mae:8.4f} {r2:6.3f} {spear:10.3f} {vs:+11.1f}%")

    print(f"\nTime-only baseline RMSE: {time_rmse:.4f}")
    print(f"Published SOTA (CNN-GRU-MHA on FEMTO, 2024): nRMSE ≈ 0.044")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder', type=str,
                         default=f'{CHECKPOINT_DIR}/jepa_v8_best.pt')
    parser.add_argument('--label-type', type=str, default='linear',
                         choices=['linear', 'piecewise'])
    parser.add_argument('--summary-only', action='store_true')
    args = parser.parse_args()

    if args.summary_only:
        results_path = os.path.join(RESULTS_DIR, f'rul_baselines_{args.label_type}.json')
        if os.path.exists(results_path):
            print_summary_table(results_path, args.label_type)
        else:
            print(f"No results at {results_path}")
    else:
        cross = run_cross_dataset_evaluation(
            encoder_path=args.encoder,
            label_type=args.label_type,
        )
