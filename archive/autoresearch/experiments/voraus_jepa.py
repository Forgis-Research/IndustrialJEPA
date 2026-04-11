#!/usr/bin/env python3
"""
Exp 52: Mechanical-JEPA Pretraining on Voraus-AD

Same JEPA architecture as aursad_jepa.py but for Voraus-AD.
Voraus-AD: Yu-Cobot pick-and-place, 6 DOF, voltage signals, 2122 episodes.

Cross-dataset comparison: AURSAD vs Voraus-AD — does JEPA benefit scale
with dataset size? (Voraus: 2122 ep vs AURSAD: 4094 ep)
"""

import sys
import time
import json
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Reuse JEPA model and training functions from aursad_jepa
from autoresearch.experiments.aursad_jepa import (
    MechanicalJEPA,
    pretrain_jepa,
    eval_linear_probe,
    JEPA_CONFIG,
    DEVICE,
    SEEDS,
)
from industrialjepa.data.factorynet import FactoryNetDataset, FactoryNetConfig

import logging
logging.getLogger('industrialjepa').setLevel(logging.WARNING)
logging.getLogger('datasets').setLevel(logging.WARNING)


def load_voraus_data(max_episodes=None, window_size=64):
    """Load Voraus-AD via FactoryNetDataset."""
    print("[Data] Loading Voraus-AD from HuggingFace...")

    config = FactoryNetConfig(
        dataset_name="Forgis/FactoryNet_Dataset",
        config_name="normalized",
        data_source="voraus",
        window_size=window_size,
        stride=window_size // 2,
        normalize=True,
        norm_mode="episode",
        max_episodes=max_episodes,
        train_healthy_only=False,
    )

    train_ds = FactoryNetDataset(config, split='train')
    shared = train_ds.get_shared_data()
    val_ds = FactoryNetDataset(config, split='val', shared_data=shared)
    test_ds = FactoryNetDataset(config, split='test', shared_data=shared)

    sample = train_ds[0]
    setpoint, effort, meta = sample
    n_setpoint = setpoint.shape[1]
    n_effort = effort.shape[1]
    n_channels = n_setpoint + n_effort

    print(f"  Train: {len(train_ds)} windows")
    print(f"  Val: {len(val_ds)} windows")
    print(f"  Test: {len(test_ds)} windows")
    print(f"  Channels: setpoint={n_setpoint}, effort={n_effort}, total={n_channels}")

    def extract_arrays(ds):
        setpoints, efforts, labels = [], [], []
        for i in range(len(ds)):
            sp, ef, m = ds[i]
            setpoints.append(sp)
            efforts.append(ef)
            labels.append(1 if m['is_anomaly'] else 0)
        return (torch.stack(setpoints),
                torch.stack(efforts),
                torch.tensor(labels, dtype=torch.long))

    print("  Extracting arrays...")
    train_sp, train_ef, train_labels = extract_arrays(train_ds)
    val_sp, val_ef, val_labels = extract_arrays(val_ds)
    test_sp, test_ef, test_labels = extract_arrays(test_ds)

    train_X = torch.cat([train_sp, train_ef], dim=-1)
    val_X = torch.cat([val_sp, val_ef], dim=-1)
    test_X = torch.cat([test_sp, test_ef], dim=-1)

    print(f"  Train X shape: {train_X.shape}")
    print(f"  Anomaly rate: train={train_labels.float().mean():.3f}, test={test_labels.float().mean():.3f}")

    return {
        'train': (train_X, train_labels),
        'val': (val_X, val_labels),
        'test': (test_X, test_labels),
        'n_channels': n_channels,
    }


def main():
    print("=" * 60)
    print("EXP 52: Mechanical-JEPA on Voraus-AD")
    print(f"Device: {DEVICE}")
    print("=" * 60)

    t0 = time.time()
    config = JEPA_CONFIG

    # Load data
    data = load_voraus_data(max_episodes=None, window_size=config['window_size'])
    n_channels = data['n_channels']
    train_X, train_y = data['train']
    val_X, val_y = data['val']
    test_X, test_y = data['test']

    print(f"\nData loaded in {time.time()-t0:.1f}s")

    all_results = []

    for seed in SEEDS:
        print(f"\n{'='*50}")
        print(f"SEED {seed}")
        print(f"{'='*50}")
        torch.manual_seed(seed)
        np.random.seed(seed)

        # ---- JEPA pretrained ----
        print("\n[1] JEPA Pretraining...")
        model_jepa = MechanicalJEPA(
            n_channels=n_channels,
            window_size=config['window_size'],
            d_model=config['d_model'],
            n_heads=config['n_heads'],
            n_layers=config['n_layers'],
            predictor_layers=config['predictor_layers'],
            patch_len=8,
            mask_ratio=config['mask_ratio'],
            ema_decay=config['ema_decay'],
        ).to(DEVICE)

        n_params = sum(p.numel() for p in model_jepa.parameters())
        print(f"  Model params: {n_params:,}")

        t_pretrain = time.time()
        losses = pretrain_jepa(model_jepa, train_X, config, seed=seed)
        print(f"  Pretraining done in {time.time()-t_pretrain:.1f}s")
        print(f"  Loss: {losses[0]:.4f} -> {losses[-1]:.4f}")

        probe_results = eval_linear_probe(model_jepa, train_X, train_y, test_X, test_y)
        print(f"  AUROC={probe_results['auroc']:.4f}, F1={probe_results['f1']:.4f}")

        # ---- Random init ----
        print("\n[2] Random init...")
        model_random = MechanicalJEPA(
            n_channels=n_channels,
            window_size=config['window_size'],
            d_model=config['d_model'],
            n_heads=config['n_heads'],
            n_layers=config['n_layers'],
            predictor_layers=config['predictor_layers'],
            patch_len=8,
            mask_ratio=config['mask_ratio'],
            ema_decay=config['ema_decay'],
        ).to(DEVICE)
        probe_random = eval_linear_probe(model_random, train_X, train_y, test_X, test_y)
        print(f"  AUROC={probe_random['auroc']:.4f}, F1={probe_random['f1']:.4f}")

        seed_result = {
            'seed': seed,
            'n_params': n_params,
            'n_channels': n_channels,
            'pretrain_loss_start': float(losses[0]),
            'pretrain_loss_end': float(losses[-1]),
            'jepa_probe_auroc': float(probe_results['auroc']),
            'jepa_probe_f1': float(probe_results['f1']),
            'random_probe_auroc': float(probe_random['auroc']),
            'random_probe_f1': float(probe_random['f1']),
            'jepa_vs_random_delta': float(probe_results['auroc'] - probe_random['auroc']),
        }
        all_results.append(seed_result)

    jepa_aurocs = [r['jepa_probe_auroc'] for r in all_results]
    random_aurocs = [r['random_probe_auroc'] for r in all_results]
    deltas = [r['jepa_vs_random_delta'] for r in all_results]

    summary = {
        'dataset': 'Voraus-AD',
        'n_channels': n_channels,
        'window_size': config['window_size'],
        'epochs_pretrain': config['epochs_pretrain'],
        'n_params': all_results[0]['n_params'],
        'jepa_auroc_mean': float(np.mean(jepa_aurocs)),
        'jepa_auroc_std': float(np.std(jepa_aurocs)),
        'random_auroc_mean': float(np.mean(random_aurocs)),
        'random_auroc_std': float(np.std(random_aurocs)),
        'delta_mean': float(np.mean(deltas)),
        'delta_std': float(np.std(deltas)),
        'verdict': 'JEPA_BETTER' if np.mean(deltas) > 0.01 else 'NO_BENEFIT',
        'seeds': all_results,
    }

    print("\n" + "=" * 60)
    print("VORAUS-AD JEPA RESULTS")
    print("=" * 60)
    print(f"JEPA pretrained:  AUROC = {summary['jepa_auroc_mean']:.4f} ± {summary['jepa_auroc_std']:.4f}")
    print(f"Random init:      AUROC = {summary['random_auroc_mean']:.4f} ± {summary['random_auroc_std']:.4f}")
    print(f"Delta:            {summary['delta_mean']:+.4f} ± {summary['delta_std']:.4f}")
    print(f"Verdict:          {summary['verdict']}")
    print(f"Total time: {time.time()-t0:.1f}s")

    out_path = PROJECT_ROOT / "datasets" / "data" / "voraus_jepa_results.json"
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {out_path}")

    return summary


if __name__ == "__main__":
    main()
