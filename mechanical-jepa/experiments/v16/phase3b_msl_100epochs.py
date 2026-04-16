"""
V16 Phase 3b: MSL Anomaly Detection with 100-epoch Full-Data Pretraining.

MSL: Mars Science Laboratory dataset.
- 55 channels, 58K train / 74K test, 10.5% anomaly rate

This is the companion to Phase 3 (SMAP). Both use the same pretraining+probe pipeline.
Target: non-PA F1 > random baseline (~0.10 for 10.5% anomaly rate).

Outputs:
  experiments/v16/phase3b_msl_results.json
  analysis/plots/v16/phase3b_msl_anomaly_scores.png
"""

import sys, json, time, copy
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

V15_DIR = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/experiments/v15')
V16_DIR = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/experiments/v16')
PLOT_DIR = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/analysis/plots/v16')
SMAP_MSL_DIR = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa')

sys.path.insert(0, str(V15_DIR))
sys.path.insert(0, str(SMAP_MSL_DIR))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from data.smap_msl import (
    load_msl, AnomalyPretrainDataset, collate_anomaly_pretrain,
    compute_anomaly_scores, evaluate_anomaly_detection,
)
from evaluation.grey_swan_metrics import anomaly_metrics
from phase1_sigreg import V15JEPA

try:
    import wandb
    HAS_WANDB = True
except Exception:
    HAS_WANDB = False

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
V16_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# Hyperparameters
D_MODEL = 256
N_HEADS = 4
N_LAYERS = 2
BATCH_SIZE = 64
LR = 3e-4
LAMBDA_SIG = 0.05
N_EPOCHS = 100
FULL_DATA = True


def pretrain_on_msl(data: dict, n_epochs: int = N_EPOCHS, seed: int = 42):
    """Pretrain V15 JEPA on full MSL training data."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    n_channels = data['n_channels']
    n_train = len(data['train'])
    print(f"\n=== Pretraining on MSL ({n_channels} channels, {n_train} timesteps) ===")
    print(f"  Mode: {'full data' if FULL_DATA else 'subset'}, {n_epochs} epochs")

    model = V15JEPA(
        n_sensors=n_channels, d_model=D_MODEL, n_heads=N_HEADS,
        n_layers=N_LAYERS, mode='sigreg',
        lambda_sig=LAMBDA_SIG, sigreg_m=256,
    ).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model params: {n_params:,}")

    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, n_epochs)

    n_samples = min(n_train * 2, n_train * 5)
    ds = AnomalyPretrainDataset(
        data['train'], n_samples=n_samples,
        seed=seed, min_context=50, max_context=100, min_horizon=5, max_horizon=20)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True,
                        collate_fn=collate_anomaly_pretrain, num_workers=0)

    run = None
    if HAS_WANDB:
        try:
            run = wandb.init(project='industrialjepa',
                             name=f'v16-msl-100ep-s{seed}',
                             tags=['v16', 'msl', 'anomaly', '100epochs'],
                             config={'n_epochs': n_epochs, 'n_samples': n_samples,
                                     'n_channels': n_channels, 'full_data': FULL_DATA},
                             reinit=True)
        except Exception:
            pass

    t0 = time.time()
    history = {'loss': []}
    for epoch in range(1, n_epochs + 1):
        model.train()
        total_loss = 0.0
        nbatch = 0
        for x_past, past_mask, x_full, full_mask, k in loader:
            x_past, past_mask = x_past.to(DEVICE), past_mask.to(DEVICE)
            x_full, full_mask = x_full.to(DEVICE), full_mask.to(DEVICE)
            k = k.to(DEVICE)

            optim.zero_grad()
            loss, _, _ = model.forward_pretrain(x_past, past_mask, x_full, full_mask, k)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            total_loss += loss.item() * x_past.shape[0]
            nbatch += x_past.shape[0]

        avg_loss = total_loss / nbatch
        history['loss'].append(avg_loss)
        sched.step()

        elapsed = (time.time() - t0) / 60
        print(f"  Ep {epoch:3d} | loss={avg_loss:.4f} | {elapsed:.1f}min", flush=True)

        if run is not None:
            run.log({'epoch': epoch, 'train_loss': avg_loss})

    if run is not None:
        run.finish()

    elapsed = (time.time() - t0) / 60
    print(f"  Done in {elapsed:.1f} min")
    return model, history


def evaluate_msl_anomaly(model, data: dict, threshold_percentile: float = 95.0):
    """Evaluate anomaly detection on MSL test set."""
    print("\n=== Evaluating MSL Anomaly Detection ===")
    model.eval()

    test_scores = compute_anomaly_scores(model, data['test'])
    labels = data['labels']

    threshold = float(np.percentile(test_scores, threshold_percentile))
    print(f"  Anomaly score stats: min={test_scores.min():.3f}, "
          f"max={test_scores.max():.3f}, mean={test_scores.mean():.3f}, "
          f"std={test_scores.std():.3f}")
    print(f"  Threshold ({threshold_percentile}th pct): {threshold:.3f}")

    metrics = anomaly_metrics(labels, test_scores, threshold)

    anon_scores = test_scores[labels == 1]
    norm_scores = test_scores[labels == 0]
    print(f"  Anomaly window scores: mean={anon_scores.mean():.3f}, "
          f"std={anon_scores.std():.3f}")
    print(f"  Normal window scores:  mean={norm_scores.mean():.3f}, "
          f"std={norm_scores.std():.3f}")

    from sklearn.metrics import f1_score
    rng = np.random.RandomState(42)
    rand_pred = (rng.rand(len(labels)) > 0.5).astype(int)
    random_f1 = f1_score(labels, rand_pred, zero_division=0)

    print(f"\n  Results:")
    print(f"    non-PA F1: {metrics['non_pa_f1']:.4f}")
    print(f"    PA F1:     {metrics['pa_f1']:.4f}")
    print(f"    AUC-PR:    {metrics['auc_pr']:.4f}")
    print(f"    Random F1: {random_f1:.4f}")

    return metrics, test_scores, threshold, float(random_f1)


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings('ignore')

    print("=" * 60)
    print("V16 Phase 3b: MSL Anomaly - 100 Epoch Full-Data Pretraining")
    print("=" * 60)

    print("\nLoading MSL data...")
    msl_data = load_msl(normalize=True)
    print(f"  Train: {msl_data['train'].shape}, Test: {msl_data['test'].shape}")
    print(f"  Anomaly rate: {msl_data['anomaly_rate']:.1%}")
    print(f"  Channels: {msl_data['n_channels']}")

    # Pretrain
    model, history = pretrain_on_msl(msl_data, n_epochs=N_EPOCHS, seed=42)

    # Evaluate
    metrics, test_scores, threshold, random_f1 = evaluate_msl_anomaly(model, msl_data)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(test_scores[:5000], label='anomaly score', linewidth=0.5)
    axes[0].axhline(threshold, color='r', linestyle='--', label='threshold')
    labels_arr = msl_data['labels']
    for i in range(0, len(labels_arr), 1):
        if labels_arr[i] and i < 5000:
            axes[0].axvspan(i, min(i+1, 5000), alpha=0.1, color='red')
    axes[0].set_title('MSL Anomaly Scores (first 5000 steps)')
    axes[0].legend()

    import scipy.stats
    anon_scores = test_scores[labels_arr == 1]
    norm_scores = test_scores[labels_arr == 0]
    axes[1].hist(norm_scores, bins=50, alpha=0.5, label='normal', density=True)
    axes[1].hist(anon_scores, bins=50, alpha=0.5, label='anomaly', density=True)
    axes[1].axvline(threshold, color='k', linestyle='--', label='threshold')
    axes[1].set_title('Score Distribution')
    axes[1].legend()

    plt.tight_layout()
    plot_path = PLOT_DIR / 'phase3b_msl_100ep_anomaly_scores.png'
    fig.savefig(str(plot_path), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Plot saved: {plot_path}")

    # Save results
    results = {
        'config': 'v16_msl_100epochs',
        'n_epochs': N_EPOCHS,
        'n_train_samples': len(msl_data['train']),
        'non_pa_f1': metrics['non_pa_f1'],
        'pa_f1': metrics['pa_f1'],
        'auc_pr': metrics['auc_pr'],
        'random_baseline_f1': random_f1,
        'anomaly_score_stats': {
            'mean': float(test_scores.mean()),
            'std': float(test_scores.std()),
            'min': float(test_scores.min()),
            'max': float(test_scores.max()),
        },
        'score_discrimination': {
            'anomaly_mean': float(anon_scores.mean()),
            'anomaly_std': float(anon_scores.std()),
            'normal_mean': float(norm_scores.mean()),
            'normal_std': float(norm_scores.std()),
            'separation': float(anon_scores.mean() - norm_scores.mean()),
        },
        'train_loss_final': float(history['loss'][-1]),
    }

    out_path = V16_DIR / 'phase3b_msl_results.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")
    print(f"\n{'='*60}")
    print(f"PHASE 3b FINAL: MSL non-PA F1 = {metrics['non_pa_f1']:.4f}")
    print(f"  (random baseline: {random_f1:.4f})")
    print(f"{'='*60}")
