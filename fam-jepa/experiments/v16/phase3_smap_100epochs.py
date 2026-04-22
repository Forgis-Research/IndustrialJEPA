"""
V16 Phase 3: SMAP Anomaly Detection with 100-epoch Full-Data Pretraining.

V15 result: non-PA F1=0.069 (barely beats random=0.071) after only 20 epochs
on 20K samples. Anomaly scores near-constant (mean=0.838, std=0.039).
Model assigned high error to EVERYTHING - no discrimination.

V16 fix: 100 epochs on full 135K training set (~6.75x more training).

Expected improvement: anomaly scores should become more discriminative.
Target: non-PA F1 > 0.10 (beats random baseline meaningfully).

Outputs:
  experiments/v16/phase3_smap_results.json
  analysis/plots/v16/phase3_smap_anomaly_scores.png
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
    load_smap, AnomalyPretrainDataset, collate_anomaly_pretrain,
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
N_EPOCHS = 100  # V16: 100 epochs (vs 20 in V15)
FULL_DATA = True  # Use full 135K training set (vs 20K in V15)


def pretrain_on_smap(data: dict, n_epochs: int = N_EPOCHS, seed: int = 42):
    """Pretrain V15 JEPA on full SMAP training data."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    n_channels = data['n_channels']
    n_train = len(data['train'])
    print(f"\n=== Pretraining on SMAP ({n_channels} channels, {n_train} timesteps) ===")
    print(f"  Mode: {'full data' if FULL_DATA else '20K samples'}, {n_epochs} epochs")

    model = V15JEPA(
        n_sensors=n_channels, d_model=D_MODEL, n_heads=N_HEADS,
        n_layers=N_LAYERS, mode='sigreg',
        lambda_sig=LAMBDA_SIG, sigreg_m=256,  # more projections than V15's 128
    ).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model params: {n_params:,}")

    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, n_epochs)

    # V16: use full training data (135K samples), one pass per epoch
    # This is 6.75x more than V15's 20K samples
    n_samples = n_train * 2 if FULL_DATA else 20000  # 270K samples ~ full coverage
    ds = AnomalyPretrainDataset(
        data['train'], n_samples=min(n_samples, n_train * 5),
        seed=seed, min_context=50, max_context=100, min_horizon=5, max_horizon=20)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True,
                        collate_fn=collate_anomaly_pretrain, num_workers=0)

    run = None
    if HAS_WANDB:
        try:
            run = wandb.init(project='industrialjepa',
                             name=f'v16-smap-100ep-s{seed}',
                             tags=['v16', 'smap', 'anomaly', '100epochs'],
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


def evaluate_smap_anomaly(model, data: dict, threshold_percentile: float = 95.0):
    """
    Evaluate anomaly detection.
    Anomaly score: L1 prediction error.
    Threshold: `threshold_percentile` of scores on first 20% of test data (normal heuristic).
    """
    print("\n=== Evaluating SMAP Anomaly Detection ===")
    model.eval()

    # Compute anomaly scores on test data
    test_scores = compute_anomaly_scores(model, data['test'])
    labels = data['labels']

    # Compute threshold on test data (use score distribution)
    threshold = float(np.percentile(test_scores, threshold_percentile))
    print(f"  Anomaly score stats: min={test_scores.min():.3f}, "
          f"max={test_scores.max():.3f}, mean={test_scores.mean():.3f}, "
          f"std={test_scores.std():.3f}")
    print(f"  Threshold ({threshold_percentile}th pct): {threshold:.3f}")

    # Compute metrics
    metrics = anomaly_metrics(labels, test_scores, threshold)

    # Diagnostic: score distribution for anomaly vs normal
    anon_scores = test_scores[labels == 1]
    norm_scores = test_scores[labels == 0]
    print(f"  Anomaly window scores: mean={anon_scores.mean():.3f}, "
          f"std={anon_scores.std():.3f}")
    print(f"  Normal window scores:  mean={norm_scores.mean():.3f}, "
          f"std={norm_scores.std():.3f}")

    # Random baseline
    rng = np.random.RandomState(42)
    rand_pred = (rng.rand(len(labels)) > 0.5).astype(int)
    from sklearn.metrics import f1_score
    random_f1 = f1_score(labels, rand_pred, zero_division=0)

    print(f"\n  Results:")
    print(f"    non-PA F1: {metrics['non_pa_f1']:.4f}")
    print(f"    PA F1:     {metrics['pa_f1']:.4f}")
    print(f"    AUC-PR:    {metrics['auc_pr']:.4f}")
    print(f"    Random F1: {random_f1:.4f}")
    print(f"    V15 baseline (20ep): non-PA F1=0.069 (barely beats random=0.071)")

    return metrics, test_scores, threshold


def plot_scores(test_scores, labels, threshold, save_path):
    """Plot anomaly score distribution and time series."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    # Time series plot (first 5000 points)
    t = np.arange(min(5000, len(test_scores)))
    axes[0].plot(t, test_scores[:len(t)], 'b-', linewidth=0.5, alpha=0.7, label='anomaly score')
    axes[0].axhline(threshold, color='r', linestyle='--', linewidth=1, label=f'threshold={threshold:.3f}')
    # Shade anomaly regions
    lbl_slice = labels[:len(t)]
    for i in range(len(t) - 1):
        if lbl_slice[i] == 1:
            axes[0].axvspan(i, i + 1, alpha=0.2, color='red')
    axes[0].set_title('Anomaly Scores - First 5000 Timesteps (red=anomaly region)')
    axes[0].set_xlabel('Timestep')
    axes[0].set_ylabel('Prediction Error')
    axes[0].legend(loc='upper right')

    # Score distribution
    axes[1].hist(test_scores[labels == 0], bins=50, alpha=0.5, label='Normal', color='blue', density=True)
    axes[1].hist(test_scores[labels == 1], bins=50, alpha=0.5, label='Anomaly', color='red', density=True)
    axes[1].axvline(threshold, color='k', linestyle='--', linewidth=2, label=f'threshold')
    axes[1].set_title('Score Distribution: Normal vs Anomaly')
    axes[1].set_xlabel('Anomaly Score')
    axes[1].set_ylabel('Density')
    axes[1].legend()

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Plot saved: {save_path}")


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings('ignore')

    print("=" * 60)
    print("V16 Phase 3: SMAP Anomaly - 100 Epoch Full-Data Pretraining")
    print("V15 result: non-PA F1=0.069 (barely > random=0.071, 20 epochs)")
    print("V16 goal: > 0.10 non-PA F1")
    print("=" * 60)

    print("\nLoading SMAP data...")
    smap_data = load_smap(normalize=True)
    print(f"  Train: {smap_data['train'].shape}, Test: {smap_data['test'].shape}")
    print(f"  Anomaly rate: {smap_data['anomaly_rate']:.1%}")
    print(f"  Channels: {smap_data['n_channels']}")

    # Pretrain
    model, history = pretrain_on_smap(smap_data, n_epochs=N_EPOCHS, seed=42)

    # Evaluate
    metrics, test_scores, threshold = evaluate_smap_anomaly(model, smap_data)

    # Plot
    plot_path = PLOT_DIR / 'phase3_smap_100ep_anomaly_scores.png'
    plot_scores(test_scores, smap_data['labels'], threshold, plot_path)

    # Random baseline
    rng = np.random.RandomState(42)
    rand_pred = (rng.rand(len(smap_data['labels'])) > 0.5).astype(int)
    from sklearn.metrics import f1_score
    random_f1 = f1_score(smap_data['labels'], rand_pred, zero_division=0)

    # Save results
    results = {
        'config': 'v16_smap_100epochs',
        'n_epochs': N_EPOCHS,
        'full_data': FULL_DATA,
        'n_train_samples': len(smap_data['train']),
        'non_pa_f1': metrics['non_pa_f1'],
        'pa_f1': metrics['pa_f1'],
        'auc_pr': metrics['auc_pr'],
        'random_baseline_f1': float(random_f1),
        'anomaly_score_stats': {
            'mean': float(test_scores.mean()),
            'std': float(test_scores.std()),
            'min': float(test_scores.min()),
            'max': float(test_scores.max()),
        },
        'threshold': float(threshold),
        'loss_history': history['loss'],
        'v15_baseline_20epochs': {
            'non_pa_f1': 0.069,
            'pa_f1': 0.625,
            'auc_pr': 0.113,
            'random_f1': 0.071,
            'anomaly_score_mean': 0.838,
            'anomaly_score_std': 0.039,
        }
    }

    out_path = V16_DIR / 'phase3_smap_results.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    print("\n" + "=" * 60)
    print("V16 Phase 3 Summary")
    print("=" * 60)
    print(f"  non-PA F1: {metrics['non_pa_f1']:.4f} (V15: 0.069, random: {random_f1:.3f})")
    print(f"  PA F1:     {metrics['pa_f1']:.4f} (V15: 0.625)")
    print(f"  AUC-PR:    {metrics['auc_pr']:.4f}")
    print(f"  Score std: {test_scores.std():.3f} (V15: 0.039 - was near-constant)")
    if metrics['non_pa_f1'] > random_f1 + 0.01:
        print(f"  IMPROVEMENT vs random: +{metrics['non_pa_f1'] - random_f1:.3f}")
    else:
        print(f"  NO SIGNIFICANT IMPROVEMENT vs random")
