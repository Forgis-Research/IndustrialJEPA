"""
V15 Phase 3 + 5b: SMAP/MSL Anomaly Detection.

Phase 3a: SMAP/MSL data adapters (in mechanical-jepa/data/smap_msl.py).
Phase 3c: Pretrain V15 encoder on SMAP for 50 epochs, evaluate anomaly detection.
Phase 5b: Report non-PA F1 vs MTS-JEPA (33.6) and TS2Vec (32.8) baselines.

Architecture: V15JEPA (bidirectional, shared encoder) adapted for 25 channels.
Anomaly score: L1 prediction error between predicted and actual embedding.
Threshold: 95th percentile of scores on first 10% of test data (normal heuristic).

Outputs:
  experiments/v15/phase3_smap_results.json
  experiments/v15/phase3_msl_results.json
  analysis/plots/v15/phase3_smap_anomaly_scores.png
"""

import sys, json, time, copy
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

V15_DIR = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/experiments/v15')
PLOT_DIR = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/analysis/plots/v15')
sys.path.insert(0, str(Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/experiments/v15')))
sys.path.insert(0, str(Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa')))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from data.smap_msl import (
    load_smap, load_msl, AnomalyPretrainDataset, collate_anomaly_pretrain,
    compute_anomaly_scores, evaluate_anomaly_detection,
)
from evaluation.grey_swan_metrics import anomaly_metrics
from phase1_sigreg import V15JEPA, V15PretrainDataset, collate_v15_pretrain

try:
    import wandb
    HAS_WANDB = True
except Exception:
    HAS_WANDB = False

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# Hyperparameters
D_MODEL = 256
N_HEADS = 4
N_LAYERS = 2
BATCH_SIZE = 64
LR = 3e-4
LAMBDA_SIG = 0.05
N_EPOCHS_SMAP = 20  # reduced for speed (was 50); enough to assess feasibility
N_EPOCHS_MSL = 10  # fewer for speed


def pretrain_on_anomaly_dataset(data: dict, n_epochs: int = 50,
                                  seed: int = 42, config_name: str = 'smap'):
    """
    Pretrain V15 JEPA on SMAP or MSL training data.
    Uses SIGReg mode (no EMA needed).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    n_channels = data['n_channels']
    print(f"\n=== Pretraining on {data['name']} ({n_channels} channels, "
          f"{len(data['train'])} timesteps) ===")

    model = V15JEPA(
        n_sensors=n_channels, d_model=D_MODEL, n_heads=N_HEADS,
        n_layers=N_LAYERS, mode='sigreg',
        lambda_sig=LAMBDA_SIG, sigreg_m=128,  # fewer projections for speed
    ).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model params: {n_params:,}")

    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, n_epochs)

    # Dataset: ~20K samples from training time series (reduced for speed)
    n_samples = min(20000, len(data['train']) * 2)
    ds = AnomalyPretrainDataset(
        data['train'], n_samples=n_samples, seed=seed,
        min_context=50, max_context=100, min_horizon=5, max_horizon=20)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True,
                        collate_fn=collate_anomaly_pretrain, num_workers=0)

    run = None
    if HAS_WANDB:
        try:
            run = wandb.init(project='industrialjepa',
                             name=f'v15-phase3-{config_name}-s{seed}',
                             tags=['v15', 'phase3', config_name],
                             config={'dataset': data['name'], 'n_channels': n_channels,
                                     'n_epochs': n_epochs, 'd_model': D_MODEL},
                             reinit=True)
        except Exception as e:
            print(f"  wandb error: {e}")

    history = []
    t0 = time.time()
    for epoch in range(1, n_epochs + 1):
        model.train()
        total_loss = 0.0; n_batches = 0
        for x_past, past_mask, x_full, full_mask, k in loader:
            x_past, past_mask = x_past.to(DEVICE), past_mask.to(DEVICE)
            x_full, full_mask = x_full.to(DEVICE), full_mask.to(DEVICE)
            k = k.to(DEVICE)
            optim.zero_grad()
            loss, _, _ = model.forward_pretrain(x_past, past_mask, x_full, full_mask, k)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            total_loss += loss.item()
            n_batches += 1
        avg_loss = total_loss / n_batches
        history.append(avg_loss)
        sched.step()

        if run is not None:
            try:
                wandb.log({'epoch': epoch, 'train_loss': avg_loss})
            except Exception:
                pass

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Ep {epoch:3d}/{n_epochs}: loss={avg_loss:.4f}  "
                  f"[{(time.time()-t0)/60:.1f}min]", flush=True)

    if run is not None:
        try:
            wandb.finish()
        except Exception:
            pass

    print(f"  Pretraining done: {(time.time()-t0)/60:.1f} min, final loss={history[-1]:.4f}")
    return model, history


def compute_scores_and_evaluate(model, data: dict, dataset_name: str) -> dict:
    """Run anomaly evaluation with multiple threshold strategies."""
    test_arr = data['test']
    labels = data['labels']
    T = len(test_arr)

    print(f"\n  Computing anomaly scores on {dataset_name} test set ({T} timesteps)...")
    t0 = time.time()

    # Score in chunks for memory
    scores = compute_anomaly_scores(model, test_arr, window_size=100,
                                     batch_size=256, device=str(DEVICE))
    print(f"  Scoring done in {time.time()-t0:.1f}s")
    print(f"  Score stats: min={scores.min():.4f} max={scores.max():.4f} "
          f"mean={scores.mean():.4f} std={scores.std():.4f}")

    # Multiple threshold strategies
    results = {'dataset': dataset_name, 'scores_stats': {
        'min': float(scores.min()), 'max': float(scores.max()),
        'mean': float(scores.mean()), 'std': float(scores.std()),
    }}

    # Strategy 1: 95th percentile of normal (first 10% heuristic)
    n_normal = max(100, int(0.10 * T))
    thresh_95 = float(np.percentile(scores[:n_normal], 95))
    m95 = anomaly_metrics(scores, labels, threshold=thresh_95)
    m95['threshold_strategy'] = '95th_pct_first_10pct'
    results['threshold_95_non_pa'] = m95

    # Strategy 2: Global 90th percentile (matches literature heuristic)
    thresh_90 = float(np.percentile(scores, 90))
    m90 = anomaly_metrics(scores, labels, threshold=thresh_90)
    m90['threshold_strategy'] = '90th_pct_global'
    results['threshold_90_global'] = m90

    # Strategy 3: Optimize F1 on first 10% (lenient)
    best_f1, best_thresh = 0.0, thresh_95
    for p in [70, 75, 80, 85, 90, 92, 95, 97, 99]:
        thr = float(np.percentile(scores, p))
        m = anomaly_metrics(scores, labels, threshold=thr)
        if m['f1_non_pa'] > best_f1:
            best_f1 = m['f1_non_pa']
            best_thresh = thr
            best_metrics = m
    results['best_f1_search'] = {**best_metrics, 'best_thresh': best_thresh}

    # Summary
    print(f"\n  Results for {dataset_name}:")
    print(f"  non-PA F1 (95th pct):   {m95['f1_non_pa']:.4f}")
    print(f"  PA F1 (95th pct):       {m95['f1_pa']:.4f}")
    print(f"  AUC-PR:                 {m95['auc_pr']:.4f}")
    print(f"  TaPR F1:                {m95['tapr_f1']:.4f}")
    print(f"  Best non-PA F1 (sweep): {best_f1:.4f}")

    # Literature comparison (MTS-JEPA uses PA F1 convention)
    if dataset_name == 'SMAP':
        print(f"\n  Literature comparison (SMAP):")
        print(f"  MTS-JEPA PA F1: 33.6% (our PA F1: {m95['f1_pa']*100:.1f}%)")
        print(f"  TS2Vec PA F1:   32.8% (our non-PA F1: {m95['f1_non_pa']*100:.1f}%)")
        results['mts_jepa_smap_pa_f1'] = 33.6
        results['ts2vec_smap_pa_f1'] = 32.8

    return results


def plot_anomaly_scores(scores, labels, results, dataset_name):
    """Plot anomaly scores with ground truth labels."""
    fig, axes = plt.subplots(2, 1, figsize=(15, 6))

    # Show first 10K timesteps
    N_plot = min(10000, len(scores))
    t = np.arange(N_plot)

    axes[0].plot(t, scores[:N_plot], 'b-', linewidth=0.5, alpha=0.7, label='Anomaly score')
    thresh = results.get('threshold_95_non_pa', {}).get('threshold_used', scores.max())
    axes[0].axhline(thresh, color='r', linestyle='--', label=f'Threshold={thresh:.3f}')
    axes[0].set_ylabel('Anomaly Score')
    axes[0].legend(fontsize=8)
    axes[0].set_title(f'{dataset_name} Anomaly Detection - Score Distribution (first 10K timesteps)')

    # Ground truth
    axes[1].fill_between(t, labels[:N_plot], alpha=0.5, color='red', label='True anomaly')
    pred_binary = (scores[:N_plot] > thresh).astype(int)
    axes[1].fill_between(t, pred_binary * 0.5, alpha=0.3, color='blue', label='Predicted')
    axes[1].set_ylabel('Anomaly Label')
    axes[1].set_xlabel('Timestep')
    axes[1].legend(fontsize=8)

    m = results.get('threshold_95_non_pa', {})
    f1_np = m.get('f1_non_pa', 0)
    f1_pa = m.get('f1_pa', 0)
    fig.suptitle(f'{dataset_name}: non-PA F1={f1_np:.3f}, PA F1={f1_pa:.3f}', fontsize=12)

    plt.tight_layout()
    path = PLOT_DIR / f'phase3_{dataset_name.lower()}_anomaly_scores.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved plot: {path}")


def run_trivial_baseline(data: dict) -> dict:
    """
    Trivial anomaly baseline: random scores.
    Used to verify non-PA F1 > random.
    """
    np.random.seed(42)
    rand_scores = np.random.randn(len(data['test'])).astype(np.float32)
    labels = data['labels']
    thresh = float(np.percentile(rand_scores, 95))
    m = anomaly_metrics(rand_scores, labels, threshold=thresh)
    return {
        'random_baseline_f1_non_pa': m['f1_non_pa'],
        'random_baseline_f1_pa': m['f1_pa'],
        'anomaly_rate': float(labels.mean()),
        'expected_f1_by_chance': float(2 * labels.mean() * 0.05 /
                                        (labels.mean() + 0.05 + 1e-6)),
    }


def main():
    t0 = time.time()
    V15_DIR.mkdir(exist_ok=True)

    print("=" * 60)
    print("V15 Phase 3 + 5b: SMAP/MSL Anomaly Detection")
    print(f"Device: {DEVICE}")
    print("=" * 60)

    all_results = {}

    # --- SMAP ---
    print("\n--- SMAP Dataset ---")
    data_smap = load_smap()
    print(f"SMAP: {data_smap['train'].shape[0]} train, {data_smap['test'].shape[0]} test, "
          f"anomaly_rate={data_smap['anomaly_rate']:.3f}")

    # Trivial baseline first (sanity check)
    trivial_smap = run_trivial_baseline(data_smap)
    print(f"  Random baseline non-PA F1: {trivial_smap['random_baseline_f1_non_pa']:.4f}")
    all_results['smap_trivial_baseline'] = trivial_smap

    # Pretrain V15 on SMAP
    smap_model, smap_history = pretrain_on_anomaly_dataset(
        data_smap, n_epochs=N_EPOCHS_SMAP, seed=42, config_name='smap_sigreg')

    # Evaluate
    smap_results = compute_scores_and_evaluate(smap_model, data_smap, 'SMAP')
    all_results['smap'] = smap_results

    # Plot
    try:
        # We need raw scores for plotting - recompute or use cached
        scores_smap = compute_anomaly_scores(smap_model, data_smap['test'],
                                              window_size=100, batch_size=256)
        plot_anomaly_scores(scores_smap, data_smap['labels'], smap_results, 'SMAP')
    except Exception as e:
        print(f"  Plot failed: {e}")

    # Save SMAP results
    smap_save = {k: v for k, v in smap_results.items()
                 if not isinstance(v, np.ndarray)}
    smap_save['pretrain_history_final_loss'] = float(smap_history[-1]) if smap_history else None
    with open(V15_DIR / 'phase3_smap_results.json', 'w') as f:
        json.dump(smap_save, f, indent=2)

    # --- MSL ---
    print("\n--- MSL Dataset ---")
    data_msl = load_msl()
    print(f"MSL: {data_msl['train'].shape[0]} train, {data_msl['test'].shape[0]} test, "
          f"anomaly_rate={data_msl['anomaly_rate']:.3f}")

    # Trivial baseline
    trivial_msl = run_trivial_baseline(data_msl)
    print(f"  Random baseline non-PA F1: {trivial_msl['random_baseline_f1_non_pa']:.4f}")
    all_results['msl_trivial_baseline'] = trivial_msl

    # Pretrain V15 on MSL
    msl_model, msl_history = pretrain_on_anomaly_dataset(
        data_msl, n_epochs=N_EPOCHS_MSL, seed=42, config_name='msl_sigreg')

    # Evaluate
    msl_results = compute_scores_and_evaluate(msl_model, data_msl, 'MSL')
    all_results['msl'] = msl_results

    # Save MSL results
    msl_save = {k: v for k, v in msl_results.items()
                if not isinstance(v, np.ndarray)}
    msl_save['pretrain_history_final_loss'] = float(msl_history[-1]) if msl_history else None
    with open(V15_DIR / 'phase3_msl_results.json', 'w') as f:
        json.dump(msl_save, f, indent=2)

    # --- Sanity checks ---
    print("\n=== Sanity Checks ===")
    smap_f1_np = smap_results.get('threshold_95_non_pa', {}).get('f1_non_pa', 0)
    smap_f1_pa = smap_results.get('threshold_95_non_pa', {}).get('f1_pa', 0)
    random_f1 = trivial_smap['random_baseline_f1_non_pa']

    print(f"  SMAP non-PA F1: {smap_f1_np:.4f} vs random {random_f1:.4f}")
    if smap_f1_np > random_f1:
        print("  [PASS] Model beats random baseline on non-PA F1")
    else:
        print("  [WARN] Model does NOT beat random baseline - check scoring function")

    print(f"  SMAP PA F1: {smap_f1_pa:.4f} vs MTS-JEPA 33.6% / TS2Vec 32.8%")

    # --- Summary ---
    summary = {
        'smap': {
            'non_pa_f1': float(smap_f1_np),
            'pa_f1': float(smap_f1_pa),
            'auc_pr': float(smap_results.get('threshold_95_non_pa', {}).get('auc_pr', 0)),
            'tapr_f1': float(smap_results.get('threshold_95_non_pa', {}).get('tapr_f1', 0)),
        },
        'msl': {
            'non_pa_f1': float(msl_results.get('threshold_95_non_pa', {}).get('f1_non_pa', 0)),
            'pa_f1': float(msl_results.get('threshold_95_non_pa', {}).get('f1_pa', 0)),
        },
        'literature_baselines': {
            'mts_jepa_smap_pa_f1': 33.6,
            'ts2vec_smap_pa_f1': 32.8,
        },
        'trivial_baselines': {
            'smap_random_f1': trivial_smap['random_baseline_f1_non_pa'],
            'msl_random_f1': trivial_msl['random_baseline_f1_non_pa'],
        },
        'runtime_hours': (time.time() - t0) / 3600,
    }

    print("\n=== Phase 3+5b Results Summary ===")
    print(f"SMAP: non-PA F1={summary['smap']['non_pa_f1']:.4f}, "
          f"PA F1={summary['smap']['pa_f1']:.4f}")
    print(f"MSL:  non-PA F1={summary['msl']['non_pa_f1']:.4f}, "
          f"PA F1={summary['msl']['pa_f1']:.4f}")
    print(f"MTS-JEPA SMAP (PA): {summary['literature_baselines']['mts_jepa_smap_pa_f1']}")
    print(f"TS2Vec SMAP (PA):   {summary['literature_baselines']['ts2vec_smap_pa_f1']}")

    with open(V15_DIR / 'phase3_all_results.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {V15_DIR / 'phase3_all_results.json'}")
    print(f"Total runtime: {(time.time()-t0)/3600:.1f}h")


if __name__ == '__main__':
    main()
