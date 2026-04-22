"""V21 Phase 0: Infrastructure sanity check on SMAP.

Goals:
  1. Verify EventHead + BCE training loop works end-to-end.
  2. Train pred-FT on SMAP (1 seed) with chronological 60/10/30 split.
  3. Store p_surface + compute AUPRC / AUROC / PA-F1.
  4. Compare PA-F1 to v18 Mahalanobis baseline (0.793) — must be in ballpark.
  5. Run monotonicity sanity check on the surface.

If everything checks out, Phase 1 uses the same runner across datasets.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

# Paths
ROOT = Path('/home/sagemaker-user/IndustrialJEPA')
FAM = ROOT / 'fam-jepa'
V11 = FAM / 'experiments' / 'v11'
V21 = FAM / 'experiments' / 'v21'
CKPT_OLD = ROOT / 'mechanical-jepa' / 'experiments'

sys.path.insert(0, str(V11))
sys.path.insert(0, str(V21))
sys.path.insert(0, str(FAM))

from models import TrajectoryJEPA  # noqa: E402
from pred_ft_utils import (  # noqa: E402
    AnomalyWindowDataset, collate_anomaly_window,
    EventHead, ProbeHEvent,
    train_bce, evaluate_surface, estimate_pos_weight,
    save_surface, HORIZONS_STEPS,
)
from surface_to_legacy import anomaly_legacy_metrics  # noqa: E402
from data.smap_msl import load_smap  # noqa: E402
from evaluation.surface_metrics import (  # noqa: E402
    evaluate_probability_surface, auprc_per_horizon,
    monotonicity_violation_rate,
)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# SMAP arch (from v17 pretraining)
SMAP_ARCH = dict(n_sensors=25, patch_length=1,
                 d_model=256, n_heads=4, n_layers=2, d_ff=1024,
                 dropout=0.1, ema_momentum=0.99, predictor_hidden=1024)

WINDOW = 100  # encoder context length
STRIDE_TRAIN = 20
STRIDE_EVAL = 10
BATCH_SIZE = 64
MAX_FUTURE = max(HORIZONS_STEPS) + 1


def load_pretrained_smap(seed: int) -> TrajectoryJEPA:
    if seed == 42:
        ckpt = CKPT_OLD / 'v17' / 'ckpts' / 'v17_smap_seed42.pt'
    else:
        ckpt = CKPT_OLD / 'v18' / 'ckpts' / f'v18_smap_seed{seed}.pt'
    assert ckpt.exists(), f'missing checkpoint: {ckpt}'
    model = TrajectoryJEPA(**SMAP_ARCH).to(DEVICE)
    sd = torch.load(ckpt, map_location=DEVICE, weights_only=False)
    # Some checkpoints wrap state in {'model': sd}
    if isinstance(sd, dict) and 'model' in sd and 'context_encoder.embed.weight' not in sd:
        sd = sd['model']
    model.load_state_dict(sd, strict=False)
    return model


def make_splits(data: dict) -> dict:
    """Chronological 60/10/30 split of the test stream."""
    test = data['test']
    labels = data['labels']
    T = len(test)
    t1 = int(0.60 * T)
    t2 = int(0.70 * T)
    return {
        'ft_train_range': (WINDOW, t1),
        'ft_val_range':   (t1, t2),
        'ft_test_range':  (t2, T),
        'test_arr': test,
        'test_labels': labels,
    }


def run_phase0(seed: int = 42) -> dict:
    t0 = time.time()
    data = load_smap()
    print(f"SMAP: train={data['train'].shape} test={data['test'].shape} "
          f"anom={data['anomaly_rate']:.3f}", flush=True)

    splits = make_splits(data)
    test_arr = splits['test_arr']
    labels = splits['test_labels']

    def make_ds(range_):
        t0_, t1_ = range_
        return AnomalyWindowDataset(
            test_arr, labels, window=WINDOW, stride=STRIDE_TRAIN,
            max_future=MAX_FUTURE, t_start=t0_, t_end=t1_)

    def make_ds_eval(range_):
        t0_, t1_ = range_
        return AnomalyWindowDataset(
            test_arr, labels, window=WINDOW, stride=STRIDE_EVAL,
            max_future=MAX_FUTURE, t_start=t0_, t_end=t1_)

    tr_ds = make_ds(splits['ft_train_range'])
    va_ds = make_ds(splits['ft_val_range'])
    te_ds = make_ds_eval(splits['ft_test_range'])
    print(f"splits: tr={len(tr_ds)} va={len(va_ds)} te={len(te_ds)}", flush=True)

    tr = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True,
                    collate_fn=collate_anomaly_window, num_workers=0)
    va = DataLoader(va_ds, batch_size=BATCH_SIZE, shuffle=False,
                    collate_fn=collate_anomaly_window, num_workers=0)
    te = DataLoader(te_ds, batch_size=BATCH_SIZE, shuffle=False,
                    collate_fn=collate_anomaly_window, num_workers=0)

    # pos_weight from train labels
    pw = estimate_pos_weight(tr, HORIZONS_STEPS)
    print(f"pos_weight = {pw:.3f}", flush=True)

    # ------------------------------------------------------------------
    # Pred-FT with BCE
    # ------------------------------------------------------------------
    torch.manual_seed(seed); np.random.seed(seed)
    model = load_pretrained_smap(seed)
    head = EventHead(SMAP_ARCH['d_model']).to(DEVICE)

    print(f"training pred_ft BCE ...", flush=True)
    train_out = train_bce(model, head, tr, va, mode='pred_ft',
                          lr=1e-3, wd=1e-2, n_epochs=30, patience=5,
                          pos_weight=pw,
                          horizons_eval=HORIZONS_STEPS,
                          device=DEVICE, verbose=True)
    print(f"pred_ft done in {(time.time()-t0)/60:.1f} min, "
          f"val={train_out['best_val']:.4f} ep={train_out['final_epoch']}",
          flush=True)

    surf = evaluate_surface(model, head, te, mode='pred_ft',
                            horizons=HORIZONS_STEPS, device=DEVICE)
    # Save surface
    V21.mkdir(exist_ok=True)
    (V21 / 'surfaces').mkdir(exist_ok=True)
    surf_path = V21 / 'surfaces' / f'smap_seed{seed}_pred_ft.npz'
    save_surface(surf_path, surf['p_surface'], surf['y_surface'],
                 surf['horizons'], surf['t_index'],
                 metadata={'dataset': 'SMAP', 'seed': seed, 'mode': 'pred_ft'})

    # Primary + per-horizon + monotonicity
    prim = evaluate_probability_surface(surf['p_surface'], surf['y_surface'])
    per_h = auprc_per_horizon(surf['p_surface'], surf['y_surface'],
                              horizon_labels=HORIZONS_STEPS)
    mono = monotonicity_violation_rate(surf['p_surface'])
    legacy = anomaly_legacy_metrics(surf['p_surface'], surf['t_index'],
                                    labels, surf['horizons'],
                                    horizon_for_score=100)

    print("\n=== PHASE 0 SMAP RESULTS (pred_ft, seed=42) ===")
    print(f"  AUPRC (pooled) = {prim['auprc']:.4f}")
    print(f"  AUROC (pooled) = {prim['auroc']:.4f}")
    print(f"  F1-best        = {prim['f1_best']:.4f} (P={prim['precision_best']:.3f}, R={prim['recall_best']:.3f})")
    print(f"  prevalence     = {prim['prevalence']:.4f} ({prim['n_positive']}/{prim['n_cells']})")
    print(f"  monotonicity violation rate = {mono['violation_rate']:.4f}")
    print(f"  PA-F1 (legacy, Δt=100, on ft_test 30%): {legacy['pa_f1']:.4f} "
          f"(P={legacy['pa_precision']:.3f}, R={legacy['pa_recall']:.3f})")
    print(f"  non-PA F1 (legacy, Δt=100, on ft_test 30%): {legacy['non_pa_f1']:.4f}")
    print(f"  Per-horizon AUPRC: {[f'{v:.3f}' for v in per_h['auprc_per_k']]}")

    return {
        'dataset': 'SMAP', 'seed': seed, 'mode': 'pred_ft',
        'primary': prim, 'per_horizon': per_h,
        'monotonicity': mono, 'legacy': legacy,
        'train': {k: v for k, v in train_out.items() if k != 'losses'},
        'surface_file': str(surf_path),
        'runtime_min': (time.time() - t0) / 60,
    }


def main():
    V21.mkdir(exist_ok=True)
    out_path = V21 / 'phase0_infrastructure.json'
    print("=" * 80)
    print("V21 Phase 0: Infrastructure sanity on SMAP pred-FT (1 seed)")
    print("=" * 80, flush=True)
    res = run_phase0(seed=42)
    with open(out_path, 'w') as f:
        json.dump(res, f, indent=2, default=float)
    print(f"\nSaved: {out_path}")


if __name__ == '__main__':
    main()
