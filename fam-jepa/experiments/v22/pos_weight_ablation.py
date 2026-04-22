"""V22 extra: pos_weight ablation on SMAP.

Test the hypothesis that the aggressive pos_weight (~40) drives AUROC
below 0.5 by over-biasing the ranking toward "predict positive".

Sweeps pw in {1, 5, 10, 40 (auto)} on SMAP, single seed (42), full
entity split pipeline from anomaly_pred_ft_runner.py.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path('/home/sagemaker-user/IndustrialJEPA')
FAM = ROOT / 'fam-jepa'
V22 = FAM / 'experiments' / 'v22'
sys.path.insert(0, str(FAM))
sys.path.insert(0, str(FAM / 'experiments' / 'v11'))
sys.path.insert(0, str(FAM / 'experiments' / 'v21'))
sys.path.insert(0, str(V22))

from anomaly_pred_ft_runner import (  # noqa: E402
    DATASETS, _load_entity_split, _build_ds_from_entities,
    _load_model, WINDOW, STRIDE_TR, STRIDE_EV, TRAIN_CFG,
)
from pred_ft_utils import (  # noqa: E402
    EventHead, train_bce, evaluate_surface, estimate_pos_weight,
    save_surface, HORIZONS_STEPS, collate_anomaly_window,
)
from evaluation.surface_metrics import (  # noqa: E402
    evaluate_probability_surface, monotonicity_violation_rate,
)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MAX_FUT = max(HORIZONS_STEPS) + 1

DATASET = 'SMAP'
SEED = 42

def run_pw(pw_val: float) -> dict:
    torch.manual_seed(SEED); np.random.seed(SEED)
    sp = _load_entity_split(DATASET)
    tr_ds = _build_ds_from_entities(sp['ft_train'], WINDOW, STRIDE_TR, MAX_FUT)
    va_ds = _build_ds_from_entities(sp['ft_val'],   WINDOW, STRIDE_EV, MAX_FUT)
    te_ds = _build_ds_from_entities(sp['ft_test'],  WINDOW, STRIDE_EV, MAX_FUT)
    tr = DataLoader(tr_ds, batch_size=256, shuffle=True,
                    collate_fn=collate_anomaly_window, num_workers=0)
    va = DataLoader(va_ds, batch_size=256, shuffle=False,
                    collate_fn=collate_anomaly_window, num_workers=0)
    te = DataLoader(te_ds, batch_size=256, shuffle=False,
                    collate_fn=collate_anomaly_window, num_workers=0)
    model = _load_model(DATASET, SEED)
    head = EventHead(256).to(DEVICE)
    if pw_val == -1:
        pw = estimate_pos_weight(tr, HORIZONS_STEPS)
    else:
        pw = float(pw_val)
    t0 = time.time()
    train_out = train_bce(model, head, tr, va, mode='pred_ft',
                          pos_weight=pw, horizons_eval=HORIZONS_STEPS,
                          device=DEVICE, **TRAIN_CFG)
    surf = evaluate_surface(model, head, te, mode='pred_ft',
                            horizons=HORIZONS_STEPS, device=DEVICE)
    p, y = surf['p_surface'], surf['y_surface']
    prim = evaluate_probability_surface(p, y)
    mono = monotonicity_violation_rate(p)
    return {
        'pw_requested': pw_val, 'pw_used': float(pw),
        'auprc': float(prim['auprc']), 'auroc': float(prim['auroc']),
        'f1_best': float(prim['f1_best']),
        'precision_best': float(prim['precision_best']),
        'recall_best': float(prim['recall_best']),
        'mono_violation': float(mono['violation_rate']),
        'final_epoch': train_out['final_epoch'],
        'runtime_s': time.time() - t0,
    }


def main():
    results = []
    for pw in [-1, 1.0, 5.0, 10.0]:  # -1 => auto estimate
        print(f'\n--- pw={pw} ---', flush=True)
        r = run_pw(pw)
        print(f'  pw_used={r["pw_used"]:.2f}  AUPRC={r["auprc"]:.3f}  '
              f'AUROC={r["auroc"]:.3f}  F1={r["f1_best"]:.3f}  '
              f'P={r["precision_best"]:.3f}  R={r["recall_best"]:.3f}  '
              f'ep={r["final_epoch"]}  ({r["runtime_s"]:.0f}s)', flush=True)
        results.append(r)
    with open(V22 / 'pos_weight_ablation.json', 'w') as f:
        json.dump({'dataset': DATASET, 'seed': SEED, 'results': results},
                  f, indent=2, default=float)
    print('\nSummary:')
    print(f"{'pw_used':>10s} | {'AUPRC':>7s} | {'AUROC':>7s} | {'F1':>7s} | "
          f"{'P':>7s} | {'R':>7s}")
    print('-' * 60)
    for r in results:
        print(f"{r['pw_used']:>10.2f} | {r['auprc']:.3f}  | {r['auroc']:.3f}  | "
              f"{r['f1_best']:.3f}  | {r['precision_best']:.3f}  | {r['recall_best']:.3f}")


if __name__ == '__main__':
    main()
