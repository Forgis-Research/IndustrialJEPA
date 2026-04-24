"""V27: compute Chronos-2 baseline p-surfaces for every dataset in the cache.

Reuses the cached 768-d Chronos-2 features at
``experiments/v24/chronos_features/<dataset>_s42_chronos2.pt`` (produced by
``experiments/v24/baseline_chronos2.py``). For each dataset we retrain the
same linear probe the v26 Chronos-2 comparison used, then apply it to the
cached test features to produce a full pooled test ``p_surface``.

Outputs:
  experiments/v27/surfaces/chronos2_<dataset>_s42.npz  (gitignored)
  experiments/v27/results/chronos2_surfaces_summary.json

The .npz format matches the FAM v26/v27 surface format so downstream code
can treat them interchangeably:
  p_surface (N, K) float32, y_surface (N, K) int8, horizons (K,) int32,
  t_index (N,) int64, meta (dict).
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

FAM_DIR = Path('/home/sagemaker-user/IndustrialJEPA/fam-jepa')
V24 = FAM_DIR / 'experiments/v24'
V27 = FAM_DIR / 'experiments/v27'
CACHE = V24 / 'chronos_features'
SURF_OUT = V27 / 'surfaces'
RES_OUT = V27 / 'results'
SURF_OUT.mkdir(parents=True, exist_ok=True)
RES_OUT.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(FAM_DIR))
sys.path.insert(0, str(V24))

from baseline_chronos2 import Probe, train_probe, get_horizons
from evaluation.losses import build_label_surface
from evaluation.surface_metrics import (
    evaluate_probability_surface, auprc_per_horizon,
)
from sklearn.metrics import roc_auc_score, average_precision_score


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def per_horizon_rows(p, y, horizons):
    rows = []
    for i, h in enumerate(horizons):
        yi = y[:, i].astype(int)
        pi = p[:, i]
        if yi.sum() == 0 or yi.sum() == len(yi):
            auroc = auprc = float('nan')
            gap = float('nan')
        else:
            auroc = float(roc_auc_score(yi, pi))
            auprc = float(average_precision_score(yi, pi))
            gap = float(pi[yi == 1].mean() - pi[yi == 0].mean())
        rows.append({'dt': int(h), 'auroc': auroc, 'auprc': auprc,
                     'pos_rate': float(yi.mean()), 'pred_gap': gap})
    return rows


def run_dataset(dataset: str, seed: int = 42) -> dict:
    cache_path = CACHE / f'{dataset}_s{seed}_chronos2.pt'
    if not cache_path.exists():
        print(f"  SKIP {dataset}: no cached features at {cache_path}")
        return None

    torch.manual_seed(seed)
    np.random.seed(seed)
    horizons = get_horizons(dataset)

    print(f"\n=== {dataset} (s{seed}) ===")
    cache = torch.load(cache_path, map_location='cpu', weights_only=False)
    Xtr, tte_tr, tidx_tr = cache['tr']
    Xva, tte_va, tidx_va = cache['va']
    Xte, tte_te, tidx_te = cache['te']
    print(f"  features: tr={tuple(Xtr.shape)} va={tuple(Xva.shape)} "
          f"te={tuple(Xte.shape)} horizons={horizons}")

    h_tensor = torch.tensor(horizons, dtype=torch.float32)
    ytr = build_label_surface(tte_tr.unsqueeze(1), h_tensor).squeeze(1)
    yva = build_label_surface(tte_va.unsqueeze(1), h_tensor).squeeze(1)
    yte = build_label_surface(tte_te.unsqueeze(1), h_tensor).squeeze(1)

    t0 = time.time()
    probe, best_val = train_probe(Xtr, ytr, Xva, yva, horizons, device=DEVICE)
    print(f"  probe: best_val={best_val:.4f} in {time.time()-t0:.1f}s")

    probe.eval()
    with torch.no_grad():
        logits = probe(Xte.to(DEVICE))
        p_te = torch.sigmoid(logits).cpu().numpy()
    y_te = yte.numpy().astype(np.int8)

    primary = evaluate_probability_surface(p_te, y_te)
    per_h = auprc_per_horizon(p_te, y_te, horizon_labels=horizons)
    rows = per_horizon_rows(p_te, y_te, horizons)

    # Save .npz (gitignored, stays on VM)
    out_npz = SURF_OUT / f'chronos2_{dataset}_s{seed}.npz'
    np.savez(out_npz,
             p_surface=p_te.astype(np.float32),
             y_surface=y_te,
             horizons=np.asarray(horizons, dtype=np.int32),
             t_index=tidx_te.numpy().astype(np.int64),
             meta=np.asarray(list({
                 'dataset': dataset, 'baseline': 'chronos-2',
                 'seed': seed, 'probe_best_val': float(best_val),
             }.items()), dtype=object))
    print(f"  wrote {out_npz}")

    print(f"  pooled AUPRC={primary['auprc']:.4f}  "
          f"AUROC={primary['auroc']:.4f}")
    for r in rows:
        print(f"    dt={r['dt']:>4}  AUROC={r['auroc']:.3f}  "
              f"AUPRC={r['auprc']:.3f}  gap={r['pred_gap']:+.3f}")

    return {
        'dataset': dataset, 'seed': seed,
        'n_train': int(len(Xtr)), 'n_test': int(len(Xte)),
        'probe_best_val': float(best_val),
        'pooled_auprc': float(primary['auprc']),
        'pooled_auroc': float(primary['auroc']),
        'per_horizon': rows,
        'surface_path': str(out_npz),
    }


def main():
    datasets = ['FD001', 'FD002', 'FD003', 'SMAP', 'MSL', 'PSM', 'MBA',
                'GECCO', 'BATADAL']
    results = {}
    for ds in datasets:
        r = run_dataset(ds, seed=42)
        if r is not None:
            results[ds] = r

    out = RES_OUT / 'chronos2_surfaces_summary.json'
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {out}")

    print(f"\n=== SUMMARY (Chronos-2 baseline, s42) ===")
    print(f"  {'dataset':<10}  {'AUPRC':>8}  {'AUROC':>8}")
    for ds, r in results.items():
        print(f"  {ds:<10}  {r['pooled_auprc']:>8.4f}  {r['pooled_auroc']:>8.4f}")


if __name__ == '__main__':
    main()
