"""V27: dense-Δt pooled-test surfaces for FAM v26, FAM v27 'none', and Chronos-2.

Re-evaluates each pred-FT checkpoint on the same deterministic test loader at
EVERY integer Δt in [1, max_h]. The FAM predictor takes Δt as a continuous
scalar so dense evaluation is just a horizon swap; Chronos-2's linear probe
needs to be retrained with K=max_h output heads on the cached features
(probe training is ~10s per dataset).

Outputs (all gitignored .npz under experiments/v27/surfaces/):
  dense_fam_v26_<ds>_s42.npz   — FAM v26 revin
  dense_fam_v27_<ds>_s42.npz   — FAM v27 'none' (C-MAPSS only)
  dense_chronos2_<ds>_s42.npz  — Chronos-2 linear probe, dense horizons

Also writes a JSON summary at experiments/v27/results/dense_pooled_summary.json.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

FAM_DIR = Path('/home/sagemaker-user/IndustrialJEPA/fam-jepa')
V24 = FAM_DIR / 'experiments/v24'
V26 = FAM_DIR / 'experiments/v26'
V27 = FAM_DIR / 'experiments/v27'
CACHE = V24 / 'chronos_features'
SURF = V27 / 'surfaces'
RES = V27 / 'results'
SURF.mkdir(parents=True, exist_ok=True)
RES.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(FAM_DIR))
sys.path.insert(0, str(V24))
sys.path.insert(0, str(V27))

from model import FAM
from train import collate_event, evaluate
from _runner import LOADERS, _global_zscore, _build_event_concat
from baseline_chronos2 import Probe, train_probe
from evaluation.losses import build_label_surface
from sklearn.metrics import roc_auc_score, average_precision_score

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def dense_horizons(dataset: str):
    if dataset.startswith('FD'):
        return list(range(1, 151))     # match training max_h on C-MAPSS
    return list(range(1, 201))         # anomaly datasets


def pooled_metrics(p, y):
    yi = y.ravel().astype(int)
    if yi.sum() == 0 or yi.sum() == len(yi):
        return float('nan'), float('nan')
    return (float(average_precision_score(yi, p.ravel())),
            float(roc_auc_score(yi, p.ravel())))


# ---------------------------------------------------------------------------
# FAM dense eval
# ---------------------------------------------------------------------------

def _fam_test_loader(dataset: str, norm_mode: str):
    bundle = LOADERS[dataset]()
    if norm_mode == 'none':
        bundle = _global_zscore(bundle)
    test_ft = _build_event_concat(bundle['ft_test'], stride=1, max_context=512)
    return DataLoader(test_ft, batch_size=128, shuffle=False,
                      collate_fn=collate_event), bundle['n_channels']


def fam_dense_surface(dataset: str, norm_mode: str, ckpt_path: Path,
                      out_path: Path) -> dict:
    if not ckpt_path.exists():
        print(f"  SKIP {dataset} {norm_mode}: ckpt missing ({ckpt_path})")
        return None
    horizons = dense_horizons(dataset)
    print(f"  FAM {norm_mode} on {dataset}: K={len(horizons)} dense horizons")
    test_loader, n_ch = _fam_test_loader(dataset, norm_mode)

    model = FAM(n_channels=n_ch, norm_mode=norm_mode).to(DEVICE).eval()
    state = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(state)

    t0 = time.time()
    out = evaluate(model, test_loader, horizons, mode='pred_ft', device=DEVICE)
    p, y = out['p_surface'], out['y_surface']
    auprc, auroc = pooled_metrics(p, y)
    print(f"    pooled AUPRC={auprc:.4f}  AUROC={auroc:.4f}  "
          f"({time.time()-t0:.1f}s)")

    np.savez(out_path,
             p_surface=p.astype(np.float32),
             y_surface=y.astype(np.int8),
             horizons=np.asarray(horizons, dtype=np.int32),
             t_index=out['t_index'].astype(np.int64),
             meta=np.asarray(list({'dataset': dataset, 'norm_mode': norm_mode,
                                   'seed': 42, 'kind': 'fam_dense_pooled',
                                   'ckpt_path': str(ckpt_path)}.items()),
                             dtype=object))
    print(f"    wrote {out_path.relative_to(FAM_DIR)}")
    return {'auprc': auprc, 'auroc': auroc, 'K': len(horizons)}


# ---------------------------------------------------------------------------
# Chronos-2 dense probe + eval
# ---------------------------------------------------------------------------

def chronos_dense_surface(dataset: str, out_path: Path) -> dict:
    cache_path = CACHE / f'{dataset}_s42_chronos2.pt'
    if not cache_path.exists():
        print(f"  SKIP Chronos-2 {dataset}: no cached features")
        return None
    horizons = dense_horizons(dataset)
    print(f"  Chronos-2 on {dataset}: K={len(horizons)} dense horizons")
    cache = torch.load(cache_path, map_location='cpu', weights_only=False)
    Xtr, tte_tr, _ = cache['tr']
    Xva, tte_va, _ = cache['va']
    Xte, tte_te, tidx_te = cache['te']

    h = torch.tensor(horizons, dtype=torch.float32)
    ytr = build_label_surface(tte_tr.unsqueeze(1), h).squeeze(1)
    yva = build_label_surface(tte_va.unsqueeze(1), h).squeeze(1)
    yte = build_label_surface(tte_te.unsqueeze(1), h).squeeze(1)

    torch.manual_seed(42)
    np.random.seed(42)
    t0 = time.time()
    probe, best_val = train_probe(Xtr, ytr, Xva, yva, horizons, device=DEVICE)
    probe.eval()
    with torch.no_grad():
        p = torch.sigmoid(probe(Xte.to(DEVICE))).cpu().numpy()
    y = yte.numpy().astype(np.int8)
    auprc, auroc = pooled_metrics(p, y)
    print(f"    probe best_val={best_val:.4f}  pooled AUPRC={auprc:.4f}  "
          f"AUROC={auroc:.4f}  ({time.time()-t0:.1f}s)")

    np.savez(out_path,
             p_surface=p.astype(np.float32),
             y_surface=y,
             horizons=np.asarray(horizons, dtype=np.int32),
             t_index=tidx_te.numpy().astype(np.int64),
             meta=np.asarray(list({'dataset': dataset, 'baseline': 'chronos-2',
                                   'seed': 42, 'kind': 'chronos2_dense_pooled',
                                   'probe_best_val': float(best_val)}.items()),
                             dtype=object))
    print(f"    wrote {out_path.relative_to(FAM_DIR)}")
    return {'auprc': auprc, 'auroc': auroc, 'K': len(horizons)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    summary = {}
    cmapss = ['FD001', 'FD002', 'FD003']
    anomaly = ['SMAP', 'MSL', 'PSM', 'MBA']

    for ds in cmapss + anomaly:
        print(f"\n=== {ds} ===")
        rec = {}

        # FAM v26 revin (use v26 ckpts)
        v26_ckpt = V26 / 'ckpts' / f'{ds}_s42_pred_ft.pt'
        out = SURF / f'dense_fam_v26_{ds}_s42.npz'
        r = fam_dense_surface(ds, 'revin', v26_ckpt, out)
        if r: rec['fam_v26_revin'] = r

        # FAM v27 none (only C-MAPSS)
        if ds in cmapss:
            v27_ckpt = V27 / 'ckpts' / f'{ds}_none_s42_pred_ft.pt'
            out = SURF / f'dense_fam_v27_{ds}_s42.npz'
            r = fam_dense_surface(ds, 'none', v27_ckpt, out)
            if r: rec['fam_v27_none'] = r

        # Chronos-2
        out = SURF / f'dense_chronos2_{ds}_s42.npz'
        r = chronos_dense_surface(ds, out)
        if r: rec['chronos2'] = r

        summary[ds] = rec

    # Also do GECCO + BATADAL for Chronos-2 only (FAM not run on these)
    for ds in ['GECCO', 'BATADAL']:
        print(f"\n=== {ds} (Chronos-2 only) ===")
        out = SURF / f'dense_chronos2_{ds}_s42.npz'
        r = chronos_dense_surface(ds, out)
        if r:
            summary[ds] = {'chronos2': r}

    out_json = RES / 'dense_pooled_summary.json'
    with open(out_json, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nwrote {out_json}")

    print(f"\n=== DENSE POOLED SUMMARY ===")
    for ds, rec in summary.items():
        print(f"  {ds}:")
        for model, r in rec.items():
            print(f"    {model:<18} K={r['K']:>4}  "
                  f"AUPRC={r['auprc']:.4f}  AUROC={r['auroc']:.4f}")


if __name__ == '__main__':
    main()
