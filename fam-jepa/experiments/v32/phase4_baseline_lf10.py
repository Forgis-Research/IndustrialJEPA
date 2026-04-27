"""V32 Phase 4: Chronos-2 + MOMENT baselines at 10% labels.

For each dataset (with cached features) and each seed:
  1. Load cached (Xtr, ytr_tte, ttr_idx) features.
  2. Identify engine boundaries via tidx resets (where t resets to 128).
  3. Keep 10% of engines (engine-level label fraction, matches FAM lf10).
  4. Train the same Chr2MLP head used at 100% labels.
  5. Evaluate on the FULL test set; report mean h-AUROC and AUPRC.

This produces the apples-to-apples 10%-label column for Table 4.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score

FAM = Path('/home/sagemaker-user/IndustrialJEPA/fam-jepa')
CHR_FEAT = FAM / 'experiments/v24/chronos_features'
MOM_FEAT = FAM / 'experiments/v31/moment_features'
V32_RES = FAM / 'experiments/v32/results'
V32_RES.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(FAM))
sys.path.insert(0, str(FAM / 'experiments/v11'))
sys.path.insert(0, str(FAM / 'experiments/v24'))
sys.path.insert(0, str(FAM / 'experiments/v27'))
sys.path.insert(0, str(FAM / 'experiments/v29'))
sys.path.insert(0, str(FAM / 'experiments/v30'))

from evaluation.losses import build_label_surface  # noqa: E402

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SEEDS = [42, 123, 456]

# Match v30 phase1_ablation horizons (sparse, K=7 or 8)
SPARSE_HORIZONS = {
    'FD001': [1, 5, 10, 20, 50, 100, 150],
    'FD002': [1, 5, 10, 20, 50, 100, 150],
    'FD003': [1, 5, 10, 20, 50, 100, 150],
    'MBA':   [1, 5, 10, 20, 50, 100, 150, 200],
    'BATADAL': [1, 5, 10, 20, 50, 100, 150, 200],
    'GECCO': [1, 5, 10, 20, 50, 100, 150, 200],
    'MSL':   [1, 5, 10, 20, 50, 100, 150, 200],
    'PSM':   [1, 5, 10, 20, 50, 100, 150, 200],
    'SMAP':  [1, 5, 10, 20, 50, 100, 150, 200],
}


class Chr2MLP(nn.Module):
    """Same Chr2MLP as v30/phase1_ablation: (h+1) → 256 → 256 → 1."""
    def __init__(self, d_input: int, d_hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_input + 1, d_hidden), nn.GELU(),
            nn.Linear(d_hidden, d_hidden), nn.GELU(),
            nn.Linear(d_hidden, 1),
        )

    def forward(self, h: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        if dt.dim() == 1:
            dt = dt.unsqueeze(0).expand(h.shape[0], -1)
        K = dt.shape[1]
        h_exp = h.unsqueeze(1).expand(h.shape[0], K, h.shape[1])
        dt_exp = dt.unsqueeze(-1).float()
        x = torch.cat([h_exp, dt_exp], dim=-1)
        return self.net(x).squeeze(-1)


def detect_engine_boundaries(tidx: torch.Tensor) -> np.ndarray:
    """Indices where engine boundaries are (start of each engine).

    A new engine begins when tidx[i] < tidx[i-1] (the time-within-engine
    counter resets back to its minimum).
    """
    t = tidx.cpu().numpy().astype(np.int64)
    boundaries = [0]
    for i in range(1, len(t)):
        if t[i] < t[i - 1]:
            boundaries.append(i)
    boundaries.append(len(t))
    return np.asarray(boundaries, dtype=np.int64)


def subsample_engines(Xtr: torch.Tensor, ytr_tte: torch.Tensor,
                      ttr_idx: torch.Tensor, label_frac: float,
                      seed: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Engine-level subsampling: keep `label_frac` of engines."""
    if label_frac >= 1.0:
        return Xtr, ytr_tte, ttr_idx
    boundaries = detect_engine_boundaries(ttr_idx)
    n_eng = len(boundaries) - 1
    n_keep = max(1, int(round(label_frac * n_eng)))
    rng = np.random.RandomState(seed + 7777)
    keep = sorted(rng.choice(n_eng, size=n_keep, replace=False).tolist())
    keep_idx = []
    for k in keep:
        keep_idx.extend(range(boundaries[k], boundaries[k + 1]))
    keep_idx = torch.tensor(keep_idx, dtype=torch.long)
    return Xtr[keep_idx], ytr_tte[keep_idx], ttr_idx[keep_idx]


def per_horizon_metrics(p_te: np.ndarray, y_te: np.ndarray,
                        horizons: List[int]) -> Dict:
    """Per-horizon AUROC + AUPRC; mean across horizons."""
    aurocs = []; auprcs = []
    per_h = {}
    for k, h in enumerate(horizons):
        score = p_te[:, k]; label = y_te[:, k].astype(np.int32)
        if label.sum() == 0 or label.sum() == len(label):
            per_h[f'h{h}'] = {'auroc': float('nan'), 'auprc': float('nan'),
                              'pos_rate': float(label.mean())}
            continue
        au = float(roc_auc_score(label, score))
        ap = float(average_precision_score(label, score))
        aurocs.append(au); auprcs.append(ap)
        per_h[f'h{h}'] = {'auroc': au, 'auprc': ap,
                          'pos_rate': float(label.mean())}
    return {
        'mean_h_auroc': float(np.mean(aurocs)) if aurocs else float('nan'),
        'mean_h_auprc': float(np.mean(auprcs)) if auprcs else float('nan'),
        'pooled_auprc': float(average_precision_score(
            y_te.ravel().astype(np.int32), p_te.ravel())) if y_te.sum() > 0 else float('nan'),
        'per_h': per_h,
    }


def train_baseline(cache: Dict, dataset: str, seed: int, label_frac: float,
                   ft_epochs: int = 50, batch: int = 2048,
                   lr: float = 1e-3, wd: float = 1e-4) -> Dict:
    horizons = SPARSE_HORIZONS[dataset]
    h_t = torch.tensor(horizons, dtype=torch.float32, device=DEVICE)
    Xtr, ytr_tte, ttr_idx = cache['tr']
    Xva, yva_tte, _ = cache['va']
    Xte, yte_tte, te_idx = cache['te']

    Xtr, ytr_tte, ttr_idx = subsample_engines(Xtr, ytr_tte, ttr_idx, label_frac, seed)

    ytr = build_label_surface(ytr_tte.unsqueeze(1), h_t.cpu()).squeeze(1).to(DEVICE)
    yva = build_label_surface(yva_tte.unsqueeze(1), h_t.cpu()).squeeze(1).to(DEVICE)
    yte = build_label_surface(yte_tte.unsqueeze(1), h_t.cpu()).squeeze(1).to(DEVICE)
    Xtr, Xva, Xte = Xtr.to(DEVICE), Xva.to(DEVICE), Xte.to(DEVICE)

    n_pos = ytr.sum().item(); n_tot = ytr.numel()
    pw = torch.tensor(max(1.0, min(1000.0, (n_tot - n_pos) / max(n_pos, 1))),
                      device=DEVICE)

    torch.manual_seed(seed); np.random.seed(seed)
    head = Chr2MLP(d_input=Xtr.shape[1], d_hidden=256).to(DEVICE)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=wd)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, ft_epochs)
    best_state, best_val, wait = None, float('inf'), 0
    for ep in range(ft_epochs):
        head.train()
        perm = torch.randperm(len(Xtr), device=DEVICE)
        for i in range(0, len(Xtr), batch):
            idx = perm[i:i + batch]
            x = Xtr[idx]; y = ytr[idx]
            dt_grid = h_t.unsqueeze(0).expand(x.shape[0], -1)
            logits = head(x, dt_grid)
            loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pw)
            opt.zero_grad(); loss.backward(); opt.step()
        sch.step()
        head.eval()
        with torch.no_grad():
            dt_grid = h_t.unsqueeze(0).expand(Xva.shape[0], -1)
            lv = head(Xva, dt_grid)
            vl = F.binary_cross_entropy_with_logits(lv, yva, pos_weight=pw).item()
        if vl < best_val:
            best_val = vl
            best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= 10:
                break
    head.load_state_dict(best_state)
    head.eval()
    with torch.no_grad():
        dt_grid = h_t.unsqueeze(0).expand(Xte.shape[0], -1)
        p_te = torch.sigmoid(head(Xte, dt_grid)).cpu().numpy()
    yte_np = yte.cpu().numpy().astype(np.int32)
    metrics = per_horizon_metrics(p_te, yte_np, horizons)
    return {
        'dataset': dataset, 'seed': seed, 'label_frac': label_frac,
        'n_train_obs': int(Xtr.shape[0]),
        'n_train_engines': int(len(detect_engine_boundaries(
            cache['tr'][2])) - 1) if label_frac >= 1.0 else None,
        'val_loss': float(best_val), 'epochs': ep + 1,
        **metrics,
    }


def run_chr2(dataset: str, seed: int, label_frac: float) -> Optional[Dict]:
    p = CHR_FEAT / f'{dataset}_s{seed}_chronos2.pt'
    if not p.exists():
        return None
    cache = torch.load(p, map_location='cpu', weights_only=False)
    return train_baseline(cache, dataset, seed, label_frac)


def run_moment(dataset: str, seed: int, label_frac: float) -> Optional[Dict]:
    p = MOM_FEAT / f'{dataset}_s{seed}_moment.pt'
    if not p.exists():
        return None
    cache = torch.load(p, map_location='cpu', weights_only=False)
    return train_baseline(cache, dataset, seed, label_frac)


def main():
    chr2_datasets = ['FD001', 'FD002', 'FD003', 'MBA', 'BATADAL',
                     'GECCO', 'MSL', 'PSM', 'SMAP']
    moment_datasets = ['FD001', 'FD003', 'MBA', 'BATADAL']

    all_results = {'chr2': {}, 'moment': {}}
    out_path = V32_RES / 'baseline_lf10.json'

    for label_frac in [0.1, 1.0]:  # both 10% and 100% so we can audit gap
        for ds in chr2_datasets:
            for seed in SEEDS:
                t0 = time.time()
                try:
                    r = run_chr2(ds, seed, label_frac)
                    if r is None:
                        print(f'  SKIP chr2 {ds} s{seed} lf{label_frac}: no cache')
                        continue
                    r['elapsed_sec'] = time.time() - t0
                    key = f'{ds}_lf{int(label_frac*100)}_s{seed}'
                    all_results['chr2'][key] = r
                    print(f'  chr2 {ds} s{seed} lf{label_frac}: '
                          f'h-AUROC={r["mean_h_auroc"]:.3f} '
                          f'h-AUPRC={r["mean_h_auprc"]:.3f} '
                          f'(n_train={r["n_train_obs"]}, took {r["elapsed_sec"]:.0f}s)')
                except Exception as e:
                    import traceback; traceback.print_exc()
                    all_results['chr2'][f'{ds}_lf{int(label_frac*100)}_s{seed}'] = {'error': str(e)}
                with open(out_path, 'w') as f:
                    json.dump(all_results, f, indent=2)

        for ds in moment_datasets:
            for seed in SEEDS:
                t0 = time.time()
                try:
                    r = run_moment(ds, seed, label_frac)
                    if r is None:
                        print(f'  SKIP moment {ds} s{seed} lf{label_frac}: no cache')
                        continue
                    r['elapsed_sec'] = time.time() - t0
                    key = f'{ds}_lf{int(label_frac*100)}_s{seed}'
                    all_results['moment'][key] = r
                    print(f'  moment {ds} s{seed} lf{label_frac}: '
                          f'h-AUROC={r["mean_h_auroc"]:.3f} '
                          f'h-AUPRC={r["mean_h_auprc"]:.3f}')
                except Exception as e:
                    import traceback; traceback.print_exc()
                    all_results['moment'][f'{ds}_lf{int(label_frac*100)}_s{seed}'] = {'error': str(e)}
                with open(out_path, 'w') as f:
                    json.dump(all_results, f, indent=2)

    # Aggregate
    def aggr(model_key):
        out = {}
        for k, v in all_results[model_key].items():
            if 'error' in v: continue
            ds, lf, s = k.rsplit('_', 2)
            agg_key = f'{ds}_{lf}'
            out.setdefault(agg_key, {'aurocs': [], 'auprcs': [], 'pooled': []})
            out[agg_key]['aurocs'].append(v['mean_h_auroc'])
            out[agg_key]['auprcs'].append(v['mean_h_auprc'])
            out[agg_key]['pooled'].append(v['pooled_auprc'])
        for ak, vals in out.items():
            ar = np.array(vals['aurocs'])
            ap = np.array(vals['auprcs'])
            pl = np.array(vals['pooled'])
            out[ak] = {
                'mean_h_auroc': float(ar.mean()),
                'std_h_auroc': float(ar.std(ddof=1)) if len(ar) > 1 else 0.0,
                'mean_h_auprc': float(ap.mean()),
                'std_h_auprc': float(ap.std(ddof=1)) if len(ap) > 1 else 0.0,
                'pooled_auprc_mean': float(pl.mean()),
                'pooled_auprc_std': float(pl.std(ddof=1)) if len(pl) > 1 else 0.0,
                'n_seeds': int(len(ar)),
            }
        return out

    all_results['agg'] = {
        'chr2': aggr('chr2'),
        'moment': aggr('moment'),
    }
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print('\n=== SUMMARY ===')
    for model in ['chr2', 'moment']:
        print(f'\n{model.upper()}:')
        for k, v in sorted(all_results['agg'][model].items()):
            print(f'  {k}: h-AUROC={v["mean_h_auroc"]:.3f}±{v["std_h_auroc"]:.3f}, '
                  f'h-AUPRC={v["mean_h_auprc"]:.3f}±{v["std_h_auprc"]:.3f}')


if __name__ == '__main__':
    main()
