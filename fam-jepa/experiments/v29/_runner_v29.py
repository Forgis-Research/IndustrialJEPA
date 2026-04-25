"""V29 shared runner: v28 + new datasets (SKAB, ETTm1, CHB-MIT) +
predictor_kind option ('mlp' | 'transformer').

Imports v28 LOADERS (which already covers v27 set + GECCO + BATADAL),
adds the three new dataset loaders, and exposes ``run_one`` with a
single ``predictor_kind`` flag that wires through to FAM.

Ground rule from CLAUDE.md/SESSION_PROMPT: import model + train code,
do NOT copy. The transformer predictor lives in fam-jepa/model.py.
"""

from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset
from sklearn.metrics import roc_auc_score, average_precision_score

FAM_DIR = Path('/home/sagemaker-user/IndustrialJEPA/fam-jepa')
V29_DIR = FAM_DIR / 'experiments/v29'
CKPT_DIR = V29_DIR / 'ckpts'
SURF_DIR = V29_DIR / 'surfaces'
LOG_DIR = V29_DIR / 'logs'
RES_DIR = V29_DIR / 'results'
for d in [CKPT_DIR, SURF_DIR, LOG_DIR, RES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(FAM_DIR))
sys.path.insert(0, str(FAM_DIR / 'experiments/v24'))
sys.path.insert(0, str(FAM_DIR / 'experiments/v27'))
sys.path.insert(0, str(FAM_DIR / 'experiments/v28'))

from model import FAM
from train import (
    PretrainDataset, EventDataset, collate_pretrain, collate_event,
    pretrain as pretrain_default, finetune as finetune_default,
    evaluate, save_surface,
)
from _runner import (
    LOADERS as V27_LOADERS, _global_zscore, per_horizon_diag, print_diag,
    _build_event_concat,
)
from runner_v28 import LOADERS as V28_LOADERS  # adds GECCO, BATADAL

CMAPSS_HORIZONS = [1, 5, 10, 20, 50, 100, 150]
ANOMALY_HORIZONS = [1, 5, 10, 20, 50, 100, 150, 200]
CHBMIT_HORIZONS = [32, 160, 320, 960, 1920, 3840, 9600]   # at 32 Hz


# ---------------------------------------------------------------------------
# New dataset adapters → unified bundle format
# ---------------------------------------------------------------------------

def _skab() -> Dict:
    from data.skab import load_skab
    d = load_skab(normalize=False)
    return {
        'pretrain_seqs': d['pretrain_stream'],
        'ft_train': d['ft_train'], 'ft_val': d['ft_val'], 'ft_test': d['ft_test'],
        'n_channels': d['n_channels'], 'horizons': ANOMALY_HORIZONS,
        'subset': 'SKAB',
    }


def _ettm1() -> Dict:
    from data.ettm import load_ettm1
    d = load_ettm1(normalize=False)
    return {
        'pretrain_seqs': d['pretrain_stream'],
        'ft_train': d['ft_train'], 'ft_val': d['ft_val'], 'ft_test': d['ft_test'],
        'n_channels': d['n_channels'], 'horizons': ANOMALY_HORIZONS,
        'subset': 'ETTm1',
    }


def _chbmit() -> Dict:
    from data.chbmit import load_chbmit
    d = load_chbmit(normalize=False)
    return {
        'pretrain_seqs': d['pretrain_stream'],
        'ft_train': d['ft_train'], 'ft_val': d['ft_val'], 'ft_test': d['ft_test'],
        'n_channels': d['n_channels'], 'horizons': CHBMIT_HORIZONS,
        'subset': 'CHBMIT',
    }


LOADERS = dict(V28_LOADERS)
LOADERS['SKAB'] = _skab
LOADERS['ETTm1'] = _ettm1
LOADERS['CHBMIT'] = _chbmit


# Norm-mode policy (lifecycle vs streaming) per the architecture spec.
NORM_POLICY = {
    'FD001': 'none', 'FD002': 'none', 'FD003': 'none',
    'SMAP': 'revin', 'MSL': 'revin', 'PSM': 'revin', 'SMD': 'revin',
    'MBA': 'revin', 'GECCO': 'revin', 'BATADAL': 'revin',
    'SKAB': 'revin', 'ETTm1': 'revin', 'CHBMIT': 'revin',
}


# ---------------------------------------------------------------------------
# Honest-metrics helper (mean per-horizon AUROC + base-rate baseline)
# ---------------------------------------------------------------------------

def honest_metrics(p_surface: np.ndarray, y_surface: np.ndarray,
                   horizons: List[int]) -> Dict:
    """Returns dict with mean h-AUROC, pooled AUPRC, and base-rate baselines."""
    valid = [i for i in range(len(horizons)) if 0 < y_surface[:, i].mean() < 1]
    mean_h_auroc = float(np.mean([
        roc_auc_score(y_surface[:, i], p_surface[:, i]) for i in valid
    ])) if valid else float('nan')
    pooled_auprc = float(average_precision_score(
        y_surface.ravel(), p_surface.ravel()))
    base_rates = y_surface.mean(axis=0)
    rng = np.random.RandomState(0)
    p_base = (np.tile(base_rates, (y_surface.shape[0], 1))
              + rng.normal(0, 1e-6, y_surface.shape))
    base_pooled = float(average_precision_score(
        y_surface.ravel(), p_base.ravel()))
    base_h_auroc = float(np.mean([
        roc_auc_score(y_surface[:, i], p_base[:, i]) for i in valid
    ])) if valid else float('nan')
    return {
        'mean_h_auroc': mean_h_auroc,
        'mean_h_auroc_base': base_h_auroc,
        'pooled_auprc': pooled_auprc,
        'pooled_auprc_base': base_pooled,
    }


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_one(dataset: str, seed: int,
            predictor_kind: str = 'mlp',
            norm_mode: Optional[str] = None,
            tag_suffix: str = '',
            pre_epochs: int = 30, pre_patience: int = 5,
            ft_epochs: int = 30, ft_patience: int = 8,
            n_cuts_train: int = 40, n_cuts_val: int = 10,
            max_context: int = 512,
            pre_batch: int = 64, ft_batch: int = 128,
            device: str = 'cuda') -> Dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if norm_mode is None:
        norm_mode = NORM_POLICY[dataset]
    extra = '' if predictor_kind == 'mlp' else '_xpred'
    if tag_suffix:
        extra += f'_{tag_suffix}'
    tag = f"{dataset}_{norm_mode}{extra}_s{seed}"
    print(f"\n{'='*70}\n=== {tag} (predictor={predictor_kind})\n{'='*70}",
          flush=True)

    bundle = LOADERS[dataset]()
    if norm_mode == 'none':
        bundle = _global_zscore(bundle)
    horizons = bundle['horizons']
    n_channels = bundle['n_channels']
    print(f"  loaded {dataset}: pretrain_seqs={len(bundle['pretrain_seqs'])}, "
          f"n_channels={n_channels}, horizons={horizons}", flush=True)

    # 1. Build model
    model = FAM(n_channels=n_channels, patch_size=16, d_model=256,
                n_heads=4, n_layers=2, d_ff=256, dropout=0.1,
                ema_momentum=0.99, predictor_hidden=256,
                norm_mode=norm_mode, predictor_kind=predictor_kind)
    n_pred = sum(p.numel() for p in model.predictor.parameters())
    n_total = sum(p.numel() for p in model.parameters())
    print(f"  predictor_kind={predictor_kind}  predictor_params={n_pred:,}  "
          f"total={n_total:,}", flush=True)

    pre_ckpt = CKPT_DIR / f'{tag}_pretrain.pt'
    pre_time, pre_best, pre_hist_len = 0.0, float('nan'), 0

    # delta_t for pretraining. Capped at 960 for CHB-MIT (30s @ 32Hz) to keep
    # target-encoder memory in check; the predictor extrapolates to longer
    # horizons during eval via its smooth Δt embedding (same trick the v27
    # pipeline uses to evaluate at Δt=200 from a delta_t_max=150 pretrain).
    pretrain_delta_t_max = 960 if dataset == 'CHBMIT' else max(horizons)
    eval_max_future = max(horizons)

    if pre_ckpt.exists():
        print(f"  [pretrain] ckpt exists: {pre_ckpt.name}", flush=True)
        model.load_state_dict(torch.load(pre_ckpt, map_location='cpu'))
    else:
        train_pre = PretrainDataset(bundle['pretrain_seqs'],
                                    n_cuts=n_cuts_train,
                                    max_context=max_context,
                                    delta_t_max=pretrain_delta_t_max,
                                    delta_t_min=1, seed=seed)
        val_seqs = {}
        for k, seq in bundle['pretrain_seqs'].items():
            L = len(seq)
            cut = int(0.9 * L)
            if L - cut >= 128:
                val_seqs[k] = seq[cut:]
        if not val_seqs:
            val_seqs = bundle['pretrain_seqs']
        val_pre = PretrainDataset(val_seqs, n_cuts=n_cuts_val,
                                  max_context=max_context,
                                  delta_t_max=pretrain_delta_t_max,
                                  delta_t_min=1, seed=seed + 10000)
        print(f"  [pretrain] train={len(train_pre)}, val={len(val_pre)}",
              flush=True)
        tlo = DataLoader(train_pre, batch_size=pre_batch, shuffle=True,
                         collate_fn=collate_pretrain, num_workers=0)
        vlo = DataLoader(val_pre, batch_size=pre_batch, shuffle=False,
                         collate_fn=collate_pretrain, num_workers=0)
        t0 = time.time()
        pre_out = pretrain_default(model, tlo, vlo, lr=3e-4,
                                   n_epochs=pre_epochs, patience=pre_patience,
                                   device=device)
        pre_time = time.time() - t0
        pre_best = float(pre_out['best_loss'])
        pre_hist_len = len(pre_out['history'])
        print(f"  [pretrain] done in {pre_time:.1f}s best_val={pre_best:.4f}",
              flush=True)
        torch.save(model.state_dict(), pre_ckpt)

    # 2. Finetune
    # CHB-MIT @ 32Hz has ~8M training samples at stride=4 → 13min/epoch.
    # Use stride=32 (gives ~250K samples → ~3min/epoch) for tractable
    # 30-epoch training. Test stride=8 keeps the dense 32Hz evaluation
    # while bringing one seed under ~5 min instead of 44 min.
    if dataset == 'CHBMIT':
        train_stride, val_stride, test_stride = 32, 16, 8
    else:
        train_stride, val_stride, test_stride = 4, 4, 1
    train_ft = _build_event_concat(bundle['ft_train'], stride=train_stride,
                                   max_context=max_context,
                                   max_future=eval_max_future)
    val_ft = _build_event_concat(bundle['ft_val'], stride=val_stride,
                                 max_context=max_context,
                                 max_future=eval_max_future)
    test_ft = _build_event_concat(bundle['ft_test'], stride=test_stride,
                                  max_context=max_context,
                                  max_future=eval_max_future)
    print(f"  [ft] train={len(train_ft)}, val={len(val_ft)}, "
          f"test={len(test_ft)}", flush=True)
    if len(train_ft) == 0 or len(test_ft) == 0:
        print(f"  SKIP {tag}: empty FT datasets", flush=True)
        return None

    tloader = DataLoader(train_ft, batch_size=ft_batch, shuffle=True,
                         collate_fn=collate_event, num_workers=0)
    vloader = DataLoader(val_ft, batch_size=ft_batch, shuffle=False,
                         collate_fn=collate_event, num_workers=0)
    test_loader = DataLoader(test_ft, batch_size=ft_batch, shuffle=False,
                             collate_fn=collate_event, num_workers=0)

    t0 = time.time()
    ft_out = finetune_default(model, tloader, vloader, horizons, mode='pred_ft',
                              lr=1e-3, n_epochs=ft_epochs,
                              patience=ft_patience, device=device)
    ft_time = time.time() - t0
    ft_ckpt = CKPT_DIR / f'{tag}_pred_ft.pt'
    torch.save(model.state_dict(), ft_ckpt)
    print(f"  [ft] done in {ft_time:.1f}s best_val={ft_out['best_val']:.4f}",
          flush=True)

    # 3. Eval
    t0 = time.time()
    eval_out = evaluate(model, test_loader, horizons, mode='pred_ft',
                        device=device)
    eval_time = time.time() - t0
    p_surf, y_surf = eval_out['p_surface'], eval_out['y_surface']
    rows = per_horizon_diag(p_surf, y_surf, horizons)
    pooled_auprc = float(eval_out['primary']['auprc'])
    pooled_auroc = float(eval_out['primary']['auroc'])
    mono = float(eval_out['monotonicity']['violation_rate'])
    print_diag(tag, rows, pooled_auprc, pooled_auroc)

    h = honest_metrics(p_surf, y_surf, horizons)
    print(f"  [{tag}] mean h-AUROC={h['mean_h_auroc']:.4f} "
          f"(base {h['mean_h_auroc_base']:.4f}, "
          f"Δ={h['mean_h_auroc']-h['mean_h_auroc_base']:+.4f})  "
          f"pooled AUPRC={h['pooled_auprc']:.4f} "
          f"(base {h['pooled_auprc_base']:.4f}, "
          f"Δ={h['pooled_auprc']-h['pooled_auprc_base']:+.4f})", flush=True)

    surf_path = SURF_DIR / f'{tag}.npz'
    save_surface(surf_path, p_surf, y_surf, horizons, eval_out['t_index'],
                 metadata={'dataset': dataset, 'norm_mode': norm_mode,
                           'seed': seed, 'predictor_kind': predictor_kind,
                           'phase': 'v29'})

    return {
        'tag': tag, 'dataset': dataset, 'norm_mode': norm_mode, 'seed': seed,
        'predictor_kind': predictor_kind,
        'n_params': n_total, 'predictor_params': n_pred,
        'pretrain_best_loss': pre_best, 'pretrain_epochs_run': pre_hist_len,
        'pretrain_time_s': pre_time, 'ft_best_val': float(ft_out['best_val']),
        'ft_time_s': ft_time, 'eval_time_s': eval_time,
        'pooled_auprc': pooled_auprc, 'pooled_auroc': pooled_auroc,
        **h,
        'monotonicity_violation_rate': mono,
        'per_horizon': rows,
        'surface_path': str(surf_path),
        'pretrain_ckpt': str(pre_ckpt), 'ft_ckpt': str(ft_ckpt),
    }


def run_and_persist(dataset: str, seeds: List[int], out_json: Path,
                    **kw) -> List[Dict]:
    results = []
    for seed in seeds:
        res = run_one(dataset, seed, **kw)
        if res is not None:
            results.append(res)
        with open(out_json, 'w') as f:
            json.dump({'dataset': dataset,
                       'options': {k: kw.get(k) for k in
                                   ('predictor_kind', 'norm_mode',
                                    'tag_suffix')},
                       'seeds': seeds[:len(results)],
                       'results': results}, f, indent=2, default=str)
        print(f"  wrote {out_json}", flush=True)
    return results
