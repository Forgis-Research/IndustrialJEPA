"""V27 shared runner: full pretrain → pred-FT → eval pipeline.

One function: ``run_one(dataset, norm_mode, seed) -> dict``.

Handles:
  - dataset-specific loading (FD001/2/3, SMAP, MSL, PSM, SMD, MBA, PhysioNet)
  - per-norm-mode data preprocessing (global z-score only when
    ``norm_mode == 'none'`` — other modes expect raw data)
  - pretrain, predictor-finetune, evaluate
  - surface + per-horizon AUROC/AUPRC + prediction-gap diagnostic

Every run stores:
  - ``ckpts/<dataset>_<mode>_s<seed>_pretrain.pt``
  - ``ckpts/<dataset>_<mode>_s<seed>_pred_ft.pt``
  - ``surfaces/<dataset>_<mode>_s<seed>.npz``
  - returns full metrics dict, including per-horizon rows for the PRIMARY
    diagnostic table (AUROC + AUPRC + pos_rate + prediction gap).
"""

import copy
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset
from sklearn.metrics import roc_auc_score, average_precision_score

FAM_DIR = Path('/home/sagemaker-user/IndustrialJEPA/fam-jepa')
V27_DIR = FAM_DIR / 'experiments/v27'
CKPT_DIR = V27_DIR / 'ckpts'
SURF_DIR = V27_DIR / 'surfaces'
LOG_DIR = V27_DIR / 'logs'
RES_DIR = V27_DIR / 'results'
for d in [CKPT_DIR, SURF_DIR, LOG_DIR, RES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(FAM_DIR))
sys.path.insert(0, str(FAM_DIR / 'experiments/v24'))  # _cmapss_raw
sys.path.insert(0, str(FAM_DIR / 'experiments/v11'))  # data_utils

from model import FAM
from train import (
    PretrainDataset, EventDataset, collate_pretrain, collate_event,
    pretrain, finetune, evaluate, save_surface,
)

CMAPSS_HORIZONS = [1, 5, 10, 20, 50, 100, 150]
ANOMALY_HORIZONS = [1, 5, 10, 20, 50, 100, 150, 200]


# ---------------------------------------------------------------------------
# Dataset loading  →  unified format
#
# Every loader returns a dict with:
#   pretrain_seqs: Dict[int, np.ndarray]  (T_i, C)   — raw pretrain sequences
#   ft_train, ft_val, ft_test: List[Dict]             — EventDataset inputs
#     each dict has keys 'test' (array (T, C)) and 'labels' (array (T,)).
#     C-MAPSS treats each engine's last cycle as the positive label.
#   n_channels, horizons
# ---------------------------------------------------------------------------

def _cmapss(subset: str) -> Dict:
    from _cmapss_raw import load_cmapss_raw
    d = load_cmapss_raw(subset)
    # Build entity records with test=seq, labels[T-1]=1
    def engines_to_entities(engines):
        out = []
        for eid, seq in engines.items():
            labels = np.zeros(len(seq), dtype=np.int32)
            labels[-1] = 1
            out.append({'entity_id': int(eid), 'test': seq, 'labels': labels})
        return out
    return {
        'pretrain_seqs': d['train_engines'],
        'ft_train': engines_to_entities(d['train_engines']),
        'ft_val': engines_to_entities(d['val_engines']),
        'ft_test': engines_to_entities(d['test_engines']),
        'n_channels': 14,
        'horizons': CMAPSS_HORIZONS,
        'subset': subset,
    }


def _smap() -> Dict:
    from data.smap_msl import load_smap_entities, split_smap_entities
    ents = load_smap_entities(normalize=False)
    pretrain_seqs = {i: e['train'].astype(np.float32)
                     for i, e in enumerate(ents) if len(e['train']) >= 128}
    ft = split_smap_entities(normalize=False)
    return {
        'pretrain_seqs': pretrain_seqs,
        'ft_train': ft['ft_train'], 'ft_val': ft['ft_val'], 'ft_test': ft['ft_test'],
        'n_channels': 25, 'horizons': ANOMALY_HORIZONS, 'subset': 'SMAP',
    }


def _msl() -> Dict:
    from data.smap_msl import load_msl_entities, split_msl_entities
    ents = load_msl_entities(normalize=False)
    pretrain_seqs = {i: e['train'].astype(np.float32)
                     for i, e in enumerate(ents) if len(e['train']) >= 128}
    ft = split_msl_entities(normalize=False)
    return {
        'pretrain_seqs': pretrain_seqs,
        'ft_train': ft['ft_train'], 'ft_val': ft['ft_val'], 'ft_test': ft['ft_test'],
        'n_channels': 55, 'horizons': ANOMALY_HORIZONS, 'subset': 'MSL',
    }


def _single_stream_intra_split(train: np.ndarray, test: np.ndarray,
                               labels: np.ndarray, gap: int = 200) -> Dict:
    """Chronological intra-stream split for PSM/SMD/MBA/etc."""
    T = len(test)
    t1 = int(0.6 * T)
    t2 = int(0.7 * T)
    return {
        'ft_train': [{'entity_id': 'E', 'test': test[:t1], 'labels': labels[:t1]}],
        'ft_val':   [{'entity_id': 'E', 'test': test[t1 + gap:t2], 'labels': labels[t1 + gap:t2]}],
        'ft_test':  [{'entity_id': 'E', 'test': test[t2 + gap:], 'labels': labels[t2 + gap:]}],
    }


def _psm() -> Dict:
    from data.psm import load_psm
    d = load_psm(normalize=False)
    train = d['train'].astype(np.float32)
    test = d['test'].astype(np.float32)
    labels = d['labels'].astype(np.int32)
    ft = _single_stream_intra_split(train, test, labels)
    return {
        'pretrain_seqs': {0: train}, **ft,
        'n_channels': train.shape[1], 'horizons': ANOMALY_HORIZONS, 'subset': 'PSM',
    }


def _smd() -> Dict:
    from data.smd import load_smd_entities, split_smd_entities
    ents = load_smd_entities(normalize=False)
    pretrain_seqs = {i: e['train'].astype(np.float32)
                     for i, e in enumerate(ents) if len(e['train']) >= 128}
    ft = split_smd_entities(normalize=False)
    return {
        'pretrain_seqs': pretrain_seqs,
        'ft_train': ft['ft_train'], 'ft_val': ft['ft_val'], 'ft_test': ft['ft_test'],
        'n_channels': ents[0]['train'].shape[1],
        'horizons': ANOMALY_HORIZONS, 'subset': 'SMD',
    }


def _mba() -> Dict:
    from data.mba import load_mba
    d = load_mba(normalize=False)
    if d is None:
        raise FileNotFoundError("MBA data unavailable")
    train = d['train'].astype(np.float32)
    test = d['test'].astype(np.float32)
    labels = d['labels'].astype(np.int32)
    ft = _single_stream_intra_split(train, test, labels)
    return {
        'pretrain_seqs': {0: train}, **ft,
        'n_channels': train.shape[1], 'horizons': ANOMALY_HORIZONS, 'subset': 'MBA',
    }


LOADERS = {
    'FD001': lambda: _cmapss('FD001'),
    'FD002': lambda: _cmapss('FD002'),
    'FD003': lambda: _cmapss('FD003'),
    'SMAP':  _smap,
    'MSL':   _msl,
    'PSM':   _psm,
    'SMD':   _smd,
    'MBA':   _mba,
}


# ---------------------------------------------------------------------------
# Global z-score (used only when norm_mode == 'none')
# ---------------------------------------------------------------------------

def _global_zscore(bundle: Dict) -> Dict:
    """Compute mean/std from concatenated pretrain train data, apply to all."""
    train_arrs = list(bundle['pretrain_seqs'].values())
    all_train = np.concatenate(train_arrs, axis=0)  # (N_total, C)
    mu = all_train.mean(axis=0, keepdims=True).astype(np.float32)   # (1, C)
    std = (all_train.std(axis=0, keepdims=True) + 1e-5).astype(np.float32)

    def norm_arr(a):
        return ((a - mu) / std).astype(np.float32)

    new = dict(bundle)
    new['pretrain_seqs'] = {k: norm_arr(v) for k, v in bundle['pretrain_seqs'].items()}
    for key in ('ft_train', 'ft_val', 'ft_test'):
        new[key] = [{**e, 'test': norm_arr(e['test'])} for e in bundle[key]]
    new['_global_mu'] = mu
    new['_global_std'] = std
    return new


# ---------------------------------------------------------------------------
# Diagnostic table  (per-horizon AUROC + AUPRC + pred-gap)
# ---------------------------------------------------------------------------

def per_horizon_diag(p_surface: np.ndarray, y_surface: np.ndarray,
                     horizons: List[int]) -> List[Dict]:
    rows = []
    for i, h in enumerate(horizons):
        y_i = y_surface[:, i].astype(int)
        p_i = p_surface[:, i]
        n_pos = int(y_i.sum())
        n_neg = int(len(y_i) - n_pos)
        if n_pos == 0 or n_neg == 0:
            auroc = auprc = float('nan')
            gap = float('nan')
        else:
            auroc = float(roc_auc_score(y_i, p_i))
            auprc = float(average_precision_score(y_i, p_i))
            gap = float(p_i[y_i == 1].mean() - p_i[y_i == 0].mean())
        rows.append({
            'dt': int(h), 'auroc': auroc, 'auprc': auprc,
            'pos_rate': float(y_i.mean()), 'pred_gap': gap,
            'n_pos': n_pos, 'n_neg': n_neg,
        })
    return rows


def print_diag(tag: str, rows: List[Dict], pooled_auprc: float, pooled_auroc: float):
    print(f"  [{tag}] pooled AUPRC={pooled_auprc:.4f} AUROC={pooled_auroc:.4f}",
          flush=True)
    print(f"  {'dt':>4}  {'AUROC':>7}  {'AUPRC':>7}  {'pos':>6}  {'gap':>8}",
          flush=True)
    for r in rows:
        print(f"  {r['dt']:>4}  {r['auroc']:>7.3f}  {r['auprc']:>7.3f}  "
              f"{r['pos_rate']:>6.3f}  {r['pred_gap']:>+8.4f}", flush=True)


# ---------------------------------------------------------------------------
# Event dataset builder
# ---------------------------------------------------------------------------

def _build_event_concat(entity_list, stride, max_context=512,
                        max_future=200, min_context=128):
    datasets = []
    for e in entity_list:
        x = e['test']
        y = e['labels']
        T = len(x)
        if T <= min_context + 1:
            continue
        ds = EventDataset(x, y, max_context=max_context, stride=stride,
                          max_future=max_future, min_context=min_context)
        if len(ds) > 0:
            datasets.append(ds)
    return ConcatDataset(datasets) if datasets else ConcatDataset([])


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_one(dataset: str, norm_mode: str, seed: int,
            pre_epochs: int = 30, pre_patience: int = 5,
            ft_epochs: int = 30, ft_patience: int = 8,
            n_cuts_train: int = 40, n_cuts_val: int = 10,
            max_context: int = 512,
            pre_batch: int = 64, ft_batch: int = 128,
            device: str = 'cuda') -> Dict:
    """Run pretrain + pred-FT + eval for one (dataset, norm_mode, seed)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    tag = f"{dataset}_{norm_mode}_s{seed}"
    print(f"\n{'='*70}\n=== {tag}\n{'='*70}", flush=True)

    # 1. Load raw data
    t0 = time.time()
    bundle = LOADERS[dataset]()
    if norm_mode == 'none':
        bundle = _global_zscore(bundle)
        mu = bundle['_global_mu']
        print(f"  global z-score: mu_mean={float(mu.mean()):.4f}, "
              f"mu_std={float(mu.std()):.4f}", flush=True)
    horizons = bundle['horizons']
    n_channels = bundle['n_channels']
    print(f"  loaded {dataset} in {time.time()-t0:.1f}s "
          f"(pretrain_seqs={len(bundle['pretrain_seqs'])}, "
          f"n_channels={n_channels}, horizons={horizons})", flush=True)

    # 2. Pretrain
    model = FAM(n_channels=n_channels, patch_size=16, d_model=256,
                n_heads=4, n_layers=2, d_ff=256, dropout=0.1,
                ema_momentum=0.99, predictor_hidden=256, norm_mode=norm_mode)
    pre_ckpt = CKPT_DIR / f'{tag}_pretrain.pt'
    pre_hist_len = 0
    pre_time = 0.0
    pre_best = float('nan')
    if pre_ckpt.exists():
        print(f"  [pretrain] ckpt exists, loading: {pre_ckpt}", flush=True)
        model.load_state_dict(torch.load(pre_ckpt, map_location='cpu'))
    else:
        train_pre = PretrainDataset(bundle['pretrain_seqs'],
                                    n_cuts=n_cuts_train,
                                    max_context=max_context, delta_t_max=150,
                                    delta_t_min=1, seed=seed)
        # Build val as 10% tail of each pretrain seq (fall back to full seqs)
        val_seqs = {}
        for k, seq in bundle['pretrain_seqs'].items():
            L = len(seq)
            cut = int(0.9 * L)
            if L - cut >= 128:
                val_seqs[k] = seq[cut:]
        if not val_seqs:
            val_seqs = bundle['pretrain_seqs']
        val_pre = PretrainDataset(val_seqs, n_cuts=n_cuts_val,
                                  max_context=max_context, delta_t_max=150,
                                  delta_t_min=1, seed=seed + 10000)
        print(f"  [pretrain] train={len(train_pre)}, val={len(val_pre)}", flush=True)
        tlo = DataLoader(train_pre, batch_size=pre_batch, shuffle=True,
                         collate_fn=collate_pretrain, num_workers=0)
        vlo = DataLoader(val_pre, batch_size=pre_batch, shuffle=False,
                         collate_fn=collate_pretrain, num_workers=0)
        t0 = time.time()
        pre_out = pretrain(model, tlo, vlo, lr=3e-4, n_epochs=pre_epochs,
                           patience=pre_patience, device=device)
        pre_time = time.time() - t0
        pre_best = float(pre_out['best_loss'])
        pre_hist_len = len(pre_out['history'])
        print(f"  [pretrain] done in {pre_time:.1f}s best_val={pre_best:.4f}", flush=True)
        torch.save(model.state_dict(), pre_ckpt)

    # 3. Finetune
    train_ft = _build_event_concat(bundle['ft_train'], stride=4,
                                   max_context=max_context)
    val_ft = _build_event_concat(bundle['ft_val'], stride=4,
                                 max_context=max_context)
    test_ft = _build_event_concat(bundle['ft_test'], stride=1,
                                  max_context=max_context)
    print(f"  [ft] train={len(train_ft)}, val={len(val_ft)}, test={len(test_ft)}", flush=True)
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
    ft_out = finetune(model, tloader, vloader, horizons, mode='pred_ft',
                      lr=1e-3, n_epochs=ft_epochs, patience=ft_patience,
                      device=device)
    ft_time = time.time() - t0
    print(f"  [ft] done in {ft_time:.1f}s best_val={ft_out['best_val']:.4f}", flush=True)
    ft_ckpt = CKPT_DIR / f'{tag}_pred_ft.pt'
    torch.save(model.state_dict(), ft_ckpt)

    # 4. Evaluate
    t0 = time.time()
    eval_out = evaluate(model, test_loader, horizons, mode='pred_ft',
                        device=device)
    eval_time = time.time() - t0
    p_surf = eval_out['p_surface']
    y_surf = eval_out['y_surface']
    rows = per_horizon_diag(p_surf, y_surf, horizons)
    pooled_auprc = float(eval_out['primary']['auprc'])
    pooled_auroc = float(eval_out['primary']['auroc'])
    mono = float(eval_out['monotonicity']['violation_rate'])
    print(f"  [eval] done in {eval_time:.1f}s, mono_violation={mono:.6f}", flush=True)
    print_diag(tag, rows, pooled_auprc, pooled_auroc)

    # 5. Save surface
    surf_path = SURF_DIR / f'{tag}.npz'
    save_surface(surf_path, p_surf, y_surf, horizons, eval_out['t_index'],
                 metadata={'dataset': dataset, 'norm_mode': norm_mode,
                           'seed': seed, 'phase': 'v27'})

    return {
        'tag': tag,
        'dataset': dataset,
        'norm_mode': norm_mode,
        'seed': seed,
        'n_params': sum(p.numel() for p in model.parameters()),
        'pretrain_best_loss': pre_best,
        'pretrain_epochs_run': pre_hist_len,
        'pretrain_time_s': pre_time,
        'ft_best_val': float(ft_out['best_val']),
        'ft_time_s': ft_time,
        'eval_time_s': eval_time,
        'pooled_auprc': pooled_auprc,
        'pooled_auroc': pooled_auroc,
        'f1_best': float(eval_out['primary']['f1_best']),
        'precision_best': float(eval_out['primary']['precision_best']),
        'recall_best': float(eval_out['primary']['recall_best']),
        'prevalence': float(eval_out['primary']['prevalence']),
        'monotonicity_violation_rate': mono,
        'per_horizon': rows,
        'surface_path': str(surf_path),
        'pretrain_ckpt': str(pre_ckpt),
        'ft_ckpt': str(ft_ckpt),
    }


def run_and_persist(dataset: str, norm_mode: str, seeds: List[int],
                    out_json: Path, **kw) -> List[Dict]:
    """Run all seeds, persist JSON after each one (idempotent/resumable)."""
    results = []
    for seed in seeds:
        res = run_one(dataset, norm_mode, seed, **kw)
        if res is not None:
            results.append(res)
        with open(out_json, 'w') as f:
            json.dump({
                'dataset': dataset, 'norm_mode': norm_mode,
                'seeds': seeds[:len(results)],
                'results': results,
            }, f, indent=2)
        print(f"  wrote {out_json}", flush=True)
    return results
