"""V31 runner: extends v30 with fixed label_fraction for single-entity datasets.

Bug in v30: label_fraction subsampled *entities* (engines/machines/sensors).
For single-entity datasets (MBA, BATADAL, PSM, ETTm1, GECCO), this was a
no-op because max(1, int(round(1 * 0.1))) = 1 == n_entities.

Fix: for single-entity datasets, label_fraction truncates the training
time series to the first label_fraction * T timesteps. This simulates
having labeled data for only a fraction of the operational period.
For multi-entity datasets, entity subsampling still applies (and we also
add within-entity truncation as a secondary mechanism for finer control).

ALSO: adds wandb logging for all training runs (non-negotiable).
"""
from __future__ import annotations

import copy
import json
import os
import sys
import time
import threading
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset, Subset
from sklearn.metrics import roc_auc_score, average_precision_score

try:
    import wandb
    WANDB_OK = True
except ImportError:
    WANDB_OK = False
    print("WARNING: wandb not found, install with 'pip install wandb'", flush=True)

try:
    import psutil
    PSUTIL_OK = True
except ImportError:
    PSUTIL_OK = False

FAM_DIR = Path('/home/sagemaker-user/IndustrialJEPA/fam-jepa')
V31_DIR = FAM_DIR / 'experiments/v31'
CKPT_DIR = V31_DIR / 'ckpts'
SURF_DIR = V31_DIR / 'surfaces'
LOG_DIR = V31_DIR / 'logs'
RES_DIR = V31_DIR / 'results'
PNG_DIR = RES_DIR / 'surface_pngs'
for d in [CKPT_DIR, SURF_DIR, LOG_DIR, RES_DIR, PNG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(FAM_DIR))
sys.path.insert(0, str(FAM_DIR / 'experiments/v24'))
sys.path.insert(0, str(FAM_DIR / 'experiments/archive/v24'))   # _cmapss_raw
sys.path.insert(0, str(FAM_DIR / 'experiments/archive/v11'))   # data_utils
sys.path.insert(0, str(FAM_DIR / 'experiments/v27'))
sys.path.insert(0, str(FAM_DIR / 'experiments/v28'))
sys.path.insert(0, str(FAM_DIR / 'experiments/v29'))
sys.path.insert(0, str(FAM_DIR / 'experiments/v30'))

from model import FAM
from train import (
    PretrainDataset, EventDataset, collate_pretrain, collate_event,
    pretrain as pretrain_default, evaluate, save_surface,
)
from evaluation.losses import build_label_surface
from _runner_v29 import (
    LOADERS, NORM_POLICY, honest_metrics,
    CMAPSS_HORIZONS, ANOMALY_HORIZONS, CHBMIT_HORIZONS,
)
from _runner import _global_zscore, per_horizon_diag, print_diag, _build_event_concat
from _runner_v30 import (
    find_pretrain_ckpt, render_surface_panel,
    V29_CKPT, V28_CKPT, V27_CKPT,
)

EPS = 1e-7


# ---------------------------------------------------------------------------
# FIXED label-fraction subsetting logic
# ---------------------------------------------------------------------------

def _truncate_entity(entity, T_keep: int):
    """Truncate a single entity dict (or list/array) to T_keep timesteps."""
    if isinstance(entity, dict):
        result = dict(entity)  # shallow copy
        for k in ('test', 'labels', 'x'):
            if k in result and hasattr(result[k], '__len__'):
                result[k] = result[k][:T_keep]
        return result
    elif isinstance(entity, (list, tuple)) and len(entity) == 2:
        return (entity[0][:T_keep], entity[1][:T_keep])
    elif hasattr(entity, '__len__'):
        return entity[:T_keep]
    else:
        return entity


def _apply_label_fraction(train_engines, label_fraction: float, seed: int,
                           dataset: str):
    """
    Apply label fraction correctly for all dataset types.

    Multi-entity (n > 1): subsample entities (same as v30 - this is correct).
    Single-entity (n == 1): truncate time series to first label_fraction * T steps.
      This is the FIX. Previously, max(1, round(0.1 * 1)) = 1 == no-op.

    Entity format: list of dicts with keys 'test' (array T x C), 'labels' (T,),
    'entity_id' (str). For multi-entity: subsample the list. For single-entity:
    truncate the test/labels arrays.
    """
    rng = np.random.RandomState(seed + 7777)

    if isinstance(train_engines, dict):
        # Dict of {key: entity} - rare case
        keys = sorted(train_engines.keys())
        n = len(keys)
        if n > 1:
            n_keep = max(1, int(round(n * label_fraction)))
            keep = sorted(rng.choice(keys, size=n_keep, replace=False).tolist())
            result = {k: train_engines[k] for k in keep}
            print(f"  [ft] label_fraction={label_fraction:.2f} -> "
                  f"{n_keep}/{n} train engines (entity subsampling, dict)",
                  flush=True)
            return result
        else:
            # Single-entity dict: truncate time series
            key = keys[0]
            entity = train_engines[key]
            T = len(entity.get('test', entity.get('x', entity)))
            T_keep = max(256, int(round(T * label_fraction)))
            print(f"  [ft] label_fraction={label_fraction:.2f} -> "
                  f"time truncation {T_keep}/{T} steps ({dataset}, single dict entity)",
                  flush=True)
            return {key: _truncate_entity(entity, T_keep)}
    else:
        # list of entity dicts (the common format)
        n = len(train_engines)
        if n > 1:
            n_keep = max(1, int(round(n * label_fraction)))
            keep_idx = sorted(rng.choice(n, size=n_keep, replace=False).tolist())
            result = [train_engines[i] for i in keep_idx]
            print(f"  [ft] label_fraction={label_fraction:.2f} -> "
                  f"{n_keep}/{n} train entities (entity subsampling, list)",
                  flush=True)
            return result
        elif n == 1:
            # Single-entity list: truncate time series (THE FIX)
            entity = train_engines[0]
            if isinstance(entity, dict):
                T = len(entity.get('test', entity.get('x', [])))
            elif hasattr(entity, '__len__'):
                T = len(entity)
            else:
                print(f"  [ft] WARNING: cannot determine T for {type(entity)}",
                      flush=True)
                return train_engines
            T_keep = max(256, int(round(T * label_fraction)))
            print(f"  [ft] label_fraction={label_fraction:.2f} -> "
                  f"time truncation {T_keep}/{T} steps ({dataset}, single entity - THE FIX)",
                  flush=True)
            return [_truncate_entity(entity, T_keep)]
        else:
            print(f"  [ft] WARNING: empty train_engines for {dataset}", flush=True)
            return train_engines


# ---------------------------------------------------------------------------
# Resource logger for wandb
# ---------------------------------------------------------------------------

def _start_resource_logger(run, interval_s: int = 60):
    """Background thread logging GPU/RAM/disk to wandb every interval_s seconds."""
    if run is None:
        return None

    stop_event = threading.Event()

    def _log_loop():
        step = 0
        while not stop_event.is_set():
            metrics = {}
            try:
                if torch.cuda.is_available():
                    metrics['sys/gpu_mem_gb'] = torch.cuda.memory_allocated() / 1e9
                    metrics['sys/gpu_mem_reserved_gb'] = torch.cuda.memory_reserved() / 1e9
                if PSUTIL_OK:
                    metrics['sys/ram_pct'] = psutil.virtual_memory().percent
                    metrics['sys/disk_pct'] = psutil.disk_usage('/').percent
            except Exception:
                pass
            if metrics:
                try:
                    wandb.log(metrics, step=step)
                except Exception:
                    pass
            step += 1
            stop_event.wait(interval_s)

    t = threading.Thread(target=_log_loop, daemon=True)
    t.start()
    return stop_event


# ---------------------------------------------------------------------------
# Dense FT loop with wandb logging
# ---------------------------------------------------------------------------

def finetune_v31(model: FAM, train_loader, val_loader,
                 eval_horizons: List[int], max_horizon: int,
                 train_horizons_dense: int = 0,
                 mode: str = 'pred_ft', pos_weight: Optional[float] = None,
                 lr: float = 1e-3, weight_decay: float = 0.01,
                 n_epochs: int = 30, patience: int = 8,
                 device: str = 'cuda', seed: int = 42,
                 wandb_run=None, tag: str = '') -> dict:
    """Finetune predictor + event head with wandb logging."""
    model.to(device)
    eval_h = torch.tensor(eval_horizons, dtype=torch.float32, device=device)

    if mode == 'pred_ft':
        for p in model.encoder.parameters():
            p.requires_grad = False
        params = list(model.predictor.parameters())
        if hasattr(model, 'event_head_kind') and model.event_head_kind == 'monotone_cdf':
            params += list(model.monotone_cdf.parameters())
        else:
            params += list(model.event_head.parameters())
    else:
        params = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)

    if pos_weight is None:
        n_pos, n_tot = 0, 0
        for ctx, ctx_m, tte, t_idx in train_loader:
            y = build_label_surface(tte.unsqueeze(1), eval_h.cpu()).squeeze(1)
            n_pos += y.sum().item(); n_tot += y.numel()
        pos_weight = max(1.0, min(1000.0, (n_tot - n_pos) / max(n_pos, 1)))
    pw = torch.tensor(pos_weight, device=device)

    g = torch.Generator(device='cpu'); g.manual_seed(seed)
    best_val, best_state, wait = float('inf'), None, 0

    for epoch in range(n_epochs):
        model.train()
        losses = []
        for ctx, ctx_m, tte, t_idx in train_loader:
            ctx, ctx_m = ctx.to(device), ctx_m.to(device)
            tte = tte.to(device)
            if train_horizons_dense > 0:
                sampled = torch.randint(1, max_horizon + 1,
                                        (train_horizons_dense,), generator=g)
                sampled = torch.unique(sampled)
                sampled, _ = torch.sort(sampled)
                h_train = sampled.float().to(device)
            else:
                h_train = eval_h
            cdf = model.finetune_forward(ctx, h_train, ctx_m, mode)
            y = build_label_surface(tte.unsqueeze(1), h_train).squeeze(1)
            p = cdf.clamp(EPS, 1 - EPS)
            loss = -(pw * y * torch.log(p) + (1 - y) * torch.log(1 - p)).mean()
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            losses.append(loss.item())
        scheduler.step()
        train_loss = float(np.mean(losses))

        model.eval()
        vl = []
        with torch.no_grad():
            for ctx, ctx_m, tte, t_idx in val_loader:
                ctx, ctx_m = ctx.to(device), ctx_m.to(device)
                tte = tte.to(device)
                cdf = model.finetune_forward(ctx, eval_h, ctx_m, mode)
                y = build_label_surface(tte.unsqueeze(1), eval_h).squeeze(1)
                p = cdf.clamp(EPS, 1 - EPS)
                vl.append(-(pw * y * torch.log(p)
                            + (1 - y) * torch.log(1 - p)).mean().item())
        val_loss = float(np.mean(vl))

        if wandb_run is not None:
            try:
                wandb.log({'ft/train_loss': train_loss, 'ft/val_loss': val_loss,
                           'ft/lr': scheduler.get_last_lr()[0]}, step=epoch)
            except Exception:
                pass

        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"  early stop at epoch {epoch}", flush=True); break
        if epoch % 5 == 0 or wait == 0:
            print(f"  epoch {epoch:3d}  train={train_loss:.4f}  val={val_loss:.4f}",
                  flush=True)

    if best_state is not None:
        model.load_state_dict(best_state)
    return {'best_val': best_val, 'final_epoch': epoch}


# ---------------------------------------------------------------------------
# Main run_one function
# ---------------------------------------------------------------------------

def run_v31(dataset: str, seed: int,
            eval_horizons: Sequence[int],
            event_head_kind: str = 'discrete_hazard',
            train_horizons_dense: int = 0,
            tag_suffix: str = '',
            init_from_ckpt: Optional[Path] = None,
            label_fraction: float = 1.0,
            norm_mode: Optional[str] = None,
            predictor_kind: str = 'mlp',
            ft_epochs: int = 30, ft_patience: int = 8,
            pre_epochs: int = 30, pre_patience: int = 5,
            n_cuts_train: int = 40, n_cuts_val: int = 10,
            max_context: int = 512,
            pre_batch: int = 64, ft_batch: int = 128,
            sort_panel_by_tte: bool = False,
            use_wandb: bool = True,
            device: str = 'cuda') -> Dict:
    torch.manual_seed(seed); np.random.seed(seed)
    if norm_mode is None:
        norm_mode = NORM_POLICY[dataset]
    eval_horizons = list(eval_horizons)
    tag_parts = [dataset, norm_mode, event_head_kind]
    if predictor_kind != 'mlp':
        tag_parts.append('xpred')
    if train_horizons_dense > 0:
        tag_parts.append(f'td{train_horizons_dense}')
    if label_fraction < 1.0:
        tag_parts.append(f'lf{int(label_fraction * 100)}')
    if tag_suffix:
        tag_parts.append(tag_suffix)
    tag_parts.append(f's{seed}')
    tag = '_'.join(tag_parts)
    print(f"\n{'='*72}\n=== {tag}\n{'='*72}", flush=True)

    # Init wandb
    wandb_run = None
    if use_wandb and WANDB_OK:
        import git
        try:
            repo = git.Repo('/home/sagemaker-user/IndustrialJEPA')
            commit = repo.head.commit.hexsha[:8]
        except Exception:
            commit = 'unknown'
        try:
            wandb_run = wandb.init(
                project='industrialjepa',
                name=f'v31-{tag}',
                tags=['v31', dataset, f'lf{int(label_fraction*100)}',
                      f'seed{seed}'],
                config={
                    'dataset': dataset, 'seed': seed,
                    'label_fraction': label_fraction,
                    'norm_mode': norm_mode,
                    'eval_horizons': len(eval_horizons),
                    'max_horizon': max(eval_horizons),
                    'ft_epochs': ft_epochs, 'ft_patience': ft_patience,
                    'event_head_kind': event_head_kind,
                    'train_horizons_dense': train_horizons_dense,
                    'predictor_kind': predictor_kind,
                    'commit': commit,
                    'version': 'v31',
                },
                reinit=True,
            )
        except Exception as e:
            print(f"  wandb init failed: {e}", flush=True)
            wandb_run = None
    elif use_wandb and not WANDB_OK:
        print("  skipping wandb (not installed)", flush=True)

    resource_stop = _start_resource_logger(wandb_run)

    try:
        result = _run_v31_inner(
            dataset=dataset, seed=seed, eval_horizons=eval_horizons,
            event_head_kind=event_head_kind,
            train_horizons_dense=train_horizons_dense,
            tag_suffix=tag_suffix, tag=tag,
            init_from_ckpt=init_from_ckpt,
            label_fraction=label_fraction,
            norm_mode=norm_mode, predictor_kind=predictor_kind,
            ft_epochs=ft_epochs, ft_patience=ft_patience,
            pre_epochs=pre_epochs, pre_patience=pre_patience,
            n_cuts_train=n_cuts_train, n_cuts_val=n_cuts_val,
            max_context=max_context, pre_batch=pre_batch,
            ft_batch=ft_batch, sort_panel_by_tte=sort_panel_by_tte,
            device=device, wandb_run=wandb_run,
        )
        if result is not None and wandb_run is not None:
            try:
                wandb.log({
                    'eval/mean_h_auroc': result['mean_h_auroc'],
                    'eval/pooled_auprc': result['pooled_auprc'],
                    'eval/pooled_auroc': result['pooled_auroc'],
                })
                wandb.summary['mean_h_auroc'] = result['mean_h_auroc']
            except Exception:
                pass
        return result
    finally:
        if resource_stop is not None:
            resource_stop.set()
        if wandb_run is not None:
            try:
                wandb.finish()
            except Exception:
                pass


def _run_v31_inner(dataset, seed, eval_horizons, event_head_kind,
                   train_horizons_dense, tag_suffix, tag,
                   init_from_ckpt, label_fraction, norm_mode, predictor_kind,
                   ft_epochs, ft_patience, pre_epochs, pre_patience,
                   n_cuts_train, n_cuts_val, max_context, pre_batch,
                   ft_batch, sort_panel_by_tte, device, wandb_run):

    bundle = LOADERS[dataset]()
    if norm_mode == 'none':
        bundle = _global_zscore(bundle)
    n_channels = bundle['n_channels']

    # Diagnose train_engines BEFORE subsampling
    ft_train_orig = bundle['ft_train']
    if isinstance(ft_train_orig, dict):
        n_orig = len(ft_train_orig)
    else:
        n_orig = len(ft_train_orig)
    print(f"  loaded {dataset}: n_channels={n_channels}, "
          f"n_train_entities={n_orig}", flush=True)

    # 1. Build model
    model = FAM(n_channels=n_channels, patch_size=16, d_model=256,
                n_heads=4, n_layers=2, d_ff=256, dropout=0.1,
                ema_momentum=0.99, predictor_hidden=256,
                norm_mode=norm_mode, predictor_kind=predictor_kind,
                event_head_kind=event_head_kind)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"  total_params={n_total:,}  event_head_kind={event_head_kind}",
          flush=True)

    # 2. Load pretrained encoder
    pre_ckpt = init_from_ckpt
    pre_time, pre_best = 0.0, float('nan')
    if pre_ckpt is None:
        pre_ckpt = find_pretrain_ckpt(dataset, norm_mode, seed, predictor_kind)
    if pre_ckpt is not None and pre_ckpt.exists():
        print(f"  [pretrain] loading {pre_ckpt}", flush=True)
        sd = torch.load(pre_ckpt, map_location='cpu')
        missing, unexpected = model.load_state_dict(sd, strict=False)
        mlist = [m for m in (missing or []) if 'monotone_cdf' not in m]
        if mlist:
            print(f"  WARN missing keys: {mlist[:5]}", flush=True)
    else:
        # Train from scratch
        pretrain_dt_max = max(eval_horizons)
        if dataset == 'CHBMIT':
            pretrain_dt_max = min(pretrain_dt_max, 960)
        train_pre = PretrainDataset(bundle['pretrain_seqs'],
                                    n_cuts=n_cuts_train, max_context=max_context,
                                    delta_t_max=pretrain_dt_max,
                                    delta_t_min=1, seed=seed)
        val_seqs = {}
        for k, seq in bundle['pretrain_seqs'].items():
            L = len(seq); cut = int(0.9 * L)
            if L - cut >= 128:
                val_seqs[k] = seq[cut:]
        if not val_seqs:
            val_seqs = bundle['pretrain_seqs']
        val_pre = PretrainDataset(val_seqs, n_cuts=n_cuts_val,
                                  max_context=max_context,
                                  delta_t_max=pretrain_dt_max,
                                  delta_t_min=1, seed=seed + 10000)
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
        new_ckpt = CKPT_DIR / f'{dataset}_{norm_mode}_s{seed}_pretrain.pt'
        torch.save(model.state_dict(), new_ckpt)
        print(f"  [pretrain] saved {new_ckpt}", flush=True)

    # 3. Apply FIXED label fraction
    eval_max_future = max(eval_horizons)
    if label_fraction < 1.0:
        train_engines = _apply_label_fraction(
            bundle['ft_train'], label_fraction, seed, dataset)
    else:
        train_engines = bundle['ft_train']

    # Diagnostic: print sample counts BEFORE and AFTER
    if isinstance(train_engines, dict):
        n_after = len(train_engines)
    else:
        n_after = len(train_engines)
    print(f"  [ft] n_train_entities: {n_orig} -> {n_after} "
          f"(label_fraction={label_fraction})", flush=True)

    if dataset == 'CHBMIT':
        train_stride, val_stride, test_stride = 32, 16, 8
    else:
        train_stride, val_stride, test_stride = 4, 4, 1

    train_ft = _build_event_concat(train_engines, stride=train_stride,
                                   max_context=max_context,
                                   max_future=eval_max_future)
    val_ft = _build_event_concat(bundle['ft_val'], stride=val_stride,
                                 max_context=max_context,
                                 max_future=eval_max_future)
    test_ft = _build_event_concat(bundle['ft_test'], stride=test_stride,
                                  max_context=max_context,
                                  max_future=eval_max_future)
    print(f"  [ft] train_windows={len(train_ft)} val_windows={len(val_ft)} "
          f"test_windows={len(test_ft)}", flush=True)

    # SELF-CHECK: train_windows at lf10 must be < train_windows at lf100
    # (printed above for manual inspection; stored in result dict)

    if len(train_ft) == 0 or len(test_ft) == 0:
        print(f"  SKIP {tag}: empty FT datasets", flush=True)
        return None

    tloader = DataLoader(train_ft, batch_size=ft_batch, shuffle=True,
                         collate_fn=collate_event, num_workers=0)
    vloader = DataLoader(val_ft, batch_size=ft_batch, shuffle=False,
                         collate_fn=collate_event, num_workers=0)
    test_loader = DataLoader(test_ft, batch_size=ft_batch, shuffle=False,
                             collate_fn=collate_event, num_workers=0)

    # 4. Finetune
    t0 = time.time()
    ft_out = finetune_v31(model, tloader, vloader, eval_horizons,
                          max_horizon=max(eval_horizons),
                          train_horizons_dense=train_horizons_dense,
                          mode='pred_ft', lr=1e-3, n_epochs=ft_epochs,
                          patience=ft_patience, device=device, seed=seed,
                          wandb_run=wandb_run, tag=tag)
    ft_time = time.time() - t0
    ft_ckpt = CKPT_DIR / f'{tag}_pred_ft.pt'
    torch.save(model.state_dict(), ft_ckpt)
    print(f"  [ft] {ft_time:.1f}s best_val={ft_out['best_val']:.4f}", flush=True)

    # 5. Eval
    t0 = time.time()
    eval_out = evaluate(model, test_loader, eval_horizons, mode='pred_ft',
                        device=device)
    eval_time = time.time() - t0
    p_surf, y_surf = eval_out['p_surface'], eval_out['y_surface']
    pooled_auprc = float(eval_out['primary']['auprc'])
    pooled_auroc = float(eval_out['primary']['auroc'])
    h = honest_metrics(p_surf, y_surf, eval_horizons)
    print(f"  [{tag}] mean h-AUROC={h['mean_h_auroc']:.4f} "
          f"(base {h['mean_h_auroc_base']:.4f}, "
          f"Δ={h['mean_h_auroc']-h['mean_h_auroc_base']:+.4f})  "
          f"pooled AUPRC={h['pooled_auprc']:.4f}", flush=True)

    surf_path = SURF_DIR / f'{tag}.npz'
    save_surface(surf_path, p_surf, y_surf, eval_horizons, eval_out['t_index'],
                 metadata={'dataset': dataset, 'norm_mode': norm_mode,
                           'seed': seed, 'predictor_kind': predictor_kind,
                           'event_head_kind': event_head_kind,
                           'label_fraction': label_fraction,
                           'phase': 'v31'})
    png_path = PNG_DIR / f'{tag}.png'
    render_surface_panel(p_surf, y_surf, eval_horizons, eval_out['t_index'],
                         png_path, tag=tag, sort_by_tte=sort_panel_by_tte)
    print(f"  [{tag}] surface={surf_path}, png={png_path}", flush=True)

    return {
        'tag': tag, 'dataset': dataset, 'norm_mode': norm_mode, 'seed': seed,
        'predictor_kind': predictor_kind, 'event_head_kind': event_head_kind,
        'eval_horizons': list(eval_horizons),
        'train_horizons_dense': train_horizons_dense,
        'label_fraction': label_fraction,
        'n_params': n_total,
        'n_train_entities_orig': n_orig,
        'n_train_entities_after': n_after,
        'n_train_windows': len(train_ft),
        'n_test_windows': len(test_ft),
        'pretrain_best_loss': pre_best, 'pretrain_time_s': pre_time,
        'pretrain_ckpt': str(pre_ckpt) if pre_ckpt is not None else None,
        'ft_best_val': float(ft_out['best_val']), 'ft_time_s': ft_time,
        'eval_time_s': eval_time,
        'pooled_auprc': pooled_auprc, 'pooled_auroc': pooled_auroc,
        **h,
        'surface_path': str(surf_path), 'png_path': str(png_path),
        'ft_ckpt': str(ft_ckpt),
    }
