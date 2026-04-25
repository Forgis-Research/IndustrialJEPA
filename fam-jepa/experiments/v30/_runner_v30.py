"""V30 runner: extends v29 with dense horizons + MonotoneCDF support.

Importing v29 LOADERS / NORM_POLICY / honest_metrics. Adds:
  - `eval_horizons` override (e.g. dense range(1, 151))
  - `event_head_kind` ('discrete_hazard' | 'monotone_cdf')
  - `init_from_ckpt`: warm-start from a v27/v28/v29 pretrained encoder
  - `train_horizons_dense`: sample K dense horizons per batch during FT
  - `freeze_predictor`: optionally keep the pretrained-init predictor frozen

Used by phase0/phase1/phase2/phase3 scripts.
"""
from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
from sklearn.metrics import roc_auc_score, average_precision_score

FAM_DIR = Path('/home/sagemaker-user/IndustrialJEPA/fam-jepa')
V30_DIR = FAM_DIR / 'experiments/v30'
CKPT_DIR = V30_DIR / 'ckpts'
SURF_DIR = V30_DIR / 'surfaces'
LOG_DIR = V30_DIR / 'logs'
RES_DIR = V30_DIR / 'results'
PNG_DIR = RES_DIR / 'surface_pngs'
for d in [CKPT_DIR, SURF_DIR, LOG_DIR, RES_DIR, PNG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(FAM_DIR))
sys.path.insert(0, str(FAM_DIR / 'experiments/v24'))
sys.path.insert(0, str(FAM_DIR / 'experiments/v27'))
sys.path.insert(0, str(FAM_DIR / 'experiments/v28'))
sys.path.insert(0, str(FAM_DIR / 'experiments/v29'))

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

EPS = 1e-7


# ---------------------------------------------------------------------------
# Existing-checkpoint discovery
# ---------------------------------------------------------------------------

V29_CKPT = FAM_DIR / 'experiments/v29/ckpts'
V28_CKPT = FAM_DIR / 'experiments/v28/ckpts'
V27_CKPT = FAM_DIR / 'experiments/v27/ckpts'


def find_pretrain_ckpt(dataset: str, norm_mode: str, seed: int,
                       predictor_kind: str = 'mlp') -> Optional[Path]:
    """Return path of the canonical v29/v28/v27 pretrained MLP checkpoint."""
    if predictor_kind != 'mlp':
        return None  # transformer-predictor ckpts only in v29 with _xpred suffix
    candidates = [
        V29_CKPT / f'{dataset}_{norm_mode}_s{seed}_pretrain.pt',
        V28_CKPT / f'{dataset}_{norm_mode}_s{seed}_pretrain.pt',
        V27_CKPT / f'{dataset}_{norm_mode}_s{seed}_pretrain.pt',
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


# ---------------------------------------------------------------------------
# Dense FT loop — adapted from v28.runner_v28.finetune_dense, supports any
# event_head_kind via FAM.finetune_forward dispatch.
# ---------------------------------------------------------------------------

def finetune_v30(model: FAM, train_loader, val_loader,
                 eval_horizons: List[int], max_horizon: int,
                 train_horizons_dense: int = 0,
                 mode: str = 'pred_ft', pos_weight: Optional[float] = None,
                 lr: float = 1e-3, weight_decay: float = 0.01,
                 n_epochs: int = 30, patience: int = 8,
                 device: str = 'cuda', seed: int = 42) -> dict:
    """Finetune predictor + event head (discrete or monotone CDF).

    train_horizons_dense > 0: sample that many random horizons per batch from
      [1, max_horizon]. =0 means use the eval_horizons every batch.
    Validation always uses fixed eval_horizons.
    """
    model.to(device)
    eval_h = torch.tensor(eval_horizons, dtype=torch.float32, device=device)

    if mode == 'pred_ft':
        for p in model.encoder.parameters():
            p.requires_grad = False
        params = list(model.predictor.parameters())
        if model.event_head_kind == 'monotone_cdf':
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
# Surface PNG renderer
# ---------------------------------------------------------------------------

def render_surface_panel(p_surface: np.ndarray, y_surface: np.ndarray,
                         horizons, t_index: np.ndarray, png_path: Path,
                         tag: str, sort_by_tte: bool = False):
    """Three-panel: predicted p (viridis) | GT (viridis) | |p-y| (gray_r)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    if sort_by_tte:
        order = np.argsort(t_index)
        p_surface = p_surface[order]
        y_surface = y_surface[order]

    err = np.abs(p_surface - y_surface)
    mae = float(err.mean())

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    extent = [0, p_surface.shape[0], horizons[0], horizons[-1]]
    for ax, mat, title, cmap, vmin, vmax in [
        (axes[0], p_surface.T, f'p(t,Δt) — {tag}', 'viridis', 0, 1),
        (axes[1], y_surface.T, 'y(t,Δt) GT', 'viridis', 0, 1),
        (axes[2], err.T, f'|p-y|  (mean={mae:.3f})', 'gray_r', 0, 1),
    ]:
        im = ax.imshow(mat, aspect='auto', origin='lower',
                       cmap=cmap, vmin=vmin, vmax=vmax,
                       extent=extent, interpolation='nearest')
        ax.set_xlabel('test sample t')
        ax.set_ylabel('horizon Δt')
        ax.set_title(title)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(png_path, dpi=120, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main run_one — single dataset × seed × variant
# ---------------------------------------------------------------------------

def run_v30(dataset: str, seed: int,
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

    bundle = LOADERS[dataset]()
    if norm_mode == 'none':
        bundle = _global_zscore(bundle)
    n_channels = bundle['n_channels']
    print(f"  loaded {dataset}: pretrain_seqs={len(bundle['pretrain_seqs'])}, "
          f"n_channels={n_channels}, n_eval_horizons={len(eval_horizons)}",
          flush=True)

    # 1. Build model
    model = FAM(n_channels=n_channels, patch_size=16, d_model=256,
                n_heads=4, n_layers=2, d_ff=256, dropout=0.1,
                ema_momentum=0.99, predictor_hidden=256,
                norm_mode=norm_mode, predictor_kind=predictor_kind,
                event_head_kind=event_head_kind)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"  total_params={n_total:,}  event_head_kind={event_head_kind}",
          flush=True)

    # 2. Load pretrained encoder (warm-start). Allow strict=False so the
    # MonotoneCDF path silently ignores missing event_head keys.
    pre_ckpt = init_from_ckpt
    pre_time, pre_best = 0.0, float('nan')
    if pre_ckpt is None:
        pre_ckpt = find_pretrain_ckpt(dataset, norm_mode, seed, predictor_kind)
    if pre_ckpt is not None and pre_ckpt.exists():
        print(f"  [pretrain] loading {pre_ckpt}", flush=True)
        sd = torch.load(pre_ckpt, map_location='cpu')
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing:
            mlist = [m for m in missing if 'monotone_cdf' not in m]
            if mlist:
                print(f"  WARN missing keys: {mlist[:5]}{'…' if len(mlist) > 5 else ''}",
                      flush=True)
        if unexpected:
            print(f"  WARN unexpected: {unexpected[:5]}", flush=True)
    else:
        # Train pretraining from scratch.
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
        print(f"  [pretrain] train={len(train_pre)} val={len(val_pre)} "
              f"dt_max={pretrain_dt_max}", flush=True)
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

    # 3. Build FT loaders
    eval_max_future = max(eval_horizons)
    if dataset == 'CHBMIT':
        train_stride, val_stride, test_stride = 32, 16, 8
    else:
        train_stride, val_stride, test_stride = 4, 4, 1

    train_engines = bundle['ft_train']
    if label_fraction < 1.0:
        keys = sorted(train_engines.keys())
        n_keep = max(1, int(round(len(keys) * label_fraction)))
        rng = np.random.RandomState(seed + 7777)
        keep = sorted(rng.choice(keys, size=n_keep, replace=False).tolist())
        train_engines = {k: train_engines[k] for k in keep}
        print(f"  [ft] label_fraction={label_fraction} → "
              f"{len(train_engines)}/{len(bundle['ft_train'])} train engines",
              flush=True)

    train_ft = _build_event_concat(train_engines, stride=train_stride,
                                   max_context=max_context,
                                   max_future=eval_max_future)
    val_ft = _build_event_concat(bundle['ft_val'], stride=val_stride,
                                 max_context=max_context,
                                 max_future=eval_max_future)
    test_ft = _build_event_concat(bundle['ft_test'], stride=test_stride,
                                  max_context=max_context,
                                  max_future=eval_max_future)
    print(f"  [ft] train={len(train_ft)} val={len(val_ft)} test={len(test_ft)}",
          flush=True)
    if len(train_ft) == 0 or len(test_ft) == 0:
        print(f"  SKIP {tag}: empty FT datasets", flush=True)
        return None

    tloader = DataLoader(train_ft, batch_size=ft_batch, shuffle=True,
                         collate_fn=collate_event, num_workers=0)
    vloader = DataLoader(val_ft, batch_size=ft_batch, shuffle=False,
                         collate_fn=collate_event, num_workers=0)
    test_loader = DataLoader(test_ft, batch_size=ft_batch, shuffle=False,
                             collate_fn=collate_event, num_workers=0)

    # 4. FT
    t0 = time.time()
    ft_out = finetune_v30(model, tloader, vloader, eval_horizons,
                          max_horizon=max(eval_horizons),
                          train_horizons_dense=train_horizons_dense,
                          mode='pred_ft', lr=1e-3, n_epochs=ft_epochs,
                          patience=ft_patience, device=device, seed=seed)
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
                           'phase': 'v30'})
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
        'pretrain_best_loss': pre_best, 'pretrain_time_s': pre_time,
        'pretrain_ckpt': str(pre_ckpt) if pre_ckpt is not None else None,
        'ft_best_val': float(ft_out['best_val']), 'ft_time_s': ft_time,
        'eval_time_s': eval_time,
        'pooled_auprc': pooled_auprc, 'pooled_auroc': pooled_auroc,
        **h,
        'surface_path': str(surf_path), 'png_path': str(png_path),
        'ft_ckpt': str(ft_ckpt),
    }
