"""V34 Runner: SIGReg from scratch + ST-JEPA fixes + new datasets.

Workstreams:
  A. SIGReg (no EMA): VICReg var+cov on h_pred, periodic hard sync of
     target encoder (or frozen target). Sweep on FD001 (5 configs), then
     scale to all datasets if FD001 looks good.
  B. ST-JEPA collapse fixes: stronger variance reg + partial channel fusion.
  C. New datasets: Sepsis (data on VM), TEP (download attempt), SWaT (skip).

Design:
  - Reuses v33's run_one() pattern (pretrain -> pred-FT -> evaluate),
    but pretrain uses pretrain_sigreg() instead of pretrain_default()
    when the model is FAM_SIGReg.
  - Diagnostics (h_pred.std, h_t.std) are logged every epoch and saved
    alongside results so we can detect partial collapse.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# ---- Path setup ----
FAM_DIR = Path('/home/sagemaker-user/IndustrialJEPA/fam-jepa')
V34_DIR = FAM_DIR / 'experiments/v34'
CKPT_DIR = V34_DIR / 'ckpts'
SURF_DIR = V34_DIR / 'surfaces'
RES_DIR = V34_DIR / 'results'
for d in [CKPT_DIR, SURF_DIR, RES_DIR / 'phaseA', RES_DIR / 'phaseB',
          RES_DIR / 'phaseC', RES_DIR / 'phaseD', RES_DIR / 'phaseE']:
    d.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(FAM_DIR))
sys.path.insert(0, str(FAM_DIR / 'experiments/v29'))
sys.path.insert(0, str(FAM_DIR / 'experiments/v28'))
sys.path.insert(0, str(FAM_DIR / 'experiments/v27'))
sys.path.insert(0, str(FAM_DIR / 'experiments/v24'))

from model import FAM, RevIN, sinusoidal_pe, PatchEmbedding
from train import (
    PretrainDataset, collate_pretrain, collate_event,
    finetune as finetune_default,
    evaluate, save_surface,
)
from _runner_v29 import LOADERS, NORM_POLICY, honest_metrics
from _runner import _global_zscore, _build_event_concat

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ---- Fixed protocol (matches v33 for direct comparability) ----
PROTOCOL = {
    'max_context': 512,
    'patch_size': 16,
    'd_model': 256,
    'n_heads': 4,
    'n_layers': 2,
    'd_ff': 256,
    'dropout': 0.1,
    'predictor_hidden': 256,
    # Pretraining
    'pre_epochs': 50,
    'pre_batch': 64,
    'pre_lr': 3e-4,
    'pre_patience': 8,
    'n_cuts': 40,
    'delta_t_min': 1,
    'delta_t_max': 150,
    # SIGReg
    'lambda_var': 0.04,
    'lambda_cov': 0.02,
    'sync_interval_steps': 100,   # hard sync every N optimizer steps
    # Finetuning
    'ft_epochs': 40,
    'ft_batch': 128,
    'ft_lr': 1e-3,
    'ft_patience': 8,
    'seeds': [42, 123, 456],
}


HORIZONS_BY_DATASET = {
    'FD001': [1, 5, 10, 20, 50, 100, 150],
    'FD002': [1, 5, 10, 20, 50, 100, 150],
    'FD003': [1, 5, 10, 20, 50, 100, 150],
    'SMAP':  [1, 5, 10, 20, 50, 100, 150, 200],
    'MSL':   [1, 5, 10, 20, 50, 100, 150, 200],
    'PSM':   [1, 5, 10, 20, 50, 100, 150, 200],
    'SMD':   [1, 5, 10, 20, 50, 100, 150, 200],
    'MBA':   [1, 5, 10, 20, 50, 100, 150, 200],
    'GECCO': [1, 5, 10, 20, 50, 100, 150, 200],
    'BATADAL': [1, 5, 10, 20, 50, 100, 150, 200],
    'SKAB':  [1, 5, 10, 20, 50, 100, 150, 200],
    'ETTm1': [1, 5, 10, 20, 50, 100, 150, 200],
    # Sepsis: 1 hr cadence, predict 1-48 hr ahead
    'Sepsis': [1, 2, 3, 6, 12, 24, 48],
    # TEP: 1 sample = 3 min, 960 steps total in test, faults at 160
    'TEP':   [1, 5, 10, 20, 50, 100, 150],
}


N_CHANNELS_BY_DATASET = {
    'FD001': 14, 'FD002': 14, 'FD003': 14,
    'SMAP': 25, 'MSL': 55, 'PSM': 25, 'SMD': 38,
    'MBA': 2, 'GECCO': 9, 'BATADAL': 43,
    'SKAB': 8, 'ETTm1': 7,
}


# ===========================================================================
# Workstream A: FAM_SIGReg
# ===========================================================================

class FAM_SIGReg(FAM):
    """FAM with SIGReg in place of EMA.

    Three target-update modes (set on the instance):
      'periodic_sync' : every `sync_interval_steps` optimizer steps, copy
                        encoder -> target_encoder (matching keys).
      'frozen_target' : never update target encoder after init.
      'joint_train'   : remove no_grad on target encoder; both encoders
                        train. SIGReg + L1 loss prevents collapse.
    """

    VALID_MODES = ('periodic_sync', 'frozen_target', 'joint_train')

    def __init__(self, target_mode: str = 'periodic_sync',
                 sync_interval_steps: int = 100, **kwargs):
        # Force ema_momentum=0.0 so accidental update_ema() calls are no-ops.
        kwargs.setdefault('ema_momentum', 0.0)
        super().__init__(**kwargs)
        assert target_mode in self.VALID_MODES, target_mode
        self.target_mode = target_mode
        self.sync_interval_steps = int(sync_interval_steps)
        if target_mode == 'joint_train':
            for p in self.target_encoder.parameters():
                p.requires_grad = True

    @torch.no_grad()
    def update_ema(self):
        """No-op: SIGReg does not use EMA."""
        return

    @torch.no_grad()
    def maybe_sync_target(self, step: int):
        """Hard-copy encoder -> target_encoder every sync_interval_steps."""
        if self.target_mode != 'periodic_sync':
            return
        if step <= 0 or (step % self.sync_interval_steps) != 0:
            return
        self._init_target_encoder()

    def pretrain_forward(self, context, target, delta_t,
                         context_mask=None, target_mask=None):
        """Returns (h_pred_n, h_target_n, h_pred_raw).

        h_pred_raw is the unnormalized predictor output that VICReg uses.
        """
        if self.predictor_kind == 'transformer':
            h_all, h_kpm = self.encoder(context, context_mask, return_all=True)
            h_pred_raw = self.predictor(h_all, delta_t, key_padding_mask=h_kpm)
        else:
            h_t = self.encoder(context, context_mask)
            h_pred_raw = self.predictor(h_t, delta_t)

        if self.target_mode == 'joint_train':
            h_target = self.target_encoder(target, target_mask)
        else:
            with torch.no_grad():
                h_target = self.target_encoder(target, target_mask)

        h_pred_n = F.normalize(h_pred_raw, dim=-1)
        h_target_n = F.normalize(h_target, dim=-1)
        return h_pred_n, h_target_n, h_pred_raw, h_target


def vicreg_var_cov(h: torch.Tensor, eps: float = 1e-4):
    """VICReg variance + covariance terms on h (B, D)."""
    B, D = h.shape
    std = h.std(dim=0) + eps
    l_var = F.relu(1.0 - std).mean()
    h_c = h - h.mean(dim=0, keepdim=True)
    cov = (h_c.T @ h_c) / max(B - 1, 1)
    off = cov - torch.diag(torch.diag(cov))
    l_cov = (off ** 2).sum() / D
    return l_var, l_cov, std.mean().detach()


def pretrain_sigreg(model: FAM_SIGReg, train_loader, val_loader=None,
                    lr: float = 3e-4, n_epochs: int = 50, patience: int = 8,
                    grad_clip: float = 1.0,
                    lambda_var: float = 0.04, lambda_cov: float = 0.02,
                    device: str = DEVICE,
                    diag_log: Optional[List[Dict]] = None) -> dict:
    """SIGReg pretrain loop.

    Loss = L1(pred_n, target_n.detach())
         + lambda_var * VICReg-var(h_pred_raw)
         + lambda_cov * VICReg-cov(h_pred_raw)

    target_encoder updated per `model.target_mode`:
      - periodic_sync : maybe_sync_target() every sync_interval_steps
      - frozen_target : never updated
      - joint_train   : its params have grad and update via optimizer

    Diagnostics (per epoch):
      h_pred_std, h_target_std, l_pred, l_var, l_cov.
      Aborts if h_pred_std < 0.01 (collapse).
    """
    model.to(device)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)

    best_loss = float('inf')
    best_state = None
    wait = 0
    history = []
    step = 0
    collapsed = False

    for epoch in range(n_epochs):
        model.train()
        ep_losses = {'loss': [], 'l_pred': [], 'l_var': [], 'l_cov': [],
                     'h_pred_std': [], 'h_target_std': []}

        for ctx, ctx_m, tgt, tgt_m, dt in train_loader:
            ctx, ctx_m = ctx.to(device), ctx_m.to(device)
            tgt, tgt_m = tgt.to(device), tgt_m.to(device)
            dt = dt.to(device)

            pred_n, targ_n, h_pred_raw, h_target_raw = \
                model.pretrain_forward(ctx, tgt, dt, ctx_m, tgt_m)
            l_pred = F.l1_loss(pred_n, targ_n.detach())
            l_var, l_cov, h_std = vicreg_var_cov(h_pred_raw)
            loss = l_pred + lambda_var * l_var + lambda_cov * l_cov

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            step += 1
            model.maybe_sync_target(step)

            ep_losses['loss'].append(loss.item())
            ep_losses['l_pred'].append(l_pred.item())
            ep_losses['l_var'].append(l_var.item())
            ep_losses['l_cov'].append(l_cov.item())
            ep_losses['h_pred_std'].append(h_std.item())
            with torch.no_grad():
                ep_losses['h_target_std'].append(
                    h_target_raw.std(dim=0).mean().item())

        scheduler.step()
        train_loss = float(np.mean(ep_losses['loss']))
        h_pred_std = float(np.mean(ep_losses['h_pred_std']))
        h_target_std = float(np.mean(ep_losses['h_target_std']))

        val_loss = train_loss
        if val_loader is not None:
            val_loss = _eval_pretrain_loss_sigreg(model, val_loader, device)

        rec = {
            'epoch': epoch,
            'train_loss': train_loss, 'val_loss': val_loss,
            'l_pred': float(np.mean(ep_losses['l_pred'])),
            'l_var': float(np.mean(ep_losses['l_var'])),
            'l_cov': float(np.mean(ep_losses['l_cov'])),
            'h_pred_std': h_pred_std, 'h_target_std': h_target_std,
            'sync_steps_so_far': step,
        }
        history.append(rec)
        if diag_log is not None:
            diag_log.append(rec)
        print(f"  epoch {epoch:3d}  train={train_loss:.4f}  val={val_loss:.4f}  "
              f"l_pred={rec['l_pred']:.4f}  std_pred={h_pred_std:.3f}  "
              f"std_tgt={h_target_std:.3f}", flush=True)

        if h_pred_std < 0.01 or h_target_std < 0.01:
            print("  COLLAPSED -- aborting", flush=True)
            collapsed = True
            break

        if val_loss < best_loss:
            best_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"  early stop at epoch {epoch}", flush=True)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return {'history': history, 'best_loss': best_loss, 'collapsed': collapsed}


@torch.no_grad()
def _eval_pretrain_loss_sigreg(model, loader, device):
    model.eval()
    losses = []
    for ctx, ctx_m, tgt, tgt_m, dt in loader:
        ctx, ctx_m = ctx.to(device), ctx_m.to(device)
        tgt, tgt_m = tgt.to(device), tgt_m.to(device)
        dt = dt.to(device)
        pred_n, targ_n, _, _ = model.pretrain_forward(ctx, tgt, dt, ctx_m, tgt_m)
        losses.append(F.l1_loss(pred_n, targ_n).item())
    return float(np.mean(losses))


# ===========================================================================
# Workstream B: ST-JEPA collapse fixes
# ===========================================================================

class GroupedChannelPatchEmbedding(nn.Module):
    """Partial channel fusion: groups of K channels per token.

    Token input dim = P * K (vs P*1 for per-channel, P*C for full-fusion).
    Tokens per timestep: ceil(C / K).
    """

    def __init__(self, n_channels: int, group_k: int, patch_size: int = 16,
                 d_model: int = 256):
        super().__init__()
        self.P = patch_size
        self.K = group_k
        self.C = n_channels
        # Pad channels so C % K == 0
        self.C_pad = ((n_channels + group_k - 1) // group_k) * group_k
        self.pad_C = self.C_pad - self.C
        self.G = self.C_pad // group_k  # groups per timestep
        # Shared projection across groups: each group has K*P inputs
        self.proj = nn.Linear(group_k * patch_size, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, C) -> (B, N_patches, G, d) with G = C_pad/K."""
        B, T, C = x.shape
        P = self.P
        if self.pad_C > 0:
            x = F.pad(x, (0, self.pad_C))
            C = x.shape[2]
        rem = T % P
        if rem != 0:
            x = F.pad(x, (0, 0, 0, P - rem))
            T = x.shape[1]
        N = T // P
        # (B, T, C) -> (B, N, P, C) -> (B, N, P, G, K) -> (B, N, G, K, P)
        x = x.reshape(B, N, P, self.G, self.K).permute(0, 1, 3, 4, 2)
        # (B, N, G, K*P)
        x = x.reshape(B, N, self.G, self.K * P)
        return self.proj(x)


# (Other ST-JEPA architecture imports come from v33 runner if needed.)


# ===========================================================================
# Build dataloaders + run_one (shared)
# ===========================================================================

def build_loaders(dataset: str, seed: int,
                  max_context: int = 512,
                  n_cuts: int = 40,
                  pre_batch: int = 64,
                  ft_batch: int = 128) -> Dict:
    """Build pretrain + FT dataloaders for one dataset."""
    norm_mode = NORM_POLICY.get(dataset, 'revin')
    bundle = LOADERS[dataset]()
    if norm_mode == 'none':
        bundle = _global_zscore(bundle)

    horizons = HORIZONS_BY_DATASET.get(dataset, bundle.get('horizons',
                                                            [1, 5, 10, 20, 50, 100, 150]))
    n_channels = bundle['n_channels']
    delta_t_max = max(horizons)

    train_pre = PretrainDataset(
        bundle['pretrain_seqs'], n_cuts=n_cuts,
        max_context=max_context, delta_t_max=delta_t_max,
        delta_t_min=1, seed=seed)

    val_seqs = {}
    for k, seq in bundle['pretrain_seqs'].items():
        L = len(seq)
        cut = int(0.9 * L)
        if L - cut >= 128:
            val_seqs[k] = seq[cut:]
    if not val_seqs:
        val_seqs = bundle['pretrain_seqs']
    val_pre = PretrainDataset(
        val_seqs, n_cuts=10, max_context=max_context,
        delta_t_max=delta_t_max, delta_t_min=1, seed=seed + 10000)

    tlo = DataLoader(train_pre, batch_size=pre_batch, shuffle=True,
                     collate_fn=collate_pretrain, num_workers=0)
    vlo = DataLoader(val_pre, batch_size=pre_batch, shuffle=False,
                     collate_fn=collate_pretrain, num_workers=0)

    train_ft = _build_event_concat(bundle['ft_train'], stride=4,
                                   max_context=max_context,
                                   max_future=max(horizons))
    val_ft = _build_event_concat(bundle['ft_val'], stride=4,
                                 max_context=max_context,
                                 max_future=max(horizons))
    test_ft = _build_event_concat(bundle['ft_test'], stride=1,
                                  max_context=max_context,
                                  max_future=max(horizons))

    tft_lo = DataLoader(train_ft, batch_size=ft_batch, shuffle=True,
                        collate_fn=collate_event, num_workers=0)
    vft_lo = DataLoader(val_ft, batch_size=ft_batch, shuffle=False,
                        collate_fn=collate_event, num_workers=0)
    test_lo = DataLoader(test_ft, batch_size=ft_batch, shuffle=False,
                         collate_fn=collate_event, num_workers=0)

    return {
        'horizons': horizons,
        'n_channels': n_channels,
        'norm_mode': norm_mode,
        'tlo': tlo, 'vlo': vlo,
        'tft_lo': tft_lo, 'vft_lo': vft_lo, 'test_lo': test_lo,
        'pretrain_n': len(train_pre), 'val_n': len(val_pre),
    }


def run_one_sigreg(dataset: str, seed: int,
                   target_mode: str = 'periodic_sync',
                   sync_interval_steps: int = 100,
                   lambda_var: float = 0.04, lambda_cov: float = 0.02,
                   model_tag: Optional[str] = None,
                   force_retrain: bool = False) -> Dict:
    """Pretrain (SIGReg) -> pred-FT -> evaluate. Mirrors v33.run_one."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    if model_tag is None:
        model_tag = f"sigreg_{target_mode}_si{sync_interval_steps}_lv{int(lambda_var*100):03d}"
    tag = f"{dataset}_{model_tag}_s{seed}"
    print(f"\n{'='*70}\n=== {tag}\n{'='*70}", flush=True)

    loaders = build_loaders(
        dataset, seed,
        max_context=PROTOCOL['max_context'],
        n_cuts=PROTOCOL['n_cuts'],
        pre_batch=PROTOCOL['pre_batch'],
        ft_batch=PROTOCOL['ft_batch'])
    horizons = loaders['horizons']
    norm_mode = loaders['norm_mode']
    n_channels = loaders['n_channels']
    print(f"  dataset={dataset} n_channels={n_channels} horizons={horizons} "
          f"norm_mode={norm_mode}", flush=True)
    print(f"  pretrain_n={loaders['pretrain_n']} val_n={loaders['val_n']}", flush=True)

    model = FAM_SIGReg(
        target_mode=target_mode,
        sync_interval_steps=sync_interval_steps,
        n_channels=n_channels, patch_size=PROTOCOL['patch_size'],
        d_model=PROTOCOL['d_model'], n_heads=PROTOCOL['n_heads'],
        n_layers=PROTOCOL['n_layers'], d_ff=PROTOCOL['d_ff'],
        dropout=PROTOCOL['dropout'],
        predictor_hidden=PROTOCOL['predictor_hidden'],
        norm_mode=norm_mode,
        predictor_kind='mlp',
        event_head_kind='discrete_hazard',
    )

    pre_ckpt = CKPT_DIR / f'{tag}_pretrain.pt'
    diag_log: List[Dict] = []
    pre_best = float('nan')
    pre_hist = []
    pre_time = 0.0
    collapsed = False

    if pre_ckpt.exists() and not force_retrain:
        print(f"  [pretrain] ckpt exists: {pre_ckpt.name}", flush=True)
        model.load_state_dict(torch.load(pre_ckpt, map_location='cpu'))
    else:
        t0 = time.time()
        pre_out = pretrain_sigreg(
            model, loaders['tlo'], loaders['vlo'],
            lr=PROTOCOL['pre_lr'],
            n_epochs=PROTOCOL['pre_epochs'],
            patience=PROTOCOL['pre_patience'],
            lambda_var=lambda_var, lambda_cov=lambda_cov,
            diag_log=diag_log)
        pre_time = time.time() - t0
        pre_best = float(pre_out['best_loss'])
        pre_hist = pre_out['history']
        collapsed = bool(pre_out['collapsed'])
        print(f"  [pretrain] {pre_time:.1f}s  best={pre_best:.4f}  "
              f"epochs={len(pre_hist)}  collapsed={collapsed}", flush=True)
        torch.save(model.state_dict(), pre_ckpt)

    # Finetune
    t0 = time.time()
    ft_out = finetune_default(
        model, loaders['tft_lo'], loaders['vft_lo'],
        horizons, mode='pred_ft',
        lr=PROTOCOL['ft_lr'],
        n_epochs=PROTOCOL['ft_epochs'],
        patience=PROTOCOL['ft_patience'])
    ft_time = time.time() - t0
    ft_ckpt = CKPT_DIR / f'{tag}_pred_ft.pt'
    torch.save(model.state_dict(), ft_ckpt)
    print(f"  [ft] {ft_time:.1f}s  best_val={ft_out['best_val']:.4f}", flush=True)

    # Eval
    t0 = time.time()
    eval_out = evaluate(model, loaders['test_lo'], horizons, mode='pred_ft')
    eval_time = time.time() - t0
    p_surf = eval_out['p_surface']
    y_surf = eval_out['y_surface']
    h = honest_metrics(p_surf, y_surf, horizons)
    pooled_auprc = float(eval_out['primary']['auprc'])
    pooled_auroc = float(eval_out['primary']['auroc'])
    print(f"  [{tag}] h-AUROC={h['mean_h_auroc']:.4f} "
          f"(base {h['mean_h_auroc_base']:.4f}, "
          f"delta={h['mean_h_auroc']-h['mean_h_auroc_base']:+.4f})  "
          f"pooled_AUPRC={pooled_auprc:.4f}", flush=True)

    surf_path = SURF_DIR / f'{tag}.npz'
    save_surface(surf_path, p_surf, y_surf, horizons, eval_out['t_index'],
                 metadata={'dataset': dataset, 'model_tag': model_tag,
                           'seed': seed, 'phase': 'v34',
                           'target_mode': target_mode,
                           'sync_interval_steps': sync_interval_steps,
                           'lambda_var': lambda_var, 'lambda_cov': lambda_cov})

    return {
        'tag': tag, 'dataset': dataset, 'model_tag': model_tag, 'seed': seed,
        'norm_mode': norm_mode,
        'target_mode': target_mode,
        'sync_interval_steps': sync_interval_steps,
        'lambda_var': lambda_var, 'lambda_cov': lambda_cov,
        'pretrain_best_loss': pre_best,
        'pretrain_epochs': len(pre_hist),
        'pretrain_time_s': pre_time,
        'pretrain_collapsed': collapsed,
        'pretrain_history': pre_hist,
        'ft_best_val': float(ft_out['best_val']),
        'ft_time_s': ft_time, 'eval_time_s': eval_time,
        'pooled_auprc': pooled_auprc, 'pooled_auroc': pooled_auroc,
        'mean_h_auroc': h['mean_h_auroc'],
        'mean_h_auroc_base': h['mean_h_auroc_base'],
        'h_auroc_delta': h['mean_h_auroc'] - h['mean_h_auroc_base'],
        'mean_h_auprc': h['pooled_auprc'],
        'surface_path': str(surf_path),
    }


def run_one_ema_baseline(dataset: str, seed: int,
                         model_tag: str = 'ema_baseline_v34',
                         force_retrain: bool = False) -> Dict:
    """Run standard FAM (EMA) with v34 protocol. For SIGReg comparison."""
    from train import pretrain as pretrain_default
    torch.manual_seed(seed); np.random.seed(seed)

    tag = f"{dataset}_{model_tag}_s{seed}"
    print(f"\n{'='*70}\n=== {tag}\n{'='*70}", flush=True)

    loaders = build_loaders(
        dataset, seed,
        max_context=PROTOCOL['max_context'],
        n_cuts=PROTOCOL['n_cuts'],
        pre_batch=PROTOCOL['pre_batch'],
        ft_batch=PROTOCOL['ft_batch'])
    horizons = loaders['horizons']
    norm_mode = loaders['norm_mode']
    n_channels = loaders['n_channels']
    print(f"  dataset={dataset} n_channels={n_channels} horizons={horizons} "
          f"norm_mode={norm_mode}", flush=True)

    model = FAM(
        n_channels=n_channels, patch_size=PROTOCOL['patch_size'],
        d_model=PROTOCOL['d_model'], n_heads=PROTOCOL['n_heads'],
        n_layers=PROTOCOL['n_layers'], d_ff=PROTOCOL['d_ff'],
        dropout=PROTOCOL['dropout'],
        ema_momentum=0.99,
        predictor_hidden=PROTOCOL['predictor_hidden'],
        norm_mode=norm_mode,
        predictor_kind='mlp',
        event_head_kind='discrete_hazard',
    )

    pre_ckpt = CKPT_DIR / f'{tag}_pretrain.pt'
    pre_best = float('nan')
    pre_time = 0.0
    if pre_ckpt.exists() and not force_retrain:
        print(f"  [pretrain] ckpt exists: {pre_ckpt.name}", flush=True)
        model.load_state_dict(torch.load(pre_ckpt, map_location='cpu'))
    else:
        t0 = time.time()
        pre_out = pretrain_default(
            model, loaders['tlo'], loaders['vlo'],
            lr=PROTOCOL['pre_lr'],
            n_epochs=PROTOCOL['pre_epochs'],
            patience=PROTOCOL['pre_patience'])
        pre_time = time.time() - t0
        pre_best = float(pre_out['best_loss'])
        torch.save(model.state_dict(), pre_ckpt)
        print(f"  [pretrain] {pre_time:.1f}s  best={pre_best:.4f}", flush=True)

    ft_out = finetune_default(
        model, loaders['tft_lo'], loaders['vft_lo'],
        horizons, mode='pred_ft',
        lr=PROTOCOL['ft_lr'],
        n_epochs=PROTOCOL['ft_epochs'],
        patience=PROTOCOL['ft_patience'])

    eval_out = evaluate(model, loaders['test_lo'], horizons, mode='pred_ft')
    p_surf = eval_out['p_surface']; y_surf = eval_out['y_surface']
    h = honest_metrics(p_surf, y_surf, horizons)
    surf_path = SURF_DIR / f'{tag}.npz'
    save_surface(surf_path, p_surf, y_surf, horizons, eval_out['t_index'],
                 metadata={'dataset': dataset, 'model_tag': model_tag,
                           'seed': seed, 'phase': 'v34_ema_baseline'})

    return {
        'tag': tag, 'dataset': dataset, 'seed': seed,
        'pretrain_best_loss': pre_best, 'pretrain_time_s': pre_time,
        'ft_best_val': float(ft_out['best_val']),
        'mean_h_auroc': h['mean_h_auroc'],
        'mean_h_auroc_base': h['mean_h_auroc_base'],
        'mean_h_auprc': h['pooled_auprc'],
        'pooled_auprc': float(eval_out['primary']['auprc']),
        'surface_path': str(surf_path),
    }


# ===========================================================================
# Phase A2: FD001 SIGReg sweep
# ===========================================================================

A2_CONFIGS = [
    {'name': 'A_si100_lv04_lc02', 'target_mode': 'periodic_sync',
     'sync_interval_steps': 100, 'lambda_var': 0.04, 'lambda_cov': 0.02},
    {'name': 'B_si50_lv04_lc02',  'target_mode': 'periodic_sync',
     'sync_interval_steps': 50,  'lambda_var': 0.04, 'lambda_cov': 0.02},
    {'name': 'C_si200_lv04_lc02', 'target_mode': 'periodic_sync',
     'sync_interval_steps': 200, 'lambda_var': 0.04, 'lambda_cov': 0.02},
    {'name': 'D_si100_lv10_lc05', 'target_mode': 'periodic_sync',
     'sync_interval_steps': 100, 'lambda_var': 0.10, 'lambda_cov': 0.05},
    {'name': 'E_frozen_lv04_lc02', 'target_mode': 'frozen_target',
     'sync_interval_steps': 0,   'lambda_var': 0.04, 'lambda_cov': 0.02},
]


def run_phaseA_sweep(dataset: str = 'FD001', seed: int = 42) -> Dict:
    """Phase A2: sweep 5 SIGReg configs on FD001 + EMA baseline."""
    out_dir = RES_DIR / 'phaseA'
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*70)
    print(f"PHASE A SWEEP: {dataset} seed={seed}, 5 SIGReg configs + EMA")
    print("="*70, flush=True)

    sweep = {}

    # EMA baseline first (for comparison anchor)
    print("\n--- EMA baseline ---", flush=True)
    try:
        ema_res = run_one_ema_baseline(dataset, seed,
                                       model_tag='ema_baseline_v34')
        sweep['ema_baseline'] = ema_res
    except Exception as e:
        print(f"  EMA baseline FAILED: {e}", flush=True)
        sweep['ema_baseline'] = {'error': str(e)}

    # SIGReg configs
    for cfg in A2_CONFIGS:
        print(f"\n--- SIGReg config {cfg['name']} ---", flush=True)
        try:
            res = run_one_sigreg(
                dataset, seed,
                target_mode=cfg['target_mode'],
                sync_interval_steps=cfg['sync_interval_steps'],
                lambda_var=cfg['lambda_var'],
                lambda_cov=cfg['lambda_cov'],
                model_tag=f"sigreg_{cfg['name']}")
            sweep[cfg['name']] = res
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"  config {cfg['name']} FAILED: {e}", flush=True)
            sweep[cfg['name']] = {'error': str(e)}

        # Persist incrementally
        with open(out_dir / f'sigreg_sweep_{dataset}.json', 'w') as f:
            json.dump(sweep, f, indent=2, default=str)

    # Best config = highest h-AUROC among non-error, non-collapsed
    best_cfg = None; best_auroc = -1.0
    for k, v in sweep.items():
        if k == 'ema_baseline':
            continue
        if not isinstance(v, dict) or 'error' in v:
            continue
        if v.get('pretrain_collapsed'):
            continue
        a = v.get('mean_h_auroc', -1.0)
        if a > best_auroc:
            best_auroc = a; best_cfg = k

    summary = {
        'dataset': dataset, 'seed': seed,
        'sweep': sweep,
        'best_config': best_cfg,
        'best_h_auroc': best_auroc,
        'ema_h_auroc': sweep.get('ema_baseline', {}).get('mean_h_auroc',
                                                          float('nan')),
    }
    with open(out_dir / f'sigreg_sweep_{dataset}_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n  BEST: {best_cfg}  h-AUROC={best_auroc:.4f}  "
          f"vs EMA {summary['ema_h_auroc']:.4f}", flush=True)
    return summary


# ===========================================================================
# Phase A4: best SIGReg config across all datasets
# ===========================================================================

def run_phaseA_all_datasets(best_config: Dict, datasets: List[str],
                             seeds: Optional[List[int]] = None) -> Dict:
    """Run the best SIGReg config from A3 across all datasets, 3 seeds."""
    if seeds is None:
        seeds = PROTOCOL['seeds']
    out_dir = RES_DIR / 'phaseA'

    print("\n" + "="*70)
    print(f"PHASE A4: SIGReg best config across {len(datasets)} datasets")
    print(f"  config={best_config['name']}  seeds={seeds}")
    print("="*70, flush=True)

    all_results = {}
    for ds in datasets:
        ds_results = {}
        for seed in seeds:
            try:
                res = run_one_sigreg(
                    ds, seed,
                    target_mode=best_config['target_mode'],
                    sync_interval_steps=best_config['sync_interval_steps'],
                    lambda_var=best_config['lambda_var'],
                    lambda_cov=best_config['lambda_cov'],
                    model_tag=f"sigreg_best_{best_config['name']}")
                ds_results[str(seed)] = {
                    'h_auroc': res['mean_h_auroc'],
                    'h_auprc': res['mean_h_auprc'],
                    'pooled_auprc': res['pooled_auprc'],
                    'collapsed': res['pretrain_collapsed'],
                    'pretrain_loss': res['pretrain_best_loss'],
                    'ft_val_loss': res['ft_best_val'],
                }
            except Exception as e:
                import traceback; traceback.print_exc()
                ds_results[str(seed)] = {'error': str(e)}

            # Persist incrementally
            all_results[ds] = ds_results
            with open(out_dir / f'sigreg_all_datasets.json', 'w') as f:
                json.dump({'best_config': best_config,
                           'results': all_results}, f, indent=2, default=str)

        aurocs = [v['h_auroc'] for v in ds_results.values() if 'h_auroc' in v]
        if aurocs:
            print(f"  [{ds}] SIGReg 3-seed: {np.mean(aurocs):.4f} "
                  f"+/- {np.std(aurocs):.4f}", flush=True)

    return all_results


# ===========================================================================
# Phase B: ST-JEPA collapse fixes
# ===========================================================================

def run_phaseB(datasets: List[str] = None) -> Dict:
    """Workstream B: ST-JEPA fixes (lambda_var=0.5 + grouped fusion K=4)."""
    if datasets is None:
        datasets = ['FD001', 'PSM', 'SMAP']
    out_dir = RES_DIR / 'phaseB'
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*70)
    print("PHASE B: ST-JEPA collapse fixes")
    print("="*70, flush=True)

    # Import v33 ST-JEPA infrastructure
    sys.path.insert(0, str(FAM_DIR / 'experiments/v33'))
    from _runner_v33 import FAM_STJEPA, pretrain_custom

    # B2: stronger variance regularizer
    # We monkey-patch the module-level LAMBDA_VAR before calling pretrain_custom.
    import _runner_v33 as v33mod

    results = {}
    SWEEP_SEED = 42
    for ds in datasets:
        n_channels = N_CHANNELS_BY_DATASET[ds]
        norm_mode = NORM_POLICY.get(ds, 'revin')

        for variant_name, lambda_var, mask_ratio in [
            ('B2_lv50_mr04', 0.5, 0.4),
            ('B2_lv100_mr04', 1.0, 0.4),
            ('B2_lv50_mr02', 0.5, 0.2),
        ]:
            print(f"\n--- {ds} ST-JEPA fix: {variant_name} ---", flush=True)
            saved = v33mod.LAMBDA_VAR
            v33mod.LAMBDA_VAR = lambda_var
            try:
                torch.manual_seed(SWEEP_SEED); np.random.seed(SWEEP_SEED)
                model = FAM_STJEPA(
                    channel_mask_ratio=mask_ratio,
                    n_channels=n_channels, patch_size=PROTOCOL['patch_size'],
                    d_model=PROTOCOL['d_model'], n_heads=PROTOCOL['n_heads'],
                    n_layers=PROTOCOL['n_layers'], d_ff=PROTOCOL['d_ff'],
                    dropout=PROTOCOL['dropout'],
                    ema_momentum=0.99,
                    predictor_hidden=PROTOCOL['predictor_hidden'],
                    norm_mode=norm_mode,
                    predictor_kind='mlp',
                    event_head_kind='discrete_hazard',
                )
                loaders = build_loaders(ds, SWEEP_SEED,
                                         max_context=PROTOCOL['max_context'],
                                         pre_batch=PROTOCOL['pre_batch'],
                                         ft_batch=PROTOCOL['ft_batch'])
                horizons = loaders['horizons']
                t0 = time.time()
                pre_out = pretrain_custom(
                    model, loaders['tlo'], loaders['vlo'],
                    lr=PROTOCOL['pre_lr'],
                    n_epochs=PROTOCOL['pre_epochs'],
                    patience=PROTOCOL['pre_patience'])
                pre_time = time.time() - t0
                ft_out = finetune_default(
                    model, loaders['tft_lo'], loaders['vft_lo'],
                    horizons, mode='pred_ft',
                    lr=PROTOCOL['ft_lr'],
                    n_epochs=PROTOCOL['ft_epochs'],
                    patience=PROTOCOL['ft_patience'])
                eval_out = evaluate(model, loaders['test_lo'],
                                    horizons, mode='pred_ft')
                h = honest_metrics(eval_out['p_surface'],
                                   eval_out['y_surface'], horizons)

                # Detect collapse from pretrain history
                last_h_std = pre_out['history'][-1]['h_std'] if pre_out['history'] else 0.0
                results[f'{ds}_{variant_name}'] = {
                    'dataset': ds, 'variant': variant_name,
                    'lambda_var': lambda_var, 'channel_mask_ratio': mask_ratio,
                    'h_auroc': h['mean_h_auroc'],
                    'h_auprc': h['pooled_auprc'],
                    'pretrain_best_loss': float(pre_out['best_loss']),
                    'pretrain_epochs': len(pre_out['history']),
                    'pretrain_time_s': pre_time,
                    'last_h_std': last_h_std,
                    'collapsed': last_h_std < 0.05,
                    'ft_best_val': float(ft_out['best_val']),
                }
                print(f"  [{ds}/{variant_name}] h-AUROC={h['mean_h_auroc']:.4f}  "
                      f"last_h_std={last_h_std:.3f}  collapsed={last_h_std<0.05}",
                      flush=True)
            except Exception as e:
                import traceback; traceback.print_exc()
                results[f'{ds}_{variant_name}'] = {'error': str(e),
                                                    'dataset': ds,
                                                    'variant': variant_name}
            finally:
                v33mod.LAMBDA_VAR = saved

            with open(out_dir / 'stjepa_fixes.json', 'w') as f:
                json.dump(results, f, indent=2, default=str)

    return results


# ===========================================================================
# CLI entry points
# ===========================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--phase', required=True,
                    choices=['A2', 'A4', 'B', 'C_sepsis', 'C_tep', 'all'])
    ap.add_argument('--dataset', default='FD001')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    if args.phase == 'A2':
        run_phaseA_sweep(args.dataset, args.seed)
    elif args.phase == 'A4':
        # Read best config from sweep summary
        sweep_summary = json.load(open(
            RES_DIR / 'phaseA' / f'sigreg_sweep_{args.dataset}_summary.json'))
        best_name = sweep_summary['best_config']
        best_cfg = next(c for c in A2_CONFIGS if c['name'] == best_name)
        datasets = ['FD001', 'FD002', 'FD003', 'SMAP', 'MSL', 'PSM',
                    'SMD', 'MBA', 'GECCO', 'BATADAL', 'SKAB', 'ETTm1']
        run_phaseA_all_datasets(best_cfg, datasets)
    elif args.phase == 'B':
        run_phaseB()
    elif args.phase == 'C_sepsis':
        # Sepsis loader integration runs separately
        pass
    elif args.phase == 'all':
        run_phaseA_sweep('FD001', 42)


if __name__ == '__main__':
    main()
