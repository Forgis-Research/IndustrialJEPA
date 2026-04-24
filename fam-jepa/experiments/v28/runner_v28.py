"""V28 shared runner: extends v27 _runner with three improvement options.

New options on top of v27:

  - ``lag_features``: list[int] of lag indices (e.g. [10, 50, 100]). When
    set, every channel is augmented with its lag-shifted versions BEFORE
    patching: ``x_aug[:, t, c]`` for original c, then ``x_aug[:, t, C+c]``
    for the lag-10 version, etc. Effective n_channels becomes C * (1+L).
    Zero-padded for the first ``max(lags)`` positions per stream. The
    intent is to encode drift gradient inside the token vector so that
    RevIN no longer erases it.

  - ``aux_stat_loss``: when True, FAM is wrapped with an extra stat-head
    that predicts the TARGET interval's (mean, std, slope) per-channel
    from h_t. Stats are computed from RAW (un-normalized) target. The
    aux loss is added at weight 0.1. Must be paired with a norm_mode that
    actually normalizes (e.g. 'revin' or 'none' with pre-z-scored data),
    so that the encoder genuinely needs the stats.

  - ``dense_ft``: when True, finetune samples K_dense random horizons per
    batch from [1, max_h] instead of using the fixed sparse horizons list.
    Eval still runs at the fixed horizons (so AUPRC/AUROC are comparable).

The three options are independent and can in principle be combined, but
v28 phase 2 runs each in isolation against the v27 baseline.

Datasets: v27 set + GECCO + BATADAL (both have local data + cached
Chronos-2 features but no v27 FAM benchmark).
"""

from __future__ import annotations

import copy
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset, Dataset
from sklearn.metrics import roc_auc_score, average_precision_score

FAM_DIR = Path('/home/sagemaker-user/IndustrialJEPA/fam-jepa')
V28_DIR = FAM_DIR / 'experiments/v28'
CKPT_DIR = V28_DIR / 'ckpts'
SURF_DIR = V28_DIR / 'surfaces'
LOG_DIR  = V28_DIR / 'logs'
RES_DIR  = V28_DIR / 'results'
for d in [CKPT_DIR, SURF_DIR, LOG_DIR, RES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(FAM_DIR))
sys.path.insert(0, str(FAM_DIR / 'experiments/v24'))   # _cmapss_raw
sys.path.insert(0, str(FAM_DIR / 'experiments/v27'))   # reuse v27 _runner

from model import FAM
from train import (
    PretrainDataset, EventDataset, collate_pretrain, collate_event,
    pretrain as pretrain_default, finetune as finetune_default,
    evaluate, save_surface, build_label_surface, EPS, LAMBDA_VAR,
)
from _runner import (
    LOADERS as V27_LOADERS, _global_zscore, per_horizon_diag, print_diag,
    _build_event_concat,
)

CMAPSS_HORIZONS = [1, 5, 10, 20, 50, 100, 150]
ANOMALY_HORIZONS = [1, 5, 10, 20, 50, 100, 150, 200]


# ---------------------------------------------------------------------------
# Extended dataset loaders (GECCO + BATADAL)
# ---------------------------------------------------------------------------

def _gecco() -> Dict:
    from data.gecco import load_gecco
    d = load_gecco(normalize=False)
    pre = d['pretrain_stream']
    return {
        'pretrain_seqs': pre,            # already a dict
        'ft_train': d['ft_train'], 'ft_val': d['ft_val'], 'ft_test': d['ft_test'],
        'n_channels': d['ft_train'][0]['test'].shape[1],
        'horizons': ANOMALY_HORIZONS,
        'subset': 'GECCO',
    }


def _batadal() -> Dict:
    from data.batadal import load_batadal
    d = load_batadal(normalize=False)
    pre = d['pretrain_stream']
    return {
        'pretrain_seqs': pre,
        'ft_train': d['ft_train'], 'ft_val': d['ft_val'], 'ft_test': d['ft_test'],
        'n_channels': d['ft_train'][0]['test'].shape[1],
        'horizons': ANOMALY_HORIZONS,
        'subset': 'BATADAL',
    }


LOADERS = dict(V27_LOADERS)
LOADERS['GECCO'] = _gecco
LOADERS['BATADAL'] = _batadal


# ---------------------------------------------------------------------------
# Try A: lag-feature augmentation
# ---------------------------------------------------------------------------

def _augment_lags(x: np.ndarray, lags: List[int]) -> np.ndarray:
    """Concatenate original x with lag-shifted versions along channel dim.

    x: (T, C). Output: (T, C * (1 + len(lags))). For each lag L, the output
    contains x[t-L] at position t (and zeros for t < L).
    """
    T, C = x.shape
    parts = [x]
    for L in lags:
        if L <= 0 or L >= T:
            shifted = np.zeros_like(x)
        else:
            shifted = np.zeros_like(x)
            shifted[L:] = x[:-L]
        parts.append(shifted)
    return np.concatenate(parts, axis=1).astype(np.float32)


def apply_lag_features(bundle: Dict, lags: List[int]) -> Dict:
    """Apply lag augmentation to all arrays in a bundle. Returns new bundle."""
    new = dict(bundle)
    new['pretrain_seqs'] = {k: _augment_lags(v, lags)
                            for k, v in bundle['pretrain_seqs'].items()}
    for key in ('ft_train', 'ft_val', 'ft_test'):
        new[key] = [{**e, 'test': _augment_lags(e['test'], lags)}
                    for e in bundle[key]]
    new['n_channels'] = bundle['n_channels'] * (1 + len(lags))
    new['lag_features'] = lags
    return new


# ---------------------------------------------------------------------------
# Try B: auxiliary stat-prediction head
# ---------------------------------------------------------------------------

class StatHead(nn.Module):
    """MLP head that predicts (mean, std, slope) per channel from h_t.

    Used during pretraining only. Output shape (B, 3 * C).
    """

    def __init__(self, d_model: int, n_channels: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, 3 * n_channels),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)


def _target_stats(target_raw: torch.Tensor,
                  target_mask: Optional[torch.Tensor]) -> torch.Tensor:
    """Compute per-channel (mean, std, slope) from a raw target interval.

    target_raw: (B, T, C). target_mask: (B, T) True = padding.
    Returns (B, 3 * C): [mean | std | slope] concatenated.
    """
    B, T, C = target_raw.shape
    if target_mask is not None:
        valid = (~target_mask).unsqueeze(-1).float()    # (B, T, 1)
        n = valid.sum(dim=1).clamp(min=1)               # (B, 1)
        mean = (target_raw * valid).sum(dim=1) / n      # (B, C)
        diff = (target_raw - mean.unsqueeze(1)) * valid
        var = (diff * diff).sum(dim=1) / n
        std = (var + 1e-6).sqrt()
        # Slope: linear regression of value vs t over valid positions.
        # For each batch element we use simple (last_valid - first_valid) / span
        # because per-batch lstsq is overkill; this captures direction+magnitude.
        first_idx = torch.zeros(B, dtype=torch.long, device=target_raw.device)
        last_idx = (valid.squeeze(-1) * torch.arange(T, device=target_raw.device)
                    .unsqueeze(0).float()).argmax(dim=1)
        last_idx = last_idx.clamp(min=1)
        span = (last_idx - first_idx).clamp(min=1).float()  # (B,)
        first_val = target_raw[torch.arange(B), first_idx]   # (B, C)
        last_val = target_raw[torch.arange(B), last_idx]
        slope = (last_val - first_val) / span.unsqueeze(-1)
    else:
        mean = target_raw.mean(dim=1)
        std = target_raw.std(dim=1) + 1e-6
        if T >= 2:
            slope = (target_raw[:, -1] - target_raw[:, 0]) / (T - 1)
        else:
            slope = torch.zeros_like(mean)
    return torch.cat([mean, std, slope], dim=-1)


@torch.no_grad()
def estimate_stat_normalization(loader, device: str = 'cuda',
                                max_batches: int = 50) -> tuple:
    """Scan loader for ~max_batches and compute global (μ, σ) of target stats.

    Returns (stat_mu, stat_sigma) tensors of shape (3 * n_channels,) on `device`.
    These are used to standardise the aux stat loss so its magnitude does not
    dominate the JEPA L1 loss (the v28 Try B failure mode).
    """
    samples = []
    n = 0
    for ctx, ctx_m, tgt, tgt_m, dt in loader:
        tgt, tgt_m = tgt.to(device), tgt_m.to(device)
        s = _target_stats(tgt, tgt_m)             # (B, 3*C)
        samples.append(s)
        n += s.shape[0]
        if len(samples) >= max_batches:
            break
    s_all = torch.cat(samples, dim=0)              # (N, 3*C)
    mu = s_all.mean(dim=0)
    sigma = s_all.std(dim=0).clamp(min=1e-3)
    print(f"  [stat-norm] estimated over N={n} samples; "
          f"mu range [{mu.min():.3f},{mu.max():.3f}]  "
          f"sigma range [{sigma.min():.3f},{sigma.max():.3f}]", flush=True)
    return mu, sigma


def pretrain_with_stat(model: FAM, stat_head: StatHead,
                       train_loader, val_loader,
                       lr: float = 3e-4, weight_decay: float = 0.01,
                       n_epochs: int = 30, patience: int = 5,
                       grad_clip: float = 1.0, device: str = 'cuda',
                       stat_weight: float = 0.1,
                       stat_normalize: bool = False) -> dict:
    """Pretrain FAM with auxiliary stat-prediction loss.

    The collate_pretrain function returns RAW (un-normalized) target tensors
    when the bundle was built without global z-score. When the bundle uses
    norm_mode='none' with pre-normalization, the target arrives already
    z-scored — that's the desired behaviour for the encoder, but the stats
    we ask the head to predict come from this same array so they're still
    a meaningful supervision signal (they describe the post-norm distribution
    of the target window).
    """
    model.to(device)
    stat_head.to(device)
    params = [p for p in model.parameters() if p.requires_grad] + list(stat_head.parameters())
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)

    # Estimate stat-normalization constants once if requested. The pred
    # head then learns to predict the standardised stats, which keeps the
    # L1 loss in the same order of magnitude as the JEPA L1.
    stat_mu, stat_sigma = None, None
    if stat_normalize:
        stat_mu, stat_sigma = estimate_stat_normalization(train_loader, device)

    best_loss = float('inf')
    best_state, best_stat_state, wait = None, None, 0
    history = []

    for epoch in range(n_epochs):
        model.train(); stat_head.train()
        losses, l_preds, l_stats = [], [], []
        h_pred_raw = None
        for ctx, ctx_m, tgt, tgt_m, dt in train_loader:
            ctx, ctx_m = ctx.to(device), ctx_m.to(device)
            tgt, tgt_m = tgt.to(device), tgt_m.to(device)
            dt = dt.to(device)

            h_t = model.encoder(ctx, ctx_m)
            h_pred_raw = model.predictor(h_t, dt)
            with torch.no_grad():
                h_target = model.target_encoder(tgt, tgt_m)
            pred_n = F.normalize(h_pred_raw, dim=-1)
            targ_n = F.normalize(h_target, dim=-1)
            l_pred = F.l1_loss(pred_n, targ_n.detach())
            l_var = F.relu(1.0 - h_pred_raw.std(dim=0)).mean()

            stat_pred = stat_head(h_t)                 # (B, 3 * C)
            target_stats = _target_stats(tgt, tgt_m)
            if stat_normalize:
                # Standardise both prediction and target so the L1 loss
                # is in z-units, comparable scale to the JEPA L1.
                stat_pred = (stat_pred - stat_mu) / stat_sigma
                target_stats = (target_stats - stat_mu) / stat_sigma
            l_stat = F.l1_loss(stat_pred, target_stats.detach())

            loss = l_pred + LAMBDA_VAR * l_var + stat_weight * l_stat
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, grad_clip)
            optimizer.step()
            model.update_ema()
            losses.append(loss.item())
            l_preds.append(l_pred.item())
            l_stats.append(l_stat.item())

        scheduler.step()
        train_loss = float(np.mean(losses))

        # Validation
        model.eval(); stat_head.eval()
        val_losses = []
        with torch.no_grad():
            for ctx, ctx_m, tgt, tgt_m, dt in val_loader:
                ctx, ctx_m = ctx.to(device), ctx_m.to(device)
                tgt, tgt_m = tgt.to(device), tgt_m.to(device)
                dt = dt.to(device)
                h_t = model.encoder(ctx, ctx_m)
                h_pred_raw = model.predictor(h_t, dt)
                h_target = model.target_encoder(tgt, tgt_m)
                l_pred = F.l1_loss(F.normalize(h_pred_raw, dim=-1),
                                   F.normalize(h_target, dim=-1))
                stat_pred = stat_head(h_t)
                tstats = _target_stats(tgt, tgt_m)
                if stat_normalize:
                    stat_pred = (stat_pred - stat_mu) / stat_sigma
                    tstats = (tstats - stat_mu) / stat_sigma
                l_stat = F.l1_loss(stat_pred, tstats)
                val_losses.append((l_pred + stat_weight * l_stat).item())
        val_loss = float(np.mean(val_losses)) if val_losses else train_loss

        with torch.no_grad():
            h_std = h_pred_raw.std(dim=0).mean().item() if h_pred_raw is not None else 0.0
        history.append({'epoch': epoch, 'train_loss': train_loss,
                        'val_loss': val_loss, 'h_std': h_std,
                        'l_pred': float(np.mean(l_preds)),
                        'l_stat': float(np.mean(l_stats))})
        print(f"  epoch {epoch:3d}  train={train_loss:.4f}  val={val_loss:.4f}  "
              f"l_pred={np.mean(l_preds):.4f}  l_stat={np.mean(l_stats):.4f}  "
              f"h_std={h_std:.3f}", flush=True)

        if h_std < 0.01:
            print("  COLLAPSED — aborting", flush=True); break

        if val_loss < best_loss:
            best_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            best_stat_state = copy.deepcopy(stat_head.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"  early stop at epoch {epoch}", flush=True); break

    if best_state is not None:
        model.load_state_dict(best_state)
        stat_head.load_state_dict(best_stat_state)
    return {'history': history, 'best_loss': best_loss}


# ---------------------------------------------------------------------------
# Try C: dense-horizon finetuning
# ---------------------------------------------------------------------------

def finetune_dense(model: FAM, train_loader, val_loader,
                   eval_horizons: List[int], max_horizon: int,
                   k_dense: int = 20, mode: str = 'pred_ft',
                   pos_weight: Optional[float] = None,
                   lr: float = 1e-3, weight_decay: float = 0.01,
                   n_epochs: int = 40, patience: int = 8,
                   device: str = 'cuda', seed: int = 42) -> dict:
    """Finetune predictor + event_head, sampling K random horizons per batch.

    The training horizons each batch are sorted ascending so the discrete-
    hazard CDF parameterization is well-defined.

    Validation uses the FIXED ``eval_horizons`` so val loss is comparable
    to the v27 baseline.
    """
    model.to(device)
    eval_h = torch.tensor(eval_horizons, dtype=torch.float32, device=device)

    if mode == 'pred_ft':
        for p in model.encoder.parameters():
            p.requires_grad = False
        params = list(model.predictor.parameters()) + list(model.event_head.parameters())
    else:
        params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)

    if pos_weight is None:
        # Compute pos_weight on EVAL horizons (the observable label set).
        n_pos, n_tot = 0, 0
        for ctx, ctx_m, tte, t_idx in train_loader:
            y = build_label_surface(tte.unsqueeze(1), eval_h.cpu()).squeeze(1)
            n_pos += y.sum().item(); n_tot += y.numel()
        pos_weight = max(1.0, min(1000.0, (n_tot - n_pos) / max(n_pos, 1)))
    pw = torch.tensor(pos_weight, device=device)

    g = torch.Generator(device='cpu')
    g.manual_seed(seed)
    best_val, best_state, wait = float('inf'), None, 0

    for epoch in range(n_epochs):
        model.train()
        losses = []
        for ctx, ctx_m, tte, t_idx in train_loader:
            ctx, ctx_m = ctx.to(device), ctx_m.to(device)
            tte = tte.to(device)
            # Sample K_dense horizons in [1, max_horizon], sort ascending.
            sampled = torch.randint(1, max_horizon + 1, (k_dense,), generator=g)
            sampled = torch.unique(sampled)
            sampled, _ = torch.sort(sampled)
            h_train = sampled.float().to(device)
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
                vl.append(-(pw * y * torch.log(p) + (1 - y) * torch.log(1 - p)).mean().item())
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
# Main runner — supports any combination of the three options
# ---------------------------------------------------------------------------

def run_one(dataset: str, norm_mode: str, seed: int,
            tag_suffix: str = '',
            lag_features: Optional[List[int]] = None,
            aux_stat: bool = False,
            stat_normalize: bool = False,
            dense_ft: bool = False,
            k_dense: int = 20,
            pre_epochs: int = 30, pre_patience: int = 5,
            ft_epochs: int = 30, ft_patience: int = 8,
            n_cuts_train: int = 40, n_cuts_val: int = 10,
            max_context: int = 512,
            pre_batch: int = 64, ft_batch: int = 128,
            device: str = 'cuda') -> Dict:
    torch.manual_seed(seed); np.random.seed(seed)
    extra = ''
    if lag_features:
        extra += f'_lag{"_".join(str(L) for L in lag_features)}'
    if aux_stat:
        extra += '_statz' if stat_normalize else '_stat'
    if dense_ft:
        extra += f'_dense{k_dense}'
    if tag_suffix:
        extra += f'_{tag_suffix}'
    tag = f"{dataset}_{norm_mode}{extra}_s{seed}"
    print(f"\n{'='*70}\n=== {tag}\n{'='*70}", flush=True)

    bundle = LOADERS[dataset]()
    if norm_mode == 'none':
        bundle = _global_zscore(bundle)
    if lag_features:
        bundle = apply_lag_features(bundle, lag_features)
        print(f"  lag_features={lag_features} -> n_channels={bundle['n_channels']}",
              flush=True)
    horizons = bundle['horizons']
    n_channels = bundle['n_channels']
    print(f"  loaded {dataset}: pretrain_seqs={len(bundle['pretrain_seqs'])}, "
          f"n_channels={n_channels}, horizons={horizons}", flush=True)

    # 1. Pretrain
    model = FAM(n_channels=n_channels, patch_size=16, d_model=256,
                n_heads=4, n_layers=2, d_ff=256, dropout=0.1,
                ema_momentum=0.99, predictor_hidden=256, norm_mode=norm_mode)
    pre_ckpt = CKPT_DIR / f'{tag}_pretrain.pt'
    pre_time, pre_best, pre_hist_len = 0.0, float('nan'), 0
    if pre_ckpt.exists():
        print(f"  [pretrain] loading existing ckpt: {pre_ckpt.name}", flush=True)
        model.load_state_dict(torch.load(pre_ckpt, map_location='cpu'))
    else:
        train_pre = PretrainDataset(bundle['pretrain_seqs'],
                                    n_cuts=n_cuts_train, max_context=max_context,
                                    delta_t_max=150, delta_t_min=1, seed=seed)
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
        if aux_stat:
            stat_head = StatHead(d_model=256, n_channels=n_channels, hidden=256)
            pre_out = pretrain_with_stat(model, stat_head, tlo, vlo,
                                         lr=3e-4, n_epochs=pre_epochs,
                                         patience=pre_patience, device=device,
                                         stat_normalize=stat_normalize)
        else:
            pre_out = pretrain_default(model, tlo, vlo, lr=3e-4, n_epochs=pre_epochs,
                                       patience=pre_patience, device=device)
        pre_time = time.time() - t0
        pre_best = float(pre_out['best_loss'])
        pre_hist_len = len(pre_out['history'])
        print(f"  [pretrain] done in {pre_time:.1f}s best_val={pre_best:.4f}", flush=True)
        torch.save(model.state_dict(), pre_ckpt)

    # 2. Finetune
    train_ft = _build_event_concat(bundle['ft_train'], stride=4, max_context=max_context)
    val_ft   = _build_event_concat(bundle['ft_val'], stride=4, max_context=max_context)
    test_ft  = _build_event_concat(bundle['ft_test'], stride=1, max_context=max_context)
    print(f"  [ft] train={len(train_ft)}, val={len(val_ft)}, test={len(test_ft)}",
          flush=True)
    if len(train_ft) == 0 or len(test_ft) == 0:
        print(f"  SKIP {tag}: empty FT datasets"); return None

    tloader = DataLoader(train_ft, batch_size=ft_batch, shuffle=True,
                         collate_fn=collate_event, num_workers=0)
    vloader = DataLoader(val_ft, batch_size=ft_batch, shuffle=False,
                         collate_fn=collate_event, num_workers=0)
    test_loader = DataLoader(test_ft, batch_size=ft_batch, shuffle=False,
                             collate_fn=collate_event, num_workers=0)

    t0 = time.time()
    if dense_ft:
        ft_out = finetune_dense(model, tloader, vloader, eval_horizons=horizons,
                                max_horizon=max(horizons), k_dense=k_dense,
                                mode='pred_ft', lr=1e-3, n_epochs=ft_epochs,
                                patience=ft_patience, device=device, seed=seed)
    else:
        ft_out = finetune_default(model, tloader, vloader, horizons, mode='pred_ft',
                                  lr=1e-3, n_epochs=ft_epochs, patience=ft_patience,
                                  device=device)
    ft_time = time.time() - t0
    ft_ckpt = CKPT_DIR / f'{tag}_pred_ft.pt'
    torch.save(model.state_dict(), ft_ckpt)
    print(f"  [ft] done in {ft_time:.1f}s best_val={ft_out['best_val']:.4f}", flush=True)

    # 3. Evaluate
    t0 = time.time()
    eval_out = evaluate(model, test_loader, horizons, mode='pred_ft', device=device)
    eval_time = time.time() - t0
    p_surf, y_surf = eval_out['p_surface'], eval_out['y_surface']
    rows = per_horizon_diag(p_surf, y_surf, horizons)
    pooled_auprc = float(eval_out['primary']['auprc'])
    pooled_auroc = float(eval_out['primary']['auroc'])
    mono = float(eval_out['monotonicity']['violation_rate'])
    print_diag(tag, rows, pooled_auprc, pooled_auroc)

    # Mean per-horizon AUROC + base-rate baseline (the v28 honest metrics)
    valid = [r for r in rows if not (np.isnan(r['auroc']))]
    mean_h_auroc = float(np.mean([r['auroc'] for r in valid])) if valid else float('nan')
    rng = np.random.RandomState(0)
    base_rates = y_surf.mean(axis=0)
    p_base = (np.tile(base_rates, (y_surf.shape[0], 1)) +
              rng.normal(0, 1e-6, y_surf.shape))
    base_auprc = float(average_precision_score(y_surf.ravel(), p_base.ravel()))
    base_h_auroc = float(np.mean([
        roc_auc_score(y_surf[:, i], p_base[:, i])
        for i in range(len(horizons)) if 0 < y_surf[:, i].mean() < 1
    ])) if valid else float('nan')
    print(f"  [{tag}] mean h-AUROC={mean_h_auroc:.4f} (base {base_h_auroc:.4f}, "
          f"Δ={mean_h_auroc-base_h_auroc:+.4f})  pooled AUPRC={pooled_auprc:.4f} "
          f"(base {base_auprc:.4f}, Δ={pooled_auprc-base_auprc:+.4f})", flush=True)

    surf_path = SURF_DIR / f'{tag}.npz'
    save_surface(surf_path, p_surf, y_surf, horizons, eval_out['t_index'],
                 metadata={'dataset': dataset, 'norm_mode': norm_mode, 'seed': seed,
                           'lag_features': lag_features, 'aux_stat': aux_stat,
                           'dense_ft': dense_ft, 'k_dense': k_dense if dense_ft else 0,
                           'phase': 'v28'})

    return {
        'tag': tag, 'dataset': dataset, 'norm_mode': norm_mode, 'seed': seed,
        'lag_features': lag_features, 'aux_stat': aux_stat,
        'dense_ft': dense_ft, 'k_dense': k_dense if dense_ft else 0,
        'n_params': sum(p.numel() for p in model.parameters()),
        'pretrain_best_loss': pre_best, 'pretrain_epochs_run': pre_hist_len,
        'pretrain_time_s': pre_time, 'ft_best_val': float(ft_out['best_val']),
        'ft_time_s': ft_time, 'eval_time_s': eval_time,
        'pooled_auprc': pooled_auprc, 'pooled_auroc': pooled_auroc,
        'pooled_auprc_base': base_auprc,
        'pooled_auprc_above_base': pooled_auprc - base_auprc,
        'mean_h_auroc': mean_h_auroc,
        'mean_h_auroc_base': base_h_auroc,
        'mean_h_auroc_above_base': mean_h_auroc - base_h_auroc,
        'monotonicity_violation_rate': mono,
        'per_horizon': rows,
        'surface_path': str(surf_path),
        'pretrain_ckpt': str(pre_ckpt), 'ft_ckpt': str(ft_ckpt),
    }


def run_and_persist(dataset: str, norm_mode: str, seeds: List[int],
                    out_json: Path, **kw) -> List[Dict]:
    results = []
    for seed in seeds:
        res = run_one(dataset, norm_mode, seed, **kw)
        if res is not None:
            results.append(res)
        with open(out_json, 'w') as f:
            json.dump({'dataset': dataset, 'norm_mode': norm_mode,
                       'options': {k: kw.get(k) for k in
                                   ('lag_features', 'aux_stat', 'dense_ft',
                                    'k_dense', 'tag_suffix')},
                       'seeds': seeds[:len(results)], 'results': results},
                      f, indent=2, default=str)
        print(f"  wrote {out_json}", flush=True)
    return results
