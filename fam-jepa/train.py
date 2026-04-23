"""
FAM training and evaluation — single entry point.

Usage:
  python train.py pretrain  --dataset FD001 --epochs 50
  python train.py finetune  --dataset FD001 --checkpoint ckpt.pt --seeds 3
  python train.py evaluate  --dataset FD001 --checkpoint ckpt.pt

See ARCHITECTURE.md for the specification.
"""

import argparse
import copy
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset

from model import FAM
from evaluation.surface_metrics import (
    evaluate_probability_surface, auprc_per_horizon, monotonicity_violation_rate,
)
from evaluation.losses import build_label_surface

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

P = 16               # global patch size
D_MODEL = 256
N_HEADS = 4
N_LAYERS = 2
D_FF = 256
DROPOUT = 0.1
EMA_MOMENTUM = 0.99
PREDICTOR_HIDDEN = 256
LAMBDA_VAR = 0.04

# Horizons per dataset (in native time units)
HORIZONS = {
    'default': [1, 5, 10, 20, 50, 100, 150],
    'sepsis':  [1, 2, 3, 6, 12, 24, 48],
}

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def get_horizons(dataset: str) -> List[int]:
    key = dataset.lower()
    for prefix, h in HORIZONS.items():
        if key.startswith(prefix):
            return h
    return HORIZONS['default']


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

class PretrainDataset(Dataset):
    """
    Sample random (t, Δt) pairs from each time series.

    Given t and Δt, the context and target are deterministic:
      context = x[0:t]       (variable length, up to max_context)
      target  = x(t : t+Δt]  (cumulative interval, no gap)
    """

    def __init__(self, sequences: Dict[int, np.ndarray],
                 n_cuts: int = 30, max_context: int = 512,
                 delta_t_max: int = 150, delta_t_min: int = 1,
                 seed: int = 42):
        self.max_context = max_context
        rng = np.random.RandomState(seed)
        self.samples = []

        for sid, seq in sequences.items():
            T = len(seq)
            for _ in range(n_cuts):
                # Need: t >= 1 (at least 1 context step)
                #        t + delta_t <= T
                dt_hi = min(delta_t_max, T - 1)
                if dt_hi < delta_t_min:
                    continue
                # Sample Δt ~ LogUniform
                u = rng.uniform(math.log(delta_t_min), math.log(dt_hi))
                dt = max(delta_t_min, min(int(math.exp(u)), dt_hi))
                # Sample t
                t_lo = 1
                t_hi = T - dt
                if t_hi < t_lo:
                    continue
                t = int(rng.randint(t_lo, t_hi + 1))
                self.samples.append((sid, t, dt))

        self.sequences = sequences

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sid, t, dt = self.samples[idx]
        seq = self.sequences[sid]
        # Context: last max_context steps up to t
        ctx_start = max(0, t - self.max_context)
        context = torch.from_numpy(seq[ctx_start:t].copy()).float()
        # Target: cumulative interval (t, t+dt]
        target = torch.from_numpy(seq[t:t + dt].copy()).float()
        return context, target, dt


def collate_pretrain(batch):
    """Pad context and target to batch max lengths."""
    contexts, targets, dts = zip(*batch)
    B = len(contexts)

    max_ctx = max(c.shape[0] for c in contexts)
    max_tgt = max(t.shape[0] for t in targets)
    C = contexts[0].shape[1]

    ctx_padded = torch.zeros(B, max_ctx, C)
    ctx_mask = torch.ones(B, max_ctx, dtype=torch.bool)  # True = padding
    tgt_padded = torch.zeros(B, max_tgt, C)
    tgt_mask = torch.ones(B, max_tgt, dtype=torch.bool)

    for i in range(B):
        tc = contexts[i].shape[0]
        ctx_padded[i, :tc] = contexts[i]
        ctx_mask[i, :tc] = False
        tt = targets[i].shape[0]
        tgt_padded[i, :tt] = targets[i]
        tgt_mask[i, :tt] = False

    dt_tensor = torch.tensor(dts, dtype=torch.float32)
    return ctx_padded, ctx_mask, tgt_padded, tgt_mask, dt_tensor


class EventDataset(Dataset):
    """
    Sliding-context dataset for finetuning / evaluation.

    Each item: (context, tte, t_index)
      context: (ctx_len, C)  — observations up to time t
      tte: scalar time-to-next-event from t. inf if none.
      t_index: absolute time index

    ``min_context`` (default 128 = 8 tokens at P=16) enforces the
    ARCHITECTURE.md rule: a transformer with fewer than 8 tokens degenerates.
    Timesteps t < min_context are skipped entirely.
    """

    def __init__(self, x: np.ndarray, labels: np.ndarray,
                 max_context: int = 512, stride: int = 1,
                 max_future: int = 200, min_context: int = 128):
        self.x = torch.from_numpy(np.asarray(x, dtype=np.float32))
        self.labels = np.asarray(labels, dtype=np.int32)
        self.max_context = max_context
        self.stride = stride
        self.max_future = max_future
        self.min_context = min_context

        T = len(x)
        t_start = max(1, min_context)
        t_end = min(T, T - 1)  # at least 1 future step for tte
        if t_end <= t_start:
            self.starts = []
        else:
            self.starts = list(range(t_start, t_end, stride))
        self._tte = self._compute_tte(self.labels, max_future)

    @staticmethod
    def _compute_tte(labels, max_future):
        """For each t, tte[t] = next d>=1 s.t. labels[t+d]==1, else inf."""
        T = len(labels)
        tte = np.full(T, np.inf, dtype=np.float32)
        next_anom = -1
        for t in range(T - 1, -1, -1):
            if next_anom != -1 and (next_anom - t) <= max_future:
                tte[t] = float(next_anom - t)
            if labels[t] == 1:
                next_anom = t
        return tte

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, i):
        t = self.starts[i]
        ctx_start = max(0, t - self.max_context)
        context = self.x[ctx_start:t]
        tte = float(self._tte[t])
        return context, torch.tensor(tte), torch.tensor(t, dtype=torch.long)


def collate_event(batch):
    """Pad contexts to batch max length."""
    contexts, ttes, ts = zip(*batch)
    B = len(contexts)
    max_len = max(c.shape[0] for c in contexts)
    C = contexts[0].shape[1]

    ctx_padded = torch.zeros(B, max_len, C)
    ctx_mask = torch.ones(B, max_len, dtype=torch.bool)
    for i in range(B):
        tc = contexts[i].shape[0]
        ctx_padded[i, :tc] = contexts[i]
        ctx_mask[i, :tc] = False

    return ctx_padded, ctx_mask, torch.stack(ttes), torch.stack(ts)


# ---------------------------------------------------------------------------
# Pretraining
# ---------------------------------------------------------------------------

def pretrain(model: FAM, train_loader, val_loader=None,
             lr: float = 3e-4, weight_decay: float = 0.01,
             n_epochs: int = 50, patience: int = 5,
             grad_clip: float = 1.0, device: str = DEVICE) -> dict:
    """Pretrain with L1 loss on L2-normalized representations + var_reg."""
    model.to(device)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)

    best_loss = float('inf')
    best_state = None
    wait = 0
    history = []

    for epoch in range(n_epochs):
        model.train()
        losses = []
        for ctx, ctx_m, tgt, tgt_m, dt in train_loader:
            ctx, ctx_m = ctx.to(device), ctx_m.to(device)
            tgt, tgt_m = tgt.to(device), tgt_m.to(device)
            dt = dt.to(device)

            h_t = model.encoder(ctx, ctx_m)
            h_pred_raw = model.predictor(h_t, dt)

            with torch.no_grad():
                h_target = model.target_encoder(tgt, tgt_m)

            # L1 loss on L2-normalized representations
            pred_n = F.normalize(h_pred_raw, dim=-1)
            targ_n = F.normalize(h_target, dim=-1)
            l_pred = F.l1_loss(pred_n, targ_n.detach())
            # Variance regularizer
            l_var = F.relu(1.0 - h_pred_raw.std(dim=0)).mean()
            loss = l_pred + LAMBDA_VAR * l_var

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            model.update_ema()
            losses.append(loss.item())

        scheduler.step()
        train_loss = np.mean(losses)

        # Validation
        val_loss = train_loss
        if val_loader is not None:
            val_loss = _eval_pretrain_loss(model, val_loader, device)

        # Collapse check (use last batch's predictor output)
        with torch.no_grad():
            h_std = h_pred_raw.std(dim=0).mean().item()

        history.append({
            'epoch': epoch, 'train_loss': train_loss,
            'val_loss': val_loss, 'h_std': h_std,
        })
        print(f"  epoch {epoch:3d}  train={train_loss:.4f}  "
              f"val={val_loss:.4f}  h_std={h_std:.3f}", flush=True)

        if h_std < 0.01:
            print("  COLLAPSED — aborting", flush=True)
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
    return {'history': history, 'best_loss': best_loss}


@torch.no_grad()
def _eval_pretrain_loss(model, loader, device):
    model.eval()
    losses = []
    for ctx, ctx_m, tgt, tgt_m, dt in loader:
        ctx, ctx_m = ctx.to(device), ctx_m.to(device)
        tgt, tgt_m = tgt.to(device), tgt_m.to(device)
        dt = dt.to(device)
        pred, target = model.pretrain_forward(ctx, tgt, dt, ctx_m, tgt_m)
        losses.append(F.l1_loss(pred, target).item())
    return np.mean(losses)


# ---------------------------------------------------------------------------
# Finetuning (pred-FT)
# ---------------------------------------------------------------------------

def finetune(model: FAM, train_loader, val_loader,
             horizons: List[int], mode: str = 'pred_ft',
             pos_weight: Optional[float] = None,
             lr: float = 1e-3, weight_decay: float = 0.01,
             n_epochs: int = 40, patience: int = 8,
             device: str = DEVICE) -> dict:
    """Finetune predictor + event_head with pos-weighted BCE."""
    model.to(device)
    h_tensor = torch.tensor(horizons, dtype=torch.float32, device=device)

    # Set up trainable params based on mode
    if mode == 'pred_ft':
        for p in model.encoder.parameters():
            p.requires_grad = False
        params = list(model.predictor.parameters()) + list(model.event_head.parameters())
    else:  # e2e
        params = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)

    # Estimate pos_weight from training data if not provided
    if pos_weight is None:
        pos_weight = _estimate_pos_weight(train_loader, h_tensor)
    pw_tensor = torch.tensor(pos_weight, device=device)

    best_val = float('inf')
    best_state = None
    wait = 0

    for epoch in range(n_epochs):
        model.train()
        losses = []
        for ctx, ctx_m, tte, t_idx in train_loader:
            ctx, ctx_m = ctx.to(device), ctx_m.to(device)
            tte = tte.to(device)

            logits = model.finetune_forward(ctx, h_tensor, ctx_m, mode)  # (B, K)
            y = build_label_surface(tte.unsqueeze(1), h_tensor)  # (B, 1, K)
            y = y.squeeze(1)  # (B, K)

            loss = F.binary_cross_entropy_with_logits(
                logits, y, pos_weight=pw_tensor, reduction='mean')

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        scheduler.step()
        train_loss = np.mean(losses)

        # Validation
        val_loss = _eval_ft_loss(model, val_loader, h_tensor, pw_tensor, mode, device)

        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"  early stop at epoch {epoch}", flush=True)
                break

        if epoch % 5 == 0 or wait == 0:
            print(f"  epoch {epoch:3d}  train={train_loss:.4f}  "
                  f"val={val_loss:.4f}", flush=True)

    if best_state is not None:
        model.load_state_dict(best_state)
    return {'best_val': best_val, 'final_epoch': epoch}


@torch.no_grad()
def _eval_ft_loss(model, loader, h_tensor, pw_tensor, mode, device):
    model.eval()
    losses = []
    for ctx, ctx_m, tte, t_idx in loader:
        ctx, ctx_m = ctx.to(device), ctx_m.to(device)
        tte = tte.to(device)
        logits = model.finetune_forward(ctx, h_tensor, ctx_m, mode)
        y = build_label_surface(tte.unsqueeze(1), h_tensor).squeeze(1)
        loss = F.binary_cross_entropy_with_logits(
            logits, y, pos_weight=pw_tensor, reduction='mean')
        losses.append(loss.item())
    return np.mean(losses)


def _estimate_pos_weight(loader, horizons, clamp_max=1000.0):
    n_pos, n_tot = 0, 0
    for ctx, ctx_m, tte, t_idx in loader:
        y = build_label_surface(tte.unsqueeze(1), horizons.cpu()).squeeze(1)
        n_pos += y.sum().item()
        n_tot += y.numel()
    n_neg = max(n_tot - n_pos, 0)
    return min(max(n_neg / max(n_pos, 1), 1.0), clamp_max)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model: FAM, test_loader, horizons: List[int],
             mode: str = 'pred_ft', device: str = DEVICE) -> dict:
    """
    Evaluate on test set → probability surface → AUPRC.

    Returns dict with primary metrics + surfaces for storage.
    """
    model.to(device)
    model.eval()
    h_tensor = torch.tensor(horizons, dtype=torch.float32, device=device)

    p_list, y_list, t_list = [], [], []
    for ctx, ctx_m, tte, t_idx in test_loader:
        ctx, ctx_m = ctx.to(device), ctx_m.to(device)
        tte = tte.to(device)

        logits = model.finetune_forward(ctx, h_tensor, ctx_m, mode)
        p = torch.sigmoid(logits)
        y = build_label_surface(tte.unsqueeze(1), h_tensor).squeeze(1)

        p_list.append(p.cpu().numpy())
        y_list.append(y.cpu().numpy())
        t_list.append(t_idx.numpy())

    p_surface = np.concatenate(p_list, axis=0)  # (N, K)
    y_surface = np.concatenate(y_list, axis=0)
    t_index = np.concatenate(t_list, axis=0)

    # Metrics
    primary = evaluate_probability_surface(p_surface, y_surface)
    per_h = auprc_per_horizon(p_surface, y_surface, horizon_labels=horizons)
    mono = monotonicity_violation_rate(p_surface)

    return {
        'primary': primary,
        'per_horizon': per_h,
        'monotonicity': mono,
        'p_surface': p_surface,
        'y_surface': y_surface,
        't_index': t_index,
    }


def save_surface(path, p_surface, y_surface, horizons, t_index,
                 metadata: Optional[dict] = None):
    """Save probability surface to .npz."""
    meta = {} if metadata is None else dict(metadata)
    np.savez(path,
             p_surface=np.asarray(p_surface, dtype=np.float32),
             y_surface=np.asarray(y_surface, dtype=np.int8),
             horizons=np.asarray(horizons, dtype=np.int32),
             t_index=np.asarray(t_index, dtype=np.int64),
             meta=np.asarray(list(meta.items()), dtype=object)
             if meta else np.array([], dtype=object))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='FAM training')
    sub = parser.add_subparsers(dest='command')

    p_pre = sub.add_parser('pretrain')
    p_pre.add_argument('--dataset', required=True)
    p_pre.add_argument('--epochs', type=int, default=50)
    p_pre.add_argument('--batch-size', type=int, default=64)
    p_pre.add_argument('--lr', type=float, default=3e-4)
    p_pre.add_argument('--seed', type=int, default=42)
    p_pre.add_argument('--out', type=str, default='checkpoints/')

    p_ft = sub.add_parser('finetune')
    p_ft.add_argument('--dataset', required=True)
    p_ft.add_argument('--checkpoint', required=True)
    p_ft.add_argument('--mode', default='pred_ft', choices=['pred_ft', 'e2e'])
    p_ft.add_argument('--epochs', type=int, default=40)
    p_ft.add_argument('--batch-size', type=int, default=256)
    p_ft.add_argument('--seed', type=int, default=42)
    p_ft.add_argument('--out', type=str, default='results/')

    p_ev = sub.add_parser('evaluate')
    p_ev.add_argument('--dataset', required=True)
    p_ev.add_argument('--checkpoint', required=True)
    p_ev.add_argument('--out', type=str, default='results/')

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    print(f"FAM {args.command} — dataset={args.dataset}, device={DEVICE}")
    # Dataset loading is left to experiment scripts that import this module.
    # The CLI is a template — actual experiments populate the data loaders.
    print("Use this module via import, or extend the CLI with dataset loaders.")


if __name__ == '__main__':
    main()
