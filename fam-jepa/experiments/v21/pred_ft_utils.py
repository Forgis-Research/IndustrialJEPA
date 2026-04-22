"""V21 Predictor Finetuning Utilities — Probability Surface Edition.

Changes from v20:
  - Head: shared Linear(d, 1) applied per horizon (EventHead) — replaces 16d->1
  - Loss: positive-weighted BCE — replaces MSE
  - Output: probability surface p(t, Δt) on a fixed grid of Δt (in STEPS)
  - Storage: .npz surfaces — recompute any metric from stored arrays

Downstream modes (C-MAPSS and anomaly):
  probe_h   : freeze enc+pred.  head on h_past, broadcast across horizons
  pred_ft   : freeze enc.       unfreeze pred + head
  e2e       : unfreeze enc+pred + head
  scratch   : random init everything + head

Horizons: default fixed grid HORIZONS_STEPS = [1,2,3,5,10,15,20,30,50,100]
During training Δt is sampled uniformly in [1, 100] per batch to densify
supervision; during eval the surface is read off the fixed grid.
"""
from __future__ import annotations

import copy
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Canonical evaluation grid (in STEPS). Same across datasets.
HORIZONS_STEPS: list = [1, 2, 3, 5, 10, 15, 20, 30, 50, 100]
MAX_TRAIN_HORIZON = 100  # sample Δt uniformly from [1, MAX_TRAIN_HORIZON]


# ---------------------------------------------------------------------------
# Heads
# ---------------------------------------------------------------------------

class EventHead(nn.Module):
    """Shared Linear(d, 1) applied per-horizon. Returns logits.

    Takes predictor outputs h_hat stacked across horizons and produces
    per-horizon logits. The same linear layer is used at every Δt, so the
    predictor's horizon conditioning is what differentiates Δt columns.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.linear = nn.Linear(d_model, 1)

    def forward(self, h_hats: torch.Tensor) -> torch.Tensor:
        # h_hats: (B, K, d) -> logits: (B, K)
        x = self.norm(h_hats)
        return self.linear(x).squeeze(-1)


class ProbeHEvent(nn.Module):
    """Probe-h baseline: Linear(d, K) on h_past only. Ignores predictor.

    Uses a separate output per horizon since there is no horizon input.
    """

    def __init__(self, d_model: int, n_horizons: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.linear = nn.Linear(d_model, n_horizons)

    def forward(self, h_past: torch.Tensor) -> torch.Tensor:
        return self.linear(self.norm(h_past))  # (B, K)


# ---------------------------------------------------------------------------
# Predictor multi-horizon forward (vectorized)
# ---------------------------------------------------------------------------

def predictor_multi_horizon(predictor: nn.Module,
                            h_past: torch.Tensor,
                            horizons: torch.Tensor) -> torch.Tensor:
    """Run predictor at every horizon in `horizons`, vectorized.

    h_past: (B, d). horizons: (K,) long. Returns (B, K, d).
    """
    B, d = h_past.shape
    K = horizons.shape[0]
    h = h_past.unsqueeze(1).expand(B, K, d).reshape(B * K, d)
    k = horizons.unsqueeze(0).expand(B, K).reshape(-1).to(
        device=h_past.device, dtype=torch.float32)
    out = predictor(h, k)  # (B*K, d)
    return out.view(B, K, d)


# ---------------------------------------------------------------------------
# Anomaly windowed dataset with time-to-event labels
# ---------------------------------------------------------------------------

class AnomalyWindowDataset(torch.utils.data.Dataset):
    """Sliding-window dataset for anomaly datasets.

    Each item: (x_past, tte)
      x_past: (W, C) context window ending at time t
      tte:    scalar time-to-next-event (from t+1 onward). inf if none in
              the next max_future steps.

    Args:
        x:            (T, C) time series
        labels:       (T,) binary (1=anomalous timestep). May be all zeros
                      (unlabeled) — in that case tte will always be inf and
                      BCE training is impossible.
        window:       context length
        stride:       step between windows
        max_future:   look-ahead for tte; must be >= max horizon used at
                      training or eval. Anything beyond is treated as inf.
        t_start, t_end: optional index bounds for chronological splits.
    """

    def __init__(self, x: np.ndarray, labels: np.ndarray,
                 window: int = 100, stride: int = 1,
                 max_future: int = 200,
                 t_start: Optional[int] = None,
                 t_end: Optional[int] = None):
        self.x = torch.from_numpy(np.asarray(x, dtype=np.float32))
        self.labels = np.asarray(labels, dtype=np.int32)
        self.window = int(window)
        self.stride = int(stride)
        self.max_future = int(max_future)

        T = len(x)
        t_start = window if t_start is None else max(int(t_start), window)
        t_end = T if t_end is None else min(int(t_end), T)
        # We need max_future lookahead AFTER t. Clamp so tte is well-defined.
        t_end = min(t_end, T - 1)  # at least 1 future step
        self.starts = list(range(t_start, t_end, self.stride))
        # Precompute tte at every index in the labels range
        self._tte = self._compute_tte(self.labels, max_future)

    @staticmethod
    def _compute_tte(labels: np.ndarray, max_future: int) -> np.ndarray:
        """For each t, tte[t] = next d >= 1 such that labels[t+d]==1, else inf.

        Runs in O(T). Only looks up to max_future ahead. Anything beyond is inf.
        """
        T = len(labels)
        tte = np.full(T, np.inf, dtype=np.float32)
        # Scan right-to-left; maintain index of next anomaly
        next_anom = -1
        # We only want t+d with d >= 1, so next_anom must be strictly > t
        for t in range(T - 1, -1, -1):
            if next_anom != -1 and (next_anom - t) <= max_future:
                tte[t] = float(next_anom - t)
            # Update: if this timestep IS an anomaly, it becomes the next
            # anomaly for earlier t's (earlier t's have t < current_t so
            # d = current_t - t >= 1).
            if labels[t] == 1:
                next_anom = t
        return tte

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, i):
        t = self.starts[i]
        past = self.x[t - self.window: t]  # (W, C)
        tte = float(self._tte[t])  # time-to-event from time t
        return past, torch.tensor(tte, dtype=torch.float32), torch.tensor(t, dtype=torch.long)


def collate_anomaly_window(batch):
    """Stack fixed-size windows (no padding needed)."""
    pasts, ttes, ts = zip(*batch)
    x = torch.stack(pasts, dim=0)  # (B, W, C)
    # Dense (non-padded) mask: False everywhere.
    B, W, _ = x.shape
    mask = torch.zeros(B, W, dtype=torch.bool)
    tte = torch.stack(ttes, dim=0)  # (B,)
    t = torch.stack(ts, dim=0)
    return x, mask, tte, t


# ---------------------------------------------------------------------------
# C-MAPSS surface dataset (wraps existing CMAPSSWindowedDataset)
# ---------------------------------------------------------------------------

class CMAPSSSurfaceDataset(torch.utils.data.Dataset):
    """For each engine, sample cuts. tte = raw RUL at cut time.

    Uses same cut sampling as v20's CMAPSSWindowedDataset.
    """

    def __init__(self, engines, n_cuts_per_engine: int = 5,
                 seed: int = 42, min_past: int = 10):
        self.items = []
        rng = np.random.default_rng(seed)
        for eid, seq in engines.items():
            T = len(seq)
            if T - min_past <= 0:
                continue
            n_cuts = min(n_cuts_per_engine, T - min_past)
            cuts = sorted(rng.integers(min_past, T, size=n_cuts).tolist())
            for t in cuts:
                rul_raw = float(T - t)  # cycles until failure
                past = torch.from_numpy(seq[:t]).float()
                self.items.append((past, rul_raw))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        past, tte = self.items[i]
        return past, torch.tensor(tte, dtype=torch.float32)


class CMAPSSSurfaceTestDataset(torch.utils.data.Dataset):
    """Test set: one item per engine at its last observed cycle.

    tte = test_rul[i] (uncapped RUL at observation time).
    """

    def __init__(self, test_engines, test_rul):
        self.items = []
        for idx, eid in enumerate(sorted(test_engines.keys())):
            seq = test_engines[eid]
            past = torch.from_numpy(seq).float()
            rul_raw = float(test_rul[idx])
            self.items.append((past, rul_raw))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        past, tte = self.items[i]
        return past, torch.tensor(tte, dtype=torch.float32)


def collate_cmapss_surface(batch):
    """Pad variable-length pasts."""
    pasts, ttes = zip(*batch)
    max_t = max(p.shape[0] for p in pasts)
    B = len(pasts)
    S = pasts[0].shape[1]
    x = torch.zeros(B, max_t, S)
    mask = torch.zeros(B, max_t, dtype=torch.bool)
    for i, p in enumerate(pasts):
        T = p.shape[0]
        x[i, :T] = p
        mask[i, T:] = True
    tte = torch.stack(ttes, dim=0)
    return x, mask, tte, torch.zeros(B, dtype=torch.long)  # t unused for cmapss


# ---------------------------------------------------------------------------
# Label surface construction from tte
# ---------------------------------------------------------------------------

def build_label_surface(tte: torch.Tensor, horizons: torch.Tensor) -> torch.Tensor:
    """y(t, Δt) = 1 if tte(t) <= Δt and tte finite, else 0.

    tte: (B,) float. horizons: (K,) float/long. Returns (B, K) float.
    """
    t = tte.unsqueeze(-1)  # (B, 1)
    h = horizons.to(device=tte.device).float().unsqueeze(0)  # (1, K)
    y = (t <= h) & torch.isfinite(t)
    return y.float()


# ---------------------------------------------------------------------------
# Forward (mode-aware)
# ---------------------------------------------------------------------------

def _forward_logits(model, head, x, mask, horizons, mode: str) -> torch.Tensor:
    """Return per-sample per-horizon logits, shape (B, K)."""
    enc_detach = mode in ('probe_h', 'pred_ft')
    if enc_detach:
        with torch.no_grad():
            h_past = model.encode_past(x, mask)
    else:
        h_past = model.encode_past(x, mask)

    if mode == 'probe_h':
        return head(h_past)  # (B, K)

    if mode == 'pred_ft':
        h_hats = predictor_multi_horizon(model.predictor, h_past, horizons)
    else:  # e2e, scratch
        h_hats = predictor_multi_horizon(model.predictor, h_past, horizons)
    return head(h_hats)  # (B, K)


# ---------------------------------------------------------------------------
# Trainable params
# ---------------------------------------------------------------------------

def get_trainable_params(model, head, mode: str):
    for p in model.parameters():
        p.requires_grad = False
    if hasattr(model, 'target_encoder'):
        for p in model.target_encoder.parameters():
            p.requires_grad = False

    params = list(head.parameters())
    if mode == 'probe_h':
        pass
    elif mode == 'pred_ft':
        for p in model.predictor.parameters():
            p.requires_grad = True
        params += list(model.predictor.parameters())
    elif mode in ('e2e', 'scratch'):
        for p in model.context_encoder.parameters():
            p.requires_grad = True
        for p in model.predictor.parameters():
            p.requires_grad = True
        params += (list(model.context_encoder.parameters())
                   + list(model.predictor.parameters()))
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return params


# ---------------------------------------------------------------------------
# BCE training loop
# ---------------------------------------------------------------------------

def train_bce(model, head, tr_loader, va_loader, mode: str,
              *, lr: float, wd: float, n_epochs: int, patience: int,
              pos_weight: float,
              horizons_eval: Sequence[int] = HORIZONS_STEPS,
              max_train_horizon: int = MAX_TRAIN_HORIZON,
              n_train_horizons: int = 10,
              device: str = 'cuda', verbose: bool = False) -> dict:
    """Train with positive-weighted BCE on a sampled set of horizons per batch.

    Training: each batch draws n_train_horizons integers uniformly from
    [1, max_train_horizon]. Always includes the eval grid to ensure best-val
    is measured on the actual eval distribution (via val loop below).

    Validation: BCE on the eval grid.
    """
    params = get_trainable_params(model, head, mode)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=wd)

    eval_h = torch.tensor(list(horizons_eval), dtype=torch.long, device=device)
    pw_t = torch.tensor(float(pos_weight), device=device)

    best_val = float('inf')
    best_head = None
    best_pred = None
    best_enc = None
    no_impr = 0
    losses = []

    for ep in range(n_epochs):
        model.eval() if mode in ('probe_h', 'pred_ft') else model.train()
        head.train()

        # Sample training horizons for this epoch.
        # probe_h's head is fixed to the eval grid K, so we MUST use exactly
        # the eval grid. Other modes can densify with random horizons.
        if mode == 'probe_h':
            tr_h_list = list(horizons_eval)
        else:
            rng = np.random.default_rng(ep)
            rand_h = rng.integers(1, max_train_horizon + 1,
                                  size=n_train_horizons).tolist()
            tr_h_list = sorted(set(rand_h + list(horizons_eval)))
        tr_h = torch.tensor(tr_h_list, dtype=torch.long, device=device)

        for batch in tr_loader:
            x, mask, tte, _t = batch
            x = x.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            tte = tte.to(device, non_blocking=True)
            opt.zero_grad()
            logits = _forward_logits(model, head, x, mask, tr_h, mode)  # (B, K_tr)
            y = build_label_surface(tte, tr_h.float())
            loss = F.binary_cross_entropy_with_logits(
                logits, y, pos_weight=pw_t, reduction='mean')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()

        # Validation BCE on eval grid
        model.eval(); head.eval()
        vs = 0.0; vn = 0
        with torch.no_grad():
            for batch in va_loader:
                x, mask, tte, _t = batch
                x = x.to(device, non_blocking=True)
                mask = mask.to(device, non_blocking=True)
                tte = tte.to(device, non_blocking=True)
                logits = _forward_logits(model, head, x, mask, eval_h, mode)
                y = build_label_surface(tte, eval_h.float())
                loss = F.binary_cross_entropy_with_logits(
                    logits, y, pos_weight=pw_t, reduction='sum')
                vs += float(loss.item()); vn += int(y.numel())
        val_loss = vs / max(vn, 1)
        losses.append(val_loss)
        if verbose and ep % 5 == 0:
            print(f"    ep{ep:3d} val_bce={val_loss:.5f}", flush=True)

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_head = copy.deepcopy(head.state_dict())
            if mode == 'pred_ft':
                best_pred = copy.deepcopy(model.predictor.state_dict())
            elif mode in ('e2e', 'scratch'):
                best_pred = copy.deepcopy(model.predictor.state_dict())
                best_enc = copy.deepcopy(model.context_encoder.state_dict())
            no_impr = 0
        else:
            no_impr += 1
            if no_impr >= patience:
                break

    # restore best
    if best_head is not None:
        head.load_state_dict(best_head)
    if best_pred is not None:
        model.predictor.load_state_dict(best_pred)
    if best_enc is not None:
        model.context_encoder.load_state_dict(best_enc)
    return {'best_val': best_val, 'final_epoch': ep, 'losses': losses}


# ---------------------------------------------------------------------------
# Evaluation: produce probability + label surfaces
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_surface(model, head, loader, mode: str,
                     horizons: Sequence[int] = HORIZONS_STEPS,
                     device: str = 'cuda') -> dict:
    """Run inference and return probability + label surfaces.

    Returns:
      p_surface: (N, K) float32 in [0, 1]
      y_surface: (N, K) int8 binary
      t_index:   (N,)    long — observation time index (anomaly datasets)
      horizons:  (K,)    long
    """
    model.eval(); head.eval()
    eval_h = torch.tensor(list(horizons), dtype=torch.long, device=device)
    p_list, y_list, t_list = [], [], []
    for batch in loader:
        x, mask, tte, t = batch
        x = x.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        tte = tte.to(device, non_blocking=True)
        logits = _forward_logits(model, head, x, mask, eval_h, mode)
        p = torch.sigmoid(logits)
        y = build_label_surface(tte, eval_h.float())
        p_list.append(p.cpu().numpy().astype(np.float32))
        y_list.append(y.cpu().numpy().astype(np.int8))
        t_list.append(t.cpu().numpy().astype(np.int64))

    return {
        'p_surface': np.concatenate(p_list, axis=0),
        'y_surface': np.concatenate(y_list, axis=0),
        't_index':   np.concatenate(t_list, axis=0),
        'horizons':  np.asarray(list(horizons), dtype=np.int32),
    }


# ---------------------------------------------------------------------------
# pos_weight estimation from the train loader (one pass)
# ---------------------------------------------------------------------------

def estimate_pos_weight(loader, horizons: Sequence[int] = HORIZONS_STEPS,
                        clamp_min: float = 1.0,
                        clamp_max: float = 1000.0) -> float:
    """Scan loader, build label surface, return N_neg/N_pos clamped."""
    n_pos, n_tot = 0, 0
    h = torch.tensor(list(horizons), dtype=torch.float32)
    for batch in loader:
        _x, _m, tte, _t = batch
        y = build_label_surface(tte, h)  # (B, K)
        n_pos += int(y.sum().item())
        n_tot += int(y.numel())
    n_neg = max(n_tot - n_pos, 0)
    if n_pos == 0:
        return float(clamp_max)
    pw = n_neg / max(n_pos, 1)
    return float(max(clamp_min, min(clamp_max, pw)))


# ---------------------------------------------------------------------------
# Surface storage
# ---------------------------------------------------------------------------

def save_surface(path, p_surface, y_surface, horizons, t_index,
                 metadata: dict | None = None):
    """Save probability surface to .npz for later metric recomputation."""
    meta = {} if metadata is None else dict(metadata)
    np.savez(path,
             p_surface=np.asarray(p_surface, dtype=np.float32),
             y_surface=np.asarray(y_surface, dtype=np.int8),
             horizons=np.asarray(horizons, dtype=np.int32),
             t_index=np.asarray(t_index, dtype=np.int64),
             meta=np.asarray(list(meta.items()), dtype=object)
             if meta else np.array([], dtype=object))


def load_surface(path) -> dict:
    d = np.load(path, allow_pickle=True)
    meta = {}
    if 'meta' in d and d['meta'].size > 0:
        meta = {str(k): v for k, v in d['meta']}
    return {
        'p_surface': d['p_surface'],
        'y_surface': d['y_surface'],
        'horizons': d['horizons'],
        't_index': d['t_index'],
        'meta': meta,
    }
