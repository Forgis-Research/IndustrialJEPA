"""V20 Predictor Finetuning Utilities.

Core infrastructure for the paper's main contribution:
  Freeze the pretrained JEPA encoder, finetune only the horizon-conditioned
  predictor + a linear head, and evaluate with per-window binary F1.

Five downstream modes, shared scalar-RUL regression objective:
  probe_h        : freeze enc+pred.  head = Linear(d -> 1) on h_past        (~257 p)
  frozen_multi   : freeze enc+pred.  head = Linear(16d -> 1) on concat h_hat(~4K p)
  pred_ft        : freeze enc.       unfreeze pred + Linear(16d -> 1)       (~198K p)
  e2e            : unfreeze enc+pred + Linear(16d -> 1)                     (~1.26M p)
  scratch        : random init everything + Linear(16d -> 1)                (~1.26M p)

RUL is sigmoid-normalized to [0,1] by rul_cap for the loss. Predictions are
un-normalized before metric computation.

Per-window binary F1 is derived from scalar RUL predictions by thresholding
at each window boundary: y_w_true = (RUL_true <= w), y_w_pred = (RUL_pred <= w).
"""

from __future__ import annotations
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Predictor multi-horizon call (vectorized)
# ---------------------------------------------------------------------------

def predictor_multi_horizon(predictor: nn.Module,
                            h_past: torch.Tensor,
                            n_windows: int = 16) -> torch.Tensor:
    """Run predictor at horizons k=1..n_windows in one vectorized forward.

    h_past: (B, d)  ->  returns (B, W, d)
    """
    B, d = h_past.shape
    W = n_windows
    h = h_past.unsqueeze(1).expand(B, W, d).reshape(B * W, d)
    k = (torch.arange(1, W + 1, device=h_past.device, dtype=torch.float32)
         .unsqueeze(0).expand(B, W).reshape(-1))
    out = predictor(h, k)                           # (B*W, d)
    return out.view(B, W, d)


# ---------------------------------------------------------------------------
# Heads
# ---------------------------------------------------------------------------

class ProbeH(nn.Module):
    """Scalar RUL probe on h_past only. Params = d_model + 1."""

    def __init__(self, d_model: int):
        super().__init__()
        self.lin = nn.Linear(d_model, 1)

    def forward(self, h_past: torch.Tensor,
                h_hats: torch.Tensor | None = None) -> torch.Tensor:
        return torch.sigmoid(self.lin(h_past)).squeeze(-1)  # (B,)


class MultiHorizonHead(nn.Module):
    """Scalar RUL head on concatenated h_hat_1..h_hat_W. Params ~= W*d + 1.

    LayerNorm applied per-horizon before concatenation. The predictor outputs
    in this architecture have std ~5 (no trailing norm) which would saturate
    a downstream sigmoid, so we normalize first.
    """

    def __init__(self, d_model: int, n_windows: int = 16):
        super().__init__()
        self.n_windows = n_windows
        self.norm = nn.LayerNorm(d_model)
        self.lin = nn.Linear(d_model * n_windows, 1)

    def forward(self, h_past: torch.Tensor,
                h_hats: torch.Tensor) -> torch.Tensor:
        # h_hats: (B, W, d)
        x = self.norm(h_hats)                        # (B, W, d)
        B = x.shape[0]
        x = x.reshape(B, -1)                         # (B, W*d)
        return torch.sigmoid(self.lin(x)).squeeze(-1)  # (B,)


# ---------------------------------------------------------------------------
# C-MAPSS windowed dataset
# ---------------------------------------------------------------------------

class CMAPSSWindowedDataset(torch.utils.data.Dataset):
    """For each engine, sample cut points. Label = capped RUL at cut time.

    Uses raw (uncapped) RUL for event-in-window labels.
    """

    def __init__(self, engines, n_cuts_per_engine: int = 5,
                 rul_cap: int = 125, seed: int = 42,
                 min_past: int = 10):
        self.rul_cap = rul_cap
        self.items = []
        rng = np.random.default_rng(seed)
        for eid, seq in engines.items():
            T = len(seq)
            if T - min_past <= 0:
                continue
            n_cuts = min(n_cuts_per_engine, T - min_past)
            cuts = sorted(rng.integers(min_past, T, size=n_cuts).tolist())
            for t in cuts:
                rul_raw = float(T - t)              # uncapped
                rul_norm = min(rul_raw, rul_cap) / rul_cap
                past = torch.from_numpy(seq[:t]).float()
                self.items.append((past, rul_norm, rul_raw))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        past, rul_norm, rul_raw = self.items[i]
        return past, torch.tensor(rul_norm, dtype=torch.float32), \
               torch.tensor(rul_raw, dtype=torch.float32)


class CMAPSSTestWindowedDataset(torch.utils.data.Dataset):
    """Test set: one item per engine at its last observed cycle.

    test_rul[i] is the uncapped RUL at that cycle.
    """

    def __init__(self, test_engines, test_rul, rul_cap: int = 125):
        self.items = []
        for idx, eid in enumerate(sorted(test_engines.keys())):
            seq = test_engines[eid]
            past = torch.from_numpy(seq).float()
            rul_raw = float(test_rul[idx])
            rul_norm = min(rul_raw, rul_cap) / rul_cap
            self.items.append((past, rul_norm, rul_raw))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        past, rul_norm, rul_raw = self.items[i]
        return past, torch.tensor(rul_norm, dtype=torch.float32), \
               torch.tensor(rul_raw, dtype=torch.float32)


def collate_windowed(batch):
    """Pad variable-length pasts."""
    past_list, rul_norm_list, rul_raw_list = zip(*batch)
    max_t = max(p.shape[0] for p in past_list)
    B = len(past_list)
    S = past_list[0].shape[1]
    past = torch.zeros(B, max_t, S)
    mask = torch.zeros(B, max_t, dtype=torch.bool)
    for i, p in enumerate(past_list):
        T = p.shape[0]
        past[i, :T] = p
        mask[i, T:] = True
    rul_norm = torch.stack(rul_norm_list)
    rul_raw = torch.stack(rul_raw_list)
    return past, mask, rul_norm, rul_raw


# ---------------------------------------------------------------------------
# Per-window F1 from scalar RUL
# ---------------------------------------------------------------------------

def per_window_metrics_from_rul(pred_rul: np.ndarray, true_rul: np.ndarray,
                                n_windows: int = 16) -> dict:
    """Derive per-window binary F1 from scalar RUL predictions.

    y_w_true = (true_rul <= w), y_w_pred = (pred_rul <= w), for w=1..W.

    Returns macro F1 across W (averaged), plus per-window breakdown.
    AUROC per window uses -pred_rul as the score (lower RUL -> higher prob event).
    """
    pred_rul = np.asarray(pred_rul, dtype=float)
    true_rul = np.asarray(true_rul, dtype=float)

    from sklearn.metrics import roc_auc_score

    per_w = []
    f1s, precs, recs, aurocs = [], [], [], []
    for w in range(1, n_windows + 1):
        yt = (true_rul <= w).astype(int)
        yp = (pred_rul <= w).astype(int)
        tp = int(((yp == 1) & (yt == 1)).sum())
        fp = int(((yp == 1) & (yt == 0)).sum())
        fn = int(((yp == 0) & (yt == 1)).sum())
        denom = 2 * tp + fp + fn
        f1 = 2 * tp / denom if denom > 0 else 0.0
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if 0 < yt.sum() < len(yt):
            try:
                auc = float(roc_auc_score(yt, -pred_rul))
            except Exception:
                auc = float('nan')
        else:
            auc = float('nan')
        per_w.append({'w': w, 'f1': f1, 'precision': p, 'recall': r,
                      'auroc': auc, 'n_pos': int(yt.sum())})
        f1s.append(f1); precs.append(p); recs.append(r)
        if not np.isnan(auc):
            aurocs.append(auc)

    return {
        'f1_mean': float(np.mean(f1s)),
        'precision_mean': float(np.mean(precs)),
        'recall_mean': float(np.mean(recs)),
        'auroc_mean': float(np.mean(aurocs)) if aurocs else float('nan'),
        'n_windows': n_windows,
        'per_window': per_w,
    }


def rul_metrics(pred_rul: np.ndarray, true_rul: np.ndarray) -> dict:
    """Legacy RMSE + NASA-S for C-MAPSS."""
    err = pred_rul - true_rul
    rmse = float(np.sqrt(np.mean(err ** 2)))
    nasa = np.where(err < 0,
                    np.exp(-err / 13.0) - 1.0,
                    np.exp(err / 10.0) - 1.0)
    return {'rmse': rmse, 'nasa_score': float(np.sum(nasa)),
            'mae': float(np.mean(np.abs(err))),
            'bias': float(np.mean(err))}


# ---------------------------------------------------------------------------
# Forward pass (mode-aware)
# ---------------------------------------------------------------------------

def _forward(model, head, past, mask, mode: str, n_windows: int = 16):
    """Run the full forward pass for a given mode and return scalar RUL pred."""
    # Encoder: whether to detach depends on mode
    enc_detach = mode in ('probe_h', 'frozen_multi', 'pred_ft')
    if enc_detach:
        with torch.no_grad():
            h_past = model.encode_past(past, mask)
    else:
        h_past = model.encode_past(past, mask)

    if mode == 'probe_h':
        return head(h_past, None)

    # Predictor call
    pred_detach = mode == 'frozen_multi'
    if pred_detach:
        with torch.no_grad():
            h_hats = predictor_multi_horizon(model.predictor, h_past, n_windows)
    else:
        h_hats = predictor_multi_horizon(model.predictor, h_past, n_windows)
    return head(h_past, h_hats)


# ---------------------------------------------------------------------------
# Parameter selection for optimizer
# ---------------------------------------------------------------------------

def get_trainable_params(model, head, mode: str):
    # Start by freezing everything then unfreeze by mode
    for p in model.parameters():
        p.requires_grad = False
    # target_encoder always frozen
    for p in model.target_encoder.parameters():
        p.requires_grad = False

    params = list(head.parameters())
    if mode in ('probe_h', 'frozen_multi'):
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
# Train / evaluate loops
# ---------------------------------------------------------------------------

def train_one(model, head, train_loader, val_loader, mode: str,
              lr: float, wd: float, n_epochs: int, patience: int,
              n_windows: int = 16, device: str = 'cuda',
              verbose: bool = False):
    """Train head (+ predictor / + encoder) until val loss stops improving."""
    params = get_trainable_params(model, head, mode)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=wd)

    best_val = float('inf')
    best_head = None
    best_pred = None
    best_enc = None
    no_impr = 0
    losses = []

    for ep in range(n_epochs):
        if mode in ('probe_h', 'frozen_multi', 'pred_ft'):
            model.eval()                # no dropout / running stats in enc
        else:
            model.train()
        head.train()
        for past, mask, rul_norm, _ in train_loader:
            past = past.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            rul_norm = rul_norm.to(device, non_blocking=True)
            opt.zero_grad()
            pred = _forward(model, head, past, mask, mode, n_windows)
            loss = F.mse_loss(pred, rul_norm)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()

        # Validation
        model.eval(); head.eval()
        val_sum = 0.0; val_n = 0
        with torch.no_grad():
            for past, mask, rul_norm, _ in val_loader:
                past = past.to(device, non_blocking=True)
                mask = mask.to(device, non_blocking=True)
                rul_norm = rul_norm.to(device, non_blocking=True)
                pred = _forward(model, head, past, mask, mode, n_windows)
                val_sum += F.mse_loss(pred, rul_norm, reduction='sum').item()
                val_n += rul_norm.numel()
        val_loss = val_sum / max(val_n, 1)
        losses.append(val_loss)
        if verbose and ep % 10 == 0:
            print(f"    ep{ep:3d} val_mse={val_loss:.5f}", flush=True)

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


@torch.no_grad()
def evaluate(model, head, test_loader, mode: str,
             n_windows: int = 16, rul_cap: int = 125,
             device: str = 'cuda'):
    """Run test inference, return RUL predictions + all metrics."""
    model.eval(); head.eval()
    preds_norm, truths_norm, truths_raw = [], [], []
    for past, mask, rul_norm, rul_raw in test_loader:
        past = past.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        pred = _forward(model, head, past, mask, mode, n_windows)
        preds_norm.append(pred.cpu().numpy())
        truths_norm.append(rul_norm.numpy())
        truths_raw.append(rul_raw.numpy())
    preds_norm = np.concatenate(preds_norm)
    truths_raw = np.concatenate(truths_raw)
    preds_raw = preds_norm * rul_cap              # un-normalize
    # For legacy RMSE on test: compare capped pred vs capped true
    truths_capped = np.minimum(truths_raw, float(rul_cap))
    legacy = rul_metrics(preds_raw, truths_capped)
    per_win = per_window_metrics_from_rul(preds_raw, truths_raw, n_windows)
    return {
        'preds_raw': preds_raw,
        'truths_raw': truths_raw,
        'legacy': legacy,
        'per_window': per_win,
    }
