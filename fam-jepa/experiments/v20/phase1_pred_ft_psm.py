"""V20 Phase 1b: Pred-FT on PSM (non-C-MAPSS demonstration).

Paper success criterion: pred-FT on at least one non-C-MAPSS dataset.

Uses the v19 PSM-pretrained encoder (3 seeds already trained).
Splits the labeled PSM test set into pred-FT-train / pred-FT-val /
pred-FT-test (chronological 60/10/30). Trains a per-window binary head
under pred-FT mode (freeze encoder, finetune predictor + head) to predict
future anomaly activity at horizons k=1..W.

For comparison also runs frozen_multi (freeze enc + pred) and probe_h.

Labels per cut at time t: y_w = any(labels[t+w]) (patch size = 1 timestep
for PSM; W=16 windows = 16 steps ahead).
"""
import sys, json, copy, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

ROOT = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa')
V11 = ROOT / 'experiments' / 'v11'
V19 = ROOT / 'experiments' / 'v19'
V20 = ROOT / 'experiments' / 'v20'
sys.path.insert(0, str(V11)); sys.path.insert(0, str(V20)); sys.path.insert(0, str(ROOT))

from models import TrajectoryJEPA
from pred_ft_utils import predictor_multi_horizon

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

CONTEXT = 100                                  # encoder input length
N_WINDOWS = 16                                 # horizons k=1..16
TRAIN_STRIDE = 10                              # stride for pred-FT train cuts
EVAL_STRIDE = 1                                # stride for pred-FT test (dense)
BATCH = 64
MAX_EPOCHS = {'probe_h': 30, 'frozen_multi': 30, 'pred_ft': 30}
PATIENCE = {'probe_h': 8, 'frozen_multi': 8, 'pred_ft': 8}
LR = {'probe_h': 1e-3, 'frozen_multi': 1e-3, 'pred_ft': 5e-4}
WD = 1e-2

ARCH = dict(patch_length=1, d_model=256, n_heads=4, n_layers=2, d_ff=1024,
            dropout=0.1, ema_momentum=0.99, predictor_hidden=1024)

SEEDS = [42, 123, 456]

PSM_DIR = Path('/home/sagemaker-user/IndustrialJEPA/paper-replications/mts-jepa/data/PSM')


def load_psm():
    tr = np.load(PSM_DIR / 'train.npy').astype(np.float32)
    te = np.load(PSM_DIR / 'test.npy').astype(np.float32)
    lb = np.load(PSM_DIR / 'test_labels.npy').astype(np.int32)
    n = min(len(te), len(lb))
    te = te[:n]; lb = lb[:n]
    return tr, te, lb


class PSMWindowedDataset(Dataset):
    """Produces (x_past, y_windows) pairs from PSM test series.

    x_past: (CONTEXT, C) ending at time t
    y_windows: (W,) binary labels: y_w = labels[t + w - 1]
    """

    def __init__(self, series, labels, t_starts, context=CONTEXT,
                 n_windows=N_WINDOWS):
        self.series = series
        self.labels = labels
        self.t_starts = t_starts        # list/array of context START indices
        self.context = context
        self.n_windows = n_windows

    def __len__(self):
        return len(self.t_starts)

    def __getitem__(self, i):
        s = int(self.t_starts[i])
        e = s + self.context                # exclusive
        x = self.series[s:e]                 # (C, C)
        y = self.labels[e: e + self.n_windows]  # W labels
        if len(y) < self.n_windows:
            y = np.concatenate([y, np.zeros(self.n_windows - len(y), dtype=np.int32)])
        return (torch.from_numpy(x).float(),
                torch.from_numpy(y).float())


def build_splits(test_len, context=CONTEXT, n_windows=N_WINDOWS,
                 train_stride=TRAIN_STRIDE, eval_stride=EVAL_STRIDE,
                 train_frac=0.6, val_frac=0.1):
    """Chronological 60/10/30 split of test timeline into pred-FT train/val/test.

    Returns dict of t_start arrays (context start indices).
    """
    valid_last_start = test_len - context - n_windows
    boundary_tv = int(train_frac * valid_last_start)
    boundary_vte = int((train_frac + val_frac) * valid_last_start)

    train_starts = np.arange(0, boundary_tv, train_stride)
    val_starts = np.arange(boundary_tv, boundary_vte, train_stride)
    test_starts = np.arange(boundary_vte, valid_last_start, eval_stride)
    return train_starts, val_starts, test_starts


class MultiHorizonHeadPerWindow(nn.Module):
    """Per-window binary head. W outputs (vs scalar in Phase 0).

    Shared across windows: single linear(d -> 1) applied to each h_hat_w.
    Params = d + 1 = 257.
    """
    def __init__(self, d_model):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.lin = nn.Linear(d_model, 1)

    def forward(self, h_past, h_hats):
        # h_hats: (B, W, d)
        x = self.norm(h_hats)
        return self.lin(x).squeeze(-1)           # (B, W) logits


class ProbeHPerWindow(nn.Module):
    """Per-window binary probe on h_past only. W outputs.
    head = linear(d -> W). Params = d*W + W = 256*16+16 = 4112.
    """
    def __init__(self, d_model, n_windows=N_WINDOWS):
        super().__init__()
        self.lin = nn.Linear(d_model, n_windows)

    def forward(self, h_past, h_hats=None):
        return self.lin(h_past)                  # (B, W) logits


def forward_anomaly(model, head, x, mode, n_windows=N_WINDOWS):
    mask = torch.zeros(x.shape[0], x.shape[1], dtype=torch.bool, device=x.device)
    enc_detach = mode in ('probe_h', 'frozen_multi', 'pred_ft')
    if enc_detach:
        with torch.no_grad():
            h_past = model.encode_past(x, mask)
    else:
        h_past = model.encode_past(x, mask)
    if mode == 'probe_h':
        return head(h_past, None)
    pred_detach = mode == 'frozen_multi'
    if pred_detach:
        with torch.no_grad():
            h_hats = predictor_multi_horizon(model.predictor, h_past, n_windows)
    else:
        h_hats = predictor_multi_horizon(model.predictor, h_past, n_windows)
    return head(h_past, h_hats)


def load_v19_psm_ckpt(seed, n_channels):
    ckpt_path = V19 / 'ckpts' / f'v19_psm_seed{seed}_ep50.pt'
    model = TrajectoryJEPA(n_sensors=n_channels, **ARCH).to(DEVICE)
    sd = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(sd)
    return model


def configure_grads(model, head, mode):
    for p in model.parameters():
        p.requires_grad = False
    for p in model.target_encoder.parameters():
        p.requires_grad = False
    params = list(head.parameters())
    if mode == 'pred_ft':
        for p in model.predictor.parameters():
            p.requires_grad = True
        params += list(model.predictor.parameters())
    return params


def per_window_f1(logits, y):
    logits = np.asarray(logits); y = np.asarray(y).astype(int)
    preds = (logits > 0).astype(int)
    W = y.shape[1]
    f1s, precs, recs, aurocs, per_w = [], [], [], [], []
    from sklearn.metrics import roc_auc_score
    for w in range(W):
        yt = y[:, w]; yp = preds[:, w]; sc = logits[:, w]
        tp = int(((yp == 1) & (yt == 1)).sum())
        fp = int(((yp == 1) & (yt == 0)).sum())
        fn = int(((yp == 0) & (yt == 1)).sum())
        denom = 2 * tp + fp + fn
        f1 = 2 * tp / denom if denom > 0 else 0.0
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        try:
            auc = float(roc_auc_score(yt, sc)) if 0 < yt.sum() < len(yt) else float('nan')
        except Exception:
            auc = float('nan')
        per_w.append({'w': w + 1, 'f1': f1, 'precision': p, 'recall': r,
                      'auroc': auc, 'n_pos': int(yt.sum())})
        f1s.append(f1); precs.append(p); recs.append(r)
        if not np.isnan(auc):
            aurocs.append(auc)
    return {
        'f1_mean': float(np.mean(f1s)),
        'precision_mean': float(np.mean(precs)),
        'recall_mean': float(np.mean(recs)),
        'auroc_mean': float(np.mean(aurocs)) if aurocs else float('nan'),
        'per_window': per_w,
    }


def run_one(series, labels, seed, mode):
    torch.manual_seed(seed); np.random.seed(seed)
    n_channels = series.shape[1] if series.ndim == 2 else 1
    model = load_v19_psm_ckpt(seed, n_channels)

    if mode == 'probe_h':
        head = ProbeHPerWindow(ARCH['d_model'], N_WINDOWS).to(DEVICE)
    else:
        head = MultiHorizonHeadPerWindow(ARCH['d_model']).to(DEVICE)

    # The encoder was trained on train.npy; we use test.npy (labeled) for FT
    tr_starts, va_starts, te_starts = build_splits(len(series))

    tr_ds = PSMWindowedDataset(series, labels, tr_starts)
    va_ds = PSMWindowedDataset(series, labels, va_starts)
    te_ds = PSMWindowedDataset(series, labels, te_starts)

    tr = DataLoader(tr_ds, batch_size=BATCH, shuffle=True, num_workers=0)
    va = DataLoader(va_ds, batch_size=BATCH, shuffle=False, num_workers=0)
    te = DataLoader(te_ds, batch_size=BATCH, shuffle=False, num_workers=0)

    # Weighted BCE: anomaly rate ~0.28 in train slice, upweight positives
    y_train = np.concatenate([labels[s + CONTEXT: s + CONTEXT + N_WINDOWS]
                              for s in tr_starts])
    pos_frac = max(1e-4, float((y_train > 0).mean()))
    pos_weight = torch.tensor(min(10.0, (1 - pos_frac) / pos_frac),
                              device=DEVICE)

    params = configure_grads(model, head, mode)
    opt = torch.optim.AdamW(params, lr=LR[mode], weight_decay=WD)

    best_val = float('inf'); best_head = None; best_pred = None
    no_impr = 0
    for ep in range(MAX_EPOCHS[mode]):
        if mode == 'pred_ft':
            model.eval()                     # frozen encoder
        head.train()
        for x, y in tr:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            logits = forward_anomaly(model, head, x, mode)
            loss = F.binary_cross_entropy_with_logits(logits, y,
                                                      pos_weight=pos_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()

        # Val
        model.eval(); head.eval()
        val_sum = 0.0; n = 0
        with torch.no_grad():
            for x, y in va:
                x, y = x.to(DEVICE), y.to(DEVICE)
                logits = forward_anomaly(model, head, x, mode)
                val_sum += F.binary_cross_entropy_with_logits(
                    logits, y, pos_weight=pos_weight, reduction='sum').item()
                n += y.numel()
        val_loss = val_sum / max(n, 1)

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_head = copy.deepcopy(head.state_dict())
            if mode == 'pred_ft':
                best_pred = copy.deepcopy(model.predictor.state_dict())
            no_impr = 0
        else:
            no_impr += 1
            if no_impr >= PATIENCE[mode]:
                break

    head.load_state_dict(best_head)
    if best_pred is not None:
        model.predictor.load_state_dict(best_pred)

    # Evaluate
    model.eval(); head.eval()
    all_logits, all_y = [], []
    with torch.no_grad():
        for x, y in te:
            x = x.to(DEVICE)
            logits = forward_anomaly(model, head, x, mode)
            all_logits.append(logits.cpu().numpy())
            all_y.append(y.numpy())
    all_logits = np.concatenate(all_logits)
    all_y = np.concatenate(all_y)

    m = per_window_f1(all_logits, all_y)
    m['val_loss'] = float(best_val)
    m['final_epoch'] = int(ep)
    m['mode'] = mode
    m['seed'] = int(seed)
    m['n_train'] = int(len(tr_ds))
    m['n_val'] = int(len(va_ds))
    m['n_test'] = int(len(te_ds))
    m['pos_weight'] = float(pos_weight.item())
    # Global non-per-window metrics (flatten)
    logits_flat = all_logits.reshape(-1)
    y_flat = all_y.reshape(-1).astype(int)
    try:
        from sklearn.metrics import roc_auc_score, f1_score
        m['global_auroc'] = float(roc_auc_score(y_flat, logits_flat))
        m['global_f1'] = float(f1_score(y_flat, (logits_flat > 0).astype(int)))
    except Exception:
        m['global_auroc'] = float('nan')
        m['global_f1'] = float('nan')
    return m


def main():
    V20.mkdir(exist_ok=True)
    out_path = V20 / 'phase1_pred_ft_psm.json'
    print("=" * 80)
    print("V20 Phase 1b: Pred-FT on PSM")
    print(f"  3 modes x 3 seeds = 9 runs, v19 PSM ckpts (seed 42, 123, 456)")
    print("=" * 80, flush=True)

    train_series, test_series, test_labels = load_psm()
    print(f"PSM test: {test_series.shape}, anomaly rate: {test_labels.mean():.3f}",
          flush=True)
    # Drop near-constant channels like v19 did (align with pretrained model)
    # v19 trained on whatever channels were in train.npy after preprocessing.
    # Load checks: v19_psm expects 25 channels. We use train as-is.

    t0 = time.time()
    results = {}
    for mode in ['probe_h', 'frozen_multi', 'pred_ft']:
        results[mode] = []
        for seed in SEEDS:
            t1 = time.time()
            r = run_one(test_series, test_labels, seed, mode)
            dt = time.time() - t1
            results[mode].append(r)
            print(f"  [{mode:14s} s={seed}] "
                  f"val={r['val_loss']:.4f} F1w={r['f1_mean']:.3f} "
                  f"AUROCw={r['auroc_mean']:.3f} P={r['precision_mean']:.3f} "
                  f"R={r['recall_mean']:.3f} globalF1={r['global_f1']:.3f} "
                  f"ep={r['final_epoch']} ({dt:.0f}s)", flush=True)

            with open(out_path, 'w') as f:
                json.dump({'config': 'v20_phase1_pred_ft_psm',
                           'seeds': SEEDS, 'runtime_min': (time.time()-t0)/60,
                           'results': results}, f, indent=2, default=float)

    print("\n" + "=" * 80)
    print("V20 Phase 1b Summary")
    print("=" * 80)
    for mode, rs in results.items():
        f1s = [r['f1_mean'] for r in rs]
        aurocs = [r['auroc_mean'] for r in rs]
        gf1s = [r['global_f1'] for r in rs]
        print(f"  {mode:14s}: F1w {np.mean(f1s):.3f}±{np.std(f1s):.3f}  "
              f"AUROCw {np.mean(aurocs):.3f}  global F1 {np.mean(gf1s):.3f}")
    print(f"Runtime: {(time.time()-t0)/60:.1f} min")


if __name__ == '__main__':
    main()
