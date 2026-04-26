"""
V19 Phase 5: Paderborn bearing vibration - real mechanical data, 3-class fault.

K001 (healthy), KA01 (outer race), KI01 (inner race). 80 .mat files each;
each has vibration_1 channel (~256K samples @ 64kHz, ~4s recording).

Protocol:
  1. Load vibration_1 from all 240 files. Window into 1024-sample chunks
     (16ms at 64kHz; covers ~15 shaft rotations at 1500 RPM).
  2. Per-window instance normalisation (standard in bearing literature).
  3. File-level split (not window-level): 60/20/20 train/val/test -> 48/16/16
     files per class. No within-file leakage.
  4. Pretrain FAM on train+val windows, labels unused (pretext learning).
  5. Linear probe for 3-class (K001/KA01/KI01) classification.
  6. Compare to supervised CNN end-to-end baseline.

Output: experiments/v19/phase5_paderborn_results.json
"""

import sys, json, time, gc, copy
from pathlib import Path
import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset

V11 = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/experiments/v11')
V17 = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/experiments/v17')
V19 = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/experiments/v19')
ROOT = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa')
sys.path.insert(0, str(V11)); sys.path.insert(0, str(V17)); sys.path.insert(0, str(ROOT))

from models import TrajectoryJEPA
from phase5_smap_anomaly import (
    SMAPV17Dataset, collate_smap, v17_loss,
    D_MODEL, N_HEADS, N_LAYERS, D_FF, BATCH_SIZE, LR,
    WEIGHT_DECAY, EMA_MOMENTUM, N_SAMPLES,
)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CKPT_DIR = V19 / 'ckpts'
CKPT_DIR.mkdir(parents=True, exist_ok=True)

PADERBORN = Path('/home/sagemaker-user/IndustrialJEPA/datasets/data/paderborn')
CLASSES = ['K001', 'KA01', 'KI01']
WINDOW = 1024
STRIDE_ENCODING = 1024   # non-overlapping
SEEDS = [42, 123, 456]


def load_vibration_file(path):
    m = sio.loadmat(path)
    key = [k for k in m if not k.startswith('_')][0]
    data = m[key][0, 0]
    Y = data['Y']
    vib = None
    for i in range(Y.shape[1]):
        ch = Y[0, i]
        name = ch['Name'][0] if 'Name' in ch.dtype.names else ''
        if name == 'vibration_1':
            vib = ch['Data'][0].astype(np.float32)
            break
    if vib is None:
        raise RuntimeError(f'no vibration_1 in {path}')
    return vib


def build_windowed_dataset(cls, files, window=WINDOW):
    """Returns (X, file_ids) where X is (N, window, 1), file_ids is (N,)."""
    X_list, fid_list = [], []
    for fi, fp in enumerate(files):
        v = load_vibration_file(fp)
        n_win = len(v) // window
        X = v[:n_win * window].reshape(n_win, window, 1)
        # Instance normalise
        mu = X.mean(axis=1, keepdims=True); sd = X.std(axis=1, keepdims=True) + 1e-6
        X = (X - mu) / sd
        X_list.append(X.astype(np.float32))
        fid_list.extend([fi] * n_win)
    X = np.concatenate(X_list, axis=0)
    return X, np.array(fid_list)


def load_paderborn_split(seed=0):
    """File-level 60/20/20 split per class."""
    rng = np.random.RandomState(seed)
    data_by_split = {'train': [], 'val': [], 'test': []}
    labels_by_split = {'train': [], 'val': [], 'test': []}
    for cls_idx, cls in enumerate(CLASSES):
        files = sorted((PADERBORN / cls).glob('*.mat'))
        rng.shuffle(files)
        n = len(files)
        n_tr = int(0.6 * n); n_va = int(0.2 * n)
        splits = {
            'train': files[:n_tr],
            'val':   files[n_tr:n_tr+n_va],
            'test':  files[n_tr+n_va:],
        }
        for split, fs in splits.items():
            X, _ = build_windowed_dataset(cls, fs)
            data_by_split[split].append(X)
            labels_by_split[split].append(np.full(X.shape[0], cls_idx, dtype=np.int64))
    out = {}
    for s in ['train', 'val', 'test']:
        X = np.concatenate(data_by_split[s], axis=0)
        y = np.concatenate(labels_by_split[s], axis=0)
        out[f'X_{s}'] = X
        out[f'y_{s}'] = y
    return out


class PaderbornPretrainDataset(Dataset):
    """Sample (past, future, k, t) tuples from Paderborn windows.

    Each window is a (1024, 1) vibration segment. Past/future split within window.
    """
    def __init__(self, X, n_samples=N_SAMPLES, min_ctx=50, max_ctx=200,
                 K_max=500, w=10, seed=42):
        self.X = X  # (N, 1024, 1)
        self.w = w; self.K_max = K_max
        rng = np.random.RandomState(seed)
        import math
        self.samples = []
        N, T, C = X.shape
        for _ in range(n_samples):
            i = int(rng.randint(0, N))
            ctx = int(rng.randint(min_ctx, max_ctx + 1))
            k = int(math.exp(rng.uniform(0.0, math.log(K_max))))
            k = max(1, min(k, K_max))
            t_min = ctx
            t_max = T - k - w - 1
            if t_max <= t_min: continue
            t = int(rng.randint(t_min, t_max + 1))
            self.samples.append((i, t, ctx, k))

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        i, t, ctx, k = self.samples[idx]
        past = self.X[i, t-ctx:t]         # (ctx, 1)
        future = self.X[i, t+k:t+k+self.w]  # (w, 1)
        return (torch.from_numpy(past).float(),
                torch.from_numpy(future).float(), k)


def collate_pad(batch):
    pasts, futures, ks = zip(*batch)
    T_max = max(p.shape[0] for p in pasts)
    B = len(pasts); C = pasts[0].shape[1]
    x_past = torch.zeros(B, T_max, C)
    past_mask = torch.ones(B, T_max, dtype=torch.bool)
    for i, p in enumerate(pasts):
        x_past[i, :p.shape[0]] = p
        past_mask[i, :p.shape[0]] = False
    x_fut = torch.stack(list(futures), dim=0)
    fut_mask = torch.zeros(B, x_fut.shape[1], dtype=torch.bool)
    return x_past, past_mask, x_fut, fut_mask, torch.tensor(ks, dtype=torch.long)


def pretrain(split, seed, ckpt_path, n_epochs=50):
    torch.manual_seed(seed); np.random.seed(seed)
    X_pretrain = np.concatenate([split['X_train'], split['X_val']], axis=0)
    C = X_pretrain.shape[-1]
    model = TrajectoryJEPA(
        n_sensors=C, patch_length=1,
        d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS, d_ff=D_FF,
        dropout=0.1, ema_momentum=EMA_MOMENTUM, predictor_hidden=D_FF,
    ).to(DEVICE)
    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR, weight_decay=WEIGHT_DECAY,
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, n_epochs)
    t0 = time.time()
    for epoch in range(1, n_epochs + 1):
        ds = PaderbornPretrainDataset(X_pretrain, n_samples=N_SAMPLES,
                                        seed=seed * 1000 + epoch)
        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True,
                            collate_fn=collate_pad)
        model.train()
        tot = 0.0; n = 0
        for x_past, past_mask, x_fut, fut_mask, k in loader:
            x_past, past_mask = x_past.to(DEVICE), past_mask.to(DEVICE)
            x_fut, fut_mask = x_fut.to(DEVICE), fut_mask.to(DEVICE)
            k = k.to(DEVICE)
            optim.zero_grad()
            pred, targ, _ = model.forward_pretrain(x_past, past_mask, x_fut, fut_mask, k)
            loss, _, _ = v17_loss(pred, targ, lambda_var=0.04)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step(); model.update_ema()
            B = x_past.shape[0]; tot += loss.item() * B; n += B
        sched.step()
        if epoch % 10 == 0 or epoch in (1, n_epochs):
            print(f"  ep {epoch:3d} | L={tot/n:.4f}", flush=True)
    print(f"  pretrain done in {(time.time()-t0)/60:.1f} min", flush=True)
    torch.save(model.state_dict(), ckpt_path)
    return model


@torch.no_grad()
def encode_windows_direct(model, X):
    """Given X of shape (N, 1024, 1), encode each full window -> (N, D)."""
    N = X.shape[0]
    H = []
    batch = 256
    for i in range(0, N, batch):
        x = torch.from_numpy(X[i:i+batch]).float().to(DEVICE)
        mask = torch.zeros(x.shape[0], x.shape[1], dtype=torch.bool, device=DEVICE)
        h = model.encode_past(x, mask)
        H.append(h.cpu().numpy())
    return np.concatenate(H, axis=0)


def linear_probe_3class(X_tr, y_tr, X_va, y_va, X_te, y_te, seed):
    torch.manual_seed(seed)
    D = X_tr.shape[1]
    probe = nn.Linear(D, 3).to(DEVICE)
    opt = torch.optim.AdamW(probe.parameters(), lr=1e-3, weight_decay=1e-2)
    X_tr_t = torch.from_numpy(X_tr).float().to(DEVICE)
    y_tr_t = torch.from_numpy(y_tr).long().to(DEVICE)
    X_va_t = torch.from_numpy(X_va).float().to(DEVICE)
    y_va_t = torch.from_numpy(y_va).long().to(DEVICE)
    X_te_t = torch.from_numpy(X_te).float().to(DEVICE)

    best_va = 0.0; best = None; no_impr = 0
    for ep in range(200):
        probe.train()
        idx = torch.randperm(X_tr_t.shape[0])
        for i in range(0, X_tr_t.shape[0], 256):
            b = idx[i:i+256]
            loss = F.cross_entropy(probe(X_tr_t[b]), y_tr_t[b])
            opt.zero_grad(); loss.backward(); opt.step()
        probe.eval()
        with torch.no_grad():
            acc_va = (probe(X_va_t).argmax(-1) == y_va_t).float().mean().item()
        if acc_va > best_va:
            best_va = acc_va; best = copy.deepcopy(probe.state_dict()); no_impr = 0
        else:
            no_impr += 1
            if no_impr >= 25: break
    probe.load_state_dict(best); probe.eval()
    with torch.no_grad():
        pred_te = probe(X_te_t).argmax(-1).cpu().numpy()
    acc_te = float((pred_te == y_te).mean())
    # macro F1
    from sklearn.metrics import f1_score, confusion_matrix
    macro_f1 = float(f1_score(y_te, pred_te, average='macro'))
    per_cls_f1 = f1_score(y_te, pred_te, average=None).tolist()
    cm = confusion_matrix(y_te, pred_te).tolist()
    return {'val_acc': best_va, 'test_acc': acc_te, 'macro_f1': macro_f1,
            'per_class_f1': per_cls_f1, 'confusion_matrix': cm}


def run_seed(seed):
    print(f"\n=== Paderborn seed {seed} ===", flush=True)
    split = load_paderborn_split(seed)
    print(f"  train X: {split['X_train'].shape}  y distr: "
          f"{np.bincount(split['y_train'])}", flush=True)
    print(f"  val   X: {split['X_val'].shape}  y distr: "
          f"{np.bincount(split['y_val'])}", flush=True)
    print(f"  test  X: {split['X_test'].shape}  y distr: "
          f"{np.bincount(split['y_test'])}", flush=True)

    ckpt = CKPT_DIR / f'v19_paderborn_seed{seed}.pt'
    if ckpt.exists():
        print(f"  ckpt exists, loading", flush=True)
        model = TrajectoryJEPA(
            n_sensors=1, patch_length=1,
            d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS, d_ff=D_FF,
            dropout=0.1, ema_momentum=EMA_MOMENTUM, predictor_hidden=D_FF,
        ).to(DEVICE)
        model.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=False))
    else:
        model = pretrain(split, seed, ckpt, n_epochs=50)
    model.eval()
    for p in model.parameters(): p.requires_grad = False

    # Encode all windows -> embeddings
    H_tr = encode_windows_direct(model, split['X_train'])
    H_va = encode_windows_direct(model, split['X_val'])
    H_te = encode_windows_direct(model, split['X_test'])
    print(f"  H shapes: {H_tr.shape}, {H_va.shape}, {H_te.shape}", flush=True)

    result = linear_probe_3class(H_tr, split['y_train'], H_va, split['y_val'],
                                   H_te, split['y_test'], seed)
    result['seed'] = seed
    print(f"  test acc: {result['test_acc']:.3f}  macro-F1: {result['macro_f1']:.3f}",
          flush=True)
    print(f"  per-class F1: {result['per_class_f1']}", flush=True)
    print(f"  confusion:\n{np.array(result['confusion_matrix'])}", flush=True)

    del model; gc.collect(); torch.cuda.empty_cache()
    return result


def main():
    t0 = time.time()
    results = []
    for seed in SEEDS:
        r = run_seed(seed)
        results.append(r)
        with open(V19/'phase5_paderborn_results.json', 'w') as f:
            json.dump({'config': 'v19_phase5_paderborn',
                       'classes': CLASSES, 'results': results,
                       'runtime_min': (time.time()-t0)/60}, f, indent=2, default=float)

    agg = {
        'test_acc_mean': float(np.mean([r['test_acc'] for r in results])),
        'test_acc_std': float(np.std([r['test_acc'] for r in results])),
        'macro_f1_mean': float(np.mean([r['macro_f1'] for r in results])),
        'macro_f1_std': float(np.std([r['macro_f1'] for r in results])),
    }
    summary = {'config': 'v19_phase5_paderborn',
               'classes': CLASSES, 'results': results,
               'aggregate': agg,
               'runtime_min': (time.time()-t0)/60}
    with open(V19/'phase5_paderborn_results.json', 'w') as f:
        json.dump(summary, f, indent=2, default=float)

    print("\n" + "=" * 60)
    print("V19 Phase 5: Paderborn 3-class bearing fault classification")
    print("=" * 60)
    print(f"  test acc:   {agg['test_acc_mean']:.3f} +/- {agg['test_acc_std']:.3f}")
    print(f"  macro F1:   {agg['macro_f1_mean']:.3f} +/- {agg['macro_f1_std']:.3f}")
    print(f"  runtime:    {(time.time()-t0)/60:.1f} min")


if __name__ == '__main__':
    main()
