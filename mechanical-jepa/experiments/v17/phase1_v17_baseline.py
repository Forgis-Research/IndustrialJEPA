"""
V17 Phase 1: Generalized horizon + fixed-window target baseline.

Changes from V2 (v11):
  - Target: ema_target_encoder(x_{t+k:t+k+w}) with w=10 (fixed)
  - k ~ LogUniform[1, K_max=150] (was U[5,30])
  - Everything else unchanged: causal encoder, 2L, d_model=256, EMA 0.99,
    L1 loss on normalized embeddings, cosine LR from 3e-4, batch=64, 200 epochs

Evaluates:
  - Frozen linear probe on h_past (raw RUL RMSE, RUL_CAP=125)
  - F1 / precision / recall / AUC-PR at k=30 binary ("fails within 30 cycles?")

Saves:
  - experiments/v17/phase1_results.json  (aggregated across seeds)
  - experiments/v17/ckpts/v17_seed{S}_best.pt   (best by probe RMSE)
  - experiments/v17/ckpts/v17_seed{S}_ep100.pt  (mid-training - for Phase 3)
"""

import sys, math, json, copy, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

V11 = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/experiments/v11')
V17 = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/experiments/v17')
sys.path.insert(0, str(V11))

from models import TrajectoryJEPA, RULProbe
from data_utils import (
    load_cmapss_subset, N_SENSORS, RUL_CAP,
    CMAPSSFinetuneDataset, collate_finetune, compute_rul_labels,
)
sys.path.insert(0, '/home/sagemaker-user/IndustrialJEPA/mechanical-jepa')
from evaluation.grey_swan_metrics import anomaly_metrics as _anomaly_metrics

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Config ---
D_MODEL = 256
N_HEADS = 4
N_LAYERS = 2
D_FF = 4 * D_MODEL
BATCH_SIZE = 64
N_EPOCHS = 200
LR = 3e-4
WEIGHT_DECAY = 0.01
EMA_MOMENTUM = 0.99
K_MAX = 150
W_WIN = 10
MIN_PAST = 10
N_CUTS = 30
PROBE_EVERY = 10
PROBE_LR = 1e-3
PROBE_EPOCHS = 100
PROBE_PATIENCE = 15
SEEDS = [42, 123, 456]
CKPT_DIR = V17 / 'ckpts'
CKPT_DIR.mkdir(parents=True, exist_ok=True)

# F1 target horizon (cycles). Binary: RUL <= k_eval means "fail within k cycles"
K_EVAL_F1 = 30


# ============================================================
# V17 dataset: LogUniform k, fixed-window target
# ============================================================

class V17PretrainDataset(Dataset):
    """(past=x_{:t}, target=x_{t+k:t+k+w}, k)"""

    def __init__(self, engines, n_cuts=N_CUTS, min_past=MIN_PAST,
                 K_max=K_MAX, w=W_WIN, seed=42):
        self.engines = engines
        self.w = w
        rng = np.random.RandomState(seed)
        self.samples = []
        for eid, arr in engines.items():
            T = len(arr)
            # We need: t >= min_past, k >= 1, t+k+w <= T
            # => t+1+w <= T => t_max = T-1-w
            t_max = T - 1 - w
            if t_max < min_past:
                continue
            for _ in range(n_cuts):
                t = int(rng.randint(min_past, t_max + 1))
                k_hi = min(K_max, T - t - w)  # largest valid k
                if k_hi < 1:
                    continue
                # LogUniform[1, k_hi]
                u = rng.uniform(0.0, math.log(max(2.0, float(k_hi))))
                k = int(math.exp(u))
                k = max(1, min(k, k_hi))
                self.samples.append((eid, t, k))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        eid, t, k = self.samples[idx]
        arr = self.engines[eid]
        past = torch.from_numpy(arr[:t]).float()
        future = torch.from_numpy(arr[t + k: t + k + self.w]).float()
        return past, future, k, t


def collate_v17_pretrain(batch):
    pasts, futures, ks, ts = zip(*batch)
    T_max = max(p.shape[0] for p in pasts)
    B = len(pasts)
    S = pasts[0].shape[1]
    x_past = torch.zeros(B, T_max, S)
    past_mask = torch.ones(B, T_max, dtype=torch.bool)
    for i, p in enumerate(pasts):
        x_past[i, :p.shape[0]] = p
        past_mask[i, :p.shape[0]] = False
    # all futures are length w
    x_fut = torch.stack(list(futures), dim=0)  # (B, w, S)
    fut_mask = torch.zeros(B, x_fut.shape[1], dtype=torch.bool)
    k_t = torch.tensor(ks, dtype=torch.long)
    t_t = torch.tensor(ts, dtype=torch.long)
    return x_past, past_mask, x_fut, fut_mask, k_t, t_t


# ============================================================
# L1 pretraining loss (on L2-normalized embeddings) + var-reg
# ============================================================

def v17_loss(pred, target, lambda_var=0.04):
    pred_n = F.normalize(pred, dim=-1)
    targ_n = F.normalize(target.detach(), dim=-1)
    l_pred = F.l1_loss(pred_n, targ_n)
    # variance penalty on predictions (anti-collapse)
    pred_std = pred.std(dim=0)
    l_var = F.relu(1.0 - pred_std).mean()
    return l_pred + lambda_var * l_var, l_pred, l_var


# ============================================================
# Linear probe (frozen encoder) - standard V2 protocol
# ============================================================

def linear_probe_rmse_and_f1(model, train_engines, val_engines, test_engines,
                              test_rul, seed=42, return_preds=False):
    """
    Train frozen probe on h_past.
    Returns dict with val_rmse, test_rmse, val_f1@k=30, test_f1@k=30, etc.
    """
    torch.manual_seed(seed)
    probe = nn.Sequential(
        nn.Linear(D_MODEL, 1),
        nn.Sigmoid(),
    ).to(DEVICE)
    opt = torch.optim.Adam(probe.parameters(), lr=PROBE_LR)

    tr_ds = CMAPSSFinetuneDataset(train_engines, n_cuts_per_engine=5, seed=seed)
    va_ds = CMAPSSFinetuneDataset(val_engines, use_last_only=True)
    tr = DataLoader(tr_ds, batch_size=32, shuffle=True, collate_fn=collate_finetune)
    va = DataLoader(va_ds, batch_size=32, shuffle=False, collate_fn=collate_finetune)

    best_val = float('inf')
    best_state = None
    no_impr = 0

    model.eval()
    for ep in range(PROBE_EPOCHS):
        probe.train()
        for past, mask, rul in tr:
            past, mask, rul = past.to(DEVICE), mask.to(DEVICE), rul.to(DEVICE)
            with torch.no_grad():
                h = model.encode_past(past, mask)
            p = probe(h).squeeze(-1)
            loss = F.mse_loss(p, rul)
            opt.zero_grad(); loss.backward(); opt.step()

        probe.eval()
        pv, tv = [], []
        with torch.no_grad():
            for past, mask, rul in va:
                past, mask = past.to(DEVICE), mask.to(DEVICE)
                h = model.encode_past(past, mask)
                pv.append(probe(h).squeeze(-1).cpu().numpy())
                tv.append(rul.numpy())
        preds = np.concatenate(pv) * RUL_CAP
        targs = np.concatenate(tv) * RUL_CAP
        val_rmse = float(np.sqrt(np.mean((preds - targs) ** 2)))
        if val_rmse < best_val:
            best_val = val_rmse
            best_state = copy.deepcopy(probe.state_dict())
            no_impr = 0
        else:
            no_impr += 1
            if no_impr >= PROBE_PATIENCE:
                break

    # Restore best
    if best_state is not None:
        probe.load_state_dict(best_state)

    # ---- Compute test RMSE + F1 metrics at k=30 ----
    probe.eval()
    # Build test loader (last cycle of each test engine, raw rul)
    from data_utils import CMAPSSTestDataset, collate_test
    te_ds = CMAPSSTestDataset(test_engines, test_rul)
    te = DataLoader(te_ds, batch_size=32, shuffle=False, collate_fn=collate_test)

    preds_test, targs_test = [], []
    with torch.no_grad():
        for past, mask, rul_gt in te:
            past, mask = past.to(DEVICE), mask.to(DEVICE)
            h = model.encode_past(past, mask)
            p = probe(h).squeeze(-1).cpu().numpy() * RUL_CAP
            preds_test.append(p)
            targs_test.append(rul_gt.numpy())
    preds_test = np.concatenate(preds_test)
    targs_test = np.concatenate(targs_test)
    test_rmse = float(np.sqrt(np.mean((preds_test - targs_test) ** 2)))

    # Binary labels y=1 if RUL <= K_EVAL_F1
    y_test = (targs_test <= K_EVAL_F1).astype(int)
    # Higher "score" = more imminent failure; score = (K_EVAL_F1 - pred)/K_EVAL_F1
    # Use negative predicted RUL so higher = more imminent.
    score_test = -preds_test
    # anomaly_metrics expects scores where higher = positive class
    try:
        threshold = float(np.percentile(score_test[y_test == 0], 95)) \
            if (y_test == 0).sum() > 0 else 0.0
    except Exception:
        threshold = 0.0
    m_test = _anomaly_metrics(score_test, y_test, threshold=threshold)

    # Also on val (for tracking)
    y_val = (targs <= K_EVAL_F1).astype(int)
    score_val = -preds
    try:
        th_v = float(np.percentile(score_val[y_val == 0], 95)) \
            if (y_val == 0).sum() > 0 else 0.0
    except Exception:
        th_v = 0.0
    m_val = _anomaly_metrics(score_val, y_val, threshold=th_v)

    out = dict(
        val_rmse=best_val,
        test_rmse=test_rmse,
        val_f1=m_val['f1_non_pa'],
        test_f1=m_test['f1_non_pa'],
        val_auc_pr=m_val['auc_pr'],
        test_auc_pr=m_test['auc_pr'],
        val_precision=m_val['precision_non_pa'],
        val_recall=m_val['recall_non_pa'],
        test_precision=m_test['precision_non_pa'],
        test_recall=m_test['recall_non_pa'],
        k_eval=K_EVAL_F1,
    )
    if return_preds:
        out['preds_test'] = preds_test.tolist()
        out['targs_test'] = targs_test.tolist()
    return out


# ============================================================
# Main pretraining + eval
# ============================================================

def pretrain_one_seed(seed, data, n_epochs=N_EPOCHS, verbose=True):
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = TrajectoryJEPA(
        n_sensors=N_SENSORS, patch_length=1,
        d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS, d_ff=D_FF,
        dropout=0.1, ema_momentum=EMA_MOMENTUM, predictor_hidden=D_FF,
    ).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[seed {seed}] params={n_params:,}", flush=True)

    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR, weight_decay=WEIGHT_DECAY,
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, n_epochs)

    history = {'loss': [], 'pred': [], 'var': [],
               'probe_rmse': [], 'probe_epochs': [],
               'test_rmse': [], 'test_f1': []}
    best_val = float('inf')
    best_state = None
    t0 = time.time()

    for epoch in range(1, n_epochs + 1):
        ds = V17PretrainDataset(
            data['train_engines'], n_cuts=N_CUTS, min_past=MIN_PAST,
            K_max=K_MAX, w=W_WIN, seed=seed * 1000 + epoch,
        )
        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True,
                            collate_fn=collate_v17_pretrain, num_workers=0)

        model.train()
        tot = pred_l = var_l = 0.0; n = 0
        for x_past, past_mask, x_fut, fut_mask, k, _ in loader:
            x_past, past_mask = x_past.to(DEVICE), past_mask.to(DEVICE)
            x_fut, fut_mask = x_fut.to(DEVICE), fut_mask.to(DEVICE)
            k = k.to(DEVICE)

            optim.zero_grad()
            pred, targ, h_past = model.forward_pretrain(
                x_past, past_mask, x_fut, fut_mask, k,
            )
            loss, lp, lv = v17_loss(pred, targ, lambda_var=0.04)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            model.update_ema()

            B = x_past.shape[0]
            tot += loss.item() * B; pred_l += lp.item() * B
            var_l += lv.item() * B; n += B

        avg_loss = tot / n
        history['loss'].append(avg_loss)
        history['pred'].append(pred_l / n)
        history['var'].append(var_l / n)
        sched.step()

        # Save ckpt at epoch 100 for Phase 3
        if epoch == 100:
            path = CKPT_DIR / f'v17_seed{seed}_ep100.pt'
            torch.save(model.state_dict(), path)
            print(f"  [seed {seed}] saved epoch-100 ckpt -> {path.name}", flush=True)

        extra = ''
        if epoch % PROBE_EVERY == 0 or epoch == 1 or epoch == n_epochs:
            metrics = linear_probe_rmse_and_f1(
                model, data['train_engines'], data['val_engines'],
                data['test_engines'], data['test_rul'], seed=seed,
            )
            history['probe_rmse'].append(metrics['val_rmse'])
            history['probe_epochs'].append(epoch)
            history['test_rmse'].append(metrics['test_rmse'])
            history['test_f1'].append(metrics['test_f1'])

            if metrics['val_rmse'] < best_val:
                best_val = metrics['val_rmse']
                best_state = copy.deepcopy(model.state_dict())
                best_metrics = metrics
                torch.save(best_state, CKPT_DIR / f'v17_seed{seed}_best.pt')
            extra = (f" | probe_val={metrics['val_rmse']:.2f} "
                     f"test={metrics['test_rmse']:.2f} "
                     f"F1@{K_EVAL_F1}={metrics['test_f1']:.3f}")

        if verbose:
            print(f"  Ep {epoch:3d} | L={avg_loss:.4f} "
                  f"pred={pred_l/n:.4f} var={var_l/n:.4f}{extra}", flush=True)

    elapsed = (time.time() - t0) / 60
    print(f"[seed {seed}] done in {elapsed:.1f} min, best probe_val={best_val:.2f}",
          flush=True)

    # Load best state for final metric
    if best_state is not None:
        model.load_state_dict(best_state)
    final_metrics = linear_probe_rmse_and_f1(
        model, data['train_engines'], data['val_engines'],
        data['test_engines'], data['test_rul'], seed=seed,
    )

    return {
        'seed': seed,
        'best_val_rmse': float(best_val),
        'final': final_metrics,
        'history': history,
        'elapsed_min': elapsed,
    }


def main():
    V17.mkdir(exist_ok=True)
    data = load_cmapss_subset('FD001')
    print(f"FD001: train={len(data['train_engines'])} "
          f"val={len(data['val_engines'])} test={len(data['test_engines'])}",
          flush=True)

    all_results = []
    t0 = time.time()
    for seed in SEEDS:
        r = pretrain_one_seed(seed, data, n_epochs=N_EPOCHS)
        all_results.append(r)
        # Save intermediate (for crash safety)
        out = {
            'config': 'v17_baseline',
            'w': W_WIN, 'K_max': K_MAX, 'k_eval_f1': K_EVAL_F1,
            'd_model': D_MODEL, 'n_layers': N_LAYERS, 'n_epochs': N_EPOCHS,
            'batch_size': BATCH_SIZE, 'lr': LR, 'ema_momentum': EMA_MOMENTUM,
            'seeds_done': [rr['seed'] for rr in all_results],
            'per_seed': [{k: v for k, v in rr.items() if k != 'history'}
                         for rr in all_results],
            'v2_baseline_rmse': 17.81,
        }
        with open(V17 / 'phase1_results.json', 'w') as f:
            json.dump(out, f, indent=2, default=float)

    # Aggregate
    rmses_val = [r['best_val_rmse'] for r in all_results]
    rmses_test = [r['final']['test_rmse'] for r in all_results]
    f1s_test = [r['final']['test_f1'] for r in all_results]
    aucpr_test = [r['final']['test_auc_pr'] for r in all_results]

    summary = {
        'config': 'v17_baseline',
        'seeds': SEEDS,
        'w': W_WIN, 'K_max': K_MAX, 'k_eval_f1': K_EVAL_F1,
        'probe_val_rmse_per_seed': rmses_val,
        'probe_val_rmse_mean': float(np.mean(rmses_val)),
        'probe_val_rmse_std': float(np.std(rmses_val)),
        'test_rmse_per_seed': rmses_test,
        'test_rmse_mean': float(np.mean(rmses_test)),
        'test_rmse_std': float(np.std(rmses_test)),
        'f1_at_k30_per_seed': f1s_test,
        'f1_at_k30_mean': float(np.mean(f1s_test)),
        'auc_pr_at_k30_mean': float(np.mean(aucpr_test)),
        'v2_baseline_rmse': 17.81,
        'histories': [r['history'] for r in all_results],
        'runtime_hours': (time.time() - t0) / 3600,
    }
    with open(V17 / 'phase1_results.json', 'w') as f:
        json.dump(summary, f, indent=2, default=float)

    print("\n" + "=" * 60)
    print("V17 Phase 1 SUMMARY")
    print("=" * 60)
    print(f"Seeds                   : {SEEDS}")
    print(f"Probe val RMSE per seed : {rmses_val}")
    print(f"Probe val RMSE mean     : {np.mean(rmses_val):.2f} +/- {np.std(rmses_val):.2f}")
    print(f"Test  RMSE per seed     : {rmses_test}")
    print(f"Test  RMSE mean         : {np.mean(rmses_test):.2f} +/- {np.std(rmses_test):.2f}")
    print(f"Test  F1@k={K_EVAL_F1}          : {np.mean(f1s_test):.3f}")
    print(f"Test  AUC-PR@k={K_EVAL_F1}     : {np.mean(aucpr_test):.3f}")
    print(f"V2 baseline (reference) : 17.81 (n=5)")
    print(f"Runtime                 : {(time.time()-t0)/3600:.2f} h")


if __name__ == '__main__':
    main()
