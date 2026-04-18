"""
V17 Phase 2: Trajectory probing.

Using Phase 1 best checkpoints (encoder + predictor FROZEN), train linear probes on:
  A) h_past only                    (256-dim, matches V2/Phase 1 baseline)
  B) concat(h, gamma(5), gamma(10), gamma(20), gamma(50), gamma(100))  [1536-dim]
  C) each gamma(k) alone            (256-dim, per-k informativeness)

Report RMSE + F1/AUC-PR at k=30 for each variant, averaged across the 3 Phase 1 seeds.

Success: trajectory probe (B) RMSE < 15.0.
"""

import sys, json, copy, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

V11 = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/experiments/v11')
V17 = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/experiments/v17')
sys.path.insert(0, str(V11))
sys.path.insert(0, '/home/sagemaker-user/IndustrialJEPA/mechanical-jepa')

from models import TrajectoryJEPA
from data_utils import (load_cmapss_subset, N_SENSORS, RUL_CAP,
                         CMAPSSFinetuneDataset, CMAPSSTestDataset,
                         collate_finetune, collate_test)
from evaluation.grey_swan_metrics import anomaly_metrics as _anomaly_metrics

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

D_MODEL = 256
N_HEADS = 4
N_LAYERS = 2
D_FF = 4 * D_MODEL
EMA_MOMENTUM = 0.99

PROBE_LR = 1e-3
PROBE_EPOCHS = 200
PROBE_PATIENCE = 25
K_SET = [5, 10, 20, 50, 100]
K_EVAL_F1 = 30
SEEDS = [42, 123, 456]
CKPT_DIR = V17 / 'ckpts'


def load_phase1_model(seed):
    model = TrajectoryJEPA(
        n_sensors=N_SENSORS, patch_length=1,
        d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS, d_ff=D_FF,
        dropout=0.1, ema_momentum=EMA_MOMENTUM, predictor_hidden=D_FF,
    ).to(DEVICE)
    state = torch.load(CKPT_DIR / f'v17_seed{seed}_best.pt', map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


@torch.no_grad()
def extract_features(model, past, mask, k_set=K_SET, variant='traj'):
    """
    variant='h'      -> h_past (D,)
    variant='traj'   -> concat(h, gamma(k1), ..., gamma(kn))  [D*(1+len(k_set))]
    variant='g5'..'g100' -> gamma(k) alone for that k
    """
    h = model.encode_past(past, mask)  # (B, D)
    if variant == 'h':
        return h
    feats = [h]
    k_bs = {k: torch.full((h.shape[0],), k, dtype=torch.long, device=h.device)
            for k in k_set}
    for k in k_set:
        g = model.predictor(h, k_bs[k])
        feats.append(g)
    if variant == 'traj':
        return torch.cat(feats, dim=-1)
    # Per-k variants: g5, g10, g20, g50, g100
    key_to_idx = {f'g{k}': i for i, k in enumerate(k_set, start=1)}
    return feats[key_to_idx[variant]]


def train_probe(model, variant, train_engines, val_engines, test_engines,
                test_rul, seed=42, verbose=False):
    """
    Train a linear probe on extracted features from frozen model.
    Returns test metrics at best val.
    """
    torch.manual_seed(seed)

    # Determine feature dim
    if variant == 'h':
        feat_dim = D_MODEL
    elif variant == 'traj':
        feat_dim = D_MODEL * (1 + len(K_SET))
    else:
        feat_dim = D_MODEL

    probe = nn.Sequential(nn.Linear(feat_dim, 1), nn.Sigmoid()).to(DEVICE)
    opt = torch.optim.Adam(probe.parameters(), lr=PROBE_LR)

    tr_ds = CMAPSSFinetuneDataset(train_engines, n_cuts_per_engine=5, seed=seed)
    va_ds = CMAPSSFinetuneDataset(val_engines, use_last_only=True)
    te_ds = CMAPSSTestDataset(test_engines, test_rul)
    tr = DataLoader(tr_ds, batch_size=32, shuffle=True, collate_fn=collate_finetune)
    va = DataLoader(va_ds, batch_size=32, shuffle=False, collate_fn=collate_finetune)
    te = DataLoader(te_ds, batch_size=32, shuffle=False, collate_fn=collate_test)

    best_val = float('inf')
    best_state = None
    no_impr = 0

    for ep in range(PROBE_EPOCHS):
        probe.train()
        for past, mask, rul in tr:
            past, mask, rul = past.to(DEVICE), mask.to(DEVICE), rul.to(DEVICE)
            feat = extract_features(model, past, mask, variant=variant)
            p = probe(feat).squeeze(-1)
            loss = F.mse_loss(p, rul)
            opt.zero_grad(); loss.backward(); opt.step()

        probe.eval()
        pv, tv = [], []
        with torch.no_grad():
            for past, mask, rul in va:
                past, mask = past.to(DEVICE), mask.to(DEVICE)
                feat = extract_features(model, past, mask, variant=variant)
                pv.append(probe(feat).squeeze(-1).cpu().numpy())
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

    probe.load_state_dict(best_state)

    # Test metrics
    probe.eval()
    p_test, t_test = [], []
    with torch.no_grad():
        for past, mask, rul_gt in te:
            past, mask = past.to(DEVICE), mask.to(DEVICE)
            feat = extract_features(model, past, mask, variant=variant)
            p_test.append(probe(feat).squeeze(-1).cpu().numpy() * RUL_CAP)
            t_test.append(rul_gt.numpy())
    preds_test = np.concatenate(p_test)
    targs_test = np.concatenate(t_test)
    test_rmse = float(np.sqrt(np.mean((preds_test - targs_test) ** 2)))

    y = (targs_test <= K_EVAL_F1).astype(int)
    score = -preds_test
    try:
        thr = float(np.percentile(score[y == 0], 95)) if (y == 0).sum() > 0 else 0.0
    except Exception:
        thr = 0.0
    m = _anomaly_metrics(score, y, threshold=thr)

    return dict(
        variant=variant, seed=seed, val_rmse=best_val, test_rmse=test_rmse,
        f1=m['f1_non_pa'], auc_pr=m['auc_pr'],
        precision=m['precision_non_pa'], recall=m['recall_non_pa'],
    )


def main():
    data = load_cmapss_subset('FD001')
    print(f"FD001 loaded", flush=True)

    variants = ['h', 'traj'] + [f'g{k}' for k in K_SET]
    all_results = {v: [] for v in variants}

    t0 = time.time()
    for seed in SEEDS:
        ckpt = CKPT_DIR / f'v17_seed{seed}_best.pt'
        if not ckpt.exists():
            print(f"  [seed {seed}] ckpt missing: {ckpt}, skipping", flush=True)
            continue
        print(f"\n=== seed {seed} ===", flush=True)
        model = load_phase1_model(seed)
        for v in variants:
            r = train_probe(model, v, data['train_engines'], data['val_engines'],
                            data['test_engines'], data['test_rul'], seed=seed)
            all_results[v].append(r)
            print(f"  variant={v:5s} val_rmse={r['val_rmse']:.2f} "
                  f"test_rmse={r['test_rmse']:.2f} "
                  f"F1@{K_EVAL_F1}={r['f1']:.3f} AUC-PR={r['auc_pr']:.3f}",
                  flush=True)
        del model
        torch.cuda.empty_cache()

    # Aggregate
    summary = {'config': 'v17_phase2_trajectory_probe',
               'K_set': K_SET, 'k_eval_f1': K_EVAL_F1,
               'seeds': SEEDS, 'variants': {}}
    for v in variants:
        rs = all_results[v]
        if not rs:
            continue
        summary['variants'][v] = {
            'n_seeds': len(rs),
            'val_rmse_per_seed': [r['val_rmse'] for r in rs],
            'test_rmse_per_seed': [r['test_rmse'] for r in rs],
            'test_rmse_mean': float(np.mean([r['test_rmse'] for r in rs])),
            'test_rmse_std': float(np.std([r['test_rmse'] for r in rs])),
            'val_rmse_mean': float(np.mean([r['val_rmse'] for r in rs])),
            'val_rmse_std': float(np.std([r['val_rmse'] for r in rs])),
            'f1_mean': float(np.mean([r['f1'] for r in rs])),
            'auc_pr_mean': float(np.mean([r['auc_pr'] for r in rs])),
        }
    summary['v2_baseline_rmse'] = 17.81
    summary['runtime_minutes'] = (time.time() - t0) / 60

    with open(V17 / 'phase2_trajectory_probe_results.json', 'w') as f:
        json.dump(summary, f, indent=2, default=float)

    print("\n" + "=" * 60)
    print("V17 Phase 2 Trajectory Probe SUMMARY")
    print("=" * 60)
    print(f"{'variant':<8} {'val_rmse':>10} {'test_rmse':>10} {'f1':>8} {'auc_pr':>8}")
    print("-" * 60)
    for v, s in summary['variants'].items():
        print(f"{v:<8} {s['val_rmse_mean']:>7.2f}+/-{s['val_rmse_std']:<4.2f} "
              f"{s['test_rmse_mean']:>7.2f}+/-{s['test_rmse_std']:<4.2f} "
              f"{s['f1_mean']:>7.3f} {s['auc_pr_mean']:>7.3f}")
    print(f"V2 ref          {17.81:>7.2f}                 -")


if __name__ == '__main__':
    main()
