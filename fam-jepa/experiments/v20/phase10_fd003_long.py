"""V20 Phase 10: FD003 pretrained for 200 epochs (matching v17 FD001 compute).

Phase 6 ran FD003 with 100 epochs → F1w ~0.146 at 100% labels (vs FD001's
0.391 at matched 200 epochs). Hypothesis: the absolute F1w gap is mostly
pretraining under-compute, not FD003 being intrinsically harder. Test by
running 200 epochs and re-running the label-efficiency sweep.
"""
import sys, json, time, math
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa')
V11 = ROOT / 'experiments' / 'v11'
V17 = ROOT / 'experiments' / 'v17'
V20 = ROOT / 'experiments' / 'v20'
sys.path.insert(0, str(V11)); sys.path.insert(0, str(V17)); sys.path.insert(0, str(V20)); sys.path.insert(0, str(ROOT))

from models import TrajectoryJEPA
from data_utils import load_cmapss_subset, N_SENSORS, RUL_CAP
from train_utils import subsample_engines
from pred_ft_utils import (MultiHorizonHead, CMAPSSWindowedDataset,
                           CMAPSSTestWindowedDataset, collate_windowed,
                           train_one, evaluate)
from phase0_pred_ft import CFG as DOWNSTREAM_CFG
from phase1_v17_baseline import (V17PretrainDataset, collate_v17_pretrain,
                                  v17_loss, D_MODEL, N_HEADS, N_LAYERS, D_FF,
                                  BATCH_SIZE, LR, WEIGHT_DECAY, EMA_MOMENTUM,
                                  K_MAX, W_WIN, MIN_PAST, N_CUTS)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CKPT_DIR = V20 / 'ckpts_fd003'
CKPT_DIR.mkdir(exist_ok=True, parents=True)

FD003_PRETRAIN_EPOCHS = 200            # matches v17 FD001
FD003_CKPT_SEED = 42
ARCH = dict(n_sensors=N_SENSORS, patch_length=1,
            d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS, d_ff=D_FF,
            dropout=0.1, ema_momentum=EMA_MOMENTUM, predictor_hidden=D_FF)
N_WINDOWS = 16
BUDGETS = [1.0, 0.1, 0.05]             # keep essential budgets
SEEDS = [0, 1, 2, 3, 4]


def pretrain_long(data, seed, n_epochs):
    torch.manual_seed(seed); np.random.seed(seed)
    model = TrajectoryJEPA(**ARCH).to(DEVICE)
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR, weight_decay=WEIGHT_DECAY,
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_epochs)
    t0 = time.time()
    for ep in range(1, n_epochs + 1):
        ds = V17PretrainDataset(data['train_engines'], n_cuts=N_CUTS,
                                 seed=seed * 1000 + ep)
        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True,
                            collate_fn=collate_v17_pretrain, num_workers=0)
        model.train()
        tot = 0.0; n = 0
        for past, past_mask, fut, fut_mask, k, t in loader:
            past, past_mask = past.to(DEVICE), past_mask.to(DEVICE)
            fut, fut_mask = fut.to(DEVICE), fut_mask.to(DEVICE)
            k = k.to(DEVICE)
            opt.zero_grad()
            pred, targ, _ = model.forward_pretrain(past, past_mask,
                                                    fut, fut_mask, k)
            loss, _, _ = v17_loss(pred, targ, lambda_var=0.04)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); model.update_ema()
            B = past.shape[0]
            tot += loss.item() * B; n += B
        sched.step()
        if ep % 25 == 0 or ep == 1:
            print(f"  FD003 long pretrain ep {ep:3d} | L={tot/n:.4f} "
                  f"({(time.time()-t0)/60:.1f} min)", flush=True)
    ckpt = CKPT_DIR / f'v20_fd003_seed{seed}_ep{n_epochs}.pt'
    torch.save(model.state_dict(), ckpt)
    print(f"Done in {(time.time()-t0)/60:.1f} min -> {ckpt.name}", flush=True)
    return ckpt


def run_one(data, ckpt, mode, budget, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    model = TrajectoryJEPA(**ARCH).to(DEVICE)
    if mode != 'scratch':
        sd = torch.load(ckpt, map_location=DEVICE, weights_only=False)
        model.load_state_dict(sd)
    head = MultiHorizonHead(ARCH['d_model'], n_windows=N_WINDOWS).to(DEVICE)
    sub = subsample_engines(data['train_engines'], budget, seed=seed)
    tr_ds = CMAPSSWindowedDataset(sub, n_cuts_per_engine=5, rul_cap=RUL_CAP, seed=seed)
    va_ds = CMAPSSWindowedDataset(data['val_engines'], n_cuts_per_engine=10,
                                  rul_cap=RUL_CAP, seed=seed+111)
    te_ds = CMAPSSTestWindowedDataset(data['test_engines'], data['test_rul'])
    tr = DataLoader(tr_ds, batch_size=16, shuffle=True, collate_fn=collate_windowed)
    va = DataLoader(va_ds, batch_size=16, shuffle=False, collate_fn=collate_windowed)
    te = DataLoader(te_ds, batch_size=16, shuffle=False, collate_fn=collate_windowed)
    t = train_one(model, head, tr, va, mode=mode, n_windows=N_WINDOWS,
                  device=DEVICE, **DOWNSTREAM_CFG[mode])
    ev = evaluate(model, head, te, mode=mode, n_windows=N_WINDOWS,
                  rul_cap=RUL_CAP, device=DEVICE)
    return {
        'mode': mode, 'budget': budget, 'seed': seed,
        'val_mse': t['best_val'], 'final_epoch': t['final_epoch'],
        'test_rmse': ev['legacy']['rmse'],
        'per_window_f1_mean': ev['per_window']['f1_mean'],
        'per_window_auroc_mean': ev['per_window']['auroc_mean'],
    }


def main():
    out_path = V20 / 'phase10_fd003_long.json'
    data = load_cmapss_subset('FD003')
    print("=" * 80)
    print(f"V20 Phase 10: FD003 pretrained 200 epochs + label-efficiency")
    print("=" * 80, flush=True)

    t0 = time.time()
    ckpt = CKPT_DIR / f'v20_fd003_seed{FD003_CKPT_SEED}_ep{FD003_PRETRAIN_EPOCHS}.pt'
    if ckpt.exists():
        print(f"200-ep ckpt exists: {ckpt.name}", flush=True)
    else:
        ckpt = pretrain_long(data, FD003_CKPT_SEED, FD003_PRETRAIN_EPOCHS)

    from scipy.stats import ttest_rel
    results = {}
    for mode in ['pred_ft', 'e2e']:
        for b in BUDGETS:
            k = f'{mode}@{b}'
            results[k] = []
            for s in SEEDS:
                t1 = time.time()
                r = run_one(data, ckpt, mode, b, s)
                dt = time.time() - t1
                results[k].append(r)
                print(f"  [{mode:8s} b={b*100:4.0f}% s={s}] "
                      f"RMSE={r['test_rmse']:.2f} "
                      f"F1w={r['per_window_f1_mean']:.3f} "
                      f"AUROCw={r['per_window_auroc_mean']:.3f} ({dt:.0f}s)",
                      flush=True)

    # Aggregate + compare to FD001
    print("\nFD003 200-ep vs phase 6 (100-ep):")
    print(f"{'Budget':>7s} | {'200-ep pred-FT':>20s} | {'100-ep pred-FT':>20s} | ΔF1w(200-100)")
    print("-" * 80)
    with open(V20 / 'phase6_fd003.json') as f:
        ph6 = json.load(f)
    for b in BUDGETS:
        p_new = [r['per_window_f1_mean'] for r in results[f'pred_ft@{b}']]
        p_old = [r['per_window_f1_mean'] for r in ph6['results'][f'pred_ft@{b}']['per_seed']]
        print(f'{b*100:4.0f}%   | {np.mean(p_new):.3f} ± {np.std(p_new, ddof=1):.3f}        | '
              f'{np.mean(p_old):.3f} ± {np.std(p_old, ddof=1):.3f}        | '
              f'{np.mean(p_new) - np.mean(p_old):+.3f}')

    print("\nPaired pred-FT vs E2E at each budget (200-ep ckpt):")
    for b in BUDGETS:
        pred = np.array([r['per_window_f1_mean'] for r in results[f'pred_ft@{b}']])
        e2e = np.array([r['per_window_f1_mean'] for r in results[f'e2e@{b}']])
        t, p = ttest_rel(pred, e2e)
        print(f'  {b*100:4.0f}%: pred {pred.mean():.3f} vs e2e {e2e.mean():.3f}, '
              f'diff {pred.mean()-e2e.mean():+.3f}, paired t p={p:.3f}')

    summary = {
        'config': 'v20_phase10_fd003_long',
        'pretrain_epochs': FD003_PRETRAIN_EPOCHS,
        'seeds': SEEDS, 'budgets': BUDGETS,
        'ckpt': str(ckpt),
        'runtime_min': (time.time()-t0)/60,
        'results': {k: {'per_seed': v,
                        'f1_mean': float(np.mean([r['per_window_f1_mean'] for r in v])),
                        'f1_std': float(np.std([r['per_window_f1_mean'] for r in v], ddof=1)),
                        'rmse_mean': float(np.mean([r['test_rmse'] for r in v])),
                        'rmse_std': float(np.std([r['test_rmse'] for r in v], ddof=1))}
                    for k, v in results.items()},
    }
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"\nRuntime: {(time.time()-t0)/60:.1f} min")


if __name__ == '__main__':
    main()
