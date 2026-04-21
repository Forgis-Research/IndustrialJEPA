"""V20 Phase 6: Pretrain V17 on FD003 + replicate label-efficiency curve.

Reviewer feedback: pred-FT crossover claim needs replication on another
C-MAPSS subset. FD003 has single operating condition like FD001 but different
fault modes. Pretrain one V17-arch encoder on FD003 (seed 42), then run pred-FT
vs E2E at 100/50/20/10/5% labels with 5 seeds.
"""
import sys, math, json, time, copy
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

ROOT = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa')
V11 = ROOT / 'experiments' / 'v11'
V17 = ROOT / 'experiments' / 'v17'
V20 = ROOT / 'experiments' / 'v20'
sys.path.insert(0, str(V11)); sys.path.insert(0, str(V20)); sys.path.insert(0, str(ROOT))

from models import TrajectoryJEPA
from data_utils import load_cmapss_subset, N_SENSORS, RUL_CAP
from train_utils import subsample_engines
from pred_ft_utils import (MultiHorizonHead, CMAPSSWindowedDataset,
                           CMAPSSTestWindowedDataset, collate_windowed,
                           train_one, evaluate)
from phase0_pred_ft import CFG as DOWNSTREAM_CFG

# Import v17 pretraining helpers
sys.path.insert(0, str(V17))
from phase1_v17_baseline import (
    V17PretrainDataset, collate_v17_pretrain, v17_loss,
    D_MODEL, N_HEADS, N_LAYERS, D_FF, BATCH_SIZE, LR, WEIGHT_DECAY,
    EMA_MOMENTUM, K_MAX, W_WIN, MIN_PAST, N_CUTS,
)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CKPT_DIR = V20 / 'ckpts_fd003'
CKPT_DIR.mkdir(exist_ok=True, parents=True)

# Shorter pretraining (100 epochs) - FD003 has similar scale to FD001
FD003_PRETRAIN_EPOCHS = 100
FD003_CKPT_SEED = 42
ARCH = dict(n_sensors=N_SENSORS, patch_length=1,
            d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS, d_ff=D_FF,
            dropout=0.1, ema_momentum=EMA_MOMENTUM, predictor_hidden=D_FF)
N_WINDOWS = 16
BUDGETS = [1.0, 0.5, 0.2, 0.1, 0.05]
SEEDS = [0, 1, 2, 3, 4]


def pretrain_fd003(data, seed=FD003_CKPT_SEED, n_epochs=FD003_PRETRAIN_EPOCHS):
    torch.manual_seed(seed); np.random.seed(seed)
    model = TrajectoryJEPA(**ARCH).to(DEVICE)
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR, weight_decay=WEIGHT_DECAY,
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_epochs)
    history = []
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
        history.append(tot / n)
        if ep % 10 == 0 or ep == 1:
            print(f"  FD003 pretrain ep {ep:3d} | L={tot/n:.4f} "
                  f"({(time.time()-t0)/60:.1f} min)", flush=True)
    ckpt_path = CKPT_DIR / f'v20_fd003_seed{seed}_ep{n_epochs}.pt'
    torch.save(model.state_dict(), ckpt_path)
    print(f"FD003 pretraining done in {(time.time()-t0)/60:.1f} min, "
          f"saved to {ckpt_path.name}", flush=True)
    return ckpt_path


def run_one_pred_ft(data, ckpt_path, mode, budget, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    model = TrajectoryJEPA(**ARCH).to(DEVICE)
    if mode == 'scratch':
        pass                        # random init
    else:
        sd = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(sd)

    head = MultiHorizonHead(ARCH['d_model'], n_windows=N_WINDOWS).to(DEVICE)

    sub = subsample_engines(data['train_engines'], budget, seed=seed)
    tr_ds = CMAPSSWindowedDataset(sub, n_cuts_per_engine=5, rul_cap=RUL_CAP, seed=seed)
    va_ds = CMAPSSWindowedDataset(data['val_engines'], n_cuts_per_engine=10,
                                  rul_cap=RUL_CAP, seed=seed + 111)
    te_ds = CMAPSSTestWindowedDataset(data['test_engines'], data['test_rul'])
    tr = DataLoader(tr_ds, batch_size=16, shuffle=True, collate_fn=collate_windowed)
    va = DataLoader(va_ds, batch_size=16, shuffle=False, collate_fn=collate_windowed)
    te = DataLoader(te_ds, batch_size=16, shuffle=False, collate_fn=collate_windowed)

    cfg = DOWNSTREAM_CFG[mode]
    t = train_one(model, head, tr, va, mode=mode,
                  n_windows=N_WINDOWS, device=DEVICE, **cfg)
    ev = evaluate(model, head, te, mode=mode,
                  n_windows=N_WINDOWS, rul_cap=RUL_CAP, device=DEVICE)
    return {
        'mode': mode, 'budget': budget, 'seed': seed,
        'val_mse': t['best_val'], 'final_epoch': t['final_epoch'],
        'test_rmse': ev['legacy']['rmse'], 'test_nasa': ev['legacy']['nasa_score'],
        'per_window_f1_mean': ev['per_window']['f1_mean'],
        'per_window_auroc_mean': ev['per_window']['auroc_mean'],
        'per_window_precision_mean': ev['per_window']['precision_mean'],
        'per_window_recall_mean': ev['per_window']['recall_mean'],
    }


def aggregate(rs):
    from scipy.stats import t as t_dist
    keys = ['test_rmse', 'per_window_f1_mean', 'per_window_auroc_mean']
    out = {'n_seeds': len(rs), 'per_seed': rs}
    for k in keys:
        vals = np.array([r[k] for r in rs if np.isfinite(r[k])])
        if not len(vals):
            out[f'{k}_mean'] = float('nan'); out[f'{k}_std'] = float('nan'); continue
        mean = float(vals.mean()); std = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        out[f'{k}_mean'] = mean; out[f'{k}_std'] = std
        if len(vals) > 1:
            tc = float(t_dist.ppf(0.975, df=len(vals) - 1))
            out[f'{k}_ci95_lo'] = mean - tc * std / np.sqrt(len(vals))
            out[f'{k}_ci95_hi'] = mean + tc * std / np.sqrt(len(vals))
    return out


def main():
    out_path = V20 / 'phase6_fd003.json'
    print("=" * 80)
    print("V20 Phase 6: FD003 pretraining + label-efficiency replication")
    print("=" * 80, flush=True)

    t0 = time.time()
    data = load_cmapss_subset('FD003')
    print(f"FD003: {len(data['train_engines'])} train, "
          f"{len(data['val_engines'])} val, {len(data['test_engines'])} test",
          flush=True)

    ckpt_path = CKPT_DIR / f'v20_fd003_seed{FD003_CKPT_SEED}_ep{FD003_PRETRAIN_EPOCHS}.pt'
    if ckpt_path.exists():
        print(f"Pretrained FD003 ckpt found: {ckpt_path.name}", flush=True)
    else:
        ckpt_path = pretrain_fd003(data)

    # Label-efficiency sweep
    results = {}
    for mode in ['pred_ft', 'e2e']:
        for budget in BUDGETS:
            key = f'{mode}@{budget}'
            results[key] = []
            for seed in SEEDS:
                t1 = time.time()
                r = run_one_pred_ft(data, ckpt_path, mode, budget, seed)
                dt = time.time() - t1
                results[key].append(r)
                print(f"  [{mode:8s} b={budget*100:4.0f}% s={seed}] "
                      f"RMSE={r['test_rmse']:.2f} F1w={r['per_window_f1_mean']:.3f} "
                      f"AUROCw={r['per_window_auroc_mean']:.3f} ({dt:.0f}s)",
                      flush=True)
                out = {'config': 'v20_phase6_fd003',
                       'seeds': SEEDS, 'budgets': BUDGETS,
                       'modes': ['pred_ft', 'e2e'],
                       'ckpt': str(ckpt_path),
                       'runtime_min': (time.time()-t0)/60,
                       'results': {k: aggregate(v) for k, v in results.items()}}
                with open(out_path, 'w') as f:
                    json.dump(out, f, indent=2, default=float)

    print(f"\nRuntime: {(time.time()-t0)/60:.1f} min")


if __name__ == '__main__':
    main()
