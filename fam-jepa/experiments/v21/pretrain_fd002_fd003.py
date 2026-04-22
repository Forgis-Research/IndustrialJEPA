"""V21 Pretrain extra seeds for FD002 and FD003.

Reuses the v17-style pretraining loop from v20 phase11/phase10 but runs
seeds 123 and 456 with early stopping (patience=5, max 50ep). Writes
checkpoints into the v20 ckpts dirs so cmapss_runner picks them up.

Estimated time: 4 (2 datasets × 2 seeds) × ~10 min = ~40 min.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path('/home/sagemaker-user/IndustrialJEPA')
FAM = ROOT / 'fam-jepa'
V11 = FAM / 'experiments' / 'v11'
V17 = FAM / 'experiments' / 'v17'
V20_OLD = ROOT / 'mechanical-jepa' / 'experiments' / 'v20'
V21 = FAM / 'experiments' / 'v21'

sys.path.insert(0, str(V11))
sys.path.insert(0, str(V17))
sys.path.insert(0, str(V20_OLD))
sys.path.insert(0, str(FAM))

from models import TrajectoryJEPA  # noqa: E402
from data_utils import load_cmapss_subset, N_SENSORS  # noqa: E402
# phase1_v17_baseline sets: v17_loss, V17PretrainDataset, collate_v17_pretrain,
# D_MODEL, N_HEADS, N_LAYERS, D_FF, BATCH_SIZE, LR, WEIGHT_DECAY, EMA_MOMENTUM, N_CUTS
from phase1_v17_baseline import (  # noqa: E402
    V17PretrainDataset, collate_v17_pretrain, v17_loss,
    D_MODEL, N_HEADS, N_LAYERS, D_FF,
    BATCH_SIZE, LR, WEIGHT_DECAY, EMA_MOMENTUM, N_CUTS,
)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
ARCH = dict(n_sensors=N_SENSORS, patch_length=1,
            d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS, d_ff=D_FF,
            dropout=0.1, ema_momentum=EMA_MOMENTUM, predictor_hidden=D_FF)

FD002_DIR = V20_OLD / 'ckpts_fd002'
FD003_DIR = V20_OLD / 'ckpts_fd003'


def pretrain_one(data: dict, seed: int, max_epochs: int = 50,
                 patience: int = 5) -> tuple[float, int, 'TrajectoryJEPA']:
    torch.manual_seed(seed); np.random.seed(seed)
    model = TrajectoryJEPA(**ARCH).to(DEVICE)
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, max_epochs)

    best_loss = float('inf'); best_sd = None; no_impr = 0
    losses = []
    t0 = time.time()
    for ep in range(1, max_epochs + 1):
        ds = V17PretrainDataset(data['train_engines'], n_cuts=N_CUTS,
                                seed=seed * 1000 + ep)
        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True,
                            collate_fn=collate_v17_pretrain, num_workers=0)
        model.train()
        tot = 0.0; n = 0
        for past, past_mask, fut, fut_mask, k, _t in loader:
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
            B = past.shape[0]; tot += loss.item() * B; n += B
        sched.step()
        ep_loss = tot / max(n, 1)
        losses.append(ep_loss)
        improved = ep_loss < best_loss - 1e-4
        if improved:
            best_loss = ep_loss; best_sd = copy.deepcopy(model.state_dict()); no_impr = 0
        else:
            no_impr += 1
        if ep % 5 == 0 or ep == 1:
            print(f"    ep {ep:3d} | L={ep_loss:.4f} best={best_loss:.4f} "
                  f"no_impr={no_impr} ({(time.time()-t0)/60:.1f}m)", flush=True)
        if no_impr >= patience:
            print(f"    Early stop at ep {ep} (best={best_loss:.4f})", flush=True)
            break
    if best_sd is not None:
        model.load_state_dict(best_sd)
    return best_loss, ep, model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--subsets', nargs='+', default=['FD002', 'FD003'])
    ap.add_argument('--seeds', nargs='+', type=int, default=[123, 456])
    ap.add_argument('--max-epochs', type=int, default=50)
    ap.add_argument('--patience', type=int, default=5)
    args = ap.parse_args()

    FD002_DIR.mkdir(exist_ok=True, parents=True)
    FD003_DIR.mkdir(exist_ok=True, parents=True)
    out_log = {}
    t_all = time.time()

    for subset in args.subsets:
        out_dir = FD002_DIR if subset == 'FD002' else FD003_DIR
        print(f'\n=== {subset} pretrain (max_ep={args.max_epochs}, patience={args.patience}) ===',
              flush=True)
        data = load_cmapss_subset(subset)
        print(f'  loaded {len(data["train_engines"])} train engines',
              flush=True)
        for seed in args.seeds:
            print(f'  -- seed={seed} --', flush=True)
            t0 = time.time()
            best_loss, final_ep, model = pretrain_one(
                data, seed, max_epochs=args.max_epochs, patience=args.patience)
            ckpt_name = f'v21_{subset.lower()}_seed{seed}_best.pt'
            ckpt = out_dir / ckpt_name
            torch.save(model.state_dict(), ckpt)
            dt = (time.time() - t0) / 60
            print(f'  -> {ckpt} (best L={best_loss:.4f} ep={final_ep} {dt:.1f}m)',
                  flush=True)
            out_log.setdefault(subset, {})[seed] = {
                'best_loss': best_loss, 'final_ep': final_ep,
                'ckpt': str(ckpt), 'runtime_min': dt,
            }
            del model
            torch.cuda.empty_cache()

    log_path = V21 / 'pretrain_fd002_fd003.json'
    json.dump({'log': out_log, 'runtime_min': (time.time() - t_all) / 60,
               'config': vars(args)}, open(log_path, 'w'), indent=2, default=float)
    print(f'\nDONE in {(time.time()-t_all)/60:.1f} min. Log: {log_path}')


if __name__ == '__main__':
    main()
