"""
V14 Phase 2e: full-sequence target encoder on FD004.

Hardest subset: 6 operating conditions + 2 fault modes (fan + HPC),
249 train engines. Per-condition normalization.

V2 baseline (from paper tab:multisubset): frozen 29.35, E2E 25.62
(single-seed numbers in paper, re-use). STAR FD004: 14.25.

Protocol: in-domain pretrain on FD004 with full-sequence target,
3-seed frozen + E2E fine-tune at 100% labels. Tests whether
Phase 2 full-sequence generalizes to the hardest multi-condition
multi-fault subset.

Output: experiments/v14/phase2e_fd004_results.json
"""

import sys, json, time, copy
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

V11_DIR = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/experiments/v11')
V14_DIR = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/experiments/v14')
sys.path.insert(0, str(V11_DIR))
sys.path.insert(0, str(V14_DIR))

from data_utils import (
    load_cmapss_subset, N_SENSORS, RUL_CAP,
    CMAPSSFinetuneDataset, CMAPSSTestDataset, collate_finetune, collate_test,
)
from models import TrajectoryJEPA, RULProbe, trajectory_jepa_loss
from phase2_full_sequence import (
    CMAPSSFullSequenceDataset, collate_full_sequence, eval_probe_rmse,
    D_MODEL, BATCH_SIZE, N_CUTS, LAMBDA_VAR, PROBE_EVERY,
)

try:
    import wandb; HAS_WANDB = True
except Exception: HAS_WANDB = False

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
N_EPOCHS = 100  # tighter budget given larger dataset
PATIENCE_PROBE = 8


def pretrain(data, ckpt, log, seed=42):
    torch.manual_seed(seed); np.random.seed(seed)
    model = TrajectoryJEPA(
        n_sensors=N_SENSORS, patch_length=1, d_model=D_MODEL, n_heads=4,
        n_layers=2, d_ff=512, dropout=0.1,
        ema_momentum=0.99, predictor_hidden=256,
    ).to(DEVICE)
    optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                              lr=3e-4, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, N_EPOCHS)
    history = {'loss': [], 'probe_rmse': [], 'probe_epochs': []}
    best = float('inf'); best_state = None; no = 0
    run = None
    if HAS_WANDB:
        try:
            run = wandb.init(project='industrialjepa', name='v14-phase2e-fullseq-fd004',
                             tags=['v14-phase2e-fullseq-fd004'], reinit=True)
        except Exception: pass
    t0 = time.time()
    with open(log, 'w') as f: f.write("V14 Phase 2e FD004 full-sequence\n")
    for epoch in range(1, N_EPOCHS + 1):
        ds = CMAPSSFullSequenceDataset(data['train_engines'], n_cuts_per_engine=N_CUTS,
                                        min_past=10, min_horizon=5, max_horizon=30,
                                        seed=epoch)
        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True,
                            collate_fn=collate_full_sequence)
        model.train(); tot = 0; n = 0
        for past, past_m, full, full_m, k, _ in loader:
            past, past_m = past.to(DEVICE), past_m.to(DEVICE)
            full, full_m = full.to(DEVICE), full_m.to(DEVICE); k = k.to(DEVICE)
            optim.zero_grad()
            h_past = model.context_encoder(past, past_m)
            with torch.no_grad():
                h_full = model.target_encoder(full, full_m)
            pred = model.predictor(h_past, k)
            loss, _, _ = trajectory_jepa_loss(pred, h_full, LAMBDA_VAR)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step(); model.update_ema()
            tot += loss.item() * past.shape[0]; n += past.shape[0]
        history['loss'].append(tot / n); sched.step()
        extra = ''
        if epoch % PROBE_EVERY == 0 or epoch == 1:
            probe_rmse = eval_probe_rmse(model, data['train_engines'], data['val_engines'])
            history['probe_rmse'].append(probe_rmse); history['probe_epochs'].append(epoch)
            if probe_rmse < best:
                best = probe_rmse; best_state = copy.deepcopy(model.state_dict())
                torch.save(best_state, ckpt); no = 0
            else: no += 1
            extra = f" | probe={probe_rmse:.2f} (best={best:.2f}, ni={no})"
        line = f"Ep {epoch:3d} | loss={tot/n:.4f}{extra}"
        print(line, flush=True)
        with open(log, 'a') as f: f.write(line + '\n')
        if run is not None:
            try:
                d = {'epoch': epoch, 'loss': tot / n}
                if epoch % PROBE_EVERY == 0 or epoch == 1:
                    d['probe_rmse'] = probe_rmse; d['best_probe_rmse'] = best
                wandb.log(d)
            except Exception: pass
        if no >= PATIENCE_PROBE: print(f"Early stop at {epoch}"); break
    if run is not None:
        try: wandb.finish()
        except Exception: pass
    if best_state is not None: model.load_state_dict(best_state)
    print(f"Pretrain done {(time.time()-t0)/60:.1f} min, best probe={best:.2f}")
    return model, history, best


def run_finetune(ckpt, data, mode, seed):
    model = TrajectoryJEPA(
        n_sensors=N_SENSORS, patch_length=1, d_model=D_MODEL, n_heads=4,
        n_layers=2, d_ff=512, dropout=0.1,
    ).to(DEVICE)
    model.load_state_dict(torch.load(str(ckpt), map_location=DEVICE))
    probe = RULProbe(D_MODEL).to(DEVICE)
    torch.manual_seed(seed); np.random.seed(seed)
    tr = DataLoader(CMAPSSFinetuneDataset(data['train_engines'], n_cuts_per_engine=5, seed=seed),
                    batch_size=16, shuffle=True, collate_fn=collate_finetune)
    va = DataLoader(CMAPSSFinetuneDataset(data['val_engines'], use_last_only=True),
                    batch_size=16, shuffle=False, collate_fn=collate_finetune)
    te = DataLoader(CMAPSSTestDataset(data['test_engines'], data['test_rul']),
                    batch_size=16, shuffle=False, collate_fn=collate_test)
    if mode == 'frozen':
        for p in model.parameters(): p.requires_grad = False
        optim = torch.optim.Adam(probe.parameters(), lr=1e-3)
    else:
        for p in model.context_encoder.parameters(): p.requires_grad = True
        optim = torch.optim.Adam(
            list(model.context_encoder.parameters()) + list(probe.parameters()), lr=1e-4)
    best_val = float('inf'); best_ps = None; best_es = None; no = 0
    for _ in range(100):
        if mode == 'frozen': model.eval()
        else: model.train()
        probe.train()
        for past, mask, rul in tr:
            past, mask, rul = past.to(DEVICE), mask.to(DEVICE), rul.to(DEVICE)
            optim.zero_grad()
            if mode == 'frozen':
                with torch.no_grad(): h = model.encode_past(past, mask)
            else: h = model.encode_past(past, mask)
            F.mse_loss(probe(h), rul).backward()
            torch.nn.utils.clip_grad_norm_(probe.parameters(), 1.0)
            optim.step()
        model.eval(); probe.eval(); pv, tv = [], []
        with torch.no_grad():
            for past, mask, rul in va:
                past, mask = past.to(DEVICE), mask.to(DEVICE)
                h = model.encode_past(past, mask)
                pv.append(probe(h).cpu().numpy()); tv.append(rul.numpy())
        val_rmse = float(np.sqrt(np.mean(
            (np.concatenate(pv)*RUL_CAP - np.concatenate(tv)*RUL_CAP)**2)))
        if val_rmse < best_val:
            best_val = val_rmse; best_ps = copy.deepcopy(probe.state_dict())
            if mode == 'e2e': best_es = copy.deepcopy(model.context_encoder.state_dict())
            no = 0
        else:
            no += 1
            if no >= 20: break
    probe.load_state_dict(best_ps)
    if mode == 'e2e' and best_es is not None:
        model.context_encoder.load_state_dict(best_es)
    model.eval(); probe.eval()
    pt, tt = [], []
    with torch.no_grad():
        for past, mask, rul_gt in te:
            past, mask = past.to(DEVICE), mask.to(DEVICE)
            h = model.encode_past(past, mask)
            pt.append(probe(h).cpu().numpy() * RUL_CAP); tt.append(rul_gt.numpy())
    return float(np.sqrt(np.mean((np.concatenate(pt) - np.concatenate(tt))**2))), best_val


def main():
    print(f"V14 Phase 2e: full-sequence on FD004 (hardest subset)")
    t0 = time.time()
    data = load_cmapss_subset('FD004')
    print(f"FD004: {len(data['train_engines'])} train, "
          f"{len(data['val_engines'])} val, {len(data['test_engines'])} test")

    ckpt = V14_DIR / 'best_pretrain_full_sequence_fd004.pt'
    log = V14_DIR / 'phase2e_output.log'
    model, history, best_probe = pretrain(data, ckpt, log, seed=42)

    print("\n==== FINE-TUNE @ 100% (3 seeds) ====")
    seeds = [42, 123, 456]
    res = {'pretrain_best_probe': best_probe, 'pretrain_history': history,
           'frozen': [], 'e2e': []}
    for seed in seeds:
        for mode in ['frozen', 'e2e']:
            rmse, val = run_finetune(ckpt, data, mode, seed)
            print(f"  seed={seed} {mode:6s} test={rmse:.3f} val={val:.3f}")
            res[mode].append({'seed': seed, 'test_rmse': rmse, 'val_rmse': val})
    for mode in ['frozen', 'e2e']:
        vals = [r['test_rmse'] for r in res[mode]]
        res[f'{mode}_mean'] = float(np.mean(vals))
        res[f'{mode}_std'] = float(np.std(vals))

    res['v2_fd004'] = {'frozen_mean': 29.35, 'e2e_mean': 25.62}  # from paper tab:multisubset
    res['star_fd004'] = 14.25
    res['wall_time_s'] = time.time() - t0
    print(f"\n=== FD004 SUMMARY ===")
    print(f"Full-seq frozen: {res['frozen_mean']:.3f} +/- {res['frozen_std']:.3f} "
          f"(V2: 29.35, STAR: 14.25)")
    print(f"Full-seq E2E:    {res['e2e_mean']:.3f} +/- {res['e2e_std']:.3f} "
          f"(V2: 25.62)")
    print(f"Total wall time: {(time.time()-t0)/60:.1f} min")

    with open(V14_DIR / 'phase2e_fd004_results.json', 'w') as f:
        json.dump(res, f, indent=2, default=float)


if __name__ == '__main__':
    main()
