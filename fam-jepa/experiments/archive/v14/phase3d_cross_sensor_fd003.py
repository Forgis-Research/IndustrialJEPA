"""
V14 Phase 3d: Cross-sensor attention on FD003 (2 fault modes: fan + HPC).

Tests whether the physics-aligned attention pattern (s*→s14 during
degradation) generalizes to a different fault-mode subset.

Protocol: in-domain pretrain on FD003 (not FD001), fine-tune frozen
3 seeds @ 100% labels, extract attention maps (healthy vs degradation
averaged across engines). Compare attention shifts to FD001 (Phase 3).

Output: experiments/v14/phase3d_fd003_results.json
        experiments/v14/phase3d_fd003_attention_maps.json
        analysis/plots/v14/cross_sensor_fd003_attention_*.png
"""

import sys, json, time, copy
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

V11_DIR = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/experiments/v11')
V14_DIR = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/experiments/v14')
PLOT_PNG = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/analysis/plots/v14')
sys.path.insert(0, str(V11_DIR))
sys.path.insert(0, str(V14_DIR))

from data_utils import (
    load_cmapss_subset, N_SENSORS, RUL_CAP,
    CMAPSSPretrainDataset, CMAPSSFinetuneDataset, CMAPSSTestDataset,
    collate_pretrain, collate_finetune, collate_test,
)
from models import RULProbe, trajectory_jepa_loss
from phase3_cross_sensor import (
    CrossSensorJEPA, D_MODEL, N_HEADS, N_PAIRS,
    extract_attention_maps, plot_attention_maps, eval_probe_rmse,
)

try:
    import wandb
    HAS_WANDB = True
except Exception:
    HAS_WANDB = False

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

N_EPOCHS = 120
BATCH_SIZE = 4
N_CUTS = 20
LAMBDA_VAR = 0.01
PROBE_EVERY = 5
PATIENCE_PROBE = 8


def pretrain(data, ckpt_path, log_path, seed=42):
    torch.manual_seed(seed); np.random.seed(seed)
    model = CrossSensorJEPA(d_model=D_MODEL, n_heads=N_HEADS, n_pairs=N_PAIRS,
                             dropout=0.1, ema_momentum=0.99).to(DEVICE)
    optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                              lr=3e-4, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, N_EPOCHS)
    history = {'loss': [], 'probe_rmse': [], 'probe_epochs': []}
    best = float('inf'); best_state = None; no_improve = 0
    t0 = time.time()
    run = None
    if HAS_WANDB:
        try:
            run = wandb.init(project='industrialjepa', name='v14-phase3d-crosssensor-fd003',
                             tags=['v14-phase3d-cross-sensor-fd003'], reinit=True)
        except Exception: pass

    with open(log_path, 'w') as f: f.write("V14 Phase 3d FD003\n")
    for epoch in range(1, N_EPOCHS + 1):
        ds = CMAPSSPretrainDataset(data['train_engines'], n_cuts_per_engine=N_CUTS,
                                    min_past=10, min_horizon=5, max_horizon=30, seed=epoch)
        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_pretrain)
        model.train(); tot = 0; n = 0
        for past, past_m, fut, fut_m, k, _ in loader:
            past, past_m = past.to(DEVICE), past_m.to(DEVICE)
            fut, fut_m = fut.to(DEVICE), fut_m.to(DEVICE); k = k.to(DEVICE)
            optim.zero_grad()
            pred, h_fut, _ = model.forward_pretrain(past, past_m, fut, fut_m, k)
            loss, _, _ = trajectory_jepa_loss(pred, h_fut, LAMBDA_VAR)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step(); model.update_ema()
            tot += loss.item() * past.shape[0]; n += past.shape[0]
        history['loss'].append(tot / n); sched.step()
        extra = ''
        if epoch % PROBE_EVERY == 0 or epoch == 1:
            probe_rmse = eval_probe_rmse(model, data['train_engines'], data['val_engines'],
                                          d_model=D_MODEL)
            history['probe_rmse'].append(probe_rmse); history['probe_epochs'].append(epoch)
            if probe_rmse < best:
                best = probe_rmse; best_state = copy.deepcopy(model.state_dict())
                torch.save(best_state, ckpt_path); no_improve = 0
            else: no_improve += 1
            extra = f" | probe={probe_rmse:.2f} (best={best:.2f}, ni={no_improve})"
        line = f"Ep {epoch:3d} | loss={tot/n:.4f}{extra}"
        print(line, flush=True)
        with open(log_path, 'a') as f: f.write(line + '\n')
        if run is not None:
            try:
                d = {'epoch': epoch, 'loss': tot / n}
                if epoch % PROBE_EVERY == 0 or epoch == 1:
                    d['probe_rmse'] = probe_rmse; d['best_probe_rmse'] = best
                wandb.log(d)
            except Exception: pass
        if no_improve >= PATIENCE_PROBE:
            print(f"Early stop at {epoch}"); break
    if run is not None:
        try: wandb.finish()
        except Exception: pass
    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"Pretrain done {(time.time()-t0)/60:.1f} min, best probe={best:.2f}")
    return model, history, best


def run_frozen(ckpt_path, data, seed):
    model = CrossSensorJEPA(d_model=D_MODEL, n_heads=N_HEADS, n_pairs=N_PAIRS,
                             dropout=0.1).to(DEVICE)
    model.load_state_dict(torch.load(str(ckpt_path), map_location=DEVICE))
    for p in model.parameters(): p.requires_grad = False
    model.eval()
    probe = RULProbe(D_MODEL).to(DEVICE)
    torch.manual_seed(seed); np.random.seed(seed)
    optim = torch.optim.Adam(probe.parameters(), lr=1e-3)

    tr = DataLoader(CMAPSSFinetuneDataset(data['train_engines'], n_cuts_per_engine=5, seed=seed),
                    batch_size=16, shuffle=True, collate_fn=collate_finetune)
    va = DataLoader(CMAPSSFinetuneDataset(data['val_engines'], use_last_only=True),
                    batch_size=16, shuffle=False, collate_fn=collate_finetune)
    te = DataLoader(CMAPSSTestDataset(data['test_engines'], data['test_rul']),
                    batch_size=16, shuffle=False, collate_fn=collate_test)
    best_val = float('inf'); best_ps = None; no = 0
    for _ in range(100):
        probe.train()
        for past, mask, rul in tr:
            past, mask, rul = past.to(DEVICE), mask.to(DEVICE), rul.to(DEVICE)
            with torch.no_grad(): h = model.encode_past(past, mask)
            optim.zero_grad(); F.mse_loss(probe(h), rul).backward(); optim.step()
        probe.eval(); pv, tv = [], []
        with torch.no_grad():
            for past, mask, rul in va:
                past, mask = past.to(DEVICE), mask.to(DEVICE)
                h = model.encode_past(past, mask)
                pv.append(probe(h).cpu().numpy()); tv.append(rul.numpy())
        val_rmse = float(np.sqrt(np.mean(
            (np.concatenate(pv)*RUL_CAP - np.concatenate(tv)*RUL_CAP)**2)))
        if val_rmse < best_val:
            best_val = val_rmse; best_ps = copy.deepcopy(probe.state_dict()); no = 0
        else:
            no += 1
            if no >= 20: break
    probe.load_state_dict(best_ps); probe.eval()
    pt, tt = [], []
    with torch.no_grad():
        for past, mask, rul_gt in te:
            past, mask = past.to(DEVICE), mask.to(DEVICE)
            h = model.encode_past(past, mask)
            pt.append(probe(h).cpu().numpy() * RUL_CAP); tt.append(rul_gt.numpy())
    return float(np.sqrt(np.mean((np.concatenate(pt) - np.concatenate(tt))**2))), best_val


def main():
    print(f"V14 Phase 3d: cross-sensor on FD003 (fan + HPC faults)")
    print(f"Device: {DEVICE}")
    t0 = time.time()

    data = load_cmapss_subset('FD003')
    print(f"FD003: {len(data['train_engines'])} train, {len(data['val_engines'])} val, "
          f"{len(data['test_engines'])} test")

    ckpt_path = V14_DIR / 'best_pretrain_cross_sensor_fd003.pt'
    log_path = V14_DIR / 'phase3d_output.log'
    model, history, best_probe = pretrain(data, ckpt_path, log_path, seed=42)

    print("\n==== FROZEN FINE-TUNE @ 100% (3 seeds) ====")
    seeds = [42, 123, 456]
    results = {'pretrain_best_probe': best_probe, 'pretrain_history': history,
               'frozen_100': []}
    for seed in seeds:
        rmse, val = run_frozen(ckpt_path, data, seed)
        print(f"  seed={seed} frozen test={rmse:.3f} val={val:.3f}")
        results['frozen_100'].append({'seed': seed, 'test_rmse': rmse, 'val_rmse': val})
    vals = [r['test_rmse'] for r in results['frozen_100']]
    results['frozen_100_mean'] = float(np.mean(vals))
    results['frozen_100_std'] = float(np.std(vals))

    # Extract attention maps
    print("\n==== ATTENTION MAPS ====")
    model.load_state_dict(torch.load(str(ckpt_path), map_location=DEVICE))
    model.eval()
    maps = extract_attention_maps(model, {**data['train_engines'], **data['val_engines']})
    plot_attention_maps(maps, str(PLOT_PNG / 'cross_sensor_fd003_attention'))
    with open(V14_DIR / 'phase3d_fd003_attention_maps.json', 'w') as f:
        json.dump({k: (v.tolist() if isinstance(v, np.ndarray) else v)
                   for k, v in maps.items()}, f, indent=2)
    print("Saved attention maps")

    results['wall_time_s'] = time.time() - t0
    results['v2_fd003_frozen'] = 19.25  # from paper tab:multisubset
    print(f"\n=== SUMMARY ===")
    print(f"FD003 cross-sensor frozen: {results['frozen_100_mean']:.3f} +/- "
          f"{results['frozen_100_std']:.3f}  (V2 FD003 frozen: 19.25)")

    with open(V14_DIR / 'phase3d_fd003_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=float)
    print(f"Total wall time: {(time.time()-t0)/60:.1f} min")


if __name__ == '__main__':
    main()
