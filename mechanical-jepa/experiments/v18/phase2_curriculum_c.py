"""
V18 Phase 2: Accelerated curriculum SIGReg - Schedule C (gradual EMA fade).

The v17 Phase 3 curriculum had a hard EMA-off cutoff at epoch 150 that caused a
4x loss spike. Schedule C replaces the binary cutoff with a gradual EMA momentum
fade from 0.99 to 1.0 over 20 epochs, so the target network smoothly becomes
frozen (equivalent to stop-grad on stale weights) before the full SG-only phase.

Starting from v17_seed{S}_ep100.pt (same as Schedule A):

  Ep 100-120: EMA (momentum=0.99), SIGReg lambda ramp 0 -> 0.05  (linear)
  Ep 120-140: EMA (momentum fades 0.99 -> 1.0), lambda = 0.05
  Ep 140-150: no EMA (stop-grad on live encoder), lambda = 0.05

SIGReg placement: predictor only (v17 showed encoder placement destroys RUL).

3 seeds (42, 123, 456). Tracks loss trajectory for comparison with Schedule A.

Output:
  - experiments/v18/phase2_curriculum_c_results.json
  - ckpts/v18_phase2c_seed{S}_best.pt
"""

import sys, math, json, copy, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

V11 = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/experiments/v11')
V15 = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/experiments/v15')
V17 = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/experiments/v17')
V18 = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/experiments/v18')
ROOT = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa')
for p in [V11, V15, V17, ROOT]:
    sys.path.insert(0, str(p))

from models import TrajectoryJEPA
from data_utils import (load_cmapss_subset, N_SENSORS, RUL_CAP,
                        CMAPSSFinetuneDataset, CMAPSSTestDataset,
                        collate_finetune, collate_test)
from phase1_sigreg import SIGRegEP
from phase1_v17_baseline import (
    V17PretrainDataset, collate_v17_pretrain,
    v17_loss, D_MODEL, N_HEADS, N_LAYERS, D_FF, BATCH_SIZE,
    LR, WEIGHT_DECAY, EMA_MOMENTUM, K_MAX, W_WIN, MIN_PAST, N_CUTS,
)
from evaluation.grey_swan_metrics import anomaly_metrics as _anomaly_metrics

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CURRICULUM_START = 100
RAMP_END = 120   # SIGReg lam ramp end, EMA fade start
FADE_END = 140   # EMA fade end (momentum hits 1.0)
N_EPOCHS = 150   # earlier than 200
LAMBDA_SIG_MAX = 0.05
SIGREG_M = 512
SEEDS = [42, 123, 456]
CKPT_DIR = V18 / 'ckpts'
CKPT_DIR.mkdir(parents=True, exist_ok=True)
PROBE_EVERY = 5
K_EVAL_LIST = [10, 20, 30, 50]


def honest_probe_metrics(model, data, seed):
    import torch.nn as nn
    torch.manual_seed(seed)
    probe = nn.Sequential(nn.Linear(D_MODEL, 1), nn.Sigmoid()).to(DEVICE)
    opt = torch.optim.AdamW(probe.parameters(), lr=1e-3, weight_decay=1e-2)

    tr_ds = CMAPSSFinetuneDataset(data['train_engines'], n_cuts_per_engine=5, seed=seed)
    va_ds = CMAPSSFinetuneDataset(data['val_engines'], n_cuts_per_engine=10, seed=seed + 111)
    te_ds = CMAPSSTestDataset(data['test_engines'], data['test_rul'])
    tr = DataLoader(tr_ds, batch_size=32, shuffle=True, collate_fn=collate_finetune)
    va = DataLoader(va_ds, batch_size=32, shuffle=False, collate_fn=collate_finetune)
    te = DataLoader(te_ds, batch_size=32, shuffle=False, collate_fn=collate_test)

    best_val = float('inf'); best_state = None; no_impr = 0
    for ep in range(100):  # fewer epochs for probe during tracking
        probe.train()
        for past, mask, rul in tr:
            past, mask, rul = past.to(DEVICE), mask.to(DEVICE), rul.to(DEVICE)
            with torch.no_grad(): h = model.encode_past(past, mask)
            loss = F.mse_loss(probe(h).squeeze(-1), rul)
            opt.zero_grad(); loss.backward(); opt.step()

        probe.eval()
        pv, tv = [], []
        with torch.no_grad():
            for past, mask, rul in va:
                past, mask = past.to(DEVICE), mask.to(DEVICE)
                h = model.encode_past(past, mask)
                pv.append(probe(h).squeeze(-1).cpu().numpy()); tv.append(rul.numpy())
        preds = np.concatenate(pv) * RUL_CAP; targs = np.concatenate(tv) * RUL_CAP
        val_rmse = float(np.sqrt(np.mean((preds - targs) ** 2)))
        if val_rmse < best_val:
            best_val = val_rmse; best_state = copy.deepcopy(probe.state_dict()); no_impr = 0
        else:
            no_impr += 1
            if no_impr >= 15: break
    probe.load_state_dict(best_state)

    p_test, t_test = [], []
    with torch.no_grad():
        for past, mask, rul_gt in te:
            past, mask = past.to(DEVICE), mask.to(DEVICE)
            h = model.encode_past(past, mask)
            p_test.append(probe(h).squeeze(-1).cpu().numpy() * RUL_CAP)
            t_test.append(rul_gt.numpy())
    pt = np.concatenate(p_test); tt = np.concatenate(t_test)
    test_rmse = float(np.sqrt(np.mean((pt - tt) ** 2)))

    f1_by_k = {}
    for ke in K_EVAL_LIST:
        y = (tt <= ke).astype(int); score = -pt
        thr = float(np.percentile(score[y == 0], 95)) if (y == 0).sum() > 0 else 0.0
        m = _anomaly_metrics(score, y, threshold=thr)
        f1_by_k[ke] = {'f1': float(m['f1_non_pa']),
                       'auc_pr': float(m['auc_pr'])}
    return {'val_rmse': best_val, 'test_rmse': test_rmse, 'f1_by_k': f1_by_k}


def schedule_c(epoch):
    """Returns (lam_sig, use_ema, ema_momentum) for epoch (in 101..N_EPOCHS)."""
    if epoch <= RAMP_END:
        # 101-120: ramp SIGReg 0 -> MAX, EMA on
        frac = (epoch - CURRICULUM_START) / (RAMP_END - CURRICULUM_START)
        return LAMBDA_SIG_MAX * frac, True, EMA_MOMENTUM
    if epoch <= FADE_END:
        # 121-140: EMA momentum fade 0.99 -> 1.0
        frac = (epoch - RAMP_END) / (FADE_END - RAMP_END)
        momentum = EMA_MOMENTUM + (1.0 - EMA_MOMENTUM) * frac
        return LAMBDA_SIG_MAX, True, momentum
    # 141-150: SG-only
    return LAMBDA_SIG_MAX, False, 1.0


def run_schedule_c(seed, data, verbose=True):
    torch.manual_seed(seed); np.random.seed(seed)

    model = TrajectoryJEPA(
        n_sensors=N_SENSORS, patch_length=1,
        d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS, d_ff=D_FF,
        dropout=0.1, ema_momentum=EMA_MOMENTUM, predictor_hidden=D_FF,
    ).to(DEVICE)
    ck = V17 / 'ckpts' / f'v17_seed{seed}_ep100.pt'
    if not ck.exists():
        raise FileNotFoundError(ck)
    model.load_state_dict(torch.load(ck, map_location=DEVICE, weights_only=False))

    sigreg = SIGRegEP(embed_dim=D_MODEL, n_projections=SIGREG_M).to(DEVICE)
    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR * 0.3, weight_decay=WEIGHT_DECAY,
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, N_EPOCHS - CURRICULUM_START)

    history = {'epoch': [], 'loss': [], 'l_pred': [], 'l_sig': [],
               'lambda_sig': [], 'ema_mom': [], 'mode': [],
               'probe_epochs': [], 'val_rmse': [], 'test_rmse': []}
    best_val = float('inf'); best_state = None; t0 = time.time()

    for epoch in range(CURRICULUM_START + 1, N_EPOCHS + 1):
        lam_sig, use_ema, ema_mom = schedule_c(epoch)
        model.ema_momentum = ema_mom  # dynamic

        ds = V17PretrainDataset(
            data['train_engines'], n_cuts=N_CUTS, min_past=MIN_PAST,
            K_max=K_MAX, w=W_WIN, seed=seed * 1000 + epoch,
        )
        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True,
                            collate_fn=collate_v17_pretrain, num_workers=0)
        model.train()
        tot = lpred_acc = lsig_acc = 0.0; n = 0
        for x_past, past_mask, x_fut, fut_mask, k, _ in loader:
            x_past, past_mask = x_past.to(DEVICE), past_mask.to(DEVICE)
            x_fut, fut_mask = x_fut.to(DEVICE), fut_mask.to(DEVICE)
            k = k.to(DEVICE)
            optim.zero_grad()
            if use_ema:
                pred, targ, h_past = model.forward_pretrain(
                    x_past, past_mask, x_fut, fut_mask, k)
            else:
                h_past = model.context_encoder(x_past, past_mask)
                with torch.no_grad():
                    targ = model.context_encoder(x_fut, fut_mask)
                pred = model.predictor(h_past, k)
            l_total, l_pred, _ = v17_loss(pred, targ, lambda_var=0.04)
            l_sig = sigreg(pred)
            loss = l_total + lam_sig * l_sig
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            if use_ema:
                model.update_ema()
            B = x_past.shape[0]
            tot += loss.item() * B; lpred_acc += l_pred.item() * B
            lsig_acc += l_sig.item() * B; n += B

        avg_loss = tot / n
        history['epoch'].append(epoch); history['loss'].append(avg_loss)
        history['l_pred'].append(lpred_acc / n); history['l_sig'].append(lsig_acc / n)
        history['lambda_sig'].append(lam_sig); history['ema_mom'].append(ema_mom)
        history['mode'].append('ema' if use_ema else 'sg')
        sched.step()

        extra = ''
        if epoch % PROBE_EVERY == 0 or epoch == N_EPOCHS:
            m = honest_probe_metrics(model, data, seed)
            history['probe_epochs'].append(epoch)
            history['val_rmse'].append(m['val_rmse'])
            history['test_rmse'].append(m['test_rmse'])
            if m['val_rmse'] < best_val:
                best_val = m['val_rmse']
                best_state = copy.deepcopy(model.state_dict())
                torch.save(best_state, CKPT_DIR / f'v18_phase2c_seed{seed}_best.pt')
            extra = (f" | probe_val={m['val_rmse']:.2f} "
                     f"test={m['test_rmse']:.2f} F1@30={m['f1_by_k'][30]['f1']:.3f}")

        if verbose:
            print(f"  Ep {epoch:3d} [{'EMA' if use_ema else 'SG'} m={ema_mom:.3f}] "
                  f"lam={lam_sig:.3f} L={avg_loss:.4f} "
                  f"l_pred={lpred_acc/n:.4f}{extra}", flush=True)

    elapsed = (time.time() - t0) / 60
    print(f"  [s{seed}] done {elapsed:.1f}min, best_val={best_val:.2f}", flush=True)

    if best_state is not None:
        model.load_state_dict(best_state)
    final = honest_probe_metrics(model, data, seed)
    return {'seed': seed, 'best_val_rmse': best_val, 'final': final,
            'history': history, 'elapsed_min': elapsed}


def main():
    V18.mkdir(exist_ok=True)
    data = load_cmapss_subset('FD001')
    t0 = time.time()
    all_results = []
    for seed in SEEDS:
        print(f"\n=== Schedule C seed {seed} ===", flush=True)
        r = run_schedule_c(seed, data)
        all_results.append(r)
        with open(V18 / 'phase2_curriculum_c_results.json', 'w') as f:
            json.dump({'config': 'v18_phase2_schedule_C',
                       'curriculum_start': CURRICULUM_START,
                       'ramp_end': RAMP_END, 'fade_end': FADE_END,
                       'n_epochs': N_EPOCHS,
                       'lambda_sig_max': LAMBDA_SIG_MAX,
                       'sigreg_on': 'predictor',
                       'ema_fade': f'{EMA_MOMENTUM} -> 1.0 over {FADE_END-RAMP_END} epochs',
                       'seeds': SEEDS,
                       'per_run': all_results,
                       'runtime_hours': (time.time() - t0) / 3600}, f,
                      indent=2, default=float)

    rmse_final = [r['final']['test_rmse'] for r in all_results]
    val_final = [r['best_val_rmse'] for r in all_results]
    print(f"\nSchedule C final test_rmse: {np.mean(rmse_final):.2f} +/- {np.std(rmse_final):.2f}")
    print(f"  (ref v17 schedule A pred: 15.38 at 200 ep)")
    print(f"  val_rmse: {np.mean(val_final):.2f} +/- {np.std(val_final):.2f}")


if __name__ == '__main__':
    main()
