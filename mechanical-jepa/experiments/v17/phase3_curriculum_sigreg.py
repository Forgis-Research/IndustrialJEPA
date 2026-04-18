"""
V17 Phase 3: Curriculum EMA -> SIGReg.

Starting from Phase 1 epoch-100 checkpoint, continue training with graduated schedule:

  Epochs 100-150 (50 epochs):
    EMA (0.99) + SIGReg lambda ramps linearly 0 -> 0.05
  Epochs 150-200 (50 epochs):
    No EMA - target = context_encoder(future).detach() (stop-grad)
    SIGReg lambda = 0.05

Two SIGReg placement variants:
  (a) after encoder  - SIGReg(h_past)
  (b) after predictor - SIGReg(gamma(k) = predictor(h, k))

3 seeds per variant, FD001.
Track: frozen probe RMSE + F1 + PC1 explained variance every 10 epochs.

Success: matches Phase 1 frozen probe quality without target network at the end.
"""

import sys, math, json, copy, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA

V11 = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/experiments/v11')
V15 = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/experiments/v15')
V17 = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/experiments/v17')
sys.path.insert(0, str(V11))
sys.path.insert(0, str(V15))
sys.path.insert(0, str(V17))
sys.path.insert(0, '/home/sagemaker-user/IndustrialJEPA/mechanical-jepa')

from models import TrajectoryJEPA
from data_utils import (load_cmapss_subset, N_SENSORS, RUL_CAP,
                         CMAPSSFinetuneDataset, CMAPSSTestDataset,
                         collate_finetune, collate_test)
from phase1_sigreg import SIGRegEP
from phase1_v17_baseline import (
    V17PretrainDataset, collate_v17_pretrain, linear_probe_rmse_and_f1,
    v17_loss, D_MODEL, N_HEADS, N_LAYERS, D_FF, BATCH_SIZE, LR,
    WEIGHT_DECAY, EMA_MOMENTUM, K_MAX, W_WIN, MIN_PAST, N_CUTS,
    PROBE_EVERY, K_EVAL_F1,
)
from evaluation.grey_swan_metrics import anomaly_metrics as _anomaly_metrics

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CURRICULUM_START = 100   # load this Phase 1 ckpt
EMA_END = 150            # switch to no-EMA at this epoch
N_EPOCHS = 200
LAMBDA_SIG_MAX = 0.05
SIGREG_M = 512
SEEDS = [42, 123, 456]
CKPT_DIR = V17 / 'ckpts'


def pc1_explained_var(model, engines, device=DEVICE, max_samples=500):
    """Compute PC1 explained variance of h_past on a random sample."""
    model.eval()
    from data_utils import CMAPSSFinetuneDataset, collate_finetune
    ds = CMAPSSFinetuneDataset(engines, n_cuts_per_engine=5, seed=0)
    loader = DataLoader(ds, batch_size=32, shuffle=False, collate_fn=collate_finetune)
    all_h = []
    n = 0
    with torch.no_grad():
        for past, mask, rul in loader:
            past, mask = past.to(device), mask.to(device)
            h = model.encode_past(past, mask)
            all_h.append(h.cpu().numpy())
            n += h.shape[0]
            if n >= max_samples:
                break
    if not all_h:
        return float('nan')
    H = np.vstack(all_h)
    if len(H) < 10:
        return float('nan')
    pca = PCA(n_components=min(5, H.shape[1]))
    pca.fit(H)
    return float(pca.explained_variance_ratio_[0])


def run_curriculum(seed, variant, data, verbose=True):
    """
    variant in {'enc', 'pred'} :
      'enc' -> SIGReg on h_past
      'pred'-> SIGReg on predictor output (gamma)
    """
    torch.manual_seed(seed); np.random.seed(seed)

    # ---- Build model, load phase1 ep100 checkpoint ----
    model = TrajectoryJEPA(
        n_sensors=N_SENSORS, patch_length=1,
        d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS, d_ff=D_FF,
        dropout=0.1, ema_momentum=EMA_MOMENTUM, predictor_hidden=D_FF,
    ).to(DEVICE)
    ck = CKPT_DIR / f'v17_seed{seed}_ep100.pt'
    if not ck.exists():
        raise FileNotFoundError(f"Missing phase1 ep100 ckpt: {ck}")
    model.load_state_dict(torch.load(ck, map_location=DEVICE))
    print(f"  [s{seed} {variant}] loaded phase1 ep100 ckpt", flush=True)

    # SIGReg module
    sigreg = SIGRegEP(embed_dim=D_MODEL, n_projections=SIGREG_M).to(DEVICE)

    # Optimizer: fresh, same LR/cosine scheduled only over remaining 100 epochs
    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR * 0.3, weight_decay=WEIGHT_DECAY,  # reduced LR for continuation
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, N_EPOCHS - CURRICULUM_START)

    history = {
        'epoch': [], 'loss': [], 'l_pred': [], 'l_sig': [], 'lambda_sig': [],
        'mode': [],
        'probe_epochs': [], 'val_rmse': [], 'test_rmse': [], 'test_f1': [],
        'pc1_var': [],
    }
    best_val = float('inf')
    best_state = None

    t0 = time.time()
    for epoch in range(CURRICULUM_START + 1, N_EPOCHS + 1):
        # Schedule
        if epoch <= EMA_END:
            frac = (epoch - CURRICULUM_START) / (EMA_END - CURRICULUM_START)
            lam_sig = LAMBDA_SIG_MAX * frac
            use_ema = True
        else:
            lam_sig = LAMBDA_SIG_MAX
            use_ema = False

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
                # Standard V2 forward: targets from EMA target_encoder
                pred, targ, h_past = model.forward_pretrain(
                    x_past, past_mask, x_fut, fut_mask, k,
                )
            else:
                # Stop-grad on SAME encoder for target
                h_past = model.context_encoder(x_past, past_mask)
                with torch.no_grad():
                    # Use context_encoder causally on fut window, take last token
                    targ = model.context_encoder(x_fut, fut_mask)
                pred = model.predictor(h_past, k)

            # Prediction loss (L1 on normalized, + var-reg on pred)
            l_total_pred, l_pred, _ = v17_loss(pred, targ, lambda_var=0.04)

            # SIGReg target
            if variant == 'enc':
                l_sig = sigreg(h_past)
            else:  # 'pred'
                l_sig = sigreg(pred)

            loss = l_total_pred + lam_sig * l_sig
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            if use_ema:
                model.update_ema()

            B = x_past.shape[0]
            tot += loss.item() * B; lpred_acc += l_pred.item() * B
            lsig_acc += l_sig.item() * B; n += B

        avg_loss = tot / n
        history['epoch'].append(epoch)
        history['loss'].append(avg_loss)
        history['l_pred'].append(lpred_acc / n)
        history['l_sig'].append(lsig_acc / n)
        history['lambda_sig'].append(lam_sig)
        history['mode'].append('ema' if use_ema else 'sg')
        sched.step()

        extra = ''
        if epoch % PROBE_EVERY == 0 or epoch == N_EPOCHS:
            metrics = linear_probe_rmse_and_f1(
                model, data['train_engines'], data['val_engines'],
                data['test_engines'], data['test_rul'], seed=seed,
            )
            pc1 = pc1_explained_var(model, data['val_engines'])
            history['probe_epochs'].append(epoch)
            history['val_rmse'].append(metrics['val_rmse'])
            history['test_rmse'].append(metrics['test_rmse'])
            history['test_f1'].append(metrics['test_f1'])
            history['pc1_var'].append(pc1)
            if metrics['val_rmse'] < best_val:
                best_val = metrics['val_rmse']
                best_state = copy.deepcopy(model.state_dict())
            extra = (f" | probe_val={metrics['val_rmse']:.2f} "
                     f"test={metrics['test_rmse']:.2f} "
                     f"F1={metrics['test_f1']:.3f} PC1={pc1:.3f}")

        if verbose:
            print(f"  Ep {epoch:3d} [{'EMA' if use_ema else 'SG '}] "
                  f"lam_sig={lam_sig:.4f} L={avg_loss:.4f} "
                  f"l_pred={lpred_acc/n:.4f} l_sig={lsig_acc/n:.4f}{extra}",
                  flush=True)

    elapsed = (time.time() - t0) / 60
    print(f"  [s{seed} {variant}] done in {elapsed:.1f} min, best_val={best_val:.2f}",
          flush=True)

    # Final metrics at best state
    if best_state is not None:
        model.load_state_dict(best_state)
    final = linear_probe_rmse_and_f1(
        model, data['train_engines'], data['val_engines'],
        data['test_engines'], data['test_rul'], seed=seed,
    )
    # Save ckpt for later analysis
    torch.save(model.state_dict(),
               CKPT_DIR / f'v17_phase3_{variant}_seed{seed}_best.pt')

    return {
        'seed': seed, 'variant': variant,
        'best_val_rmse': float(best_val),
        'final': final,
        'history': history,
        'elapsed_min': elapsed,
    }


def main():
    data = load_cmapss_subset('FD001')
    all_results = []
    t0 = time.time()
    for variant in ['enc', 'pred']:
        print(f"\n{'=' * 60}\nVariant: SIGReg after {variant}\n{'=' * 60}",
              flush=True)
        for seed in SEEDS:
            r = run_curriculum(seed, variant, data)
            all_results.append(r)
            # Save intermediate
            out = {
                'config': 'v17_phase3_curriculum',
                'seeds': SEEDS,
                'curriculum_start_ep': CURRICULUM_START,
                'ema_end_ep': EMA_END, 'n_epochs': N_EPOCHS,
                'lambda_sig_max': LAMBDA_SIG_MAX,
                'per_run': [
                    {k: v for k, v in rr.items() if k != 'history'}
                    for rr in all_results
                ],
            }
            with open(V17 / 'phase3_curriculum_results.json', 'w') as f:
                json.dump(out, f, indent=2, default=float)

    # Aggregate per variant
    summary = {'config': 'v17_phase3_curriculum', 'by_variant': {}}
    for variant in ['enc', 'pred']:
        rs = [r for r in all_results if r['variant'] == variant]
        if not rs:
            continue
        summary['by_variant'][variant] = {
            'n_seeds': len(rs),
            'val_rmse_per_seed': [r['best_val_rmse'] for r in rs],
            'test_rmse_per_seed': [r['final']['test_rmse'] for r in rs],
            'test_rmse_mean': float(np.mean([r['final']['test_rmse'] for r in rs])),
            'test_rmse_std': float(np.std([r['final']['test_rmse'] for r in rs])),
            'val_rmse_mean': float(np.mean([r['best_val_rmse'] for r in rs])),
            'f1_mean': float(np.mean([r['final']['test_f1'] for r in rs])),
            'auc_pr_mean': float(np.mean([r['final']['test_auc_pr'] for r in rs])),
            'pc1_final_mean': float(np.mean([r['history']['pc1_var'][-1]
                                              for r in rs if r['history']['pc1_var']])),
        }

    summary['v2_baseline_rmse'] = 17.81
    summary['histories'] = [r['history'] for r in all_results]
    summary['runtime_hours'] = (time.time() - t0) / 3600

    with open(V17 / 'phase3_curriculum_results.json', 'w') as f:
        json.dump(summary, f, indent=2, default=float)

    print("\n" + "=" * 60)
    print("V17 Phase 3 Curriculum SIGReg SUMMARY")
    print("=" * 60)
    print(f"{'variant':<8} {'val_rmse':>10} {'test_rmse':>10} {'F1@30':>8} {'PC1':>6}")
    for v, s in summary['by_variant'].items():
        print(f"{v:<8} {s['val_rmse_mean']:>10.2f} {s['test_rmse_mean']:>10.2f} "
              f"{s['f1_mean']:>8.3f} {s['pc1_final_mean']:>6.3f}")
    print(f"Runtime: {summary['runtime_hours']:.2f}h")


if __name__ == '__main__':
    main()
