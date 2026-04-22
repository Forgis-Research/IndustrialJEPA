"""
V16 Supplementary: V15-SIGReg Re-run WITH Checkpoint Saving.

V15-SIGReg seed 42 achieved best_probe=10.21 (epoch 110) but no checkpoint
was saved to disk. This script re-runs with checkpoint saving to:
1. Verify the 10.21 result is reproducible
2. Save the checkpoint for cross-machine evaluation
3. Verify the saved checkpoint actually achieves the claimed RMSE

Architecture: V15-SIGReg (bidirectional shared encoder, EP-SIGReg regularizer)
  - Same as V15 phase1_sigreg.py but WITH disk checkpointing

Output:
  experiments/v16/v15sigreg_best_seed42.pt   (best checkpoint)
  experiments/v16/v15sigreg_seed42_results.json

Critical question: Does seed 42 reproducibly achieve probe < 15 over 3 runs?
This will determine if the 10.21 result is reliable or a lucky seed.
"""

import sys, json, time, copy, math
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

V11_DIR = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/experiments/v11')
V15_DIR = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/experiments/v15')
V16_DIR = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/experiments/v16')
sys.path.insert(0, str(V11_DIR))
sys.path.insert(0, str(V15_DIR))

from data_utils import (
    load_cmapss_subset, N_SENSORS, RUL_CAP,
    CMAPSSFinetuneDataset, CMAPSSTestDataset,
    collate_finetune, collate_test,
)
from phase1_sigreg import (
    V15JEPA, V15PretrainDataset, collate_v15_pretrain,
    eval_probe_rmse, get_v15_encoder_fn, D_MODEL, N_HEADS, N_LAYERS,
    LAMBDA_SIG, EMA_MOMENTUM, M_SLICES, N_CUTS, BATCH_SIZE, LR,
)

try:
    import wandb
    HAS_WANDB = True
except Exception:
    HAS_WANDB = False

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
V16_DIR.mkdir(parents=True, exist_ok=True)

N_EPOCHS = 200
PATIENCE = 20
PROBE_EVERY = 10
# Run seed 42 only (the one that achieved 10.21) to verify + save checkpoint
SEEDS = [42]


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def pretrain_v15sigreg_with_ckpt(data, seed=42, ckpt_path=None):
    """
    Identical to V15 phase1 SIGReg training but saves best checkpoint to disk.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = V15JEPA(
        n_sensors=N_SENSORS, d_model=D_MODEL, n_heads=N_HEADS,
        n_layers=N_LAYERS, mode='sigreg',
        lambda_sig=LAMBDA_SIG, lambda_var=0.04,
        ema_momentum=EMA_MOMENTUM, sigreg_m=M_SLICES,
    ).to(DEVICE)
    n_params = count_params(model)
    print(f"  [v15_sigreg_ckpt] params={n_params:,}")

    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, N_EPOCHS)

    history = {'loss': [], 'probe_rmse': [], 'probe_epochs': [], 'seed': seed}
    best_probe = float('inf')
    best_epoch = -1
    no_impr = 0

    run = None
    if HAS_WANDB:
        try:
            run = wandb.init(project='industrialjepa',
                             name=f'v16-v15sigreg-ckpt-s{seed}',
                             tags=['v16', 'v15sigreg', 'checkpoint_verify'],
                             config={'seed': seed, 'n_epochs': N_EPOCHS,
                                     'architecture': 'v15_sigreg_with_checkpoint'},
                             reinit=True)
        except Exception as e:
            print(f"  wandb init failed: {e}")

    t0 = time.time()
    for epoch in range(1, N_EPOCHS + 1):
        ds = V15PretrainDataset(data['train_engines'], n_cuts_per_engine=N_CUTS,
                                 min_past=10, min_horizon=5, max_horizon=30,
                                 seed=epoch + seed)
        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True,
                            collate_fn=collate_v15_pretrain, num_workers=0)

        model.train()
        total_loss = 0.0
        nbatch = 0
        for x_past, past_mask, x_full, full_mask, k in loader:
            x_past, past_mask = x_past.to(DEVICE), past_mask.to(DEVICE)
            x_full, full_mask = x_full.to(DEVICE), full_mask.to(DEVICE)
            k = k.to(DEVICE)

            optim.zero_grad()
            loss, h_t, h_tk = model.forward_pretrain(x_past, past_mask, x_full, full_mask, k)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            total_loss += loss.item() * x_past.shape[0]
            nbatch += x_past.shape[0]

        avg_loss = total_loss / nbatch
        history['loss'].append(avg_loss)
        sched.step()

        extra = ''
        if epoch % PROBE_EVERY == 0 or epoch == 1:
            model.eval()
            enc_fn = get_v15_encoder_fn(model)
            probe_rmse = eval_probe_rmse(enc_fn, data['train_engines'],
                                          data['val_engines'])
            history['probe_rmse'].append(probe_rmse)
            history['probe_epochs'].append(epoch)

            if probe_rmse < best_probe:
                best_probe = probe_rmse
                best_epoch = epoch
                no_impr = 0
                # Save checkpoint to disk (the key difference from V15!)
                if ckpt_path is not None:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'probe_rmse': probe_rmse,
                        'seed': seed,
                    }, ckpt_path)
                    print(f"  ** Checkpoint saved at epoch {epoch} (probe={probe_rmse:.2f})")
            else:
                no_impr += 1

            extra = f" | probe={probe_rmse:.2f} (best={best_probe:.2f}@ep{best_epoch})"

            if run is not None:
                run.log({'epoch': epoch, 'train_loss': avg_loss,
                         'probe_rmse': probe_rmse, 'best_probe': best_probe})

        print(f"  Ep {epoch:3d} | loss={avg_loss:.4f}{extra}", flush=True)

        if no_impr >= PATIENCE and epoch > 50:
            print(f"  Early stop at epoch {epoch}")
            break

    if run is not None:
        run.finish()

    elapsed = (time.time() - t0) / 60
    print(f"  done in {elapsed:.1f} min, best_probe={best_probe:.2f} at epoch {best_epoch}")
    return model, history, best_probe, best_epoch


def verify_checkpoint(ckpt_path, data, d_model=D_MODEL):
    """
    Reload checkpoint and re-run probe evaluation to verify claimed RMSE.
    This catches any probe training randomness that might inflate/deflate results.
    """
    print(f"\n=== Checkpoint Verification ===")
    if not Path(ckpt_path).exists():
        print(f"  Checkpoint not found: {ckpt_path}")
        return None

    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    epoch = ckpt['epoch']
    saved_probe = ckpt['probe_rmse']
    seed = ckpt['seed']
    print(f"  Checkpoint: epoch={epoch}, saved_probe={saved_probe:.2f}, seed={seed}")

    model = V15JEPA(
        n_sensors=N_SENSORS, d_model=D_MODEL, n_heads=N_HEADS,
        n_layers=N_LAYERS, mode='sigreg',
        lambda_sig=LAMBDA_SIG, lambda_var=0.04,
        ema_momentum=EMA_MOMENTUM, sigreg_m=M_SLICES,
    ).to(DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    # Re-run probe 5 times with different seeds to get stable estimate
    probe_rmses = []
    for probe_seed in [42, 123, 456, 789, 1234]:
        enc_fn = get_v15_encoder_fn(model)
        rmse = eval_probe_rmse(enc_fn, data['train_engines'], data['val_engines'],
                                seed=probe_seed)
        probe_rmses.append(rmse)
        print(f"    Probe seed {probe_seed}: RMSE={rmse:.2f}")

    probe_mean = float(np.mean(probe_rmses))
    probe_std = float(np.std(probe_rmses))
    print(f"\n  Verified probe RMSE: {probe_mean:.2f} +/- {probe_std:.2f}")
    print(f"  Claimed probe RMSE:  {saved_probe:.2f}")

    consistent = abs(probe_mean - saved_probe) < 3.0  # within 3 cycles
    if consistent:
        print(f"  CONSISTENT (within 3 cycles) - result is valid")
    else:
        print(f"  INCONSISTENT (diff={abs(probe_mean - saved_probe):.2f}) - investigate!")

    return {
        'epoch': epoch,
        'saved_probe': saved_probe,
        'verified_probe_mean': probe_mean,
        'verified_probe_std': probe_std,
        'probe_rmses': probe_rmses,
        'consistent': consistent,
    }


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings('ignore')

    print("=" * 60)
    print("V15-SIGReg Re-run WITH Checkpoint Saving (Verification)")
    print("V15 seed 42 claimed probe=10.21 at epoch 110")
    print("Goal: reproduce + save checkpoint for cross-machine eval")
    print("=" * 60)

    data = load_cmapss_subset('FD001')

    results = {
        'description': 'V15-SIGReg with checkpoint saving for verification',
        'v15_original_best': {'seed': 42, 'probe': 10.21, 'epoch': 110},
        'seeds': SEEDS,
    }

    all_probes = []
    for seed in SEEDS:
        print(f"\n--- Seed {seed} ---")
        ckpt_path = str(V16_DIR / f'v15sigreg_best_seed{seed}.pt')
        model, history, best_probe, best_epoch = pretrain_v15sigreg_with_ckpt(
            data, seed=seed, ckpt_path=ckpt_path)

        all_probes.append(best_probe)

        # Verify checkpoint
        verify_result = verify_checkpoint(ckpt_path, data)

        results[f'seed_{seed}'] = {
            'best_probe': best_probe,
            'best_epoch': best_epoch,
            'loss_history': history['loss'],
            'probe_history': history['probe_rmse'],
            'probe_epochs': history['probe_epochs'],
            'verification': verify_result,
        }

        with open(V16_DIR / 'v15sigreg_seed42_results.json', 'w') as f:
            json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print("V15-SIGReg Re-run Summary")
    print("=" * 60)
    for seed, probe in zip(SEEDS, all_probes):
        print(f"  Seed {seed}: best_probe={probe:.2f}")
    print(f"  V15 original: 10.21 (seed 42)")
    if all_probes[0] < 15:
        print(f"  REPRODUCED: {all_probes[0]:.2f} < 15 target")
    else:
        print(f"  NOT REPRODUCED: {all_probes[0]:.2f} >= 15")

    with open(V16_DIR / 'v15sigreg_seed42_results.json', 'w') as f:
        json.dump(results, f, indent=2)
