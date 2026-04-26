"""
V16b: Stable Training Fix for V16a.

V16a problem: 2/3 seeds fail to learn (init artifacts).
Seeds 123/456 converge too fast (loss ~0.004 in 10 ep vs 0.011 for seed 42).
Early convergence locks representations into non-RUL-informative minima.

V16b fixes:
  1. Stronger variance regularization (lambda_var=0.1 vs 0.04) - prevents representation collapse
  2. LR warmup (20 epochs linear warmup then cosine) - prevents premature convergence
  3. VICReg-style covariance regularization (lambda_cov=0.01) - forces diverse dimensions
  4. Slightly lower initial EMA momentum (0.996 vs 0.99) - keeps target closer to context early on
     Note: 0.99 was already low; going to 0.996 makes target SLOWER to diverge from context init

Goal: All 3 seeds should show genuine learning (not init artifact).
Target: 3-seed frozen probe mean < 12 cycles.

Comparison:
  V16a: seed42=4.75 (genuine), seeds 123/456 = init artifacts (8.53, 12.45)
  V2: frozen=17.81 +/- 1.7 (5 seeds)

Output:
  experiments/v16/phase1_v16b_results.json
  experiments/v16/best_v16b_seed{seed}.pt
"""

import sys, json, time, copy, math
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

V11_DIR = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/experiments/v11')
V16_DIR = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/experiments/v16')
sys.path.insert(0, str(V11_DIR))
# Import architectures from V16a to reuse them
sys.path.insert(0, str(V16_DIR))

from data_utils import (
    load_cmapss_subset, N_SENSORS, RUL_CAP, SELECTED_SENSORS,
    CMAPSSFinetuneDataset, CMAPSSTestDataset,
    collate_finetune, collate_test,
)
from phase1_v16a import (
    BidiContextEncoder, FutureTargetEncoder, V16aPredictor, V16aJEPA,
    V16aPretrainDataset, collate_v16a,
    eval_probe_rmse, sinusoidal_pe, horizon_pe,
    D_MODEL, N_HEADS, N_LAYERS,
)

try:
    import wandb
    HAS_WANDB = True
except Exception:
    HAS_WANDB = False

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---- Hyperparameters (changes from V16a highlighted) ----
N_CUTS = 30
BATCH_SIZE = 64
N_EPOCHS = 200
PROBE_EVERY = 10
PATIENCE = 20
EMA_MOMENTUM = 0.996       # was 0.99 - slightly higher to slow target divergence
LR = 3e-4
WARMUP_EPOCHS = 20         # NEW: linear LR warmup to prevent premature convergence
LAMBDA_VAR = 0.1           # was 0.04 - stronger variance reg to prevent collapse
LAMBDA_COV = 0.01          # NEW: covariance reg to force diverse representation dims
SEEDS = [42, 123, 456]
V16_DIR.mkdir(parents=True, exist_ok=True)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def vicreg_loss(h_hat: torch.Tensor, h_tgt: torch.Tensor,
                lambda_var: float = 0.1, lambda_cov: float = 0.01) -> torch.Tensor:
    """
    VICReg-style regularization loss on predicted embeddings.
    Prevents representational collapse via variance + covariance terms.

    h_hat: (B, D) - predicted embeddings
    h_tgt: (B, D) - target embeddings
    Returns: scalar regularization loss
    """
    B, D = h_hat.shape

    # Variance term: std of each dim should be >= 1 (penalize collapse)
    std_hat = h_hat.std(dim=0)
    std_tgt = h_tgt.std(dim=0)
    l_var = (F.relu(1.0 - std_hat).mean() + F.relu(1.0 - std_tgt).mean()) / 2

    # Covariance term: off-diagonal cov should be near 0 (decorrelation)
    h_hat_c = h_hat - h_hat.mean(dim=0)
    cov_hat = (h_hat_c.T @ h_hat_c) / (B - 1)
    # Penalize off-diagonal elements
    diag_mask = torch.eye(D, device=h_hat.device, dtype=torch.bool)
    l_cov = cov_hat[~diag_mask].pow(2).sum() / D

    return lambda_var * l_var + lambda_cov * l_cov


class WarmupCosineScheduler:
    """Linear warmup then cosine annealing."""

    def __init__(self, optimizer, warmup_epochs: int, total_epochs: int, base_lr: float):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = base_lr
        self.epoch = 0

    def step(self):
        self.epoch += 1
        if self.epoch <= self.warmup_epochs:
            lr = self.base_lr * self.epoch / self.warmup_epochs
        else:
            progress = (self.epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.base_lr * 0.5 * (1 + math.cos(math.pi * progress))
        for pg in self.optimizer.param_groups:
            pg['lr'] = lr
        return lr


def pretrain_v16b(data, seed=42, ckpt_path=None):
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = V16aJEPA(
        n_sensors=N_SENSORS, d_model=D_MODEL, n_heads=N_HEADS,
        n_layers=N_LAYERS, dropout=0.1, ema_momentum=EMA_MOMENTUM,
    ).to(DEVICE)
    n_params = count_params(model)
    print(f"  [v16b] params={n_params:,}")

    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR, weight_decay=0.01)
    sched = WarmupCosineScheduler(optim, WARMUP_EPOCHS, N_EPOCHS, LR)

    history = {'loss': [], 'probe_rmse': [], 'probe_epochs': [], 'seed': seed}
    best_probe = float('inf')
    no_impr = 0

    run = None
    if HAS_WANDB:
        try:
            run = wandb.init(project='industrialjepa',
                             name=f'v16b-s{seed}',
                             tags=['v16', 'v16b', 'stable-training', 'vicreg'],
                             config={'seed': seed, 'd_model': D_MODEL,
                                     'n_params': n_params, 'ema_momentum': EMA_MOMENTUM,
                                     'warmup_epochs': WARMUP_EPOCHS,
                                     'lambda_var': LAMBDA_VAR, 'lambda_cov': LAMBDA_COV,
                                     'architecture': 'bidi_context+causal_target+vicreg'},
                             reinit=True)
        except Exception as e:
            print(f"  wandb init failed: {e}")

    t0 = time.time()
    for epoch in range(1, N_EPOCHS + 1):
        ds = V16aPretrainDataset(data['train_engines'], n_cuts=N_CUTS,
                                  min_past=10, min_horizon=5, max_horizon=30,
                                  seed=epoch + seed)
        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True,
                            collate_fn=collate_v16a, num_workers=0)

        model.train()
        total_loss = 0.0
        nbatch = 0
        for x_past, past_mask, x_future, future_mask, k in loader:
            x_past = x_past.to(DEVICE)
            past_mask = past_mask.to(DEVICE)
            x_future = x_future.to(DEVICE)
            future_mask = future_mask.to(DEVICE)
            k = k.to(DEVICE)

            optim.zero_grad()

            # Forward pretrain (prediction loss + basic variance reg)
            h_ctx = model.context_encoder(x_past, key_padding_mask=past_mask)
            with torch.no_grad():
                h_tgt = model.target_encoder(x_future, key_padding_mask=future_mask)
            h_hat = model.predictor(h_ctx, k)

            # L1 prediction loss on normalized embeddings
            l_pred = F.l1_loss(F.normalize(h_hat, dim=-1), F.normalize(h_tgt, dim=-1))

            # VICReg regularization (stronger + covariance term)
            l_reg = vicreg_loss(h_hat, h_tgt, lambda_var=LAMBDA_VAR, lambda_cov=LAMBDA_COV)

            loss = l_pred + l_reg

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            model.update_ema()
            total_loss += loss.item() * x_past.shape[0]
            nbatch += x_past.shape[0]

        avg_loss = total_loss / nbatch
        history['loss'].append(avg_loss)
        current_lr = sched.step()

        extra = ''
        if epoch % PROBE_EVERY == 0 or epoch == 1:
            model.eval()
            encode_fn = lambda past, mask: model.encode_context(past, mask)
            probe_rmse = eval_probe_rmse(encode_fn, data['train_engines'],
                                          data['val_engines'])
            history['probe_rmse'].append(probe_rmse)
            history['probe_epochs'].append(epoch)

            if probe_rmse < best_probe:
                best_probe = probe_rmse
                no_impr = 0
                if ckpt_path is not None:
                    torch.save(model.state_dict(), ckpt_path)
            else:
                no_impr += 1

            extra = f" | probe={probe_rmse:.2f} (best={best_probe:.2f}) | lr={current_lr:.2e}"

            if run is not None:
                run.log({'epoch': epoch, 'train_loss': avg_loss,
                         'probe_rmse': probe_rmse, 'best_probe': best_probe,
                         'lr': current_lr})

        print(f"  Ep {epoch:3d} | loss={avg_loss:.4f}{extra}", flush=True)

        if no_impr >= PATIENCE:
            print(f"  Early stop at epoch {epoch} (patience={PATIENCE})")
            break

    if run is not None:
        run.finish()

    elapsed = (time.time() - t0) / 60
    print(f"  done in {elapsed:.1f} min, best_probe={best_probe:.2f}")
    return model, history, best_probe


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings('ignore')

    print("=" * 60)
    print("V16b: Stable Training with VICReg + Warmup")
    print(f"LR warmup: {WARMUP_EPOCHS} epochs, lambda_var={LAMBDA_VAR}, lambda_cov={LAMBDA_COV}")
    print(f"EMA momentum: {EMA_MOMENTUM}, Seeds: {SEEDS}")
    print("=" * 60)

    print("\nLoading FD001 data...")
    data = load_cmapss_subset('FD001')

    results = {
        'config': 'v16b',
        'architecture': 'bidi_context_encoder + causal_target_encoder (EMA)',
        'fixes': [
            f'warmup={WARMUP_EPOCHS}ep',
            f'lambda_var={LAMBDA_VAR}',
            f'lambda_cov={LAMBDA_COV}',
            f'ema_momentum={EMA_MOMENTUM}',
        ],
        'seeds': SEEDS,
        'n_epochs': N_EPOCHS,
        'frozen_probe_rmse_per_seed': [],
        'probe_histories': [],
        'baselines': {
            'v2_frozen': {'mean': 17.81, 'std': 1.7},
            'v16a_seed42': 4.75,
            'v16a_seed123': 8.53,
            'v16a_seed456': 12.45,
        }
    }

    for seed in SEEDS:
        print(f"\n--- Seed {seed} ---")
        ckpt_path = V16_DIR / f'best_v16b_seed{seed}.pt'
        model, history, best_probe = pretrain_v16b(data, seed=seed, ckpt_path=ckpt_path)

        results['frozen_probe_rmse_per_seed'].append(float(best_probe))
        results['probe_histories'].append(history)

        # Save after each seed
        with open(V16_DIR / 'phase1_v16b_results.json', 'w') as f:
            json.dump(results, f, indent=2)

    valid = results['frozen_probe_rmse_per_seed']
    results['frozen_probe_mean'] = float(np.mean(valid))
    results['frozen_probe_std'] = float(np.std(valid))

    print("\n" + "=" * 60)
    print("V16b 3-Seed Summary")
    print("=" * 60)
    for seed, rmse in zip(SEEDS, results['frozen_probe_rmse_per_seed']):
        print(f"  Seed {seed}: frozen probe RMSE = {rmse:.2f}")
    print(f"\n  Mean: {results['frozen_probe_mean']:.2f} +/- {results['frozen_probe_std']:.2f}")
    print(f"  V2 baseline: 17.81 +/- 1.7")
    print(f"  V16a seed42: 4.75 (single genuine seed)")

    with open(V16_DIR / 'phase1_v16b_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {V16_DIR / 'phase1_v16b_results.json'}")
