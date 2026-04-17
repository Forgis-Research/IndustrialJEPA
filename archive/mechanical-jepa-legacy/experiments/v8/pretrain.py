"""
Phase 2: JEPA V8 Pretraining on all bearing sources.

Training config:
- 100 epochs, AdamW, lr=1e-4, cosine schedule with 5-epoch warmup
- Batch size 64
- Collapse detection: prediction variance < 0.01 triggers warning
- Save checkpoint every 20 epochs + best (lowest val loss)
"""

import os
import sys
import json
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

sys.path.insert(0, '/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/v8')
from data_pipeline import load_pretrain_windows
from jepa_v8 import MechanicalJEPAV8, count_parameters

# Config
CHECKPOINT_DIR = '/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/v8/checkpoints'
RESULTS_DIR = '/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/v8/results'

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS = 100
BATCH_SIZE = 64
LR = 1e-4
WARMUP_EPOCHS = 5
VAL_FRAC = 0.1
SEED = 42


def get_cosine_lr(epoch: int, max_lr: float, epochs: int, warmup: int) -> float:
    """Cosine schedule with linear warmup."""
    import math
    if epoch < warmup:
        return max_lr * (epoch + 1) / warmup
    progress = (epoch - warmup) / (epochs - warmup)
    return max_lr * 0.5 * (1 + math.cos(math.pi * progress))


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    total_var = 0
    n_batches = 0
    for batch in loader:
        x = batch[0].to(device)
        loss, preds, _ = model(x)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        model.update_ema()
        total_loss += loss.item()
        total_var += preds.var(dim=1).mean().item()
        n_batches += 1
    return total_loss / n_batches, total_var / n_batches


@torch.no_grad()
def val_epoch(model, loader, device):
    model.eval()
    total_loss = 0
    total_var = 0
    n_batches = 0
    for batch in loader:
        x = batch[0].to(device)
        loss, preds, _ = model(x)
        total_loss += loss.item()
        total_var += preds.var(dim=1).mean().item()
        n_batches += 1
    return total_loss / n_batches, total_var / n_batches


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print(f"=== JEPA V8 Pretraining ===")
    print(f"Device: {DEVICE}")
    print(f"Epochs: {EPOCHS}, Batch: {BATCH_SIZE}, LR: {LR}")

    # Load data
    print("\n--- Loading data ---")
    t0 = time.time()
    X, sources = load_pretrain_windows(verbose=True)
    print(f"Data loaded: {X.shape} in {time.time()-t0:.1f}s")

    # Dataset split
    X_tensor = torch.from_numpy(X).unsqueeze(1)  # (N, 1, 1024)
    full_dataset = TensorDataset(X_tensor)
    n_val = int(len(full_dataset) * VAL_FRAC)
    n_train = len(full_dataset) - n_val
    train_ds, val_ds = random_split(full_dataset, [n_train, n_val],
                                     generator=torch.Generator().manual_seed(SEED))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                               num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=4, pin_memory=True)
    print(f"Train: {n_train}, Val: {n_val}")

    # Model
    model = MechanicalJEPAV8(
        n_channels=1,
        window_size=1024,
        patch_size=64,
        embed_dim=256,
        encoder_depth=4,
        predictor_depth=4,
        n_heads=4,
        mask_ratio=0.625,
        ema_decay=0.996,
        var_reg_lambda=0.1,
    ).to(DEVICE)
    print(f"Parameters: {count_parameters(model):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    # Training loop
    history = []
    best_val_loss = float('inf')
    best_epoch = 0

    print(f"\n{'Epoch':>5} {'Train Loss':>12} {'Val Loss':>10} {'Pred Var':>10} {'LR':>10} {'Time':>8}")
    print('-' * 65)

    for epoch in range(EPOCHS):
        t_ep = time.time()

        # Update LR
        lr = get_cosine_lr(epoch, LR, EPOCHS, WARMUP_EPOCHS)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        train_loss, train_var = train_epoch(model, train_loader, optimizer, DEVICE)
        val_loss, val_var = val_epoch(model, val_loader, DEVICE)

        t_ep = time.time() - t_ep

        # Collapse detection
        collapse_flag = ''
        if val_var < 0.01:
            collapse_flag = ' *** COLLAPSE DETECTED ***'
            print(f"WARNING: prediction variance {val_var:.4f} < 0.01 at epoch {epoch}")

        print(f"{epoch+1:5d} {train_loss:12.6f} {val_loss:10.6f} {val_var:10.6f} {lr:10.2e} {t_ep:7.1f}s{collapse_flag}")

        rec = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'pred_var': val_var,
            'lr': lr,
        }
        history.append(rec)

        # Save checkpoint
        if (epoch + 1) % 20 == 0:
            ckpt_path = os.path.join(CHECKPOINT_DIR, f'jepa_v8_epoch{epoch+1}.pt')
            torch.save({
                'epoch': epoch + 1,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'val_loss': val_loss,
                'history': history,
            }, ckpt_path)
            print(f"  Saved checkpoint: {ckpt_path}")

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            ckpt_path = os.path.join(CHECKPOINT_DIR, 'jepa_v8_best.pt')
            torch.save({
                'epoch': epoch + 1,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'val_loss': val_loss,
                'history': history,
            }, ckpt_path)

    # Save training history
    hist_path = os.path.join(RESULTS_DIR, 'pretrain_history.json')
    with open(hist_path, 'w') as f:
        json.dump({
            'history': history,
            'best_epoch': best_epoch,
            'best_val_loss': best_val_loss,
            'n_train': n_train,
            'n_val': n_val,
            'config': {
                'epochs': EPOCHS, 'batch_size': BATCH_SIZE, 'lr': LR,
                'warmup_epochs': WARMUP_EPOCHS, 'mask_ratio': 0.625,
                'embed_dim': 256, 'n_patches': 16, 'n_mask': 10,
            }
        }, f, indent=2)
    print(f"\nTraining history saved: {hist_path}")
    print(f"Best val loss: {best_val_loss:.6f} at epoch {best_epoch}")
    print(f"\nBest checkpoint: {CHECKPOINT_DIR}/jepa_v8_best.pt")


if __name__ == '__main__':
    main()
