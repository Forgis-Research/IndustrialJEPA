"""
Round 5: Spectral/FFT input experiments.

Tests whether providing FFT magnitude alongside time-domain improves JEPA pretraining.

Approaches:
1. FFT-only: Replace raw time signal with FFT magnitude spectrum
2. Dual-domain: Concatenate time + FFT as extra channels
3. Log-FFT: Log-scale FFT magnitude (better dynamic range for vibration)

Usage:
    python train_spectral.py --input-type fft --epochs 100 --seed 42
    python train_spectral.py --input-type dual --epochs 100 --seed 42
    python train_spectral.py --input-type log_fft --epochs 100 --seed 42
"""

import argparse
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / '.env')
except ImportError:
    pass

import wandb

import sys
sys.path.insert(0, str(Path(__file__).parent))

from src.data import create_dataloaders
from src.models import MechanicalJEPAV2


def compute_fft_features(x: torch.Tensor, log_scale: bool = False) -> torch.Tensor:
    """
    Compute FFT magnitude features from raw vibration signal.

    Args:
        x: (B, C, T) raw time-domain signal
        log_scale: If True, apply log1p scaling

    Returns:
        fft_mag: (B, C, T//2) FFT magnitude (one-sided spectrum)
                  Then padded/resampled to T for easy concatenation
    """
    B, C, T = x.shape

    # FFT magnitude (one-sided)
    fft = torch.fft.rfft(x, dim=-1)  # (B, C, T//2 + 1)
    fft_mag = fft.abs()              # (B, C, T//2 + 1)

    if log_scale:
        fft_mag = torch.log1p(fft_mag)

    # Resize to match time dimension using interpolation
    fft_mag = torch.nn.functional.interpolate(
        fft_mag, size=T, mode='linear', align_corners=False
    )  # (B, C, T)

    return fft_mag


class SpectralBearingDataset(Dataset):
    """
    Wrapper dataset that adds spectral features to bearing windows.
    """

    def __init__(self, base_dataset, input_type: str = 'dual'):
        """
        input_type:
          'raw': original time-domain only (baseline)
          'fft': FFT magnitude only (replaces raw)
          'dual': concatenate raw + FFT as extra channels
          'log_fft': log-scale FFT only
          'dual_log': raw + log-FFT
        """
        self.base = base_dataset
        self.input_type = input_type

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        signal, label, bearing_id = self.base[idx]  # (C, T)

        if self.input_type == 'raw':
            return signal, label, bearing_id

        # Compute FFT features
        x = signal.unsqueeze(0)  # (1, C, T) for batch dim
        if self.input_type == 'fft':
            fft = compute_fft_features(x, log_scale=False).squeeze(0)
            return fft, label, bearing_id
        elif self.input_type == 'log_fft':
            fft = compute_fft_features(x, log_scale=True).squeeze(0)
            return fft, label, bearing_id
        elif self.input_type == 'dual':
            fft = compute_fft_features(x, log_scale=False).squeeze(0)
            combined = torch.cat([signal, fft], dim=0)  # (2C, T)
            return combined, label, bearing_id
        elif self.input_type == 'dual_log':
            fft = compute_fft_features(x, log_scale=True).squeeze(0)
            combined = torch.cat([signal, fft], dim=0)  # (2C, T)
            return combined, label, bearing_id
        else:
            raise ValueError(f"Unknown input_type: {self.input_type}")


def get_n_channels(input_type: str, base_n_channels: int = 3) -> int:
    """Return number of channels for given input type."""
    if input_type in ('raw', 'fft', 'log_fft'):
        return base_n_channels
    elif input_type in ('dual', 'dual_log'):
        return base_n_channels * 2
    raise ValueError(f"Unknown input_type: {input_type}")


DEFAULT_CONFIG = {
    'data_dir': 'data/bearings',
    'dataset_filter': 'cwru',
    'batch_size': 32,
    'window_size': 4096,
    'stride': 2048,
    'n_channels': 3,   # Will be overridden based on input_type
    'test_ratio': 0.2,
    'num_workers': 0,
    'patch_size': 256,
    'embed_dim': 512,
    'encoder_depth': 4,
    'predictor_depth': 4,
    'n_heads': 4,
    'mask_ratio': 0.625,
    'ema_decay': 0.996,
    'predictor_pos': 'sinusoidal',
    'separate_mask_tokens': False,
    'loss_fn': 'l1',
    'var_reg_lambda': 0.1,
    'vicreg_lambda': 0.0,
    'epochs': 100,
    'lr': 1e-4,
    'weight_decay': 0.05,
    'warmup_epochs': 5,
    'min_lr': 1e-6,
    'probe_epochs': 20,
    'probe_lr': 1e-3,
    'seed': 42,
    'input_type': 'raw',  # 'raw' | 'fft' | 'log_fft' | 'dual' | 'dual_log'
}


def cosine_scheduler(base_value, final_value, epochs, warmup_epochs):
    warmup_schedule = np.linspace(0, base_value, warmup_epochs)
    iters = np.arange(epochs - warmup_epochs)
    schedule = final_value + 0.5 * (base_value - final_value) * (
        1 + np.cos(np.pi * iters / len(iters))
    )
    return np.concatenate([warmup_schedule, schedule])


def quick_diagnose(model, device):
    model.eval()
    B = 8
    x = torch.randn(B, model.encoder.patch_embed.n_channels, 4096).to(device)
    with torch.no_grad():
        n_patches = model.n_patches
        n_context = n_patches // 2
        ctx_idx = torch.arange(n_context).unsqueeze(0).expand(B, -1).to(device)
        all_patches = model.encoder(x, return_all_tokens=True)[:, 1:]
        ctx_embeds = all_patches[:, :n_context]
        preds = []
        for pos in range(n_context, n_patches):
            mask_idx = torch.tensor([[pos]]).expand(B, -1).to(device)
            pred = model.predictor(ctx_embeds, ctx_idx, mask_idx)
            preds.append(pred[:, 0])
        preds = torch.stack(preds, dim=1)
        pred_var = preds.var(dim=1).mean().item()
        _, predictions, targets = model(x)
        pred_std = predictions.std(dim=1).mean().item()
        targ_std = targets.std(dim=1).mean().item()
        spread = pred_std / (targ_std + 1e-8)
    return {'pred_var_across_pos': pred_var, 'spread_ratio': spread, 'collapse': pred_var < 0.001}


def train_and_eval(config: dict, device: torch.device):
    input_type = config['input_type']
    base_n_ch = 3
    actual_n_ch = get_n_channels(input_type, base_n_ch)

    run_name = f"spectral_{input_type}_s{config['seed']}"
    print(f"\nRun: {run_name} | n_channels={actual_n_ch}")

    use_wandb = not config.get('no_wandb', False)
    if use_wandb:
        wandb.init(
            project='mechanical-jepa',
            config=config,
            name=run_name,
        )

    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])

    # Create base dataloaders (with 3 channels)
    config_for_data = dict(config)
    config_for_data['n_channels'] = base_n_ch
    train_loader_base, test_loader_base, data_info = create_dataloaders(
        data_dir=config['data_dir'],
        batch_size=config['batch_size'],
        window_size=config['window_size'],
        stride=config['stride'],
        test_ratio=config['test_ratio'],
        seed=config['seed'],
        num_workers=config['num_workers'],
        dataset_filter=config['dataset_filter'],
        n_channels=base_n_ch,
    )

    if input_type != 'raw':
        # Wrap datasets with spectral features
        train_ds = SpectralBearingDataset(train_loader_base.dataset, input_type)
        test_ds = SpectralBearingDataset(test_loader_base.dataset, input_type)
        train_loader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True, num_workers=0)
        test_loader = DataLoader(test_ds, batch_size=config['batch_size'], shuffle=False, num_workers=0)
    else:
        train_loader, test_loader = train_loader_base, test_loader_base

    print(f"Train: {len(train_loader.dataset)} windows, Test: {len(test_loader.dataset)}")

    # Create model with correct n_channels
    model = MechanicalJEPAV2(
        n_channels=actual_n_ch,
        window_size=config['window_size'],
        patch_size=config['patch_size'],
        embed_dim=config['embed_dim'],
        encoder_depth=config['encoder_depth'],
        predictor_depth=config['predictor_depth'],
        n_heads=config['n_heads'],
        mask_ratio=config['mask_ratio'],
        ema_decay=config['ema_decay'],
        predictor_pos=config['predictor_pos'],
        separate_mask_tokens=config['separate_mask_tokens'],
        loss_fn=config['loss_fn'],
        var_reg_lambda=config['var_reg_lambda'],
        vicreg_lambda=config['vicreg_lambda'],
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    lr_schedule = cosine_scheduler(config['lr'], config['min_lr'], config['epochs'], config['warmup_epochs'])

    start = time.time()
    for epoch in range(config['epochs']):
        model.train()
        total_loss = 0
        n_batches = len(train_loader)
        for batch_idx, (signals, labels, _) in enumerate(train_loader):
            lr = lr_schedule[min(epoch, len(lr_schedule) - 1)]
            for pg in optimizer.param_groups:
                pg['lr'] = lr
            signals = signals.to(device)
            optimizer.zero_grad()
            loss = model.train_step(signals)
            loss.backward()
            optimizer.step()
            model.update_ema()
            total_loss += loss.item()
        avg_loss = total_loss / n_batches
        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}: loss={avg_loss:.4f}")
        if use_wandb:
            wandb.log({'epoch': epoch+1, 'loss': avg_loss})

    print(f"  Training done in {(time.time()-start)/60:.1f}min")

    # Diagnostic
    diag = quick_diagnose(model, device)
    print(f"  Collapse: {diag['collapse']}, spread_ratio: {diag['spread_ratio']:.3f}")

    # Linear probe
    model.eval()
    def extract(loader):
        all_e, all_l = [], []
        with torch.no_grad():
            for signals, labels, _ in loader:
                embeds = model.get_embeddings(signals.to(device), pool='mean')
                all_e.append(embeds.cpu())
                all_l.append(labels)
        return torch.cat(all_e), torch.cat(all_l)

    train_e, train_l = extract(train_loader)
    test_e, test_l = extract(test_loader)

    probe = nn.Linear(config['embed_dim'], 4).to(device)
    opt_probe = optim.Adam(probe.parameters(), lr=config['probe_lr'])
    crit = nn.CrossEntropyLoss()
    train_e, train_l = train_e.to(device), train_l.to(device)
    test_e, test_l = test_e.to(device), test_l.to(device)

    best_acc = 0
    for ep in range(config['probe_epochs']):
        probe.train()
        opt_probe.zero_grad()
        crit(probe(train_e), train_l).backward()
        opt_probe.step()
        probe.eval()
        with torch.no_grad():
            preds = probe(test_e).argmax(1)
            acc = (preds == test_l).float().mean().item()
        if acc > best_acc:
            best_acc = acc

    print(f"  Linear probe: {best_acc:.4f}")

    if use_wandb:
        wandb.log({'probe/test_acc': best_acc, 'diag/spread_ratio': diag['spread_ratio'],
                   'diag/collapsed': int(diag['collapse'])})
        wandb.summary['final_test_acc'] = best_acc
        wandb.finish()

    return best_acc, diag


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-type', type=str, default='raw',
                        choices=['raw', 'fft', 'log_fft', 'dual', 'dual_log'])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no-wandb', action='store_true')
    args = parser.parse_args()

    config = DEFAULT_CONFIG.copy()
    config['input_type'] = args.input_type
    config['epochs'] = args.epochs
    config['seed'] = args.seed
    if args.no_wandb:
        config['no_wandb'] = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Input type: {args.input_type}")

    acc, diag = train_and_eval(config, device)
    print(f"\nFinal: acc={acc:.4f}, collapsed={diag['collapse']}")


if __name__ == '__main__':
    main()
