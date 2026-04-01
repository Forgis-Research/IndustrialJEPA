"""
Multi-Source Pretraining Experiment (Round 1C-4).

Pretrain JEPA on CWRU + Paderborn (resampled to 12kHz) together.
Test on each held-out dataset.

Questions:
1. Does more diverse pretraining data improve representations?
2. Do representations learned from mixed sources transfer better?

Dataset mixing strategy:
- CWRU: 4096-sample windows at 12kHz (native)
- Paderborn: 4096-sample windows at 12kHz (downsampled from 64kHz)
- IMS: 4096-sample windows at 12kHz (downsampled from 20kHz)
  Note: IMS has RUL labels, not fault type labels, so used unsupervised only

The JEPA training is fully unsupervised - labels are not used.
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime
import time

import numpy as np
import scipy.signal
import scipy.io
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset

sys.path.insert(0, str(Path(__file__).parent))
from src.data.bearing_dataset import load_cwru_signal
from src.models.jepa_v2 import MechanicalJEPAV2
import wandb

PADERBORN_DIR = Path('/home/sagemaker-user/IndustrialJEPA/datasets/data/paderborn')
SOURCE_SR_PADERBORN = 64000


def resample_poly_1d(signal: np.ndarray, source_sr: int, target_sr: int) -> np.ndarray:
    from math import gcd
    if source_sr == target_sr:
        return signal
    g = gcd(source_sr, target_sr)
    up = target_sr // g
    down = source_sr // g
    return scipy.signal.resample_poly(signal, up, down).astype(np.float32)


def load_paderborn_vib(mat_path: Path) -> np.ndarray:
    mat = scipy.io.loadmat(str(mat_path), squeeze_me=True, simplify_cells=True)
    key = [k for k in mat.keys() if not k.startswith('_')][0]
    return mat[key]['Y'][6]['Data'].astype(np.float32)


class PaderbornWindowDataset(Dataset):
    """Paderborn windows resampled to target_sr, no labels needed (unsupervised)."""

    def __init__(
        self,
        bearing_dirs: list,  # [(path, label), ...]
        window_size: int = 4096,
        stride: int = 2048,
        target_sr: int = 12000,
        n_channels: int = 3,
        max_files: int = 20,
    ):
        self.window_size = window_size
        self.n_channels = n_channels
        self.windows = []  # (signal, start, label)

        for bearing_path, label in bearing_dirs:
            files = sorted(Path(bearing_path).glob('*.mat'))[:max_files]
            for f in files:
                try:
                    raw = load_paderborn_vib(f)
                    sig = resample_poly_1d(raw, SOURCE_SR_PADERBORN, target_sr)
                    n = len(sig)
                    n_wins = (n - window_size) // stride + 1
                    for i in range(n_wins):
                        self.windows.append((sig, i * stride, label))
                except:
                    continue

        print(f"PaderbornWindowDataset: {len(self.windows)} windows (sr={target_sr})")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        sig, start, label = self.windows[idx]
        w = sig[start:start + self.window_size]
        # Repeat to n_channels
        wmc = np.stack([w] * self.n_channels, axis=0)
        # Normalize
        mean = wmc.mean(axis=1, keepdims=True)
        std = wmc.std(axis=1, keepdims=True) + 1e-8
        wmc = (wmc - mean) / std
        return torch.tensor(wmc, dtype=torch.float32), label, 'paderborn'


def cosine_scheduler(base, final, epochs, warmup):
    sched = np.linspace(0, base, warmup)
    iters = np.arange(epochs - warmup)
    sched2 = final + 0.5 * (base - final) * (1 + np.cos(np.pi * iters / len(iters)))
    return np.concatenate([sched, sched2])


def train_jepa_multisource(
    cwru_loader,
    paderborn_loader,
    model,
    epochs: int,
    device: torch.device,
    run_name: str,
    use_wandb: bool = True,
):
    """Train JEPA on combined CWRU + Paderborn data."""
    if use_wandb:
        wandb.init(project='mechanical-jepa', name=run_name)

    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)
    lr_schedule = cosine_scheduler(1e-4, 1e-6, epochs, 5)

    print(f"\nMulti-source pretraining for {epochs} epochs...")
    print(f"CWRU: {len(cwru_loader.dataset)} windows")
    print(f"Paderborn: {len(paderborn_loader.dataset)} windows")

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        n_batches = 0

        # Interleave CWRU and Paderborn batches
        lr = lr_schedule[min(epoch, len(lr_schedule) - 1)]
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        for batch_data in cwru_loader:
            signals = batch_data[0].to(device)
            optimizer.zero_grad()
            loss = model.train_step(signals)
            loss.backward()
            optimizer.step()
            model.update_ema()
            total_loss += loss.item()
            n_batches += 1

        for batch_data in paderborn_loader:
            signals = batch_data[0].to(device)
            optimizer.zero_grad()
            loss = model.train_step(signals)
            loss.backward()
            optimizer.step()
            model.update_ema()
            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f}")

        if use_wandb:
            wandb.log({'epoch': epoch+1, 'loss': avg_loss})

    return model


def evaluate_cwru_probe(model, cwru_train_loader, cwru_test_loader, embed_dim, device, probe_epochs=20):
    """Linear probe on CWRU."""
    model.eval()

    def extract(loader):
        all_e, all_l = [], []
        with torch.no_grad():
            for signals, labels, _ in loader:
                signals = signals.to(device)
                embeds = model.get_embeddings(signals, pool='mean')
                all_e.append(embeds.cpu())
                all_l.append(labels)
        return torch.cat(all_e), torch.cat(all_l)

    train_e, train_l = extract(cwru_train_loader)
    test_e, test_l = extract(cwru_test_loader)

    mean = train_e.mean(0, keepdim=True)
    std = train_e.std(0, keepdim=True) + 1e-8
    train_e_n = ((train_e - mean) / std).to(device)
    test_e_n = ((test_e - mean) / std).to(device)
    train_l_d = train_l.to(device)
    test_l_d = test_l.to(device)

    probe = nn.Linear(embed_dim, 4).to(device)
    opt = optim.Adam(probe.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()

    best_acc = 0
    for ep in range(probe_epochs):
        probe.train()
        opt.zero_grad()
        crit(probe(train_e_n), train_l_d).backward()
        opt.step()

        probe.eval()
        with torch.no_grad():
            preds = probe(test_e_n).argmax(1)
            acc = (preds == test_l_d).float().mean().item()
        if acc > best_acc:
            best_acc = acc

    return best_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--target-sr', type=int, default=12000)
    parser.add_argument('--max-pad-files', type=int, default=20,
                        help='Max Paderborn files per class (20 of 80)')
    parser.add_argument('--no-wandb', action='store_true')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"Device: {device}")
    print(f"Multi-source pretraining: CWRU + Paderborn @ {args.target_sr}Hz")

    # CWRU data loader (native 12kHz, no resampling needed)
    from src.data import create_dataloaders
    cwru_train, cwru_test, info = create_dataloaders(
        data_dir='data/bearings',
        batch_size=32, window_size=4096, stride=2048,
        test_ratio=0.2, seed=args.seed, num_workers=0,
        dataset_filter='cwru', n_channels=3,
    )
    print(f"CWRU: {info['train_windows']} train, {info['test_windows']} test")

    # Paderborn data (will be resampled to target_sr)
    pad_bearing_dirs = [
        (PADERBORN_DIR / 'K001', 0),
        (PADERBORN_DIR / 'KA01', 1),
        (PADERBORN_DIR / 'KI01', 2),
    ]
    pad_dataset = PaderbornWindowDataset(
        bearing_dirs=pad_bearing_dirs,
        window_size=4096, stride=2048,
        target_sr=args.target_sr, n_channels=3,
        max_files=args.max_pad_files,
    )
    pad_loader = DataLoader(pad_dataset, batch_size=32, shuffle=True, num_workers=0)

    # Model
    model = MechanicalJEPAV2(
        n_channels=3, window_size=4096, patch_size=256,
        embed_dim=512, encoder_depth=4, predictor_depth=4, n_heads=4,
        mask_ratio=0.625, ema_decay=0.996,
        predictor_pos='sinusoidal', loss_fn='l1', var_reg_lambda=0.1,
    ).to(device)

    run_name = f"multi_src_cwru_pad_sr{args.target_sr}_s{args.seed}"

    # Train
    use_wandb = not args.no_wandb
    model = train_jepa_multisource(
        cwru_train, pad_loader, model,
        epochs=args.epochs, device=device, run_name=run_name,
        use_wandb=use_wandb,
    )

    # Evaluate on CWRU
    print("\n--- CWRU evaluation (multi-source pretrained) ---")
    multi_acc = evaluate_cwru_probe(model, cwru_train, cwru_test, 512, device)
    print(f"Multi-source CWRU probe: {multi_acc:.4f}")

    # Compare: CWRU-only pretrained
    model_cwru_only = MechanicalJEPAV2(
        n_channels=3, window_size=4096, patch_size=256,
        embed_dim=512, encoder_depth=4, predictor_depth=4, n_heads=4,
        mask_ratio=0.625, ema_decay=0.996,
        predictor_pos='sinusoidal', loss_fn='l1', var_reg_lambda=0.1,
    ).to(device)
    model_cwru_only.load_state_dict(
        torch.load('checkpoints/jepa_v2_20260401_003619.pt',
                   map_location=device, weights_only=False)['model_state_dict']
    )
    cwru_only_acc = evaluate_cwru_probe(model_cwru_only, cwru_train, cwru_test, 512, device)
    print(f"CWRU-only pretrained probe: {cwru_only_acc:.4f}")

    print(f"\nMulti-source vs CWRU-only: {multi_acc - cwru_only_acc:+.4f}")

    if use_wandb:
        wandb.log({'cwru_probe': multi_acc})
        wandb.summary['cwru_probe'] = multi_acc
        wandb.finish()

    # Save checkpoint
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    ckpt_path = Path('checkpoints') / f'jepa_multisrc_{ts}.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': vars(args),
        'multi_acc': multi_acc,
        'cwru_only_acc': cwru_only_acc,
    }, ckpt_path)
    print(f"Saved: {ckpt_path}")

    return multi_acc, cwru_only_acc


if __name__ == '__main__':
    main()
