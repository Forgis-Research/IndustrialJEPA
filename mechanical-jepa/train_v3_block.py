"""
Train V3 with temporal block masking (Round 5A).

Block masking forces the predictor to extrapolate across time, not just interpolate
between randomly scattered context patches. This is harder and may force better
temporal representations.

Usage:
    python train_v3_block.py --epochs 100 --seed 42 --mask-ratio 0.625
"""

import argparse
import sys
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / '.env')
except ImportError:
    pass

import wandb

sys.path.insert(0, str(Path(__file__).parent))
from src.data import create_dataloaders
from src.models.jepa_v2 import MechanicalJEPAV2


class MechanicalJEPAV2BlockMask(MechanicalJEPAV2):
    """
    V2 with temporal block masking.

    Instead of randomly scattering masked patches, we mask contiguous temporal blocks.
    This is harder for the predictor: it must extrapolate across time.

    The block masking strategy uses a single contiguous block starting at a random position.
    """

    def _generate_mask(self, batch_size: int, device: torch.device):
        """Generate contiguous temporal block masks."""
        n_mask = int(self.n_patches * self.mask_ratio)
        n_context = self.n_patches - n_mask

        mask_indices_all = []
        context_indices_all = []

        for _ in range(batch_size):
            # Pick random start for contiguous block
            max_start = self.n_patches - n_mask
            if max_start <= 0:
                start = 0
            else:
                start = torch.randint(0, max_start + 1, (1,), device=device).item()

            # Build mask: contiguous block [start, start+n_mask)
            mask_idx = [(start + i) % self.n_patches for i in range(n_mask)]
            mask_set = set(mask_idx)
            context_idx = [i for i in range(self.n_patches) if i not in mask_set]

            mask_indices_all.append(torch.tensor(mask_idx[:n_mask], device=device))
            context_indices_all.append(torch.tensor(context_idx[:n_context], device=device))

        return torch.stack(mask_indices_all), torch.stack(context_indices_all)


def cosine_scheduler(base_value, final_value, epochs, warmup_epochs):
    warmup_schedule = np.linspace(0, base_value, warmup_epochs)
    iters = np.arange(epochs - warmup_epochs)
    schedule = final_value + 0.5 * (base_value - final_value) * (
        1 + np.cos(np.pi * iters / len(iters))
    )
    return np.concatenate([warmup_schedule, schedule])


def train_linear_probe(model, train_loader, test_loader, embed_dim, device, probe_epochs=20):
    model.eval()
    n_classes = 4

    def extract_embeddings(loader):
        all_embeds, all_labels = [], []
        with torch.no_grad():
            for signals, labels, _ in loader:
                signals = signals.to(device)
                embeds = model.get_embeddings(signals, pool='mean')
                all_embeds.append(embeds.cpu())
                all_labels.append(labels)
        return torch.cat(all_embeds), torch.cat(all_labels)

    train_embeds, train_labels = extract_embeddings(train_loader)
    test_embeds, test_labels = extract_embeddings(test_loader)

    probe = nn.Linear(embed_dim, n_classes).to(device)
    optimizer = optim.Adam(probe.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    train_embeds = train_embeds.to(device)
    train_labels = train_labels.to(device)
    test_embeds = test_embeds.to(device)
    test_labels = test_labels.to(device)

    best_test_acc = 0
    for epoch in range(probe_epochs):
        probe.train()
        optimizer.zero_grad()
        logits = probe(train_embeds)
        loss = criterion(logits, train_labels)
        loss.backward()
        optimizer.step()

        probe.eval()
        with torch.no_grad():
            test_preds = probe(test_embeds).argmax(dim=1)
            test_acc = (test_preds == test_labels).float().mean().item()
        if test_acc > best_test_acc:
            best_test_acc = test_acc

    return best_test_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--embed-dim', type=int, default=512)
    parser.add_argument('--mask-ratio', type=float, default=0.625)
    parser.add_argument('--var-reg', type=float, default=0.1)
    parser.add_argument('--no-save', action='store_true')
    parser.add_argument('--run-name', type=str, default=None)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    run_name = args.run_name or f"v3_block_mr{args.mask_ratio}_s{args.seed}"

    wandb.init(
        project='mechanical-jepa',
        config=vars(args),
        name=run_name,
    )

    train_loader, test_loader, info = create_dataloaders(
        data_dir='data/bearings',
        batch_size=32,
        window_size=4096,
        stride=2048,
        test_ratio=0.2,
        seed=args.seed,
        num_workers=0,
        dataset_filter='cwru',
        n_channels=3,
    )

    model = MechanicalJEPAV2BlockMask(
        n_channels=3, window_size=4096, patch_size=256,
        embed_dim=args.embed_dim, encoder_depth=4, predictor_depth=4, n_heads=4,
        mask_ratio=args.mask_ratio, ema_decay=0.996,
        predictor_pos='sinusoidal', loss_fn='l1',
        var_reg_lambda=args.var_reg,
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)
    lr_schedule = cosine_scheduler(1e-4, 1e-6, args.epochs, 5)

    print(f"\nTraining {run_name} for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for signals, labels, _ in train_loader:
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

        avg_loss = total_loss / len(train_loader)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{args.epochs}: loss={avg_loss:.4f}")
        wandb.log({'epoch': epoch+1, 'loss': avg_loss})

    # Linear probe
    test_acc = train_linear_probe(model, train_loader, test_loader, args.embed_dim, device)
    print(f"\nBest test accuracy: {test_acc:.4f}")
    wandb.log({'probe/test_acc': test_acc})
    wandb.summary['final_test_acc'] = test_acc

    if not args.no_save:
        save_dir = Path('checkpoints')
        save_dir.mkdir(exist_ok=True)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        ckpt_path = save_dir / f'jepa_v3_block_{ts}.pt'
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': vars(args),
            'test_acc': test_acc,
        }, ckpt_path)
        print(f"Saved: {ckpt_path}")

    wandb.finish()
    return test_acc


if __name__ == '__main__':
    main()
