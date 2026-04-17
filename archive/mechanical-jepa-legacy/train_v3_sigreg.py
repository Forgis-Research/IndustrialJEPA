"""
Mechanical-JEPA V3 Training Script (SIGReg, No EMA).

V3 key change: Replace EMA target encoder with SIGReg regularization.
This removes 50% of model parameters (no target_encoder copy) and simplifies training.

Usage:
    # Default V3 (SIGReg=0.1)
    python train_v3_sigreg.py --epochs 30 --seed 42

    # Different SIGReg coefficients
    python train_v3_sigreg.py --epochs 30 --seed 42 --sigreg 0.01
    python train_v3_sigreg.py --epochs 30 --seed 42 --sigreg 0.1
    python train_v3_sigreg.py --epochs 30 --seed 42 --sigreg 1.0

    # Full training
    python train_v3_sigreg.py --epochs 100 --seed 42 --sigreg 0.1
"""

import argparse
import os
import sys
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / '.env')
except ImportError:
    pass

import wandb

sys.path.insert(0, str(Path(__file__).parent))
from src.data import create_dataloaders
from src.models import MechanicalJEPAV3

CHECKPOINT_DIR = Path('/mnt/sagemaker-nvme/jepa_checkpoints')
CHECKPOINT_DIR.mkdir(exist_ok=True, parents=True)


DEFAULT_CONFIG = {
    # Data
    'data_dir': 'data/bearings',
    'dataset_filter': 'cwru',
    'batch_size': 32,
    'window_size': 4096,
    'stride': 2048,
    'n_channels': 3,
    'test_ratio': 0.2,
    'num_workers': 0,

    # Model
    'patch_size': 256,
    'embed_dim': 512,
    'encoder_depth': 4,
    'predictor_depth': 4,
    'n_heads': 4,
    'mask_ratio': 0.625,

    # V3-specific
    'predictor_pos': 'sinusoidal',
    'loss_fn': 'l1',
    'sigreg_coeff': 0.1,
    'sigreg_projections': 64,
    'var_reg_lambda': 0.0,

    # Training
    'epochs': 30,
    'lr': 1e-4,
    'weight_decay': 0.05,
    'warmup_epochs': 5,
    'min_lr': 1e-6,

    # Linear probe
    'probe_epochs': 20,
    'probe_lr': 1e-3,

    # Other
    'seed': 42,
}


def cosine_scheduler(base_value, final_value, epochs, warmup_epochs):
    warmup_schedule = np.linspace(0, base_value, warmup_epochs)
    iters = np.arange(epochs - warmup_epochs)
    schedule = final_value + 0.5 * (base_value - final_value) * (
        1 + np.cos(np.pi * iters / len(iters))
    )
    return np.concatenate([warmup_schedule, schedule])


class LinearProbe(nn.Module):
    def __init__(self, embed_dim, n_classes):
        super().__init__()
        self.fc = nn.Linear(embed_dim, n_classes)

    def forward(self, x):
        return self.fc(x)


def extract_embeddings(model, loader, device):
    model.eval()
    all_embeds, all_labels = [], []
    with torch.no_grad():
        for signals, labels, _ in loader:
            signals = signals.to(device)
            embeds = model.get_embeddings(signals, pool='mean')
            all_embeds.append(embeds.cpu())
            all_labels.append(labels.cpu())
    return torch.cat(all_embeds), torch.cat(all_labels)


def train_probe_get_f1(train_embeds, train_labels, test_embeds, test_labels,
                        embed_dim, device, seed=42, probe_epochs=20):
    torch.manual_seed(seed)
    probe = LinearProbe(embed_dim, 4).to(device)
    optimizer = optim.Adam(probe.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    te_dev = train_embeds.to(device)
    tl_dev = train_labels.to(device)

    best_f1 = 0.0
    best_acc = 0.0
    best_preds = None

    for ep in range(probe_epochs):
        probe.train()
        optimizer.zero_grad()
        logits = probe(te_dev)
        loss = criterion(logits, tl_dev)
        loss.backward()
        optimizer.step()

        probe.eval()
        with torch.no_grad():
            test_logits = probe(test_embeds.to(device))
            preds = test_logits.argmax(dim=1).cpu().numpy()
            f1 = f1_score(test_labels.numpy(), preds, average='macro', zero_division=0)
            acc = (preds == test_labels.numpy()).mean()

        if f1 > best_f1:
            best_f1 = f1
            best_acc = acc
            best_preds = preds.copy()

    return best_f1, best_acc, best_preds


def quick_diagnose(model, device):
    """Collapse diagnostic for V3 (no EMA)."""
    model.eval()
    B = 8
    x = torch.randn(B, 3, 4096).to(device)

    with torch.no_grad():
        n_patches = model.n_patches
        n_context = int(n_patches * (1 - model.mask_ratio))

        context_indices = torch.arange(n_context).unsqueeze(0).expand(B, -1).to(device)
        all_patches = model.encoder(x, return_all_tokens=True)[:, 1:]
        context_embeds = all_patches[:, :n_context]

        single_preds = []
        for pos in range(n_context, n_patches):
            mask_idx = torch.tensor([[pos]]).expand(B, -1).to(device)
            pred = model.predictor(context_embeds, context_indices, mask_idx)
            single_preds.append(pred[:, 0])

        single_preds = torch.stack(single_preds, dim=1)
        pred_var_across_pos = single_preds.var(dim=1).mean().item()

        loss, predictions, targets = model(x)
        pred_std = predictions.std(dim=1).mean().item()
        targ_std = targets.std(dim=1).mean().item()
        spread_ratio = pred_std / (targ_std + 1e-8)

        # Check SIGReg value
        sigreg_val = model.sigreg(all_patches).item()

        # Embedding isotropy: ratio of min/max singular values (simple check)
        flat = all_patches.reshape(-1, all_patches.shape[-1])  # (B*N, D)
        flat_centered = flat - flat.mean(dim=0, keepdim=True)
        # Approximate via std per dimension
        dim_std = flat_centered.std(dim=0)
        isotropy = dim_std.min().item() / (dim_std.max().item() + 1e-8)

    return {
        'pred_var_across_pos': pred_var_across_pos,
        'spread_ratio': spread_ratio,
        'sigreg_val': sigreg_val,
        'isotropy': isotropy,
        'collapse': pred_var_across_pos < 0.001,
    }


def train(config: dict, device: torch.device):
    print("=" * 60)
    print("MECHANICAL-JEPA V3 TRAINING (SIGReg, No EMA)")
    print("=" * 60)

    run_name = config.get('run_name') or (
        f"v3_sigreg{config['sigreg_coeff']}"
        f"_pd{config['predictor_depth']}"
        f"_{config['loss_fn']}"
        f"_s{config['seed']}"
    )

    print(f"\nRun: {run_name}")

    use_wandb = not config.get('no_wandb', False)
    if use_wandb:
        wandb.init(
            project=config.get('wandb_project', 'mechanical-jepa'),
            config={k: v for k, v in config.items() if not k.startswith('wandb') and k != 'no_wandb'},
            name=run_name,
            tags=['v3', 'sigreg', 'no-ema'],
        )
        print(f"WandB: {wandb.run.url}")

    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])

    print("\nLoading data...")
    train_loader, test_loader, data_info = create_dataloaders(
        data_dir=config['data_dir'],
        batch_size=config['batch_size'],
        window_size=config['window_size'],
        stride=config['stride'],
        test_ratio=config['test_ratio'],
        seed=config['seed'],
        num_workers=config['num_workers'],
        dataset_filter=config['dataset_filter'],
        n_channels=config['n_channels'],
    )
    print(f"Train: {data_info['train_windows']} windows, Test: {data_info['test_windows']}")

    print("\nCreating V3 model (SIGReg, no EMA)...")
    model = MechanicalJEPAV3(
        n_channels=config['n_channels'],
        window_size=config['window_size'],
        patch_size=config['patch_size'],
        embed_dim=config['embed_dim'],
        encoder_depth=config['encoder_depth'],
        predictor_depth=config['predictor_depth'],
        n_heads=config['n_heads'],
        mask_ratio=config['mask_ratio'],
        predictor_pos=config['predictor_pos'],
        loss_fn=config['loss_fn'],
        sigreg_coeff=config['sigreg_coeff'],
        sigreg_projections=config['sigreg_projections'],
        var_reg_lambda=config['var_reg_lambda'],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")
    print(f"(V2 had ~2x more due to EMA target_encoder copy)")

    optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    lr_schedule = cosine_scheduler(config['lr'], config['min_lr'], config['epochs'], config['warmup_epochs'])

    print(f"\nTraining for {config['epochs']} epochs...")
    history = {'loss': []}
    start_time = time.time()

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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / n_batches
        history['loss'].append(avg_loss)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{config['epochs']}: loss={avg_loss:.4f}")

        if use_wandb:
            wandb.log({'epoch': epoch + 1, 'loss': avg_loss, 'lr': lr})

    total_time = time.time() - start_time
    print(f"\nTraining done in {total_time/60:.1f}min")

    # Collapse diagnostic
    print("\nRunning collapse diagnostic...")
    diag = quick_diagnose(model, device)
    print(f"  pred_var_across_pos: {diag['pred_var_across_pos']:.6f}")
    print(f"  spread_ratio: {diag['spread_ratio']:.4f}")
    print(f"  sigreg_val: {diag['sigreg_val']:.4f}")
    print(f"  isotropy: {diag['isotropy']:.4f}")
    print(f"  collapsed: {diag['collapse']}")
    if diag['collapse']:
        print("  [X] PREDICTOR STILL COLLAPSED")
    else:
        print("  [OK] Predictor NOT collapsed")

    if use_wandb:
        wandb.log({
            'diag/pred_var_across_pos': diag['pred_var_across_pos'],
            'diag/spread_ratio': diag['spread_ratio'],
            'diag/sigreg_val': diag['sigreg_val'],
            'diag/isotropy': diag['isotropy'],
            'diag/collapsed': int(diag['collapse']),
        })

    # Linear probe evaluation
    print("\n" + "=" * 60)
    print("LINEAR PROBE EVALUATION")
    print("=" * 60)

    train_embeds, train_labels = extract_embeddings(model, train_loader, device)
    test_embeds, test_labels = extract_embeddings(model, test_loader, device)

    f1, acc, preds = train_probe_get_f1(
        train_embeds, train_labels, test_embeds, test_labels,
        config['embed_dim'], device, seed=config['seed']
    )

    # Random init baseline
    torch.manual_seed(config['seed'] + 10000)
    rand_model = MechanicalJEPAV3(
        n_channels=config['n_channels'],
        window_size=config['window_size'],
        patch_size=config['patch_size'],
        embed_dim=config['embed_dim'],
        encoder_depth=config['encoder_depth'],
        sigreg_coeff=0.0,
    ).to(device)

    rand_train, rand_train_labels = extract_embeddings(rand_model, train_loader, device)
    rand_test, rand_test_labels = extract_embeddings(rand_model, test_loader, device)
    rand_f1, rand_acc, _ = train_probe_get_f1(
        rand_train, rand_train_labels, rand_test, rand_test_labels,
        config['embed_dim'], device, seed=config['seed']
    )

    print(f"\nV3 JEPA Macro F1: {f1:.4f} (acc: {acc:.4f})")
    print(f"Random init F1:   {rand_f1:.4f} (acc: {rand_acc:.4f})")
    print(f"F1 gain:          {f1 - rand_f1:+.4f}")

    probe_results = {
        'macro_f1': f1,
        'test_acc': acc,
        'rand_f1': rand_f1,
        'rand_acc': rand_acc,
        'f1_gain': f1 - rand_f1,
    }

    if use_wandb:
        wandb.log({
            'probe/macro_f1': f1,
            'probe/test_acc': acc,
            'probe/rand_f1': rand_f1,
            'probe/f1_gain': f1 - rand_f1,
        })
        wandb.summary['macro_f1'] = f1
        wandb.summary['f1_gain'] = f1 - rand_f1

    # Save checkpoint to NVMe
    if not config.get('no_save', False):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        checkpoint_path = CHECKPOINT_DIR / f'jepa_v3_{timestamp}_sig{config["sigreg_coeff"]}_s{config["seed"]}.pt'
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': config,
            'history': history,
            'probe_results': probe_results,
            'diag': diag,
        }, checkpoint_path)
        print(f"\nSaved: {checkpoint_path}")

    if use_wandb:
        wandb.finish()

    return model, probe_results, diag


def get_args():
    parser = argparse.ArgumentParser(description='Train Mechanical-JEPA V3 (SIGReg)')
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--sigreg', type=float, default=None, help='SIGReg coefficient')
    parser.add_argument('--sigreg-projections', type=int, default=None)
    parser.add_argument('--predictor-depth', type=int, default=None)
    parser.add_argument('--loss-fn', type=str, default=None, choices=['mse', 'l1', 'smooth_l1'])
    parser.add_argument('--mask-ratio', type=float, default=None)
    parser.add_argument('--embed-dim', type=int, default=None)
    parser.add_argument('--var-reg', type=float, default=None)
    parser.add_argument('--no-wandb', action='store_true')
    parser.add_argument('--no-save', action='store_true')
    parser.add_argument('--run-name', type=str, default=None)
    parser.add_argument('--wandb-project', type=str, default='mechanical-jepa')
    return parser.parse_args()


def main():
    args = get_args()
    config = DEFAULT_CONFIG.copy()

    if args.epochs is not None: config['epochs'] = args.epochs
    if args.seed is not None: config['seed'] = args.seed
    if args.sigreg is not None: config['sigreg_coeff'] = args.sigreg
    if args.sigreg_projections is not None: config['sigreg_projections'] = args.sigreg_projections
    if args.predictor_depth is not None: config['predictor_depth'] = args.predictor_depth
    if args.loss_fn is not None: config['loss_fn'] = args.loss_fn
    if args.mask_ratio is not None: config['mask_ratio'] = args.mask_ratio
    if args.embed_dim is not None: config['embed_dim'] = args.embed_dim
    if args.var_reg is not None: config['var_reg_lambda'] = args.var_reg
    if args.no_wandb: config['no_wandb'] = True
    if args.no_save: config['no_save'] = True
    if args.run_name: config['run_name'] = args.run_name
    config['wandb_project'] = args.wandb_project

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    model, probe_results, diag = train(config, device)

    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Macro F1: {probe_results['macro_f1']:.4f}")
    print(f"F1 gain:  {probe_results['f1_gain']:+.4f}")
    print(f"Collapsed: {diag['collapse']}")


if __name__ == '__main__':
    main()
