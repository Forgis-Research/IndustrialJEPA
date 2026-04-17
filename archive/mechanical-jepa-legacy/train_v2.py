"""
Mechanical-JEPA V2 Training Script.

Usage:
    # Baseline: sinusoidal pos encoding (main fix for predictor collapse)
    python train_v2.py --epochs 30 --seed 42 --predictor-pos sinusoidal

    # With deeper predictor
    python train_v2.py --epochs 30 --seed 42 --predictor-pos sinusoidal --predictor-depth 4

    # With L1 loss
    python train_v2.py --epochs 30 --seed 42 --predictor-pos sinusoidal --loss-fn l1

    # With variance regularization
    python train_v2.py --epochs 30 --seed 42 --predictor-pos sinusoidal --var-reg 0.1

    # Full fix (sinusoidal + separate tokens + L1 + var reg)
    python train_v2.py --epochs 30 --seed 42 --predictor-pos sinusoidal --separate-mask-tokens --loss-fn l1 --var-reg 0.05

    # VICReg encoder regularization
    python train_v2.py --epochs 30 --seed 42 --predictor-pos sinusoidal --vicreg 0.01
"""

import argparse
import os
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

from src.data import create_dataloaders
from src.models import MechanicalJEPAV2


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
    'predictor_depth': 4,      # V2 default: 4 (was 2 in V1)
    'n_heads': 4,
    'mask_ratio': 0.5,
    'ema_decay': 0.996,

    # V2-specific
    'predictor_pos': 'sinusoidal',   # 'sinusoidal' | 'learnable'
    'separate_mask_tokens': False,
    'loss_fn': 'mse',               # 'mse' | 'l1' | 'smooth_l1'
    'var_reg_lambda': 0.0,          # prediction variance regularization
    'vicreg_lambda': 0.0,           # encoder VICReg

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
    'log_interval': 10,
    'save_dir': 'checkpoints',
}


def get_args():
    parser = argparse.ArgumentParser(description='Train Mechanical-JEPA V2')

    # Standard args
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--embed-dim', type=int, default=None)
    parser.add_argument('--encoder-depth', type=int, default=None)
    parser.add_argument('--predictor-depth', type=int, default=None)
    parser.add_argument('--mask-ratio', type=float, default=None)
    parser.add_argument('--patch-size', type=int, default=None,
                        help='Patch size in samples (default: 256, gives 16 patches per window)')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--dataset', type=str, default=None)

    # V2-specific args
    parser.add_argument('--predictor-pos', type=str, default=None,
                        choices=['sinusoidal', 'learnable'],
                        help='Positional encoding type for predictor')
    parser.add_argument('--separate-mask-tokens', action='store_true', default=None,
                        help='Use per-position mask tokens instead of shared token')
    parser.add_argument('--loss-fn', type=str, default=None,
                        choices=['mse', 'l1', 'smooth_l1'],
                        help='Loss function for prediction target')
    parser.add_argument('--var-reg', type=float, default=None,
                        help='Variance regularization lambda (0=off)')
    parser.add_argument('--vicreg', type=float, default=None,
                        help='VICReg lambda for encoder (0=off)')

    # Modes
    parser.add_argument('--no-save', action='store_true')
    parser.add_argument('--no-wandb', action='store_true')
    parser.add_argument('--wandb-project', type=str, default='mechanical-jepa')
    parser.add_argument('--run-name', type=str, default=None,
                        help='Custom wandb run name')

    return parser.parse_args()


def cosine_scheduler(base_value, final_value, epochs, warmup_epochs):
    warmup_schedule = np.linspace(0, base_value, warmup_epochs)
    iters = np.arange(epochs - warmup_epochs)
    schedule = final_value + 0.5 * (base_value - final_value) * (
        1 + np.cos(np.pi * iters / len(iters))
    )
    return np.concatenate([warmup_schedule, schedule])


class LinearProbe(nn.Module):
    def __init__(self, embed_dim: int, n_classes: int):
        super().__init__()
        self.fc = nn.Linear(embed_dim, n_classes)

    def forward(self, x):
        return self.fc(x)


def train_linear_probe(model, train_loader, test_loader, config, device):
    print("\n" + "=" * 60)
    print("LINEAR PROBE EVALUATION")
    print("=" * 60)

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

    print(f"Train: {train_embeds.shape}, Test: {test_embeds.shape}")

    probe = LinearProbe(config['embed_dim'], n_classes).to(device)
    optimizer = optim.Adam(probe.parameters(), lr=config['probe_lr'])
    criterion = nn.CrossEntropyLoss()

    train_embeds = train_embeds.to(device)
    train_labels = train_labels.to(device)
    test_embeds = test_embeds.to(device)
    test_labels = test_labels.to(device)

    best_test_acc = 0
    for epoch in range(config['probe_epochs']):
        probe.train()
        optimizer.zero_grad()
        logits = probe(train_embeds)
        loss = criterion(logits, train_labels)
        loss.backward()
        optimizer.step()

        probe.eval()
        with torch.no_grad():
            test_logits = probe(test_embeds)
            test_preds = test_logits.argmax(dim=1)
            test_acc = (test_preds == test_labels).float().mean().item()

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_test_preds = test_preds.clone()

        if (epoch + 1) % 5 == 0:
            print(f"  Probe epoch {epoch+1}: test_acc={test_acc:.4f}")

    print(f"\nBest test accuracy: {best_test_acc:.4f}")

    # Per-class accuracy
    class_names = ['healthy', 'outer_race', 'inner_race', 'ball']
    per_class_acc = {}
    for i, name in enumerate(class_names):
        mask = test_labels == i
        if mask.sum() > 0:
            acc = (best_test_preds[mask] == test_labels[mask]).float().mean().item()
            per_class_acc[name] = acc
            print(f"  {name:12s}: {acc:.4f} ({mask.sum().item()} samples)")

    return {'test_acc': best_test_acc, 'per_class_acc': per_class_acc}


def quick_diagnose(model, device, embed_dim):
    """
    Run quick predictor collapse diagnostic inline.
    Returns dict with key metrics.
    """
    model.eval()
    B = 8
    x = torch.randn(B, 3, 4096).to(device)

    with torch.no_grad():
        n_patches = model.n_patches  # Use model's actual n_patches
        n_context = n_patches // 2  # Use half as context

        context_indices = torch.arange(n_context).unsqueeze(0).expand(B, -1).to(device)
        all_patches = model.encoder(x, return_all_tokens=True)[:, 1:]
        context_embeds = all_patches[:, :n_context]

        # Predict each mask position individually
        single_preds = []
        for pos in range(n_context, n_patches):
            mask_idx = torch.tensor([[pos]]).expand(B, -1).to(device)
            pred = model.predictor(context_embeds, context_indices, mask_idx)
            single_preds.append(pred[:, 0])

        single_preds = torch.stack(single_preds, dim=1)  # (B, 8, D)

        # Key metrics
        pred_var_across_pos = single_preds.var(dim=1).mean().item()

        # Full forward pass for spread ratio
        loss, predictions, targets = model(x)
        pred_std = predictions.std(dim=1).mean().item()
        targ_std = targets.std(dim=1).mean().item()
        spread_ratio = pred_std / (targ_std + 1e-8)

        # Cosine similarity
        pred_norm = torch.nn.functional.normalize(predictions, dim=-1)
        targ_norm = torch.nn.functional.normalize(targets, dim=-1)
        cos_sim = (pred_norm * targ_norm).sum(dim=-1).mean().item()

    metrics = {
        'pred_var_across_pos': pred_var_across_pos,
        'spread_ratio': spread_ratio,
        'cos_sim': cos_sim,
        'collapse': pred_var_across_pos < 0.001,
    }
    return metrics


def train(config: dict, device: torch.device):
    print("=" * 60)
    print("MECHANICAL-JEPA V2 TRAINING")
    print("=" * 60)

    # Build run name
    run_name = config.get('run_name') or (
        f"v2_{config['predictor_pos']}"
        f"_pd{config['predictor_depth']}"
        f"_{config['loss_fn']}"
        f"_var{config['var_reg_lambda']}"
        f"_vic{config['vicreg_lambda']}"
        f"_sep{int(config['separate_mask_tokens'])}"
        f"_s{config['seed']}"
    )

    print(f"\nRun: {run_name}")
    for k, v in config.items():
        if not k.startswith('wandb') and k != 'run_name':
            print(f"  {k}: {v}")

    use_wandb = not config.get('no_wandb', False)
    if use_wandb:
        wandb.init(
            project=config.get('wandb_project', 'mechanical-jepa'),
            config={k: v for k, v in config.items() if not k.startswith('wandb') and k != 'no_wandb'},
            name=run_name,
        )
        print(f"\nWandB: {wandb.run.url}")

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

    print("\nCreating model (V2)...")
    model = MechanicalJEPAV2(
        n_channels=config['n_channels'],
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

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")

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
            optimizer.step()
            model.update_ema()
            total_loss += loss.item()

        avg_loss = total_loss / n_batches
        history['loss'].append(avg_loss)
        epoch_time = time.time() - start_time

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{config['epochs']}: loss={avg_loss:.4f}")

        if use_wandb:
            wandb.log({'epoch': epoch + 1, 'loss': avg_loss, 'lr': lr})

    total_time = time.time() - start_time
    print(f"\nTraining done in {total_time/60:.1f}min")

    # Run collapse diagnostic
    print("\nRunning predictor collapse diagnostic...")
    diag = quick_diagnose(model, device, config['embed_dim'])
    print(f"  pred_var_across_pos: {diag['pred_var_across_pos']:.6f}")
    print(f"  spread_ratio: {diag['spread_ratio']:.4f}")
    print(f"  cos_sim: {diag['cos_sim']:.4f}")
    print(f"  collapsed: {diag['collapse']}")
    if diag['collapse']:
        print("  [X] PREDICTOR STILL COLLAPSED")
    else:
        print("  [OK] Predictor NOT collapsed")

    if use_wandb:
        wandb.log({
            'diag/pred_var_across_pos': diag['pred_var_across_pos'],
            'diag/spread_ratio': diag['spread_ratio'],
            'diag/cos_sim': diag['cos_sim'],
            'diag/collapsed': int(diag['collapse']),
        })

    # Linear probe evaluation
    probe_results = train_linear_probe(model, train_loader, test_loader, config, device)

    if use_wandb:
        wandb.log({
            'probe/test_acc': probe_results['test_acc'],
            **{f'probe/{k}_acc': v for k, v in probe_results['per_class_acc'].items()},
        })
        wandb.summary['final_test_acc'] = probe_results['test_acc']
        wandb.summary['predictor_collapsed'] = int(diag['collapse'])

    # Save checkpoint
    if not config.get('no_save', False):
        save_dir = Path(config['save_dir'])
        save_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        checkpoint_path = save_dir / f'jepa_v2_{timestamp}.pt'

        torch.save({
            'model_state_dict': model.state_dict(),
            'config': config,
            'history': history,
            'probe_results': probe_results,
            'diag': diag,
        }, checkpoint_path)
        print(f"\nSaved: {checkpoint_path}")

        # Skip artifact upload to conserve disk space (metrics are in wandb)
        # if use_wandb:
        #     artifact = wandb.Artifact('model-v2', type='model')
        #     artifact.add_file(str(checkpoint_path))
        #     wandb.log_artifact(artifact)

    if use_wandb:
        wandb.finish()

    return model, probe_results, diag


def main():
    args = get_args()
    config = DEFAULT_CONFIG.copy()

    if args.epochs is not None: config['epochs'] = args.epochs
    if args.batch_size is not None: config['batch_size'] = args.batch_size
    if args.lr is not None: config['lr'] = args.lr
    if args.embed_dim is not None: config['embed_dim'] = args.embed_dim
    if args.encoder_depth is not None: config['encoder_depth'] = args.encoder_depth
    if args.predictor_depth is not None: config['predictor_depth'] = args.predictor_depth
    if args.mask_ratio is not None: config['mask_ratio'] = args.mask_ratio
    if args.patch_size is not None: config['patch_size'] = args.patch_size
    if args.seed is not None: config['seed'] = args.seed
    if args.dataset is not None: config['dataset_filter'] = args.dataset
    if args.predictor_pos is not None: config['predictor_pos'] = args.predictor_pos
    if args.separate_mask_tokens: config['separate_mask_tokens'] = True
    if args.loss_fn is not None: config['loss_fn'] = args.loss_fn
    if args.var_reg is not None: config['var_reg_lambda'] = args.var_reg
    if args.vicreg is not None: config['vicreg_lambda'] = args.vicreg
    if args.no_save: config['no_save'] = True
    if args.no_wandb: config['no_wandb'] = True
    config['wandb_project'] = args.wandb_project
    if args.run_name: config['run_name'] = args.run_name

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    model, probe_results, diag = train(config, device)

    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Test accuracy: {probe_results['test_acc']:.4f}")
    for cls, acc in probe_results['per_class_acc'].items():
        print(f"  {cls}: {acc:.4f}")
    print(f"Predictor collapsed: {diag['collapse']}")
    print(f"Spread ratio: {diag['spread_ratio']:.4f} (>0.5 is good)")


if __name__ == '__main__':
    main()
