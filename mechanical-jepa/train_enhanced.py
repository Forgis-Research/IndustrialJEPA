"""
Train with enhanced JEPA (structured masking).

Quick script to test if temporal_block masking improves transfer.
"""

import sys
import torch
import torch.optim as optim
import numpy as np
from pathlib import Path

from src.data import create_dataloaders
from src.models import MechanicalJEPAEnhanced
from train import cosine_scheduler, train_epoch, train_linear_probe

# Config
config = {
    'data_dir': 'data/bearings',
    'dataset_filter': 'cwru',
    'batch_size': 32,
    'window_size': 4096,
    'stride': 2048,
    'n_channels': 3,
    'test_ratio': 0.2,
    'num_workers': 0,
    'patch_size': 256,
    'embed_dim': 256,
    'encoder_depth': 4,
    'predictor_depth': 2,
    'n_heads': 4,
    'mask_ratio': 0.5,
    'ema_decay': 0.996,
    'masking_strategy': sys.argv[1] if len(sys.argv) > 1 else 'temporal_block',
    'epochs': int(sys.argv[2]) if len(sys.argv) > 2 else 30,
    'lr': 1e-4,
    'weight_decay': 0.05,
    'warmup_epochs': 5,
    'min_lr': 1e-6,
    'probe_epochs': 20,
    'probe_lr': 1e-3,
    'seed': int(sys.argv[3]) if len(sys.argv) > 3 else 42,
    'log_interval': 10,
}

print("="*60)
print(f"ENHANCED JEPA TRAINING")
print("="*60)
print(f"Masking strategy: {config['masking_strategy']}")
print(f"Epochs: {config['epochs']}")
print(f"Seed: {config['seed']}")

# Set seed
torch.manual_seed(config['seed'])
np.random.seed(config['seed'])

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# Data
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

# Model
print("\nCreating enhanced model...")
model = MechanicalJEPAEnhanced(
    n_channels=config['n_channels'],
    window_size=config['window_size'],
    patch_size=config['patch_size'],
    embed_dim=config['embed_dim'],
    encoder_depth=config['encoder_depth'],
    predictor_depth=config['predictor_depth'],
    n_heads=config['n_heads'],
    mask_ratio=config['mask_ratio'],
    ema_decay=config['ema_decay'],
    masking_strategy=config['masking_strategy'],
    block_size=4,
).to(device)

n_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {n_params:,}")

# Optimizer
optimizer = optim.AdamW(
    model.parameters(),
    lr=config['lr'],
    weight_decay=config['weight_decay'],
)

# LR schedule
lr_schedule = cosine_scheduler(
    config['lr'], config['min_lr'],
    config['epochs'], config['warmup_epochs']
)

# Training
print(f"\nTraining for {config['epochs']} epochs...")
history = {'loss': [], 'lr': []}

for epoch in range(config['epochs']):
    avg_loss = train_epoch(
        model, train_loader, optimizer,
        epoch, config, device, lr_schedule
    )
    history['loss'].append(avg_loss)
    history['lr'].append(lr_schedule[min(epoch, len(lr_schedule) - 1)])
    print(f"Epoch {epoch+1}/{config['epochs']}: loss={avg_loss:.4f}")

# Evaluation
print("\nEvaluating...")
probe_results = train_linear_probe(model, train_loader, test_loader, config, device)

print("\n" + "="*60)
print(f"RESULTS")
print("="*60)
print(f"Test accuracy: {probe_results['test_acc']:.4f}")
print("\nPer-class:")
for cls, acc in probe_results['per_class_acc'].items():
    print(f"  {cls}: {acc:.4f}")

# Save
save_path = Path('checkpoints') / f"jepa_enhanced_{config['masking_strategy']}_{config['seed']}.pt"
save_path.parent.mkdir(exist_ok=True)

torch.save({
    'model_state_dict': model.state_dict(),
    'config': config,
    'history': history,
    'probe_results': probe_results,
    'data_info': data_info,
}, save_path)

print(f"\nSaved to {save_path}")
