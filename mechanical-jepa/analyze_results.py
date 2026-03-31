"""Analyze multi-seed results."""
import torch
import numpy as np
from pathlib import Path

checkpoints = [
    Path('checkpoints/jepa_20260330_232646.pt'),
    Path('checkpoints/jepa_20260331_080504.pt'),
]

results = []
for cp in checkpoints:
    ckpt = torch.load(cp, map_location='cpu', weights_only=False)
    config = ckpt['config']
    probe = ckpt['probe_results']
    results.append({
        'seed': config['seed'],
        'test_acc': probe['test_acc'],
        'train_acc': probe['train_acc'],
        'per_class': probe['per_class_acc'],
    })

print('Multi-Seed Results (30 epochs):')
print('='*60)
print(f"{'Seed':<10} {'Test Acc':<15} {'Train Acc':<15}")
print('='*60)

for r in results:
    print(f"{r['seed']:<10} {r['test_acc']:.4f} ({r['test_acc']*100:.1f}%)  {r['train_acc']:.4f} ({r['train_acc']*100:.1f}%)")

test_accs = [r['test_acc'] for r in results]

print('='*60)
print(f"Mean:      {np.mean(test_accs):.4f} ({np.mean(test_accs)*100:.1f}%)")
print(f"Std:       {np.std(test_accs):.4f} ({np.std(test_accs)*100:.1f}%)")
print(f"Min:       {np.min(test_accs):.4f} ({np.min(test_accs)*100:.1f}%)")
print(f"Max:       {np.max(test_accs):.4f} ({np.max(test_accs)*100:.1f}%)")

print('\nPer-Class Breakdown:')
print('='*60)

# Get all class names
all_classes = set()
for r in results:
    all_classes.update(r['per_class'].keys())

for cls in sorted(all_classes):
    accs = [r['per_class'].get(cls, 0) for r in results]
    print(f"{cls:15s}: {np.mean(accs):.4f} ± {np.std(accs):.4f}  (min={np.min(accs):.2f}, max={np.max(accs):.2f})")
