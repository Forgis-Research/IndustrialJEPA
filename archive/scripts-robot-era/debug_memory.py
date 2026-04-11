#!/usr/bin/env python
"""Debug memory usage when loading both datasets."""

import psutil
import os

def mem_gb():
    return psutil.Process(os.getpid()).memory_info().rss / 1024**3

print(f'Memory before: {mem_gb():.2f} GB')

from industrialjepa.data.factorynet import FactoryNetDataset, FactoryNetConfig

# Load AURSAD first (like transfer script does)
config = FactoryNetConfig(data_source='aursad', max_episodes=500, window_size=256, stride=128)

print('Loading AURSAD train...')
ds1 = FactoryNetDataset(config, split='train')
print(f'After AURSAD train: {mem_gb():.2f} GB')

print('Loading AURSAD test...')
ds2 = FactoryNetDataset(config, split='test')
print(f'After AURSAD test: {mem_gb():.2f} GB')

# Now load Voraus
config2 = FactoryNetConfig(data_source='voraus', max_episodes=500, window_size=256, stride=128)

print('Loading Voraus train...')
ds3 = FactoryNetDataset(config2, split='train')
print(f'After Voraus train: {mem_gb():.2f} GB')

print('Loading Voraus test...')
ds4 = FactoryNetDataset(config2, split='test')
print(f'After Voraus test: {mem_gb():.2f} GB')

print('SUCCESS - all datasets loaded')
