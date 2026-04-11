#!/usr/bin/env python
"""Debug Voraus dataset loading."""

import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

from industrialjepa.data.factorynet import FactoryNetDataset, FactoryNetConfig

print('Creating config...')
config = FactoryNetConfig(data_source='voraus', max_episodes=500, window_size=256, stride=128)

print('Loading train split...')
ds1 = FactoryNetDataset(config, split='train')
print(f'Train: {len(ds1)} windows')

print('Loading test split...')
ds2 = FactoryNetDataset(config, split='test')
print(f'Test: {len(ds2)} windows')

print('SUCCESS - both splits loaded')
