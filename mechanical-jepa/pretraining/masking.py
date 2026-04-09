"""
Masking strategies for JEPA pretraining.
"""

import numpy as np
import torch
from typing import List, Tuple


def random_mask(n_patches: int, mask_ratio: float = 0.625,
                rng: np.random.RandomState = None) -> List[int]:
    """
    Random patch masking (V8 default).
    Returns list of masked patch indices.
    """
    n_mask = int(n_patches * mask_ratio)
    if rng is None:
        rng = np.random.default_rng()
    all_idx = list(range(n_patches))
    rng.shuffle(all_idx) if hasattr(rng, 'shuffle') else np.random.shuffle(all_idx)
    return sorted(all_idx[:n_mask])


def contiguous_block_mask(n_patches: int, block_size: int = 10) -> List[int]:
    """
    Contiguous block masking: mask a single contiguous block.
    Block start is randomized. Better for temporal signals.
    """
    max_start = n_patches - block_size
    if max_start <= 0:
        return list(range(n_patches))
    start = np.random.randint(0, max_start + 1)
    return list(range(start, start + block_size))


def multi_block_mask(n_patches: int, n_blocks: int = 3,
                     target_ratio: float = 0.625) -> List[int]:
    """
    Multiple contiguous block masks.
    Total mask ratio approximately target_ratio.
    """
    n_mask = int(n_patches * target_ratio)
    block_size = n_mask // n_blocks

    masked = set()
    for _ in range(n_blocks * 3):  # retry to get target count
        if len(masked) >= n_mask:
            break
        max_start = n_patches - block_size
        if max_start <= 0:
            break
        start = np.random.randint(0, max_start + 1)
        for i in range(start, min(start + block_size, n_patches)):
            masked.add(i)

    return sorted(list(masked))[:n_mask]


def get_mask_fn(strategy: str = 'random'):
    """Return mask function by strategy name."""
    if strategy == 'random':
        return lambda n: random_mask(n)
    elif strategy == 'block':
        return lambda n: contiguous_block_mask(n)
    elif strategy == 'multi_block':
        return lambda n: multi_block_mask(n)
    else:
        raise ValueError(f"Unknown masking strategy: {strategy}")
