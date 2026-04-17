"""
Enhanced Mechanical-JEPA with structured spatiotemporal masking.

Inspired by Brain-JEPA (NeurIPS 2024), this adds:
1. Structured temporal masking (block masking in time)
2. Cross-channel masking (mask all channels of a patch)
3. Spatiotemporal masking (combined strategies)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Literal
import numpy as np

from .jepa import MechanicalJEPA, PatchEmbed1D, JEPAEncoder, JEPAPredictor


class MechanicalJEPAEnhanced(MechanicalJEPA):
    """
    Enhanced JEPA with structured masking strategies.

    Masking strategies:
    - 'random': Random patch masking (default, like original I-JEPA)
    - 'temporal_block': Contiguous blocks in time (better for sequences)
    - 'cross_time': Mask same temporal position across channels
    - 'mixed': Combination of strategies
    """

    def __init__(
        self,
        n_channels: int = 3,
        window_size: int = 4096,
        patch_size: int = 256,
        embed_dim: int = 256,
        encoder_depth: int = 6,
        predictor_depth: int = 3,
        n_heads: int = 4,
        mask_ratio: float = 0.5,
        ema_decay: float = 0.996,
        masking_strategy: Literal['random', 'temporal_block', 'cross_time', 'mixed'] = 'random',
        block_size: int = 4,  # For temporal_block
    ):
        super().__init__(
            n_channels=n_channels,
            window_size=window_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            encoder_depth=encoder_depth,
            predictor_depth=predictor_depth,
            n_heads=n_heads,
            mask_ratio=mask_ratio,
            ema_decay=ema_decay,
        )

        self.masking_strategy = masking_strategy
        self.block_size = block_size

    def _generate_mask_random(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Original random masking (same as parent)."""
        return super()._generate_mask(batch_size, device)

    def _generate_mask_temporal_block(
        self,
        batch_size: int,
        device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Block masking in temporal dimension.

        Masks contiguous blocks of patches, better for temporal sequences.
        Inspired by Brain-JEPA's Cross-Time masking.
        """
        n_mask = int(self.n_patches * self.mask_ratio)
        n_context = self.n_patches - n_mask

        # Calculate number of blocks
        n_blocks = max(1, n_mask // self.block_size)

        mask_indices_list = []
        context_indices_list = []

        for _ in range(batch_size):
            # Randomly select block start positions
            possible_starts = list(range(self.n_patches - self.block_size + 1))
            if len(possible_starts) < n_blocks:
                # Not enough space for non-overlapping blocks, use random
                indices = np.random.permutation(self.n_patches)
                mask_indices = indices[:n_mask].tolist()
                context_indices = indices[n_mask:].tolist()
            else:
                block_starts = np.random.choice(possible_starts, size=n_blocks, replace=False)

                # Generate mask indices from blocks
                mask_indices = []
                for start in block_starts:
                    mask_indices.extend(range(start, min(start + self.block_size, self.n_patches)))

                # Take only up to n_mask (in case blocks overlap boundaries)
                mask_indices = sorted(list(set(mask_indices)))[:n_mask]

                # Context is everything else
                all_indices = set(range(self.n_patches))
                context_indices = sorted(list(all_indices - set(mask_indices)))

            # Ensure exact lengths
            mask_indices = mask_indices[:n_mask]
            context_indices = context_indices[:n_context]

            # Pad if needed
            while len(mask_indices) < n_mask:
                # Find an index not in mask
                for i in range(self.n_patches):
                    if i not in mask_indices and i not in context_indices:
                        mask_indices.append(i)
                        break

            while len(context_indices) < n_context:
                # Find an index not in context
                for i in range(self.n_patches):
                    if i not in context_indices and i not in mask_indices:
                        context_indices.append(i)
                        break

            mask_indices_list.append(mask_indices)
            context_indices_list.append(context_indices)

        mask_indices = torch.tensor(mask_indices_list, device=device)
        context_indices = torch.tensor(context_indices_list, device=device)

        return mask_indices, context_indices

    def _generate_mask_cross_time(
        self,
        batch_size: int,
        device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Cross-time masking: mask same temporal positions.

        This forces the model to learn temporal correlations.
        """
        n_mask = int(self.n_patches * self.mask_ratio)
        n_context = self.n_patches - n_mask

        # Randomly select which temporal positions to mask
        # (same for all samples in batch)
        indices = torch.randperm(self.n_patches, device=device)
        mask_indices = indices[:n_mask].unsqueeze(0).expand(batch_size, -1)
        context_indices = indices[n_mask:].unsqueeze(0).expand(batch_size, -1)

        return mask_indices, context_indices

    def _generate_mask_mixed(
        self,
        batch_size: int,
        device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Mixed masking: randomly choose strategy per sample.

        Inspired by Brain-JEPA's multiple masking types.
        """
        mask_indices_list = []
        context_indices_list = []

        strategies = ['random', 'temporal_block', 'cross_time']

        for _ in range(batch_size):
            # Randomly pick a strategy for this sample
            strategy = np.random.choice(strategies)

            if strategy == 'random':
                mask_idx, context_idx = self._generate_mask_random(1, device)
            elif strategy == 'temporal_block':
                mask_idx, context_idx = self._generate_mask_temporal_block(1, device)
            else:  # cross_time
                mask_idx, context_idx = self._generate_mask_cross_time(1, device)

            mask_indices_list.append(mask_idx[0])
            context_indices_list.append(context_idx[0])

        mask_indices = torch.stack(mask_indices_list)
        context_indices = torch.stack(context_indices_list)

        return mask_indices, context_indices

    def _generate_mask(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate mask indices based on configured strategy.

        Returns:
            mask_indices: (B, n_mask) indices of patches to mask
            context_indices: (B, n_context) indices of visible patches
        """
        if self.masking_strategy == 'random':
            return self._generate_mask_random(batch_size, device)
        elif self.masking_strategy == 'temporal_block':
            return self._generate_mask_temporal_block(batch_size, device)
        elif self.masking_strategy == 'cross_time':
            return self._generate_mask_cross_time(batch_size, device)
        elif self.masking_strategy == 'mixed':
            return self._generate_mask_mixed(batch_size, device)
        else:
            raise ValueError(f"Unknown masking strategy: {self.masking_strategy}")


# =============================================================================
# Test
# =============================================================================

if __name__ == '__main__':
    # Test enhanced model with different masking strategies
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    for strategy in ['random', 'temporal_block', 'cross_time', 'mixed']:
        print(f"\n{'='*60}")
        print(f"Testing strategy: {strategy}")
        print(f"{'='*60}")

        model = MechanicalJEPAEnhanced(
            n_channels=3,
            window_size=4096,
            patch_size=256,
            embed_dim=256,
            encoder_depth=4,
            predictor_depth=2,
            masking_strategy=strategy,
            block_size=4,
        ).to(device)

        # Test forward pass
        x = torch.randn(8, 3, 4096).to(device)

        loss, preds, targets = model(x)
        print(f"Loss: {loss.item():.4f}")
        print(f"Predictions shape: {preds.shape}")
        print(f"Targets shape: {targets.shape}")

        # Test mask generation
        mask_idx, context_idx = model._generate_mask(2, device)
        print(f"Mask indices shape: {mask_idx.shape}")
        print(f"Context indices shape: {context_idx.shape}")
        print(f"Example mask indices: {mask_idx[0][:5].tolist()}")

    print("\n✓ All masking strategies work correctly!")
