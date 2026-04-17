"""
Mechanical-JEPA: Joint Embedding Predictive Architecture for bearing signals.

Architecture follows I-JEPA/Brain-JEPA:
- Encoder: Transformer on 1D signal patches
- Predictor: Lightweight transformer predicting masked embeddings
- Target encoder: EMA of main encoder
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, List
import copy


# =============================================================================
# Patch Embedding
# =============================================================================

class PatchEmbed1D(nn.Module):
    """
    Converts (B, C, T) signal to (B, N, D) patch embeddings.

    Example: (32, 3, 4096) with patch_size=256 -> (32, 16, embed_dim)
    """

    def __init__(
        self,
        n_channels: int = 3,
        patch_size: int = 256,
        embed_dim: int = 256,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        # Linear projection of flattened patches
        self.proj = nn.Linear(n_channels * patch_size, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T) input signal
        Returns:
            patches: (B, N, D) patch embeddings where N = T // patch_size
        """
        B, C, T = x.shape
        assert T % self.patch_size == 0, f"Signal length {T} not divisible by patch_size {self.patch_size}"

        n_patches = T // self.patch_size

        # Reshape to patches: (B, C, T) -> (B, N, C*patch_size)
        x = x.reshape(B, C, n_patches, self.patch_size)
        x = x.permute(0, 2, 1, 3)  # (B, N, C, patch_size)
        x = x.reshape(B, n_patches, -1)  # (B, N, C*patch_size)

        # Project to embedding dimension
        x = self.proj(x)  # (B, N, D)

        return x


# =============================================================================
# Transformer Blocks
# =============================================================================

class TransformerBlock(nn.Module):
    """Standard transformer block with pre-norm."""

    def __init__(
        self,
        embed_dim: int,
        n_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim, n_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention with residual
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, key_padding_mask=mask)
        x = x + attn_out

        # MLP with residual
        x = x + self.mlp(self.norm2(x))

        return x


# =============================================================================
# JEPA Encoder
# =============================================================================

class JEPAEncoder(nn.Module):
    """
    Encoder for Mechanical-JEPA.

    Takes signal patches and produces embeddings.
    Similar to ViT encoder but for 1D signals.
    """

    def __init__(
        self,
        n_channels: int = 3,
        window_size: int = 4096,
        patch_size: int = 256,
        embed_dim: int = 256,
        depth: int = 6,
        n_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.n_patches = window_size // patch_size
        self.embed_dim = embed_dim

        # Patch embedding
        self.patch_embed = PatchEmbed1D(n_channels, patch_size, embed_dim)

        # Positional embedding (learnable)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # CLS token (optional, for downstream classification)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, n_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        mask_indices: Optional[torch.Tensor] = None,
        return_all_tokens: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, C, T) input signal
            mask_indices: (B, n_mask) indices of patches to mask (for context encoder)
            return_all_tokens: If True, return all tokens; else return CLS token

        Returns:
            If mask_indices provided: (B, n_visible, D) embeddings of visible patches
            If return_all_tokens: (B, N+1, D) all embeddings including CLS
            Else: (B, D) CLS token embedding
        """
        B = x.shape[0]

        # Patch embedding
        x = self.patch_embed(x)  # (B, N, D)

        # Add positional embedding
        x = x + self.pos_embed

        # If masking, keep only visible patches
        if mask_indices is not None:
            # Create visible mask (True = visible)
            visible_mask = torch.ones(B, self.n_patches, dtype=torch.bool, device=x.device)
            visible_mask.scatter_(1, mask_indices, False)

            # Gather visible patches
            visible_indices = visible_mask.nonzero(as_tuple=True)[1].reshape(B, -1)
            x = torch.gather(x, 1, visible_indices.unsqueeze(-1).expand(-1, -1, self.embed_dim))

        # Prepend CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        if return_all_tokens:
            return x  # (B, N+1, D) or (B, n_visible+1, D)
        else:
            return x[:, 0]  # (B, D) CLS token


# =============================================================================
# JEPA Predictor
# =============================================================================

class JEPAPredictor(nn.Module):
    """
    Predictor for Mechanical-JEPA.

    Takes context embeddings and mask tokens, predicts target embeddings.
    """

    def __init__(
        self,
        n_patches: int = 16,
        embed_dim: int = 256,
        predictor_dim: int = 128,
        depth: int = 3,
        n_heads: int = 4,
    ):
        super().__init__()
        self.n_patches = n_patches
        self.embed_dim = embed_dim

        # Project encoder dim to predictor dim
        self.input_proj = nn.Linear(embed_dim, predictor_dim)

        # Mask token (learnable)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        # Positional embedding for predictor
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches, predictor_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(predictor_dim, n_heads, mlp_ratio=4.0, dropout=0.1)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(predictor_dim)

        # Project back to encoder dim for loss computation
        self.output_proj = nn.Linear(predictor_dim, embed_dim)

    def forward(
        self,
        context_embeds: torch.Tensor,
        context_indices: torch.Tensor,
        mask_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            context_embeds: (B, n_context, D) embeddings of visible patches
            context_indices: (B, n_context) indices of visible patches
            mask_indices: (B, n_mask) indices of masked patches

        Returns:
            predictions: (B, n_mask, D) predicted embeddings for masked patches
        """
        B = context_embeds.shape[0]
        n_context = context_embeds.shape[1]
        n_mask = mask_indices.shape[1]

        # Project context to predictor dimension
        context = self.input_proj(context_embeds)  # (B, n_context, predictor_dim)

        # Add positional embeddings to context
        context_pos = torch.gather(
            self.pos_embed.expand(B, -1, -1),
            1,
            context_indices.unsqueeze(-1).expand(-1, -1, context.shape[-1])
        )
        context = context + context_pos

        # Create mask tokens with positional embeddings
        mask_tokens = self.mask_token.expand(B, n_mask, -1)
        mask_pos = torch.gather(
            self.pos_embed.expand(B, -1, -1),
            1,
            mask_indices.unsqueeze(-1).expand(-1, -1, mask_tokens.shape[-1])
        )
        mask_tokens = mask_tokens + mask_pos

        # Concatenate context and mask tokens
        x = torch.cat([context, mask_tokens], dim=1)  # (B, n_context + n_mask, predictor_dim)

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        # Extract predictions for masked positions (last n_mask tokens)
        predictions = x[:, n_context:]  # (B, n_mask, predictor_dim)

        # Project to encoder dimension
        predictions = self.output_proj(predictions)  # (B, n_mask, D)

        return predictions


# =============================================================================
# Full JEPA Model
# =============================================================================

class MechanicalJEPA(nn.Module):
    """
    Complete Mechanical-JEPA model.

    Components:
    - Context encoder: Encodes visible patches
    - Target encoder: EMA of context encoder, encodes all patches for targets
    - Predictor: Predicts masked patch embeddings from context
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
    ):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.ema_decay = ema_decay
        self.n_patches = window_size // patch_size

        # Context encoder (trainable)
        self.encoder = JEPAEncoder(
            n_channels=n_channels,
            window_size=window_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=encoder_depth,
            n_heads=n_heads,
        )

        # Target encoder (EMA of context encoder)
        self.target_encoder = copy.deepcopy(self.encoder)
        for param in self.target_encoder.parameters():
            param.requires_grad = False

        # Predictor
        self.predictor = JEPAPredictor(
            n_patches=self.n_patches,
            embed_dim=embed_dim,
            predictor_dim=embed_dim // 2,
            depth=predictor_depth,
            n_heads=n_heads,
        )

    @torch.no_grad()
    def _update_target_encoder(self):
        """Update target encoder with EMA."""
        for param_q, param_k in zip(self.encoder.parameters(), self.target_encoder.parameters()):
            param_k.data = self.ema_decay * param_k.data + (1 - self.ema_decay) * param_q.data

    def _generate_mask(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate random mask indices.

        Returns:
            mask_indices: (B, n_mask) indices of patches to mask
            context_indices: (B, n_context) indices of visible patches
        """
        n_mask = int(self.n_patches * self.mask_ratio)
        n_context = self.n_patches - n_mask

        # Random permutation for each sample
        indices = torch.stack([
            torch.randperm(self.n_patches, device=device)
            for _ in range(batch_size)
        ])

        mask_indices = indices[:, :n_mask]
        context_indices = indices[:, n_mask:]

        return mask_indices, context_indices

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for training.

        Args:
            x: (B, C, T) input signal

        Returns:
            loss: scalar loss
            predictions: (B, n_mask, D) predicted embeddings
            targets: (B, n_mask, D) target embeddings
        """
        B = x.shape[0]
        device = x.device

        # Generate mask
        mask_indices, context_indices = self._generate_mask(B, device)

        # Get target embeddings (all patches, no masking)
        with torch.no_grad():
            target_embeds = self.target_encoder(x, return_all_tokens=True)[:, 1:]  # Remove CLS, (B, N, D)
            # Gather target embeddings for masked patches
            targets = torch.gather(
                target_embeds,
                1,
                mask_indices.unsqueeze(-1).expand(-1, -1, target_embeds.shape[-1])
            )

        # Get context embeddings (visible patches only)
        context_embeds = self.encoder(x, mask_indices=mask_indices, return_all_tokens=True)[:, 1:]  # Remove CLS

        # Predict masked patch embeddings
        predictions = self.predictor(context_embeds, context_indices, mask_indices)

        # L2 loss on normalized embeddings (like I-JEPA)
        predictions_norm = F.normalize(predictions, dim=-1)
        targets_norm = F.normalize(targets, dim=-1)
        loss = F.mse_loss(predictions_norm, targets_norm)

        return loss, predictions, targets

    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Get CLS embeddings for downstream tasks."""
        return self.encoder(x, return_all_tokens=False)

    def train_step(self, x: torch.Tensor) -> torch.Tensor:
        """Single training step with EMA update."""
        loss, _, _ = self.forward(x)
        return loss

    @torch.no_grad()
    def update_ema(self):
        """Call after optimizer step to update target encoder."""
        self._update_target_encoder()


# =============================================================================
# Test
# =============================================================================

if __name__ == '__main__':
    # Test model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = MechanicalJEPA(
        n_channels=3,
        window_size=4096,
        patch_size=256,
        embed_dim=256,
        encoder_depth=4,
        predictor_depth=2,
        mask_ratio=0.5,
    ).to(device)

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {n_params:,}")
    print(f"Trainable parameters: {n_trainable:,}")

    # Test forward pass
    x = torch.randn(8, 3, 4096).to(device)

    loss, preds, targets = model(x)
    print(f"\nForward pass:")
    print(f"  Input shape: {x.shape}")
    print(f"  Loss: {loss.item():.4f}")
    print(f"  Predictions shape: {preds.shape}")
    print(f"  Targets shape: {targets.shape}")

    # Test embedding extraction
    embeds = model.get_embeddings(x)
    print(f"\nEmbedding extraction:")
    print(f"  Embedding shape: {embeds.shape}")

    print("\nModel test PASSED!")
