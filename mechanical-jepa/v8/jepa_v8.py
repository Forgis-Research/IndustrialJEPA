"""
JEPA V8: Adapted for single-channel 1024-sample windows at 12,800 Hz.

Changes from V2:
- n_channels=1 (was 3)
- window_size=1024 (was 4096)
- patch_size=64 (was 256)
- embed_dim=256 (was 512)
- predictor_dim=128 (was 256)
- 16 patches, mask 10 (62.5%)
- L1 loss + variance regularization (anti-collapse)
- Sinusoidal positional encoding (not learnable)
- EMA momentum=0.996
"""

import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple


# ============================================================
# SINUSOIDAL POSITIONAL ENCODING
# ============================================================

def sinusoidal_pe(n_positions: int, d_model: int) -> torch.Tensor:
    """Returns (1, n_positions, d_model) fixed sinusoidal PE."""
    pe = torch.zeros(n_positions, d_model)
    pos = torch.arange(0, n_positions, dtype=torch.float).unsqueeze(1)
    div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe.unsqueeze(0)  # (1, N, D)


# ============================================================
# TRANSFORMER BLOCK
# ============================================================

class TransformerBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        mlp_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention with pre-norm
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        # MLP with pre-norm
        x = x + self.mlp(self.norm2(x))
        return x


# ============================================================
# PATCH EMBEDDING (1D)
# ============================================================

class PatchEmbed1D(nn.Module):
    """
    Partition a 1D signal into non-overlapping patches and project each patch
    to an embedding vector.

    Input: (B, C, L) where C=1, L=1024
    Output: (B, N, D) where N = L // patch_size = 16, D = embed_dim
    """
    def __init__(self, n_channels: int = 1, patch_size: int = 64, embed_dim: int = 256):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Linear(n_channels * patch_size, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, L)
        B, C, L = x.shape
        n_patches = L // self.patch_size
        # (B, C, L) → (B, n_patches, C*patch_size)
        x = x.unfold(2, self.patch_size, self.patch_size)  # (B, C, n_patches, patch_size)
        x = x.permute(0, 2, 1, 3).contiguous()             # (B, n_patches, C, patch_size)
        x = x.view(B, n_patches, -1)                        # (B, n_patches, C*patch_size)
        return self.proj(x)                                  # (B, n_patches, D)


# ============================================================
# ENCODER
# ============================================================

class JEPAEncoderV8(nn.Module):
    """
    Context encoder: processes visible (non-masked) patches.
    Also used as target encoder (EMA copy).
    """
    def __init__(
        self,
        n_channels: int = 1,
        window_size: int = 1024,
        patch_size: int = 64,
        embed_dim: int = 256,
        depth: int = 4,
        n_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.n_patches = window_size // patch_size
        self.embed_dim = embed_dim

        self.patch_embed = PatchEmbed1D(n_channels, patch_size, embed_dim)

        # Fixed sinusoidal PE — no learnable PE to avoid collapse
        pe = sinusoidal_pe(self.n_patches, embed_dim)
        self.register_buffer('pos_embed', pe)  # (1, N, D)

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, n_heads, mlp_ratio=4.0, dropout=dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        mask_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, C, L) input signal
            mask_indices: (B, n_mask) indices of masked patches.
                          If None, process all patches (for target encoder).

        Returns:
            (B, n_patches_visible, D) if mask_indices given
            (B, n_patches, D) if no mask
        """
        # Embed all patches
        patches = self.patch_embed(x)  # (B, N, D)
        patches = patches + self.pos_embed  # add positional encoding

        if mask_indices is not None:
            # Remove masked patches — keep only context patches
            B, N, D = patches.shape
            # Create visibility mask
            visible_mask = torch.ones(B, N, dtype=torch.bool, device=x.device)
            for b in range(B):
                visible_mask[b, mask_indices[b]] = False
            # Extract visible patches per batch
            # Since mask is same for all batch items (uniform), do it efficiently
            # Actually mask_indices varies per batch item → use gather approach
            # Get context indices: complement of mask_indices
            n_mask = mask_indices.shape[1]
            n_context = N - n_mask
            all_idx = torch.arange(N, device=x.device).unsqueeze(0).expand(B, -1)
            # Build context indices by excluding mask indices
            context_patches = []
            for b in range(B):
                keep = [i for i in range(N) if i not in set(mask_indices[b].tolist())]
                context_patches.append(patches[b, keep])  # (n_context, D)
            patches = torch.stack(context_patches, dim=0)  # (B, n_context, D)

        for block in self.blocks:
            patches = block(patches)
        patches = self.norm(patches)
        return patches


class JEPAEncoderV8Fast(nn.Module):
    """
    Optimized encoder using gather instead of list comprehension.
    Requires context_indices to be provided when masking.
    """
    def __init__(
        self,
        n_channels: int = 1,
        window_size: int = 1024,
        patch_size: int = 64,
        embed_dim: int = 256,
        depth: int = 4,
        n_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.n_patches = window_size // patch_size
        self.embed_dim = embed_dim

        self.patch_embed = PatchEmbed1D(n_channels, patch_size, embed_dim)
        pe = sinusoidal_pe(self.n_patches, embed_dim)
        self.register_buffer('pos_embed', pe)

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, n_heads, mlp_ratio=4.0, dropout=dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        context_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, C, L)
            context_indices: (B, n_context) — if None, process all patches

        Returns: (B, n_context, D) or (B, N, D)
        """
        patches = self.patch_embed(x)   # (B, N, D)
        patches = patches + self.pos_embed

        if context_indices is not None:
            # Gather context patches
            patches = torch.gather(
                patches, 1,
                context_indices.unsqueeze(-1).expand(-1, -1, self.embed_dim)
            )  # (B, n_context, D)

        for block in self.blocks:
            patches = block(patches)
        patches = self.norm(patches)
        return patches


# ============================================================
# PREDICTOR
# ============================================================

class JEPAPredictorV8(nn.Module):
    """
    Predictor: takes context patch embeddings + mask positions,
    predicts target embeddings at masked positions.
    """
    def __init__(
        self,
        n_patches: int = 16,
        embed_dim: int = 256,
        predictor_dim: int = 128,
        depth: int = 4,
        n_heads: int = 4,
    ):
        super().__init__()
        self.n_patches = n_patches
        self.embed_dim = embed_dim
        self.predictor_dim = predictor_dim

        self.input_proj = nn.Linear(embed_dim, predictor_dim)

        # Fixed sinusoidal PE — prevents position collapse
        pe = sinusoidal_pe(n_patches, predictor_dim)
        self.register_buffer('pos_embed', pe)  # (1, N, predictor_dim)

        # Shared learnable mask token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        self.blocks = nn.ModuleList([
            TransformerBlock(predictor_dim, n_heads, mlp_ratio=4.0, dropout=0.1)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(predictor_dim)
        self.output_proj = nn.Linear(predictor_dim, embed_dim)

    def forward(
        self,
        context_embeds: torch.Tensor,
        context_indices: torch.Tensor,
        mask_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            context_embeds: (B, n_context, embed_dim)
            context_indices: (B, n_context) — patch indices of context
            mask_indices: (B, n_mask) — patch indices of masked positions

        Returns: (B, n_mask, embed_dim) predicted target embeddings
        """
        B = context_embeds.shape[0]
        n_context = context_embeds.shape[1]
        n_mask = mask_indices.shape[1]

        # Project to predictor dim
        context = self.input_proj(context_embeds)  # (B, n_context, predictor_dim)

        # Add positional encoding to context
        pe = self.pos_embed.expand(B, -1, -1)  # (B, N, predictor_dim)
        ctx_pe = torch.gather(pe, 1, context_indices.unsqueeze(-1).expand(-1, -1, self.predictor_dim))
        context = context + ctx_pe

        # Build mask tokens with positional encoding
        mask_tokens = self.mask_token.expand(B, n_mask, -1)  # (B, n_mask, predictor_dim)
        mask_pe = torch.gather(pe, 1, mask_indices.unsqueeze(-1).expand(-1, -1, self.predictor_dim))
        mask_tokens = mask_tokens + mask_pe

        # Concatenate: [context | mask tokens]
        x = torch.cat([context, mask_tokens], dim=1)  # (B, n_context+n_mask, predictor_dim)

        for block in self.blocks:
            x = block(x)
        x = self.norm(x)

        # Extract predictions for masked positions
        preds = x[:, n_context:]  # (B, n_mask, predictor_dim)
        return self.output_proj(preds)  # (B, n_mask, embed_dim)


# ============================================================
# FULL JEPA V8 MODEL
# ============================================================

class MechanicalJEPAV8(nn.Module):
    """
    JEPA V8: Single-channel bearing JEPA for RUL pretraining.

    Key design choices:
    - L1 loss on normalized predictions (not MSE — avoids mean-prediction collapse)
    - Variance regularization (lambda=0.1) on predictions
    - EMA target encoder (momentum=0.996)
    - Sinusoidal PE throughout (no learnable PE — prevents position collapse)
    """

    def __init__(
        self,
        n_channels: int = 1,
        window_size: int = 1024,
        patch_size: int = 64,
        embed_dim: int = 256,
        encoder_depth: int = 4,
        predictor_depth: int = 4,
        n_heads: int = 4,
        mask_ratio: float = 0.625,  # mask 10 of 16 patches
        ema_decay: float = 0.996,
        var_reg_lambda: float = 0.1,
    ):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.ema_decay = ema_decay
        self.n_patches = window_size // patch_size
        self.embed_dim = embed_dim
        self.var_reg_lambda = var_reg_lambda

        n_mask = int(self.n_patches * mask_ratio)
        self.n_mask = n_mask
        self.n_context = self.n_patches - n_mask

        # Context encoder (trained via backprop)
        self.encoder = JEPAEncoderV8Fast(
            n_channels=n_channels,
            window_size=window_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=encoder_depth,
            n_heads=n_heads,
        )

        # Target encoder (EMA copy, no gradients)
        self.target_encoder = copy.deepcopy(self.encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False

        # Predictor
        self.predictor = JEPAPredictorV8(
            n_patches=self.n_patches,
            embed_dim=embed_dim,
            predictor_dim=embed_dim // 2,
            depth=predictor_depth,
            n_heads=n_heads,
        )

    @torch.no_grad()
    def update_ema(self):
        """Update target encoder via EMA."""
        for p_q, p_k in zip(self.encoder.parameters(), self.target_encoder.parameters()):
            p_k.data = self.ema_decay * p_k.data + (1 - self.ema_decay) * p_q.data

    def _generate_mask(self, batch_size: int, device: torch.device):
        """Generate random mask for each batch item."""
        n_mask = self.n_mask
        n_context = self.n_context

        # Random shuffle for each batch item
        indices = torch.stack([
            torch.randperm(self.n_patches, device=device)
            for _ in range(batch_size)
        ])  # (B, N)

        mask_indices = indices[:, :n_mask]      # (B, n_mask)
        context_indices = indices[:, n_mask:]   # (B, n_context)

        # Sort for reproducibility
        mask_indices, _ = torch.sort(mask_indices, dim=1)
        context_indices, _ = torch.sort(context_indices, dim=1)

        return mask_indices, context_indices

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Training forward pass.

        Returns: (loss, predictions, targets)
        """
        B = x.shape[0]
        device = x.device

        # Generate masks
        mask_indices, context_indices = self._generate_mask(B, device)

        # Target: process ALL patches with EMA encoder
        with torch.no_grad():
            all_targets = self.target_encoder(x, context_indices=None)  # (B, N, D)
            # Gather targets at masked positions
            targets = torch.gather(
                all_targets, 1,
                mask_indices.unsqueeze(-1).expand(-1, -1, self.embed_dim)
            )  # (B, n_mask, D)

        # Context: process only visible patches
        context_embeds = self.encoder(x, context_indices=context_indices)  # (B, n_context, D)

        # Predict masked patch embeddings
        predictions = self.predictor(context_embeds, context_indices, mask_indices)  # (B, n_mask, D)

        # L1 loss on L2-normalized predictions
        pred_norm = F.normalize(predictions, dim=-1)
        tgt_norm = F.normalize(targets, dim=-1)
        loss = F.l1_loss(pred_norm, tgt_norm)

        # Variance regularization: penalize low variance across predicted positions
        if self.var_reg_lambda > 0:
            pred_var = predictions.var(dim=1).mean()  # variance across positions
            var_loss = F.relu(0.1 - pred_var)
            loss = loss + self.var_reg_lambda * var_loss

        return loss, predictions, targets

    @torch.no_grad()
    def get_embeddings(self, x: torch.Tensor, pool: str = 'mean') -> torch.Tensor:
        """
        Get embeddings for downstream tasks.

        Args:
            x: (B, C, L) input signal
            pool: 'mean' (recommended) or 'cls' (not applicable here)

        Returns: (B, D) mean-pooled patch embeddings
        """
        all_patches = self.encoder(x, context_indices=None)  # (B, N, D)
        if pool == 'mean':
            return all_patches.mean(dim=1)  # (B, D)
        else:
            return all_patches[:, 0]  # first patch as proxy for CLS


# ============================================================
# MODEL SUMMARY
# ============================================================

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    print("=== JEPA V8 Architecture Test ===\n")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    model = MechanicalJEPAV8().to(device)
    print(f"\nModel parameters: {count_parameters(model):,}")

    # Test forward pass
    x = torch.randn(8, 1, 1024).to(device)
    loss, preds, targets = model(x)

    print(f"\nForward pass:")
    print(f"  Input: {x.shape}")
    print(f"  Loss: {loss.item():.4f}")
    print(f"  Predictions: {preds.shape}")
    print(f"  Targets: {targets.shape}")

    # Check prediction variance (anti-collapse metric)
    pred_var = preds.var(dim=1).mean().item()
    print(f"  Prediction variance across positions: {pred_var:.4f} (>0.01 = OK)")

    # Test EMA update
    model.update_ema()
    print("\nEMA update: OK")

    # Test get_embeddings
    model.eval()
    with torch.no_grad():
        embeds = model.get_embeddings(x)
    print(f"\nEmbeddings: {embeds.shape}, "
          f"mean={embeds.mean().item():.4f}, "
          f"std={embeds.std().item():.4f}")

    print("\n=== JEPA V8 Architecture OK ===")
