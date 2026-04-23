"""
FAM — Forecast-Anything Model.

Canonical architecture. See ARCHITECTURE.md for the specification.

Components:
  RevIN              — per-context instance normalization
  PatchEmbedding     — group P timesteps, project to d
  CausalEncoder      — causal transformer → h_t
  TargetEncoder      — bidirectional transformer + attention pool → h*
  Predictor          — MLP(h_t, Δt) → predicted future embedding
  EventHead          — shared linear → σ → p(t, Δt)
  FAM                — full model (pretrain + finetune)
"""

import copy
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Positional encoding
# ---------------------------------------------------------------------------

def sinusoidal_pe(positions: torch.Tensor, d: int) -> torch.Tensor:
    """Sinusoidal positional encoding. positions: (N,) → (N, d)."""
    pe = torch.zeros(len(positions), d, device=positions.device)
    pos = positions.float().unsqueeze(1)  # (N, 1)
    div = torch.exp(torch.arange(0, d, 2, device=positions.device).float()
                    * -(math.log(10000.0) / d))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


# ---------------------------------------------------------------------------
# RevIN — Reversible Instance Normalization (Kim et al. 2022)
# ---------------------------------------------------------------------------

class RevIN(nn.Module):
    """Per-context, per-channel normalization. No learnable parameters."""

    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor,
                mask: Optional[torch.Tensor] = None):
        """
        x: (B, T, C). mask: (B, T) bool, True = padding.
        Returns (x_norm, stats) where stats = (mean, std) for denorm.
        """
        if mask is not None:
            # Exclude padding from stats
            valid = (~mask).unsqueeze(-1).float()  # (B, T, 1)
            n = valid.sum(dim=1, keepdim=True).clamp(min=1)  # (B, 1, 1)
            mean = (x * valid).sum(dim=1, keepdim=True) / n   # (B, 1, C)
            var = ((x - mean) ** 2 * valid).sum(dim=1, keepdim=True) / n
            std = (var + self.eps).sqrt()
        else:
            mean = x.mean(dim=1, keepdim=True)       # (B, 1, C)
            std = x.std(dim=1, keepdim=True) + self.eps
        return (x - mean) / std, (mean, std)


# ---------------------------------------------------------------------------
# Patch embedding
# ---------------------------------------------------------------------------

class PatchEmbedding(nn.Module):
    """Group P timesteps across all C channels into one token."""

    def __init__(self, n_channels: int, patch_size: int = 16, d_model: int = 256):
        super().__init__()
        self.P = patch_size
        self.proj = nn.Linear(n_channels * patch_size, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, C) → (B, N_tokens, d)."""
        B, T, C = x.shape
        P = self.P
        # Pad to multiple of P
        remainder = T % P
        if remainder != 0:
            x = F.pad(x, (0, 0, 0, P - remainder))  # pad time dim
            T = x.shape[1]
        x = x.reshape(B, T // P, C * P)
        return self.proj(x)


# ---------------------------------------------------------------------------
# Transformer blocks
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,
                                          batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model), nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, key_padding_mask=None, attn_mask=None):
        # Pre-norm
        x2 = self.norm1(x)
        a, _ = self.attn(x2, x2, x2, key_padding_mask=key_padding_mask,
                         attn_mask=attn_mask)
        x = x + self.drop(a)
        x = x + self.ff(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Context encoder (causal)
# ---------------------------------------------------------------------------

class CausalEncoder(nn.Module):
    """Causal transformer encoder → h_t (last valid token)."""

    def __init__(self, n_channels: int, patch_size: int = 16,
                 d_model: int = 256, n_heads: int = 4, n_layers: int = 2,
                 d_ff: int = 256, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.P = patch_size
        self.revin = RevIN()
        self.patch_embed = PatchEmbedding(n_channels, patch_size, d_model)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: (B, T, C). mask: (B, T) bool, True=padding.
        Returns: h_t (B, d).
        """
        B, T, C = x.shape
        # RevIN
        x, _ = self.revin(x, mask)
        # Patch embed
        tokens = self.patch_embed(x)  # (B, N, d)
        N = tokens.shape[1]
        # Positional encoding
        tokens = tokens + sinusoidal_pe(torch.arange(N, device=x.device), self.d_model)
        # Patch-level padding mask
        patch_mask = self._compute_patch_mask(mask, T, N, x.device) if mask is not None else None
        # Causal attention mask
        causal = nn.Transformer.generate_square_subsequent_mask(N, device=x.device)
        # Forward through layers
        h = tokens
        for layer in self.layers:
            h = layer(h, key_padding_mask=patch_mask, attn_mask=causal)
        h = self.norm(h)  # (B, N, d)
        # Extract last valid token
        if patch_mask is not None:
            valid = (~patch_mask).long()
            last_idx = (valid * torch.arange(N, device=x.device)).argmax(dim=1)
            h_t = h[torch.arange(B, device=x.device), last_idx]
        else:
            h_t = h[:, -1]
        return h_t

    def _compute_patch_mask(self, mask, T, N, device):
        """Convert timestep mask (B, T) to patch mask (B, N)."""
        B = mask.shape[0]
        P = self.P
        # Pad mask to match padded T
        T_padded = N * P
        if T < T_padded:
            mask = F.pad(mask, (0, T_padded - T), value=True)
        # A patch is padding if ALL its timesteps are padding
        mask = mask[:, :T_padded].reshape(B, N, P)
        return mask.all(dim=-1)  # (B, N)


# ---------------------------------------------------------------------------
# Target encoder (bidirectional + attention pool)
# ---------------------------------------------------------------------------

class TargetEncoder(nn.Module):
    """Bidirectional transformer + attention pool → h* (B, d)."""

    def __init__(self, n_channels: int, patch_size: int = 16,
                 d_model: int = 256, n_heads: int = 4, n_layers: int = 2,
                 d_ff: int = 256, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.P = patch_size
        self.revin = RevIN()
        self.patch_embed = PatchEmbedding(n_channels, patch_size, d_model)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        # Attention pool
        self.pool_query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pool_attn = nn.MultiheadAttention(d_model, n_heads, dropout=0.0,
                                               batch_first=True)

    def forward(self, x: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: (B, T, C) — the target interval x(t : t+Δt].
        mask: (B, T) bool, True=padding.
        Returns: h* (B, d).
        """
        B = x.shape[0]
        x, _ = self.revin(x, mask)
        tokens = self.patch_embed(x)
        N = tokens.shape[1]
        tokens = tokens + sinusoidal_pe(torch.arange(N, device=x.device), self.d_model)

        patch_mask = None
        if mask is not None:
            T = mask.shape[1]
            T_padded = N * self.P
            if T < T_padded:
                mask_padded = F.pad(mask, (0, T_padded - T), value=True)
            else:
                mask_padded = mask[:, :T_padded]
            patch_mask = mask_padded.reshape(B, N, self.P).all(dim=-1)

        h = tokens
        for layer in self.layers:
            h = layer(h, key_padding_mask=patch_mask)  # bidirectional
        h = self.norm(h)

        # Attention pool
        query = self.pool_query.expand(B, -1, -1)
        pooled, _ = self.pool_attn(query, h, h, key_padding_mask=patch_mask)
        return pooled.squeeze(1)  # (B, d)


# ---------------------------------------------------------------------------
# Predictor (horizon-aware MLP)
# ---------------------------------------------------------------------------

class Predictor(nn.Module):
    """MLP: (h_t, Δt) → predicted future embedding."""

    def __init__(self, d_model: int = 256, hidden: int = 1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model + 1, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model),
        )

    def forward(self, h: torch.Tensor, delta_t: torch.Tensor) -> torch.Tensor:
        """h: (B, d). delta_t: (B,) float. Returns (B, d)."""
        dt = delta_t.float().unsqueeze(-1)  # (B, 1)
        return self.net(torch.cat([h, dt], dim=-1))


# ---------------------------------------------------------------------------
# Event head (shared linear → sigmoid)
# ---------------------------------------------------------------------------

class EventHead(nn.Module):
    """Shared linear head: predicted embedding → event logit."""

    def __init__(self, d_model: int = 256):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.linear = nn.Linear(d_model, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: (B, d) or (B, K, d). Returns logits same shape minus last dim."""
        return self.linear(self.norm(h)).squeeze(-1)


# ---------------------------------------------------------------------------
# FAM — full model
# ---------------------------------------------------------------------------

class FAM(nn.Module):
    """
    Forecast-Anything Model.

    Pretraining: encoder + target_encoder + predictor (self-supervised).
    Finetuning:  freeze encoder, train predictor + event_head (supervised).
    """

    def __init__(self, n_channels: int, patch_size: int = 16,
                 d_model: int = 256, n_heads: int = 4, n_layers: int = 2,
                 d_ff: int = 256, dropout: float = 0.1,
                 ema_momentum: float = 0.99, predictor_hidden: int = 256):
        super().__init__()
        self.d_model = d_model
        self.ema_momentum = ema_momentum

        self.encoder = CausalEncoder(
            n_channels, patch_size, d_model, n_heads, n_layers, d_ff, dropout)
        self.target_encoder = TargetEncoder(
            n_channels, patch_size, d_model, n_heads, n_layers, d_ff, dropout)
        self.predictor = Predictor(d_model, predictor_hidden)
        self.event_head = EventHead(d_model)

        # Initialize target encoder from encoder weights (matching keys)
        self._init_target_encoder()
        # Freeze target encoder
        for p in self.target_encoder.parameters():
            p.requires_grad = False

    def _init_target_encoder(self):
        """Copy matching weights from encoder to target encoder."""
        enc_state = self.encoder.state_dict()
        tgt_state = self.target_encoder.state_dict()
        for k in tgt_state:
            if k in enc_state and enc_state[k].shape == tgt_state[k].shape:
                tgt_state[k] = enc_state[k].clone()
        self.target_encoder.load_state_dict(tgt_state)

    @torch.no_grad()
    def update_ema(self):
        """EMA update: target_encoder ← m * target + (1-m) * encoder."""
        m = self.ema_momentum
        for p_enc, p_tgt in zip(self.encoder.parameters(),
                                self.target_encoder.parameters()):
            p_tgt.data.mul_(m).add_(p_enc.data, alpha=1 - m)

    # --- Pretraining forward ---

    def pretrain_forward(self, context: torch.Tensor, target: torch.Tensor,
                         delta_t: torch.Tensor,
                         context_mask: Optional[torch.Tensor] = None,
                         target_mask: Optional[torch.Tensor] = None):
        """
        context: (B, T_ctx, C) — observations up to time t.
        target:  (B, T_tgt, C) — observations in (t, t+Δt].
        delta_t: (B,) — horizon values.
        Returns: (pred, target_repr) both (B, d), L2-normalized.
        """
        h_t = self.encoder(context, context_mask)
        h_pred = self.predictor(h_t, delta_t)

        with torch.no_grad():
            h_target = self.target_encoder(target, target_mask)

        # L2 normalize
        h_pred = F.normalize(h_pred, dim=-1)
        h_target = F.normalize(h_target, dim=-1)
        return h_pred, h_target

    # --- Finetuning forward ---

    def finetune_forward(self, context: torch.Tensor,
                         horizons: torch.Tensor,
                         context_mask: Optional[torch.Tensor] = None,
                         mode: str = 'pred_ft') -> torch.Tensor:
        """
        context: (B, T, C).
        horizons: (K,) — fixed set of horizons.
        mode: 'pred_ft' (freeze encoder) or 'e2e' (train all).
        Returns: logits (B, K).
        """
        if mode == 'pred_ft':
            with torch.no_grad():
                h_t = self.encoder(context, context_mask)
            h_t = h_t.detach()
        else:
            h_t = self.encoder(context, context_mask)

        # Vectorized: run predictor at all K horizons
        B, d = h_t.shape
        K = horizons.shape[0]
        h_exp = h_t.unsqueeze(1).expand(B, K, d).reshape(B * K, d)
        dt_exp = horizons.unsqueeze(0).expand(B, K).reshape(B * K).to(
            device=h_t.device, dtype=torch.float32)
        h_pred = self.predictor(h_exp, dt_exp).view(B, K, d)

        logits = self.event_head(h_pred)  # (B, K)
        return logits

    # --- Convenience ---

    def encode(self, context: torch.Tensor,
               mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Encode context → h_t. For probing / analysis."""
        return self.encoder(context, mask)
