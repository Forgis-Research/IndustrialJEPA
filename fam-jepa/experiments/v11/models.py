"""
V11 Model Architectures: Trajectory JEPA for C-MAPSS

Key differences from V10:
- NO CNN Stage 1, NO FFT (C-MAPSS cycles are already 14-dim sensor vectors)
- Continuous-time PE indexed by cycle number
- EMA TargetEncoder
- Horizon-aware predictor
- ~1.2M total parameters
"""

import math
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# ============================================================
# Positional Encoding
# ============================================================

def sinusoidal_pe(positions: torch.Tensor, d_model: int) -> torch.Tensor:
    """
    Sinusoidal positional encoding for arbitrary positions.
    positions: (...,) integer or float tensor
    returns: (..., d_model)
    """
    positions = positions.float()
    half_d = d_model // 2
    div_term = torch.exp(
        torch.arange(half_d, device=positions.device).float() *
        -(math.log(10000.0) / half_d)
    )
    # Broadcast: positions shape (...), div_term shape (half_d,)
    sin_enc = torch.sin(positions.unsqueeze(-1) * div_term)
    cos_enc = torch.cos(positions.unsqueeze(-1) * div_term)
    pe = torch.cat([sin_enc, cos_enc], dim=-1)  # (..., d_model)
    if d_model % 2 == 1:
        pe = pe[..., :d_model]
    return pe


# ============================================================
# Sensor Projection (patch tokenizer)
# ============================================================

class SensorProjection(nn.Module):
    """
    Project raw sensor readings to d_model tokens.
    patch_length=1: cycle-as-token (primary)
    patch_length=L: L consecutive cycles as one token (ablation)
    """

    def __init__(self, n_sensors: int = 14, patch_length: int = 1, d_model: int = 128):
        super().__init__()
        self.patch_length = patch_length
        self.n_sensors = n_sensors
        self.d_model = d_model
        self.proj = nn.Linear(n_sensors * patch_length, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, S) -> (B, T', d_model)"""
        B, T, S = x.shape
        L = self.patch_length
        if L > 1:
            T_trim = (T // L) * L
            if T_trim == 0:
                T_trim = L
                x = F.pad(x, (0, 0, L - T % L, 0))
            else:
                x = x[:, :T_trim, :]
            x = x.reshape(B, T_trim // L, L * S)
        return self.proj(x)


# ============================================================
# Transformer Encoder (with optional causal mask)
# ============================================================

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,
                                                batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor,
                key_padding_mask: Optional[torch.Tensor] = None,
                attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention with pre-norm
        x2 = self.norm1(x)
        x2, _ = self.self_attn(x2, x2, x2,
                                key_padding_mask=key_padding_mask,
                                attn_mask=attn_mask,
                                need_weights=False)
        x = x + self.drop(x2)
        # FFN
        x2 = self.norm2(x)
        x = x + self.drop(self.ff(x2))
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_layers: int,
                 d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor,
                key_padding_mask: Optional[torch.Tensor] = None,
                causal: bool = False) -> torch.Tensor:
        T = x.shape[1]
        attn_mask = None
        if causal:
            # Upper triangular causal mask
            attn_mask = torch.triu(
                torch.full((T, T), float('-inf'), device=x.device), diagonal=1
            )
        for layer in self.layers:
            x = layer(x, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        return self.norm(x)


# ============================================================
# Context Encoder (causal)
# ============================================================

class ContextEncoder(nn.Module):
    """
    Encodes past observations causally.
    Outputs last (non-padded) hidden state as h_past.
    """

    def __init__(self, n_sensors: int = 14, patch_length: int = 1,
                 d_model: int = 128, n_heads: int = 4, n_layers: int = 2,
                 d_ff: int = 256, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.patch_length = patch_length
        self.proj = SensorProjection(n_sensors, patch_length, d_model)
        self.transformer = TransformerEncoder(d_model, n_heads, n_layers, d_ff, dropout)

    def forward(self, x: torch.Tensor,
                key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: (B, T, S)
        key_padding_mask: (B, T) bool, True = padding position
        returns: h_past (B, d_model) - last non-padded token
        """
        B, T, S = x.shape
        # Project to tokens
        tokens = self.proj(x)  # (B, T', d_model)
        T_prime = tokens.shape[1]

        # Add sinusoidal positional encoding
        positions = torch.arange(T_prime, device=x.device)  # (T',)
        pe = sinusoidal_pe(positions, self.d_model)  # (T', d_model)
        tokens = tokens + pe.unsqueeze(0)

        # Pad mask for projected tokens (approximate: use T' if L=1)
        proj_mask = None
        if key_padding_mask is not None and self.patch_length == 1:
            proj_mask = key_padding_mask[:, :T_prime]

        # Causal encoding
        out = self.transformer(tokens, key_padding_mask=proj_mask, causal=True)
        # (B, T', d_model)

        # Extract last non-padded position
        if proj_mask is not None:
            # Find last non-masked position per batch
            # proj_mask: True = padding. Find last False.
            valid = (~proj_mask).long()  # (B, T')
            last_idx = (valid * torch.arange(T_prime, device=x.device).unsqueeze(0)).argmax(dim=1)
            h_past = out[torch.arange(B, device=x.device), last_idx]  # (B, d_model)
        else:
            h_past = out[:, -1]  # (B, d_model)

        return h_past


# ============================================================
# Target Encoder (bidirectional, EMA copy)
# ============================================================

class TargetEncoder(nn.Module):
    """
    Encodes future observations bidirectionally.
    Uses attention pooling with a learned query to get h_future.
    """

    def __init__(self, n_sensors: int = 14, patch_length: int = 1,
                 d_model: int = 128, n_heads: int = 4, n_layers: int = 2,
                 d_ff: int = 256, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.patch_length = patch_length
        self.proj = SensorProjection(n_sensors, patch_length, d_model)
        self.transformer = TransformerEncoder(d_model, n_heads, n_layers, d_ff, dropout)
        # Learned pooling query
        self.pool_query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pool_attn = nn.MultiheadAttention(d_model, n_heads, dropout=0.0,
                                                batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor,
                key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: (B, K, S) future sensor observations
        returns: h_future (B, d_model)
        """
        B, K, S = x.shape
        tokens = self.proj(x)  # (B, K', d_model)
        K_prime = tokens.shape[1]

        positions = torch.arange(K_prime, device=x.device)
        pe = sinusoidal_pe(positions, self.d_model)
        tokens = tokens + pe.unsqueeze(0)

        proj_mask = None
        if key_padding_mask is not None and self.patch_length == 1:
            proj_mask = key_padding_mask[:, :K_prime]

        out = self.transformer(tokens, key_padding_mask=proj_mask, causal=False)

        # Attention pooling
        query = self.pool_query.expand(B, -1, -1)  # (B, 1, d_model)
        pooled, _ = self.pool_attn(query, out, out,
                                    key_padding_mask=proj_mask,
                                    need_weights=False)
        h_future = self.norm(pooled[:, 0])  # (B, d_model)
        return h_future


# ============================================================
# Predictor (horizon-aware)
# ============================================================

class Predictor(nn.Module):
    """
    Predicts h_future given h_past and horizon k.
    Input: concat(h_past, PE(k)) -> MLP -> predicted h_future
    """

    def __init__(self, d_model: int = 128, d_hidden: int = 256):
        super().__init__()
        self.d_model = d_model
        self.net = nn.Sequential(
            nn.Linear(2 * d_model, d_hidden),
            nn.ReLU(),
            nn.LayerNorm(d_hidden),
            nn.Linear(d_hidden, d_model),
        )

    def forward(self, h_past: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """
        h_past: (B, d_model)
        k: (B,) integer horizon
        returns: pred_future (B, d_model)
        """
        k_pe = sinusoidal_pe(k, self.d_model)  # (B, d_model)
        x = torch.cat([h_past, k_pe], dim=-1)  # (B, 2*d_model)
        return self.net(x)


# ============================================================
# Full Trajectory JEPA Model
# ============================================================

class TrajectoryJEPA(nn.Module):
    """
    Trajectory JEPA for C-MAPSS RUL prediction.
    Components:
    - ContextEncoder (causal, trained by gradient)
    - TargetEncoder (EMA copy of ContextEncoder, no gradient)
    - Predictor (small MLP, horizon-aware)
    """

    def __init__(self,
                 n_sensors: int = 14,
                 patch_length: int = 1,
                 d_model: int = 128,
                 n_heads: int = 4,
                 n_layers: int = 2,
                 d_ff: int = 256,
                 dropout: float = 0.1,
                 ema_momentum: float = 0.996,
                 predictor_hidden: int = 256):
        super().__init__()
        self.d_model = d_model
        self.ema_momentum = ema_momentum
        self.patch_length = patch_length

        # Context encoder (gradients flow here)
        self.context_encoder = ContextEncoder(
            n_sensors, patch_length, d_model, n_heads, n_layers, d_ff, dropout
        )

        # Target encoder (EMA, no gradients needed during forward)
        self.target_encoder = TargetEncoder(
            n_sensors, patch_length, d_model, n_heads, n_layers, d_ff, dropout
        )

        # Predictor
        self.predictor = Predictor(d_model, predictor_hidden)

        # Initialize target encoder as copy of context encoder
        self._init_target_encoder()

    def _init_target_encoder(self):
        """Copy context encoder weights to target encoder (shared architecture)."""
        # Target encoder has same transformer structure but separate proj/transformer
        # Copy transformer weights from context to target
        ctx_state = self.context_encoder.state_dict()
        tgt_state = self.target_encoder.state_dict()
        # Map context -> target (same keys)
        for key in tgt_state:
            if key in ctx_state:
                tgt_state[key] = ctx_state[key].clone()
        self.target_encoder.load_state_dict(tgt_state)

        # Freeze target encoder (EMA update only)
        for p in self.target_encoder.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def update_ema(self):
        """EMA update of target encoder from context encoder."""
        m = self.ema_momentum
        ctx_params = dict(self.context_encoder.named_parameters())
        tgt_params = dict(self.target_encoder.named_parameters())
        for key in tgt_params:
            if key in ctx_params:
                tgt_params[key].data.mul_(m).add_(ctx_params[key].data, alpha=1.0 - m)

    def forward_pretrain(self,
                          past: torch.Tensor,
                          past_mask: torch.Tensor,
                          future: torch.Tensor,
                          future_mask: torch.Tensor,
                          k: torch.Tensor):
        """
        Forward pass for pretraining.
        Returns: pred_future, h_future (target)
        """
        # Context encoding (with gradients)
        h_past = self.context_encoder(past, past_mask)

        # Target encoding (no gradients, EMA model)
        with torch.no_grad():
            h_future = self.target_encoder(future, future_mask)

        # Prediction
        pred_future = self.predictor(h_past, k)

        return pred_future, h_future, h_past

    def encode_past(self, past: torch.Tensor,
                    past_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Encode past observations to get h_past embedding."""
        return self.context_encoder(past, past_mask)


# ============================================================
# RUL Probe (frozen encoder)
# ============================================================

class RULProbe(nn.Module):
    """Linear probe on top of frozen JEPA encoder."""

    def __init__(self, d_model: int = 128):
        super().__init__()
        self.linear = nn.Linear(d_model, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: (B, d_model) -> predicted RUL in [0, 1]"""
        return self.sigmoid(self.linear(h)).squeeze(-1)


class RULMSE(nn.Module):
    """MSE loss on RUL predictions."""

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(pred, target)


# ============================================================
# Supervised LSTM baseline
# ============================================================

class SupervisedLSTM(nn.Module):
    """
    Supervised LSTM baseline for RUL prediction.
    Trained from scratch with full labels.
    """

    def __init__(self, n_sensors: int = 14, hidden_size: int = 64,
                 n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(n_sensors, hidden_size, n_layers,
                            batch_first=True, dropout=dropout if n_layers > 1 else 0.0)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: (B, T, S)
        mask: (B, T) True = padding
        returns: (B,) RUL predictions in [0, 1]
        """
        # Pack for variable length
        if mask is not None:
            lengths = (~mask).sum(dim=1).clamp(min=1).cpu()
            x_packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths, batch_first=True, enforce_sorted=False
            )
            out_packed, (h_n, _) = self.lstm(x_packed)
        else:
            out, (h_n, _) = self.lstm(x)
            h_n = h_n

        # Use last hidden state from last layer
        h_last = h_n[-1]  # (B, hidden_size)
        return self.fc(h_last).squeeze(-1)


# ============================================================
# Pretraining loss
# ============================================================

def trajectory_jepa_loss(pred_future: torch.Tensor,
                          h_future: torch.Tensor,
                          lambda_var: float = 0.01) -> torch.Tensor:
    """
    JEPA pretraining loss:
    pred_loss = MSE(pred_future, h_future.detach())
    var_loss = variance collapse prevention
    """
    pred_loss = F.mse_loss(pred_future, h_future.detach())

    # Variance collapse prevention: penalize if std < 1.0 across batch
    std = h_future.std(dim=0)  # (d_model,)
    var_loss = torch.relu(1.0 - std).mean()

    return pred_loss + lambda_var * var_loss, pred_loss, var_loss


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
