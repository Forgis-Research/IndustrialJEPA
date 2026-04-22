"""V22 Encoder Variants — Cross-Channel Attention Architectures.

Two alternative context encoders that add cross-channel attention:

  Variant A (pure iTransformer, fixed window T=100)
    - Transpose (B,T,C) -> (B,C,T), project each channel's full window to d
      with a SHARED Linear(T, d).
    - Add sinusoidal PE(channel_index) — fixed, not learned.
    - Self-attention across the C channel tokens (L=2, H=4, non-causal).
    - Mean-pool C tokens to h_past.

  Variant B (hybrid temporal + cross-channel, variable-length)
    - Temporal path unchanged: Linear(C, d) -> causal Transformer -> (B,T,d).
      Extract last non-padded temporal token h_temporal.
    - Cross-channel path: shared Linear(W, d_ch) on last W=10 timesteps
      per channel, + sinusoidal PE(channel_index), + 1-layer cross-channel
      MHA, + mean-pool -> h_channel.
    - Fuse: Linear([h_temporal; h_channel]) -> h_past.

HARD RULE: no learnable per-channel embeddings.  Channel identity comes
only from fixed sinusoidal PE on the channel index (V15 sensor-ID
embeddings made pretrain collapse; V14 used cross-channel WITHOUT channel
IDs but hit low-label regression from another source).

TrajectoryJEPAVariant reuses the baseline TargetEncoder and Predictor —
only the context encoder swaps.  This lets us freeze the encoder during
pred-FT and compare representation quality head-to-head.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path('/home/sagemaker-user/IndustrialJEPA')
sys.path.insert(0, str(ROOT / 'fam-jepa' / 'experiments' / 'v11'))
from models import (  # noqa: E402
    sinusoidal_pe, TransformerEncoder,
    TargetEncoder, Predictor, TrajectoryJEPA,
)


# ---------------------------------------------------------------------------
# Variant A: pure iTransformer (fixed window T)
# ---------------------------------------------------------------------------

class ITransformerContextEncoder(nn.Module):
    """Pure iTransformer-style encoder.

    Input : (B, T, C) with fixed T.  Shorter inputs must be left-padded
            to T before the call (or pass a key_padding_mask — ignored by
            this variant because projection is position-specific).
    Output: h_past (B, d_model)

    Architecture:
      - Transpose to (B, C, T).  Shared Linear(T, d_model) applied to each
        of the C channels.  This produces one token per channel, whose
        temporal history is encoded by the projection weights.
      - Add fixed sinusoidal PE(channel_index) to distinguish channels.
      - Non-causal self-attention over the C tokens (channels attend to
        each other).  L=2, H=4 transformer.
      - Mean-pool the C tokens to a single (B, d_model) representation.
    """

    def __init__(self, n_sensors: int, window: int, d_model: int = 256,
                 n_heads: int = 4, n_layers: int = 2, d_ff: int = 1024,
                 dropout: float = 0.1):
        super().__init__()
        self.n_sensors = n_sensors
        self.window = window
        self.d_model = d_model
        # Shared Linear(T, d): each channel's entire temporal window -> d_model.
        self.temporal_proj = nn.Linear(window, d_model)
        # Cross-channel transformer (non-causal).
        self.transformer = TransformerEncoder(
            d_model, n_heads, n_layers, d_ff, dropout)

    def _pad_or_truncate(self, x: torch.Tensor) -> torch.Tensor:
        """Left-pad (with zeros) or truncate temporal axis to self.window."""
        B, T, C = x.shape
        if T == self.window:
            return x
        if T > self.window:
            return x[:, -self.window:, :]
        # left-pad with zeros so the last token of the window is still x[:, -1]
        pad = torch.zeros(B, self.window - T, C, device=x.device, dtype=x.dtype)
        return torch.cat([pad, x], dim=1)

    def forward(self, x: torch.Tensor,
                key_padding_mask: Optional[torch.Tensor] = None
                ) -> torch.Tensor:
        # x: (B, T, C) — pad/truncate to T=self.window
        x = self._pad_or_truncate(x)  # (B, W, C)
        B, W, C = x.shape
        assert C == self.n_sensors, f"expected C={self.n_sensors}, got {C}"

        # (B, W, C) -> (B, C, W) -> Linear(W, d) -> (B, C, d)
        x_ct = x.transpose(1, 2)  # (B, C, W)
        tokens = self.temporal_proj(x_ct)  # (B, C, d_model)

        # Fixed sinusoidal PE on channel index.
        ch_idx = torch.arange(C, device=x.device)
        ch_pe = sinusoidal_pe(ch_idx, self.d_model)  # (C, d_model)
        tokens = tokens + ch_pe.unsqueeze(0)

        # Cross-channel self-attention (non-causal: channels can attend freely).
        out = self.transformer(tokens, key_padding_mask=None, causal=False)
        # (B, C, d_model)

        # Mean-pool across channel dim.
        h_past = out.mean(dim=1)  # (B, d_model)
        return h_past


class TrajectoryJEPAVariantA(nn.Module):
    """TrajectoryJEPA with iTransformer context encoder.

    Target encoder and predictor are unchanged (reuse baseline classes).
    """

    def __init__(self, n_sensors: int = 14, window: int = 100,
                 d_model: int = 256, n_heads: int = 4, n_layers: int = 2,
                 d_ff: int = 1024, dropout: float = 0.1,
                 ema_momentum: float = 0.99, predictor_hidden: int = 1024,
                 patch_length: int = 1):
        super().__init__()
        assert patch_length == 1, 'Variant A requires patch_length=1'
        self.d_model = d_model
        self.ema_momentum = ema_momentum

        self.context_encoder = ITransformerContextEncoder(
            n_sensors=n_sensors, window=window, d_model=d_model,
            n_heads=n_heads, n_layers=n_layers, d_ff=d_ff, dropout=dropout)

        # Baseline target encoder (bidirectional, variable-length on future).
        self.target_encoder = TargetEncoder(
            n_sensors=n_sensors, patch_length=1, d_model=d_model,
            n_heads=n_heads, n_layers=n_layers, d_ff=d_ff, dropout=dropout)

        self.predictor = Predictor(d_model, predictor_hidden)

        # Freeze target encoder — EMA update of the context encoder's
        # temporal_proj weights isn't meaningful here, so the target encoder
        # stays as a standalone trained-by-self module (same as v11 behaviour:
        # target encoder copies context encoder for TEMPORAL transformer, but
        # variant A has no temporal transformer. We train target encoder
        # from scratch via its own forward path during pretraining).
        for p in self.target_encoder.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def update_ema(self):
        # No shared structure between context and target encoders -> no EMA.
        # Target encoder is simply random-init and gets slowly adjusted
        # via the BCE-only path. Keep it frozen for stability.
        pass

    def forward_pretrain(self, past, past_mask, future, future_mask, k):
        h_past = self.context_encoder(past, past_mask)
        with torch.no_grad():
            h_future = self.target_encoder(future, future_mask)
        pred_future = self.predictor(h_past, k)
        return pred_future, h_future, h_past

    def encode_past(self, past, past_mask=None):
        return self.context_encoder(past, past_mask)


# ---------------------------------------------------------------------------
# Variant B: hybrid (temporal + cross-channel, variable-length compatible)
# ---------------------------------------------------------------------------

class CrossChannelHead(nn.Module):
    """Extract a cross-channel summary from the last W timesteps of the raw
    input.  Returns (B, d_ch).

    - Take x[:, -W:, :].  If T < W, left-pad with zeros.
    - (B, W, C) -> transpose -> (B, C, W) -> Linear(W, d_ch) -> (B, C, d_ch)
    - Add fixed sinusoidal PE(channel_index).
    - 1-layer cross-channel MultiheadAttention (MHA), then a small FFN.
    - Mean-pool across C -> (B, d_ch).
    """

    def __init__(self, n_sensors: int, window_w: int, d_ch: int,
                 n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.n_sensors = n_sensors
        self.window_w = window_w
        self.d_ch = d_ch
        self.proj = nn.Linear(window_w, d_ch)
        self.attn = nn.MultiheadAttention(d_ch, n_heads, dropout=dropout,
                                          batch_first=True)
        self.norm1 = nn.LayerNorm(d_ch)
        self.ff = nn.Sequential(
            nn.Linear(d_ch, d_ch * 2), nn.GELU(),
            nn.Linear(d_ch * 2, d_ch))
        self.norm2 = nn.LayerNorm(d_ch)
        self.drop = nn.Dropout(dropout)

    def _last_W(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        if T == self.window_w:
            return x
        if T > self.window_w:
            return x[:, -self.window_w:, :]
        pad = torch.zeros(B, self.window_w - T, C, device=x.device, dtype=x.dtype)
        return torch.cat([pad, x], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        x_w = self._last_W(x)            # (B, W, C)
        x_w = x_w.transpose(1, 2)        # (B, C, W)
        tokens = self.proj(x_w)          # (B, C, d_ch)

        ch_idx = torch.arange(tokens.shape[1], device=x.device)
        ch_pe = sinusoidal_pe(ch_idx, self.d_ch)  # (C, d_ch)
        tokens = tokens + ch_pe.unsqueeze(0)

        # Cross-channel attention (pre-norm).
        t2 = self.norm1(tokens)
        a, _ = self.attn(t2, t2, t2, need_weights=False)
        tokens = tokens + self.drop(a)
        t2 = self.norm2(tokens)
        tokens = tokens + self.drop(self.ff(t2))

        return tokens.mean(dim=1)  # (B, d_ch)


class HybridContextEncoder(nn.Module):
    """Hybrid encoder: temporal (unchanged) + cross-channel (parallel stream).

    Step 1: temporal path = baseline ContextEncoder output at last valid
            token.  Produces h_temporal (B, d_model).
    Step 2: cross-channel path on last W=10 timesteps -> h_channel (B, d_ch).
    Step 3: fuse via Linear(d_model + d_ch, d_model) -> h_past.
    """

    def __init__(self, n_sensors: int, d_model: int = 256, n_heads: int = 4,
                 n_layers: int = 2, d_ff: int = 1024, dropout: float = 0.1,
                 cross_window: int = 10, d_ch: Optional[int] = None,
                 patch_length: int = 1):
        super().__init__()
        # Reuse the baseline temporal ContextEncoder.
        from models import ContextEncoder
        self.temporal = ContextEncoder(
            n_sensors=n_sensors, patch_length=patch_length, d_model=d_model,
            n_heads=n_heads, n_layers=n_layers, d_ff=d_ff, dropout=dropout)

        self.d_model = d_model
        self.d_ch = d_ch or d_model
        self.cross = CrossChannelHead(
            n_sensors=n_sensors, window_w=cross_window, d_ch=self.d_ch,
            n_heads=n_heads, dropout=dropout)
        self.fuse = nn.Sequential(
            nn.Linear(d_model + self.d_ch, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )

    def forward(self, x: torch.Tensor,
                key_padding_mask: Optional[torch.Tensor] = None
                ) -> torch.Tensor:
        # Temporal path
        h_temporal = self.temporal(x, key_padding_mask)  # (B, d_model)
        # Cross-channel path
        h_channel = self.cross(x)  # (B, d_ch)
        # Fuse
        h = torch.cat([h_temporal, h_channel], dim=-1)
        return self.fuse(h)


class TrajectoryJEPAVariantB(nn.Module):
    """TrajectoryJEPA with hybrid context encoder."""

    def __init__(self, n_sensors: int = 14, patch_length: int = 1,
                 d_model: int = 256, n_heads: int = 4, n_layers: int = 2,
                 d_ff: int = 1024, dropout: float = 0.1,
                 ema_momentum: float = 0.99, predictor_hidden: int = 1024,
                 cross_window: int = 10):
        super().__init__()
        self.d_model = d_model
        self.ema_momentum = ema_momentum
        self.patch_length = patch_length

        self.context_encoder = HybridContextEncoder(
            n_sensors=n_sensors, d_model=d_model, n_heads=n_heads,
            n_layers=n_layers, d_ff=d_ff, dropout=dropout,
            cross_window=cross_window, patch_length=patch_length)
        self.target_encoder = TargetEncoder(
            n_sensors=n_sensors, patch_length=patch_length, d_model=d_model,
            n_heads=n_heads, n_layers=n_layers, d_ff=d_ff, dropout=dropout)
        self.predictor = Predictor(d_model, predictor_hidden)

        # EMA init: target encoder copies the TEMPORAL transformer weights
        # from the context encoder (variant B keeps a temporal transformer).
        self._init_target_encoder()

    def _init_target_encoder(self):
        ctx_t = self.context_encoder.temporal.state_dict()
        tgt = self.target_encoder.state_dict()
        for k in tgt:
            if k in ctx_t:
                tgt[k] = ctx_t[k].clone()
        self.target_encoder.load_state_dict(tgt)
        for p in self.target_encoder.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def update_ema(self):
        m = self.ema_momentum
        ctx_t = dict(self.context_encoder.temporal.named_parameters())
        tgt = dict(self.target_encoder.named_parameters())
        for k in tgt:
            if k in ctx_t:
                tgt[k].data.mul_(m).add_(ctx_t[k].data, alpha=1.0 - m)

    def forward_pretrain(self, past, past_mask, future, future_mask, k):
        h_past = self.context_encoder(past, past_mask)
        with torch.no_grad():
            h_future = self.target_encoder(future, future_mask)
        pred_future = self.predictor(h_past, k)
        return pred_future, h_future, h_past

    def encode_past(self, past, past_mask=None):
        return self.context_encoder(past, past_mask)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_model(variant: str, n_sensors: int, window: int = 100,
                d_model: int = 256, n_heads: int = 4, n_layers: int = 2,
                d_ff: int = 1024, predictor_hidden: int = 1024,
                dropout: float = 0.1, ema_momentum: float = 0.99,
                cross_window_w: int = 10,
                ) -> nn.Module:
    """Build one of {baseline, variantA, variantB} with matching shapes."""
    if variant == 'baseline':
        return TrajectoryJEPA(
            n_sensors=n_sensors, patch_length=1, d_model=d_model,
            n_heads=n_heads, n_layers=n_layers, d_ff=d_ff, dropout=dropout,
            ema_momentum=ema_momentum, predictor_hidden=predictor_hidden)
    if variant == 'variantA':
        return TrajectoryJEPAVariantA(
            n_sensors=n_sensors, window=window, d_model=d_model,
            n_heads=n_heads, n_layers=n_layers, d_ff=d_ff, dropout=dropout,
            ema_momentum=ema_momentum, predictor_hidden=predictor_hidden)
    if variant == 'variantB':
        return TrajectoryJEPAVariantB(
            n_sensors=n_sensors, patch_length=1, d_model=d_model,
            n_heads=n_heads, n_layers=n_layers, d_ff=d_ff, dropout=dropout,
            ema_momentum=ema_momentum, predictor_hidden=predictor_hidden,
            cross_window=cross_window_w)
    raise ValueError(variant)


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


if __name__ == '__main__':
    # Quick shape + param sanity
    import torch
    for v in ['baseline', 'variantA', 'variantB']:
        m = build_model(v, n_sensors=14, window=100)
        print(f'{v}: total params={count_params(m):,}')
        x = torch.randn(2, 100, 14)
        h = m.encode_past(x)
        print(f'  encode_past({x.shape}) -> {h.shape}')
        # variable length test (only baseline + variantB)
        if v != 'variantA':
            x2 = torch.randn(2, 50, 14)
            mask = torch.zeros(2, 50, dtype=torch.bool)
            h2 = m.encode_past(x2, mask)
            print(f'  encode_past(variable T=50) -> {h2.shape}')
