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
    """Causal transformer encoder → h_t (last valid token).

    ``norm_mode`` selects the normalization strategy:
      - 'revin'       — per-context per-channel RevIN (default, v24/v26)
      - 'none'        — no normalization in the model (data pre-normalized)
      - 'last_value'  — subtract last valid timestep per channel (NLinear)
      - 'revin_stat'  — RevIN + project (mean, std) into a learnable stat
                        token prepended at position 0 of the context
                        sequence. Causal attention lets every later token
                        read the stat token. h_t = last non-stat token.
    """

    VALID_NORM_MODES = ('revin', 'none', 'last_value', 'revin_stat')

    def __init__(self, n_channels: int, patch_size: int = 16,
                 d_model: int = 256, n_heads: int = 4, n_layers: int = 2,
                 d_ff: int = 256, dropout: float = 0.1,
                 norm_mode: str = 'revin'):
        super().__init__()
        assert norm_mode in self.VALID_NORM_MODES, (
            f"norm_mode must be in {self.VALID_NORM_MODES}, got {norm_mode!r}")
        self.d_model = d_model
        self.P = patch_size
        self.n_channels = n_channels
        self.norm_mode = norm_mode

        if norm_mode in ('revin', 'revin_stat'):
            self.revin = RevIN()
        else:
            self.revin = None
        if norm_mode == 'revin_stat':
            self.stat_proj = nn.Linear(2 * n_channels, d_model)
        else:
            self.stat_proj = None

        self.patch_embed = PatchEmbedding(n_channels, patch_size, d_model)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def _preprocess(self, x, mask):
        """Return (x_processed, stat_token | None) per self.norm_mode."""
        if self.norm_mode == 'revin':
            x, _ = self.revin(x, mask)
            return x, None
        if self.norm_mode == 'revin_stat':
            x, (mu, sigma) = self.revin(x, mask)
            stat_feat = torch.cat([mu.squeeze(1), sigma.squeeze(1)], dim=-1)
            stat_token = self.stat_proj(stat_feat)
            return x, stat_token
        if self.norm_mode == 'last_value':
            B, T, _ = x.shape
            if mask is not None:
                valid = (~mask).long()
                positions = torch.arange(T, device=x.device)
                last_idx = (valid * positions).argmax(dim=1)
                last_val = x[torch.arange(B, device=x.device), last_idx].unsqueeze(1)
            else:
                last_val = x[:, -1:, :]
            return x - last_val, None
        if self.norm_mode == 'none':
            return x, None
        raise ValueError(self.norm_mode)

    def forward(self, x: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                return_all: bool = False):
        """
        x: (B, T, C). mask: (B, T) bool, True=padding.
        return_all=False (default): returns h_t (B, d) — last valid non-stat token.
        return_all=True: returns (h_all, patch_mask) where
          h_all: (B, N_total, d) — every output token, including stat token if any
          patch_mask: (B, N_total) bool, True=padding (None when no input mask)
        """
        B, T, C = x.shape
        x, stat_token = self._preprocess(x, mask)
        tokens = self.patch_embed(x)  # (B, N, d)
        N = tokens.shape[1]

        if stat_token is None:
            tokens = tokens + sinusoidal_pe(
                torch.arange(N, device=x.device), self.d_model)
        else:
            # Patches get PE(1..N); stat-token gets PE(0).
            tokens = tokens + sinusoidal_pe(
                torch.arange(1, N + 1, device=x.device), self.d_model)
            stat_pe = sinusoidal_pe(
                torch.zeros(1, dtype=torch.long, device=x.device),
                self.d_model)  # (1, d)
            stat_with_pe = stat_token.unsqueeze(1) + stat_pe.unsqueeze(0)
            tokens = torch.cat([stat_with_pe, tokens], dim=1)  # (B, N+1, d)

        N_total = tokens.shape[1]
        patch_mask = (self._compute_patch_mask(mask, T, N, x.device)
                      if mask is not None else None)
        if patch_mask is not None and stat_token is not None:
            stat_mask = torch.zeros(B, 1, dtype=torch.bool, device=x.device)
            patch_mask = torch.cat([stat_mask, patch_mask], dim=1)  # (B, N+1)

        causal = nn.Transformer.generate_square_subsequent_mask(
            N_total, device=x.device)

        h = tokens
        for layer in self.layers:
            h = layer(h, key_padding_mask=patch_mask, attn_mask=causal)
        h = self.norm(h)

        if return_all:
            return h, patch_mask

        # Extract last valid non-stat token. The stat token (if present) sits
        # at index 0 — it can never be "last" unless N_total==1 (only the stat
        # token), which would mean an empty context.
        if patch_mask is not None:
            valid = (~patch_mask).long()
            last_idx = (valid * torch.arange(N_total, device=x.device)).argmax(dim=1)
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
    """Bidirectional transformer + attention pool → h* (B, d).

    Shares ``norm_mode`` with the context encoder so pretraining uses the
    same pre-patch preprocessing on both sides. ``'revin_stat'`` collapses
    to plain ``'revin'`` on the target side — the target interval is short
    and has no "lifecycle state" to preserve, and the asymmetric stat
    token lives in the context path only.
    """

    def __init__(self, n_channels: int, patch_size: int = 16,
                 d_model: int = 256, n_heads: int = 4, n_layers: int = 2,
                 d_ff: int = 256, dropout: float = 0.1,
                 norm_mode: str = 'revin'):
        super().__init__()
        effective_mode = 'revin' if norm_mode == 'revin_stat' else norm_mode
        assert effective_mode in CausalEncoder.VALID_NORM_MODES
        self.d_model = d_model
        self.P = patch_size
        self.n_channels = n_channels
        self.norm_mode = effective_mode

        if effective_mode == 'revin':
            self.revin = RevIN()
        else:
            self.revin = None

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

    def _preprocess(self, x, mask):
        if self.norm_mode == 'revin':
            x, _ = self.revin(x, mask)
            return x
        if self.norm_mode == 'last_value':
            B, T, _ = x.shape
            if mask is not None:
                valid = (~mask).long()
                positions = torch.arange(T, device=x.device)
                last_idx = (valid * positions).argmax(dim=1)
                last_val = x[torch.arange(B, device=x.device), last_idx].unsqueeze(1)
            else:
                last_val = x[:, -1:, :]
            return x - last_val
        if self.norm_mode == 'none':
            return x
        raise ValueError(self.norm_mode)

    def forward(self, x: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: (B, T, C) — the target interval x(t : t+Δt].
        mask: (B, T) bool, True=padding.
        Returns: h* (B, d).
        """
        B = x.shape[0]
        x = self._preprocess(x, mask)
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


class TransformerPredictor(nn.Module):
    """Transformer predictor: attends over ALL encoder tokens + Δt query.

    The MLP predictor sees only h_t (last encoder token) — a single 256-d
    vector that has to summarize the entire context. This predictor sees
    the full sequence h_all (B, N, d) and appends a Δt query token; the
    attention from the Δt position over all encoder positions lets the
    predictor exploit the drift gradient and local patterns the last-token
    bottleneck collapses.

    Output is taken from the Δt query position.
    """

    def __init__(self, d_model: int = 256, n_heads: int = 4, n_layers: int = 1,
                 d_ff: int = 256, dropout: float = 0.0):
        super().__init__()
        self.dt_embed = nn.Sequential(
            nn.Linear(1, d_model), nn.GELU())
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, activation='gelu', batch_first=True,
            norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.out_norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, h_all: torch.Tensor, delta_t: torch.Tensor,
                key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """h_all: (B, N, d). delta_t: (B,). key_padding_mask: (B, N) True=pad.

        Returns: (B, d) — output from the appended Δt query position.
        """
        B = h_all.shape[0]
        dt_tok = self.dt_embed(delta_t.float().unsqueeze(-1))  # (B, d)
        tokens = torch.cat([h_all, dt_tok.unsqueeze(1)], dim=1)  # (B, N+1, d)
        if key_padding_mask is not None:
            # The Δt query token is always valid (False = not padding).
            extra = torch.zeros(B, 1, dtype=torch.bool,
                                device=key_padding_mask.device)
            kpm = torch.cat([key_padding_mask, extra], dim=1)
        else:
            kpm = None
        out = self.transformer(tokens, src_key_padding_mask=kpm)
        return self.out_proj(self.out_norm(out[:, -1]))


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
                 ema_momentum: float = 0.99, predictor_hidden: int = 256,
                 norm_mode: str = 'revin',
                 predictor_kind: str = 'mlp',
                 predictor_n_layers: int = 1):
        super().__init__()
        assert predictor_kind in ('mlp', 'transformer'), predictor_kind
        self.d_model = d_model
        self.ema_momentum = ema_momentum
        self.norm_mode = norm_mode
        self.predictor_kind = predictor_kind

        self.encoder = CausalEncoder(
            n_channels, patch_size, d_model, n_heads, n_layers, d_ff, dropout,
            norm_mode=norm_mode)
        self.target_encoder = TargetEncoder(
            n_channels, patch_size, d_model, n_heads, n_layers, d_ff, dropout,
            norm_mode=norm_mode)
        if predictor_kind == 'mlp':
            self.predictor = Predictor(d_model, predictor_hidden)
        else:
            self.predictor = TransformerPredictor(
                d_model=d_model, n_heads=n_heads,
                n_layers=predictor_n_layers, d_ff=d_model, dropout=0.0)
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
        """EMA update: target_encoder ← m * target + (1-m) * encoder.

        Only parameters with matching names (and shapes) are updated — the
        target encoder has extra modules (pool_query, pool_attn) that are
        frozen at init (target_encoder.requires_grad=False).
        """
        m = self.ema_momentum
        enc_params = dict(self.encoder.named_parameters())
        for name, p_tgt in self.target_encoder.named_parameters():
            p_enc = enc_params.get(name)
            if p_enc is not None and p_tgt.shape == p_enc.shape:
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
        if self.predictor_kind == 'transformer':
            h_all, h_kpm = self.encoder(context, context_mask, return_all=True)
            h_pred = self.predictor(h_all, delta_t, key_padding_mask=h_kpm)
        else:
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
        horizons: (K,) — fixed set of horizons, sorted ascending.
        mode: 'pred_ft' (freeze encoder) or 'e2e' (train all).
        Returns: cdf (B, K) — event probabilities in (0, 1),
                 monotonically non-decreasing in K by construction.

        Parameterization (discrete hazard → CDF):
          λ_k = σ(event_head(predictor(h_t, Δt_k)))   conditional hazard
          S_k = ∏_{j≤k} (1 - λ_j)                     survival function
          p(t, Δt_k) = 1 - S_k                         CDF (non-decreasing)
        """
        return_all = (self.predictor_kind == 'transformer')
        if mode == 'pred_ft':
            with torch.no_grad():
                enc_out = self.encoder(context, context_mask,
                                       return_all=return_all)
            if return_all:
                h_all, h_kpm = enc_out
                h_all = h_all.detach()
            else:
                h_t = enc_out.detach()
        else:
            enc_out = self.encoder(context, context_mask,
                                   return_all=return_all)
            if return_all:
                h_all, h_kpm = enc_out
            else:
                h_t = enc_out

        K = horizons.shape[0]
        if return_all:
            B, N, d = h_all.shape
            # Vectorize over K horizons by tiling along batch dim.
            h_exp = h_all.unsqueeze(1).expand(B, K, N, d).reshape(B * K, N, d)
            dt_exp = horizons.unsqueeze(0).expand(B, K).reshape(B * K).to(
                device=h_all.device, dtype=torch.float32)
            kpm_exp = (h_kpm.unsqueeze(1).expand(B, K, N).reshape(B * K, N)
                       if h_kpm is not None else None)
            h_pred = self.predictor(h_exp, dt_exp,
                                    key_padding_mask=kpm_exp).view(B, K, d)
        else:
            B, d = h_t.shape
            h_exp = h_t.unsqueeze(1).expand(B, K, d).reshape(B * K, d)
            dt_exp = horizons.unsqueeze(0).expand(B, K).reshape(B * K).to(
                device=h_t.device, dtype=torch.float32)
            h_pred = self.predictor(h_exp, dt_exp).view(B, K, d)

        hazard_logits = self.event_head(h_pred)          # (B, K)
        lambdas = torch.sigmoid(hazard_logits)            # (B, K) ∈ (0,1)
        survival = torch.cumprod(1 - lambdas, dim=-1)     # (B, K) non-increasing
        cdf = 1 - survival                                # (B, K) non-decreasing
        return cdf

    # --- Convenience ---

    def encode(self, context: torch.Tensor,
               mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Encode context → h_t. For probing / analysis."""
        return self.encoder(context, mask)
