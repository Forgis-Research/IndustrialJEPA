# SPDX-FileCopyrightText: 2025 Industrial World Model Authors
# SPDX-License-Identifier: MIT

"""
Mamba (Selective State Space Model) implementation for time series.

Based on "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
(Gu & Dao, 2024) with adaptations for industrial time series.

Key features:
- O(L) complexity vs O(L^2) for attention
- Input-dependent (selective) state transitions
- Hardware-efficient parallel scan
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from einops import rearrange, repeat
import math


class SelectiveSSM(nn.Module):
    """
    Selective State Space Model - the core of Mamba.

    Unlike traditional SSMs with fixed A, B matrices, the selective SSM
    makes these parameters input-dependent, allowing the model to
    selectively propagate or forget information.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: str = "auto",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = "random",
        dt_scale: float = 1.0,
        bias: bool = False,
        conv_bias: bool = True,
    ):
        """
        Args:
            d_model: Model dimension
            d_state: SSM state dimension
            d_conv: Local convolution width
            expand: Block expansion factor
            dt_rank: Rank of dt projection ("auto" = d_model // 16)
            dt_min, dt_max: Range for dt initialization
            dt_init: Initialization method ("random" or "constant")
            dt_scale: Scale for dt
            bias: Use bias in linear layers
            conv_bias: Use bias in conv layer
        """
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(expand * d_model)

        # Dt rank
        if dt_rank == "auto":
            self.dt_rank = max(d_model // 16, 1)
        else:
            self.dt_rank = int(dt_rank)

        # Input projection: projects to (d_inner, d_inner) for x and z branches
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=bias)

        # 1D convolution for local context
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
            bias=conv_bias,
        )

        # SSM parameter projections (input-dependent B, C, and dt)
        # Projects from d_inner to: dt_rank + d_state * 2 (for B and C)
        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + d_state * 2, bias=False
        )

        # dt projection
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # Initialize dt bias for proper range
        dt_init_std = self.dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)

        # dt bias initialization
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_min)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

        # A parameter (log-space for stability)
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32),
            "n -> d n",
            d=self.d_inner,
        )
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True

        # D parameter (skip connection)
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.D._no_weight_decay = True

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias)

    def forward(
        self,
        x: torch.Tensor,
        cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass of Selective SSM.

        Args:
            x: [B, L, D] input tensor
            cache: Optional (conv_state, ssm_state) for incremental decoding

        Returns:
            output: [B, L, D]
            new_cache: Updated cache (if cache was provided)
        """
        B, L, D = x.shape

        # Input projection and split
        xz = self.in_proj(x)  # [B, L, 2*d_inner]
        x_branch, z = xz.chunk(2, dim=-1)  # Each [B, L, d_inner]

        # Convolution (causal)
        x_conv = rearrange(x_branch, "b l d -> b d l")
        if cache is not None:
            conv_state, ssm_state = cache
            x_conv = torch.cat([conv_state, x_conv], dim=-1)
        x_conv = self.conv1d(x_conv)[:, :, :L]
        x_conv = rearrange(x_conv, "b d l -> b l d")

        # Activation
        x_conv = F.silu(x_conv)

        # SSM parameter computation (input-dependent)
        x_proj = self.x_proj(x_conv)  # [B, L, dt_rank + 2*d_state]
        dt, B_proj, C_proj = x_proj.split(
            [self.dt_rank, self.d_state, self.d_state], dim=-1
        )

        # dt projection with softplus activation
        dt = self.dt_proj(dt)  # [B, L, d_inner]
        dt = F.softplus(dt)  # Ensure positive

        # Get A (from log-space)
        A = -torch.exp(self.A_log)  # [d_inner, d_state]

        # Run SSM
        y, new_ssm_state = self.selective_scan(
            x_conv, dt, A, B_proj, C_proj,
            initial_state=ssm_state if cache is not None else None,
        )

        # Skip connection with D
        y = y + self.D * x_conv

        # Gate with z branch
        y = y * F.silu(z)

        # Output projection
        output = self.out_proj(y)

        # Update cache if provided
        new_cache = None
        if cache is not None:
            new_conv_state = x_branch[:, -self.d_conv + 1:, :].transpose(1, 2)
            new_cache = (new_conv_state, new_ssm_state)

        return output, new_cache

    def selective_scan(
        self,
        x: torch.Tensor,
        dt: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        initial_state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Selective scan (parallel implementation).

        This is a simplified version. The actual Mamba uses CUDA kernels
        for hardware-efficient parallel scan.

        Args:
            x: [B, L, d_inner] input
            dt: [B, L, d_inner] time deltas
            A: [d_inner, d_state] state matrix
            B: [B, L, d_state] input projection
            C: [B, L, d_state] output projection
            initial_state: [B, d_inner, d_state] initial hidden state

        Returns:
            y: [B, L, d_inner] output
            final_state: [B, d_inner, d_state] final hidden state
        """
        B_batch, L, d_inner = x.shape
        d_state = A.shape[1]

        # Discretize A and B
        # dA = exp(dt * A)
        dA = torch.exp(
            torch.einsum("bld,dn->bldn", dt, A)
        )  # [B, L, d_inner, d_state]

        # dB = dt * B (simplified discretization)
        dB = torch.einsum("bld,bln->bldn", dt, B)  # [B, L, d_inner, d_state]

        # Initialize state
        if initial_state is None:
            h = torch.zeros(
                B_batch, d_inner, d_state, device=x.device, dtype=x.dtype
            )
        else:
            h = initial_state

        # Sequential scan (for correctness - production would use parallel scan)
        ys = []
        for i in range(L):
            # State update: h = dA * h + dB * x
            h = dA[:, i] * h + dB[:, i] * x[:, i:i+1, :].transpose(1, 2)
            # Output: y = C * h
            y = torch.einsum("bdn,bn->bd", h, C[:, i])
            ys.append(y)

        y = torch.stack(ys, dim=1)  # [B, L, d_inner]

        return y, h


class MambaBlock(nn.Module):
    """
    Full Mamba block with normalization and residual connection.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
        use_rms_norm: bool = True,
    ):
        super().__init__()

        # Normalization
        if use_rms_norm:
            self.norm = RMSNorm(d_model)
        else:
            self.norm = nn.LayerNorm(d_model)

        # Selective SSM
        self.ssm = SelectiveSSM(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward with residual connection.

        Args:
            x: [B, L, D]
            cache: Optional cache for incremental decoding

        Returns:
            output: [B, L, D]
            new_cache: Updated cache
        """
        residual = x
        x = self.norm(x)
        x, new_cache = self.ssm(x, cache=cache)
        x = self.dropout(x)
        return residual + x, new_cache


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class BidirectionalMamba(nn.Module):
    """
    Bidirectional Mamba for non-causal time series processing.

    Processes sequence in both directions and combines outputs.
    Useful for classification/representation tasks where full context is available.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.forward_mamba = MambaBlock(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dropout=dropout,
        )

        self.backward_mamba = MambaBlock(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dropout=dropout,
        )

        # Combine forward and backward
        self.combine = nn.Linear(d_model * 2, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, L, D]

        Returns:
            [B, L, D] bidirectional output
        """
        # Forward direction
        forward_out, _ = self.forward_mamba(x)

        # Backward direction (flip, process, flip back)
        x_flip = torch.flip(x, dims=[1])
        backward_out, _ = self.backward_mamba(x_flip)
        backward_out = torch.flip(backward_out, dims=[1])

        # Combine
        combined = torch.cat([forward_out, backward_out], dim=-1)
        return self.combine(combined)
