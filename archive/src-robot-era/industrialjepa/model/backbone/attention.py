# SPDX-FileCopyrightText: 2025 Industrial World Model Authors
# SPDX-License-Identifier: MIT

"""
Sparse Attention for hybrid Mamba-Transformer.

Implements efficient attention that focuses on:
1. Anomaly tokens (high reconstruction error)
2. Action tokens (control inputs)
3. Query tokens (for LLM reasoning)

This allows O(L) overall complexity while maintaining attention
for critical interactions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class SparseAttention(nn.Module):
    """
    Sparse multi-head attention for critical tokens.

    Instead of attending to all tokens, focuses on:
    - Top-k tokens by importance score
    - Fixed special tokens (actions, queries)
    - Local window attention
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        dropout: float = 0.1,
        max_seq_len: int = 8192,
        top_k: int = 64,
        window_size: int = 128,
        use_flash: bool = True,
    ):
        """
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            dropout: Dropout rate
            max_seq_len: Maximum sequence length
            top_k: Number of top tokens to attend to
            window_size: Local window size for each position
            use_flash: Use flash attention if available
        """
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.top_k = top_k
        self.window_size = window_size
        self.use_flash = use_flash

        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        # QKV projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # Importance score predictor (for selecting which tokens to attend to)
        self.importance_predictor = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 1),
        )

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Scale
        self.scale = self.d_head ** -0.5

    def forward(
        self,
        x: torch.Tensor,
        importance_mask: Optional[torch.Tensor] = None,
        special_token_mask: Optional[torch.Tensor] = None,
        causal: bool = True,
    ) -> torch.Tensor:
        """
        Sparse attention forward pass.

        Args:
            x: [B, L, D] input tensor
            importance_mask: [B, L] pre-computed importance scores (optional)
            special_token_mask: [B, L] boolean mask for special tokens (always attend)
            causal: Whether to use causal masking

        Returns:
            [B, L, D] output tensor
        """
        B, L, D = x.shape

        # Project to Q, K, V
        Q = self.q_proj(x)  # [B, L, D]
        K = self.k_proj(x)  # [B, L, D]
        V = self.v_proj(x)  # [B, L, D]

        # Reshape for multi-head attention
        Q = Q.view(B, L, self.n_heads, self.d_head).transpose(1, 2)  # [B, H, L, d]
        K = K.view(B, L, self.n_heads, self.d_head).transpose(1, 2)  # [B, H, L, d]
        V = V.view(B, L, self.n_heads, self.d_head).transpose(1, 2)  # [B, H, L, d]

        # Compute importance scores if not provided
        if importance_mask is None:
            importance_mask = self.importance_predictor(x).squeeze(-1)  # [B, L]

        # Get top-k indices per position (sparse selection)
        # For each position, we attend to: local window + top-k global + special tokens
        sparse_indices = self._get_sparse_indices(
            importance_mask, special_token_mask, L, causal
        )

        # Compute attention with sparse pattern
        # For simplicity, we'll use a dense-sparse hybrid approach here
        # Production would use block-sparse kernels

        if L <= 512:
            # For short sequences, use full attention
            attn_output = self._full_attention(Q, K, V, causal)
        else:
            # For long sequences, use sparse attention
            attn_output = self._sparse_attention(
                Q, K, V, importance_mask, special_token_mask, causal
            )

        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, L, D)

        # Output projection
        return self.out_proj(self.dropout(attn_output))

    def _full_attention(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        causal: bool,
    ) -> torch.Tensor:
        """Standard full attention for short sequences."""
        B, H, L, d = Q.shape

        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # [B, H, L, L]

        # Apply causal mask if needed
        if causal:
            mask = torch.triu(
                torch.ones(L, L, device=Q.device, dtype=torch.bool), diagonal=1
            )
            scores = scores.masked_fill(mask, float('-inf'))

        # Softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply to values
        return torch.matmul(attn_weights, V)  # [B, H, L, d]

    def _sparse_attention(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        importance_mask: torch.Tensor,
        special_token_mask: Optional[torch.Tensor],
        causal: bool,
    ) -> torch.Tensor:
        """
        Sparse attention for long sequences.

        Uses a combination of:
        1. Local window attention
        2. Global attention to top-k important tokens
        3. Attention to special tokens (actions, queries)
        """
        B, H, L, d = Q.shape

        # For each query position, compute attention to:
        # 1. Local window of size window_size
        # 2. Top-k global tokens by importance

        # Get global top-k indices
        _, top_k_indices = torch.topk(importance_mask, min(self.top_k, L), dim=-1)  # [B, top_k]

        # Include special tokens in global attention
        if special_token_mask is not None:
            special_indices = special_token_mask.nonzero(as_tuple=True)[1]
            # Combine with top-k (simplified - full implementation would handle batches)

        # Simplified: process in chunks with local + global attention
        chunk_size = self.window_size
        outputs = []

        for start in range(0, L, chunk_size):
            end = min(start + chunk_size, L)
            chunk_len = end - start

            # Query chunk
            Q_chunk = Q[:, :, start:end, :]  # [B, H, chunk_len, d]

            # Keys/values: local window + global top-k
            # Local window: [start - window_size//2, end + window_size//2]
            local_start = max(0, start - self.window_size // 2)
            local_end = min(L, end + self.window_size // 2)

            K_local = K[:, :, local_start:local_end, :]
            V_local = V[:, :, local_start:local_end, :]

            # Global top-k
            K_global = K.gather(
                2, top_k_indices.unsqueeze(1).unsqueeze(-1).expand(-1, H, -1, d)
            )
            V_global = V.gather(
                2, top_k_indices.unsqueeze(1).unsqueeze(-1).expand(-1, H, -1, d)
            )

            # Concatenate local and global
            K_combined = torch.cat([K_local, K_global], dim=2)
            V_combined = torch.cat([V_local, V_global], dim=2)

            # Compute attention
            scores = torch.matmul(Q_chunk, K_combined.transpose(-2, -1)) * self.scale

            # Apply causal mask for local portion (if needed)
            if causal:
                local_len = local_end - local_start
                # Create mask for this chunk
                q_positions = torch.arange(start, end, device=Q.device)
                kv_positions = torch.cat([
                    torch.arange(local_start, local_end, device=Q.device),
                    top_k_indices[0]  # Simplified: use first batch
                ])
                mask = q_positions.unsqueeze(1) < kv_positions.unsqueeze(0)
                scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))

            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.dropout(attn_weights)

            chunk_output = torch.matmul(attn_weights, V_combined)
            outputs.append(chunk_output)

        return torch.cat(outputs, dim=2)

    def _get_sparse_indices(
        self,
        importance_mask: torch.Tensor,
        special_token_mask: Optional[torch.Tensor],
        seq_len: int,
        causal: bool,
    ) -> torch.Tensor:
        """Compute sparse attention indices."""
        B = importance_mask.shape[0]

        # Top-k by importance
        _, top_k_idx = torch.topk(importance_mask, min(self.top_k, seq_len), dim=-1)

        # Add special token indices
        if special_token_mask is not None:
            special_idx = special_token_mask.nonzero(as_tuple=True)
            # Would combine here in full implementation

        return top_k_idx


class LocalAttention(nn.Module):
    """
    Pure local (window) attention for efficiency.

    Each position only attends to positions within a fixed window.
    O(L * W) complexity where W is window size.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        window_size: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.window_size = window_size

        self.qkv_proj = nn.Linear(d_model, d_model * 3)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.scale = self.d_head ** -0.5

    def forward(self, x: torch.Tensor, causal: bool = True) -> torch.Tensor:
        """
        Args:
            x: [B, L, D]
            causal: Use causal masking

        Returns:
            [B, L, D]
        """
        B, L, D = x.shape
        W = self.window_size

        # QKV projection
        qkv = self.qkv_proj(x)
        Q, K, V = qkv.chunk(3, dim=-1)

        # Reshape for multi-head
        Q = Q.view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        K = K.view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        V = V.view(B, L, self.n_heads, self.d_head).transpose(1, 2)

        # Pad sequence for windowing
        pad_len = (W - L % W) % W
        if pad_len > 0:
            Q = F.pad(Q, (0, 0, 0, pad_len))
            K = F.pad(K, (0, 0, 0, pad_len))
            V = F.pad(V, (0, 0, 0, pad_len))

        L_padded = L + pad_len
        num_windows = L_padded // W

        # Reshape into windows
        Q = Q.view(B, self.n_heads, num_windows, W, self.d_head)
        K = K.view(B, self.n_heads, num_windows, W, self.d_head)
        V = V.view(B, self.n_heads, num_windows, W, self.d_head)

        # Compute attention within each window
        scores = torch.einsum('bhnqd,bnkd->bhnqk', Q, K) * self.scale

        # Causal mask within window
        if causal:
            mask = torch.triu(torch.ones(W, W, device=x.device, dtype=torch.bool), diagonal=1)
            scores = scores.masked_fill(mask, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply to values
        out = torch.einsum('bhnqk,bnkd->bhnqd', attn_weights, V)

        # Reshape back
        out = out.view(B, self.n_heads, L_padded, self.d_head)
        out = out[:, :, :L, :]  # Remove padding
        out = out.transpose(1, 2).contiguous().view(B, L, D)

        return self.out_proj(out)
