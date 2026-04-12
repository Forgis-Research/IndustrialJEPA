"""
Adapt V11 Trajectory JEPA for anomaly prediction benchmarks.

Uses the same pre-train -> freeze -> downstream classifier pipeline as MTS-JEPA,
but with Trajectory JEPA's causal Transformer encoder.
"""
import os
import sys
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Self-contained — all components defined below. No V11 imports to avoid path conflicts.


class TrajectoryJEPAForAnomalyPrediction(nn.Module):
    """
    Adapted Trajectory JEPA for anomaly prediction.

    Architecture:
    - Input: (B, T_w, V) multivariate window
    - SensorProjection: (B, T_w, V) -> (B, T_w, d_model)
    - Causal Transformer: (B, T_w, d_model) -> (B, T_w, d_model)
    - Pool last token: (B, d_model)
    - This gives a single representation per window

    For anomaly prediction:
    - Pre-train with variable-horizon prediction objective
    - Freeze encoder
    - Train MLP classifier on encoded representations
    """
    def __init__(self, n_vars, d_model=256, n_heads=4, n_layers=2,
                 d_ff=512, dropout=0.1, ema_momentum=0.99,
                 predictor_hidden=256, window_length=100):
        super().__init__()
        self.n_vars = n_vars
        self.d_model = d_model
        self.window_length = window_length

        # Sensor projection: (B, T, V) -> (B, T, d_model)
        self.sensor_proj = nn.Linear(n_vars, d_model)

        # Positional encoding
        self.pos_embed = nn.Parameter(torch.randn(1, window_length, d_model) * 0.02)

        # Causal context encoder
        self.context_encoder = nn.ModuleList([
            TransformerEncoderLayerSimple(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

        # Target encoder (EMA)
        self.target_encoder = nn.ModuleList([
            TransformerEncoderLayerSimple(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.target_norm = nn.LayerNorm(d_model)

        # Initialize target = context
        self._copy_to_target()
        for p in self.target_encoder.parameters():
            p.requires_grad = False
        for p in self.target_norm.parameters():
            p.requires_grad = False

        self.ema_momentum = ema_momentum

        # Predictor
        self.predictor = nn.Sequential(
            nn.Linear(d_model, predictor_hidden),
            nn.ReLU(),
            nn.LayerNorm(predictor_hidden),
            nn.Linear(predictor_hidden, d_model),
        )

    def _copy_to_target(self):
        for (tc, tt) in zip(self.context_encoder.parameters(),
                            self.target_encoder.parameters()):
            tt.data.copy_(tc.data)
        for (nc, nt) in zip(self.norm.parameters(), self.target_norm.parameters()):
            nt.data.copy_(nc.data)

    @torch.no_grad()
    def update_ema(self):
        m = self.ema_momentum
        for (pc, pt) in zip(self.context_encoder.parameters(),
                            self.target_encoder.parameters()):
            pt.data.mul_(m).add_(pc.data, alpha=1 - m)
        for (nc, nt) in zip(self.norm.parameters(), self.target_norm.parameters()):
            nt.data.mul_(m).add_(nc.data, alpha=1 - m)

    def _encode(self, x, encoder, norm, causal=True):
        """Encode a window through Transformer."""
        B, T, V = x.shape
        h = self.sensor_proj(x)  # (B, T, d_model)
        h = h + self.pos_embed[:, :T, :]

        # Create causal mask if needed
        if causal:
            mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        else:
            mask = None

        for layer in encoder:
            h = layer(h, mask)
        h = norm(h)

        # Pool: take last token
        return h[:, -1, :]  # (B, d_model)

    def forward_pretrain(self, x_context, x_target):
        """
        Pre-training forward pass.

        x_context: (B, T, V)
        x_target: (B, T, V)

        Returns: pred_loss, var_loss
        """
        # Encode context (causal)
        h_past = self._encode(x_context, self.context_encoder, self.norm, causal=True)

        # Predict future representation
        h_pred = self.predictor(h_past)

        # Encode target (bidirectional, EMA, no grad)
        with torch.no_grad():
            h_future = self._encode(x_target, self.target_encoder, self.target_norm, causal=False)

        # Losses
        pred_loss = F.mse_loss(h_pred, h_future.detach())

        # Variance regularizer
        std = h_future.std(dim=0)
        var_loss = F.relu(1.0 - std).mean()

        return pred_loss, var_loss

    def encode_for_downstream(self, x):
        """
        Encode context window for downstream classification.
        x: (B, T, V)
        Returns: (B, d_model) representation
        """
        with torch.no_grad():
            return self._encode(x, self.context_encoder, self.norm, causal=True)


class TransformerEncoderLayerSimple(nn.Module):
    """Simple pre-norm Transformer layer."""
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, attn_mask=None):
        x2 = self.norm1(x)
        x = x + self.attn(x2, x2, x2, attn_mask=attn_mask)[0]
        x = x + self.ffn(self.norm2(x))
        return x


def pretrain_trajectory_jepa(model, train_loader, val_loader, n_vars,
                              device='cuda', n_epochs=100, lr=3e-4):
    """Pre-train Trajectory JEPA on anomaly prediction data."""
    from data_utils import RevIN

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=0.01
    )
    revin = RevIN(n_vars).to(device)

    best_val_loss = float('inf')
    best_state = None

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0
        n_batches = 0

        for x_ctx, x_tgt in train_loader:
            x_ctx = x_ctx.to(device)
            x_tgt = x_tgt.to(device)

            x_ctx_n = revin(x_ctx)
            x_tgt_n = revin(x_tgt)

            pred_loss, var_loss = model.forward_pretrain(x_ctx_n, x_tgt_n)
            loss = pred_loss + 0.01 * var_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            model.update_ema()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / n_batches

        # Validation
        model.eval()
        val_loss = 0
        val_batches = 0
        with torch.no_grad():
            for x_ctx, x_tgt in val_loader:
                x_ctx = x_ctx.to(device)
                x_tgt = x_tgt.to(device)
                x_ctx_n = revin(x_ctx)
                x_tgt_n = revin(x_tgt)
                pl, vl = model.forward_pretrain(x_ctx_n, x_tgt_n)
                val_loss += (pl + 0.01 * vl).item()
                val_batches += 1

        avg_val = val_loss / max(val_batches, 1)

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0:
            print(f"  TrajJEPA epoch {epoch}: train={avg_loss:.4f}, val={avg_val:.4f}")

    if best_state:
        model.load_state_dict(best_state)

    return best_val_loss
