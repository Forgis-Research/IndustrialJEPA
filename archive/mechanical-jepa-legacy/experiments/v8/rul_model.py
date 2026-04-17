"""
RUL Temporal Models for Stage 2 (after JEPA pretraining).

Variant A — MLP (single snapshot, no history):
    Input: [z_t (256), elapsed_time (1)] → MLP → RUL%

Variant B — LSTM (with history, main method):
    Per-step: [z_t (256), delta_t (1)] → LSTM → hidden
    Final: [lstm_out (128), elapsed_time (1)] → Linear → RUL%
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class RULMLP(nn.Module):
    """
    Single-snapshot MLP: no temporal context.
    Tests whether JEPA embeddings alone (+ clock) can predict RUL.
    """
    def __init__(self, embed_dim: int = 256, hidden_dim: int = 128):
        super().__init__()
        # Input: [embedding (embed_dim), elapsed_time_normalized (1)]
        self.net = nn.Sequential(
            nn.Linear(embed_dim + 1, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor, elapsed_time: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (B, D) embedding
            elapsed_time: (B, 1) normalized elapsed time [0, 1]

        Returns: (B, 1) RUL prediction [0, 1]
        """
        x = torch.cat([z, elapsed_time], dim=-1)
        return self.net(x)


class RULLSTM(nn.Module):
    """
    LSTM-based temporal model: processes episode sequence.
    Main method.
    """
    def __init__(
        self,
        embed_dim: int = 256,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Per-step input: [embedding (embed_dim), delta_t_normalized (1)]
        self.lstm = nn.LSTM(
            input_size=embed_dim + 1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )

        # Final prediction: [lstm_out (hidden_size), elapsed_time (1)]
        self.output_head = nn.Sequential(
            nn.LayerNorm(hidden_size + 1),
            nn.Linear(hidden_size + 1, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        z_seq: torch.Tensor,
        delta_t_seq: torch.Tensor,
        elapsed_time: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            z_seq: (B, T, D) sequence of embeddings
            delta_t_seq: (B, T, 1) time since previous snapshot
            elapsed_time: (B, T, 1) total elapsed time at each step

        Returns: (B, T, 1) RUL predictions for each step
        """
        # Concatenate embedding with delta_t
        x = torch.cat([z_seq, delta_t_seq], dim=-1)  # (B, T, D+1)

        # LSTM
        lstm_out, _ = self.lstm(x)  # (B, T, hidden_size)

        # Predict at each step using elapsed time
        combined = torch.cat([lstm_out, elapsed_time], dim=-1)  # (B, T, hidden_size+1)
        preds = self.output_head(combined)  # (B, T, 1)
        return preds


class HandcraftedMLP(nn.Module):
    """Baseline: handcrafted features + elapsed_time → MLP → RUL%"""
    def __init__(self, n_features: int = 18, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features + 1, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, features: torch.Tensor, elapsed_time: torch.Tensor) -> torch.Tensor:
        x = torch.cat([features, elapsed_time], dim=-1)
        return self.net(x)


class HandcraftedLSTM(nn.Module):
    """Baseline: handcrafted features over time → LSTM → RUL%"""
    def __init__(self, n_features: int = 18, hidden_size: int = 64, num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features + 1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=0.1 if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size + 1),
            nn.Linear(hidden_size + 1, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, feat_seq: torch.Tensor, delta_t_seq: torch.Tensor,
                elapsed_time: torch.Tensor) -> torch.Tensor:
        x = torch.cat([feat_seq, delta_t_seq], dim=-1)
        out, _ = self.lstm(x)
        combined = torch.cat([out, elapsed_time], dim=-1)
        return self.head(combined)


class EnvelopeRMSLSTM(nn.Module):
    """Baseline: scalar envelope RMS + delta_t → LSTM → RUL%"""
    def __init__(self, hidden_size: int = 32, num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=2,  # [env_rms, delta_t]
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=0.1 if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size + 1, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid(),
        )

    def forward(self, env_rms_seq: torch.Tensor, delta_t_seq: torch.Tensor,
                elapsed_time: torch.Tensor) -> torch.Tensor:
        # env_rms_seq: (B, T, 1)
        x = torch.cat([env_rms_seq, delta_t_seq], dim=-1)  # (B, T, 2)
        out, _ = self.lstm(x)
        combined = torch.cat([out, elapsed_time], dim=-1)
        return self.head(combined)


class CNNEncoder(nn.Module):
    """1D CNN feature extractor for end-to-end CNN-LSTM baseline."""
    def __init__(self, in_channels: int = 1, out_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=16, stride=4, padding=6),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=8, stride=4, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, out_dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, L)
        return self.net(x).squeeze(-1)  # (B, out_dim)


class EndToEndCNNLSTM(nn.Module):
    """Supervised baseline: jointly trained CNN + LSTM."""
    def __init__(self, cnn_out: int = 64, hidden_size: int = 64, num_layers: int = 2):
        super().__init__()
        self.cnn = CNNEncoder(1, cnn_out)
        self.lstm = nn.LSTM(
            input_size=cnn_out + 1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=0.1 if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size + 1, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, signal_seq: torch.Tensor, delta_t_seq: torch.Tensor,
                elapsed_time: torch.Tensor) -> torch.Tensor:
        """
        signal_seq: (B, T, 1, L) — sequence of raw signals
        delta_t_seq: (B, T, 1)
        elapsed_time: (B, T, 1)
        """
        B, T, C, L = signal_seq.shape
        # Encode each step with CNN
        z = self.cnn(signal_seq.view(B * T, C, L))  # (B*T, cnn_out)
        z = z.view(B, T, -1)  # (B, T, cnn_out)
        x = torch.cat([z, delta_t_seq], dim=-1)
        out, _ = self.lstm(x)
        combined = torch.cat([out, elapsed_time], dim=-1)
        return self.head(combined)


class TCNBlock(nn.Module):
    """Dilated causal convolution block for TCN."""
    def __init__(self, n_inputs: int, n_outputs: int, kernel_size: int, dilation: int, dropout: float = 0.1):
        super().__init__()
        pad = (kernel_size - 1) * dilation  # causal padding
        self.conv1 = nn.utils.weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                                     dilation=dilation, padding=pad))
        self.conv2 = nn.utils.weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                                     dilation=dilation, padding=pad))
        self.net = nn.Sequential(
            self.conv1, nn.ReLU(), nn.Dropout(dropout),
            self.conv2, nn.ReLU(), nn.Dropout(dropout),
        )
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.pad = pad

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        out = out[:, :, :-self.pad] if self.pad > 0 else out  # remove causal padding
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class CNNGRUMHAEncoder(nn.Module):
    """
    CNN-GRU-MHA baseline (approximate replication of published SOTA on FEMTO).
    Reference: Applied Sciences 2024, nRMSE=0.044.
    """
    def __init__(
        self,
        in_channels: int = 1,
        cnn_channels: int = 32,
        gru_hidden: int = 64,
        n_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        # 1D CNN feature extractor (3 conv layers)
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels, cnn_channels, kernel_size=32, stride=2, padding=15),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(),
            nn.Conv1d(cnn_channels, cnn_channels * 2, kernel_size=16, stride=2, padding=7),
            nn.BatchNorm1d(cnn_channels * 2),
            nn.ReLU(),
            nn.Conv1d(cnn_channels * 2, cnn_channels * 2, kernel_size=8, stride=2, padding=3),
            nn.BatchNorm1d(cnn_channels * 2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        cnn_out = cnn_channels * 2  # 64

        # Bidirectional GRU
        self.gru = nn.GRU(
            input_size=cnn_out + 1,  # +1 for delta_t
            hidden_size=gru_hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )
        gru_out = gru_hidden * 2  # bidirectional

        # Multi-head self-attention over sequence
        self.attn = nn.MultiheadAttention(gru_out, n_heads, dropout=dropout, batch_first=True)
        self.attn_norm = nn.LayerNorm(gru_out)

        # FC output
        self.head = nn.Sequential(
            nn.LayerNorm(gru_out + 1),
            nn.Linear(gru_out + 1, gru_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(gru_hidden, 1),
            nn.Sigmoid(),
        )

    def forward(self, signal_seq: torch.Tensor, delta_t_seq: torch.Tensor,
                elapsed_time: torch.Tensor) -> torch.Tensor:
        """
        signal_seq: (B, T, 1, L)
        delta_t_seq: (B, T, 1)
        elapsed_time: (B, T, 1)
        """
        B, T, C, L = signal_seq.shape
        z = self.cnn(signal_seq.view(B * T, C, L)).squeeze(-1)  # (B*T, cnn_out)
        z = z.view(B, T, -1)  # (B, T, cnn_out)

        x = torch.cat([z, delta_t_seq], dim=-1)  # (B, T, cnn_out+1)
        gru_out, _ = self.gru(x)  # (B, T, gru_hidden*2)

        # Self-attention
        attn_out, _ = self.attn(gru_out, gru_out, gru_out)
        gru_out = self.attn_norm(gru_out + attn_out)

        combined = torch.cat([gru_out, elapsed_time], dim=-1)
        return self.head(combined)


class TransformerRUL(nn.Module):
    """
    Transformer encoder-decoder for RUL (baseline 10).
    Input: handcrafted features per snapshot.
    """
    def __init__(self, n_features: int = 18, d_model: int = 128, n_heads: int = 4,
                 n_layers: int = 4, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(n_features + 1, d_model)

        import math
        # Sinusoidal PE
        max_len = 2000
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pos_embed', pe.unsqueeze(0))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.head = nn.Sequential(
            nn.LayerNorm(d_model + 1),
            nn.Linear(d_model + 1, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, feat_seq: torch.Tensor, delta_t_seq: torch.Tensor,
                elapsed_time: torch.Tensor) -> torch.Tensor:
        x = torch.cat([feat_seq, delta_t_seq], dim=-1)
        x = self.input_proj(x)
        T = x.shape[1]
        x = x + self.pos_embed[:, :T, :]

        # Causal mask
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        x = self.transformer(x, mask=mask)

        combined = torch.cat([x, elapsed_time], dim=-1)
        return self.head(combined)
