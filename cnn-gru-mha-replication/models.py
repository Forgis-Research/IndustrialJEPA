"""
CNN-GRU-MHA architecture for bearing RUL prediction.

Based on:
  Yu et al., "Remaining Useful Life of the Rolling Bearings Prediction Method
  Based on Transfer Learning Integrated with CNN-GRU-MHA"
  Applied Sciences, 2024, 14, 9039.

Architecture:
  Input: (batch=1, seq_len=N_snapshots, 2560) — one snapshot per timestep

  CNN: 6 conv blocks (each Conv1d+BN+ReLU+MaxPool1d), MHA inserted after block 3
    Blocks 1-3: kernel=5, filters=[32, 64, 128]
    MHA: 2 heads, embed_dim=128
    Blocks 4-6: kernel=3, filters=[256, 512, 1024]
    GAP: AdaptiveAvgPool1d(1) -> (batch, 1024)

  GRU: 2 layers, hidden=[512, 128], processes CNN features over time
    Input: (1, N_snapshots, 1024)
    Output: (1, N_snapshots, 128)

  FC: Linear(128, 64) + ReLU -> Linear(64, 1) + Sigmoid
"""

import contextlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


# =====================================================================
# CNN Feature Extractor
# =====================================================================

class ConvBlock(nn.Module):
    """Conv1d + BN + ReLU + MaxPool1d."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int, pool: int = 2):
        super().__init__()
        padding = kernel // 2  # 'same' padding
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=kernel, padding=padding),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(pool),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CNNExtractor(nn.Module):
    """
    6-layer CNN feature extractor with MHA after block 3.

    Input: (batch, 1, 2560) — single channel preprocessed signal
    Output: (batch, 1024) — global average pooled features
    """

    def __init__(self):
        super().__init__()
        # Blocks 1-3: large kernels (5), expanding filters
        self.block1 = ConvBlock(1, 32, kernel=5, pool=2)      # -> (B, 32, 1280)
        self.block2 = ConvBlock(32, 64, kernel=5, pool=2)     # -> (B, 64, 640)
        self.block3 = ConvBlock(64, 128, kernel=5, pool=2)    # -> (B, 128, 320)

        # Multi-Head Attention after block 3
        # Input: (batch, seq_len=320, embed_dim=128)
        self.mha = nn.MultiheadAttention(embed_dim=128, num_heads=2, batch_first=True)
        self.mha_norm = nn.LayerNorm(128)

        # Blocks 4-6: smaller kernels (3), deeper filters
        self.block4 = ConvBlock(128, 256, kernel=3, pool=2)   # -> (B, 256, 160)
        self.block5 = ConvBlock(256, 512, kernel=3, pool=2)   # -> (B, 512, 80)
        self.block6 = ConvBlock(512, 1024, kernel=3, pool=2)  # -> (B, 1024, 40)

        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 1, 2560)
        Returns:
            (batch, 1024)
        """
        # Blocks 1-3
        x = self.block1(x)   # (B, 32, 1280)
        x = self.block2(x)   # (B, 64, 640)
        x = self.block3(x)   # (B, 128, 320)

        # Multi-Head Attention
        # Reshape: (B, 128, 320) -> (B, 320, 128) for MHA
        x_t = x.permute(0, 2, 1)  # (B, seq=320, embed=128)
        attn_out, _ = self.mha(x_t, x_t, x_t)  # (B, 320, 128)
        x_t = self.mha_norm(x_t + attn_out)     # residual + norm
        x = x_t.permute(0, 2, 1)               # (B, 128, 320)

        # Blocks 4-6
        x = self.block4(x)   # (B, 256, 160)
        x = self.block5(x)   # (B, 512, 80)
        x = self.block6(x)   # (B, 1024, 40)

        # Global Average Pooling
        x = self.gap(x)      # (B, 1024, 1)
        x = x.squeeze(-1)    # (B, 1024)
        return x


# =====================================================================
# Full CNN-GRU-MHA Model
# =====================================================================

class CNNGRUMHAModel(nn.Module):
    """
    Full CNN-GRU-MHA model for bearing RUL prediction.

    Designed for sequence-level prediction:
      - CNN processes each snapshot independently to extract features
      - GRU processes the feature sequence over time
      - FC head predicts RUL at each timestep

    Forward pass:
      Input: (N_snapshots, 2560) — all snapshots for ONE bearing
      Output: (N_snapshots,) — RUL prediction at each timestep
    """

    def __init__(
        self,
        cnn_out: int = 1024,
        gru_hidden1: int = 512,
        gru_hidden2: int = 128,
        fc_hidden: int = 64,
        cnn_batch_size: int = 64,  # batch size for CNN processing (memory efficiency)
    ):
        super().__init__()
        self.cnn_batch_size = cnn_batch_size

        self.cnn = CNNExtractor()

        # Two-layer GRU
        self.gru1 = nn.GRU(
            input_size=cnn_out,
            hidden_size=gru_hidden1,
            num_layers=1,
            batch_first=True,
        )
        self.gru2 = nn.GRU(
            input_size=gru_hidden1,
            hidden_size=gru_hidden2,
            num_layers=1,
            batch_first=True,
        )

        # FC prediction head
        self.fc = nn.Sequential(
            nn.Linear(gru_hidden2, fc_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(fc_hidden, 1),
            nn.Sigmoid(),
        )

    def extract_cnn_features(
        self, snapshots: torch.Tensor, use_grad: bool = True
    ) -> torch.Tensor:
        """
        Extract CNN features from all snapshots (batched for efficiency).

        Args:
            snapshots: (N, 2560) — all snapshots for one bearing
            use_grad: if False, run without gradient (for frozen CNN inference)
        Returns:
            (N, 1024) — CNN features
        """
        N = snapshots.shape[0]
        device = snapshots.device
        features_list = []

        context = torch.no_grad() if not use_grad else torch.inference_mode(mode=False)
        with context if not use_grad else contextlib.nullcontext():
            for start in range(0, N, self.cnn_batch_size):
                end = min(start + self.cnn_batch_size, N)
                batch = snapshots[start:end]  # (batch, 2560)
                batch = batch.unsqueeze(1)    # (batch, 1, 2560)
                feat = self.cnn(batch)        # (batch, 1024)
                features_list.append(feat)

        features = torch.cat(features_list, dim=0)  # (N, 1024)
        return features

    def forward(self, snapshots: torch.Tensor, cnn_requires_grad: bool = True) -> torch.Tensor:
        """
        Full forward pass for a single bearing.

        Args:
            snapshots: (N, 2560) — all snapshots for one bearing (one sequence)
            cnn_requires_grad: if False, CNN features are detached (for FC-only finetune)
        Returns:
            (N,) — RUL predictions
        """
        N = snapshots.shape[0]

        # Step 1: Extract CNN features for all snapshots
        # During FC-only fine-tuning, CNN is frozen — no need to keep CNN gradients
        cnn_grad = cnn_requires_grad and any(p.requires_grad for p in self.cnn.parameters())
        features = self.extract_cnn_features(snapshots, use_grad=cnn_grad)  # (N, 1024)

        # Detach if CNN doesn't need gradients (saves memory)
        if not cnn_grad:
            features = features.detach()

        # Step 2: GRU over time sequence
        # GRU expects (batch=1, seq=N, input=1024)
        seq = features.unsqueeze(0)     # (1, N, 1024)
        out1, _ = self.gru1(seq)        # (1, N, 512)
        out2, _ = self.gru2(out1)       # (1, N, 128)
        out2 = out2.squeeze(0)          # (N, 128)

        # Detach GRU if it's frozen too (FC-only finetune)
        gru_frozen = not any(
            p.requires_grad for p in list(self.gru1.parameters()) + list(self.gru2.parameters())
        )
        if gru_frozen:
            out2 = out2.detach()

        # Step 3: FC head per timestep
        rul = self.fc(out2)             # (N, 1)
        return rul.squeeze(-1)          # (N,)

    def forward_finetune(self, snapshots: torch.Tensor) -> torch.Tensor:
        """Alias for forward — used during fine-tuning FC only."""
        return self.forward(snapshots)

    def freeze_feature_extractor(self):
        """Freeze CNN + GRU parameters (for transfer learning)."""
        for param in self.cnn.parameters():
            param.requires_grad = False
        for param in self.gru1.parameters():
            param.requires_grad = False
        for param in self.gru2.parameters():
            param.requires_grad = False
        print("  Frozen: CNN + GRU (only FC head trainable)")

    def unfreeze_all(self):
        """Unfreeze all parameters."""
        for param in self.parameters():
            param.requires_grad = True

    def get_fc_parameters(self):
        """Return only FC head parameters (for optimizer during fine-tuning)."""
        return list(self.fc.parameters())

    def count_parameters(self) -> dict:
        """Count trainable vs total parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        cnn_params = sum(p.numel() for p in self.cnn.parameters())
        gru_params = (sum(p.numel() for p in self.gru1.parameters()) +
                      sum(p.numel() for p in self.gru2.parameters()))
        fc_params = sum(p.numel() for p in self.fc.parameters())
        return {
            "total": total,
            "trainable": trainable,
            "cnn": cnn_params,
            "gru": gru_params,
            "fc": fc_params,
        }


# =====================================================================
# Loss function
# =====================================================================

def rmse_l1_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    model: CNNGRUMHAModel,
    alpha: float = 1e-4,
) -> Tuple[torch.Tensor, dict]:
    """
    RMSE + L1 regularization loss.

    Loss = sqrt(mean((pred - target)^2)) + alpha * sum(|w|)

    Args:
        pred: (N,) predictions
        target: (N,) ground truth
        model: model for L1 regularization
        alpha: L1 regularization coefficient

    Returns:
        (loss, info_dict)
    """
    mse = F.mse_loss(pred, target)
    rmse = torch.sqrt(mse + 1e-8)

    # L1 regularization over all parameters
    l1_reg = torch.tensor(0.0, device=pred.device)
    for param in model.parameters():
        if param.requires_grad:
            l1_reg = l1_reg + param.abs().sum()

    loss = rmse + alpha * l1_reg

    return loss, {
        "rmse": rmse.item(),
        "l1_reg": (alpha * l1_reg).item(),
        "total": loss.item(),
    }
