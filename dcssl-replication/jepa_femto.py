"""
JEPA+HC Hybrid for FEMTO/PRONOSTIA Bearing RUL Prediction.

Adapts the IndustrialJEPA v8 architecture for FEMTO data:
- 2 channels (horizontal + vertical vibration)
- 2560 samples per snapshot at 25.6 kHz
- TCN encoder (same as DCSSL) for fair comparison, with JEPA-style masking

This is our "JEPA+HC" baseline:
  1. Self-supervised pretraining via JEPA (masked prediction)
  2. Feature extraction: JEPA embeddings + 18 handcrafted features
  3. RUL regression via MLP head

Architecture choices:
- Use TCN encoder (same as DCSSL baselines for fair comparison)
- Predictor: small transformer to predict masked patches
- 18 handcrafted features appended before the RUL head
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from pathlib import Path
from typing import Optional, Tuple, Dict, List

from models import TCNEncoder, RULHead, count_parameters


# =====================================================================
# Handcrafted Feature Extractor
# =====================================================================

def extract_hc_features(snapshot: np.ndarray) -> np.ndarray:
    """
    Extract 18 handcrafted features from a (2560, 2) snapshot.

    6 features per channel × 2 channels = 12 core features
    + 6 cross-channel features = 18 total
    """
    from scipy.stats import kurtosis as sp_kurt, skew as sp_skew

    feats = []
    ch_rms = []
    for ch in range(2):
        x = snapshot[:, ch]
        rms = np.sqrt(np.mean(x ** 2))
        peak = np.max(np.abs(x))
        kurt = float(sp_kurt(x, fisher=False))
        skew = float(sp_skew(x))
        crest = peak / max(rms, 1e-10)
        shape = rms / max(np.mean(np.abs(x)), 1e-10)
        feats.extend([rms, peak, kurt, skew, crest, shape])
        ch_rms.append(rms)

    # Cross-channel features
    x0, x1 = snapshot[:, 0], snapshot[:, 1]
    corr = float(np.corrcoef(x0, x1)[0, 1]) if (x0.std() > 0 and x1.std() > 0) else 0.0
    rms_sum = ch_rms[0] + ch_rms[1]
    rms_ratio = ch_rms[0] / max(ch_rms[1], 1e-10)
    energy = float(np.mean(x0 ** 2 + x1 ** 2))

    # Spectral centroid (mean over channels)
    def spectral_centroid(x):
        X = np.abs(np.fft.rfft(x))
        freqs = np.arange(len(X))
        return (freqs * X).sum() / max(X.sum(), 1e-10)

    sc0 = spectral_centroid(x0)
    sc1 = spectral_centroid(x1)
    feats.extend([corr, rms_sum, rms_ratio, energy, sc0, sc1])

    return np.array(feats, dtype=np.float32)  # 18-dim


# =====================================================================
# JEPA-style pretraining for TCN encoder
# =====================================================================

class MaskedJEPALoss(nn.Module):
    """
    JEPA-style masked prediction loss for time series.

    Instead of contrastive learning, predicts masked patches
    from visible patches using an online encoder + target encoder (EMA).

    This follows I-JEPA (Assran et al. 2023) adapted for 1D bearing signals.
    """

    def __init__(
        self,
        encoder_out: int = 128,
        predictor_hidden: int = 256,
        n_blocks: int = 8,
        momentum: float = 0.996,
        mask_ratio: float = 0.5,
    ):
        super().__init__()
        self.momentum = momentum
        self.mask_ratio = mask_ratio

        self.online_encoder = TCNEncoder(
            in_channels=2, hidden_channels=64, out_channels=encoder_out,
            kernel_size=3, n_blocks=n_blocks, dropout=0.1,
        )
        self.target_encoder = copy.deepcopy(self.online_encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False

        # Predictor: small MLP that predicts target encodings from masked inputs
        self.predictor = nn.Sequential(
            nn.Linear(encoder_out, predictor_hidden),
            nn.ReLU(),
            nn.Linear(predictor_hidden, encoder_out),
        )

        self.rul_head = RULHead(encoder_out, hidden_dim=64)

    @torch.no_grad()
    def update_target(self):
        """EMA update of target encoder."""
        for po, pt in zip(self.online_encoder.parameters(),
                           self.target_encoder.parameters()):
            pt.data = pt.data * self.momentum + po.data * (1 - self.momentum)

    def forward_loss(self, x1: torch.Tensor, x2: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Compute JEPA pretraining loss.

        x1 = masked/augmented view (online encoder input)
        x2 = clean view (target encoder input)
        """
        B, C, T = x1.shape

        # Online branch: encode masked view
        h_online = self.online_encoder(x1)  # (B, D)
        z_online = self.predictor(h_online)  # (B, D)

        # Target branch: encode clean view, no gradients
        with torch.no_grad():
            h_target = self.target_encoder(x2)  # (B, D)
            h_target = F.normalize(h_target, dim=-1)

        z_online = F.normalize(z_online, dim=-1)

        # L2 prediction loss
        loss = F.mse_loss(z_online, h_target)

        # Variance regularization (anti-collapse)
        var_loss = torch.relu(1 - z_online.std(dim=0).mean())

        total = loss + 0.01 * var_loss

        return total, {"loss_pred": loss.item(), "loss_var": var_loss.item(), "total": total.item()}

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.online_encoder(x)

    def predict_rul(self, x: torch.Tensor) -> torch.Tensor:
        h = self.online_encoder(x)
        return self.rul_head(h)

    def contrastive_loss(self, view1, view2, **kwargs):
        """Interface compatibility with train_utils."""
        loss, loss_dict = self.forward_loss(view1, view2)
        self.update_target()
        return loss, loss_dict


class JEPAHCModel(nn.Module):
    """
    JEPA + Handcrafted Features hybrid model.

    Encoder: TCN (same as DCSSL)
    Features: encoder output + 18 handcrafted features
    Head: MLP regressor
    """

    def __init__(
        self,
        in_channels: int = 2,
        encoder_hidden: int = 64,
        encoder_out: int = 128,
        n_tcn_blocks: int = 8,
        kernel_size: int = 3,
        dropout: float = 0.1,
        n_hc_features: int = 18,
        rul_hidden: int = 128,
        momentum: float = 0.996,
    ):
        super().__init__()
        self.encoder_out = encoder_out
        self.n_hc_features = n_hc_features
        self.momentum = momentum

        # Online encoder (trained)
        self.encoder = TCNEncoder(
            in_channels=in_channels,
            hidden_channels=encoder_hidden,
            out_channels=encoder_out,
            kernel_size=kernel_size,
            n_blocks=n_tcn_blocks,
            dropout=dropout,
        )
        # Target encoder (EMA copy)
        self.target_encoder = copy.deepcopy(self.encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False

        # Predictor for JEPA loss
        self.predictor = nn.Sequential(
            nn.Linear(encoder_out, 256),
            nn.ReLU(),
            nn.Linear(256, encoder_out),
        )

        # RUL head: encoder_out + n_hc_features → RUL
        self.rul_head = nn.Sequential(
            nn.Linear(encoder_out + n_hc_features, rul_hidden),
            nn.BatchNorm1d(rul_hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(rul_hidden, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        # Feature normalizer (fit on training data)
        self.feat_mean = None
        self.feat_std = None

    @torch.no_grad()
    def update_target(self):
        for po, pt in zip(self.encoder.parameters(), self.target_encoder.parameters()):
            pt.data = pt.data * self.momentum + po.data * (1 - self.momentum)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def contrastive_loss(self, view1: torch.Tensor, view2: torch.Tensor, **kwargs):
        """JEPA-style loss: predict target encoding from augmented view."""
        h_online = self.encoder(view1)
        z_pred = self.predictor(h_online)
        z_pred = F.normalize(z_pred, dim=-1)

        with torch.no_grad():
            h_target = self.target_encoder(view2)
            h_target = F.normalize(h_target, dim=-1)

        loss_pred = F.mse_loss(z_pred, h_target)
        var_loss = torch.relu(1 - z_pred.std(dim=0).mean())
        total = loss_pred + 0.01 * var_loss

        self.update_target()

        return total, {"loss_pred": loss_pred.item(), "loss_var": var_loss.item(), "total": total.item()}

    def predict_rul(self, x: torch.Tensor, hc_feats: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Predict RUL from raw signal + optional handcrafted features.
        """
        h = self.encoder(x)  # (B, encoder_out)
        if hc_feats is not None:
            feat = torch.cat([h, hc_feats], dim=-1)  # (B, encoder_out + n_hc)
        else:
            # If no HC features provided, use zeros
            zeros = torch.zeros(h.shape[0], self.n_hc_features, device=h.device)
            feat = torch.cat([h, zeros], dim=-1)
        return self.rul_head(feat).squeeze(-1)


# =====================================================================
# Extended RUL Dataset with HC features
# =====================================================================

class FEMTORULDatasetHC(torch.utils.data.Dataset):
    """
    RUL dataset that also provides handcrafted features alongside raw signals.
    """

    def __init__(
        self,
        bearing_data_list: List[Dict],
        augment: bool = False,
        crop_length: int = 2560,
        normalize_hc: bool = True,
        feat_mean: Optional[np.ndarray] = None,
        feat_std: Optional[np.ndarray] = None,
    ):
        self.augment = augment
        self.crop_length = min(crop_length, 2560)
        self.normalize_hc = normalize_hc

        self.snapshots = []
        self.rul_labels = []
        self.hc_features = []
        self.bearing_indices = []

        print("  Extracting handcrafted features...", end=" ")
        for b_idx, bdata in enumerate(bearing_data_list):
            n = bdata["n_snapshots"]
            for t in range(n):
                snap = bdata["snapshots"][t]  # (2560, 2)
                hc = extract_hc_features(snap)
                self.snapshots.append(snap)
                self.rul_labels.append(bdata["rul"][t])
                self.hc_features.append(hc)
                self.bearing_indices.append(b_idx)
        print(f"done ({len(self.rul_labels)} samples)")

        self.snapshots = np.array(self.snapshots, dtype=np.float32)
        self.rul_labels = np.array(self.rul_labels, dtype=np.float32)
        self.hc_features = np.array(self.hc_features, dtype=np.float32)

        # Normalize HC features
        if normalize_hc:
            if feat_mean is None:
                self.feat_mean = self.hc_features.mean(axis=0)
                self.feat_std = self.hc_features.std(axis=0) + 1e-8
            else:
                self.feat_mean = feat_mean
                self.feat_std = feat_std
            self.hc_features = (self.hc_features - self.feat_mean) / self.feat_std

    def __len__(self):
        return len(self.rul_labels)

    def __getitem__(self, idx):
        snap = self.snapshots[idx].copy()
        hc = self.hc_features[idx].copy()

        if self.augment:
            snap = snap * np.random.uniform(0.95, 1.05)

        if self.crop_length < 2560:
            snap = snap[:self.crop_length]

        snap = snap.T  # (2, crop_length)

        return {
            "x": torch.FloatTensor(snap),
            "hc": torch.FloatTensor(hc),
            "rul": torch.FloatTensor([self.rul_labels[idx]]),
            "bearing_idx": self.bearing_indices[idx],
        }


def finetune_jepa_hc(
    model: JEPAHCModel,
    train_loader: torch.utils.data.DataLoader,
    n_epochs: int = 100,
    lr: float = 5e-4,
    device: torch.device = None,
    verbose: bool = True,
) -> list:
    """Fine-tune JEPA+HC model with handcrafted features."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    history = []
    for epoch in range(n_epochs):
        model.train()
        losses = []
        for batch in train_loader:
            x = batch["x"].to(device)
            hc = batch["hc"].to(device)
            rul = batch["rul"].to(device).squeeze()

            pred = model.predict_rul(x, hc)
            loss = criterion(pred, rul)

            if torch.isnan(loss):
                continue

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            losses.append(loss.item())

        scheduler.step()
        avg_loss = np.mean(losses) if losses else float("nan")
        history.append({"train_mse": avg_loss})

        if verbose and (epoch % 20 == 0 or epoch == n_epochs - 1):
            print(f"  Epoch {epoch+1:4d}/{n_epochs} | Train MSE={avg_loss:.4f}")

    return history


def evaluate_jepa_hc(
    model: JEPAHCModel,
    test_data: List[Dict],
    device: torch.device,
    batch_size: int = 64,
    crop_length: int = 2560,
    train_feat_mean: Optional[np.ndarray] = None,
    train_feat_std: Optional[np.ndarray] = None,
) -> Dict:
    """Evaluate JEPA+HC model on test bearings."""
    results = {}
    model.eval()

    for bdata in test_data:
        ds = FEMTORULDatasetHC(
            [bdata], augment=False, crop_length=crop_length,
            normalize_hc=True,
            feat_mean=train_feat_mean,
            feat_std=train_feat_std,
        )
        loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False)

        all_preds, all_targets = [], []
        with torch.no_grad():
            for batch in loader:
                x = batch["x"].to(device)
                hc = batch["hc"].to(device)
                rul = batch["rul"].squeeze().cpu().numpy()

                pred = model.predict_rul(x, hc).cpu().numpy()
                if pred.ndim == 0:
                    pred = pred.reshape(1)
                if rul.ndim == 0:
                    rul = rul.reshape(1)

                all_preds.append(pred)
                all_targets.append(rul)

        preds = np.concatenate(all_preds)
        targets = np.concatenate(all_targets)
        mse = float(np.mean((preds - targets) ** 2))
        results[bdata["bearing_name"]] = {
            "mse": mse,
            "predictions": preds.tolist(),
            "targets": targets.tolist(),
            "n_snapshots": bdata["n_snapshots"],
        }

    return results
