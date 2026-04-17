"""
Comprehensive Baseline Comparison for Mechanical-JEPA.

Round 4 (MOST IMPORTANT): Prove JEPA adds value vs simpler approaches.

Baselines implemented:
1. Hand-crafted features + Logistic Regression (zero deep learning)
2. CNN Supervised (~5M params, same scale as JEPA)
3. Transformer Supervised (same arch as JEPA encoder, trained with labels)
4. MAE (reconstruct raw signal patches, not latent — JEPA comparison)

For each baseline:
- CWRU in-domain Macro F1 (3 seeds)
- Paderborn transfer gain (pretrain on CWRU, probe on Paderborn @20kHz)
- Parameter count and supervision level

Usage:
    python baselines_comparison.py --baseline all --seeds 42 123 456
    python baselines_comparison.py --baseline handcrafted
    python baselines_comparison.py --baseline cnn_supervised
    python baselines_comparison.py --baseline transformer_supervised
    python baselines_comparison.py --baseline mae
"""

import argparse
import sys
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.signal import resample_poly
from math import gcd
from sklearn.metrics import f1_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / '.env')
except ImportError:
    pass

import wandb

sys.path.insert(0, str(Path(__file__).parent))
from src.data import create_dataloaders
from src.models import MechanicalJEPAV2

CHECKPOINT_DIR = Path('/mnt/sagemaker-nvme/jepa_checkpoints')
CHECKPOINT_DIR.mkdir(exist_ok=True, parents=True)

CLASS_NAMES = ['healthy', 'outer_race', 'inner_race', 'ball']


# =============================================================================
# HAND-CRAFTED FEATURES
# =============================================================================

def extract_handcrafted_features(signals):
    """
    Extract hand-crafted vibration features from raw signals.

    Features per channel:
    - RMS: root mean square (energy)
    - Kurtosis: impulse indicator (fault sensitivity)
    - Crest factor: peak/RMS ratio
    - Spectral entropy: frequency distribution uniformity
    - Band energies: low/mid/high frequency bands

    Args:
        signals: (N, C, L) numpy array

    Returns:
        features: (N, F) numpy array
    """
    N, C, L = signals.shape
    features = []

    for i in range(N):
        sig_features = []
        for c in range(C):
            x = signals[i, c]  # (L,)

            # Time domain
            rms = np.sqrt(np.mean(x ** 2))
            mean_abs = np.mean(np.abs(x))
            peak = np.max(np.abs(x))
            kurtosis = np.mean(x ** 4) / (np.mean(x ** 2) ** 2 + 1e-10)
            crest_factor = peak / (rms + 1e-10)
            shape_factor = rms / (mean_abs + 1e-10)
            impulse_factor = peak / (mean_abs + 1e-10)

            # Frequency domain (FFT)
            fft_mag = np.abs(np.fft.rfft(x)) / L
            freqs = np.fft.rfftfreq(L)  # Normalized [0, 0.5]

            # Spectral entropy
            fft_prob = fft_mag / (fft_mag.sum() + 1e-10)
            spectral_entropy = -np.sum(fft_prob * np.log(fft_prob + 1e-10))

            # Spectral centroid
            spectral_centroid = np.sum(freqs * fft_mag) / (fft_mag.sum() + 1e-10)

            # Band energies (5 bands)
            n_fft = len(fft_mag)
            band_size = n_fft // 5
            band_energies = [
                np.sum(fft_mag[j*band_size:(j+1)*band_size] ** 2)
                for j in range(5)
            ]

            # Peak frequency
            peak_freq = freqs[np.argmax(fft_mag)]

            sig_features.extend([
                rms, kurtosis, crest_factor, shape_factor, impulse_factor,
                spectral_entropy, spectral_centroid, peak_freq,
                *band_energies
            ])

        features.append(sig_features)

    return np.array(features)


def baseline_handcrafted(seeds, device, paderborn_data=None):
    """
    Baseline 1: Hand-crafted features + Logistic Regression.
    """
    print("\n" + "=" * 60)
    print("BASELINE 1: HAND-CRAFTED FEATURES + LOGISTIC REGRESSION")
    print("=" * 60)

    results = []

    for seed in seeds:
        print(f"\n  Seed {seed}:")
        torch.manual_seed(seed)
        np.random.seed(seed)

        train_loader, test_loader, data_info = create_dataloaders(
            data_dir='data/bearings',
            batch_size=256,
            window_size=4096,
            stride=2048,
            test_ratio=0.2,
            seed=seed,
            num_workers=0,
            dataset_filter='cwru',
            n_channels=3,
        )

        # Collect raw signals
        def collect_signals(loader):
            all_signals, all_labels = [], []
            for signals, labels, _ in loader:
                all_signals.append(signals.numpy())
                all_labels.append(labels.numpy())
            return np.concatenate(all_signals, axis=0), np.concatenate(all_labels, axis=0)

        print("    Extracting features...")
        t0 = time.time()
        train_signals, train_labels = collect_signals(train_loader)
        test_signals, test_labels = collect_signals(test_loader)

        train_feats = extract_handcrafted_features(train_signals)
        test_feats = extract_handcrafted_features(test_signals)

        # Normalize
        scaler = StandardScaler()
        train_feats = scaler.fit_transform(train_feats)
        test_feats = scaler.transform(test_feats)

        t1 = time.time()
        print(f"    Feature extraction: {t1-t0:.1f}s, shape: {train_feats.shape}")

        # Logistic regression
        clf = LogisticRegression(max_iter=1000, random_state=seed, C=1.0, multi_class='multinomial')
        clf.fit(train_feats, train_labels)
        test_preds = clf.predict(test_feats)

        f1 = f1_score(test_labels, test_preds, average='macro', zero_division=0)
        acc = accuracy_score(test_labels, test_preds)

        print(f"    CWRU Macro F1: {f1:.4f} (acc: {acc:.4f})")

        # Paderborn transfer (if data available)
        pad_f1 = None
        if paderborn_data is not None:
            pad_train_signals, pad_train_labels, pad_test_signals, pad_test_labels = paderborn_data
            pad_train_feats = extract_handcrafted_features(pad_train_signals)
            pad_test_feats = extract_handcrafted_features(pad_test_signals)

            # Use scaler from CWRU training
            pad_train_feats_norm = scaler.transform(pad_train_feats)
            pad_test_feats_norm = scaler.transform(pad_test_feats)

            pad_clf = LogisticRegression(max_iter=1000, random_state=seed, C=1.0)
            pad_clf.fit(pad_train_feats_norm, pad_train_labels)
            pad_preds = pad_clf.predict(pad_test_feats_norm)
            pad_f1 = f1_score(pad_test_labels, pad_preds, average='macro', zero_division=0)
            print(f"    Paderborn transfer F1: {pad_f1:.4f}")

        results.append({
            'seed': seed,
            'cwru_f1': f1,
            'cwru_acc': acc,
            'paderborn_f1': pad_f1,
            'n_features': train_feats.shape[1],
            'n_params': 0,
        })

    cwru_f1s = [r['cwru_f1'] for r in results]
    print(f"\n  CWRU Macro F1: {np.mean(cwru_f1s):.4f} ± {np.std(cwru_f1s):.4f}")
    print(f"  Params: 0 (no neural network)")
    print(f"  Supervision: Full (supervised logistic regression)")

    return results


# =============================================================================
# CNN SUPERVISED BASELINE (~5M params)
# =============================================================================

class CNN1DSupervised(nn.Module):
    """
    1D CNN for supervised vibration fault classification.
    ~5M parameters to match JEPA scale.

    Architecture:
    - 4 convolutional blocks (Conv1D + BN + ReLU + Pool)
    - Global average pooling
    - Linear classification head
    """

    def __init__(self, n_channels=3, n_classes=4, window_size=4096):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1: 3 -> 64
            nn.Conv1d(n_channels, 64, kernel_size=7, padding=3, stride=2),  # -> L/2
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),  # -> L/4

            # Block 2: 64 -> 128
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),  # -> L/8

            # Block 3: 128 -> 256
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2),  # -> L/16

            # Block 4: 256 -> 512
            nn.Conv1d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
        )

        # Global average pooling -> (B, 512)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(512, n_classes)

    def get_embeddings(self, x):
        """Get 512-dim embeddings (for transfer evaluation)."""
        feat = self.features(x)
        feat = self.global_pool(feat).squeeze(-1)
        return feat

    def forward(self, x):
        feat = self.get_embeddings(x)
        return self.classifier(feat)


def train_supervised_model(model, train_loader, test_loader, epochs, lr, device,
                            use_wandb=False, run_name='supervised'):
    """Train a supervised model with cross-entropy loss."""
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss()

    best_f1 = 0.0
    best_state = None

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for signals, labels, _ in train_loader:
            signals, labels = signals.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(signals)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        # Evaluate
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for signals, labels, _ in test_loader:
                signals = signals.to(device)
                preds = model(signals).argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.numpy())

        f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"    Epoch {epoch+1}: loss={total_loss/len(train_loader):.4f}, test_f1={f1:.4f}")

        if use_wandb:
            wandb.log({f'{run_name}/epoch': epoch+1, f'{run_name}/f1': f1,
                       f'{run_name}/loss': total_loss/len(train_loader)})

    model.load_state_dict(best_state)
    return best_f1


def baseline_cnn_supervised(seeds, device, paderborn_loader=None):
    """
    Baseline 2: Supervised CNN (~5M params).
    """
    print("\n" + "=" * 60)
    print("BASELINE 2: CNN SUPERVISED (~5M PARAMS)")
    print("=" * 60)

    results = []

    for seed in seeds:
        print(f"\n  Seed {seed}:")
        torch.manual_seed(seed)
        np.random.seed(seed)

        train_loader, test_loader, data_info = create_dataloaders(
            data_dir='data/bearings',
            batch_size=32,
            window_size=4096,
            stride=2048,
            test_ratio=0.2,
            seed=seed,
            num_workers=0,
            dataset_filter='cwru',
            n_channels=3,
        )

        model = CNN1DSupervised(n_channels=3, n_classes=4).to(device)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"    Params: {n_params:,}")

        best_f1 = train_supervised_model(
            model, train_loader, test_loader,
            epochs=100, lr=1e-4, device=device, run_name=f'cnn_sup_s{seed}'
        )
        print(f"    CWRU Macro F1: {best_f1:.4f}")

        # Transfer: freeze encoder, linear probe on Paderborn
        pad_f1 = None
        pad_rand_f1 = None
        if paderborn_loader is not None:
            pad_train_loader, pad_test_loader = paderborn_loader

            def extract_cnn_embeddings(cnn_model, loader):
                cnn_model.eval()
                all_embeds, all_labels = [], []
                with torch.no_grad():
                    for signals, labels, _ in loader:
                        signals = signals.to(device)
                        embeds = cnn_model.get_embeddings(signals)
                        all_embeds.append(embeds.cpu())
                        all_labels.append(labels.cpu())
                return torch.cat(all_embeds), torch.cat(all_labels)

            pad_train_embeds, pad_train_labels = extract_cnn_embeddings(model, pad_train_loader)
            pad_test_embeds, pad_test_labels = extract_cnn_embeddings(model, pad_test_loader)

            # Linear probe on Paderborn
            pad_probe = nn.Linear(512, 3).to(device)
            pad_opt = optim.Adam(pad_probe.parameters(), lr=1e-3)
            pad_crit = nn.CrossEntropyLoss()

            te = pad_train_embeds.to(device)
            tl = pad_train_labels.to(device)

            best_pad_f1 = 0.0
            for ep in range(50):
                pad_probe.train()
                pad_opt.zero_grad()
                logits = pad_probe(te)
                loss = pad_crit(logits, tl)
                loss.backward()
                pad_opt.step()

                pad_probe.eval()
                with torch.no_grad():
                    preds = pad_probe(pad_test_embeds.to(device)).argmax(dim=1).cpu().numpy()
                f1_p = f1_score(pad_test_labels.numpy(), preds, average='macro', zero_division=0)
                if f1_p > best_pad_f1:
                    best_pad_f1 = f1_p

            # Random init CNN transfer
            torch.manual_seed(seed + 10000)
            rand_cnn = CNN1DSupervised(n_channels=3, n_classes=4).to(device)
            pad_rand_train_embeds, _ = extract_cnn_embeddings(rand_cnn, pad_train_loader)
            pad_rand_test_embeds, _ = extract_cnn_embeddings(rand_cnn, pad_test_loader)

            rand_pad_probe = nn.Linear(512, 3).to(device)
            rand_pad_opt = optim.Adam(rand_pad_probe.parameters(), lr=1e-3)

            best_rand_pad_f1 = 0.0
            for ep in range(50):
                rand_pad_probe.train()
                rand_pad_opt.zero_grad()
                logits = rand_pad_probe(pad_rand_train_embeds.to(device))
                loss = pad_crit(logits, pad_train_labels.to(device))
                loss.backward()
                rand_pad_opt.step()

                rand_pad_probe.eval()
                with torch.no_grad():
                    preds = rand_pad_probe(pad_rand_test_embeds.to(device)).argmax(dim=1).cpu().numpy()
                f1_r = f1_score(pad_test_labels.numpy(), preds, average='macro', zero_division=0)
                if f1_r > best_rand_pad_f1:
                    best_rand_pad_f1 = f1_r

            pad_f1 = best_pad_f1
            pad_rand_f1 = best_rand_pad_f1
            print(f"    Paderborn transfer: JEPA={pad_f1:.4f}, Rand={pad_rand_f1:.4f}, gain={pad_f1-pad_rand_f1:+.4f}")

        results.append({
            'seed': seed,
            'cwru_f1': best_f1,
            'paderborn_f1': pad_f1,
            'paderborn_rand_f1': pad_rand_f1,
            'n_params': n_params,
        })

    cwru_f1s = [r['cwru_f1'] for r in results]
    print(f"\n  CWRU Macro F1: {np.mean(cwru_f1s):.4f} ± {np.std(cwru_f1s):.4f}")
    print(f"  Supervision: Full supervised (cross-entropy on labels)")

    return results


# =============================================================================
# TRANSFORMER SUPERVISED BASELINE
# =============================================================================

class TransformerSupervised(nn.Module):
    """
    Transformer supervised: same architecture as JEPA encoder, but trained
    with cross-entropy loss on fault labels.

    This answers: "Is self-supervised pretraining better than supervised
    pretraining for transfer?"
    """

    def __init__(self, n_channels=3, n_classes=4, window_size=4096, patch_size=256,
                 embed_dim=512, depth=4, n_heads=4):
        super().__init__()
        from src.models.jepa import JEPAEncoder
        self.encoder = JEPAEncoder(
            n_channels=n_channels,
            window_size=window_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=depth,
            n_heads=n_heads,
        )
        self.n_patches = window_size // patch_size
        self.embed_dim = embed_dim
        self.classifier = nn.Linear(embed_dim, n_classes)

    def get_embeddings(self, x):
        """Mean-pool patch tokens (same as JEPA evaluation protocol)."""
        all_tokens = self.encoder(x, return_all_tokens=True)
        return all_tokens[:, 1:].mean(dim=1)  # mean over patch tokens

    def forward(self, x):
        embeds = self.get_embeddings(x)
        return self.classifier(embeds)


def baseline_transformer_supervised(seeds, device, paderborn_loader=None):
    """
    Baseline 3: Transformer trained supervised on CWRU (same arch as JEPA encoder).
    """
    print("\n" + "=" * 60)
    print("BASELINE 3: TRANSFORMER SUPERVISED (SAME ARCH AS JEPA ENCODER)")
    print("=" * 60)

    results = []

    for seed in seeds:
        print(f"\n  Seed {seed}:")
        torch.manual_seed(seed)
        np.random.seed(seed)

        train_loader, test_loader, data_info = create_dataloaders(
            data_dir='data/bearings',
            batch_size=32,
            window_size=4096,
            stride=2048,
            test_ratio=0.2,
            seed=seed,
            num_workers=0,
            dataset_filter='cwru',
            n_channels=3,
        )

        model = TransformerSupervised(n_channels=3, n_classes=4).to(device)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"    Params: {n_params:,}")

        best_f1 = train_supervised_model(
            model, train_loader, test_loader,
            epochs=100, lr=1e-4, device=device, run_name=f'transformer_sup_s{seed}'
        )
        print(f"    CWRU Macro F1: {best_f1:.4f}")

        # Transfer to Paderborn
        pad_f1 = None
        pad_gain = None
        if paderborn_loader is not None:
            pad_train_loader, pad_test_loader = paderborn_loader

            def extract_transformer_embeds(m, loader):
                m.eval()
                all_embeds, all_labels = [], []
                with torch.no_grad():
                    for signals, labels, _ in loader:
                        signals = signals.to(device)
                        embeds = m.get_embeddings(signals)
                        all_embeds.append(embeds.cpu())
                        all_labels.append(labels.cpu())
                return torch.cat(all_embeds), torch.cat(all_labels)

            pad_train_embeds, pad_train_labels = extract_transformer_embeds(model, pad_train_loader)
            pad_test_embeds, pad_test_labels = extract_transformer_embeds(model, pad_test_loader)

            # Linear probe
            probe = nn.Linear(512, 3).to(device)
            opt = optim.Adam(probe.parameters(), lr=1e-3)
            crit = nn.CrossEntropyLoss()
            best_pad_f1 = 0.0

            for ep in range(50):
                probe.train()
                opt.zero_grad()
                logits = probe(pad_train_embeds.to(device))
                loss = crit(logits, pad_train_labels.to(device))
                loss.backward()
                opt.step()

                probe.eval()
                with torch.no_grad():
                    preds = probe(pad_test_embeds.to(device)).argmax(dim=1).cpu().numpy()
                f1_p = f1_score(pad_test_labels.numpy(), preds, average='macro', zero_division=0)
                if f1_p > best_pad_f1:
                    best_pad_f1 = f1_p

            # Random init
            torch.manual_seed(seed + 10000)
            rand_model = TransformerSupervised(n_channels=3, n_classes=4).to(device)
            rand_embeds_train, _ = extract_transformer_embeds(rand_model, pad_train_loader)
            rand_embeds_test, _ = extract_transformer_embeds(rand_model, pad_test_loader)

            rand_probe = nn.Linear(512, 3).to(device)
            rand_opt = optim.Adam(rand_probe.parameters(), lr=1e-3)
            best_rand_f1 = 0.0

            for ep in range(50):
                rand_probe.train()
                rand_opt.zero_grad()
                logits = rand_probe(rand_embeds_train.to(device))
                loss = crit(logits, pad_train_labels.to(device))
                loss.backward()
                rand_opt.step()

                rand_probe.eval()
                with torch.no_grad():
                    preds = rand_probe(rand_embeds_test.to(device)).argmax(dim=1).cpu().numpy()
                f1_r = f1_score(pad_test_labels.numpy(), preds, average='macro', zero_division=0)
                if f1_r > best_rand_f1:
                    best_rand_f1 = f1_r

            pad_f1 = best_pad_f1
            pad_gain = best_pad_f1 - best_rand_f1
            print(f"    Paderborn transfer: F1={pad_f1:.4f}, gain={pad_gain:+.4f}")

        results.append({
            'seed': seed,
            'cwru_f1': best_f1,
            'paderborn_f1': pad_f1,
            'paderborn_gain': pad_gain,
            'n_params': n_params,
        })

    cwru_f1s = [r['cwru_f1'] for r in results]
    print(f"\n  CWRU Macro F1: {np.mean(cwru_f1s):.4f} ± {np.std(cwru_f1s):.4f}")
    print(f"  Supervision: Full supervised (same arch as JEPA)")

    return results


# =============================================================================
# MAE BASELINE (reconstruct raw signal, not latent)
# =============================================================================

class MAEDecoder(nn.Module):
    """Simple decoder to reconstruct raw signal patches from latent embeddings."""

    def __init__(self, embed_dim=512, patch_size=256, n_channels=3, depth=2):
        super().__init__()
        from src.models.jepa import TransformerBlock
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, n_heads=4, mlp_ratio=4.0)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        # Reconstruct raw patch
        self.output_proj = nn.Linear(embed_dim, patch_size * n_channels)
        self.patch_size = patch_size
        self.n_channels = n_channels

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return self.output_proj(x)  # (B, N_mask, patch_size * n_channels)


class MAEModel(nn.Module):
    """
    Masked Autoencoder: reconstructs raw signal patches (not latent).

    Key difference from JEPA: loss is MSE on raw pixels (patches), not L1 on normalized embeddings.
    This answers: "Is predicting in latent space (JEPA) better than reconstructing input (MAE)?"
    """

    def __init__(self, n_channels=3, window_size=4096, patch_size=256,
                 embed_dim=512, encoder_depth=4, decoder_depth=2, n_heads=4, mask_ratio=0.625):
        super().__init__()
        from src.models.jepa import JEPAEncoder
        import math

        self.mask_ratio = mask_ratio
        self.n_patches = window_size // patch_size
        self.patch_size = patch_size
        self.n_channels = n_channels

        self.encoder = JEPAEncoder(
            n_channels=n_channels,
            window_size=window_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=encoder_depth,
            n_heads=n_heads,
        )

        # Mask token for decoder
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        # Position embeddings for decoder (simple sinusoidal)
        pe = torch.zeros(self.n_patches, embed_dim)
        position = torch.arange(0, self.n_patches, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2, dtype=torch.float) * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pos_embed', pe.unsqueeze(0))  # (1, N, D)

        self.decoder = MAEDecoder(embed_dim=embed_dim, patch_size=patch_size,
                                   n_channels=n_channels, depth=decoder_depth)

    def _generate_mask(self, B, device):
        n_mask = int(self.n_patches * self.mask_ratio)
        n_context = self.n_patches - n_mask
        indices = torch.stack([torch.randperm(self.n_patches, device=device) for _ in range(B)])
        return indices[:, :n_mask], indices[:, n_mask:]

    def _patchify(self, x):
        """Split signal into patches. x: (B, C, L) -> (B, N, C*patch_size)"""
        B, C, L = x.shape
        N = L // self.patch_size
        x = x.reshape(B, C, N, self.patch_size)
        x = x.permute(0, 2, 1, 3)  # (B, N, C, patch_size)
        return x.reshape(B, N, C * self.patch_size)

    def forward(self, x):
        B = x.shape[0]
        device = x.device

        mask_indices, context_indices = self._generate_mask(B, device)

        # Encode context
        context_embeds = self.encoder(x, mask_indices=mask_indices, return_all_tokens=True)[:, 1:]  # (B, N_context, D)

        # Add positional embeddings to context
        ctx_pos = torch.gather(
            self.pos_embed.expand(B, -1, -1), 1,
            context_indices.unsqueeze(-1).expand(-1, -1, context_embeds.shape[-1])
        )
        context_embeds = context_embeds + ctx_pos

        # Mask tokens with positional embeddings
        mask_tokens = self.mask_token.expand(B, mask_indices.shape[1], -1).clone()
        mask_pos = torch.gather(
            self.pos_embed.expand(B, -1, -1), 1,
            mask_indices.unsqueeze(-1).expand(-1, -1, context_embeds.shape[-1])
        )
        mask_tokens = mask_tokens + mask_pos

        # Full sequence for decoder
        full_seq = torch.cat([context_embeds, mask_tokens], dim=1)  # (B, N, D)
        reconstructed = self.decoder(full_seq)  # (B, N, patch_size * C)

        # Only compute loss on masked patches (last n_mask tokens)
        n_mask = mask_indices.shape[1]
        recon_mask = reconstructed[:, context_indices.shape[1]:]  # (B, n_mask, patch_size*C)

        # Target: raw signal patches
        patches = self._patchify(x)  # (B, N, C*patch_size)
        target_patches = torch.gather(
            patches, 1,
            mask_indices.unsqueeze(-1).expand(-1, -1, patches.shape[-1])
        )  # (B, n_mask, C*patch_size)

        # Normalize target patches (per-patch normalization, like in MAE)
        mean = target_patches.mean(dim=-1, keepdim=True)
        std = target_patches.std(dim=-1, keepdim=True) + 1e-6
        target_normalized = (target_patches - mean) / std

        loss = F.mse_loss(recon_mask, target_normalized)
        return loss

    def get_embeddings(self, x, pool='mean'):
        all_tokens = self.encoder(x, return_all_tokens=True)
        return all_tokens[:, 1:].mean(dim=1)


def baseline_mae(seeds, device, paderborn_loader=None, epochs=100):
    """
    Baseline 4: MAE (reconstruct raw signal patches).
    """
    print("\n" + "=" * 60)
    print("BASELINE 4: MAE (RECONSTRUCT RAW SIGNAL, NOT LATENT)")
    print("=" * 60)
    print("This answers: 'Is predicting in latent space (JEPA) better than reconstructing input (MAE)?'")

    results = []

    for seed in seeds:
        print(f"\n  Seed {seed}:")
        torch.manual_seed(seed)
        np.random.seed(seed)

        train_loader, test_loader, data_info = create_dataloaders(
            data_dir='data/bearings',
            batch_size=32,
            window_size=4096,
            stride=2048,
            test_ratio=0.2,
            seed=seed,
            num_workers=0,
            dataset_filter='cwru',
            n_channels=3,
        )

        model = MAEModel(n_channels=3, window_size=4096, patch_size=256,
                          embed_dim=512, encoder_depth=4, mask_ratio=0.625).to(device)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"    Params: {n_params:,}")

        # Train MAE (self-supervised)
        optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

        print(f"    Training MAE for {epochs} epochs...")
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            for signals, labels, _ in train_loader:
                signals = signals.to(device)
                optimizer.zero_grad()
                loss = model(signals)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += loss.item()
            scheduler.step()

            if (epoch + 1) % 20 == 0 or epoch == 0:
                print(f"    Epoch {epoch+1}: loss={total_loss/len(train_loader):.4f}")

        # Evaluate linear probe
        model.eval()
        all_embeds, all_labels = [], []
        with torch.no_grad():
            for signals, labels, _ in train_loader:
                embeds = model.get_embeddings(signals.to(device))
                all_embeds.append(embeds.cpu())
                all_labels.append(labels)
        train_embeds = torch.cat(all_embeds)
        train_labels = torch.cat(all_labels)

        all_embeds, all_labels = [], []
        with torch.no_grad():
            for signals, labels, _ in test_loader:
                embeds = model.get_embeddings(signals.to(device))
                all_embeds.append(embeds.cpu())
                all_labels.append(labels)
        test_embeds = torch.cat(all_embeds)
        test_labels = torch.cat(all_labels)

        # Linear probe
        probe = nn.Linear(512, 4).to(device)
        probe_opt = optim.Adam(probe.parameters(), lr=1e-3)
        crit = nn.CrossEntropyLoss()
        best_f1 = 0.0

        te = train_embeds.to(device)
        tl = train_labels.to(device)

        for ep in range(20):
            probe.train()
            probe_opt.zero_grad()
            logits = probe(te)
            loss = crit(logits, tl)
            loss.backward()
            probe_opt.step()

            probe.eval()
            with torch.no_grad():
                preds = probe(test_embeds.to(device)).argmax(dim=1).cpu().numpy()
            f1 = f1_score(test_labels.numpy(), preds, average='macro', zero_division=0)
            if f1 > best_f1:
                best_f1 = f1

        print(f"    CWRU Macro F1: {best_f1:.4f}")

        # Paderborn transfer
        pad_f1 = None
        if paderborn_loader is not None:
            pad_train_loader, pad_test_loader = paderborn_loader

            def get_mae_embeds(m, loader):
                m.eval()
                embeds_list, labels_list = [], []
                with torch.no_grad():
                    for signals, labels, _ in loader:
                        embeds = m.get_embeddings(signals.to(device))
                        embeds_list.append(embeds.cpu())
                        labels_list.append(labels)
                return torch.cat(embeds_list), torch.cat(labels_list)

            pad_tr_emb, pad_tr_lbl = get_mae_embeds(model, pad_train_loader)
            pad_te_emb, pad_te_lbl = get_mae_embeds(model, pad_test_loader)

            probe_pad = nn.Linear(512, 3).to(device)
            opt_pad = optim.Adam(probe_pad.parameters(), lr=1e-3)
            best_pad_f1 = 0.0

            for ep in range(50):
                probe_pad.train()
                opt_pad.zero_grad()
                logits = probe_pad(pad_tr_emb.to(device))
                loss = crit(logits, pad_tr_lbl.to(device))
                loss.backward()
                opt_pad.step()

                probe_pad.eval()
                with torch.no_grad():
                    preds = probe_pad(pad_te_emb.to(device)).argmax(dim=1).cpu().numpy()
                f1_p = f1_score(pad_te_lbl.numpy(), preds, average='macro', zero_division=0)
                if f1_p > best_pad_f1:
                    best_pad_f1 = f1_p

            # Random init
            torch.manual_seed(seed + 10000)
            rand_mae = MAEModel(n_channels=3, window_size=4096, patch_size=256, embed_dim=512).to(device)
            rand_tr_emb, _ = get_mae_embeds(rand_mae, pad_train_loader)
            rand_te_emb, _ = get_mae_embeds(rand_mae, pad_test_loader)

            rand_probe = nn.Linear(512, 3).to(device)
            rand_opt = optim.Adam(rand_probe.parameters(), lr=1e-3)
            best_rand_f1 = 0.0

            for ep in range(50):
                rand_probe.train()
                rand_opt.zero_grad()
                logits = rand_probe(rand_tr_emb.to(device))
                loss = crit(logits, pad_tr_lbl.to(device))
                loss.backward()
                rand_opt.step()

                rand_probe.eval()
                with torch.no_grad():
                    preds = rand_probe(rand_te_emb.to(device)).argmax(dim=1).cpu().numpy()
                f1_r = f1_score(pad_te_lbl.numpy(), preds, average='macro', zero_division=0)
                if f1_r > best_rand_f1:
                    best_rand_f1 = f1_r

            pad_f1 = best_pad_f1
            pad_gain = best_pad_f1 - best_rand_f1
            print(f"    Paderborn: MAE={pad_f1:.4f}, rand={best_rand_f1:.4f}, gain={pad_gain:+.4f}")

        results.append({
            'seed': seed,
            'cwru_f1': best_f1,
            'paderborn_f1': pad_f1,
            'n_params': n_params,
        })

    cwru_f1s = [r['cwru_f1'] for r in results]
    print(f"\n  MAE CWRU Macro F1: {np.mean(cwru_f1s):.4f} ± {np.std(cwru_f1s):.4f}")

    return results


# =============================================================================
# PADERBORN DATA LOADING HELPER
# =============================================================================

def load_paderborn_loaders(target_sr=20000, seed=42):
    """
    Load Paderborn data at target sampling rate for transfer experiments.
    Returns (train_loader, test_loader) or None if data not available.
    """
    pad_path = Path('data/bearings')
    pad_files = list(pad_path.glob('*.parquet'))

    # Check for Paderborn data
    try:
        import pandas as pd
        import torch
        from torch.utils.data import TensorDataset, DataLoader, random_split

        # Try loading from local parquet files
        pad_data_found = False
        for f in pad_files:
            if 'paderborn' in f.name.lower() or 'pad' in f.name.lower():
                pad_data_found = True
                break

        if not pad_data_found:
            # Try loading from existing paderborn_transfer.py approach
            try:
                sys.path.insert(0, str(Path(__file__).parent))
                from paderborn_transfer import load_paderborn_data
                return load_paderborn_data(target_sr=target_sr, seed=seed)
            except Exception as e:
                print(f"    Could not load Paderborn data: {e}")
                return None

    except Exception as e:
        print(f"    Paderborn data not available: {e}")
        return None


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', type=str, default='all',
                        choices=['all', 'handcrafted', 'cnn_supervised', 'transformer_supervised', 'mae'])
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 456])
    parser.add_argument('--no-wandb', action='store_true')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--paderborn', action='store_true', help='Include Paderborn transfer eval')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Seeds: {args.seeds}")

    # Check disk
    import shutil
    free_gb = shutil.disk_usage('/home/sagemaker-user').free / 1e9
    print(f"Home disk free: {free_gb:.1f} GB")
    if free_gb < 2.0:
        print("WARNING: Less than 2GB free on home disk!")

    use_wandb = not args.no_wandb
    if use_wandb:
        wandb.init(
            project='mechanical-jepa',
            name=f'baselines_{args.baseline}',
            tags=['baselines', 'round4'],
        )

    # Load Paderborn data if requested
    paderborn_loaders = None
    if args.paderborn:
        print("\nLoading Paderborn data for transfer evaluation...")
        paderborn_loaders = load_paderborn_loaders(target_sr=20000, seed=args.seeds[0])
        if paderborn_loaders is None:
            print("Paderborn data unavailable, skipping transfer metrics")

    all_results = {}

    if args.baseline in ('all', 'handcrafted'):
        results = baseline_handcrafted(args.seeds, device, paderborn_data=None)
        all_results['handcrafted'] = results

        cwru_f1s = [r['cwru_f1'] for r in results]
        if use_wandb:
            wandb.log({
                'handcrafted/cwru_f1_mean': np.mean(cwru_f1s),
                'handcrafted/cwru_f1_std': np.std(cwru_f1s),
            })

    if args.baseline in ('all', 'cnn_supervised'):
        results = baseline_cnn_supervised(args.seeds, device, paderborn_loader=paderborn_loaders)
        all_results['cnn_supervised'] = results

        cwru_f1s = [r['cwru_f1'] for r in results]
        if use_wandb:
            wandb.log({
                'cnn_supervised/cwru_f1_mean': np.mean(cwru_f1s),
                'cnn_supervised/cwru_f1_std': np.std(cwru_f1s),
            })

    if args.baseline in ('all', 'transformer_supervised'):
        results = baseline_transformer_supervised(args.seeds, device, paderborn_loader=paderborn_loaders)
        all_results['transformer_supervised'] = results

        cwru_f1s = [r['cwru_f1'] for r in results]
        if use_wandb:
            wandb.log({
                'transformer_supervised/cwru_f1_mean': np.mean(cwru_f1s),
                'transformer_supervised/cwru_f1_std': np.std(cwru_f1s),
            })

    if args.baseline in ('all', 'mae'):
        results = baseline_mae(args.seeds, device, paderborn_loader=paderborn_loaders, epochs=args.epochs)
        all_results['mae'] = results

        cwru_f1s = [r['cwru_f1'] for r in results]
        if use_wandb:
            wandb.log({
                'mae/cwru_f1_mean': np.mean(cwru_f1s),
                'mae/cwru_f1_std': np.std(cwru_f1s),
            })

    # Print comprehensive summary table
    print("\n" + "=" * 80)
    print("COMPREHENSIVE BASELINE COMPARISON TABLE")
    print("=" * 80)
    print(f"{'Method':<35} {'CWRU F1 (mean±std)':<25} {'Seeds':<10} {'Supervision'}")
    print("-" * 80)

    if 'handcrafted' in all_results:
        r = all_results['handcrafted']
        f1s = [x['cwru_f1'] for x in r]
        print(f"{'Hand-crafted + LogReg':<35} {f'{np.mean(f1s):.3f} ± {np.std(f1s):.3f}':<25} {len(f1s):<10} {'Supervised'}")

    if 'cnn_supervised' in all_results:
        r = all_results['cnn_supervised']
        f1s = [x['cwru_f1'] for x in r]
        print(f"{'CNN Supervised (~5M)':<35} {f'{np.mean(f1s):.3f} ± {np.std(f1s):.3f}':<25} {len(f1s):<10} {'Supervised'}")

    if 'transformer_supervised' in all_results:
        r = all_results['transformer_supervised']
        f1s = [x['cwru_f1'] for x in r]
        print(f"{'Transformer Supervised (~5M)':<35} {f'{np.mean(f1s):.3f} ± {np.std(f1s):.3f}':<25} {len(f1s):<10} {'Supervised'}")

    if 'mae' in all_results:
        r = all_results['mae']
        f1s = [x['cwru_f1'] for x in r]
        print(f"{'MAE (reconstruct signal)':<35} {f'{np.mean(f1s):.3f} ± {np.std(f1s):.3f}':<25} {len(f1s):<10} {'Self-supervised'}")

    print(f"{'JEPA V2 (ours)':<35} {'0.773 ± 0.018':<25} {'3':<10} {'Self-supervised'}")
    print("-" * 80)

    if use_wandb:
        wandb.finish()

    # Save results
    import json
    save_path = Path('results/baselines_comparison.json')
    save_path.parent.mkdir(exist_ok=True, parents=True)

    # Convert to serializable
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        return obj

    with open(save_path, 'w') as f:
        json.dump(
            {k: [{kk: convert(vv) for kk, vv in r.items()} for r in v]
             for k, v in all_results.items()},
            f, indent=2
        )
    print(f"\nResults saved to {save_path}")

    return all_results


if __name__ == '__main__':
    main()
