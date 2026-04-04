"""
Trivial baselines for cross-check:
1. Random Forest on handcrafted features (RMS, kurtosis, crest factor, spectral entropy, band energies)
2. XGBoost on same features
3. Linear probe on random-init encoder (3 seeds)

Evaluated on:
- CWRU in-domain F1
- Paderborn transfer F1 (features trained on CWRU, applied to Paderborn without retraining)

Results saved to mechanical-jepa/results/trivial_baselines.json
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from scipy.stats import entropy as scipy_entropy
from pathlib import Path
import sys
import json
import time

sys.path.insert(0, str(Path(__file__).parent))
from paderborn_transfer import create_paderborn_loaders, CLASSES
from src.data import create_dataloaders

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("XGBoost not available, skipping")

CWRU_DIR = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/data/bearings')
PADERBORN_DIR = Path('/home/sagemaker-user/IndustrialJEPA/datasets/data/paderborn')
RESULTS_DIR = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/results')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_features(x_batch, sr=12000):
    """
    Handcrafted vibration features per window.
    x_batch: (N, C, L) numpy array
    Returns: (N, n_features) numpy array
    """
    N, C, L = x_batch.shape
    features = []
    for i in range(N):
        feats = []
        for c in range(C):
            sig = x_batch[i, c].astype(np.float64)
            # Time-domain
            rms = np.sqrt(np.mean(sig ** 2))
            peak = np.max(np.abs(sig))
            crest = peak / (rms + 1e-10)
            mu2 = np.mean(sig ** 2)
            mu4 = np.mean(sig ** 4)
            kurtosis = mu4 / (mu2 ** 2 + 1e-10)
            std = np.std(sig)
            skewness = np.mean((sig - np.mean(sig)) ** 3) / (std ** 3 + 1e-10)
            shape_factor = rms / (np.mean(np.abs(sig)) + 1e-10)
            impulse_factor = peak / (np.mean(np.abs(sig)) + 1e-10)
            # Frequency-domain
            fft_mag = np.abs(np.fft.rfft(sig))
            fft_freqs = np.fft.rfftfreq(L, d=1.0 / sr)
            # Spectral entropy
            psd = fft_mag ** 2
            psd_norm = psd / (psd.sum() + 1e-10)
            spec_entropy = scipy_entropy(psd_norm + 1e-10)
            # Band energies (4 bands, adaptive to sr)
            band_edges = [0, sr * 0.04, sr * 0.17, sr * 0.33, sr * 0.50]
            band_energies = []
            total_e = psd.sum() + 1e-10
            for lo, hi in zip(band_edges[:-1], band_edges[1:]):
                mask = (fft_freqs >= lo) & (fft_freqs < hi)
                band_energies.append(psd[mask].sum() / total_e)
            # Spectral centroid and spread
            spec_centroid = np.sum(fft_freqs * psd) / (psd.sum() + 1e-10)
            spec_spread = np.sqrt(np.sum(((fft_freqs - spec_centroid) ** 2) * psd) / (psd.sum() + 1e-10))
            # Collect
            feats.extend([rms, peak, crest, kurtosis, std, skewness,
                          shape_factor, impulse_factor, spec_entropy,
                          spec_centroid, spec_spread] + band_energies)
        features.append(feats)
    return np.array(features, dtype=np.float32)


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_cwru(seed, batch_size=512):
    train_loader, test_loader, _ = create_dataloaders(
        data_dir=CWRU_DIR, batch_size=batch_size, window_size=4096, stride=2048,
        test_ratio=0.2, seed=seed, num_workers=0, n_channels=3, dataset_filter='cwru'
    )
    X_tr, y_tr, X_te, y_te = [], [], [], []
    for batch in train_loader:
        X_tr.append(batch[0].numpy()); y_tr.append(batch[1].numpy())
    for batch in test_loader:
        X_te.append(batch[0].numpy()); y_te.append(batch[1].numpy())
    return (np.concatenate(X_tr), np.concatenate(y_tr),
            np.concatenate(X_te), np.concatenate(y_te))


def load_paderborn(seed, batch_size=512):
    bearing_dirs = [(str(PADERBORN_DIR / folder), label)
                    for folder, label in CLASSES.items()
                    if (PADERBORN_DIR / folder).exists()]
    _, test_loader = create_paderborn_loaders(
        bearing_dirs=bearing_dirs, window_size=4096, stride=2048,
        target_sr=20000, n_channels=3, test_ratio=0.2,
        batch_size=batch_size, seed=seed, max_files_per_bearing=20,
    )
    X_te, y_te = [], []
    for batch in test_loader:
        X_te.append(batch[0].numpy())
        y_te.append(batch[1].numpy() if isinstance(batch[1], torch.Tensor)
                    else np.array(batch[1]))
    return np.concatenate(X_te), np.concatenate(y_te)


# ---------------------------------------------------------------------------
# Random-encoder linear probe
# ---------------------------------------------------------------------------

def build_random_encoder():
    """Same architecture as JEPA V2 encoder but randomly initialized."""
    sys.path.insert(0, str(Path(__file__).parent / 'src'))
    from models.jepa_v2 import JEPAEncoder  # may not exist — fall back
    raise ImportError("Use inline definition")


class PatchEmbedding(nn.Module):
    def __init__(self, patch_size=256, d_model=512, n_channels=3):
        super().__init__()
        self.proj = nn.Linear(patch_size * n_channels, d_model)
        self.patch_size = patch_size
        self.n_channels = n_channels

    def forward(self, x):
        B, C, L = x.shape
        P = self.patch_size
        n_patches = L // P
        x = x[:, :, :n_patches * P].reshape(B, C, n_patches, P)
        x = x.permute(0, 2, 1, 3).reshape(B, n_patches, C * P)
        return self.proj(x)


class RandomEncoder(nn.Module):
    def __init__(self, patch_size=256, d_model=512, n_heads=4, n_layers=4, n_channels=3):
        super().__init__()
        self.embed = PatchEmbedding(patch_size, d_model, n_channels)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            batch_first=True, dropout=0.0, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        # Sinusoidal PE
        max_len = 32
        pos = torch.arange(max_len).unsqueeze(1).float()
        i = torch.arange(0, d_model, 2).float()
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(pos / 10000 ** (i / d_model))
        pe[:, 1::2] = torch.cos(pos / 10000 ** (i / d_model))
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        t = self.embed(x)
        t = t + self.pe[:, :t.shape[1]]
        t = self.transformer(t)
        return t.mean(dim=1)


@torch.no_grad()
def extract_encoder_features(encoder, x_batch, batch_size=256):
    encoder.eval()
    feats = []
    for i in range(0, len(x_batch), batch_size):
        xb = torch.tensor(x_batch[i:i + batch_size], dtype=torch.float32).to(DEVICE)
        feats.append(encoder(xb).cpu().numpy())
    return np.concatenate(feats)


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def run_classifier(X_tr, y_tr, X_te, y_te, X_pad, y_pad, clf, name, seed):
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)
    X_pad_s = scaler.transform(X_pad)

    clf.fit(X_tr_s, y_tr)
    cwru_f1 = f1_score(y_te, clf.predict(X_te_s), average='macro')
    pad_f1 = f1_score(y_pad, clf.predict(X_pad_s), average='macro',
                      labels=np.unique(y_pad))
    print(f"  [{name} seed={seed}] CWRU F1={cwru_f1:.3f}, Pad transfer F1={pad_f1:.3f}")
    return cwru_f1, pad_f1


def main():
    seeds = [42, 123, 456]
    results = {
        'random_forest': [],
        'xgboost': [],
        'random_encoder': [],
        '_meta': {
            'description': 'Trivial baselines: RF, XGBoost, random-init encoder',
            'features': 'RMS, peak, crest, kurtosis, std, skewness, shape_factor, '
                        'impulse_factor, spectral_entropy, spectral_centroid, '
                        'spectral_spread, 4 band energies (per channel)',
            'transfer': 'CWRU-trained features/model evaluated on Paderborn (no retraining)',
        }
    }

    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"Seed {seed}")
        print('='*60)

        # Load CWRU
        print("Loading CWRU...")
        X_cwru_tr, y_cwru_tr, X_cwru_te, y_cwru_te = load_cwru(seed)
        print(f"  CWRU train: {X_cwru_tr.shape}, test: {X_cwru_te.shape}")

        # Load Paderborn (test set only for transfer eval)
        print("Loading Paderborn...")
        X_pad_te, y_pad_te = load_paderborn(seed)
        print(f"  Paderborn test: {X_pad_te.shape}")

        # --- Handcrafted features ---
        print("Extracting features (CWRU 12kHz)...")
        F_cwru_tr = extract_features(X_cwru_tr, sr=12000)
        F_cwru_te = extract_features(X_cwru_te, sr=12000)
        print("Extracting features (Paderborn 20kHz)...")
        F_pad_te = extract_features(X_pad_te, sr=20000)
        print(f"  Feature dim: {F_cwru_tr.shape[1]}")

        # Random Forest
        print("Running Random Forest...")
        rf = RandomForestClassifier(n_estimators=200, max_depth=None,
                                    random_state=seed, n_jobs=-1)
        cwru_f1, pad_f1 = run_classifier(
            F_cwru_tr, y_cwru_tr, F_cwru_te, y_cwru_te,
            F_pad_te, y_pad_te, rf, 'RF', seed
        )
        results['random_forest'].append({'seed': seed, 'cwru_f1': cwru_f1, 'pad_f1': pad_f1})

        # XGBoost
        if HAS_XGB:
            print("Running XGBoost...")
            xgb_clf = xgb.XGBClassifier(
                n_estimators=300, max_depth=6, learning_rate=0.1,
                use_label_encoder=False, eval_metric='mlogloss',
                random_state=seed, verbosity=0, n_jobs=-1
            )
            cwru_f1, pad_f1 = run_classifier(
                F_cwru_tr, y_cwru_tr, F_cwru_te, y_cwru_te,
                F_pad_te, y_pad_te, xgb_clf, 'XGB', seed
            )
            results['xgboost'].append({'seed': seed, 'cwru_f1': cwru_f1, 'pad_f1': pad_f1})

        # Random encoder linear probe
        print("Running random-encoder linear probe...")
        torch.manual_seed(seed)
        enc = RandomEncoder().to(DEVICE)
        # No training — purely random weights
        print("  Extracting random encoder features (CWRU train)...")
        E_cwru_tr = extract_encoder_features(enc, X_cwru_tr)
        print("  Extracting random encoder features (CWRU test)...")
        E_cwru_te = extract_encoder_features(enc, X_cwru_te)
        print("  Extracting random encoder features (Paderborn test)...")
        E_pad_te = extract_encoder_features(enc, X_pad_te)

        lr_clf = LogisticRegression(C=1.0, max_iter=1000, random_state=seed)
        cwru_f1, pad_f1 = run_classifier(
            E_cwru_tr, y_cwru_tr, E_cwru_te, y_cwru_te,
            E_pad_te, y_pad_te, lr_clf, 'RandEnc', seed
        )
        results['random_encoder'].append({'seed': seed, 'cwru_f1': cwru_f1, 'pad_f1': pad_f1})

    # Summarize
    print("\n\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    method_keys = [k for k in results.keys() if not k.startswith('_')]
    summaries = {}
    for method_name in method_keys:
        rows = results[method_name]
        if not rows:
            continue
        cwru_vals = [r['cwru_f1'] for r in rows]
        pad_vals = [r['pad_f1'] for r in rows]
        print(f"{method_name:25s}  CWRU: {np.mean(cwru_vals):.3f}±{np.std(cwru_vals):.3f}  "
              f"Pad transfer: {np.mean(pad_vals):.3f}±{np.std(pad_vals):.3f}")
        summaries[f'_summary_{method_name}'] = {
            'cwru_mean': float(np.mean(cwru_vals)),
            'cwru_std': float(np.std(cwru_vals)),
            'pad_mean': float(np.mean(pad_vals)),
            'pad_std': float(np.std(pad_vals)),
        }
    results.update(summaries)

    print("\nContext:")
    print("  JEPA V2           CWRU: 0.773±0.018  Pad transfer: 0.900±0.008")
    print("  CNN Supervised    CWRU: 1.000±0.000  Pad transfer: 0.987±0.005")
    print("  Random Init       CWRU: 0.557±0.012  Pad transfer: 0.529±0.024")

    out = RESULTS_DIR / 'trivial_baselines.json'
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out}")
    return results


if __name__ == '__main__':
    main()
