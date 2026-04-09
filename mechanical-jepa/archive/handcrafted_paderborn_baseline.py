"""
Handcrafted features + classical ML baselines for Paderborn transfer.

This establishes whether handcrafted features (FFT, RMS, statistical moments)
can achieve competitive transfer on Paderborn WITHOUT pretraining.

Baselines:
1. FFT features + Ridge Regression
2. FFT features + Logistic Regression
3. Statistical features (RMS, kurtosis, crest factor) + Ridge
4. Combined (FFT + statistical) + Ridge

Goal: Establish whether the Paderborn task itself is trivial for handcrafted features.
If yes: JEPA's 0.900 F1 is not interesting (trivially solvable).
If no: JEPA's 0.900 F1 is meaningful (better than handcrafted baselines).

Expected: FFT might work well for Paderborn (bearing defect frequencies are clear).
The interesting comparison is: can CWRU-pretrained JEPA beat Paderborn FFT features
that were computed DIRECTLY ON PADERBORN DATA?
"""

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).parent))
from paderborn_transfer import create_paderborn_loaders, CLASSES

PADERBORN_DIR = Path('/home/sagemaker-user/IndustrialJEPA/datasets/data/paderborn')
RESULTS_DIR = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/results')


def extract_features(x_batch):
    """
    Extract handcrafted features from a batch of vibration windows.
    x_batch: (N, 3, L) numpy array
    Returns: (N, n_features) numpy array
    """
    N, C, L = x_batch.shape
    features = []

    for i in range(N):
        feats = []
        for c in range(C):
            sig = x_batch[i, c]

            # Time domain
            rms = np.sqrt(np.mean(sig**2))
            peak = np.max(np.abs(sig))
            crest = peak / (rms + 1e-8)
            kurtosis = np.mean(sig**4) / (np.mean(sig**2)**2 + 1e-8)
            skewness = np.mean(sig**3) / (np.mean(sig**2)**1.5 + 1e-8)

            # Frequency domain (FFT)
            fft = np.abs(np.fft.rfft(sig))
            fft_freqs = np.fft.rfftfreq(L, d=1/20000)  # 20kHz

            # Band energies (20kHz signal, bands in Hz)
            bands = [(0, 500), (500, 2000), (2000, 5000), (5000, 10000)]
            band_energies = []
            for flo, fhi in bands:
                mask = (fft_freqs >= flo) & (fft_freqs < fhi)
                band_energies.append(np.sum(fft[mask]**2) / (np.sum(fft**2) + 1e-8))

            # Top 5 FFT magnitude peaks (normalized)
            top_freqs = np.sort(fft)[-5:][::-1]
            top_freqs_norm = top_freqs / (fft.max() + 1e-8)

            feats.extend([rms, peak, crest, kurtosis, skewness] + band_energies + list(top_freqs_norm))

        features.append(feats)

    return np.array(features, dtype=np.float32)


def load_paderborn_data(seed=42):
    """Load all Paderborn data as numpy arrays."""
    bearing_dirs = []
    for folder, label in CLASSES.items():
        bp = PADERBORN_DIR / folder
        if bp.exists():
            bearing_dirs.append((str(bp), label))

    train_loader, test_loader = create_paderborn_loaders(
        bearing_dirs=bearing_dirs,
        window_size=4096, stride=2048,
        target_sr=20000, n_channels=3, test_ratio=0.2,
        batch_size=256, seed=seed, max_files_per_bearing=20,
    )

    def loader_to_numpy(loader):
        X, y = [], []
        for batch in loader:
            x_batch = batch[0].numpy() if isinstance(batch[0], torch.Tensor) else batch[0]
            y_batch = batch[1].numpy() if isinstance(batch[1], torch.Tensor) else np.array(batch[1])
            X.append(x_batch)
            y.append(y_batch)
        return np.concatenate(X), np.concatenate(y)

    X_train, y_train = loader_to_numpy(train_loader)
    X_test, y_test = loader_to_numpy(test_loader)
    return X_train, y_train, X_test, y_test


def run_handcrafted_baselines(seeds=(42, 123, 456)):
    """Run all handcrafted baselines for given seeds."""
    results = {}

    for seed in seeds:
        print(f"\n=== Seed {seed} ===")
        X_train_raw, y_train, X_test_raw, y_test = load_paderborn_data(seed=seed)
        print(f"Train: {X_train_raw.shape}, Test: {X_test_raw.shape}")
        print(f"Classes: {np.unique(y_train)} (train), {np.unique(y_test)} (test)")

        # Extract features
        print("Extracting features...")
        X_train = extract_features(X_train_raw)
        X_test = extract_features(X_test_raw)
        print(f"Feature dim: {X_train.shape[1]}")

        # Scale
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Baselines
        baselines = {
            'LogReg (C=1)': LogisticRegression(C=1, max_iter=1000, random_state=seed),
            'LogReg (C=10)': LogisticRegression(C=10, max_iter=1000, random_state=seed),
            'RidgeClassifier': RidgeClassifier(alpha=1.0),
        }

        for name, clf in baselines.items():
            clf.fit(X_train_scaled, y_train)
            y_pred = clf.predict(X_test_scaled)
            f1 = f1_score(y_test, y_pred, average='macro')
            print(f"  {name}: F1={f1:.3f}")

            if name not in results:
                results[name] = []
            results[name].append({'seed': seed, 'f1': f1})

    # Summary
    print("\n=== Summary ===")
    for name, seed_results in results.items():
        f1s = [r['f1'] for r in seed_results]
        print(f"{name}: {np.mean(f1s):.3f} +/- {np.std(f1s):.3f}")

    # Save results
    out = RESULTS_DIR / 'handcrafted_paderborn.json'
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out}")

    return results


if __name__ == '__main__':
    results = run_handcrafted_baselines(seeds=(42, 123, 456))

    # Compare to JEPA V2
    print("\n=== Final Comparison ===")
    print(f"{'Method':<30} {'Paderborn F1'}")
    print("-" * 50)
    print(f"{'JEPA V2 (pretrained CWRU)':<30} 0.900 +/- 0.008")
    for name, seed_results in results.items():
        f1s = [r['f1'] for r in seed_results]
        print(f"{name:<30} {np.mean(f1s):.3f} +/- {np.std(f1s):.3f}")
    print(f"{'Random Init':<30} 0.529 +/- 0.024")
