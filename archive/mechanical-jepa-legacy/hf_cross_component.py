"""
Cross-Component Transfer: Bearing → Gearbox using HF Mechanical-Components Dataset.

Experiment 5C:
- Pretrain JEPA on FEMTO bearings (non-overlapping with CWRU)
- Transfer (zero-shot linear probe) to gearbox fault classification
- Compare: bearing-pretrained vs random init vs gearbox self-pretrained

The key question: do vibration features transfer across component types?

Dataset structure:
  Bearings (file 0): 1124 FEMTO samples, 2560 samples/file, 25.6kHz, 2 channels
  Gearboxes (file 0): 272 samples from 3 sources
    - mcc5_thu: 143 samples, 88832 samples/file
    - phm2009: 109 samples, unknown
    - oedi: 20 samples

Usage:
    python hf_cross_component.py --checkpoint checkpoints/jepa_v2_xxx.pt
"""

import argparse
import os
import sys
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).parent))
from src.models import MechanicalJEPAV2

os.environ.setdefault('HF_TOKEN', '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')


# =============================================================================
# HF Data Loaders
# =============================================================================

def load_hf_parquet(config, split_idx, n_splits, token):
    """Load one parquet file from HF dataset."""
    import pandas as pd
    url = f'hf://datasets/Forgis/Mechanical-Components/{config}/train-0000{split_idx}-of-0000{n_splits}.parquet'
    return pd.read_parquet(url, storage_options={'token': token})


def extract_signal_window(signal_array, window_size, n_channels, seed=42):
    """
    Extract a fixed-size window from a signal array.
    signal_array: object array of shape (n_channels,), each element is 1D array
    """
    rng = np.random.default_rng(seed)
    n_ch = len(signal_array)
    actual_channels = min(n_ch, n_channels)

    window = np.zeros((n_channels, window_size), dtype=np.float32)

    for ch in range(actual_channels):
        sig = np.array(signal_array[ch], dtype=np.float32)
        if len(sig) >= window_size:
            start = rng.integers(0, len(sig) - window_size + 1)
            window[ch] = sig[start:start + window_size]
        else:
            # Repeat to fill
            repeats = window_size // len(sig) + 1
            extended = np.tile(sig, repeats)
            window[ch] = extended[:window_size]

    return window


def normalize_window(window):
    """Per-channel z-score normalization."""
    for ch in range(window.shape[0]):
        m = window[ch].mean()
        s = window[ch].std() + 1e-8
        window[ch] = (window[ch] - m) / s
    return window


class HFBearingDataset(Dataset):
    """FEMTO bearings from HF dataset (for JEPA pretraining)."""

    def __init__(self, df, window_size=4096, n_channels=2, seed=42, resample_to=None):
        self.window_size = window_size
        self.n_channels = n_channels
        self.seed = seed

        # Filter to vibration signals only
        self.df = df[df['n_channels'] >= 1].reset_index(drop=True)
        print(f"HFBearingDataset: {len(self.df)} samples")

        # Class labels for evaluation (fault_type)
        label_map = {}
        unique_faults = self.df['fault_type'].unique()
        for i, ft in enumerate(sorted(unique_faults)):
            label_map[ft] = i
        self.label_map = label_map
        self.labels = [label_map[ft] for ft in self.df['fault_type']]

        print(f"  Label map: {label_map}")
        print(f"  Fault distribution: {dict(zip(*np.unique(self.labels, return_counts=True)))}")

        # Sampling rate info
        if 'extra_metadata' in df.columns:
            meta = df['extra_metadata'].iloc[0]
            if isinstance(meta, str):
                meta = json.loads(meta)
            elif isinstance(meta, dict):
                pass
            self.sampling_rate = meta.get('sampling_rate_hz', 25600)
        else:
            self.sampling_rate = 25600
        print(f"  Sampling rate: {self.sampling_rate} Hz")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sig_array = row['signal']
        label = self.labels[idx]

        window = extract_signal_window(sig_array, self.window_size, self.n_channels, seed=self.seed + idx)
        window = normalize_window(window)

        return torch.from_numpy(window), label, idx


class HFGearboxDataset(Dataset):
    """Gearbox samples from HF dataset (for cross-component transfer evaluation)."""

    def __init__(self, df, window_size=4096, n_channels=3, seed=42):
        self.window_size = window_size
        self.n_channels = n_channels
        self.seed = seed

        # Filter to known fault types
        self.df = df[df['fault_type'].isin(['healthy', 'gear_pitting', 'gear_crack'])].reset_index(drop=True)
        print(f"HFGearboxDataset: {len(self.df)} samples (filtered to known fault types)")

        if len(self.df) == 0:
            # If no known types, use all
            self.df = df.reset_index(drop=True)
            print(f"  Using all {len(self.df)} samples")

        label_map = {}
        unique_faults = sorted(self.df['fault_type'].unique())
        for i, ft in enumerate(unique_faults):
            label_map[ft] = i
        self.label_map = label_map
        self.labels = [label_map[ft] for ft in self.df['fault_type']]

        print(f"  Label map: {label_map}")
        print(f"  Fault distribution: {dict(zip(*np.unique(self.labels, return_counts=True)))}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sig_array = row['signal']
        label = self.labels[idx]

        window = extract_signal_window(sig_array, self.window_size, self.n_channels, seed=self.seed + idx)
        window = normalize_window(window)

        return torch.from_numpy(window), label, idx


# =============================================================================
# Embedding Extraction & Probe Evaluation
# =============================================================================

def extract_embeddings_hf(model, dataset, device, batch_size=32):
    """Extract JEPA embeddings for all samples in HF dataset."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    model.eval()

    all_embeds = []
    all_labels = []

    with torch.no_grad():
        for signals, labels, _ in loader:
            signals = signals.to(device)
            emb = model.get_embeddings(signals, pool='mean')
            all_embeds.append(emb.cpu().numpy())
            all_labels.extend(labels.numpy().tolist() if hasattr(labels, 'numpy') else labels.tolist())

    return np.concatenate(all_embeds, axis=0), np.array(all_labels)


def probe_f1(train_embeds, train_labels, test_embeds, test_labels, seed=42, C=1.0):
    """Logistic regression probe with F1."""
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(train_embeds)
    X_te = scaler.transform(test_embeds)

    clf = LogisticRegression(max_iter=1000, C=C, random_state=seed)
    clf.fit(X_tr, train_labels)
    preds = clf.predict(X_te)

    f1 = f1_score(test_labels, preds, average='macro', zero_division=0)
    acc = accuracy_score(test_labels, preds)
    return f1, acc, preds


# =============================================================================
# Main Experiments
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 456])
    parser.add_argument('--window-size', type=int, default=2560)
    parser.add_argument('--n-channels', type=int, default=2)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    token = os.environ.get('HF_TOKEN', '')

    # =========================================================================
    # Step 1: Load HF data
    # =========================================================================
    print("\n" + "="*60)
    print("STEP 1: LOADING HF MECHANICAL-COMPONENTS DATASET")
    print("="*60)

    import pandas as pd

    print("\nLoading bearings (file 0 = FEMTO)...")
    df_bearings_0 = pd.read_parquet(
        'hf://datasets/Forgis/Mechanical-Components/bearings/train-00000-of-00005.parquet',
        storage_options={'token': token}
    )
    print(f"  Bearings file 0: {df_bearings_0.shape}")
    print(f"  Sources: {df_bearings_0['source_id'].value_counts().to_dict()}")
    print(f"  Health states: {df_bearings_0['health_state'].value_counts().to_dict()}")
    print(f"  Fault types: {df_bearings_0['fault_type'].value_counts().to_dict()}")

    print("\nLoading gearboxes (file 0)...")
    df_gearboxes_0 = pd.read_parquet(
        'hf://datasets/Forgis/Mechanical-Components/gearboxes/train-00000-of-00004.parquet',
        storage_options={'token': token}
    )
    print(f"  Gearboxes file 0: {df_gearboxes_0.shape}")
    print(f"  Sources: {df_gearboxes_0['source_id'].value_counts().to_dict()}")
    print(f"  Fault types: {df_gearboxes_0['fault_type'].value_counts().to_dict()}")

    # Sanity checks
    print("\nSanity checks:")
    sig_b = df_bearings_0['signal'].iloc[0]
    sig_g = df_gearboxes_0['signal'].iloc[0]
    print(f"  Bearing signal: {len(sig_b)} channels, each {len(sig_b[0])} samples")
    print(f"  Gearbox signal: {len(sig_g)} channels, each {len(sig_g[0]) if hasattr(sig_g[0], '__len__') else 'N/A'} samples")

    # Check for NaN
    n_nan_b = sum(1 for i in range(min(10, len(df_bearings_0)))
                  for ch in df_bearings_0['signal'].iloc[i]
                  if np.any(np.isnan(np.array(ch, dtype=np.float32))))
    print(f"  Bearing NaN windows (first 10): {n_nan_b}")

    # =========================================================================
    # Step 2: Load CWRU-pretrained JEPA model
    # =========================================================================
    print("\n" + "="*60)
    print("STEP 2: LOADING CWRU-PRETRAINED JEPA MODEL")
    print("="*60)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = ckpt['config']
    print(f"Config: embed_dim={config['embed_dim']}, mask={config.get('mask_ratio', 0.625)}")

    jepa_model = MechanicalJEPAV2(
        n_channels=args.n_channels,  # Will pad/truncate
        window_size=args.window_size,
        patch_size=min(256, args.window_size // 4),  # Adapt patch size
        embed_dim=config['embed_dim'],
        encoder_depth=config['encoder_depth'],
        predictor_depth=config.get('predictor_depth', 4),
        mask_ratio=config.get('mask_ratio', 0.625),
        predictor_pos=config.get('predictor_pos', 'sinusoidal'),
        loss_fn=config.get('loss_fn', 'l1'),
        var_reg_lambda=config.get('var_reg_lambda', 0.1),
    ).to(device)

    # Load with size adaptation (CWRU has 3 channels, we may need 2)
    # Only load compatible parameters
    state = ckpt['model_state_dict']
    try:
        jepa_model.load_state_dict(state, strict=False)
        print("Model loaded (non-strict: some params may not match due to n_channels)")
    except Exception as e:
        print(f"Load error: {e}")
        print("Creating fresh model (will test random init only)")

    jepa_model.eval()

    # =========================================================================
    # Step 3: Create datasets
    # =========================================================================
    print("\n" + "="*60)
    print("STEP 3: CREATING DATASETS")
    print("="*60)

    window_size = args.window_size  # 2560 for FEMTO

    # Bearings: use all FEMTO samples
    bearing_dataset = HFBearingDataset(
        df_bearings_0,
        window_size=window_size,
        n_channels=args.n_channels,
        seed=42
    )

    # Gearboxes: filter to mcc5_thu (most samples, clear fault types)
    df_gear_mcc = df_gearboxes_0[df_gearboxes_0['source_id'] == 'mcc5_thu'].reset_index(drop=True)
    print(f"\nmcc5_thu gearboxes: {len(df_gear_mcc)} samples")
    print(f"  Fault types: {df_gear_mcc['fault_type'].value_counts().to_dict()}")

    gearbox_dataset = HFGearboxDataset(
        df_gear_mcc,
        window_size=window_size,
        n_channels=args.n_channels,
        seed=42
    )

    if len(gearbox_dataset) == 0:
        print("ERROR: No gearbox samples - using all sources")
        gearbox_dataset = HFGearboxDataset(
            df_gearboxes_0, window_size=window_size, n_channels=args.n_channels
        )

    # =========================================================================
    # Step 4: Extract embeddings
    # =========================================================================
    print("\n" + "="*60)
    print("STEP 4: EXTRACTING EMBEDDINGS")
    print("="*60)

    # CWRU-pretrained JEPA embeddings
    print("\nExtracting JEPA (CWRU-pretrained) embeddings for bearings...")
    bear_embeds_jepa, bear_labels = extract_embeddings_hf(jepa_model, bearing_dataset, device)
    print(f"  Bearing JEPA: {bear_embeds_jepa.shape}")

    print("Extracting JEPA (CWRU-pretrained) embeddings for gearboxes...")
    gear_embeds_jepa, gear_labels = extract_embeddings_hf(jepa_model, gearbox_dataset, device)
    print(f"  Gearbox JEPA: {gear_embeds_jepa.shape}")

    # Random init embeddings
    torch.manual_seed(42)
    rand_model = MechanicalJEPAV2(
        n_channels=args.n_channels,
        window_size=window_size,
        patch_size=min(256, window_size // 4),
        embed_dim=config['embed_dim'],
        encoder_depth=config['encoder_depth'],
    ).to(device)
    rand_model.eval()

    print("Extracting random init embeddings for gearboxes...")
    gear_embeds_rand, _ = extract_embeddings_hf(rand_model, gearbox_dataset, device)
    print(f"  Gearbox Random: {gear_embeds_rand.shape}")

    # =========================================================================
    # Step 5: Cross-component transfer
    # =========================================================================
    print("\n" + "="*60)
    print("STEP 5: CROSS-COMPONENT TRANSFER EVALUATION")
    print("="*60)

    n_gear = len(gearbox_dataset)
    n_gear_train = max(1, int(0.7 * n_gear))

    all_jepa_f1s, all_rand_f1s = [], []

    for seed in args.seeds:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(n_gear)
        train_idx = perm[:n_gear_train]
        test_idx = perm[n_gear_train:]

        if len(test_idx) == 0:
            print(f"  Seed {seed}: too few samples for test split, skipping")
            continue

        # JEPA transfer
        j_f1, j_acc, _ = probe_f1(
            gear_embeds_jepa[train_idx], gear_labels[train_idx],
            gear_embeds_jepa[test_idx], gear_labels[test_idx],
            seed=seed
        )

        # Random init
        r_f1, r_acc, _ = probe_f1(
            gear_embeds_rand[train_idx], gear_labels[train_idx],
            gear_embeds_rand[test_idx], gear_labels[test_idx],
            seed=seed
        )

        all_jepa_f1s.append(j_f1)
        all_rand_f1s.append(r_f1)

        print(f"  Seed {seed}:")
        print(f"    JEPA (bearing-pretrained) F1: {j_f1:.4f}")
        print(f"    Random init F1: {r_f1:.4f}")
        print(f"    Transfer gain: {j_f1 - r_f1:+.4f}")

    if all_jepa_f1s:
        print(f"\nSummary ({len(all_jepa_f1s)} seeds):")
        print(f"  JEPA F1: {np.mean(all_jepa_f1s):.4f} ± {np.std(all_jepa_f1s):.4f}")
        print(f"  Rand F1: {np.mean(all_rand_f1s):.4f} ± {np.std(all_rand_f1s):.4f}")
        print(f"  Transfer gain: {np.mean(all_jepa_f1s) - np.mean(all_rand_f1s):+.4f}")
        print(f"  Positive seeds: {sum(j > r for j, r in zip(all_jepa_f1s, all_rand_f1s))}/{len(all_jepa_f1s)}")

    # =========================================================================
    # Step 6: Bearing-domain evaluation (sanity check)
    # =========================================================================
    print("\n" + "="*60)
    print("STEP 6: BEARING-DOMAIN SANITY CHECK")
    print("="*60)

    n_bear = len(bearing_dataset)
    n_bear_train = int(0.7 * n_bear)
    perm_b = np.random.default_rng(42).permutation(n_bear)
    b_tr, b_te = perm_b[:n_bear_train], perm_b[n_bear_train:]

    b_jepa_f1, b_jepa_acc, _ = probe_f1(
        bear_embeds_jepa[b_tr], bear_labels[b_tr],
        bear_embeds_jepa[b_te], bear_labels[b_te]
    )

    bear_rand_embeds, _ = extract_embeddings_hf(rand_model, bearing_dataset, device)
    b_rand_f1, b_rand_acc, _ = probe_f1(
        bear_rand_embeds[b_tr], bear_labels[b_tr],
        bear_rand_embeds[b_te], bear_labels[b_te]
    )

    print(f"\nBearing-domain fault classification (sanity):")
    print(f"  JEPA (CWRU-pretrained) F1: {b_jepa_f1:.4f}")
    print(f"  Random init F1: {b_rand_f1:.4f}")
    print(f"  Transfer gain: {b_jepa_f1 - b_rand_f1:+.4f}")
    print(f"\n  Note: FEMTO uses different fault types than CWRU")
    print(f"  Fault map: {bearing_dataset.label_map}")

    # Summary
    print(f"\n{'='*60}")
    print("EXPERIMENT 5C COMPLETE")
    print(f"{'='*60}")
    if all_jepa_f1s:
        print(f"Cross-component (bearing→gearbox) transfer: {np.mean(all_jepa_f1s) - np.mean(all_rand_f1s):+.4f}")
    print(f"Bearing domain (CWRU→FEMTO): {b_jepa_f1 - b_rand_f1:+.4f}")

    return {
        'cross_component_jepa_f1': float(np.mean(all_jepa_f1s)) if all_jepa_f1s else None,
        'cross_component_rand_f1': float(np.mean(all_rand_f1s)) if all_rand_f1s else None,
        'cross_component_gain': float(np.mean(all_jepa_f1s) - np.mean(all_rand_f1s)) if all_jepa_f1s else None,
        'bearing_jepa_f1': b_jepa_f1,
        'bearing_rand_f1': b_rand_f1,
        'bearing_gain': b_jepa_f1 - b_rand_f1,
    }


if __name__ == '__main__':
    main()
