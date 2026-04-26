"""
Phase 5: Shuffle Test for V16b Encoder Validity (Rule 4 from research protocol).

Tests whether the V16b encoder actually uses temporal information.
Ablations:
1. Original: frozen encoder -> linear probe
2. Shuffled input: temporally permuted encoder -> linear probe (should degrade if temporal matters)
3. Random features: zero-initialized features of same dim -> linear probe (lower bound)
4. Mean-pool raw: mean pool raw sensor data -> linear probe (no encoder)

If shuffled ~= original, the encoder is not using temporal order.
If random == original, the encoder contributes nothing.
"""
import sys, json, copy
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

V11_DIR = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/experiments/v11')
V16_DIR = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/experiments/v16')
sys.path.insert(0, str(V11_DIR))
sys.path.insert(0, str(V16_DIR))

from data_utils import (
    load_cmapss_subset, N_SENSORS, RUL_CAP,
    CMAPSSFinetuneDataset,
    collate_finetune,
)
from phase1_v16a import V16aJEPA, D_MODEL, N_HEADS, N_LAYERS, EMA_MOMENTUM

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PROBE_SEEDS = [42, 123, 456]
N_PROBE_EPOCHS = 30


def train_probe(features, labels, val_features, val_labels, seed=42, n_epochs=N_PROBE_EPOCHS):
    """Train a linear probe on fixed features. Returns val RMSE."""
    torch.manual_seed(seed)
    D = features.shape[1]
    probe = nn.Linear(D, 1).to(DEVICE)
    optim = torch.optim.Adam(probe.parameters(), lr=1e-3)

    feats = torch.tensor(features, dtype=torch.float32).to(DEVICE)
    labs = torch.tensor(labels, dtype=torch.float32).to(DEVICE)
    val_feats = torch.tensor(val_features, dtype=torch.float32).to(DEVICE)
    val_labs = torch.tensor(val_labels, dtype=torch.float32).to(DEVICE)

    best_val = float('inf')
    for ep in range(n_epochs):
        probe.train()
        idx = torch.randperm(len(feats), device=DEVICE)
        for i in range(0, len(feats), 64):
            batch_idx = idx[i:i+64]
            pred = probe(feats[batch_idx]).squeeze(-1)
            loss = F.mse_loss(pred, labs[batch_idx])
            optim.zero_grad()
            loss.backward()
            optim.step()

        probe.eval()
        with torch.no_grad():
            val_pred = probe(val_feats).squeeze(-1)
            val_rmse = float(torch.sqrt(F.mse_loss(val_pred * RUL_CAP, val_labs * RUL_CAP)))
        if val_rmse < best_val:
            best_val = val_rmse

    return best_val


def extract_features(model, engines, shuffle_time=False, seed=42):
    """Extract frozen encoder features. Optionally shuffle temporal order."""
    model.eval()
    all_feats, all_labels = [], []

    with torch.no_grad():
        for eng in engines:
            # eng is a 2D array (T, N_sensors)
            T, S = eng['sensors'].shape if hasattr(eng, '__getitem__') else (len(eng[0]), N_SENSORS)
            # Handle both dict and tuple formats
            if isinstance(eng, dict):
                past = torch.tensor(eng['sensors'], dtype=torch.float32)
                rul = eng['rul']
            else:
                past, rul = eng

            if len(past.shape) == 2:
                past = past.unsqueeze(0)  # (1, T, S)

            if shuffle_time:
                rng = np.random.default_rng(seed)
                perm = rng.permutation(past.shape[1])
                past = past[:, perm, :]

            past = past.to(DEVICE)
            mask = torch.ones(past.shape[0], past.shape[1], dtype=torch.bool, device=DEVICE)
            feat = model.encode_context(past, mask)
            all_feats.append(feat.cpu().numpy())
            all_labels.append([rul] if np.isscalar(rul) else rul)

    return np.concatenate(all_feats, axis=0), np.array(all_labels).flatten()


def extract_mean_pool_raw(engines):
    """Extract mean-pooled raw sensor data (no encoder)."""
    all_feats, all_labels = [], []
    for eng in engines:
        if isinstance(eng, dict):
            past = np.array(eng['sensors'])
            rul = eng['rul']
        else:
            past, rul = eng
        feat = np.mean(past, axis=0)  # (N_sensors,)
        all_feats.append(feat)
        all_labels.append(rul if np.isscalar(rul) else rul[0])
    return np.array(all_feats), np.array(all_labels)


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings('ignore')

    print("=" * 60)
    print("Phase 5: Shuffle Test for V16b Encoder Validity")
    print("=" * 60)

    # Check which checkpoints exist
    available_seeds = []
    for s in [42, 123, 456]:
        ckpt = V16_DIR / f'best_v16b_seed{s}.pt'
        if ckpt.exists():
            available_seeds.append(s)
            print(f"  Found checkpoint: best_v16b_seed{s}.pt")
        else:
            print(f"  Missing checkpoint: best_v16b_seed{s}.pt")

    if not available_seeds:
        print("No checkpoints found. Exiting.")
        import sys; sys.exit(1)

    # Load data using CMAPSSFinetuneDataset approach
    data = load_cmapss_subset('FD001')
    train_engines = data['train_engines']
    val_engines = data['val_engines']

    # Build simple dataset for feature extraction
    # Use "last window" for each engine (most informative for RUL)
    print("\nExtracting features from validation engines...")

    results = {
        'description': 'Shuffle test for V16b encoder - Rule 4 validity check',
        'checkpoint_seed_used': available_seeds[0],
        'probe_seeds': PROBE_SEEDS,
        'ablations': {}
    }

    # Load best checkpoint (use first available)
    ckpt_seed = available_seeds[0]
    model = V16aJEPA(
        n_sensors=N_SENSORS, d_model=D_MODEL, n_heads=N_HEADS,
        n_layers=N_LAYERS, dropout=0.1, ema_momentum=EMA_MOMENTUM,
    ).to(DEVICE)
    ckpt_path = V16_DIR / f'best_v16b_seed{ckpt_seed}.pt'
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    model.eval()
    print(f"\nLoaded checkpoint: best_v16b_seed{ckpt_seed}.pt")

    # Use finetuning dataset (properly handles variable-length sequences)
    tr_ds = CMAPSSFinetuneDataset(train_engines, n_cuts_per_engine=5, seed=42)
    va_ds = CMAPSSFinetuneDataset(val_engines, use_last_only=True)

    tr_loader = DataLoader(tr_ds, batch_size=64, shuffle=False, collate_fn=collate_finetune)
    va_loader = DataLoader(va_ds, batch_size=64, shuffle=False, collate_fn=collate_finetune)

    def extract_from_loader(loader, shuffle_time=False, use_random=False, use_mean_pool=False):
        model.eval()
        all_feats, all_labels = [], []
        with torch.no_grad():
            for past, mask, rul in loader:
                past, mask = past.to(DEVICE), mask.to(DEVICE)

                if use_random:
                    # Random features same dim as encoder output
                    feat = torch.randn(past.shape[0], D_MODEL, device=DEVICE)
                elif use_mean_pool:
                    # Mean pool raw sensor data (ignoring padding via mask)
                    # past: (B, T, S), mask: (B, T) where True=PADDING, False=valid
                    # Invert mask to get valid positions (1=valid, 0=padding)
                    m = (~mask).unsqueeze(-1).float()
                    feat_raw = (past * m).sum(1) / m.sum(1).clamp(min=1)  # (B, S)
                    # Pad to D_MODEL dim with zeros for fair probe comparison
                    feat = torch.zeros(past.shape[0], D_MODEL, device=DEVICE)
                    feat[:, :N_SENSORS] = feat_raw
                elif shuffle_time:
                    # Shuffle temporal order within each sequence
                    B, T, S = past.shape
                    perm = torch.randperm(T, device=DEVICE)
                    past_shuffled = past[:, perm, :]
                    feat = model.encode_context(past_shuffled, mask)
                else:
                    feat = model.encode_context(past, mask)

                all_feats.append(feat.cpu().numpy())
                all_labels.append(rul.numpy())

        return np.concatenate(all_feats), np.concatenate(all_labels)

    print("\nExtracting features for all ablations...")

    # 1. Original encoder
    print("  [1/4] Original encoder features...")
    tr_feats_orig, tr_labs = extract_from_loader(tr_loader, shuffle_time=False)
    va_feats_orig, va_labs = extract_from_loader(va_loader, shuffle_time=False)

    # 2. Shuffled temporal order
    print("  [2/4] Shuffled temporal encoder features...")
    tr_feats_shuf, _ = extract_from_loader(tr_loader, shuffle_time=True)
    va_feats_shuf, _ = extract_from_loader(va_loader, shuffle_time=True)

    # 3. Random features
    print("  [3/4] Random features (zero encoder)...")
    tr_feats_rand, _ = extract_from_loader(tr_loader, use_random=True)
    va_feats_rand, _ = extract_from_loader(va_loader, use_random=True)

    # 4. Mean-pooled raw features
    print("  [4/4] Mean-pooled raw sensor features...")
    tr_feats_pool, _ = extract_from_loader(tr_loader, use_mean_pool=True)
    va_feats_pool, _ = extract_from_loader(va_loader, use_mean_pool=True)

    # Train probes for each ablation
    print("\nTraining probes (3 seeds each)...")
    for name, tr_f, va_f in [
        ('original', tr_feats_orig, va_feats_orig),
        ('shuffled_time', tr_feats_shuf, va_feats_shuf),
        ('random_features', tr_feats_rand, va_feats_rand),
        ('mean_pool_raw', tr_feats_pool, va_feats_pool),
    ]:
        rmses = []
        for s in PROBE_SEEDS:
            rmse = train_probe(tr_f, tr_labs, va_f, va_labs, seed=s)
            rmses.append(rmse)
        mean_rmse = float(np.mean(rmses))
        std_rmse = float(np.std(rmses))
        results['ablations'][name] = {
            'val_rmse_mean': mean_rmse,
            'val_rmse_std': std_rmse,
            'val_rmse_per_seed': [float(r) for r in rmses]
        }
        print(f"  {name}: val RMSE = {mean_rmse:.2f} +/- {std_rmse:.2f}")

    # Summary
    print("\n" + "=" * 60)
    print("SHUFFLE TEST SUMMARY")
    print("=" * 60)
    orig = results['ablations']['original']['val_rmse_mean']
    shuf = results['ablations']['shuffled_time']['val_rmse_mean']
    rand = results['ablations']['random_features']['val_rmse_mean']
    pool = results['ablations']['mean_pool_raw']['val_rmse_mean']

    print(f"  Original encoder:      {orig:.2f}")
    print(f"  Shuffled time order:   {shuf:.2f}  (delta: +{shuf-orig:.2f})")
    print(f"  Random features:       {rand:.2f}  (delta: +{rand-orig:.2f})")
    print(f"  Mean-pool raw sensors: {pool:.2f}  (delta: +{pool-orig:.2f})")
    print()

    if shuf - orig < 2.0:
        print("  WARNING: Shuffled ~= original. Encoder may not use temporal order.")
    else:
        print(f"  OK: Shuffling degrades by {shuf-orig:.2f} RMSE — encoder uses temporal order.")

    if pool - orig < 2.0:
        print("  WARNING: Mean-pool ~= encoder. Encoder may not add value over raw features.")
    else:
        print(f"  OK: Encoder beats mean-pool by {pool-orig:.2f} RMSE.")

    results['summary'] = {
        'temporal_delta': float(shuf - orig),
        'vs_random_delta': float(rand - orig),
        'vs_mean_pool_delta': float(pool - orig),
        'encoder_uses_temporal_order': bool(shuf - orig > 2.0),
        'encoder_beats_mean_pool': bool(pool - orig > 2.0),
    }

    out_path = V16_DIR / 'phase5_shuffle_test_results.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")
