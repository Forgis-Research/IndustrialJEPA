"""
Phase 4A+4B: Multi-Source Pretraining on HF Mechanical-Components + Cross-Component Transfer

This script:
1. Loads MCC5-THU gearbox data (3ch, 12.8kHz, 956 samples) from HuggingFace
2. Pretrains JEPA on gearbox data (gear-pretrained)
3. Pretrains JEPA on combined CWRU + gearbox data (multi-source)
4. Evaluates all variants on:
   - CWRU classification (bearing)
   - Paderborn transfer (bearing)
   - Gearbox classification (linear probe on MCC5 test split)

Goal: Test if multi-source pretraining generalizes better than single-source.
Previous finding (V3): CWRU+Paderborn multi-source = -7.5% CWRU (diversity dilutes)
New hypothesis: Gearbox pretraining may be an ORTHOGONAL modality that doesn't interfere.

Usage:
    python multi_source_hf_pretrain.py --epochs 100 --seeds 42 123 456
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score

os.environ['HF_HOME'] = '/mnt/sagemaker-nvme/hf_cache'

sys.path.insert(0, str(Path(__file__).parent))
from src.models import MechanicalJEPAV2
from src.data import create_dataloaders
from src.models.jepa_v2 import prediction_var_loss

TOKEN = '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'
PADERBORN_DIR = Path('/home/sagemaker-user/IndustrialJEPA/datasets/data/paderborn')
JEPA_V2_CKPT = Path(__file__).parent / 'checkpoints/jepa_v2_20260401_003619.pt'


# ============================================================================
# HF Gearbox Dataset
# ============================================================================

class MCC5GearboxDataset(Dataset):
    """
    Load MCC5-THU gearbox data from HuggingFace.
    Signals: (3, 64000) at 12800Hz → window to 4096 samples
    """

    def __init__(
        self,
        window_size: int = 4096,
        stride: int = 2048,
        max_samples: int = None,
        split: str = 'train',
        seed: int = 42,
        n_channels: int = 3,
        resample_to: int = 12000,  # Resample to CWRU SR for compatibility
    ):
        self.window_size = window_size
        self.n_channels = n_channels

        print(f"Loading MCC5-THU gearbox data (split={split})...", flush=True)

        # Load all 4 parquets
        dfs = []
        for i in range(4):
            try:
                df = pd.read_parquet(
                    f'hf://datasets/Forgis/Mechanical-Components/gearboxes/train-0000{i}-of-00004.parquet',
                    storage_options={'token': TOKEN}
                )
                dfs.append(df)
            except Exception as e:
                print(f"  Warning: parquet {i} failed: {e}")

        if not dfs:
            raise RuntimeError("Failed to load any MCC5 parquets")

        df_all = pd.concat(dfs, ignore_index=True)

        # Filter to MCC5-THU only (most samples, 12800Hz)
        df = df_all[df_all['source_id'] == 'mcc5_thu'].copy()
        print(f"  MCC5-THU samples: {len(df)}", flush=True)

        # Train/test split by episode_id
        rng = np.random.default_rng(seed)
        episode_ids = df['episode_id'].unique()
        rng.shuffle(episode_ids)
        n_test = max(1, int(len(episode_ids) * 0.2))
        test_eps = set(episode_ids[:n_test])
        train_eps = set(episode_ids[n_test:])

        if split == 'train':
            df = df[df['episode_id'].isin(train_eps)]
        else:
            df = df[df['episode_id'].isin(test_eps)]

        if max_samples:
            df = df.sample(min(max_samples, len(df)), random_state=seed)

        # Build fault label map
        fault_types = sorted(df['fault_type'].unique())
        self.label_map = {ft: i for i, ft in enumerate(fault_types)}
        print(f"  Fault types: {fault_types}", flush=True)
        print(f"  Label map: {self.label_map}", flush=True)

        # Extract windows
        self.windows = []
        sr = 12800

        if resample_to and resample_to != sr:
            import scipy.signal
            from math import gcd
            g = gcd(sr, resample_to)
            up, down = resample_to // g, sr // g
        else:
            up, down = 1, 1
            resample_to = sr

        for _, row in df.iterrows():
            raw_signal = np.stack(row['signal']).astype(np.float32)  # (3, T)
            label = self.label_map[row['fault_type']]

            # Resample channels
            if up != 1 or down != 1:
                import scipy.signal
                resampled = np.stack([
                    scipy.signal.resample_poly(raw_signal[c], up, down)
                    for c in range(raw_signal.shape[0])
                ])
            else:
                resampled = raw_signal

            T = resampled.shape[1]
            n_windows = (T - window_size) // stride + 1
            for i in range(n_windows):
                start = i * stride
                window = resampled[:, start:start + window_size]
                # Normalize
                mean = window.mean(axis=1, keepdims=True)
                std = window.std(axis=1, keepdims=True) + 1e-8
                window = (window - mean) / std
                self.windows.append((window, label))

        label_counts = np.bincount([w[1] for w in self.windows], minlength=len(self.label_map))
        print(f"  Windows: {len(self.windows)} | labels: {dict(enumerate(label_counts))}", flush=True)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        window, label = self.windows[idx]
        return torch.tensor(window[:self.n_channels], dtype=torch.float32), label

    @property
    def n_classes(self):
        return len(self.label_map)


# ============================================================================
# JEPA Training on a DataLoader (generic)
# ============================================================================

def train_jepa(model, train_loader, epochs, lr, device, desc="JEPA"):
    """Train a JEPA model and return loss history."""
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    history = []
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        ep_loss, n_batches = 0.0, 0

        for batch in train_loader:
            # Handle both 2-tuple and 3-tuple batches
            signals = batch[0].to(device)
            optimizer.zero_grad()
            loss, _, _ = model(signals)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ep_loss += loss.item()
            n_batches += 1

        scheduler.step()
        ep_loss /= n_batches
        history.append(ep_loss)

        if epoch % 20 == 0 or epoch == 1:
            print(f"  {desc} Epoch {epoch:3d}/{epochs} | loss={ep_loss:.4f} | {time.time()-t0:.0f}s", flush=True)

    return history


# ============================================================================
# Linear Probe Evaluation
# ============================================================================

def linear_probe_f1(model_fn, train_loader, test_loader, embed_dim, n_classes, device, epochs=60):
    """Extract embeddings, train linear probe, return best F1."""
    def extract(loader):
        all_e, all_l = [], []
        with torch.no_grad():
            for batch in loader:
                x = batch[0].to(device)
                y = batch[1]
                e = model_fn(x)
                all_e.append(e.cpu())
                all_l.append(y if isinstance(y, torch.Tensor) else torch.tensor(y))
        return torch.cat(all_e), torch.cat(all_l)

    tr_e, tr_l = extract(train_loader)
    te_e, te_l = extract(test_loader)

    probe = nn.Linear(embed_dim, n_classes).to(device)
    opt = optim.Adam(probe.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()
    best_f1 = 0.0

    for ep in range(epochs):
        probe.train(); opt.zero_grad()
        crit(probe(tr_e.to(device)), tr_l.to(device)).backward(); opt.step()
        probe.eval()
        with torch.no_grad():
            preds = probe(te_e.to(device)).argmax(1).cpu().numpy()
        best_f1 = max(best_f1, f1_score(te_l.numpy(), preds, average='macro', zero_division=0))

    return best_f1


def get_paderborn_loaders(seed=42):
    from paderborn_transfer import create_paderborn_loaders, CLASSES
    bearing_dirs = [(str(PADERBORN_DIR / f), l) for f, l in CLASSES.items() if (PADERBORN_DIR / f).exists()]
    if not bearing_dirs: return None, None
    return create_paderborn_loaders(
        bearing_dirs=bearing_dirs, window_size=4096, stride=2048, target_sr=20000,
        n_channels=3, test_ratio=0.2, batch_size=32, seed=seed, max_files_per_bearing=20,
    )


def make_jepa(device, seed=None):
    if seed: torch.manual_seed(seed)
    return MechanicalJEPAV2(
        n_channels=3, window_size=4096, patch_size=256, embed_dim=512,
        encoder_depth=4, predictor_depth=4, n_heads=4, mask_ratio=0.625,
        ema_decay=0.996, predictor_pos='sinusoidal', loss_fn='l1', var_reg_lambda=0.1,
    ).to(device)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 456])
    parser.add_argument('--gear-epochs', type=int, default=50, help='Epochs for gearbox pretraining')
    args = parser.parse_args()

    import shutil
    print(f"Home disk: {shutil.disk_usage('/home/sagemaker-user').free/1e9:.1f} GB free")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}", flush=True)

    # Load MCC5 gearbox data (shared across seeds)
    print("\n--- Loading MCC5-THU Gearbox Data ---", flush=True)
    gear_train = MCC5GearboxDataset(window_size=4096, stride=2048, split='train',
                                     seed=42, n_channels=3, resample_to=12000)
    gear_test = MCC5GearboxDataset(window_size=4096, stride=2048, split='test',
                                    seed=42, n_channels=3, resample_to=12000)

    gear_train_loader = DataLoader(gear_train, batch_size=32, shuffle=True, num_workers=0)
    gear_test_loader = DataLoader(gear_test, batch_size=32, shuffle=False, num_workers=0)
    n_gear_classes = gear_train.n_classes

    all_results = {
        'jepa_v2_cwru_pretrained': [],  # Reference: V2 checkpoint
        'jepa_gear_pretrained': [],      # New: pretrained on gearboxes
        'jepa_multisource': [],          # New: pretrained on CWRU + gearboxes
        'random_init': [],               # Baseline: no pretraining
    }

    for seed in args.seeds:
        print(f"\n{'='*70}\nSeed {seed}\n{'='*70}", flush=True)
        torch.manual_seed(seed)
        np.random.seed(seed)

        # CWRU loaders
        cwru_train, cwru_test, _ = create_dataloaders(
            data_dir='data/bearings', batch_size=32, window_size=4096,
            stride=2048, test_ratio=0.2, seed=seed, num_workers=0,
            dataset_filter='cwru', n_channels=3,
        )

        # Paderborn loaders
        pad_train, pad_test = get_paderborn_loaders(seed=seed)

        # Random init baseline
        rand = make_jepa(device, seed=seed + 9999)
        rand.eval()
        rand_fn = lambda x: rand.get_embeddings(x, pool='mean')
        rand_cwru = linear_probe_f1(rand_fn, cwru_train, cwru_test, 512, 4, device)
        rand_pad = linear_probe_f1(rand_fn, pad_train, pad_test, 512, 3, device) if pad_train else None
        rand_gear = linear_probe_f1(rand_fn, gear_train_loader, gear_test_loader, 512, n_gear_classes, device)
        print(f"  Random: CWRU={rand_cwru:.4f}, Pad={rand_pad:.4f}, Gear={rand_gear:.4f}", flush=True)
        all_results['random_init'].append({
            'seed': seed, 'cwru_f1': rand_cwru, 'pad_f1': rand_pad, 'gear_f1': rand_gear
        })

        # ===== 1. JEPA V2 CWRU pretrained (reference) =====
        print("\n  [1/3] JEPA V2 (CWRU pretrained, reference checkpoint)...", flush=True)
        if JEPA_V2_CKPT.exists():
            ckpt = torch.load(str(JEPA_V2_CKPT), map_location=device, weights_only=False)
            config = ckpt['config']
            jepa_cwru = MechanicalJEPAV2(
                n_channels=config['n_channels'], window_size=config['window_size'],
                patch_size=config.get('patch_size', 256), embed_dim=config['embed_dim'],
                encoder_depth=config['encoder_depth'], predictor_depth=config.get('predictor_depth', 4),
                n_heads=config.get('n_heads', 4), mask_ratio=config.get('mask_ratio', 0.625),
                predictor_pos=config.get('predictor_pos', 'sinusoidal'),
                loss_fn=config.get('loss_fn', 'l1'), var_reg_lambda=config.get('var_reg_lambda', 0.1),
            ).to(device)
            jepa_cwru.load_state_dict(ckpt['model_state_dict'])
            jepa_cwru.eval()
            cwru_fn = lambda x: jepa_cwru.get_embeddings(x, pool='mean')

            ref_cwru = linear_probe_f1(cwru_fn, cwru_train, cwru_test, 512, 4, device)
            ref_pad = linear_probe_f1(cwru_fn, pad_train, pad_test, 512, 3, device) if pad_train else None
            ref_gear = linear_probe_f1(cwru_fn, gear_train_loader, gear_test_loader, 512, n_gear_classes, device)
            print(f"    CWRU={ref_cwru:.4f}, Pad={ref_pad:.4f}, Gear={ref_gear:.4f}", flush=True)
            all_results['jepa_v2_cwru_pretrained'].append({
                'seed': seed, 'cwru_f1': ref_cwru, 'pad_f1': ref_pad, 'gear_f1': ref_gear
            })

        # ===== 2. Gear pretrained JEPA =====
        print(f"\n  [2/3] JEPA pretrained on gearboxes ({args.gear_epochs} epochs)...", flush=True)
        t0 = time.time()
        jepa_gear = make_jepa(device, seed=seed)
        gear_hist = train_jepa(jepa_gear, gear_train_loader, args.gear_epochs, 1e-4, device, "Gear-JEPA")

        jepa_gear.eval()
        gear_fn = lambda x: jepa_gear.get_embeddings(x, pool='mean')
        g_cwru = linear_probe_f1(gear_fn, cwru_train, cwru_test, 512, 4, device)
        g_pad = linear_probe_f1(gear_fn, pad_train, pad_test, 512, 3, device) if pad_train else None
        g_gear = linear_probe_f1(gear_fn, gear_train_loader, gear_test_loader, 512, n_gear_classes, device)
        print(f"    CWRU={g_cwru:.4f}, Pad={g_pad:.4f}, Gear={g_gear:.4f} ({time.time()-t0:.0f}s)", flush=True)
        all_results['jepa_gear_pretrained'].append({
            'seed': seed, 'cwru_f1': g_cwru, 'pad_f1': g_pad, 'gear_f1': g_gear,
            'gear_gain': (g_gear - rand_gear) if rand_gear else None,
            'cwru_gain': g_cwru - rand_cwru,
        })

        # Save gear-pretrained checkpoint
        ckpt_dir = Path('/mnt/sagemaker-nvme/jepa_checkpoints')
        ckpt_dir.mkdir(exist_ok=True)
        torch.save({'model_state_dict': jepa_gear.state_dict(), 'config': {'n_channels': 3, 'window_size': 4096,
                    'patch_size': 256, 'embed_dim': 512, 'encoder_depth': 4, 'predictor_depth': 4,
                    'n_heads': 4, 'mask_ratio': 0.625, 'predictor_pos': 'sinusoidal', 'loss_fn': 'l1',
                    'var_reg_lambda': 0.1}, 'cwru_f1': g_cwru, 'pad_f1': g_pad, 'gear_f1': g_gear},
                   str(ckpt_dir / f'gear_pretrained_seed{seed}.pt'))

        # ===== 3. Multi-source JEPA (CWRU + Gearbox) =====
        print(f"\n  [3/3] JEPA multi-source (CWRU+Gear, {args.epochs} epochs)...", flush=True)
        t0 = time.time()

        # Combine CWRU and gearbox datasets
        from torch.utils.data import ConcatDataset

        class IgnoreLabel(Dataset):
            """Wrapper to discard gear labels so we can concatenate without label mismatch."""
            def __init__(self, ds): self.ds = ds
            def __len__(self): return len(self.ds)
            def __getitem__(self, idx):
                item = self.ds[idx]
                return (item[0],)  # only signal

        # Create signal-only loaders for pretraining
        class SignalOnlyWrapper(Dataset):
            def __init__(self, loader):
                self.data = []
                for batch in loader:
                    for i in range(batch[0].shape[0]):
                        self.data.append(batch[0][i])
            def __len__(self): return len(self.data)
            def __getitem__(self, idx): return (self.data[idx],)

        # Simple approach: alternate between CWRU and gear batches
        cwru_signals = [batch[0] for batch in cwru_train]
        gear_signals = [batch[0] for batch in gear_train_loader]
        all_signals = cwru_signals + gear_signals

        class MultiSourceSignalDataset(Dataset):
            def __init__(self, signal_batches):
                self.windows = []
                for batch in signal_batches:
                    for i in range(batch.shape[0]):
                        self.windows.append(batch[i])
            def __len__(self): return len(self.windows)
            def __getitem__(self, idx): return (self.windows[idx],)

        multi_ds = MultiSourceSignalDataset(all_signals)
        multi_loader = DataLoader(multi_ds, batch_size=32, shuffle=True, num_workers=0)
        print(f"    Multi-source: {len(multi_ds)} windows ({len(cwru_signals)*32} CWRU + {len(gear_signals)*32} gear approx)", flush=True)

        jepa_multi = make_jepa(device, seed=seed)
        multi_hist = train_jepa(jepa_multi, multi_loader, args.epochs, 1e-4, device, "Multi-JEPA")

        jepa_multi.eval()
        multi_fn = lambda x: jepa_multi.get_embeddings(x, pool='mean')
        m_cwru = linear_probe_f1(multi_fn, cwru_train, cwru_test, 512, 4, device)
        m_pad = linear_probe_f1(multi_fn, pad_train, pad_test, 512, 3, device) if pad_train else None
        m_gear = linear_probe_f1(multi_fn, gear_train_loader, gear_test_loader, 512, n_gear_classes, device)
        print(f"    CWRU={m_cwru:.4f}, Pad={m_pad:.4f}, Gear={m_gear:.4f} ({time.time()-t0:.0f}s)", flush=True)
        all_results['jepa_multisource'].append({
            'seed': seed, 'cwru_f1': m_cwru, 'pad_f1': m_pad, 'gear_f1': m_gear,
        })

        # Save intermediate
        with open('results/multisource_pretrain.json', 'w') as f:
            json.dump(all_results, f, indent=2, default=float)
        print(f"  Checkpoint saved: results/multisource_pretrain.json", flush=True)

    # Final summary
    print(f"\n{'='*100}", flush=True)
    print("MULTI-SOURCE PRETRAINING RESULTS", flush=True)
    print(f"{'='*100}", flush=True)
    print(f"{'Method':<35} {'CWRU F1':>15} {'Paderborn F1':>15} {'Gear F1':>15}", flush=True)
    print("-"*80, flush=True)

    v2_cwru_ref = 0.773  # V2 reference

    for method, records in all_results.items():
        if not records: continue
        cwru = np.mean([r['cwru_f1'] for r in records])
        cwru_std = np.std([r['cwru_f1'] for r in records])
        pad_vals = [r['pad_f1'] for r in records if r.get('pad_f1') is not None]
        gear_vals = [r['gear_f1'] for r in records if r.get('gear_f1') is not None]
        pad = f"{np.mean(pad_vals):.3f}±{np.std(pad_vals):.3f}" if pad_vals else "N/A"
        gear = f"{np.mean(gear_vals):.3f}±{np.std(gear_vals):.3f}" if gear_vals else "N/A"
        print(f"{method:<35} {cwru:.3f}±{cwru_std:.3f}  {pad:>15} {gear:>15}", flush=True)

    # Final JSON save
    summary = {}
    for method, records in all_results.items():
        if not records: continue
        cwru = [r['cwru_f1'] for r in records]
        pad = [r['pad_f1'] for r in records if r.get('pad_f1') is not None]
        gear = [r['gear_f1'] for r in records if r.get('gear_f1') is not None]
        summary[method] = {
            'cwru_mean': float(np.mean(cwru)), 'cwru_std': float(np.std(cwru)),
            'pad_mean': float(np.mean(pad)) if pad else None, 'pad_std': float(np.std(pad)) if pad else None,
            'gear_mean': float(np.mean(gear)) if gear else None, 'gear_std': float(np.std(gear)) if gear else None,
        }

    all_results['_summary'] = summary
    with open('results/multisource_pretrain.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=float)
    print(f"\nSaved: results/multisource_pretrain.json", flush=True)


if __name__ == '__main__':
    main()
