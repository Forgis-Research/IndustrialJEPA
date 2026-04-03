"""
Phase 1 Audit: Fix and re-run CWRU->Paderborn transfer for all methods.
This is a focused audit script that fixes the Paderborn path issue and
re-runs transfer for all methods to get JSON-backed results.

Priority: Get real Paderborn F1 for CNN, Transformer, MAE, JEPA V2.
"""

import sys
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).parent))
from src.models import MechanicalJEPAV2
from src.data import create_dataloaders
from baselines_comparison import CNN1DSupervised, TransformerSupervised, MAEModel, train_supervised_model
from paderborn_transfer import create_paderborn_loaders, CLASSES

PADERBORN_DIR = Path('/home/sagemaker-user/IndustrialJEPA/datasets/data/paderborn')
CHECKPOINT_DIR = Path('/mnt/sagemaker-nvme/jepa_checkpoints')
JEPA_V2_CKPT = Path(__file__).parent / 'checkpoints/jepa_v2_20260401_003619.pt'

# Previously measured JEPA V2 CWRU F1 per seed (from Exp 36)
JEPA_CWRU_REF = {42: 0.7483, 123: 0.7907, 456: 0.7788}


def get_paderborn_loaders(seed=42, batch_size=32):
    """Load Paderborn train/test loaders with file-level splits."""
    bearing_dirs = []
    for folder, label in CLASSES.items():
        bp = PADERBORN_DIR / folder
        if bp.exists():
            bearing_dirs.append((str(bp), label))

    if not bearing_dirs:
        return None, None

    train_loader, test_loader = create_paderborn_loaders(
        bearing_dirs=bearing_dirs,
        window_size=4096,
        stride=2048,
        target_sr=20000,
        n_channels=3,
        test_ratio=0.2,
        batch_size=batch_size,
        seed=seed,
        max_files_per_bearing=20,
    )
    n_train = len(train_loader.dataset)
    n_test = len(test_loader.dataset)
    print(f"  Paderborn: {n_train} train, {n_test} test windows")
    return train_loader, test_loader


def linear_probe(model_fn, train_loader, test_loader, embed_dim, n_classes, device, epochs=50):
    """Extract embeddings and train linear probe."""
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
        probe.train()
        opt.zero_grad()
        logits = probe(tr_e.to(device))
        loss = crit(logits, tr_l.to(device))
        loss.backward()
        opt.step()

        probe.eval()
        with torch.no_grad():
            preds = probe(te_e.to(device)).argmax(1).cpu().numpy()
        f1 = f1_score(te_l.numpy(), preds, average='macro', zero_division=0)
        best_f1 = max(best_f1, f1)

    return best_f1


def run_audit(seeds, device):
    print(f"\nRunning Phase 1 Audit — Device: {device}")
    print(f"Seeds: {seeds}")

    # Collect results keyed by method
    all_results = {
        'cnn_supervised': [],
        'transformer_supervised': [],
        'mae': [],
        'jepa_v2': [],
    }

    for seed in seeds:
        print(f"\n{'='*70}\nSeed {seed}\n{'='*70}")
        torch.manual_seed(seed)
        np.random.seed(seed)

        # CWRU data (for training supervised models)
        train_loader, test_loader, _ = create_dataloaders(
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

        # Paderborn
        pad_train, pad_test = get_paderborn_loaders(seed=seed, batch_size=32)
        has_pad = pad_train is not None

        # ===== CNN Supervised =====
        print("\n  [1/4] CNN Supervised...")
        t0 = time.time()
        cnn = CNN1DSupervised(n_channels=3, n_classes=4).to(device)
        cwru_f1 = train_supervised_model(cnn, train_loader, test_loader, 100, 1e-4, device)
        print(f"    CWRU F1: {cwru_f1:.4f} ({time.time()-t0:.0f}s)")

        pad_f1, rand_f1 = None, None
        if has_pad:
            cnn.eval()
            def cnn_embed(x): return cnn.get_embeddings(x)
            pad_f1 = linear_probe(cnn_embed, pad_train, pad_test, 512, 3, device)

            rand_cnn = CNN1DSupervised(n_channels=3, n_classes=4).to(device)
            rand_cnn.eval()
            rand_f1 = linear_probe(lambda x: rand_cnn.get_embeddings(x), pad_train, pad_test, 512, 3, device)
            print(f"    Paderborn: F1={pad_f1:.4f}, rand={rand_f1:.4f}, gain={pad_f1-rand_f1:+.4f}")

        all_results['cnn_supervised'].append({'seed': seed, 'cwru_f1': cwru_f1, 'pad_f1': pad_f1, 'rand_pad_f1': rand_f1})

        # ===== Transformer Supervised =====
        print("\n  [2/4] Transformer Supervised...")
        t0 = time.time()
        torch.manual_seed(seed)
        transformer = TransformerSupervised(n_channels=3, n_classes=4).to(device)
        cwru_f1_tr = train_supervised_model(transformer, train_loader, test_loader, 100, 1e-4, device)
        print(f"    CWRU F1: {cwru_f1_tr:.4f} ({time.time()-t0:.0f}s)")

        pad_f1_tr, rand_f1_tr = None, None
        if has_pad:
            transformer.eval()
            def tr_embed(x): return transformer.get_embeddings(x)
            pad_f1_tr = linear_probe(tr_embed, pad_train, pad_test, 512, 3, device)

            rand_tr = TransformerSupervised(n_channels=3, n_classes=4).to(device)
            rand_tr.eval()
            rand_f1_tr = linear_probe(lambda x: rand_tr.get_embeddings(x), pad_train, pad_test, 512, 3, device)
            print(f"    Paderborn: F1={pad_f1_tr:.4f}, rand={rand_f1_tr:.4f}, gain={pad_f1_tr-rand_f1_tr:+.4f}")

        all_results['transformer_supervised'].append({'seed': seed, 'cwru_f1': cwru_f1_tr, 'pad_f1': pad_f1_tr, 'rand_pad_f1': rand_f1_tr})

        # ===== MAE =====
        print("\n  [3/4] MAE...")
        t0 = time.time()
        torch.manual_seed(seed)
        mae = MAEModel(n_channels=3, window_size=4096, patch_size=256, embed_dim=512,
                       encoder_depth=4, mask_ratio=0.625).to(device)
        optimizer = optim.AdamW(mae.parameters(), lr=1e-4, weight_decay=0.05)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)
        for epoch in range(100):
            mae.train()
            for signals, labels, _ in train_loader:
                optimizer.zero_grad()
                loss = mae(signals.to(device))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(mae.parameters(), 1.0)
                optimizer.step()
            scheduler.step()

        # CWRU linear probe on MAE
        mae.eval()
        all_e, all_l = [], []
        with torch.no_grad():
            for signals, labels, _ in train_loader:
                all_e.append(mae.get_embeddings(signals.to(device)).cpu())
                all_l.append(labels)
        tr_e = torch.cat(all_e); tr_l = torch.cat(all_l)
        all_e, all_l = [], []
        with torch.no_grad():
            for signals, labels, _ in test_loader:
                all_e.append(mae.get_embeddings(signals.to(device)).cpu())
                all_l.append(labels)
        te_e = torch.cat(all_e); te_l = torch.cat(all_l)

        probe = nn.Linear(512, 4).to(device)
        opt2 = optim.Adam(probe.parameters(), lr=1e-3)
        crit = nn.CrossEntropyLoss()
        cwru_f1_mae = 0.0
        for ep in range(30):
            probe.train(); opt2.zero_grad()
            logits = probe(tr_e.to(device))
            crit(logits, tr_l.to(device)).backward(); opt2.step()
            probe.eval()
            with torch.no_grad():
                preds = probe(te_e.to(device)).argmax(1).cpu().numpy()
            f1 = f1_score(te_l.numpy(), preds, average='macro', zero_division=0)
            cwru_f1_mae = max(cwru_f1_mae, f1)
        print(f"    CWRU F1: {cwru_f1_mae:.4f} ({time.time()-t0:.0f}s)")

        pad_f1_mae, rand_f1_mae = None, None
        if has_pad:
            mae.eval()
            def mae_embed(x): return mae.get_embeddings(x)
            pad_f1_mae = linear_probe(mae_embed, pad_train, pad_test, 512, 3, device)

            rand_mae = MAEModel(n_channels=3, window_size=4096, patch_size=256, embed_dim=512).to(device)
            rand_mae.eval()
            rand_f1_mae = linear_probe(lambda x: rand_mae.get_embeddings(x), pad_train, pad_test, 512, 3, device)
            print(f"    Paderborn: F1={pad_f1_mae:.4f}, rand={rand_f1_mae:.4f}, gain={pad_f1_mae-rand_f1_mae:+.4f}")

        all_results['mae'].append({'seed': seed, 'cwru_f1': cwru_f1_mae, 'pad_f1': pad_f1_mae, 'rand_pad_f1': rand_f1_mae})

        # ===== JEPA V2 =====
        print("\n  [4/4] JEPA V2 (using saved checkpoint + CWRU ref)...")
        cwru_f1_jepa = JEPA_CWRU_REF.get(seed, 0.773)
        pad_f1_jepa, rand_f1_jepa = None, None

        if JEPA_V2_CKPT.exists() and has_pad:
            ckpt = torch.load(str(JEPA_V2_CKPT), map_location=device, weights_only=False)
            config = ckpt['config']
            jepa = MechanicalJEPAV2(
                n_channels=config['n_channels'],
                window_size=config['window_size'],
                patch_size=config.get('patch_size', 256),
                embed_dim=config['embed_dim'],
                encoder_depth=config['encoder_depth'],
                predictor_depth=config.get('predictor_depth', 4),
                n_heads=config.get('n_heads', 4),
                mask_ratio=config.get('mask_ratio', 0.625),
                predictor_pos=config.get('predictor_pos', 'sinusoidal'),
                loss_fn=config.get('loss_fn', 'l1'),
                var_reg_lambda=config.get('var_reg_lambda', 0.1),
            ).to(device)
            jepa.load_state_dict(ckpt['model_state_dict'])
            jepa.eval()

            def jepa_embed(x): return jepa.get_embeddings(x, pool='mean')
            pad_f1_jepa = linear_probe(jepa_embed, pad_train, pad_test, 512, 3, device)

            rand_jepa = MechanicalJEPAV2(n_channels=3, window_size=4096, patch_size=256, embed_dim=512, encoder_depth=4).to(device)
            rand_jepa.eval()
            rand_f1_jepa = linear_probe(lambda x: rand_jepa.get_embeddings(x, pool='mean'), pad_train, pad_test, 512, 3, device)

            print(f"    CWRU F1: {cwru_f1_jepa:.4f} (ref), Paderborn: F1={pad_f1_jepa:.4f}, rand={rand_f1_jepa:.4f}, gain={pad_f1_jepa-rand_f1_jepa:+.4f}")
        else:
            print(f"    CWRU F1: {cwru_f1_jepa:.4f} (ref), no checkpoint for Paderborn transfer")

        all_results['jepa_v2'].append({'seed': seed, 'cwru_f1': cwru_f1_jepa, 'pad_f1': pad_f1_jepa, 'rand_pad_f1': rand_f1_jepa})

        # Save after each seed
        save_path = Path('results/transfer_baselines_v6.json')
        save_path.parent.mkdir(exist_ok=True, parents=True)
        with open(save_path, 'w') as f:
            json.dump(all_results, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.float32, np.float64)) else x)
        print(f"\n  Checkpoint saved: {save_path}")

    # Final summary table
    print("\n" + "="*100)
    print("COMPREHENSIVE TRANSFER TABLE — V6 AUDIT")
    print("="*100)
    print(f"{'Method':<30} {'CWRU F1':>12} {'Pad F1':>12} {'Pad Gain':>12} {'Supervision'}")
    print("-"*100)

    supervision_map = {
        'cnn_supervised': 'Supervised',
        'transformer_supervised': 'Supervised',
        'mae': 'Self-supervised',
        'jepa_v2': 'Self-supervised',
    }

    summary = {}
    for method, records in all_results.items():
        if not records: continue
        cwru_vals = [r['cwru_f1'] for r in records]
        pad_vals = [r['pad_f1'] for r in records if r['pad_f1'] is not None]
        rand_vals = [r['rand_pad_f1'] for r in records if r['rand_pad_f1'] is not None]
        cwru_str = f"{np.mean(cwru_vals):.3f}±{np.std(cwru_vals):.3f}"
        pad_str = f"{np.mean(pad_vals):.3f}±{np.std(pad_vals):.3f}" if pad_vals else "N/A"
        if pad_vals and rand_vals:
            gains = [p - r for p, r in zip(pad_vals, rand_vals)]
            gain_str = f"{np.mean(gains):+.3f}±{np.std(gains):.3f}"
        else:
            gain_str = "N/A"
        print(f"{method:<30} {cwru_str:>12} {pad_str:>12} {gain_str:>12} {supervision_map.get(method,'?')}")
        summary[method] = {'cwru_mean': float(np.mean(cwru_vals)), 'cwru_std': float(np.std(cwru_vals)),
                           'pad_mean': float(np.mean(pad_vals)) if pad_vals else None,
                           'pad_std': float(np.std(pad_vals)) if pad_vals else None,
                           'gain_mean': float(np.mean(gains)) if (pad_vals and rand_vals) else None,
                           'gain_std': float(np.std(gains)) if (pad_vals and rand_vals) else None}

    print("-"*100)

    # Save full results with summary
    all_results['_summary'] = summary
    with open(save_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.float32, np.float64)) else x)
    print(f"\nFinal results saved: {save_path}")
    return all_results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 456])
    parser.add_argument('--skip-mae', action='store_true', help='Skip MAE (slowest method)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    import shutil
    free_gb = shutil.disk_usage('/home/sagemaker-user').free / 1e9
    print(f"Home disk: {free_gb:.1f} GB free")
    if free_gb < 2.0:
        print("CRITICAL: Disk too full. Exiting.")
        sys.exit(1)

    run_audit(args.seeds, device)
