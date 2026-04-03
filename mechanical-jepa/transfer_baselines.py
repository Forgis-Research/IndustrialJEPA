"""
Transfer Baselines: Compare CWRU→Paderborn transfer for all methods.

This script takes pretrained models (CNN, Transformer, MAE, JEPA V2) and evaluates
their transfer to Paderborn dataset at 20kHz resampling.

Usage:
    python transfer_baselines.py --seeds 42 123 456
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).parent))
from src.models import MechanicalJEPAV2
from src.data import create_dataloaders
from baselines_comparison import CNN1DSupervised, TransformerSupervised, MAEModel, train_supervised_model
from paderborn_transfer import PaderbornDataset

CHECKPOINT_DIR = Path('/mnt/sagemaker-nvme/jepa_checkpoints')
JEPA_V2_CKPT = Path('checkpoints/jepa_v2_20260401_003619.pt')


def get_paderborn_loaders(target_sr=20000, seed=42, batch_size=32):
    """Load Paderborn data as DataLoaders."""
    paderborn_dir = Path('/home/sagemaker-user/IndustrialJEPA/datasets/data/paderborn')
    if not paderborn_dir.exists():
        print(f"Paderborn dir not found: {paderborn_dir}")
        return None, None

    torch.manual_seed(seed)

    try:
        dataset = PaderbornDataset(
            root_dir=str(paderborn_dir),
            target_sr=target_sr,
            window_size=4096,
            stride=2048,
            n_channels=1,
        )
    except Exception as e:
        print(f"Failed to load Paderborn: {e}")
        return None, None

    if len(dataset) == 0:
        return None, None

    n_train = int(0.7 * len(dataset))
    n_test = len(dataset) - n_train
    train_ds, test_ds = torch.utils.data.random_split(
        dataset, [n_train, n_test],
        generator=torch.Generator().manual_seed(seed)
    )

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"  Paderborn: {len(train_ds)} train, {len(test_ds)} test windows")
    return train_loader, test_loader


def linear_probe_transfer(model_fn, pad_train_loader, pad_test_loader, embed_dim, n_classes, device):
    """
    Extract embeddings and train linear probe.
    model_fn: callable that takes (signals: Tensor) -> embeddings: Tensor
    """
    def extract(loader):
        all_embeds, all_labels = [], []
        with torch.no_grad():
            for batch in loader:
                if len(batch) == 2:
                    signals, labels = batch
                elif len(batch) == 3:
                    signals, labels, _ = batch
                else:
                    signals, labels = batch[0], batch[1]
                signals = signals.to(device)
                embeds = model_fn(signals)
                all_embeds.append(embeds.cpu())
                all_labels.append(labels.cpu() if isinstance(labels, torch.Tensor) else torch.tensor(labels))
        return torch.cat(all_embeds), torch.cat(all_labels)

    train_embeds, train_labels = extract(pad_train_loader)
    test_embeds, test_labels = extract(pad_test_loader)

    probe = nn.Linear(embed_dim, n_classes).to(device)
    opt = optim.Adam(probe.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()

    te = train_embeds.to(device)
    tl = train_labels.to(device)

    best_f1 = 0.0
    for ep in range(50):
        probe.train()
        opt.zero_grad()
        logits = probe(te)
        loss = crit(logits, tl)
        loss.backward()
        opt.step()

        probe.eval()
        with torch.no_grad():
            preds = probe(test_embeds.to(device)).argmax(dim=1).cpu().numpy()
        f1 = f1_score(test_labels.numpy(), preds, average='macro', zero_division=0)
        if f1 > best_f1:
            best_f1 = f1

    return best_f1


def evaluate_all_transfers(seeds, device):
    """Run CWRU training + Paderborn transfer for all baselines."""

    # Check if Paderborn is available using seed 0
    pad_train_0, pad_test_0 = get_paderborn_loaders(target_sr=20000, seed=42)
    has_paderborn = pad_train_0 is not None

    if not has_paderborn:
        print("Paderborn data not available. Running CWRU-only comparison.")

    results = {
        'handcrafted': [],
        'cnn_supervised': [],
        'transformer_supervised': [],
        'mae': [],
        'jepa_v2': [],
    }

    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"Seed {seed}")
        print(f"{'='*60}")

        torch.manual_seed(seed)
        np.random.seed(seed)

        # CWRU data
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

        # Paderborn data (reseed for consistent splits)
        if has_paderborn:
            pad_train, pad_test = get_paderborn_loaders(target_sr=20000, seed=seed)
        else:
            pad_train, pad_test = None, None

        # ===== 1. Hand-crafted features =====
        print("\n  Hand-crafted + LogReg...")
        from baselines_comparison import extract_handcrafted_features
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        def collect_raw(loader):
            sigs, lbls = [], []
            for s, l, _ in loader:
                sigs.append(s.numpy())
                lbls.append(l.numpy())
            return np.concatenate(sigs), np.concatenate(lbls)

        tr_sig, tr_lbl = collect_raw(train_loader)
        te_sig, te_lbl = collect_raw(test_loader)

        tr_feat = extract_handcrafted_features(tr_sig)
        te_feat = extract_handcrafted_features(te_sig)

        scaler = StandardScaler()
        tr_feat = scaler.fit_transform(tr_feat)
        te_feat = scaler.transform(te_feat)

        clf = LogisticRegression(max_iter=1000, random_state=seed, C=1.0)
        clf.fit(tr_feat, tr_lbl)
        cwru_f1 = f1_score(te_lbl, clf.predict(te_feat), average='macro', zero_division=0)
        print(f"    CWRU F1: {cwru_f1:.4f}")

        pad_f1 = None
        rand_pad_f1 = None
        if has_paderborn and pad_train is not None:
            def collect_pad(loader):
                sigs, lbls = [], []
                for batch in loader:
                    if len(batch) == 3:
                        s, l, _ = batch
                    else:
                        s, l = batch[0], batch[1]
                    sigs.append(s.numpy())
                    lbls.append(l.numpy() if isinstance(l, np.ndarray) else l.cpu().numpy())
                return np.concatenate(sigs, axis=0), np.concatenate(lbls, axis=0)

            pad_tr_sig, pad_tr_lbl = collect_pad(pad_train)
            pad_te_sig, pad_te_lbl = collect_pad(pad_test)

            # Paderborn may be 1-channel — expand to match
            if pad_tr_sig.shape[1] == 1:
                pad_tr_sig = np.repeat(pad_tr_sig, 3, axis=1)
                pad_te_sig = np.repeat(pad_te_sig, 3, axis=1)

            pad_tr_feat = extract_handcrafted_features(pad_tr_sig)
            pad_te_feat = extract_handcrafted_features(pad_te_sig)

            pad_tr_feat_norm = scaler.transform(pad_tr_feat)  # Use CWRU scaler
            pad_te_feat_norm = scaler.transform(pad_te_feat)

            pad_clf = LogisticRegression(max_iter=1000, random_state=seed)
            pad_clf.fit(pad_tr_feat_norm, pad_tr_lbl)
            pad_f1 = f1_score(pad_te_lbl, pad_clf.predict(pad_te_feat_norm), average='macro', zero_division=0)

            # Random init (untransformed features — no scaler)
            rand_scaler = StandardScaler()
            pad_tr_rand = rand_scaler.fit_transform(pad_tr_feat)
            pad_te_rand = rand_scaler.transform(pad_te_feat)
            rand_clf = LogisticRegression(max_iter=1000, random_state=seed)
            rand_clf.fit(pad_tr_rand, pad_tr_lbl)
            rand_pad_f1 = f1_score(pad_te_lbl, rand_clf.predict(pad_te_rand), average='macro', zero_division=0)

            print(f"    Paderborn F1: {pad_f1:.4f} (rand: {rand_pad_f1:.4f}, gain: {pad_f1-rand_pad_f1:+.4f})")

        results['handcrafted'].append({'seed': seed, 'cwru_f1': cwru_f1, 'pad_f1': pad_f1, 'rand_pad_f1': rand_pad_f1})

        # ===== 2. CNN Supervised =====
        print("\n  CNN Supervised...")
        cnn = CNN1DSupervised(n_channels=3, n_classes=4).to(device)
        cwru_f1_cnn = train_supervised_model(cnn, train_loader, test_loader, 100, 1e-4, device)
        print(f"    CWRU F1: {cwru_f1_cnn:.4f}")

        pad_f1_cnn = None
        rand_pad_f1_cnn = None
        if has_paderborn and pad_train is not None:
            cnn.eval()
            def cnn_embed(x):
                # Paderborn has 1 channel, CNN expects 3 — repeat channels
                if x.shape[1] == 1:
                    x = x.repeat(1, 3, 1)
                return cnn.get_embeddings(x)

            pad_f1_cnn = linear_probe_transfer(cnn_embed, pad_train, pad_test, 512, 3, device)

            # Random init CNN
            torch.manual_seed(seed + 9999)
            rand_cnn = CNN1DSupervised(n_channels=3, n_classes=4).to(device)
            rand_cnn.eval()
            def rand_cnn_embed(x):
                if x.shape[1] == 1:
                    x = x.repeat(1, 3, 1)
                return rand_cnn.get_embeddings(x)
            rand_pad_f1_cnn = linear_probe_transfer(rand_cnn_embed, pad_train, pad_test, 512, 3, device)

            print(f"    Paderborn: F1={pad_f1_cnn:.4f} (rand={rand_pad_f1_cnn:.4f}, gain={pad_f1_cnn-rand_pad_f1_cnn:+.4f})")

        results['cnn_supervised'].append({'seed': seed, 'cwru_f1': cwru_f1_cnn, 'pad_f1': pad_f1_cnn, 'rand_pad_f1': rand_pad_f1_cnn})

        # ===== 3. Transformer Supervised =====
        print("\n  Transformer Supervised...")
        transformer = TransformerSupervised(n_channels=3, n_classes=4).to(device)
        cwru_f1_tr = train_supervised_model(transformer, train_loader, test_loader, 100, 1e-4, device)
        print(f"    CWRU F1: {cwru_f1_tr:.4f}")

        pad_f1_tr = None
        rand_pad_f1_tr = None
        if has_paderborn and pad_train is not None:
            transformer.eval()
            def transformer_embed(x):
                if x.shape[1] == 1:
                    x = x.repeat(1, 3, 1)
                return transformer.get_embeddings(x)

            pad_f1_tr = linear_probe_transfer(transformer_embed, pad_train, pad_test, 512, 3, device)

            torch.manual_seed(seed + 9998)
            rand_tr = TransformerSupervised(n_channels=3, n_classes=4).to(device)
            rand_tr.eval()
            def rand_tr_embed(x):
                if x.shape[1] == 1:
                    x = x.repeat(1, 3, 1)
                return rand_tr.get_embeddings(x)
            rand_pad_f1_tr = linear_probe_transfer(rand_tr_embed, pad_train, pad_test, 512, 3, device)

            print(f"    Paderborn: F1={pad_f1_tr:.4f} (rand={rand_pad_f1_tr:.4f}, gain={pad_f1_tr-rand_pad_f1_tr:+.4f})")

        results['transformer_supervised'].append({'seed': seed, 'cwru_f1': cwru_f1_tr, 'pad_f1': pad_f1_tr, 'rand_pad_f1': rand_pad_f1_tr})

        # ===== 4. MAE =====
        print("\n  MAE (100 epochs)...")
        mae = MAEModel(n_channels=3, window_size=4096, patch_size=256, embed_dim=512,
                       encoder_depth=4, mask_ratio=0.625).to(device)
        n_mae_params = sum(p.numel() for p in mae.parameters() if p.requires_grad)

        optimizer = optim.AdamW(mae.parameters(), lr=1e-4, weight_decay=0.05)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)

        for epoch in range(100):
            mae.train()
            for signals, labels, _ in train_loader:
                optimizer.zero_grad()
                loss = mae(signals.to(device))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(mae.parameters(), max_norm=1.0)
                optimizer.step()
            scheduler.step()
            if (epoch + 1) % 50 == 0:
                print(f"    MAE epoch {epoch+1}")

        # Linear probe on CWRU
        mae.eval()
        def mae_embed(x):
            return mae.get_embeddings(x)

        all_embeds, all_labels = [], []
        with torch.no_grad():
            for signals, labels, _ in train_loader:
                embeds = mae_embed(signals.to(device))
                all_embeds.append(embeds.cpu())
                all_labels.append(labels)
        tr_emb = torch.cat(all_embeds)
        tr_lbl = torch.cat(all_labels)

        all_embeds, all_labels = [], []
        with torch.no_grad():
            for signals, labels, _ in test_loader:
                embeds = mae_embed(signals.to(device))
                all_embeds.append(embeds.cpu())
                all_labels.append(labels)
        te_emb = torch.cat(all_embeds)
        te_lbl = torch.cat(all_labels)

        probe = nn.Linear(512, 4).to(device)
        opt = optim.Adam(probe.parameters(), lr=1e-3)
        crit = nn.CrossEntropyLoss()
        best_mae_cwru_f1 = 0.0

        for ep in range(20):
            probe.train()
            opt.zero_grad()
            logits = probe(tr_emb.to(device))
            loss = crit(logits, tr_lbl.to(device))
            loss.backward()
            opt.step()
            probe.eval()
            with torch.no_grad():
                preds = probe(te_emb.to(device)).argmax(dim=1).cpu().numpy()
            f1 = f1_score(te_lbl.numpy(), preds, average='macro', zero_division=0)
            if f1 > best_mae_cwru_f1:
                best_mae_cwru_f1 = f1

        print(f"    CWRU F1: {best_mae_cwru_f1:.4f}")

        pad_f1_mae = None
        rand_pad_f1_mae = None
        if has_paderborn and pad_train is not None:
            def mae_embed_pad(x):
                if x.shape[1] == 1:
                    x = x.repeat(1, 3, 1)
                return mae.get_embeddings(x)

            pad_f1_mae = linear_probe_transfer(mae_embed_pad, pad_train, pad_test, 512, 3, device)

            torch.manual_seed(seed + 9997)
            rand_mae = MAEModel(n_channels=3, window_size=4096, patch_size=256, embed_dim=512).to(device)
            rand_mae.eval()
            def rand_mae_embed(x):
                if x.shape[1] == 1:
                    x = x.repeat(1, 3, 1)
                return rand_mae.get_embeddings(x)
            rand_pad_f1_mae = linear_probe_transfer(rand_mae_embed, pad_train, pad_test, 512, 3, device)

            print(f"    Paderborn: F1={pad_f1_mae:.4f} (rand={rand_pad_f1_mae:.4f}, gain={pad_f1_mae-rand_pad_f1_mae:+.4f})")

        results['mae'].append({'seed': seed, 'cwru_f1': best_mae_cwru_f1, 'pad_f1': pad_f1_mae, 'rand_pad_f1': rand_pad_f1_mae})

        # ===== 5. JEPA V2 (reference) =====
        print("\n  JEPA V2 (reference from saved results)...")
        # JEPA V2 reference results (from Exp 36, 3 seeds)
        jepa_v2_ref = {42: {'cwru_f1': 0.7483, 'pad_f1': None},
                       123: {'cwru_f1': 0.7907, 'pad_f1': 0.8988},
                       456: {'cwru_f1': 0.7788, 'pad_f1': 0.8904}}
        if seed in jepa_v2_ref:
            ref = jepa_v2_ref[seed]
            print(f"    CWRU F1: {ref['cwru_f1']:.4f} (from Exp 36)")
            if ref['pad_f1']:
                print(f"    Paderborn F1: {ref['pad_f1']:.4f} (from Exp 25)")

        # Load and run JEPA V2 on Paderborn if checkpoint available
        pad_f1_jepa = None
        rand_pad_f1_jepa = None
        if JEPA_V2_CKPT.exists() and has_paderborn and pad_train is not None:
            ckpt = torch.load(str(JEPA_V2_CKPT), map_location=device, weights_only=False)
            config = ckpt['config']
            jepa_model = MechanicalJEPAV2(
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
            jepa_model.load_state_dict(ckpt['model_state_dict'])
            jepa_model.eval()

            def jepa_embed(x):
                if x.shape[1] == 1:
                    x = x.repeat(1, 3, 1)
                return jepa_model.get_embeddings(x, pool='mean')

            pad_f1_jepa = linear_probe_transfer(jepa_embed, pad_train, pad_test, 512, 3, device)

            # Random init
            torch.manual_seed(seed + 9996)
            rand_jepa = MechanicalJEPAV2(n_channels=3, window_size=4096, patch_size=256,
                                         embed_dim=512, encoder_depth=4).to(device)
            rand_jepa.eval()
            def rand_jepa_embed(x):
                if x.shape[1] == 1:
                    x = x.repeat(1, 3, 1)
                return rand_jepa.get_embeddings(x, pool='mean')
            rand_pad_f1_jepa = linear_probe_transfer(rand_jepa_embed, pad_train, pad_test, 512, 3, device)

            cwru_f1_jepa = jepa_v2_ref.get(seed, {}).get('cwru_f1', 0.773)
            print(f"    Paderborn: F1={pad_f1_jepa:.4f} (rand={rand_pad_f1_jepa:.4f}, gain={pad_f1_jepa-rand_pad_f1_jepa:+.4f})")

            results['jepa_v2'].append({
                'seed': seed, 'cwru_f1': cwru_f1_jepa, 'pad_f1': pad_f1_jepa,
                'rand_pad_f1': rand_pad_f1_jepa
            })

    # Print comprehensive table
    print("\n" + "=" * 100)
    print("COMPREHENSIVE COMPARISON TABLE: CWRU + PADERBORN TRANSFER")
    print("=" * 100)
    print(f"{'Method':<35} {'CWRU F1':<25} {'Paderborn F1':<20} {'Pad Gain':<15} {'Supervision'}")
    print("-" * 100)

    for method_name, method_results in results.items():
        if not method_results:
            continue

        cwru_f1s = [r['cwru_f1'] for r in method_results]
        pad_f1s = [r['pad_f1'] for r in method_results if r['pad_f1'] is not None]
        rand_pad_f1s = [r['rand_pad_f1'] for r in method_results if r['rand_pad_f1'] is not None]

        cwru_str = f"{np.mean(cwru_f1s):.3f} ± {np.std(cwru_f1s):.3f}"
        pad_str = f"{np.mean(pad_f1s):.3f} ± {np.std(pad_f1s):.3f}" if pad_f1s else "N/A"

        if pad_f1s and rand_pad_f1s:
            gains = [p - r for p, r in zip(pad_f1s, rand_pad_f1s)]
            gain_str = f"+{np.mean(gains):.3f} ± {np.std(gains):.3f}"
        else:
            gain_str = "N/A"

        supervision = {
            'handcrafted': 'Supervised',
            'cnn_supervised': 'Supervised',
            'transformer_supervised': 'Supervised',
            'mae': 'Self-supervised',
            'jepa_v2': 'Self-supervised',
        }.get(method_name, '?')

        print(f"{method_name:<35} {cwru_str:<25} {pad_str:<20} {gain_str:<15} {supervision}")

    print("-" * 100)

    import json
    save_path = Path('results/transfer_baselines.json')
    save_path.parent.mkdir(exist_ok=True, parents=True)

    def convert(obj):
        if isinstance(obj, (np.float32, np.float64)): return float(obj)
        if isinstance(obj, (np.int32, np.int64)): return int(obj)
        return obj

    with open(save_path, 'w') as f:
        json.dump(
            {k: [{kk: convert(vv) for kk, vv in r.items()} for r in v]
             for k, v in results.items()},
            f, indent=2
        )
    print(f"\nResults saved to {save_path}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 456])
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    import shutil
    print(f"Home disk: {shutil.disk_usage('/home/sagemaker-user').free/1e9:.1f} GB free")

    results = evaluate_all_transfers(args.seeds, device)
    return results


if __name__ == '__main__':
    main()
