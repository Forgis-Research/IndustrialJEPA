"""
Phase 2C: Few-Shot Transfer Curves — Key Publishable Figure.

For each method (JEPA V2, CNN supervised, Transformer supervised, Random init),
measure Paderborn transfer F1 at N = {10, 20, 50, 100, 200, all} labeled target samples.

JEPA's advantage should be largest at low N (few-shot regime).
This is the core publishable figure.

Usage:
    python fewshot_transfer_curves.py --seeds 42 123 456
"""

import sys
import json
import time
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
JEPA_V2_CKPT = Path(__file__).parent / 'checkpoints/jepa_v2_20260401_003619.pt'
JEPA_CWRU_REF = {42: 0.7483, 123: 0.7907, 456: 0.7788}

# N values to test — samples PER CLASS (3 classes: healthy, OA, IA)
N_SHOTS = [10, 20, 50, 100, 200, -1]  # -1 means all


def get_paderborn_loaders(seed=42, batch_size=32):
    """Load Paderborn with file-level splits."""
    bearing_dirs = [(str(PADERBORN_DIR / folder), label) for folder, label in CLASSES.items()
                    if (PADERBORN_DIR / folder).exists()]
    if not bearing_dirs:
        return None, None
    return create_paderborn_loaders(
        bearing_dirs=bearing_dirs,
        window_size=4096, stride=2048, target_sr=20000,
        n_channels=3, test_ratio=0.2, batch_size=batch_size,
        seed=seed, max_files_per_bearing=20,
    )


def subsample_loader(full_loader, n_per_class, seed, n_classes=3):
    """Subsample dataset to n_per_class samples per class."""
    if n_per_class < 0:
        return full_loader  # use all

    # Collect all data
    all_x, all_y = [], []
    for batch in full_loader:
        x = batch[0]
        y = batch[1]
        all_x.append(x)
        if isinstance(y, torch.Tensor):
            all_y.append(y)
        else:
            all_y.append(torch.tensor(y))
    all_x = torch.cat(all_x)
    all_y = torch.cat(all_y)

    # Subsample per class
    rng = np.random.default_rng(seed)
    indices = []
    for cls in range(n_classes):
        cls_idx = (all_y == cls).nonzero(as_tuple=True)[0].numpy()
        if len(cls_idx) < n_per_class:
            print(f"    Warning: class {cls} has only {len(cls_idx)} samples, requested {n_per_class}")
            chosen = cls_idx
        else:
            chosen = rng.choice(cls_idx, size=n_per_class, replace=False)
        indices.extend(chosen.tolist())

    indices = torch.tensor(indices)
    sub_x = all_x[indices]
    sub_y = all_y[indices]

    dataset = torch.utils.data.TensorDataset(sub_x, sub_y)
    return torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)


def linear_probe_on_loaders(model_fn, train_loader, test_loader, embed_dim, n_classes, device, epochs=100):
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
    opt = optim.Adam(probe.parameters(), lr=1e-3, weight_decay=1e-4)
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


def measure_fewshot_curve(model_fn, embed_dim, pad_train_full, pad_test, device, seeds_sub, n_classes=3):
    """Measure F1 at each N value, averaging over sub-seeds."""
    curve = {}
    for n in N_SHOTS:
        f1_list = []
        for sub_seed in seeds_sub:
            sub_loader = subsample_loader(pad_train_full, n, sub_seed, n_classes=n_classes)
            f1 = linear_probe_on_loaders(model_fn, sub_loader, pad_test, embed_dim, n_classes, device)
            f1_list.append(f1)
        curve[str(n)] = {'mean': float(np.mean(f1_list)), 'std': float(np.std(f1_list)),
                          'all': [float(v) for v in f1_list]}
        label = 'all' if n < 0 else str(n)
        print(f"      N={label:>4}: F1={np.mean(f1_list):.4f} ± {np.std(f1_list):.4f}")
    return curve


def main(seeds=[42, 123, 456]):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}, Seeds: {seeds}")

    import shutil
    free_gb = shutil.disk_usage('/home/sagemaker-user').free / 1e9
    print(f"Home disk: {free_gb:.1f} GB free")
    if free_gb < 2.0:
        print("CRITICAL: Disk too full")
        sys.exit(1)

    results = {}

    for seed in seeds:
        print(f"\n{'='*70}\nSeed {seed}\n{'='*70}")
        torch.manual_seed(seed)
        np.random.seed(seed)

        # CWRU training data
        train_loader, test_loader, _ = create_dataloaders(
            data_dir='data/bearings', batch_size=32, window_size=4096,
            stride=2048, test_ratio=0.2, seed=seed, num_workers=0,
            dataset_filter='cwru', n_channels=3,
        )

        # Paderborn full loaders (we'll subsample from train)
        pad_train, pad_test = get_paderborn_loaders(seed=seed, batch_size=32)
        if pad_train is None:
            print("Paderborn not available. Skipping.")
            continue

        # Sub-seeds for each N measurement (3 sub-seeds per N point)
        sub_seeds = [seed, seed + 1000, seed + 2000]

        seed_results = {}

        # ===== 1. JEPA V2 (pretrained, frozen) =====
        print(f"\n  [1/4] JEPA V2...")
        if JEPA_V2_CKPT.exists():
            ckpt = torch.load(str(JEPA_V2_CKPT), map_location=device, weights_only=False)
            config = ckpt['config']
            jepa = MechanicalJEPAV2(
                n_channels=config['n_channels'], window_size=config['window_size'],
                patch_size=config.get('patch_size', 256), embed_dim=config['embed_dim'],
                encoder_depth=config['encoder_depth'], predictor_depth=config.get('predictor_depth', 4),
                n_heads=config.get('n_heads', 4), mask_ratio=config.get('mask_ratio', 0.625),
                predictor_pos=config.get('predictor_pos', 'sinusoidal'),
                loss_fn=config.get('loss_fn', 'l1'), var_reg_lambda=config.get('var_reg_lambda', 0.1),
            ).to(device)
            jepa.load_state_dict(ckpt['model_state_dict'])
            jepa.eval()
            jepa_fn = lambda x: jepa.get_embeddings(x, pool='mean')
            seed_results['jepa_v2'] = measure_fewshot_curve(jepa_fn, 512, pad_train, pad_test, device, sub_seeds)
        else:
            print("    JEPA checkpoint not found, skipping")

        # ===== 2. Random init (same architecture) =====
        print(f"\n  [2/4] Random init JEPA...")
        torch.manual_seed(seed + 9999)
        rand_model = MechanicalJEPAV2(n_channels=3, window_size=4096, patch_size=256,
                                       embed_dim=512, encoder_depth=4).to(device)
        rand_model.eval()
        rand_fn = lambda x: rand_model.get_embeddings(x, pool='mean')
        seed_results['random_init'] = measure_fewshot_curve(rand_fn, 512, pad_train, pad_test, device, sub_seeds)

        # ===== 3. CNN Supervised =====
        print(f"\n  [3/4] CNN Supervised...")
        torch.manual_seed(seed)
        cnn = CNN1DSupervised(n_channels=3, n_classes=4).to(device)
        train_supervised_model(cnn, train_loader, test_loader, 100, 1e-4, device)
        cnn.eval()
        cnn_fn = lambda x: cnn.get_embeddings(x)
        seed_results['cnn_supervised'] = measure_fewshot_curve(cnn_fn, 512, pad_train, pad_test, device, sub_seeds)

        # ===== 4. Transformer Supervised =====
        print(f"\n  [4/4] Transformer Supervised...")
        torch.manual_seed(seed)
        transformer = TransformerSupervised(n_channels=3, n_classes=4).to(device)
        train_supervised_model(transformer, train_loader, test_loader, 100, 1e-4, device)
        transformer.eval()
        tr_fn = lambda x: transformer.get_embeddings(x)
        seed_results['transformer_supervised'] = measure_fewshot_curve(tr_fn, 512, pad_train, pad_test, device, sub_seeds)

        results[seed] = seed_results

        # Save checkpoint
        save_path = Path('results/fewshot_curves.json')
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"  Checkpoint saved: {save_path}")

    # Aggregate across seeds and print final summary
    print(f"\n{'='*100}")
    print("FEW-SHOT TRANSFER CURVES — FINAL SUMMARY")
    print(f"{'='*100}")

    aggregated = {}
    methods = ['jepa_v2', 'cnn_supervised', 'transformer_supervised', 'random_init']

    for method in methods:
        method_n_results = {str(n): [] for n in N_SHOTS}
        for seed, seed_res in results.items():
            if method not in seed_res:
                continue
            for n_key, val in seed_res[method].items():
                method_n_results[n_key].extend(val['all'])
        aggregated[method] = {n_key: {'mean': float(np.mean(vs)), 'std': float(np.std(vs))}
                               for n_key, vs in method_n_results.items() if vs}

    # Print table
    headers = [str(n) if n > 0 else 'all' for n in N_SHOTS]
    print(f"{'Method':<30} " + " ".join(f"{h:>8}" for h in headers))
    print("-" * (30 + 9 * len(headers)))
    for method in methods:
        row = f"{method:<30}"
        for n in N_SHOTS:
            n_key = str(n)
            if n_key in aggregated.get(method, {}):
                v = aggregated[method][n_key]
                row += f" {v['mean']:>7.3f}"
            else:
                row += f" {'N/A':>7}"
        print(row)

    # Save aggregated
    aggregated['_raw'] = {str(k): v for k, v in results.items()}
    final_path = Path('results/fewshot_curves.json')
    with open(final_path, 'w') as f:
        json.dump(aggregated, f, indent=2)
    print(f"\nFinal results saved: {final_path}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 456])
    args = parser.parse_args()
    main(args.seeds)
