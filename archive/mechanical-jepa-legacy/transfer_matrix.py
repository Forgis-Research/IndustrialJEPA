"""
Complete Transfer Matrix Experiment (Round 6).

Builds the full transfer matrix: for each source->target pair,
reports JEPA gain vs random init (3 seeds, linear probe).

Sources: CWRU, Paderborn
Targets: CWRU, Paderborn, IMS

For each pair:
- Pretrain JEPA on source (or use existing checkpoint)
- Evaluate linear probe on target
- Report gain over random init

All Paderborn data resampled to 12kHz for compatibility.
IMS data loaded at native 20kHz (previous experiments showed this works).

Note: CWRU->IMS already done in Exp 18 (+8.8%).
      CWRU->Paderborn@12kHz done in Exp 24 (+8.5%).
      CWRU->Paderborn@20kHz done in Exp 25 (+14.7%).
      IMS->IMS done in Exp 20 (+6.2%).

This experiment fills in:
      Paderborn->CWRU: Use Paderborn-pretrained encoder on CWRU
      Paderborn->IMS: Use Paderborn-pretrained encoder on IMS
      IMS->CWRU: Use IMS-pretrained encoder (new ckpt) on CWRU
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import scipy.signal
import scipy.io
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import math

sys.path.insert(0, str(Path(__file__).parent))
from src.data import create_dataloaders
from src.models.jepa_v2 import MechanicalJEPAV2

PADERBORN_DIR = Path('/home/sagemaker-user/IndustrialJEPA/datasets/data/paderborn')
IMS_DIR = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/data/bearings/raw/ims')


def resample_poly_1d(sig, src_sr, tgt_sr):
    from math import gcd
    if src_sr == tgt_sr:
        return sig
    g = gcd(src_sr, tgt_sr)
    return scipy.signal.resample_poly(sig, tgt_sr//g, src_sr//g).astype(np.float32)


def load_paderborn_vib(mat_path):
    mat = scipy.io.loadmat(str(mat_path), squeeze_me=True, simplify_cells=True)
    key = [k for k in mat.keys() if not k.startswith('_')][0]
    return mat[key]['Y'][6]['Data'].astype(np.float32)


class PaderbornWindowDataset(Dataset):
    """Paderborn at 12kHz (downsampled from 64kHz)."""

    def __init__(self, window_size=4096, stride=2048, target_sr=12000,
                 n_channels=3, max_files=30, seed=42, test_only=False):
        rng = np.random.default_rng(seed)
        self.window_size = window_size
        self.n_channels = n_channels
        self.windows = []

        bearings = [
            (PADERBORN_DIR / 'K001', 0),
            (PADERBORN_DIR / 'KA01', 1),
            (PADERBORN_DIR / 'KI01', 2),
        ]

        for bearing_path, label in bearings:
            files = sorted(bearing_path.glob('*.mat'))
            # Split: use first 80% for train, last 20% for test
            n = len(files)
            n_test = max(1, n // 5)
            if test_only:
                use_files = files[:n_test]
            else:
                use_files = files[n_test:n_test + max_files]

            for f in use_files:
                try:
                    raw = load_paderborn_vib(f)
                    sig = resample_poly_1d(raw, 64000, target_sr)
                    n_wins = (len(sig) - window_size) // stride + 1
                    for i in range(n_wins):
                        self.windows.append((sig, i * stride, label))
                except:
                    continue

        print(f"  PaderbornDataset({'test' if test_only else 'train'}): "
              f"{len(self.windows)} windows")

    def __len__(self): return len(self.windows)

    def __getitem__(self, idx):
        sig, start, label = self.windows[idx]
        w = sig[start:start + self.window_size]
        wmc = np.stack([w] * self.n_channels, axis=0)
        mean = wmc.mean(1, keepdims=True)
        std = wmc.std(1, keepdims=True) + 1e-8
        return torch.tensor((wmc - mean) / std, dtype=torch.float32), label, 'paderborn'


def cosine_scheduler(base, final, epochs, warmup):
    s1 = np.linspace(0, base, warmup)
    iters = np.arange(epochs - warmup)
    s2 = final + 0.5 * (base - final) * (1 + np.cos(math.pi * iters / len(iters)))
    return np.concatenate([s1, s2])


def train_jepa(loader, model, epochs, device):
    """Train JEPA on loader, return trained model."""
    opt = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)
    sched = cosine_scheduler(1e-4, 1e-6, epochs, 5)

    for ep in range(epochs):
        model.train()
        total = 0
        for batch in loader:
            signals = batch[0].to(device)
            lr = sched[min(ep, len(sched)-1)]
            for pg in opt.param_groups:
                pg['lr'] = lr
            opt.zero_grad()
            loss = model.train_step(signals)
            loss.backward()
            opt.step()
            model.update_ema()
            total += loss.item()
        avg = total / len(loader)
        if (ep+1) % 20 == 0 or ep == 0:
            print(f"    Epoch {ep+1}/{epochs}: loss={avg:.4f}")
    return model


def evaluate_probe(model, train_loader, test_loader, embed_dim, device,
                   n_classes=4, probe_epochs=50, label=''):
    """Linear probe on frozen encoder."""
    model.eval()

    def extract(ldr):
        all_e, all_l = [], []
        with torch.no_grad():
            for batch in ldr:
                sig = batch[0].to(device)
                all_e.append(model.get_embeddings(sig, pool='mean').cpu())
                all_l.append(batch[1])
        return torch.cat(all_e), torch.cat(all_l)

    te, tl = extract(train_loader)
    ee, el = extract(test_loader)

    m = te.mean(0, keepdim=True)
    s = te.std(0, keepdim=True) + 1e-8
    te_n = ((te - m) / s).to(device)
    ee_n = ((ee - m) / s).to(device)
    tl_d = tl.to(device)
    el_d = el.to(device)

    probe = nn.Linear(embed_dim, n_classes).to(device)
    opt = optim.AdamW(probe.parameters(), lr=1e-2, weight_decay=0.01)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, probe_epochs, 1e-5)
    crit = nn.CrossEntropyLoss()

    best = 0
    for ep in range(probe_epochs):
        probe.train()
        opt.zero_grad()
        crit(probe(te_n), tl_d).backward()
        opt.step()
        sch.step()
        probe.eval()
        with torch.no_grad():
            acc = (probe(ee_n).argmax(1) == el_d).float().mean().item()
        if acc > best:
            best = acc

    print(f"  [{label:35s}]: {best:.4f}")
    return best


def make_model(device):
    return MechanicalJEPAV2(
        n_channels=3, window_size=4096, patch_size=256,
        embed_dim=512, encoder_depth=4, predictor_depth=4, n_heads=4,
        mask_ratio=0.625, ema_decay=0.996,
        predictor_pos='sinusoidal', loss_fn='l1', var_reg_lambda=0.1,
    ).to(device)


def run_paderborn_to_cwru(device, seeds=[42, 123, 456]):
    """
    Paderborn -> CWRU transfer.
    Train JEPA on Paderborn, evaluate on CWRU.
    """
    print("\n" + "="*60)
    print("Paderborn -> CWRU Transfer")
    print("="*60)

    all_gains = []

    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)

        # CWRU eval data
        cwru_train, cwru_test, info = create_dataloaders(
            data_dir='data/bearings', batch_size=32,
            window_size=4096, stride=2048, test_ratio=0.2, seed=seed,
            num_workers=0, dataset_filter='cwru', n_channels=3,
        )

        # Paderborn train data
        pad_train = PaderbornWindowDataset(
            window_size=4096, stride=2048, target_sr=12000,
            n_channels=3, max_files=30, seed=seed, test_only=False,
        )
        pad_loader = DataLoader(pad_train, batch_size=32, shuffle=True, num_workers=0)

        # Train JEPA on Paderborn
        print(f"\nSeed {seed}: Training JEPA on Paderborn (50 epochs)...")
        model = make_model(device)
        model = train_jepa(pad_loader, model, epochs=50, device=device)

        print(f"Evaluating on CWRU (4-class):")
        jepa_acc = evaluate_probe(model, cwru_train, cwru_test, 512, device,
                                  n_classes=4, probe_epochs=30, label='Paderborn->CWRU JEPA')
        rand_model = make_model(device)
        rand_acc = evaluate_probe(rand_model, cwru_train, cwru_test, 512, device,
                                  n_classes=4, probe_epochs=30, label='Random init')

        gain = jepa_acc - rand_acc
        all_gains.append(gain)
        print(f"  Transfer gain: {gain:+.4f}")

    print(f"\nPaderborn->CWRU Transfer: {np.mean(all_gains):+.4f} ± {np.std(all_gains):.4f}")
    return np.mean(all_gains), np.std(all_gains)


def run_ims_to_cwru(device, ims_ckpt_path, seeds=[42, 123, 456]):
    """
    IMS -> CWRU transfer.
    Use IMS-pretrained encoder to evaluate on CWRU 4-class fault classification.
    """
    print("\n" + "="*60)
    print("IMS -> CWRU Transfer")
    print("="*60)

    all_gains = []

    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)

        # CWRU eval data
        cwru_train, cwru_test, info = create_dataloaders(
            data_dir='data/bearings', batch_size=32,
            window_size=4096, stride=2048, test_ratio=0.2, seed=seed,
            num_workers=0, dataset_filter='cwru', n_channels=3,
        )

        # Load IMS-pretrained model
        ckpt = torch.load(ims_ckpt_path, map_location=device, weights_only=False)
        config = ckpt['config']
        model = make_model(device)
        model.load_state_dict(ckpt['model_state_dict'])

        print(f"\nSeed {seed}:")
        jepa_acc = evaluate_probe(model, cwru_train, cwru_test, 512, device,
                                  n_classes=4, probe_epochs=30, label='IMS->CWRU JEPA')
        rand_model = make_model(device)
        rand_acc = evaluate_probe(rand_model, cwru_train, cwru_test, 512, device,
                                  n_classes=4, probe_epochs=30, label='Random init')

        gain = jepa_acc - rand_acc
        all_gains.append(gain)
        print(f"  Transfer gain: {gain:+.4f}")

    print(f"\nIMS->CWRU Transfer: {np.mean(all_gains):+.4f} ± {np.std(all_gains):.4f}")
    return np.mean(all_gains), np.std(all_gains)


def run_paderborn_to_ims(device, seeds=[42, 123, 456]):
    """
    Paderborn -> IMS transfer.
    Train JEPA on Paderborn @ 12kHz, evaluate on IMS binary degradation.
    """
    print("\n" + "="*60)
    print("Paderborn -> IMS Transfer")
    print("="*60)

    # Use IMS transfer infrastructure
    sys.path.insert(0, '.')
    from ims_transfer import IMSDegradationDataset, create_ims_splits, evaluate_probe as ims_eval_probe

    all_gains = []

    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Paderborn train data
        pad_train = PaderbornWindowDataset(
            window_size=4096, stride=2048, target_sr=12000,
            n_channels=3, max_files=30, seed=seed, test_only=False,
        )
        pad_loader = DataLoader(pad_train, batch_size=32, shuffle=True, num_workers=0)

        # Train JEPA on Paderborn
        print(f"\nSeed {seed}: Training JEPA on Paderborn...")
        model = make_model(device)
        model = train_jepa(pad_loader, model, epochs=50, device=device)

        # IMS evaluation (binary)
        ims_dir = IMS_DIR
        train_idx, train_labels, test_idx, test_labels, _ = create_ims_splits(
            ims_dir, '1st_test', seed=seed, n_classes=2
        )
        train_ds = IMSDegradationDataset(
            ims_dir=ims_dir, test_set='1st_test',
            file_indices=train_idx, labels=train_labels,
            window_size=4096, n_channels=3, windows_per_file=4, seed=seed,
        )
        test_ds = IMSDegradationDataset(
            ims_dir=ims_dir, test_set='1st_test',
            file_indices=test_idx, labels=test_labels,
            window_size=4096, n_channels=3, windows_per_file=4, seed=seed,
        )
        train_ldr = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0)
        test_ldr = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0)

        jepa_res = ims_eval_probe(model, train_ldr, test_ldr, 512, device,
                                  probe_type='linear', probe_epochs=50, n_classes=2,
                                  pool='mean', label='Paderborn->IMS JEPA')
        rand_model = make_model(device)
        rand_res = ims_eval_probe(rand_model, train_ldr, test_ldr, 512, device,
                                  probe_type='linear', probe_epochs=50, n_classes=2,
                                  pool='mean', label='Random init')

        gain = jepa_res['test_acc'] - rand_res['test_acc']
        all_gains.append(gain)
        print(f"  Transfer gain: {gain:+.4f}")

    print(f"\nPaderborn->IMS Transfer: {np.mean(all_gains):+.4f} ± {np.std(all_gains):.4f}")
    return np.mean(all_gains), np.std(all_gains)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 123, 456])
    parser.add_argument('--ims-ckpt', default='checkpoints/jepa_v2_20260401_113314.pt',
                        help='IMS-pretrained checkpoint for IMS->CWRU')
    parser.add_argument('--experiments', nargs='+', default=['pad_cwru', 'ims_cwru', 'pad_ims'],
                        help='Which transfer directions to run')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    results = {}

    if 'pad_cwru' in args.experiments:
        results['pad_cwru'] = run_paderborn_to_cwru(device, args.seeds)

    if 'ims_cwru' in args.experiments:
        results['ims_cwru'] = run_ims_to_cwru(device, args.ims_ckpt, args.seeds)

    if 'pad_ims' in args.experiments:
        results['pad_ims'] = run_paderborn_to_ims(device, args.seeds)

    # Complete transfer matrix summary
    print("\n" + "="*70)
    print("COMPLETE TRANSFER MATRIX SUMMARY")
    print("="*70)
    print("\nPreviously measured (from Experiments 18, 20, 24, 25):")
    print("  CWRU->IMS binary @ 20kHz:       +8.8% ± 0.7% (3/3 seeds)")
    print("  IMS->IMS @ 20kHz:               +6.2% ± 1.7% (3/3 seeds)")
    print("  CWRU->Paderborn @ 12kHz:        +8.5% ± 3.0% (3/3 seeds)")
    print("  CWRU->Paderborn @ 20kHz:        +14.7% ± 0.8% (3/3 seeds)")
    print()
    print("New measurements:")
    for key, (mean, std) in results.items():
        print(f"  {key:25s}: {mean:+.4f} ± {std:.4f}")


if __name__ == '__main__':
    main()
