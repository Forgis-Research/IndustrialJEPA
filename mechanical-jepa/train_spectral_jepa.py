"""
Phase 3A: Spectral-Feature JEPA (SF-JEPA)

Novel objective: Multi-task JEPA that predicts BOTH:
  1. Latent embeddings of masked patches (standard JEPA)
  2. Spectral/statistical features of masked patches (new task)

Feature targets per patch (window_size=256 samples at 12kHz):
  - 4 spectral band energies (0-1.5kHz, 1.5-3kHz, 3-6kHz, 6-12kHz)
  - RMS, kurtosis, crest_factor, skewness (4 features)
  - Envelope RMS (1 feature)
  Total: 9 features per patch

Hypothesis: Adding physics-meaningful targets forces the predictor to
understand the signal structure that matters for fault detection, leading
to better representations for RUL and cross-domain transfer.

Evaluation:
  - CWRU classification (linear probe) — should not regress vs V2
  - Paderborn transfer gain — should improve
  - IMS health indicator Spearman correlation — key new metric

Usage:
    python train_spectral_jepa.py --epochs 100 --seed 42 --spec-weight 0.5
"""

import argparse
import copy
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import scipy.signal
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).parent))
from src.models import MechanicalJEPAV2
from src.data import create_dataloaders
from src.models.jepa_v2 import prediction_var_loss


# ============================================================================
# Spectral Feature Extraction (CPU, per patch)
# ============================================================================

def extract_spectral_features(patches: torch.Tensor, sample_rate: float = 12000.0) -> torch.Tensor:
    """
    Extract physics-meaningful features from signal patches.

    Args:
        patches: (B, n_mask, patch_size) float tensor
        sample_rate: sampling rate in Hz

    Returns:
        features: (B, n_mask, 9) float tensor
            [band0_energy, band1_energy, band2_energy, band3_energy,
             rms, kurtosis, crest_factor, skewness, envelope_rms]
    """
    B, N, T = patches.shape
    patches_np = patches.detach().cpu().numpy().reshape(B * N, T)
    feats = []

    # Frequency bands (at 12kHz, Nyquist=6kHz)
    band_edges = [0, 1500, 3000, 5000, 6000]  # Hz — 4 bands
    freqs = np.fft.rfftfreq(T, d=1.0 / sample_rate)

    for sig in patches_np:
        row = []

        # FFT power spectrum
        fft_vals = np.abs(np.fft.rfft(sig)) ** 2

        # Band energies (normalized by total)
        total_power = fft_vals.sum() + 1e-8
        for lo, hi in zip(band_edges[:-1], band_edges[1:]):
            mask = (freqs >= lo) & (freqs < hi)
            row.append(float(fft_vals[mask].sum() / total_power))

        # Time-domain statistics
        rms = float(np.sqrt(np.mean(sig ** 2)))
        row.append(rms)

        # Kurtosis (using scipy)
        from scipy.stats import kurtosis as sp_kurtosis
        row.append(float(sp_kurtosis(sig)))

        # Crest factor (peak / RMS)
        rms_safe = max(rms, 1e-8)
        row.append(float(np.max(np.abs(sig)) / rms_safe))

        # Skewness
        from scipy.stats import skew as sp_skew
        row.append(float(sp_skew(sig)))

        # Envelope RMS via Hilbert transform
        analytic = scipy.signal.hilbert(sig)
        envelope = np.abs(analytic)
        row.append(float(np.sqrt(np.mean(envelope ** 2))))

        feats.append(row)

    feats = np.array(feats, dtype=np.float32).reshape(B, N, 9)
    return torch.tensor(feats)


def extract_patches_from_signal(signals: torch.Tensor, mask_indices: torch.Tensor, patch_size: int) -> torch.Tensor:
    """
    Extract the raw signal patches at masked positions.

    Args:
        signals: (B, C, T) input signals
        mask_indices: (B, n_mask) patch indices
        patch_size: samples per patch

    Returns:
        patches: (B, n_mask, patch_size) — first channel only
    """
    B, C, T = signals.shape
    # Use first channel (DE) for feature extraction
    signal_1d = signals[:, 0, :]  # (B, T)

    patches = []
    for b in range(B):
        row = []
        for idx in mask_indices[b]:
            start = idx.item() * patch_size
            end = start + patch_size
            patch = signal_1d[b, start:end]
            row.append(patch)
        patches.append(torch.stack(row))  # (n_mask, patch_size)
    return torch.stack(patches)  # (B, n_mask, patch_size)


# ============================================================================
# SF-JEPA: Spectral Feature JEPA
# ============================================================================

class SpectralJEPA(nn.Module):
    """
    Spectral Feature JEPA.

    Extends V2 with an additional spectral feature prediction head.
    The main JEPA objective remains unchanged (predict latent targets).
    The spectral head adds a physics-informed auxiliary loss.
    """

    def __init__(
        self,
        n_channels: int = 3,
        window_size: int = 4096,
        patch_size: int = 256,
        embed_dim: int = 512,
        encoder_depth: int = 4,
        predictor_depth: int = 4,
        n_heads: int = 4,
        mask_ratio: float = 0.625,
        ema_decay: float = 0.996,
        loss_fn: str = 'l1',
        var_reg_lambda: float = 0.1,
        spectral_weight: float = 0.5,  # Weight of spectral auxiliary loss
        n_spectral_feats: int = 9,
        sample_rate: float = 12000.0,
    ):
        super().__init__()
        self.base = MechanicalJEPAV2(
            n_channels=n_channels,
            window_size=window_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            encoder_depth=encoder_depth,
            predictor_depth=predictor_depth,
            n_heads=n_heads,
            mask_ratio=mask_ratio,
            ema_decay=ema_decay,
            predictor_pos='sinusoidal',
            loss_fn=loss_fn,
            var_reg_lambda=var_reg_lambda,
        )
        self.spectral_weight = spectral_weight
        self.n_spectral_feats = n_spectral_feats
        self.patch_size = patch_size
        self.sample_rate = sample_rate

        # Spectral feature prediction head: maps predictor output to 9 features
        # Input: (B, n_mask, embed_dim) predictions — DETACH from main path
        self.spectral_head = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Linear(256, n_spectral_feats),
        )

    def forward(self, x: torch.Tensor):
        """
        Forward pass combining JEPA loss and spectral feature prediction.

        Returns:
            total_loss, jepa_loss, spectral_loss
        """
        B = x.shape[0]
        device = x.device

        # Generate mask (same as V2)
        n_patches = self.base.n_patches
        n_mask = int(n_patches * self.base.mask_ratio)
        n_context = n_patches - n_mask

        indices = torch.stack([torch.randperm(n_patches, device=device) for _ in range(B)])
        mask_indices = indices[:, :n_mask]
        context_indices = indices[:, n_mask:]

        # Standard JEPA forward
        with torch.no_grad():
            target_embeds = self.base.target_encoder(x, return_all_tokens=True)[:, 1:]
            targets = torch.gather(
                target_embeds, 1,
                mask_indices.unsqueeze(-1).expand(-1, -1, target_embeds.shape[-1])
            )

        context_embeds = self.base.encoder(x, mask_indices=mask_indices, return_all_tokens=True)[:, 1:]
        predictions = self.base.predictor(context_embeds, context_indices, mask_indices)

        # Main JEPA loss
        pred_norm = F.normalize(predictions, dim=-1)
        tgt_norm = F.normalize(targets, dim=-1)
        if self.base.loss_fn == 'l1':
            jepa_loss = F.l1_loss(pred_norm, tgt_norm)
        elif self.base.loss_fn == 'mse':
            jepa_loss = F.mse_loss(pred_norm, tgt_norm)
        else:
            jepa_loss = F.smooth_l1_loss(pred_norm, tgt_norm)

        if self.base.var_reg_lambda > 0:
            var_loss = prediction_var_loss(predictions, threshold=0.1)
            jepa_loss = jepa_loss + self.base.var_reg_lambda * var_loss

        # Spectral feature targets (computed on CPU, no gradient through signal)
        spectral_loss = torch.tensor(0.0, device=device)
        if self.spectral_weight > 0:
            with torch.no_grad():
                raw_patches = extract_patches_from_signal(x, mask_indices, self.patch_size)
                # raw_patches: (B, n_mask, patch_size)
                spectral_targets = extract_spectral_features(raw_patches, self.sample_rate).to(device)
                # spectral_targets: (B, n_mask, 9) — normalize to zero mean unit var
                spectral_targets = F.normalize(spectral_targets, dim=-1)

            # Predict spectral features from predictor output
            spectral_preds = self.spectral_head(predictions)  # (B, n_mask, 9)
            spectral_preds = F.normalize(spectral_preds, dim=-1)

            spectral_loss = F.l1_loss(spectral_preds, spectral_targets)

        total_loss = jepa_loss + self.spectral_weight * spectral_loss

        # EMA update
        self.base._update_target_encoder()

        return total_loss, jepa_loss.item(), spectral_loss.item() if self.spectral_weight > 0 else 0.0

    def get_embeddings(self, x: torch.Tensor, pool: str = 'mean') -> torch.Tensor:
        """Get embeddings for downstream tasks (same as V2)."""
        return self.base.get_embeddings(x, pool=pool)


# ============================================================================
# Training Loop
# ============================================================================

def train_spectral_jepa(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])

    # Data
    train_loader, test_loader, info = create_dataloaders(
        data_dir=config['data_dir'],
        batch_size=config['batch_size'],
        window_size=config['window_size'],
        stride=config['stride'],
        test_ratio=0.2,
        seed=config['seed'],
        num_workers=0,
        dataset_filter='cwru',
        n_channels=3,
    )
    print(f"Train: {len(train_loader.dataset)} windows, Test: {len(test_loader.dataset)} windows")

    # Model
    model = SpectralJEPA(
        n_channels=3,
        window_size=config['window_size'],
        patch_size=config['patch_size'],
        embed_dim=config['embed_dim'],
        encoder_depth=config['encoder_depth'],
        predictor_depth=config['predictor_depth'],
        n_heads=config['n_heads'],
        mask_ratio=config['mask_ratio'],
        ema_decay=config['ema_decay'],
        loss_fn=config['loss_fn'],
        var_reg_lambda=config['var_reg_lambda'],
        spectral_weight=config['spec_weight'],
        sample_rate=12000.0,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model params: {total_params:,}")

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=0.05)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['epochs'], eta_min=1e-6
    )

    # Training
    history = []
    t_start = time.time()

    for epoch in range(1, config['epochs'] + 1):
        model.train()
        ep_total, ep_jepa, ep_spec = 0.0, 0.0, 0.0
        n_batches = 0

        for signals, labels, _ in train_loader:
            signals = signals.to(device)
            optimizer.zero_grad()
            total_loss, jepa_l, spec_l = model(signals)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            ep_total += total_loss.item()
            ep_jepa += jepa_l
            ep_spec += spec_l
            n_batches += 1

        scheduler.step()

        ep_total /= n_batches
        ep_jepa /= n_batches
        ep_spec /= n_batches

        history.append({'epoch': epoch, 'total': ep_total, 'jepa': ep_jepa, 'spectral': ep_spec})

        if epoch % 10 == 0 or epoch == 1:
            elapsed = time.time() - t_start
            print(f"  Epoch {epoch:3d}/{config['epochs']} | total={ep_total:.4f} | "
                  f"jepa={ep_jepa:.4f} | spec={ep_spec:.4f} | {elapsed:.0f}s")

    # Linear probe evaluation on CWRU
    model.eval()
    print("\nEvaluating linear probe on CWRU...")

    def embed_fn(x):
        return model.get_embeddings(x.to(device), pool='mean')

    all_e, all_l = [], []
    with torch.no_grad():
        for signals, labels, _ in train_loader:
            all_e.append(embed_fn(signals).cpu())
            all_l.append(labels)
    tr_e = torch.cat(all_e); tr_l = torch.cat(all_l)

    all_e, all_l = [], []
    with torch.no_grad():
        for signals, labels, _ in test_loader:
            all_e.append(embed_fn(signals).cpu())
            all_l.append(labels)
    te_e = torch.cat(all_e); te_l = torch.cat(all_l)

    probe = nn.Linear(config['embed_dim'], 4).to(device)
    opt = optim.Adam(probe.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()
    best_f1 = 0.0

    for ep in range(50):
        probe.train(); opt.zero_grad()
        logits = probe(tr_e.to(device))
        crit(logits, tr_l.to(device)).backward(); opt.step()
        probe.eval()
        with torch.no_grad():
            preds = probe(te_e.to(device)).argmax(1).cpu().numpy()
        f1 = f1_score(te_l.numpy(), preds, average='macro', zero_division=0)
        best_f1 = max(best_f1, f1)

    print(f"CWRU linear probe F1: {best_f1:.4f}")

    # Save checkpoint
    ckpt_dir = Path('/mnt/sagemaker-nvme/jepa_checkpoints')
    ckpt_dir.mkdir(exist_ok=True, parents=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    ckpt_path = ckpt_dir / f"spectral_jepa_seed{config['seed']}_{ts}.pt"
    torch.save({
        'model_state_dict': model.base.state_dict(),  # Same as V2 for compatibility
        'spectral_head': model.spectral_head.state_dict(),
        'config': config,
        'cwru_f1': best_f1,
        'history': history,
    }, str(ckpt_path))
    print(f"Checkpoint saved: {ckpt_path}")

    return best_f1, ckpt_path, history


# ============================================================================
# Paderborn Transfer Evaluation
# ============================================================================

def evaluate_paderborn_transfer(model_fn, embed_dim, device):
    """Run Paderborn transfer and return F1 and random baseline."""
    from paderborn_transfer import create_paderborn_loaders, CLASSES
    PADERBORN_DIR = Path('/home/sagemaker-user/IndustrialJEPA/datasets/data/paderborn')

    bearing_dirs = [(str(PADERBORN_DIR / folder), label) for folder, label in CLASSES.items()
                    if (PADERBORN_DIR / folder).exists()]
    if not bearing_dirs:
        return None, None

    pad_train, pad_test = create_paderborn_loaders(
        bearing_dirs=bearing_dirs, window_size=4096, stride=2048, target_sr=20000,
        n_channels=3, test_ratio=0.2, batch_size=32, seed=42, max_files_per_bearing=20,
    )

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

    tr_e, tr_l = extract(pad_train)
    te_e, te_l = extract(pad_test)

    probe = nn.Linear(embed_dim, 3).to(device)
    opt = optim.Adam(probe.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()
    best_f1 = 0.0

    for ep in range(50):
        probe.train(); opt.zero_grad()
        logits = probe(tr_e.to(device))
        crit(logits, tr_l.to(device)).backward(); opt.step()
        probe.eval()
        with torch.no_grad():
            preds = probe(te_e.to(device)).argmax(1).cpu().numpy()
        f1 = f1_score(te_l.numpy(), preds, average='macro', zero_division=0)
        best_f1 = max(best_f1, f1)

    # Random baseline
    from src.models import MechanicalJEPAV2
    rand = MechanicalJEPAV2(n_channels=3, window_size=4096, patch_size=256, embed_dim=embed_dim, encoder_depth=4).to(device)
    rand.eval()
    def rand_fn(x): return rand.get_embeddings(x, pool='mean')
    tr_r, tr_l2 = extract(pad_train)

    probe2 = nn.Linear(embed_dim, 3).to(device)
    opt2 = optim.Adam(probe2.parameters(), lr=1e-3)
    rand_f1 = 0.0
    # Re-extract with rand model
    r_tr_e, r_tr_l = [], []
    r_te_e, r_te_l = [], []
    with torch.no_grad():
        for batch in pad_train:
            x = batch[0].to(device)
            y = batch[1]
            r_tr_e.append(rand_fn(x).cpu())
            r_tr_l.append(y if isinstance(y, torch.Tensor) else torch.tensor(y))
        for batch in pad_test:
            x = batch[0].to(device)
            y = batch[1]
            r_te_e.append(rand_fn(x).cpu())
            r_te_l.append(y if isinstance(y, torch.Tensor) else torch.tensor(y))
    r_tr_e = torch.cat(r_tr_e); r_tr_l = torch.cat(r_tr_l)
    r_te_e = torch.cat(r_te_e); r_te_l = torch.cat(r_te_l)

    for ep in range(50):
        probe2.train(); opt2.zero_grad()
        logits = probe2(r_tr_e.to(device))
        crit(logits, r_tr_l.to(device)).backward(); opt2.step()
        probe2.eval()
        with torch.no_grad():
            preds = probe2(r_te_e.to(device)).argmax(1).cpu().numpy()
        f1 = f1_score(r_te_l.numpy(), preds, average='macro', zero_division=0)
        rand_f1 = max(rand_f1, f1)

    return best_f1, rand_f1


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 456])
    parser.add_argument('--spec-weight', type=float, default=0.5,
                        help='Weight of spectral auxiliary loss (0=no spectral, 1=equal weight)')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--data-dir', type=str, default='data/bearings')
    args = parser.parse_args()

    import shutil
    free_gb = shutil.disk_usage('/home/sagemaker-user').free / 1e9
    print(f"Home disk: {free_gb:.1f} GB free")
    if free_gb < 2.0:
        print("CRITICAL: Not enough disk space")
        sys.exit(1)

    config_base = {
        'data_dir': args.data_dir,
        'batch_size': args.batch_size,
        'window_size': 4096,
        'stride': 2048,
        'patch_size': 256,
        'embed_dim': 512,
        'encoder_depth': 4,
        'predictor_depth': 4,
        'n_heads': 4,
        'mask_ratio': 0.625,
        'ema_decay': 0.996,
        'loss_fn': 'l1',
        'var_reg_lambda': 0.1,
        'epochs': args.epochs,
        'lr': 1e-4,
        'spec_weight': args.spec_weight,
    }

    all_results = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for seed in args.seeds:
        print(f"\n{'='*70}")
        print(f"Seed {seed} | Spectral weight: {args.spec_weight}")
        print(f"{'='*70}")
        config = {**config_base, 'seed': seed}
        t0 = time.time()

        cwru_f1, ckpt_path, history = train_spectral_jepa(config)

        # Paderborn transfer
        print("\nEvaluating Paderborn transfer...")
        ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
        model = MechanicalJEPAV2(
            n_channels=3, window_size=4096, patch_size=256, embed_dim=512,
            encoder_depth=4, predictor_depth=4, n_heads=4, mask_ratio=0.625,
            predictor_pos='sinusoidal', loss_fn='l1', var_reg_lambda=0.1,
        ).to(device)
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()
        model_fn = lambda x: model.get_embeddings(x, pool='mean')

        pad_f1, rand_f1 = evaluate_paderborn_transfer(model_fn, 512, device)
        gain = (pad_f1 - rand_f1) if (pad_f1 and rand_f1) else None

        print(f"\nSeed {seed} Results:")
        print(f"  CWRU F1:       {cwru_f1:.4f}")
        print(f"  Paderborn F1:  {pad_f1:.4f}" if pad_f1 else "  Paderborn: N/A")
        print(f"  Transfer gain: {gain:+.4f}" if gain else "  Gain: N/A")
        print(f"  Time: {time.time()-t0:.0f}s")

        all_results.append({
            'seed': seed, 'spec_weight': args.spec_weight,
            'cwru_f1': cwru_f1, 'pad_f1': pad_f1, 'rand_f1': rand_f1, 'gain': gain,
            'ckpt_path': str(ckpt_path), 'history': history[-1],
        })

    # Summary
    print(f"\n{'='*100}")
    print("SF-JEPA RESULTS SUMMARY")
    print(f"{'='*100}")
    cwru_vals = [r['cwru_f1'] for r in all_results]
    pad_vals = [r['pad_f1'] for r in all_results if r['pad_f1']]
    gain_vals = [r['gain'] for r in all_results if r['gain']]

    print(f"Spec weight: {args.spec_weight}")
    print(f"CWRU F1:       {np.mean(cwru_vals):.4f} ± {np.std(cwru_vals):.4f} (3-seed)")
    if pad_vals:
        print(f"Paderborn F1:  {np.mean(pad_vals):.4f} ± {np.std(pad_vals):.4f}")
        print(f"Transfer gain: {np.mean(gain_vals):+.4f} ± {np.std(gain_vals):.4f}")

    # Compare to V2 baseline
    v2_cwru = 0.773; v2_gain = 0.453
    print(f"\nComparison to JEPA V2 baseline:")
    print(f"  CWRU: {np.mean(cwru_vals):.4f} vs V2 {v2_cwru:.4f} "
          f"({'better' if np.mean(cwru_vals) > v2_cwru else 'worse'})")
    if gain_vals:
        print(f"  Gain: {np.mean(gain_vals):+.4f} vs V2 {v2_gain:+.4f} "
              f"({'better' if np.mean(gain_vals) > v2_gain else 'worse'})")

    # Save results
    save_path = Path('results/spectral_jepa.json')
    save_path.parent.mkdir(exist_ok=True, parents=True)

    with open(save_path, 'w') as f:
        json.dump({
            'spec_weight': args.spec_weight,
            'epochs': args.epochs,
            'per_seed': all_results,
            'summary': {
                'cwru_mean': float(np.mean(cwru_vals)),
                'cwru_std': float(np.std(cwru_vals)),
                'pad_mean': float(np.mean(pad_vals)) if pad_vals else None,
                'pad_std': float(np.std(pad_vals)) if pad_vals else None,
                'gain_mean': float(np.mean(gain_vals)) if gain_vals else None,
                'gain_std': float(np.std(gain_vals)) if gain_vals else None,
            }
        }, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.float32, np.float64)) else str(x))
    print(f"\nResults saved: {save_path}")


if __name__ == '__main__':
    main()
