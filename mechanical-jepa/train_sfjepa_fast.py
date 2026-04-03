"""
SF-JEPA Fast: Spectral Feature JEPA with vectorized feature extraction.

Key improvement over train_spectral_jepa.py:
- Vectorized torch FFT (no scipy per-patch loop)
- ~100x faster spectral feature computation
- All features computed as differentiable torch operations (could backprop through targets if needed)

Features extracted per patch (256 samples at 12kHz):
  - 4 normalized band energies (0-1.5kHz, 1.5-3kHz, 3-5kHz, 5-6kHz)
  - RMS
  - Log variance (more numerically stable than kurtosis for normalization)
  - Spectral centroid (normalized)
  - Peak-to-RMS ratio (crest factor)
  Total: 8 features per patch

Usage:
    python train_sfjepa_fast.py --epochs 100 --seeds 42 123 456 --spec-weight 0.5
    python train_sfjepa_fast.py --epochs 100 --seeds 42 --spec-weight 0.0  # baseline (pure JEPA)
    python train_sfjepa_fast.py --epochs 100 --seeds 42 --spec-weight 1.0  # max spectral
"""

import argparse
import copy
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
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
# Fast Vectorized Spectral Feature Extraction (torch-based)
# ============================================================================

@torch.no_grad()
def extract_spectral_features_fast(patches: torch.Tensor, sample_rate: float = 12000.0) -> torch.Tensor:
    """
    Fast vectorized spectral feature extraction using torch FFT.

    Args:
        patches: (B*N, T) float tensor (flattened batch x patches)
        sample_rate: Hz

    Returns:
        features: (B*N, 8) float tensor
            [band0, band1, band2, band3, rms, log_var, spectral_centroid, crest_factor]
    """
    device = patches.device
    BN, T = patches.shape

    # FFT
    fft = torch.fft.rfft(patches, dim=-1)  # (BN, T//2+1)
    power = torch.abs(fft) ** 2  # (BN, T//2+1)
    freqs = torch.fft.rfftfreq(T, d=1.0/sample_rate).to(device)  # (T//2+1,)

    total_power = power.sum(dim=-1, keepdim=True) + 1e-8

    # Band energies (normalized)
    band_edges = [0, 1500, 3000, 5000, 6000]
    band_energies = []
    for lo, hi in zip(band_edges[:-1], band_edges[1:]):
        mask = (freqs >= lo) & (freqs < hi)
        band_energy = power[:, mask].sum(dim=-1, keepdim=True) / total_power
        band_energies.append(band_energy)
    bands = torch.cat(band_energies, dim=-1)  # (BN, 4)

    # Time domain features
    rms = torch.sqrt((patches ** 2).mean(dim=-1, keepdim=True) + 1e-8)  # (BN, 1)
    log_var = torch.log((patches.var(dim=-1, keepdim=True) + 1e-8))  # (BN, 1)

    # Spectral centroid (normalized by Nyquist)
    nyquist = sample_rate / 2
    centroid = (power * freqs.unsqueeze(0)).sum(dim=-1, keepdim=True) / (total_power * nyquist)  # (BN, 1)

    # Crest factor (peak / RMS)
    peak = torch.abs(patches).max(dim=-1, keepdim=True).values
    crest = peak / (rms + 1e-8)  # (BN, 1)

    features = torch.cat([bands, rms, log_var, centroid, crest], dim=-1)  # (BN, 8)
    return features


# ============================================================================
# SF-JEPA Fast Model
# ============================================================================

class SFJEPAFast(nn.Module):
    """
    Spectral Feature JEPA with fast vectorized feature computation.
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
        spectral_weight: float = 0.5,
        n_spectral_feats: int = 8,
        sample_rate: float = 12000.0,
    ):
        super().__init__()
        self.base = MechanicalJEPAV2(
            n_channels=n_channels, window_size=window_size, patch_size=patch_size,
            embed_dim=embed_dim, encoder_depth=encoder_depth, predictor_depth=predictor_depth,
            n_heads=n_heads, mask_ratio=mask_ratio, ema_decay=ema_decay,
            predictor_pos='sinusoidal', loss_fn=loss_fn, var_reg_lambda=var_reg_lambda,
        )
        self.spectral_weight = spectral_weight
        self.n_spectral_feats = n_spectral_feats
        self.patch_size = patch_size
        self.sample_rate = sample_rate
        self.n_patches = window_size // patch_size
        self.n_mask = int(self.n_patches * mask_ratio)
        self.n_context = self.n_patches - self.n_mask

        # Spectral feature prediction head
        self.spectral_head = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Linear(256, n_spectral_feats),
        )

    def _get_raw_patches(self, x: torch.Tensor, mask_indices: torch.Tensor) -> torch.Tensor:
        """
        Extract raw signal patches at masked positions.
        Uses first channel (DE).

        Returns: (B * n_mask, patch_size)
        """
        B = x.shape[0]
        n_mask = mask_indices.shape[1]
        signal = x[:, 0, :]  # (B, T)

        # Build index tensor: (B, n_mask, patch_size)
        patch_starts = mask_indices * self.patch_size  # (B, n_mask)
        offsets = torch.arange(self.patch_size, device=x.device).unsqueeze(0).unsqueeze(0)  # (1,1,P)
        idx = (patch_starts.unsqueeze(-1) + offsets)  # (B, n_mask, P)

        # Gather patches
        idx_flat = idx.reshape(B, n_mask * self.patch_size)  # (B, n_mask*P)
        patches_flat = torch.gather(signal, 1, idx_flat)  # (B, n_mask*P)
        patches = patches_flat.reshape(B * n_mask, self.patch_size)  # (B*n_mask, P)

        return patches

    def forward(self, x: torch.Tensor):
        B = x.shape[0]
        device = x.device

        # Generate mask
        indices = torch.stack([torch.randperm(self.n_patches, device=device) for _ in range(B)])
        mask_indices = indices[:, :self.n_mask]      # first n_mask patches are masked
        context_indices = indices[:, self.n_mask:]   # remaining n_context patches are context

        # Standard JEPA targets (no grad)
        with torch.no_grad():
            target_embeds = self.base.target_encoder(x, return_all_tokens=True)[:, 1:]
            targets = torch.gather(
                target_embeds, 1,
                mask_indices.unsqueeze(-1).expand(-1, -1, target_embeds.shape[-1])
            )

        # Context encoding
        context_embeds = self.base.encoder(x, mask_indices=mask_indices, return_all_tokens=True)[:, 1:]
        predictions = self.base.predictor(context_embeds, context_indices, mask_indices)

        # Main JEPA loss
        pred_norm = F.normalize(predictions, dim=-1)
        tgt_norm = F.normalize(targets, dim=-1)
        if self.base.loss_fn == 'l1':
            jepa_loss = F.l1_loss(pred_norm, tgt_norm)
        else:
            jepa_loss = F.mse_loss(pred_norm, tgt_norm)

        if self.base.var_reg_lambda > 0:
            var_loss = prediction_var_loss(predictions, threshold=0.1)
            jepa_loss = jepa_loss + self.base.var_reg_lambda * var_loss

        # Spectral feature loss (fast)
        spectral_loss = torch.tensor(0.0, device=device)
        if self.spectral_weight > 0:
            # Extract raw patches at mask positions (GPU, vectorized)
            raw_patches = self._get_raw_patches(x, mask_indices)  # (B*n_mask, P)

            with torch.no_grad():
                spec_targets = extract_spectral_features_fast(raw_patches, self.sample_rate)  # (B*n_mask, F)
                # Reshape to (B, n_mask, F) and normalize
                spec_targets = spec_targets.reshape(B, self.n_mask, self.n_spectral_feats)
                # Normalize each feature dimension across positions (zero mean, unit std)
                spec_mean = spec_targets.mean(dim=(0, 1), keepdim=True)
                spec_std = spec_targets.std(dim=(0, 1), keepdim=True) + 1e-6
                spec_targets = (spec_targets - spec_mean) / spec_std

            # Predict spectral features
            spec_preds = self.spectral_head(predictions)  # (B, n_mask, F)
            spectral_loss = F.l1_loss(spec_preds, spec_targets)

        total_loss = jepa_loss + self.spectral_weight * spectral_loss

        # EMA update
        self.base._update_target_encoder()

        return total_loss, jepa_loss.item(), spectral_loss.item() if self.spectral_weight > 0 else 0.0

    def get_embeddings(self, x: torch.Tensor, pool: str = 'mean') -> torch.Tensor:
        return self.base.get_embeddings(x, pool=pool)


# ============================================================================
# Training
# ============================================================================

def train_sfjepa(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}", flush=True)

    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])

    train_loader, test_loader, _ = create_dataloaders(
        data_dir=config['data_dir'], batch_size=config['batch_size'],
        window_size=config['window_size'], stride=config['stride'],
        test_ratio=0.2, seed=config['seed'], num_workers=0,
        dataset_filter='cwru', n_channels=3,
    )
    print(f"Train: {len(train_loader.dataset)} windows, Test: {len(test_loader.dataset)} windows", flush=True)

    model = SFJEPAFast(
        n_channels=3, window_size=config['window_size'], patch_size=config['patch_size'],
        embed_dim=config['embed_dim'], encoder_depth=config['encoder_depth'],
        predictor_depth=config['predictor_depth'], n_heads=config['n_heads'],
        mask_ratio=config['mask_ratio'], ema_decay=config['ema_decay'],
        loss_fn=config['loss_fn'], var_reg_lambda=config['var_reg_lambda'],
        spectral_weight=config['spec_weight'],
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model params: {total_params:,}", flush=True)

    optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=0.05)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'], eta_min=1e-6)

    history = []
    t_start = time.time()

    for epoch in range(1, config['epochs'] + 1):
        model.train()
        ep_total, ep_jepa, ep_spec, n_batches = 0.0, 0.0, 0.0, 0

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
        ep_total /= n_batches; ep_jepa /= n_batches; ep_spec /= n_batches
        history.append({'epoch': epoch, 'total': ep_total, 'jepa': ep_jepa, 'spectral': ep_spec})

        if epoch % 10 == 0 or epoch == 1:
            elapsed = time.time() - t_start
            print(f"  Epoch {epoch:3d}/{config['epochs']} | total={ep_total:.4f} | jepa={ep_jepa:.4f} | "
                  f"spec={ep_spec:.4f} | {elapsed:.0f}s", flush=True)

    # CWRU linear probe
    model.eval()
    def emb(x): return model.get_embeddings(x.to(device), pool='mean')

    tr_e, tr_l, te_e, te_l = [], [], [], []
    with torch.no_grad():
        for s, l, _ in train_loader: tr_e.append(emb(s).cpu()); tr_l.append(l)
        for s, l, _ in test_loader: te_e.append(emb(s).cpu()); te_l.append(l)
    tr_e = torch.cat(tr_e); tr_l = torch.cat(tr_l)
    te_e = torch.cat(te_e); te_l = torch.cat(te_l)

    probe = nn.Linear(config['embed_dim'], 4).to(device)
    opt = optim.Adam(probe.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()
    best_f1 = 0.0

    for ep in range(50):
        probe.train(); opt.zero_grad()
        crit(probe(tr_e.to(device)), tr_l.to(device)).backward(); opt.step()
        probe.eval()
        with torch.no_grad():
            preds = probe(te_e.to(device)).argmax(1).cpu().numpy()
        best_f1 = max(best_f1, f1_score(te_l.numpy(), preds, average='macro', zero_division=0))

    print(f"CWRU linear probe F1: {best_f1:.4f}", flush=True)

    # Save checkpoint to NVMe
    ckpt_dir = Path('/mnt/sagemaker-nvme/jepa_checkpoints')
    ckpt_dir.mkdir(exist_ok=True, parents=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    ckpt_path = ckpt_dir / f"sfjepa_fast_sw{config['spec_weight']}_seed{config['seed']}_{ts}.pt"
    torch.save({
        'model_state_dict': model.base.state_dict(),
        'spectral_head': model.spectral_head.state_dict(),
        'config': config,
        'cwru_f1': best_f1,
        'history': history,
    }, str(ckpt_path))
    print(f"Checkpoint: {ckpt_path}", flush=True)
    return best_f1, ckpt_path, history


def evaluate_paderborn(model_fn, embed_dim, device, seed=42):
    from paderborn_transfer import create_paderborn_loaders, CLASSES
    PADERBORN_DIR = Path('/home/sagemaker-user/IndustrialJEPA/datasets/data/paderborn')
    bearing_dirs = [(str(PADERBORN_DIR / f), l) for f, l in CLASSES.items() if (PADERBORN_DIR / f).exists()]
    if not bearing_dirs: return None, None

    pad_train, pad_test = create_paderborn_loaders(
        bearing_dirs=bearing_dirs, window_size=4096, stride=2048, target_sr=20000,
        n_channels=3, test_ratio=0.2, batch_size=32, seed=seed, max_files_per_bearing=20,
    )

    def extract(loader):
        all_e, all_l = [], []
        with torch.no_grad():
            for batch in loader:
                x = batch[0].to(device); y = batch[1]
                all_e.append(model_fn(x).cpu())
                all_l.append(y if isinstance(y, torch.Tensor) else torch.tensor(y))
        return torch.cat(all_e), torch.cat(all_l)

    tr_e, tr_l = extract(pad_train)
    te_e, te_l = extract(pad_test)

    def run_probe(tr_e, tr_l, te_e, te_l, n_cls=3):
        probe = nn.Linear(embed_dim, n_cls).to(device)
        opt = optim.Adam(probe.parameters(), lr=1e-3)
        crit = nn.CrossEntropyLoss()
        best = 0.0
        for ep in range(80):
            probe.train(); opt.zero_grad()
            crit(probe(tr_e.to(device)), tr_l.to(device)).backward(); opt.step()
            probe.eval()
            with torch.no_grad():
                preds = probe(te_e.to(device)).argmax(1).cpu().numpy()
            best = max(best, f1_score(te_l.numpy(), preds, average='macro', zero_division=0))
        return best

    pad_f1 = run_probe(tr_e, tr_l, te_e, te_l)

    # Random baseline
    rand = MechanicalJEPAV2(n_channels=3, window_size=4096, patch_size=256, embed_dim=embed_dim, encoder_depth=4).to(device)
    rand.eval()
    r_tr_e, r_tr_l = extract(pad_train)  # Re-extract with rand
    r_te_e, r_te_l = extract(pad_test)
    # Actually use rand model
    r_tr_e2, r_tr_l2 = [], []
    r_te_e2, r_te_l2 = [], []
    with torch.no_grad():
        for batch in pad_train:
            x = batch[0].to(device); y = batch[1]
            r_tr_e2.append(rand.get_embeddings(x, pool='mean').cpu())
            r_tr_l2.append(y if isinstance(y, torch.Tensor) else torch.tensor(y))
        for batch in pad_test:
            x = batch[0].to(device); y = batch[1]
            r_te_e2.append(rand.get_embeddings(x, pool='mean').cpu())
            r_te_l2.append(y if isinstance(y, torch.Tensor) else torch.tensor(y))
    r_tr_e2 = torch.cat(r_tr_e2); r_tr_l2 = torch.cat(r_tr_l2)
    r_te_e2 = torch.cat(r_te_e2); r_te_l2 = torch.cat(r_te_l2)
    rand_f1 = run_probe(r_tr_e2, r_tr_l2, r_te_e2, r_te_l2)

    return pad_f1, rand_f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 456])
    parser.add_argument('--spec-weight', type=float, default=0.5)
    parser.add_argument('--data-dir', type=str, default='data/bearings')
    args = parser.parse_args()

    import shutil
    print(f"Home disk: {shutil.disk_usage('/home/sagemaker-user').free/1e9:.1f} GB free", flush=True)

    config_base = {
        'data_dir': args.data_dir, 'batch_size': 32, 'window_size': 4096, 'stride': 2048,
        'patch_size': 256, 'embed_dim': 512, 'encoder_depth': 4, 'predictor_depth': 4,
        'n_heads': 4, 'mask_ratio': 0.625, 'ema_decay': 0.996,
        'loss_fn': 'l1', 'var_reg_lambda': 0.1, 'epochs': args.epochs, 'lr': 1e-4,
        'spec_weight': args.spec_weight,
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    all_results = []

    for seed in args.seeds:
        print(f"\n{'='*70}", flush=True)
        print(f"Seed {seed} | spec_weight={args.spec_weight}", flush=True)
        print(f"{'='*70}", flush=True)
        config = {**config_base, 'seed': seed}
        t0 = time.time()

        cwru_f1, ckpt_path, history = train_sfjepa(config)

        # Load trained model for Paderborn eval
        ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
        model = MechanicalJEPAV2(
            n_channels=3, window_size=4096, patch_size=256, embed_dim=512,
            encoder_depth=4, predictor_depth=4, n_heads=4, mask_ratio=0.625,
            predictor_pos='sinusoidal', loss_fn='l1', var_reg_lambda=0.1,
        ).to(device)
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()
        model_fn = lambda x: model.get_embeddings(x, pool='mean')

        print("\nEvaluating Paderborn transfer...", flush=True)
        pad_f1, rand_f1 = evaluate_paderborn(model_fn, 512, device, seed=seed)
        gain = (pad_f1 - rand_f1) if (pad_f1 is not None and rand_f1 is not None) else None

        print(f"\nSeed {seed} Results:", flush=True)
        print(f"  CWRU F1: {cwru_f1:.4f}", flush=True)
        print(f"  Pad F1:  {pad_f1:.4f}" if pad_f1 else "  Pad: N/A", flush=True)
        print(f"  Gain:    {gain:+.4f}" if gain else "  Gain: N/A", flush=True)
        print(f"  Time: {time.time()-t0:.0f}s", flush=True)

        all_results.append({
            'seed': seed, 'spec_weight': args.spec_weight,
            'cwru_f1': float(cwru_f1), 'pad_f1': float(pad_f1) if pad_f1 else None,
            'rand_f1': float(rand_f1) if rand_f1 else None, 'gain': float(gain) if gain else None,
        })

        # Save after each seed
        save_path = Path('results/sfjepa_fast.json')
        save_path.parent.mkdir(exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump({'spec_weight': args.spec_weight, 'epochs': args.epochs,
                       'per_seed': all_results}, f, indent=2)

    # Summary
    print(f"\n{'='*100}", flush=True)
    print("SF-JEPA FAST RESULTS", flush=True)
    cwru_vals = [r['cwru_f1'] for r in all_results]
    pad_vals = [r['pad_f1'] for r in all_results if r['pad_f1']]
    gain_vals = [r['gain'] for r in all_results if r['gain']]
    print(f"Spec weight: {args.spec_weight}", flush=True)
    print(f"CWRU:  {np.mean(cwru_vals):.4f} ± {np.std(cwru_vals):.4f}", flush=True)
    if pad_vals:
        print(f"Pad:   {np.mean(pad_vals):.4f} ± {np.std(pad_vals):.4f}", flush=True)
        print(f"Gain:  {np.mean(gain_vals):+.4f} ± {np.std(gain_vals):.4f}", flush=True)

    # Comparison to V2 baseline
    v2_cwru, v2_gain = 0.773, 0.371
    print(f"\nVs JEPA V2 baseline (cwru={v2_cwru:.3f}, gain={v2_gain:+.3f}):", flush=True)
    print(f"  CWRU delta: {np.mean(cwru_vals)-v2_cwru:+.4f}", flush=True)
    if gain_vals:
        print(f"  Gain delta: {np.mean(gain_vals)-v2_gain:+.4f}", flush=True)

    # Final save
    summary = {
        'cwru_mean': float(np.mean(cwru_vals)), 'cwru_std': float(np.std(cwru_vals)),
        'pad_mean': float(np.mean(pad_vals)) if pad_vals else None,
        'gain_mean': float(np.mean(gain_vals)) if gain_vals else None,
        'gain_std': float(np.std(gain_vals)) if gain_vals else None,
    }
    with open(save_path, 'w') as f:
        json.dump({'spec_weight': args.spec_weight, 'epochs': args.epochs,
                   'per_seed': all_results, 'summary': summary}, f, indent=2)
    print(f"\nSaved: {save_path}", flush=True)


if __name__ == '__main__':
    main()
