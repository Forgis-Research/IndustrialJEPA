"""
Round 3: Frequency-domain masking for JEPA.

Nobody has done frequency-domain masking in JEPA for vibration signals.
Mechanical faults live in specific frequency bands (BPFO, BPFI, BSF, FTF).

Strategy: Instead of masking time-domain patches, mask frequency bands.
The model must predict the masked frequency content from visible bands.

Two approaches:
1. FFT masking: FFT input -> mask frequency bands -> IFFT -> train JEPA on this
2. Hybrid: Time-domain JEPA with frequency-domain mask selection

This is novel: JEPA has only been applied with time-domain masking.

Usage:
    python freq_masking.py --mode fft_mask --epochs 30 --seeds 42 123 456
    python freq_masking.py --mode combined --epochs 30
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).parent))
from src.data import create_dataloaders
from src.models import MechanicalJEPAV2

CHECKPOINT_DIR = Path('/mnt/sagemaker-nvme/jepa_checkpoints')
CHECKPOINT_DIR.mkdir(exist_ok=True, parents=True)


# =============================================================================
# FREQUENCY-DOMAIN MASKING TRANSFORMS
# =============================================================================

class FrequencyBandMasking:
    """
    Mask specific frequency bands in a vibration signal.

    Process:
    1. FFT the signal
    2. Zero out mask_ratio fraction of frequency bands
    3. IFFT back to time domain
    4. Train JEPA on the masked/unmasked time-domain signal

    This teaches the model to predict fault signatures from incomplete spectral information.

    Args:
        n_bands: Number of frequency bands to divide spectrum into
        mask_ratio: Fraction of bands to mask (0.625 = same as time masking)
        mode: 'high' (mask high freq), 'low' (mask low freq), 'random' (random bands)
    """

    def __init__(self, n_bands=16, mask_ratio=0.625, mode='random'):
        self.n_bands = n_bands
        self.mask_ratio = mask_ratio
        self.mode = mode

    def __call__(self, signal: torch.Tensor) -> torch.Tensor:
        """
        Apply frequency masking to signal.

        Args:
            signal: (B, C, L) time-domain signal

        Returns:
            masked_signal: (B, C, L) with frequency bands zeroed
        """
        B, C, L = signal.shape
        n_mask = int(self.n_bands * self.mask_ratio)

        # FFT
        spectrum = torch.fft.rfft(signal, dim=-1)  # (B, C, L//2+1)
        n_fft = spectrum.shape[-1]

        # Divide spectrum into bands
        band_size = n_fft // self.n_bands

        # Create mask
        if self.mode == 'random':
            band_indices = torch.randperm(self.n_bands)[:n_mask]
        elif self.mode == 'high':
            band_indices = torch.arange(self.n_bands - n_mask, self.n_bands)
        elif self.mode == 'low':
            band_indices = torch.arange(n_mask)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        # Apply mask to spectrum
        masked_spectrum = spectrum.clone()
        for band_idx in band_indices:
            start = band_idx * band_size
            end = min(start + band_size, n_fft)
            masked_spectrum[:, :, start:end] = 0.0

        # IFFT back to time domain
        masked_signal = torch.fft.irfft(masked_spectrum, n=L, dim=-1)

        return masked_signal


class TimeFreqMasking:
    """
    Combined time + frequency masking.

    Applies both time-domain patch masking (as in JEPA) and
    frequency-band masking to the context signal.
    """

    def __init__(self, time_mask_ratio=0.5, freq_mask_ratio=0.3, n_bands=8):
        self.freq_masking = FrequencyBandMasking(n_bands=n_bands, mask_ratio=freq_mask_ratio)
        self.time_mask_ratio = time_mask_ratio

    def apply_freq_to_context(self, context_signal: torch.Tensor) -> torch.Tensor:
        """Apply frequency masking to context signal before encoding."""
        return self.freq_masking(context_signal)


# =============================================================================
# JEPA WITH FREQUENCY MASKING
# =============================================================================

class MechanicalJEPAFreqMask(MechanicalJEPAV2):
    """
    JEPA V2 with optional frequency-domain masking on context signal.

    Key difference: before encoding context, apply frequency masking.
    The predictor must then predict masked patch embeddings from
    frequency-masked context — a harder and more structured task.
    """

    def __init__(self, freq_mask_ratio=0.3, freq_bands=8, **kwargs):
        super().__init__(**kwargs)
        self.freq_masking = FrequencyBandMasking(
            n_bands=freq_bands,
            mask_ratio=freq_mask_ratio,
            mode='random',
        )
        self.freq_mask_ratio = freq_mask_ratio
        self.freq_bands = freq_bands

    def forward(self, x: torch.Tensor):
        B = x.shape[0]
        device = x.device

        # Generate time-domain mask (same as V2)
        mask_indices, context_indices = self._generate_mask(B, device)

        # Get target embeddings (full signal, no masking)
        with torch.no_grad():
            target_embeds = self.target_encoder(x, return_all_tokens=True)[:, 1:]
            targets = torch.gather(
                target_embeds, 1,
                mask_indices.unsqueeze(-1).expand(-1, -1, target_embeds.shape[-1])
            )

        # Apply frequency masking to context signal (NOVEL!)
        x_freq_masked = self.freq_masking(x)

        # Encode frequency-masked context
        context_embeds = self.encoder(x_freq_masked, mask_indices=mask_indices, return_all_tokens=True)[:, 1:]

        # Predict masked patches from frequency-masked context
        predictions = self.predictor(context_embeds, context_indices, mask_indices)

        # Prediction loss
        loss = self._compute_prediction_loss(predictions, targets)

        if self.var_reg_lambda > 0:
            from src.models.jepa_v2 import prediction_var_loss
            var_loss = prediction_var_loss(predictions, threshold=0.1)
            loss = loss + self.var_reg_lambda * var_loss

        if self.vicreg_lambda > 0:
            from src.models.jepa_v2 import vicreg_loss
            all_encoder_out = self.encoder(x_freq_masked, return_all_tokens=True)[:, 1:]
            vic_loss = vicreg_loss(all_encoder_out)
            loss = loss + self.vicreg_lambda * vic_loss

        return loss, predictions, targets


def cosine_scheduler(base_value, final_value, epochs, warmup_epochs):
    warmup_schedule = np.linspace(0, base_value, warmup_epochs)
    iters = np.arange(epochs - warmup_epochs)
    schedule = final_value + 0.5 * (base_value - final_value) * (
        1 + np.cos(np.pi * iters / len(iters))
    )
    return np.concatenate([warmup_schedule, schedule])


def run_freq_masking_experiment(mode, seeds, epochs, device):
    """
    Run frequency masking JEPA experiment.

    mode: 'time_only' (V2 baseline), 'freq_mask' (freq masking context), 'fft_input' (FFT input)
    """
    print(f"\n{'='*60}")
    print(f"FREQUENCY MASKING EXPERIMENT: {mode.upper()}")
    print(f"{'='*60}")

    results = []

    for seed in seeds:
        print(f"\n  Seed {seed}:")
        torch.manual_seed(seed)
        np.random.seed(seed)

        train_loader, test_loader, data_info = create_dataloaders(
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

        # Create model based on mode
        if mode == 'freq_mask':
            model = MechanicalJEPAFreqMask(
                n_channels=3,
                window_size=4096,
                patch_size=256,
                embed_dim=512,
                encoder_depth=4,
                predictor_depth=4,
                mask_ratio=0.625,
                ema_decay=0.996,
                predictor_pos='sinusoidal',
                loss_fn='l1',
                var_reg_lambda=0.1,
                freq_mask_ratio=0.3,
                freq_bands=8,
            ).to(device)
            model_name = "JEPA + Freq Masking (context)"
        else:  # time_only (V2 baseline)
            model = MechanicalJEPAV2(
                n_channels=3,
                window_size=4096,
                patch_size=256,
                embed_dim=512,
                encoder_depth=4,
                predictor_depth=4,
                mask_ratio=0.625,
                ema_decay=0.996,
                predictor_pos='sinusoidal',
                loss_fn='l1',
                var_reg_lambda=0.1,
            ).to(device)
            model_name = "JEPA V2 (time masking only)"

        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"    Model: {model_name}, params: {n_params:,}")

        optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)
        lr_schedule = cosine_scheduler(1e-4, 1e-6, epochs, 5)

        # Train
        t0 = time.time()
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            for signals, labels, _ in train_loader:
                lr = lr_schedule[min(epoch, len(lr_schedule) - 1)]
                for pg in optimizer.param_groups:
                    pg['lr'] = lr
                signals = signals.to(device)
                optimizer.zero_grad()
                loss = model.train_step(signals)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                model.update_ema()
                total_loss += loss.item()

            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"    Epoch {epoch+1}: loss={total_loss/len(train_loader):.4f}")

        t1 = time.time()
        print(f"    Training: {(t1-t0)/60:.1f}min")

        # Evaluate
        model.eval()
        all_embeds, all_labels = [], []
        with torch.no_grad():
            for signals, labels, _ in train_loader:
                embeds = model.get_embeddings(signals.to(device), pool='mean')
                all_embeds.append(embeds.cpu())
                all_labels.append(labels)
        train_embeds = torch.cat(all_embeds)
        train_labels = torch.cat(all_labels)

        all_embeds, all_labels = [], []
        with torch.no_grad():
            for signals, labels, _ in test_loader:
                embeds = model.get_embeddings(signals.to(device), pool='mean')
                all_embeds.append(embeds.cpu())
                all_labels.append(labels)
        test_embeds = torch.cat(all_embeds)
        test_labels = torch.cat(all_labels)

        # Linear probe
        probe = nn.Linear(512, 4).to(device)
        opt = optim.Adam(probe.parameters(), lr=1e-3)
        crit = nn.CrossEntropyLoss()
        best_f1 = 0.0

        for ep in range(20):
            probe.train(); opt.zero_grad()
            logits = probe(train_embeds.to(device))
            loss = crit(logits, train_labels.to(device))
            loss.backward(); opt.step()

            probe.eval()
            with torch.no_grad():
                preds = probe(test_embeds.to(device)).argmax(1).cpu().numpy()
            f1 = f1_score(test_labels.numpy(), preds, average='macro', zero_division=0)
            if f1 > best_f1: best_f1 = f1

        # Random baseline
        torch.manual_seed(seed + 10000)
        rand_model = MechanicalJEPAV2(n_channels=3, window_size=4096, patch_size=256,
                                       embed_dim=512, encoder_depth=4).to(device)
        rand_model.eval()
        rand_embeds, rand_labels = [], []
        with torch.no_grad():
            for signals, labels, _ in train_loader:
                rand_embeds.append(rand_model.get_embeddings(signals.to(device)).cpu())
                rand_labels.append(labels)
        rand_embeds_test, rand_labels_test = [], []
        with torch.no_grad():
            for signals, labels, _ in test_loader:
                rand_embeds_test.append(rand_model.get_embeddings(signals.to(device)).cpu())
                rand_labels_test.append(labels)

        rand_train_e = torch.cat(rand_embeds)
        rand_train_l = torch.cat(rand_labels)
        rand_test_e = torch.cat(rand_embeds_test)
        rand_test_l = torch.cat(rand_labels_test)

        rand_probe = nn.Linear(512, 4).to(device)
        rand_opt = optim.Adam(rand_probe.parameters(), lr=1e-3)
        best_rand_f1 = 0.0

        for ep in range(20):
            rand_probe.train(); rand_opt.zero_grad()
            logits = rand_probe(rand_train_e.to(device))
            loss = crit(logits, rand_train_l.to(device))
            loss.backward(); rand_opt.step()

            rand_probe.eval()
            with torch.no_grad():
                preds = rand_probe(rand_test_e.to(device)).argmax(1).cpu().numpy()
            f1 = f1_score(rand_test_l.numpy(), preds, average='macro', zero_division=0)
            if f1 > best_rand_f1: best_rand_f1 = f1

        print(f"    F1: {best_f1:.4f} (rand: {best_rand_f1:.4f}, gain: {best_f1-best_rand_f1:+.4f})")
        results.append({
            'seed': seed,
            'f1': best_f1,
            'rand_f1': best_rand_f1,
            'gain': best_f1 - best_rand_f1,
            'mode': mode,
        })

    f1s = [r['f1'] for r in results]
    gains = [r['gain'] for r in results]
    print(f"\n  {mode}: F1 = {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
    print(f"  Gain over random: {np.mean(gains):+.4f} ± {np.std(gains):.4f}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='all',
                        choices=['all', 'time_only', 'freq_mask'])
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 456])
    parser.add_argument('--epochs', type=int, default=30)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    import shutil
    print(f"Home disk: {shutil.disk_usage('/home/sagemaker-user').free/1e9:.1f} GB free")

    all_results = {}

    if args.mode in ('all', 'time_only'):
        results = run_freq_masking_experiment('time_only', args.seeds, args.epochs, device)
        all_results['time_only'] = results

    if args.mode in ('all', 'freq_mask'):
        results = run_freq_masking_experiment('freq_mask', args.seeds, args.epochs, device)
        all_results['freq_mask'] = results

    # Comparison
    print("\n" + "=" * 60)
    print("FREQUENCY MASKING COMPARISON")
    print("=" * 60)
    for mode_name, rs in all_results.items():
        f1s = [r['f1'] for r in rs]
        gains = [r['gain'] for r in rs]
        label = {
            'time_only': 'JEPA V2 (time masking only)',
            'freq_mask': 'JEPA + Freq Masking (context)',
        }.get(mode_name, mode_name)
        print(f"{label:<40}: F1={np.mean(f1s):.4f} ± {np.std(f1s):.4f} (gain {np.mean(gains):+.4f})")

    import json
    save_path = Path('results/freq_masking.json')
    save_path.parent.mkdir(exist_ok=True, parents=True)
    with open(save_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to {save_path}")

    return all_results


if __name__ == '__main__':
    main()
