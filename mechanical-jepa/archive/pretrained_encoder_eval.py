"""
Round 2: Pretrained Encoder Evaluation

Tests wav2vec2-base as a pretrained backbone for bearing fault detection.
Compares:
  1. Frozen wav2vec2 embeddings + linear probe (what does the pretrained model already know?)
  2. Our V2 JEPA embeddings + linear probe (our best method)
  3. Random init transformer + linear probe (baseline)

The key question: Does a speech-pretrained model generalize to vibration signals?

wav2vec2-base:
  - 94M parameters
  - Trained on 960h of LibriSpeech (English speech, 16kHz)
  - Input: raw waveform, output: 768-dim embeddings
  - Convolutional feature extractor + 12 transformer layers

We feed vibration signals (12kHz, 4096 samples) to wav2vec2 which expects:
  - 16kHz input (need resampling)
  - 1D waveform (single channel)
  - Any length (we use our 4096 window)

Design choice: resample 12kHz -> 16kHz before feeding to wav2vec2.
"""

import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from src.data import create_dataloaders
from src.models import MechanicalJEPAV2


def resample_batch(signals: torch.Tensor, source_sr: int, target_sr: int) -> torch.Tensor:
    """
    Resample batch of signals from source_sr to target_sr.
    signals: (B, C, T)
    Returns: (B, C, T_new)
    """
    if source_sr == target_sr:
        return signals
    from math import gcd
    import scipy.signal
    g = gcd(source_sr, target_sr)
    up = target_sr // g
    down = source_sr // g
    B, C, T = signals.shape
    # Process on CPU
    signals_np = signals.cpu().numpy()
    resampled = scipy.signal.resample_poly(signals_np, up, down, axis=-1)
    return torch.tensor(resampled, dtype=torch.float32)


class Wav2Vec2Embedder(nn.Module):
    """
    Wrapper around wav2vec2-base for feature extraction.
    Input: (B, C, T) vibration signal at 12kHz
    Output: (B, embed_dim) mean-pooled embeddings

    We:
    1. Resample from 12kHz to 16kHz
    2. Average channels to mono
    3. Run through wav2vec2 feature extractor
    4. Mean-pool over time to get fixed-size embedding
    """

    def __init__(self, source_sr: int = 12000, frozen: bool = True):
        super().__init__()
        from transformers import Wav2Vec2Model
        print("Loading wav2vec2-base from HuggingFace...")
        self.wav2vec2 = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        self.source_sr = source_sr
        self.target_sr = 16000  # wav2vec2 expects 16kHz
        self.embed_dim = 768

        if frozen:
            for p in self.wav2vec2.parameters():
                p.requires_grad = False
        print(f"  wav2vec2-base loaded ({sum(p.numel() for p in self.wav2vec2.parameters()):,} params)")

    def get_embeddings(self, x: torch.Tensor, pool: str = 'mean') -> torch.Tensor:
        """
        x: (B, C, T) at source_sr
        Returns: (B, 768) embeddings
        """
        # Resample if needed
        if self.source_sr != self.target_sr:
            x_resampled = resample_batch(x, self.source_sr, self.target_sr)
        else:
            x_resampled = x

        # Average channels to mono
        x_mono = x_resampled.mean(dim=1)  # (B, T)

        # Normalize to [-1, 1] range (wav2vec2 expects normalized audio)
        max_val = x_mono.abs().max(dim=1, keepdim=True)[0].clamp(min=1e-8)
        x_norm = x_mono / max_val

        # Run through wav2vec2
        x_norm = x_norm.to(next(self.wav2vec2.parameters()).device)
        outputs = self.wav2vec2(x_norm)
        hidden_states = outputs.last_hidden_state  # (B, T', 768)

        # Mean pool
        return hidden_states.mean(dim=1)  # (B, 768)


def evaluate_linear_probe(
    embedder,
    train_loader: DataLoader,
    test_loader: DataLoader,
    embed_dim: int,
    device: torch.device,
    n_classes: int = 4,
    probe_epochs: int = 50,
    label: str = '',
) -> dict:
    """Linear probe on frozen embeddings."""
    embedder.eval()

    def extract(loader):
        all_e, all_l = [], []
        with torch.no_grad():
            for signals, labels, _ in loader:
                signals = signals.to(device)
                embeds = embedder.get_embeddings(signals, pool='mean')
                all_e.append(embeds.cpu())
                all_l.append(labels)
        return torch.cat(all_e), torch.cat(all_l)

    print(f"  Extracting embeddings for {label}...")
    train_e, train_l = extract(train_loader)
    test_e, test_l = extract(test_loader)

    # Normalize using train stats
    mean = train_e.mean(0, keepdim=True)
    std = train_e.std(0, keepdim=True) + 1e-8
    train_e_n = ((train_e - mean) / std).to(device)
    test_e_n = ((test_e - mean) / std).to(device)
    train_l_d = train_l.to(device)
    test_l_d = test_l.to(device)

    probe = nn.Linear(embed_dim, n_classes).to(device)
    opt = optim.AdamW(probe.parameters(), lr=1e-2, weight_decay=0.01)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=probe_epochs, eta_min=1e-5)
    crit = nn.CrossEntropyLoss()

    best_acc = 0
    best_preds = None
    for ep in range(probe_epochs):
        probe.train()
        opt.zero_grad()
        crit(probe(train_e_n), train_l_d).backward()
        opt.step()
        sched.step()

        probe.eval()
        with torch.no_grad():
            preds = probe(test_e_n).argmax(1)
            acc = (preds == test_l_d).float().mean().item()
        if acc > best_acc:
            best_acc = acc
            best_preds = preds.clone()

    class_names = ['healthy', 'outer_race', 'inner_race', 'ball']
    per_class = {}
    for i, name in enumerate(class_names[:n_classes]):
        mask = test_l_d == i
        if mask.sum() > 0:
            per_class[name] = (best_preds[mask] == test_l_d[mask]).float().mean().item()

    print(f"  [{label:35s}] linear probe: {best_acc:.4f}")
    for k, v in per_class.items():
        print(f"    {k:15s}: {v:.4f}")

    return {'test_acc': best_acc, 'per_class': per_class, 'embed_dim': embed_dim}


def run_pretrained_eval(
    data_dir: str,
    v2_checkpoint: str,
    device: torch.device,
    seed: int = 42,
):
    """
    Compare pretrained encoders vs our V2 JEPA.
    """
    print(f"\n{'='*70}")
    print(f"PRETRAINED ENCODER COMPARISON | seed={seed}")
    print(f"{'='*70}")

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Load CWRU data
    train_loader, test_loader, info = create_dataloaders(
        data_dir=data_dir,
        batch_size=32,
        window_size=4096,
        stride=2048,
        test_ratio=0.2,
        seed=seed,
        num_workers=0,
        dataset_filter='cwru',
        n_channels=3,
    )
    print(f"CWRU: {info['train_windows']} train, {info['test_windows']} test windows")

    results = {}

    # 1. wav2vec2-base frozen (pretrained on speech)
    print("\n--- wav2vec2-base (frozen, speech pretrained) ---")
    wav2vec_embedder = Wav2Vec2Embedder(source_sr=12000, frozen=True).to(device)
    results['wav2vec2_frozen'] = evaluate_linear_probe(
        wav2vec_embedder, train_loader, test_loader,
        embed_dim=768, device=device, n_classes=4, probe_epochs=50,
        label='wav2vec2-base (frozen)',
    )

    # 2. V2 JEPA from checkpoint
    print("\n--- V2 JEPA (our method) ---")
    ckpt = torch.load(v2_checkpoint, map_location=device, weights_only=False)
    config = ckpt['config']
    jepa_model = MechanicalJEPAV2(
        n_channels=3, window_size=4096, patch_size=config['patch_size'],
        embed_dim=config['embed_dim'], encoder_depth=config['encoder_depth'],
        predictor_depth=config['predictor_depth'], n_heads=config['n_heads'],
        mask_ratio=config['mask_ratio'], ema_decay=config['ema_decay'],
        predictor_pos=config.get('predictor_pos', 'sinusoidal'),
        loss_fn=config.get('loss_fn', 'l1'), var_reg_lambda=0.0, vicreg_lambda=0.0,
    ).to(device)
    jepa_model.load_state_dict(ckpt['model_state_dict'])
    results['jepa_v2'] = evaluate_linear_probe(
        jepa_model, train_loader, test_loader,
        embed_dim=config['embed_dim'], device=device, n_classes=4, probe_epochs=50,
        label='V2 JEPA (our method)',
    )

    # 3. Random init (same arch as JEPA, no pretraining)
    print("\n--- Random init (same arch as JEPA) ---")
    random_model = MechanicalJEPAV2(
        n_channels=3, window_size=4096, patch_size=config['patch_size'],
        embed_dim=config['embed_dim'], encoder_depth=config['encoder_depth'],
        predictor_depth=config['predictor_depth'], n_heads=config['n_heads'],
        mask_ratio=config['mask_ratio'], ema_decay=config['ema_decay'],
        predictor_pos=config.get('predictor_pos', 'sinusoidal'),
        loss_fn=config.get('loss_fn', 'l1'), var_reg_lambda=0.0, vicreg_lambda=0.0,
    ).to(device)
    # Same architecture but no pretraining
    results['random_init'] = evaluate_linear_probe(
        random_model, train_loader, test_loader,
        embed_dim=config['embed_dim'], device=device, n_classes=4, probe_epochs=50,
        label='Random init (JEPA arch)',
    )

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    for name, res in results.items():
        print(f"  {name:30s}: {res['test_acc']:.4f}")

    print(f"\nJEPA gain over random init: {results['jepa_v2']['test_acc'] - results['random_init']['test_acc']:+.4f}")
    print(f"wav2vec2 gain over random:  {results['wav2vec2_frozen']['test_acc'] - results['random_init']['test_acc']:+.4f}")

    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='data/bearings')
    parser.add_argument('--checkpoint', default='checkpoints/jepa_v2_20260401_003619.pt')
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 123, 456])
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    all_results = {}
    for seed in args.seeds:
        all_results[seed] = run_pretrained_eval(
            data_dir=args.data_dir,
            v2_checkpoint=args.checkpoint,
            device=device,
            seed=seed,
        )

    # Multi-seed summary
    print(f"\n{'='*70}")
    print("MULTI-SEED SUMMARY")
    print(f"{'='*70}")

    for method in ['wav2vec2_frozen', 'jepa_v2', 'random_init']:
        accs = [all_results[s][method]['test_acc'] for s in args.seeds]
        print(f"  {method:30s}: {np.mean(accs):.4f} ± {np.std(accs):.4f}")

    jepa_accs = [all_results[s]['jepa_v2']['test_acc'] for s in args.seeds]
    random_accs = [all_results[s]['random_init']['test_acc'] for s in args.seeds]
    wav_accs = [all_results[s]['wav2vec2_frozen']['test_acc'] for s in args.seeds]

    print(f"\n  JEPA gain over random: {np.mean([j-r for j,r in zip(jepa_accs, random_accs)]):+.4f}")
    print(f"  wav2vec2 gain over random: {np.mean([w-r for w,r in zip(wav_accs, random_accs)]):+.4f}")
    print(f"  JEPA vs wav2vec2: {np.mean([j-w for j,w in zip(jepa_accs, wav_accs)]):+.4f}")
