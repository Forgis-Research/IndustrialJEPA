#!/usr/bin/env python3
"""
Forecasting Evaluation

Tests how well the pretrained encoder enables next-state prediction.

Evaluates:
1. Copy-last baseline (state_t -> state_t as prediction for state_{t+1})
2. Linear regression on (state_t, action_t) -> state_{t+1}
3. MLP on (state_t, action_t) -> state_{t+1}
4. Pretrained encoder + linear decoder (frozen)
5. Pretrained encoder + linear decoder (fine-tuned)
6. Transformer from scratch (same arch, random init)

Usage:
    python eval_forecasting.py --checkpoint checkpoints/best.pt
"""

import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from pretrain import MechanicalJEPA, StateEncoder, CONFIGS


# ============================================================================
# Synthetic Data with Actions
# ============================================================================

def generate_forecasting_data(n_episodes=1000, seq_len=32, n_joints=7, seed=42):
    """
    Generate (state, action, next_state) tuples for forecasting eval.

    Dynamics: next_state = state + action[:7] * gain + small_noise
    - Action is the dominant driver (gain=0.5)
    - Small noise (std=0.02) makes it non-trivially deterministic
    - Copy-last is a meaningful but not perfect baseline

    This creates a well-conditioned prediction problem where:
    - Linear regression on (state, action) should work well
    - Models that understand the action-state relationship win
    """
    np.random.seed(seed)

    q = np.random.uniform(-2.0, 2.0, (n_episodes, n_joints)).astype(np.float32)
    all_states = [q.copy()]
    all_actions = []

    action_gain = 0.5  # How much action drives state change
    step_noise = 0.02  # Small stochastic noise (determinism dominates)

    for t in range(seq_len):
        # Action: desired joint velocity command
        action = np.random.randn(n_episodes, 7).astype(np.float32) * 0.3
        all_actions.append(action)

        # Dynamics: mostly deterministic, action-driven
        noise = np.random.randn(n_episodes, n_joints).astype(np.float32) * step_noise
        q = q + action[:, :n_joints] * action_gain + noise
        q = np.clip(q, -3.14, 3.14)
        all_states.append(q.copy())

    states = np.stack(all_states, axis=1)  # (n_ep, seq_len+1, n_joints)
    actions = np.stack(all_actions, axis=1)  # (n_ep, seq_len, 7)

    s_t = states[:, :-1].reshape(-1, n_joints)       # (N, 7)
    a_t = actions.reshape(-1, 7)                      # (N, 7)
    s_next = states[:, 1:].reshape(-1, n_joints)      # (N, 7)

    return (torch.tensor(s_t), torch.tensor(a_t), torch.tensor(s_next))


def generate_sequence_dataset(n_episodes=1000, seq_len=32, n_joints=7, seed=42):
    """Generate sequences for encoder-based evaluation (vectorized, fast)."""
    np.random.seed(seed)

    q = np.random.uniform(-2.0, 2.0, (n_episodes, n_joints)).astype(np.float32)
    all_states = []
    all_actions = []

    for t in range(seq_len):
        all_states.append(q.copy())
        action = np.random.randn(n_episodes, 7).astype(np.float32) * 0.3
        all_actions.append(action)

        noise = np.random.randn(n_episodes, n_joints).astype(np.float32) * 0.02
        q = q + action[:, :n_joints] * 0.5 + noise
        q = np.clip(q, -3.14, 3.14)

    states = np.stack(all_states, axis=1)   # (n_ep, seq_len, 7)
    actions = np.stack(all_actions, axis=1)  # (n_ep, seq_len, 7)

    return (torch.tensor(states), torch.tensor(actions))


# ============================================================================
# Baseline Models
# ============================================================================

class CopyLastBaseline:
    """Predict state_{t+1} = state_t"""
    def predict(self, s_t, a_t):
        return s_t


class LinearBaseline(nn.Module):
    """Linear regression on (state, action) -> next_state"""
    def __init__(self, state_dim=7, action_dim=7):
        super().__init__()
        self.fc = nn.Linear(state_dim + action_dim, state_dim)

    def forward(self, s_t, a_t):
        x = torch.cat([s_t, a_t], dim=-1)
        return self.fc(x)


class MLPBaseline(nn.Module):
    """2-layer MLP on (state, action) -> next_state"""
    def __init__(self, state_dim=7, action_dim=7, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, state_dim),
        )

    def forward(self, s_t, a_t):
        x = torch.cat([s_t, a_t], dim=-1)
        return self.net(x)


# ============================================================================
# Encoder-Based Models
# ============================================================================

class EncoderForecaster(nn.Module):
    """
    Pretrained encoder + linear decoder for next-state prediction.

    Uses the last timestep embedding to predict next state.
    """
    def __init__(self, encoder, d_model, state_dim=7, freeze_encoder=True):
        super().__init__()
        self.encoder = encoder
        self.decoder = nn.Linear(d_model, state_dim)

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

    def forward(self, state_seq):
        # state_seq: (batch, seq_len, state_dim)
        z = self.encoder(state_seq)  # (batch, seq_len, d_model)
        z_last = z[:, -1]  # Last timestep embedding
        return self.decoder(z_last)


# ============================================================================
# Training Helpers
# ============================================================================

def train_model(model, s_t, a_t, s_next, n_epochs=200, lr=1e-2, batch_size=256):
    """Train a prediction model."""
    n = len(s_t)
    n_train = int(0.8 * n)
    perm = torch.randperm(n)
    train_idx = perm[:n_train]
    val_idx = perm[n_train:]

    s_train, a_train, y_train = s_t[train_idx], a_t[train_idx], s_next[train_idx]
    s_val, a_val, y_val = s_t[val_idx], a_t[val_idx], s_next[val_idx]

    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=lr
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    best_val_mse = float('inf')
    for epoch in range(n_epochs):
        model.train()
        perm2 = torch.randperm(n_train)
        for i in range(0, n_train, batch_size):
            idx = perm2[i:i+batch_size]
            pred = model(s_train[idx], a_train[idx])
            loss = F.mse_loss(pred, y_train[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

        # Track best
        model.eval()
        with torch.no_grad():
            pred = model(s_val, a_val)
            val_mse = F.mse_loss(pred, y_val).item()
        best_val_mse = min(best_val_mse, val_mse)

    return best_val_mse


def train_encoder_forecaster(forecaster, states, s_next_seq, n_epochs=50, lr=1e-3,
                              batch_size=64, window=32, device='cpu'):
    """Train encoder-based forecaster using sequence windows."""
    # states: (n_ep, seq_len, state_dim)
    # Create (window, next_state) pairs
    n_ep, seq_len, state_dim = states.shape
    windows = []
    targets = []

    for ep in range(n_ep):
        for t in range(window, seq_len):
            windows.append(states[ep, t-window:t])
            targets.append(states[ep, t])

    windows = torch.stack(windows).to(device)   # (N, window, state_dim)
    targets = torch.stack(targets).to(device)    # (N, state_dim)

    forecaster = forecaster.to(device)

    n = len(windows)
    n_train = int(0.8 * n)
    perm = torch.randperm(n)
    train_w = windows[perm[:n_train]]
    train_t = targets[perm[:n_train]]
    val_w = windows[perm[n_train:]]
    val_t = targets[perm[n_train:]]

    optimizer = torch.optim.Adam(
        [p for p in forecaster.parameters() if p.requires_grad], lr=lr
    )

    for epoch in range(n_epochs):
        forecaster.train()
        perm2 = torch.randperm(n_train)
        for i in range(0, n_train, batch_size):
            idx = perm2[i:i+batch_size]
            pred = forecaster(train_w[idx])
            loss = F.mse_loss(pred, train_t[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    forecaster.eval()
    with torch.no_grad():
        pred = forecaster(val_w)
        val_mse = F.mse_loss(pred, val_t).item()

    return val_mse


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str,
                        default='autoresearch/mechanical_jepa/checkpoints/best.pt')
    parser.add_argument('--n-seeds', type=int, default=3)
    args = parser.parse_args()

    print("=" * 60)
    print("FORECASTING EVALUATION")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    # Load pretrained model
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = checkpoint['config']
    pretrained_model = MechanicalJEPA(state_dim=7, action_dim=7, **cfg).to(device)
    pretrained_model.load_state_dict(checkpoint['model_state_dict'])
    pretrained_model.eval()
    d_model = cfg['d_model']
    print(f"Loaded checkpoint: epoch {checkpoint['epoch']}, val_loss={checkpoint['val_loss']:.4f}")

    results = {name: [] for name in [
        'copy_last', 'linear', 'mlp',
        'encoder_frozen', 'encoder_finetuned',
        'transformer_scratch',
    ]}

    for seed in range(args.n_seeds):
        print(f"\n[Seed {seed+1}/{args.n_seeds}]")

        # Generate data
        s_t, a_t, s_next = generate_forecasting_data(
            n_episodes=2000, seq_len=128, seed=seed * 100
        )
        sequences, _ = generate_sequence_dataset(
            n_episodes=2000, seq_len=128, seed=seed * 100
        )

        print(f"  Data: {s_t.shape[0]:,} (state, action, next_state) pairs")

        # 1. Copy-last baseline
        baseline = CopyLastBaseline()
        n = len(s_t)
        n_val = int(0.2 * n)
        val_s = s_t[-n_val:]
        val_a = a_t[-n_val:]
        val_next = s_next[-n_val:]
        copy_mse = F.mse_loss(baseline.predict(val_s, val_a), val_next).item()
        results['copy_last'].append(copy_mse)

        # 2. Linear regression
        linear = LinearBaseline()
        linear_mse = train_model(linear, s_t, a_t, s_next, n_epochs=50, lr=1e-3)
        results['linear'].append(linear_mse)

        # 3. MLP
        mlp = MLPBaseline()
        mlp_mse = train_model(mlp, s_t, a_t, s_next, n_epochs=50, lr=1e-3)
        results['mlp'].append(mlp_mse)

        # 4. Pretrained encoder + frozen linear decoder
        frozen_forecaster = EncoderForecaster(
            pretrained_model.state_encoder, d_model, state_dim=7, freeze_encoder=True
        )
        frozen_mse = train_encoder_forecaster(
            frozen_forecaster, sequences, None, n_epochs=50, lr=1e-3, device=device
        )
        results['encoder_frozen'].append(frozen_mse)

        # 5. Pretrained encoder + fine-tuned linear decoder
        # Re-load to get a fresh copy to fine-tune
        finetuned_model = MechanicalJEPA(state_dim=7, action_dim=7, **cfg)
        finetuned_model.load_state_dict(checkpoint['model_state_dict'])
        finetuned_forecaster = EncoderForecaster(
            finetuned_model.state_encoder, d_model, state_dim=7, freeze_encoder=False
        )
        finetuned_mse = train_encoder_forecaster(
            finetuned_forecaster, sequences, None, n_epochs=50, lr=1e-4, device=device
        )
        results['encoder_finetuned'].append(finetuned_mse)

        # 6. Transformer from scratch (same arch, no pretraining)
        scratch_cfg = CONFIGS['small']
        scratch_encoder = StateEncoder(
            input_dim=7, d_model=scratch_cfg['d_model'],
            n_heads=scratch_cfg['n_heads'], n_layers=scratch_cfg['n_layers']
        )
        scratch_forecaster = EncoderForecaster(
            scratch_encoder, scratch_cfg['d_model'], state_dim=7, freeze_encoder=False
        )
        scratch_mse = train_encoder_forecaster(
            scratch_forecaster, sequences, None, n_epochs=50, lr=1e-4, device=device
        )
        results['transformer_scratch'].append(scratch_mse)

        print(f"  Copy-last:          MSE = {copy_mse:.5f}")
        print(f"  Linear:             MSE = {linear_mse:.5f}")
        print(f"  MLP:                MSE = {mlp_mse:.5f}")
        print(f"  Encoder (frozen):   MSE = {frozen_mse:.5f}")
        print(f"  Encoder (finetuned):MSE = {finetuned_mse:.5f}")
        print(f"  Transformer scratch:MSE = {scratch_mse:.5f}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY (mean +/- std over seeds)")
    print("=" * 60)
    print(f"\n{'Method':<30} {'Mean MSE':>12} {'Std':>10}")
    print("-" * 55)

    for name, vals in results.items():
        print(f"{name:<30} {np.mean(vals):>12.5f} {np.std(vals):>10.5f}")

    print()
    # Key comparison
    linear_mean = np.mean(results['linear'])
    finetuned_mean = np.mean(results['encoder_finetuned'])
    scratch_mean = np.mean(results['transformer_scratch'])

    print(f"Pretrained vs linear:  {finetuned_mean/linear_mean:.2f}x (want <1.0 to beat linear)")
    print(f"Pretrained vs scratch: {finetuned_mean/scratch_mean:.2f}x (want <1.0, pretraining helps)")

    return results


if __name__ == "__main__":
    results = main()
