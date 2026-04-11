#!/usr/bin/env python3
"""
Embodiment Classification Evaluation

Tests if the pretrained encoder learns to distinguish robot embodiments.
Uses a frozen encoder + linear probe.

Usage:
    python eval_classification.py --checkpoint checkpoints/best.pt
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

# Import model from pretrain.py
sys.path.insert(0, str(Path(__file__).parent))
from pretrain import StateEncoder, MechanicalJEPA, CONFIGS


# ============================================================================
# Synthetic Multi-Robot Data
# ============================================================================

ROBOT_CONFIGS = {
    'A_fast_oscillating': {'type': 'sinusoidal', 'freq_range': (0.5, 2.0), 'amp': 1.5,
                           'noise': 0.05, 'n_joints': 7},
    'B_slow_drift':        {'type': 'drift', 'vel_decay': 0.95, 'accel': 0.005,
                            'limits': 1.0, 'n_joints': 7},
    'C_6dof_jerky':        {'type': 'jerky', 'jump_prob': 0.1, 'jump_scale': 0.5,
                            'limits': 2.5, 'n_joints': 6},
    'D_sim_sinusoidal':    {'type': 'sinusoidal', 'freq_range': (0.1, 0.3), 'amp': 0.4,
                            'noise': 0.01, 'n_joints': 7},
}


def generate_robot_trajectory(robot_name, n_episodes=200, seq_len=128, seed=None):
    """
    Generate synthetic trajectories with clearly distinct robot-specific characteristics.

    Each robot has a genuinely different dynamics type:
    - A: Fast sinusoidal oscillation (franka-like, high speed)
    - B: Slow momentum-based drift (heavy industrial)
    - C: Jerky 6-DOF motion (random jumps, zero-padded)
    - D: Slow sinusoidal with tight limits (sim panda)
    """
    if seed is not None:
        np.random.seed(seed)

    cfg = ROBOT_CONFIGS[robot_name]
    n_joints = cfg['n_joints']
    trajectories = []

    for _ in range(n_episodes):
        if cfg['type'] == 'sinusoidal':
            freq_lo, freq_hi = cfg['freq_range']
            t_arr = np.arange(seq_len) * 0.05
            freqs = np.random.uniform(freq_lo, freq_hi, n_joints)
            phases = np.random.uniform(0, 2 * np.pi, n_joints)
            traj = np.sin(t_arr[:, None] * freqs[None, :] + phases[None, :]) * cfg['amp']
            traj += np.random.randn(*traj.shape) * cfg['noise']

        elif cfg['type'] == 'drift':
            q = np.random.uniform(-cfg['limits'] * 0.5, cfg['limits'] * 0.5, n_joints)
            vel = np.zeros(n_joints)
            traj = []
            for t in range(seq_len):
                traj.append(q.copy())
                vel = cfg['vel_decay'] * vel + np.random.randn(n_joints) * cfg['accel']
                q = np.clip(q + vel, -cfg['limits'], cfg['limits'])
            traj = np.array(traj, dtype=np.float32)

        elif cfg['type'] == 'jerky':
            q = np.random.uniform(-cfg['limits'], cfg['limits'], n_joints)
            traj = []
            for t in range(seq_len):
                traj.append(q.copy())
                if np.random.rand() < cfg['jump_prob']:
                    q += np.random.randn(n_joints) * cfg['jump_scale']
                else:
                    q += np.random.randn(n_joints) * 0.02
                q = np.clip(q, -cfg['limits'], cfg['limits'])
            traj = np.array(traj, dtype=np.float32)
        else:
            raise ValueError(f"Unknown type: {cfg['type']}")

        traj = np.array(traj, dtype=np.float32)

        # Pad 6-DOF to 7-DOF
        if n_joints == 6:
            traj = np.pad(traj, ((0, 0), (0, 1)), mode='constant')

        trajectories.append(traj)

    return np.stack(trajectories)


def build_multi_robot_dataset(n_per_robot=300, seq_len=128, seed=42):
    """Build dataset with multiple robot types for classification."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    all_data = []
    all_labels = []
    robot_names = list(ROBOT_CONFIGS.keys())

    for label_idx, robot_name in enumerate(robot_names):
        trajs = generate_robot_trajectory(robot_name, n_episodes=n_per_robot, seq_len=seq_len)
        all_data.append(trajs)
        all_labels.extend([label_idx] * n_per_robot)

    data = np.concatenate(all_data, axis=0)
    labels = np.array(all_labels)

    # Shuffle
    perm = np.random.permutation(len(data))
    data = data[perm]
    labels = labels[perm]

    return torch.tensor(data), torch.tensor(labels), robot_names


# ============================================================================
# Linear Probe
# ============================================================================

class LinearProbe(nn.Module):
    def __init__(self, d_model, n_classes):
        super().__init__()
        self.fc = nn.Linear(d_model, n_classes)

    def forward(self, z):
        # z: (batch, seq_len, d_model) -> mean pool -> (batch, d_model)
        z_pooled = z.mean(dim=1)
        return self.fc(z_pooled)


def extract_embeddings(model, data, batch_size=64, device='cpu'):
    """Extract frozen embeddings from the encoder."""
    model.eval()
    all_embeddings = []
    with torch.no_grad():
        for i in range(0, len(data), batch_size):
            batch = data[i:i+batch_size].to(device)
            z = model.encode(batch)  # (batch, seq, d_model)
            all_embeddings.append(z.cpu())
    return torch.cat(all_embeddings, dim=0)


def train_linear_probe(embeddings, labels, n_classes, n_epochs=100, lr=1e-3):
    """Train a linear probe on frozen embeddings."""
    probe = LinearProbe(embeddings.shape[-1], n_classes)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)

    # Train/val split
    n = len(embeddings)
    n_train = int(0.8 * n)
    idx = torch.randperm(n)
    train_idx = idx[:n_train]
    val_idx = idx[n_train:]

    train_emb, train_lbl = embeddings[train_idx], labels[train_idx]
    val_emb, val_lbl = embeddings[val_idx], labels[val_idx]

    best_val_acc = 0
    for epoch in range(n_epochs):
        # Mini-batch training
        probe.train()
        perm = torch.randperm(n_train)
        for i in range(0, n_train, 64):
            batch_idx = perm[i:i+64]
            out = probe(train_emb[batch_idx])
            loss = F.cross_entropy(out, train_lbl[batch_idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validate
        probe.eval()
        with torch.no_grad():
            val_out = probe(val_emb)
            val_pred = val_out.argmax(dim=1)
            val_acc = (val_pred == val_lbl).float().mean().item()

        if val_acc > best_val_acc:
            best_val_acc = val_acc

    return best_val_acc


# ============================================================================
# Baselines
# ============================================================================

def raw_feature_classification(data, labels, n_classes):
    """Classify using raw state features (no encoder)."""
    # Flatten: mean + std over time
    mean_feat = data.mean(dim=1)  # (n, 7)
    std_feat = data.std(dim=1)    # (n, 7)
    features = torch.cat([mean_feat, std_feat], dim=-1)  # (n, 14)

    n = len(features)
    n_train = int(0.8 * n)
    idx = torch.randperm(n)
    train_feat, train_lbl = features[idx[:n_train]], labels[idx[:n_train]]
    val_feat, val_lbl = features[idx[n_train:]], labels[idx[n_train:]]

    probe = nn.Linear(features.shape[-1], n_classes)
    optimizer = torch.optim.Adam(probe.parameters(), lr=1e-3)

    for _ in range(200):
        perm = torch.randperm(n_train)
        for i in range(0, n_train, 64):
            batch_idx = perm[i:i+64]
            out = probe(train_feat[batch_idx])
            loss = F.cross_entropy(out, train_lbl[batch_idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    probe.eval()
    with torch.no_grad():
        val_pred = probe(val_feat).argmax(dim=1)
        return (val_pred == val_lbl).float().mean().item()


def random_encoder_classification(d_model, data, labels, n_classes):
    """Classify using random (untrained) encoder."""
    cfg = CONFIGS['small']
    random_model = MechanicalJEPA(state_dim=7, action_dim=7, **cfg)
    random_model.eval()
    with torch.no_grad():
        embeddings = []
        for i in range(0, len(data), 64):
            z = random_model.encode(data[i:i+64])
            embeddings.append(z)
        embeddings = torch.cat(embeddings, dim=0)
    return train_linear_probe(embeddings, labels, n_classes)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str,
                        default='autoresearch/mechanical_jepa/checkpoints/best.pt')
    parser.add_argument('--n-seeds', type=int, default=3)
    parser.add_argument('--n-per-robot', type=int, default=300)
    args = parser.parse_args()

    print("=" * 60)
    print("EMBODIMENT CLASSIFICATION EVALUATION")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load model
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = checkpoint['config']
    model = MechanicalJEPA(state_dim=7, action_dim=7, **cfg).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Loaded checkpoint: epoch {checkpoint['epoch']}, val_loss={checkpoint['val_loss']:.4f}")

    n_classes = len(ROBOT_CONFIGS)
    chance_acc = 1.0 / n_classes
    print(f"\nTask: {n_classes}-way embodiment classification")
    print(f"Chance accuracy: {chance_acc:.1%}\n")

    all_pretrained_accs = []
    all_random_accs = []
    all_raw_accs = []

    for seed in range(args.n_seeds):
        print(f"[Seed {seed+1}/{args.n_seeds}]")

        # Build data
        data, labels, robot_names = build_multi_robot_dataset(
            n_per_robot=args.n_per_robot, seq_len=128, seed=seed * 100
        )
        print(f"  Data: {data.shape}, Labels: {labels.shape}")
        print(f"  Robots: {robot_names}")

        # Extract pretrained embeddings
        embeddings = extract_embeddings(model, data, device=device)

        # 1. Pretrained encoder probe
        pretrained_acc = train_linear_probe(embeddings, labels, n_classes)
        all_pretrained_accs.append(pretrained_acc)

        # 2. Random encoder probe
        random_acc = random_encoder_classification(cfg['d_model'], data, labels, n_classes)
        all_random_accs.append(random_acc)

        # 3. Raw features
        raw_acc = raw_feature_classification(data, labels, n_classes)
        all_raw_accs.append(raw_acc)

        print(f"  Pretrained probe: {pretrained_acc:.1%}")
        print(f"  Random encoder:   {random_acc:.1%}")
        print(f"  Raw features:     {raw_acc:.1%}")
        print()

    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\n{'Method':<25} {'Mean Acc':>10} {'Std':>8}")
    print("-" * 45)
    print(f"{'Chance':<25} {chance_acc:>10.1%} {'':>8}")
    print(f"{'Random encoder':<25} {np.mean(all_random_accs):>10.1%} {np.std(all_random_accs):>8.3f}")
    print(f"{'Raw features':<25} {np.mean(all_raw_accs):>10.1%} {np.std(all_raw_accs):>8.3f}")
    print(f"{'Pretrained encoder':<25} {np.mean(all_pretrained_accs):>10.1%} {np.std(all_pretrained_accs):>8.3f}")

    pretrained_mean = np.mean(all_pretrained_accs)
    print(f"\nPass criterion: >50% (chance=25%)")
    if pretrained_mean > 0.50:
        print(f"PASSED: {pretrained_mean:.1%} > 50%")
    else:
        print(f"FAILED: {pretrained_mean:.1%} < 50%")

    return {
        'pretrained_acc': np.mean(all_pretrained_accs),
        'pretrained_std': np.std(all_pretrained_accs),
        'random_acc': np.mean(all_random_accs),
        'raw_acc': np.mean(all_raw_accs),
        'chance': chance_acc,
    }


if __name__ == "__main__":
    results = main()
