#!/usr/bin/env python3
"""
Exp 50: KUKA Force/Contact Prediction

Given joint state sequence, predict force/contact signals.
Models: Persistence, Linear, MLP, CI-Transformer, Full-Attn Transformer,
        PhysMask Transformer, Action-conditioned variants.

Physics groups:
  joints [0-6]: joint_pos
  joint_vel [7-13]: joint_vel
  ee_state [14-19]: ee_pos(3) + ee_vel(3)
  forces [20-25]: force axes Fx,Fy,Fz,Tx,Ty,Tz
  contact [26]: contact signal
"""

import sys
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from pathlib import Path
from sklearn.metrics import roc_auc_score, accuracy_score

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
DATA_DIR = PROJECT_ROOT / "datasets" / "data" / "kuka_force"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEEDS = [42, 123, 456]

# Feature configuration
# Input: joint_pos(7) + joint_vel(7) + ee_pos(3) + ee_vel(3) = 20 channels
# Target: forces(6) + contact(1) = 7 channels
N_INPUT = 20
N_TARGET = 7
N_FORCE = 6
N_CONTACT = 1
SEQ_LEN = 30         # Use 30 of 50 timesteps as context window
PRED_HORIZON = 1     # Predict next timestep
BATCH_SIZE = 64
EPOCHS = 50
LR = 1e-3

# Physics groups (indices into the 27-channel full vector)
KUKA_GROUPS = {
    "joints":    list(range(0, 7)),      # joint_pos
    "joint_vel": list(range(7, 14)),     # joint_vel
    "ee_state":  list(range(14, 20)),    # ee_pos + ee_vel
    "forces":    list(range(20, 26)),    # Fx,Fy,Fz,Tx,Ty,Tz
    "contact":   [26],                   # contact
}


# ============================================================================
# Data
# ============================================================================

class KukaForceDataset(Dataset):
    """KUKA force prediction dataset.

    Input: (seq_len, 20) — joint_pos + joint_vel + ee_pos + ee_vel
    Target: (7,) — forces (6) + contact (1) at t+1
    Label: success (1) / fail (0) for linear probe
    """

    def __init__(self, joint_pos, joint_vel, ee_pos, ee_vel, forces, contact, success,
                 seq_len=SEQ_LEN, split='train', seed=42):
        rng = np.random.RandomState(seed)
        N = len(joint_pos)

        # Build input: joint_pos + joint_vel + ee_pos + ee_vel
        # shape: (N, 50, 20)
        X_full = np.concatenate([joint_pos, joint_vel, ee_pos, ee_vel], axis=2)  # (N, T, 20)
        Y_full = np.concatenate([forces, contact[:, :, None]], axis=2)            # (N, T, 7)

        # Build sliding windows
        T = X_full.shape[1]
        samples_X, samples_Y, samples_S = [], [], []

        for ep in range(N):
            for t in range(seq_len, T):
                window_X = X_full[ep, t-seq_len:t]     # (seq_len, 20)
                target_Y = Y_full[ep, t]                 # (7,)
                samples_X.append(window_X)
                samples_Y.append(target_Y)
                samples_S.append(success[ep])

        samples_X = np.array(samples_X, dtype=np.float32)
        samples_Y = np.array(samples_Y, dtype=np.float32)
        samples_S = np.array(samples_S, dtype=np.float32)

        # Split
        n = len(samples_X)
        idx = rng.permutation(n)
        n_train = int(0.7 * n)
        n_val = int(0.15 * n)

        if split == 'train':
            sel = idx[:n_train]
        elif split == 'val':
            sel = idx[n_train:n_train+n_val]
        else:  # test
            sel = idx[n_train+n_val:]

        self.X = torch.FloatTensor(samples_X[sel])
        self.Y = torch.FloatTensor(samples_Y[sel])
        self.S = torch.FloatTensor(samples_S[sel])

        # Normalize X
        self.X_mean = self.X.mean(dim=(0, 1), keepdim=True)
        self.X_std = self.X.std(dim=(0, 1), keepdim=True).clamp(min=1e-6)
        self.X = (self.X - self.X_mean) / self.X_std

        # Normalize Y (forces only, not contact)
        self.Y_force_mean = self.Y[:, :N_FORCE].mean(dim=0, keepdim=True)
        self.Y_force_std = self.Y[:, :N_FORCE].std(dim=0, keepdim=True).clamp(min=1e-6)

        print(f"  {split}: {len(sel)} samples")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx], self.S[idx]


def load_data(n_max=None):
    """Load KUKA force data from cache (real or synthetic)."""
    # Try real data first, then synthetic
    search_files = []
    for n in [300, 200, 100]:
        search_files.append(DATA_DIR / f"kuka_{n}ep.npz")
        search_files.append(DATA_DIR / f"kuka_synthetic_{n}ep.npz")

    for fp in search_files:
        if fp.exists():
            print(f"[Data] Loading {fp}...")
            d = np.load(fp, allow_pickle=True)
            data = {k: d[k] for k in d.files}
            if n_max:
                for k in data:
                    if isinstance(data[k], np.ndarray):
                        data[k] = data[k][:n_max]
            is_synthetic = 'synthetic' in fp.name
            print(f"  Loaded {len(data['success'])} episodes ({'synthetic' if is_synthetic else 'real'})")
            return data, is_synthetic

    print(f"[Data] No cache found in {DATA_DIR}. Using synthetic data.")
    from autoresearch.experiments.kuka.download_and_eda import generate_realistic_kuka
    data = generate_realistic_kuka(n_episodes=300, seed=42)
    return data, True


# ============================================================================
# Models
# ============================================================================

class PersistenceBaseline:
    """force_{t+1} = force_t (last value in sequence)."""
    def predict(self, X):
        # X: (B, seq_len, 20) — last step contains ee_state but not forces
        # For persistence we'd use the last force measurement
        # We don't have past forces in input, so we predict zeros
        return np.zeros((len(X), N_TARGET))


class LinearPredictor(nn.Module):
    """Linear regression from flattened input to forces+contact."""
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(SEQ_LEN * N_INPUT, N_TARGET)

    def forward(self, x):
        return self.fc(x.reshape(x.size(0), -1))


class MLPPredictor(nn.Module):
    """2-layer MLP from flattened input to forces+contact."""
    def __init__(self, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(SEQ_LEN * N_INPUT, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, N_TARGET)
        )

    def forward(self, x):
        return self.net(x.reshape(x.size(0), -1))


class CITransformerPredictor(nn.Module):
    """Channel-independent transformer for force prediction.

    Each channel is processed independently, then outputs are pooled.
    """
    def __init__(self, n_channels=N_INPUT, d=32, n_heads=4, n_layers=2, dropout=0.1):
        super().__init__()
        self.n = n_channels
        self.d = d
        self.proj = nn.Linear(1, d)
        self.pos = nn.Parameter(torch.randn(1, SEQ_LEN, d) * 0.02)
        el = nn.TransformerEncoderLayer(d, n_heads, d*4, dropout, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(el, n_layers)
        # Pool over time, then project all channels to output
        self.channel_agg = nn.Linear(d, 1)   # (d) -> (1) per channel
        self.out = nn.Linear(n_channels, N_TARGET)

    def forward(self, x):
        B, T, C = x.shape
        # Treat each channel independently: reshape to (B*C, T, 1)
        x = x.permute(0, 2, 1).reshape(B * C, T, 1)
        x = self.proj(x) + self.pos[:, :T]
        z = self.enc(x)
        # Pool over time
        z = z.mean(dim=1)              # (B*C, d)
        z = self.channel_agg(z)        # (B*C, 1)
        z = z.reshape(B, C)            # (B, C)
        return self.out(z)             # (B, N_TARGET)


class FullAttnTransformerPredictor(nn.Module):
    """Full cross-channel attention transformer."""
    def __init__(self, d=64, n_heads=4, n_layers=2, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(N_INPUT, d)
        self.pos = nn.Parameter(torch.randn(1, SEQ_LEN, d) * 0.02)
        el = nn.TransformerEncoderLayer(d, n_heads, d*4, dropout, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(el, n_layers)
        self.head = nn.Linear(d, N_TARGET)

    def forward(self, x):
        z = self.proj(x) + self.pos[:, :x.size(1)]
        z = self.enc(z)
        return self.head(z.mean(dim=1))


class PhysMaskTransformerPredictor(nn.Module):
    """Physics-masked transformer: groups joints, joint_vel, ee_state separately.

    Each physics group gets its own transformer, then group representations
    are aggregated with cross-group attention.
    """
    def __init__(self, d=32, n_heads=4, n_layers=2, dropout=0.1):
        super().__init__()
        # Groups in INPUT space (indices 0-19):
        # joints: 0-6 (7), joint_vel: 7-13 (7), ee_state: 14-19 (6)
        self.group_dims = [7, 7, 6]  # input groups
        self.group_names = ['joints', 'joint_vel', 'ee_state']

        # Per-group encoders
        self.group_projs = nn.ModuleList([nn.Linear(g, d) for g in self.group_dims])
        self.pos = nn.Parameter(torch.randn(1, SEQ_LEN, d) * 0.02)

        # Shared temporal encoder (applied per group)
        el = nn.TransformerEncoderLayer(d, n_heads, d*4, dropout, batch_first=True, norm_first=True)
        self.temp_enc = nn.TransformerEncoder(el, n_layers)

        # Cross-group attention
        n_groups = len(self.group_dims)
        cross_el = nn.TransformerEncoderLayer(d, n_heads, d*4, dropout, batch_first=True, norm_first=True)
        self.cross_enc = nn.TransformerEncoder(cross_el, 1)

        self.head = nn.Linear(d, N_TARGET)

    def forward(self, x):
        B, T, C = x.shape
        # Split into groups
        splits = [0, 7, 14, 20]
        group_reps = []
        for i, (start, end) in enumerate(zip(splits[:-1], splits[1:])):
            x_g = x[:, :, start:end]          # (B, T, g)
            z_g = self.group_projs[i](x_g) + self.pos[:, :T]  # (B, T, d)
            z_g = self.temp_enc(z_g)           # (B, T, d)
            group_reps.append(z_g.mean(dim=1, keepdim=True))  # (B, 1, d)

        # Cross-group: (B, n_groups, d)
        z_cross = torch.cat(group_reps, dim=1)
        z_cross = self.cross_enc(z_cross)
        z_pool = z_cross.mean(dim=1)           # (B, d)
        return self.head(z_pool)


class ActionConditionedWrapper(nn.Module):
    """Wraps a predictor with action conditioning."""
    def __init__(self, base_model, action_dim=4, d=32):
        super().__init__()
        self.base = base_model
        self.action_proj = nn.Sequential(
            nn.Linear(action_dim * SEQ_LEN, d),
            nn.GELU(),
            nn.Linear(d, N_TARGET)
        )
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, x, actions=None):
        base_out = self.base(x)
        if actions is not None:
            B = actions.size(0)
            action_feat = self.action_proj(actions.reshape(B, -1))
            return base_out + self.scale * action_feat
        return base_out


# ============================================================================
# Training
# ============================================================================

def train_one_epoch(model, loader, optimizer, has_actions=False):
    model.train()
    total_loss = 0
    n = 0
    for batch in loader:
        X, Y, S = [b.to(DEVICE) for b in batch]
        optimizer.zero_grad()
        pred = model(X)
        # MSE on forces, BCE on contact
        loss_force = F.mse_loss(pred[:, :N_FORCE], Y[:, :N_FORCE])
        loss_contact = F.binary_cross_entropy_with_logits(
            pred[:, N_FORCE:], Y[:, N_FORCE:])
        loss = loss_force + 0.1 * loss_contact
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        n += 1
    return total_loss / max(n, 1)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    force_mse_total = 0
    contact_probs, contact_true = [], []
    reps, labels = [], []
    n = 0

    for batch in loader:
        X, Y, S = [b.to(DEVICE) for b in batch]
        pred = model(X)

        force_mse_total += F.mse_loss(pred[:, :N_FORCE], Y[:, :N_FORCE]).item()

        # Contact prediction (threshold at 0)
        contact_prob = torch.sigmoid(pred[:, N_FORCE]).cpu().numpy()
        contact_probs.extend(contact_prob)
        contact_true.extend(Y[:, N_FORCE].cpu().numpy())

        # Representations for linear probe (use output before head if available)
        # For now use raw predictions as representation
        reps.append(pred.cpu().numpy())
        labels.append(S.cpu().numpy())

        n += 1

    force_mse = force_mse_total / max(n, 1)

    # Contact accuracy
    contact_pred_bin = (np.array(contact_probs) > 0.5).astype(float)
    contact_acc = accuracy_score(
        (np.array(contact_true) > 0.5).astype(int),
        contact_pred_bin.astype(int)
    )

    # Try AUROC for contact
    try:
        contact_auroc = roc_auc_score(
            (np.array(contact_true) > 0.5).astype(int),
            np.array(contact_probs)
        )
    except Exception:
        contact_auroc = 0.5

    reps_all = np.vstack(reps)
    labels_all = np.concatenate(labels)

    return {
        'force_mse': force_mse,
        'contact_acc': contact_acc,
        'contact_auroc': contact_auroc,
        'reps': reps_all,
        'labels': labels_all,
    }


def linear_probe_success(reps, labels):
    """Train linear probe on representations to predict episode success."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score

    if len(np.unique(labels)) < 2:
        return 0.5  # Can't evaluate with single class

    scaler = StandardScaler()
    reps_scaled = scaler.fit_transform(reps)

    clf = LogisticRegression(max_iter=500, C=1.0)
    try:
        scores = cross_val_score(clf, reps_scaled, labels.astype(int), cv=3, scoring='roc_auc')
        return float(scores.mean())
    except Exception:
        return 0.5


def run_experiment(model, train_loader, val_loader, test_loader,
                   model_name, seed, epochs=EPOCHS):
    """Train and evaluate one model."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = model.to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_mse = float('inf')
    best_state = None

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer)
        scheduler.step()

        if (epoch + 1) % 10 == 0:
            val_metrics = evaluate(model, val_loader)
            if val_metrics['force_mse'] < best_val_mse:
                best_val_mse = val_metrics['force_mse']
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

    # Restore best
    if best_state:
        model.load_state_dict(best_state)

    test_metrics = evaluate(model, test_loader)
    success_auroc = linear_probe_success(test_metrics['reps'], test_metrics['labels'])

    return {
        'model': model_name,
        'n_params': n_params,
        'seed': seed,
        'force_mse': test_metrics['force_mse'],
        'contact_acc': test_metrics['contact_acc'],
        'contact_auroc': test_metrics['contact_auroc'],
        'success_auroc': success_auroc,
    }


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 60)
    print("EXP 50: KUKA Force/Contact Prediction")
    print(f"Device: {DEVICE}")
    print("=" * 60)

    # Load data
    data, is_synthetic = load_data()

    joint_pos = data['joint_pos']    # (N, 50, 7)
    joint_vel = data['joint_vel']    # (N, 50, 7)
    ee_pos = data['ee_pos']          # (N, 50, 3)
    ee_vel = data['ee_vel']          # (N, 50, 3)
    forces = data['forces']          # (N, 50, 6)
    contact = data['contact']        # (N, 50)
    success = data['success']        # (N,)

    print(f"\nData: N={len(success)}, success_rate={success.mean():.3f}, synthetic={is_synthetic}")

    # Build datasets
    print("\nBuilding datasets...")
    train_ds = KukaForceDataset(joint_pos, joint_vel, ee_pos, ee_vel, forces, contact, success, split='train')
    val_ds   = KukaForceDataset(joint_pos, joint_vel, ee_pos, ee_vel, forces, contact, success, split='val')
    test_ds  = KukaForceDataset(joint_pos, joint_vel, ee_pos, ee_vel, forces, contact, success, split='test')

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE)
    test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE)

    # Define models to test
    model_configs = [
        ("Linear",          lambda: LinearPredictor()),
        ("MLP",             lambda: MLPPredictor(hidden=128)),
        ("CI-Transformer",  lambda: CITransformerPredictor(n_channels=N_INPUT, d=32, n_heads=4, n_layers=2)),
        ("Full-Attn",       lambda: FullAttnTransformerPredictor(d=64, n_heads=4, n_layers=2)),
        ("PhysMask",        lambda: PhysMaskTransformerPredictor(d=32, n_heads=4, n_layers=2)),
    ]

    all_results = []

    for model_name, model_fn in model_configs:
        print(f"\n--- {model_name} ---")
        seed_results = []
        for seed in SEEDS:
            model = model_fn()
            result = run_experiment(model, train_loader, val_loader, test_loader,
                                    model_name, seed, epochs=EPOCHS)
            seed_results.append(result)
            print(f"  seed={seed}: force_mse={result['force_mse']:.4f}, "
                  f"contact_acc={result['contact_acc']:.3f}, "
                  f"contact_auroc={result['contact_auroc']:.3f}, "
                  f"success_auroc={result['success_auroc']:.3f}")

        # Aggregate
        force_mses = [r['force_mse'] for r in seed_results]
        contact_accs = [r['contact_acc'] for r in seed_results]
        contact_aurocs = [r['contact_auroc'] for r in seed_results]
        success_aurocs = [r['success_auroc'] for r in seed_results]

        agg = {
            'model': model_name,
            'n_params': seed_results[0]['n_params'],
            'force_mse_mean': float(np.mean(force_mses)),
            'force_mse_std': float(np.std(force_mses)),
            'contact_acc_mean': float(np.mean(contact_accs)),
            'contact_auroc_mean': float(np.mean(contact_aurocs)),
            'success_auroc_mean': float(np.mean(success_aurocs)),
        }
        all_results.append(agg)
        print(f"  [{model_name}] force_mse={agg['force_mse_mean']:.4f}±{agg['force_mse_std']:.4f}, "
              f"contact_acc={agg['contact_acc_mean']:.3f}, "
              f"success_auroc={agg['success_auroc_mean']:.3f}")

    # Save results
    out_file = DATA_DIR / "force_prediction_results.json"
    with open(out_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[Done] Results saved to {out_file}")

    # Print summary table
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print(f"{'Model':<20} {'Params':>8} {'Force MSE':>12} {'Contact Acc':>13} {'Success AUROC':>14}")
    print("-" * 80)
    for r in all_results:
        print(f"{r['model']:<20} {r['n_params']:>8,} "
              f"{r['force_mse_mean']:>12.4f} "
              f"{r['contact_acc_mean']:>13.3f} "
              f"{r['success_auroc_mean']:>14.3f}")
    print("=" * 80)

    return all_results


def generate_synthetic_kuka(n_episodes=300, seq_len=50, seed=42):
    """Generate synthetic KUKA-like data when GCS is unavailable."""
    rng = np.random.RandomState(seed)
    print(f"[Synthetic] Generating {n_episodes} KUKA-like episodes...")

    joint_pos_all, joint_vel_all, ee_pos_all, ee_vel_all = [], [], [], []
    forces_all, contact_all, rewards_all, success_all = [], [], [], []

    for ep in range(n_episodes):
        # Simulate peg insertion: robot moves toward target, makes contact, inserts
        t = np.arange(seq_len)

        # Joint trajectories (smooth motion toward insertion pose)
        q0 = rng.uniform(-1.5, 1.5, 7).astype(np.float32)
        q_target = rng.uniform(-1.5, 1.5, 7).astype(np.float32)
        alpha = np.linspace(0, 1, seq_len)[:, None]
        joint_pos = q0[None] + alpha * (q_target[None] - q0[None])
        joint_pos += rng.randn(seq_len, 7).astype(np.float32) * 0.02
        joint_vel = np.diff(joint_pos, axis=0, prepend=joint_pos[:1])

        # EE state: approaching from above
        ee_z = np.linspace(0.5, 0.1, seq_len)  # descending
        ee_xy = rng.randn(seq_len, 2).astype(np.float32) * 0.01
        ee_pos = np.column_stack([ee_xy, ee_z]).astype(np.float32)
        ee_vel = np.diff(ee_pos, axis=0, prepend=ee_pos[:1])

        # Success probability based on endpoint accuracy
        success = rng.random() < 0.15  # 15% success rate

        # Force profile: increases during contact phase
        contact_start = int(seq_len * 0.5)  # contact at 50% of episode
        contact_sig = (t >= contact_start).astype(np.float32)

        if success:
            # Successful insertion: force ramps up then decreases (peg seated)
            fz_profile = np.where(t < contact_start, 0.0,
                         np.where(t < int(seq_len*0.8),
                                  (t - contact_start) * 0.3,
                                  (int(seq_len*0.8) - contact_start) * 0.3 * np.exp(-0.1*(t-int(seq_len*0.8)))))
        else:
            # Failed: force ramps up but doesn't seat (higher lateral forces)
            fz_profile = np.where(t < contact_start, 0.0,
                                  (t - contact_start) * 0.1)

        # 6-axis forces
        forces = np.zeros((seq_len, 6), dtype=np.float32)
        forces[:, 2] = fz_profile.astype(np.float32) + rng.randn(seq_len).astype(np.float32) * 0.05
        forces[:, :2] = rng.randn(seq_len, 2).astype(np.float32) * 0.1 * contact_sig[:, None]
        forces[:, 3:] = rng.randn(seq_len, 3).astype(np.float32) * 0.02

        joint_pos_all.append(joint_pos)
        joint_vel_all.append(joint_vel.astype(np.float32))
        ee_pos_all.append(ee_pos)
        ee_vel_all.append(ee_vel.astype(np.float32))
        forces_all.append(forces)
        contact_all.append(contact_sig)
        rewards_all.append(1.0 if success else 0.0)
        success_all.append(success)

    return {
        'joint_pos': np.stack(joint_pos_all),
        'joint_vel': np.stack(joint_vel_all),
        'ee_pos': np.stack(ee_pos_all),
        'ee_vel': np.stack(ee_vel_all),
        'forces': np.stack(forces_all),
        'contact': np.stack(contact_all),
        'rewards': np.array(rewards_all),
        'success': np.array(success_all),
        'lang': np.array(['insert the peg'] * n_episodes),
    }


if __name__ == "__main__":
    main()
