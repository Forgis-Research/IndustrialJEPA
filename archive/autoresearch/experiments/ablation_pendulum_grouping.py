#!/usr/bin/env python3
"""
Ablation: Physics vs Random grouping on Pendulum.

On C-MAPSS, random ≈ physics. But pendulum has cleaner physics (2 masses, true independence).
Does physics grouping matter when the groups are truly physically independent?

Groups:
- physics: mass_1=[theta1, omega1], mass_2=[theta2, omega2]
- random_0: [theta1, theta2], [omega1, omega2]  (mixing positions vs velocities)
- random_1: [theta1, omega2], [omega1, theta2]  (mixing across masses)
"""
import sys, time, copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy import stats

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sys.path.insert(0, '/home/sagemaker-user/IndustrialJEPA/autoresearch/experiments')
from tier1_pendulum import (load_pendulum_data, TSDataset, PhysicsGrouped,
                             train_model, evaluate, LOOKBACK, HORIZON, N_CH, PHYSICS_GROUPS)

# Different grouping conditions for 4 channels
# Channels: 0=theta1, 1=omega1, 2=theta2, 3=omega2
GROUPINGS = {
    "physics": {"mass_1": [0, 1], "mass_2": [2, 3]},          # correct physics
    "type": {"angles": [0, 2], "velocities": [1, 3]},          # by measurement type
    "cross_1": {"mix_1": [0, 3], "mix_2": [1, 2]},             # theta1+omega2, omega1+theta2
    "cross_2": {"mix_1": [0, 2], "mix_2": [1, 3]},             # same as type actually — try different
    "singleton": {"g0": [0], "g1": [1], "g2": [2], "g3": [3]}, # per-channel (like CI)
    "all": {"all": [0, 1, 2, 3]},                               # single group (like full attn)
}

# Fix cross_2 to be truly different
GROUPINGS["cross_2"] = {"mix_1": [0, 3], "mix_2": [2, 1]}  # different from cross_1 order

SEEDS = [42, 123, 456, 789, 1234]  # 5 seeds for faster


class GroupedPendulum(nn.Module):
    """Pendulum model with configurable groups."""
    def __init__(self, groups, d=32, nh=4, nl=2, do=0.1):
        super().__init__()
        self.d, self.groups = d, groups
        self.nc = len(groups)
        self.proj = nn.Linear(1, d)
        self.pos = nn.Parameter(torch.randn(1, LOOKBACK, d)*0.02)
        el = nn.TransformerEncoderLayer(d, nh, d*4, do, batch_first=True)
        self.within = nn.TransformerEncoder(el, nl)
        self.pool = nn.Linear(d, d)
        self.cemb = nn.Parameter(torch.randn(1, self.nc, d)*0.02)
        cl = nn.TransformerEncoderLayer(d, nh, d*4, do, batch_first=True)
        self.cross = nn.TransformerEncoder(cl, 1)
        self.group_dec = nn.ModuleList([
            nn.Linear(d, HORIZON * len(idx)) for idx in groups.values()
        ])

    def forward(self, x):
        B, T, C = x.shape; cs = []
        for gn, idx in self.groups.items():
            ns = len(idx)
            cx = x[:,:,idx].permute(0,2,1).reshape(B*ns, T, 1)
            cx = self.proj(cx) + self.pos[:,:T]
            cx = self.within(cx)[:, -1].reshape(B, ns, -1)
            cs.append(self.pool(cx.mean(1)))
        s = torch.stack(cs, 1) + self.cemb
        s = self.cross(s)
        out = [None] * C
        for i, (gn, idx) in enumerate(self.groups.items()):
            dec = self.group_dec[i](s[:, i]).reshape(B, HORIZON, len(idx))
            for j, ci in enumerate(idx):
                out[ci] = dec[:, :, j]
        return torch.stack(out, dim=2)


def run_pendulum_ablation():
    print("="*70)
    print("PENDULUM GROUPING ABLATION")
    print(f"Time: {time.strftime('%H:%M')} | Device: {DEVICE}")
    print("="*70)
    sys.stdout.flush()

    source, target = load_pendulum_data()
    src_ds = TSDataset(source, max_samples=8000)
    mn, sd = src_ds.mean, src_ds.std
    tgt_ds = TSDataset(target, mean=mn, std=sd, max_samples=3000)
    src_l = DataLoader(src_ds, 256)
    tgt_l = DataLoader(tgt_ds, 256)

    n = len(src_ds); nt = int(0.8*n); nv = n - nt

    results = {}

    for cond_name, groups in GROUPINGS.items():
        if cond_name in ("singleton", "all"):
            # These have different #groups, skip for clean comparison
            # Actually let's include them for completeness
            pass

        print(f"\n{'='*50}")
        print(f"CONDITION: {cond_name}")
        print(f"  Groups: {groups}")
        print(f"{'='*50}")

        src_a, tgt_a = [], []
        for seed in SEEDS:
            t0 = time.time()
            torch.manual_seed(seed); np.random.seed(seed)
            trn, val = torch.utils.data.random_split(
                src_ds, [nt, nv], generator=torch.Generator().manual_seed(seed))
            tl = DataLoader(trn, 128, shuffle=True)
            vl = DataLoader(val, 256)

            m = GroupedPendulum(groups)
            m, _ = train_model(m, tl, vl)
            sm = evaluate(m, src_l); src_a.append(sm)
            tm = evaluate(m, tgt_l); tgt_a.append(tm)
            print(f"  [{seed}] src={sm:.6f} tgt={tm:.6f} ({time.time()-t0:.0f}s)")
            sys.stdout.flush()

        sa = np.array(src_a); ta = np.array(tgt_a)
        results[cond_name] = {'src': sa, 'tgt': ta, 'ratio': ta.mean()/sa.mean()}
        print(f"  AVG: src={sa.mean():.6f}±{sa.std():.6f} tgt={ta.mean():.6f}±{ta.std():.6f} ratio={ta.mean()/sa.mean():.2f}")

    # Summary
    print("\n" + "="*70)
    print("PENDULUM GROUPING ABLATION SUMMARY")
    print("="*70)
    print(f"{'Condition':<15} {'Src MSE':>12} {'Tgt MSE':>12} {'Ratio':>8}")
    print("-"*50)
    for name in ["physics", "type", "cross_1", "cross_2", "singleton", "all"]:
        if name in results:
            r = results[name]
            print(f"{name:<15} {r['src'].mean():>12.6f} {r['tgt'].mean():>12.6f} {r['ratio']:>8.2f}")

    # Statistical tests vs physics
    print("\n--- Physics vs Others (Target MSE) ---")
    phys_tgt = results['physics']['tgt']
    for rname in ["type", "cross_1", "cross_2", "singleton", "all"]:
        if rname in results:
            other_tgt = results[rname]['tgt']
            t, p = stats.ttest_ind(phys_tgt, other_tgt)
            better = "physics" if phys_tgt.mean() < other_tgt.mean() else rname
            print(f"  Physics vs {rname}: physics={phys_tgt.mean():.6f} {rname}={other_tgt.mean():.6f} "
                  f"t={t:.2f} p={p:.3f} (better: {better})")

    print(f"\nPendulum ablation complete! Time: {time.strftime('%H:%M')}")
    return results


if __name__ == "__main__":
    run_pendulum_ablation()
