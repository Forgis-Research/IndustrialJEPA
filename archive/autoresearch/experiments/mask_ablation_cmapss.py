#!/usr/bin/env python3
"""
Mask ablation on C-MAPSS: Physics mask vs random mask vs full attention.
Same architecture (PhysMask-style), different masks.
"""
import sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from scipy import stats

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sys.path.insert(0, '/home/sagemaker-user/IndustrialJEPA/autoresearch/experiments')
from tier2_cmapss import (load_cmapss, TrainDS, TestDS, train_m, ev,
                           N_CH, SEQ_LEN, GROUP_IDX, COMPONENT_GROUPS, SC)

SEEDS = [42, 123, 456, 789, 1234]  # 5 seeds


class MaskedCMAPSS(nn.Module):
    """PhysMask-style model for C-MAPSS with configurable mask."""
    def __init__(self, mask_type="physics", d=32, nh=4, nl=2, do=0.1):
        super().__init__()
        self.d = d
        self.proj = nn.Linear(1, d)
        self.pos = nn.Parameter(torch.randn(1, SEQ_LEN, d)*0.02)
        el = nn.TransformerEncoderLayer(d, nh, d*4, do, batch_first=True)
        self.te = nn.TransformerEncoder(el, nl)
        self.ce = nn.Parameter(torch.randn(1, N_CH, d)*0.02)
        cl = nn.TransformerEncoderLayer(d, nh, d*4, do, batch_first=True)
        self.se = nn.TransformerEncoder(cl, 1)
        self.register_buffer('gmask', self._build_mask(mask_type))
        self.out = nn.Linear(N_CH*d, 1)

    def _build_mask(self, mask_type):
        if mask_type == "physics":
            m = torch.ones(N_CH, N_CH, dtype=torch.bool)
            for idx in GROUP_IDX.values():
                for i in idx:
                    for j in idx: m[i,j] = False
            reps = [idx[0] for idx in GROUP_IDX.values()]
            for i in reps:
                for j in reps: m[i,j] = False
            return m
        elif mask_type == "full":
            return torch.zeros(N_CH, N_CH, dtype=torch.bool)
        elif mask_type == "ci":
            m = torch.ones(N_CH, N_CH, dtype=torch.bool)
            for i in range(N_CH): m[i,i] = False
            return m
        elif mask_type.startswith("random"):
            seed = int(mask_type.split("_")[1]) if "_" in mask_type else 0
            rng = torch.Generator().manual_seed(seed)
            # Same sparsity as physics mask
            physics_m = self._build_mask("physics")
            sparsity = physics_m.float().mean().item()
            m = torch.rand(N_CH, N_CH, generator=rng) < sparsity
            for i in range(N_CH): m[i,i] = False
            return m.bool()
        elif mask_type == "wrong":
            # Block within-component, allow cross-component
            m = torch.zeros(N_CH, N_CH, dtype=torch.bool)
            for idx in GROUP_IDX.values():
                for i in idx:
                    for j in idx:
                        if i != j: m[i,j] = True
            return m
        else:
            raise ValueError(f"Unknown: {mask_type}")

    def forward(self, x):
        B, T, C = x.shape
        x = x.permute(0,2,1).reshape(B*C, T, 1)
        x = self.proj(x) + self.pos[:,:T]
        x = self.te(x)[:, -1].reshape(B, C, self.d)
        x = x + self.ce
        x = self.se(x, mask=self.gmask)
        return self.out(x.reshape(B, -1))


def run():
    print("="*70)
    print("C-MAPSS MASK ABLATION")
    print(f"Time: {time.strftime('%H:%M')} | Device: {DEVICE}")
    print("="*70)

    fd1 = load_cmapss("FD001")
    fd2 = load_cmapss("FD002")
    nm = fd1['ts'].mean(0); ns = fd1['ts'].std(0); ns[ns < 1e-8] = 1.0

    train_ds = TrainDS(fd1['tu'], fd1['ts'], fd1['tr'], nm, ns)
    nt = int(0.8*len(train_ds)); nv = len(train_ds) - nt
    tl1 = DataLoader(TestDS(fd1, nm, ns), 256)
    tl2 = DataLoader(TestDS(fd2, nm, ns), 256)

    conditions = ["physics", "full", "ci", "wrong", "random_0", "random_1", "random_2"]
    results = {}

    for cond in conditions:
        print(f"\n--- {cond} ---")
        r1a, r2a = [], []
        for seed in SEEDS:
            t0 = time.time()
            torch.manual_seed(seed); np.random.seed(seed)
            trn, val = torch.utils.data.random_split(
                train_ds, [nt, nv], generator=torch.Generator().manual_seed(seed))
            tl = DataLoader(trn, 256, shuffle=True)
            vl = DataLoader(val, 256)
            m = MaskedCMAPSS(cond)
            m, _ = train_m(m, tl, vl)
            e1 = ev(m, tl1); r1a.append(e1)
            e2 = ev(m, tl2); r2a.append(e2)
            print(f"  [{seed}] FD001={e1:.2f} FD002={e2:.2f} ({time.time()-t0:.0f}s)")
            sys.stdout.flush()
        r1 = np.array(r1a); r2 = np.array(r2a)
        results[cond] = {'fd001': r1, 'fd002': r2}
        print(f"  AVG: FD001={r1.mean():.2f}±{r1.std():.2f} FD002={r2.mean():.2f}±{r2.std():.2f}")

    print("\n" + "="*70)
    print("C-MAPSS MASK ABLATION SUMMARY")
    print("="*70)
    print(f"{'Condition':<15} {'FD001':>10} {'FD002':>10} {'Ratio':>8}")
    print("-"*45)
    for cond in conditions:
        r = results[cond]
        ratio = r['fd002'].mean() / r['fd001'].mean()
        print(f"{cond:<15} {r['fd001'].mean():>10.2f} {r['fd002'].mean():>10.2f} {ratio:>8.2f}")

    print("\n--- Physics vs Others (FD002) ---")
    phys = results['physics']['fd002']
    for cond in ["full", "wrong", "random_0", "random_1", "random_2"]:
        other = results[cond]['fd002']
        t_stat, p_val = stats.ttest_ind(phys, other)
        better = "physics" if phys.mean() < other.mean() else cond
        print(f"  Physics ({phys.mean():.1f}) vs {cond} ({other.mean():.1f}): t={t_stat:.2f}, p={p_val:.3f} ({better})")

    rnd = np.concatenate([results[f'random_{i}']['fd002'] for i in range(3)])
    t_stat, p_val = stats.ttest_ind(phys, rnd)
    print(f"\n  Physics vs random avg: t={t_stat:.2f}, p={p_val:.3f}")

    print(f"\nDone! Time: {time.strftime('%H:%M')}")
    return results


if __name__ == "__main__":
    run()
