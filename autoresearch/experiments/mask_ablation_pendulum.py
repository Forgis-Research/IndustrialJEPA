#!/usr/bin/env python3
"""
Mask ablation on Pendulum: Does the PHYSICS mask specifically matter?

We know PhysMask > Full-Attn on pendulum (7.4%, p=0.0002).
Question: Is this because the mask matches physics, or because any sparse mask helps?

Test:
1. PhysMask (correct physics: mass_1↔mass_1, mass_2↔mass_2)
2. RandomMask (random 50% sparsity)
3. WrongMask (anti-physics: mass_1↔mass_2, block within-mass)
4. Full-Attn (no mask)
5. CI (diagonal mask)
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
from tier1_pendulum import (load_pendulum_data, TSDataset, train_model, evaluate,
                             LOOKBACK, HORIZON, N_CH, PHYSICS_GROUPS)

SEEDS = [42, 123, 456, 789, 1234, 5678, 9012, 3456, 7890, 2468]


class MaskedAttnModel(nn.Module):
    """Same architecture as PhysicsGrouped but with configurable mask."""
    def __init__(self, mask_type="physics", d=32, nh=4, nl=2, do=0.1):
        super().__init__()
        self.d = d
        self.proj = nn.Linear(1, d)
        self.pos = nn.Parameter(torch.randn(1, LOOKBACK, d)*0.02)
        el = nn.TransformerEncoderLayer(d, nh, d*4, do, batch_first=True)
        self.time_enc = nn.TransformerEncoder(el, nl)
        self.ch_emb = nn.Parameter(torch.randn(1, N_CH, d)*0.02)
        cl = nn.TransformerEncoderLayer(d, nh, d*4, do, batch_first=True)
        self.space_enc = nn.TransformerEncoder(cl, 1)
        self.register_buffer('gmask', self._build_mask(mask_type))
        self.out = nn.Linear(d, HORIZON)

    def _build_mask(self, mask_type):
        if mask_type == "physics":
            # Within-mass + cross-mass representatives
            m = torch.ones(N_CH, N_CH, dtype=torch.bool)
            for idx in PHYSICS_GROUPS.values():
                for i in idx:
                    for j in idx: m[i,j] = False
            reps = [idx[0] for idx in PHYSICS_GROUPS.values()]
            for i in reps:
                for j in reps: m[i,j] = False
            return m
        elif mask_type == "wrong":
            # Anti-physics: BLOCK within-mass, ALLOW cross-mass
            m = torch.zeros(N_CH, N_CH, dtype=torch.bool)  # allow everything
            for idx in PHYSICS_GROUPS.values():
                for i in idx:
                    for j in idx:
                        if i != j: m[i,j] = True  # block within-mass (except self)
            return m
        elif mask_type == "full":
            return torch.zeros(N_CH, N_CH, dtype=torch.bool)  # no mask
        elif mask_type == "ci":
            m = torch.ones(N_CH, N_CH, dtype=torch.bool)
            for i in range(N_CH): m[i,i] = False  # only self-attention
            return m
        elif mask_type.startswith("random"):
            seed = int(mask_type.split("_")[1]) if "_" in mask_type else 0
            rng = torch.Generator().manual_seed(seed)
            m = torch.rand(N_CH, N_CH, generator=rng) > 0.5
            for i in range(N_CH): m[i,i] = False  # always allow self
            return m.bool()
        else:
            raise ValueError(f"Unknown mask type: {mask_type}")

    def forward(self, x):
        B, T, C = x.shape
        x = x.permute(0,2,1).reshape(B*C, T, 1)
        x = self.proj(x) + self.pos[:,:T]
        x = self.time_enc(x)[:, -1].reshape(B, C, self.d)
        x = x + self.ch_emb
        x = self.space_enc(x, mask=self.gmask)
        return self.out(x).permute(0, 2, 1)


def run_mask_ablation():
    print("="*70)
    print("MASK ABLATION: Does the physics mask specifically matter?")
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

    conditions = ["physics", "full", "ci", "wrong", "random_0", "random_1", "random_2"]
    results = {}

    for cond in conditions:
        print(f"\n--- {cond} ---")
        # Show mask
        test_m = MaskedAttnModel(cond)
        mask = test_m.gmask
        print(f"  Mask (True=blocked):\n  {mask.int().tolist()}")

        src_a, tgt_a = [], []
        for seed in SEEDS:
            t0 = time.time()
            torch.manual_seed(seed); np.random.seed(seed)
            trn, val = torch.utils.data.random_split(
                src_ds, [nt, nv], generator=torch.Generator().manual_seed(seed))
            tl = DataLoader(trn, 128, shuffle=True)
            vl = DataLoader(val, 256)

            m = MaskedAttnModel(cond)
            m, _ = train_model(m, tl, vl)
            sm = evaluate(m, src_l); src_a.append(sm)
            tm = evaluate(m, tgt_l); tgt_a.append(tm)

        sa = np.array(src_a); ta = np.array(tgt_a)
        results[cond] = {'src': sa, 'tgt': ta}
        print(f"  src={sa.mean():.6f}±{sa.std():.6f} tgt={ta.mean():.6f}±{ta.std():.6f}")
        sys.stdout.flush()

    # Summary
    print("\n" + "="*70)
    print("MASK ABLATION SUMMARY")
    print("="*70)
    print(f"{'Condition':<15} {'Tgt MSE':>12} {'± std':>10} {'vs Physics':>12}")
    print("-"*52)
    phys_tgt = results['physics']['tgt'].mean()
    for cond in conditions:
        t = results[cond]['tgt']
        delta = (1 - t.mean()/phys_tgt) * 100
        print(f"{cond:<15} {t.mean():>12.6f} {t.std():>10.6f} {delta:>11.1f}%")

    # Statistical tests
    print("\n--- Physics vs Others ---")
    phys = results['physics']['tgt']
    for cond in ["full", "wrong", "random_0", "random_1", "random_2"]:
        other = results[cond]['tgt']
        t_stat, p_val = stats.ttest_rel(phys, other)
        better = "physics" if phys.mean() < other.mean() else cond
        print(f"  Physics vs {cond}: t={t_stat:.2f}, p={p_val:.4f} (better: {better})")

    # Average random
    rnd = np.concatenate([results[f'random_{i}']['tgt'] for i in range(3)])
    t_stat, p_val = stats.ttest_ind(phys, rnd)
    print(f"\n  Physics ({phys.mean():.6f}) vs random avg ({rnd.mean():.6f}): t={t_stat:.2f}, p={p_val:.4f}")

    print(f"\nMask ablation complete! Time: {time.strftime('%H:%M')}")
    return results


if __name__ == "__main__":
    run_mask_ablation()
