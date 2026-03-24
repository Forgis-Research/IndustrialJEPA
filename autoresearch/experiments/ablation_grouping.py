#!/usr/bin/env python3
"""
Ablation: Does PHYSICS grouping matter, or does ANY grouping help?

Tests:
1. Physics grouping (fan, HPC, combustor, turbine, nozzle) — our method
2. Random grouping (same sizes, random assignment)
3. Size-matched random (3 random shuffles)
4. Wrong grouping (deliberately bad: mix sensors from different components)
5. Single group (all sensors in one group — equivalent to full attention)
6. Per-sensor (each sensor its own group — equivalent to CI)

If physics >> random, the physics knowledge matters.
If physics ≈ random >> CI, any grouping helps but physics isn't special.
"""
import sys, time, copy, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = Path("/home/sagemaker-user/IndustrialJEPA/data/cmapss")
MAX_RUL = 125; SEQ_LEN = 30
COLS = ["unit","cycle"]+[f"setting{i}" for i in range(1,4)]+[f"s{i}" for i in range(1,22)]
SC = ["s2","s3","s4","s7","s8","s9","s11","s12","s13","s14","s15","s17","s20","s21"]
N_CH = len(SC)

# Ground truth physics grouping
PHYSICS_GROUPS = {
    "fan": [SC.index(s) for s in ["s2","s8","s12","s21"]],
    "hpc": [SC.index(s) for s in ["s3","s7","s11","s20"]],
    "combustor": [SC.index(s) for s in ["s9","s14"]],
    "turbine": [SC.index(s) for s in ["s4","s13"]],
    "nozzle": [SC.index(s) for s in ["s15","s17"]],
}

SEEDS = [42, 123, 456]


def make_random_groups(n_channels=N_CH, group_sizes=[4, 4, 2, 2, 2], rng_seed=0):
    """Random grouping with same sizes as physics groups."""
    rng = random.Random(rng_seed)
    indices = list(range(n_channels))
    rng.shuffle(indices)
    groups = {}
    pos = 0
    for i, sz in enumerate(group_sizes):
        groups[f"rnd_{i}"] = indices[pos:pos+sz]
        pos += sz
    return groups


def make_wrong_groups(n_channels=N_CH):
    """Deliberately wrong: pair sensors from DIFFERENT components."""
    # Take one from each component, mix them
    return {
        "wrong_0": [0, 5, 8, 12],    # s2(fan), s9(comb), s13(turb), s20(hpc)
        "wrong_1": [1, 4, 9, 13],    # s3(hpc), s8(fan), s14(comb), s21(fan)
        "wrong_2": [2, 7, 10, 11],   # s4(turb), s12(fan), s15(noz), s17(noz)
        "wrong_3": [3, 6],           # s7(hpc), s11(hpc) — accidentally correct pair
        "wrong_4": [5, 8],           # leftover - but let me fix the indexing
    }


def make_wrong_groups_v2():
    """Maximally wrong: each group has sensors from ALL different components."""
    # fan=[0,4,7,13], hpc=[1,3,6,12], comb=[5,9], turb=[2,8], noz=[10,11]
    return {
        "w0": [0, 1, 2, 5],     # fan, hpc, turb, comb — all different
        "w1": [4, 3, 8, 9],     # fan, hpc, turb, comb — all different
        "w2": [7, 6, 10, 11],   # fan, hpc, noz, noz
        "w3": [13, 12],         # fan, hpc
    }


# ─── Data Loading (from tier2) ───

def load_cmapss(subset):
    def read_txt(path):
        rows = []
        with open(path) as f:
            for line in f:
                vals = line.strip().split()
                if len(vals) >= len(COLS):
                    rows.append([float(v) for v in vals[:len(COLS)]])
        return np.array(rows)

    train_raw = read_txt(DATA_DIR / f"train_{subset}.txt")
    test_raw = read_txt(DATA_DIR / f"test_{subset}.txt")
    with open(DATA_DIR / f"RUL_{subset}.txt") as f:
        rul_vals = [int(line.strip().split()[0]) for line in f if line.strip()]

    sensor_idx = [COLS.index(s) for s in SC]
    def build(raw):
        return raw[:, 0].astype(int), raw[:, 1].astype(int), raw[:, sensor_idx]

    tu, tc, ts = build(train_raw)
    eu, ec, es = build(test_raw)
    trul = np.zeros(len(train_raw))
    for u in np.unique(tu):
        m = tu == u; trul[m] = np.minimum(tc[m].max() - tc[m], MAX_RUL)
    return {'tu': tu, 'tc': tc, 'ts': ts, 'tr': trul,
            'eu': eu, 'ec': ec, 'es': es, 'er': rul_vals}


class TrainDS(Dataset):
    def __init__(self, units, sensors, rul, nm, ns, sl=SEQ_LEN):
        self.samples = []
        sensors = (sensors - nm) / ns
        for u in np.unique(units):
            m = units == u; s = sensors[m]; r = rul[m]
            for i in range(len(s) - sl + 1):
                self.samples.append((s[i:i+sl], min(r[i+sl-1], MAX_RUL)))
    def __len__(self): return len(self.samples)
    def __getitem__(self, i):
        x, y = self.samples[i]
        return torch.FloatTensor(x), torch.FloatTensor([y])

class TestDS(Dataset):
    def __init__(self, data, nm, ns, sl=SEQ_LEN):
        self.samples = []
        sensors = (data['es'] - nm) / ns
        units = data['eu']; rm = data['er']
        for idx, u in enumerate(np.unique(units)):
            m = units == u; s = sensors[m]
            if len(s) < sl: s = np.vstack([np.tile(s[0], (sl-len(s), 1)), s])
            self.samples.append((s[-sl:], min(rm[idx], MAX_RUL)))
    def __len__(self): return len(self.samples)
    def __getitem__(self, i):
        x, y = self.samples[i]
        return torch.FloatTensor(x), torch.FloatTensor([y])


# ─── Model ───

class GroupedTrans(nn.Module):
    """Role-Transformer with configurable groups."""
    def __init__(self, groups, d=32, nh=4, nl=2, do=0.1):
        super().__init__()
        self.d, self.groups = d, groups
        self.nc = len(groups)
        self.proj = nn.Linear(1, d)
        self.pos = nn.Parameter(torch.randn(1, SEQ_LEN, d)*0.02)
        el = nn.TransformerEncoderLayer(d, nh, d*4, do, batch_first=True)
        self.within = nn.TransformerEncoder(el, nl)
        self.pool = nn.Linear(d, d)
        self.cemb = nn.Parameter(torch.randn(1, self.nc, d)*0.02)
        cl = nn.TransformerEncoderLayer(d, nh, d*4, do, batch_first=True)
        self.cross = nn.TransformerEncoder(cl, 1)
        self.out = nn.Linear(d*self.nc, 1)

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
        return self.out(s.reshape(B, -1))


# ─── Training ───

def train_m(model, tl, vl, epochs=60, lr=1e-3, patience=12):
    model = model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    bv, bs, ni = 1e9, None, 0
    for ep in range(epochs):
        model.train()
        for x, y in tl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            loss = F.mse_loss(model(x), y)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        sch.step()
        v = ev(model, vl)
        if v < bv:
            bv = v; bs = {k: v.cpu().clone() for k, v in model.state_dict().items()}; ni = 0
        else: ni += 1
        if ni >= patience: break
    model.load_state_dict(bs); model.to(DEVICE)
    return model, bv

def ev(model, loader):
    model.eval(); ps, ts = [], []
    with torch.no_grad():
        for x, y in loader:
            ps.append(model(x.to(DEVICE)).cpu()); ts.append(y)
    return torch.sqrt(F.mse_loss(torch.cat(ps), torch.cat(ts))).item()

def finetune(model, loader, epochs=10, lr=5e-4):
    model = model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        model.train()
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            loss = F.mse_loss(model(x), y)
            opt.zero_grad(); loss.backward(); opt.step()
    return model


def run_ablation():
    print("="*70)
    print("ABLATION: Does Physics Grouping Specifically Matter?")
    print(f"Time: {time.strftime('%H:%M')} | Device: {DEVICE}")
    print("="*70)
    sys.stdout.flush()

    fd1 = load_cmapss("FD001")
    fd2 = load_cmapss("FD002")

    nm = fd1['ts'].mean(0); ns = fd1['ts'].std(0); ns[ns < 1e-8] = 1.0

    train_ds = TrainDS(fd1['tu'], fd1['ts'], fd1['tr'], nm, ns)
    nt = int(0.8*len(train_ds)); nv = len(train_ds) - nt

    tl1 = DataLoader(TestDS(fd1, nm, ns), 256)
    tl2 = DataLoader(TestDS(fd2, nm, ns), 256)

    # FD002 adapt
    u2 = np.unique(fd2['tu']); na = max(1, len(u2)//10)
    am = np.isin(fd2['tu'], u2[:na])
    adapt_ds = TrainDS(fd2['tu'][am], fd2['ts'][am], fd2['tr'][am], nm, ns)

    # Define grouping conditions
    conditions = {
        "physics": PHYSICS_GROUPS,
        "random_0": make_random_groups(rng_seed=0),
        "random_1": make_random_groups(rng_seed=1),
        "random_2": make_random_groups(rng_seed=2),
        "wrong": make_wrong_groups_v2(),
    }

    print("\nGrouping assignments:")
    for name, groups in conditions.items():
        print(f"  {name}:")
        for gn, idx in groups.items():
            sensors = [SC[i] for i in idx]
            print(f"    {gn}: {sensors}")
    print()
    sys.stdout.flush()

    results = {}
    for cond_name, groups in conditions.items():
        print(f"\n{'='*50}")
        print(f"CONDITION: {cond_name}")
        print(f"{'='*50}")

        r1a, r2a, r2aa = [], [], []
        for seed in SEEDS:
            t0 = time.time()
            torch.manual_seed(seed); np.random.seed(seed)
            trn, val = torch.utils.data.random_split(
                train_ds, [nt, nv], generator=torch.Generator().manual_seed(seed))
            tl = DataLoader(trn, 256, shuffle=True)
            vl = DataLoader(val, 256)

            m = GroupedTrans(groups)
            m, _ = train_m(m, tl, vl)

            e1 = ev(m, tl1); r1a.append(e1)
            e2 = ev(m, tl2); r2a.append(e2)

            ma = copy.deepcopy(m)
            al = DataLoader(adapt_ds, 128, shuffle=True)
            ma = finetune(ma, al)
            e2a = ev(ma, tl2); r2aa.append(e2a)

            print(f"  [{seed}] FD001={e1:.2f} FD002={e2:.2f} FD002-adapt={e2a:.2f} ({time.time()-t0:.0f}s)")
            sys.stdout.flush()

        r1 = np.array(r1a); r2 = np.array(r2a); r2a_arr = np.array(r2aa)
        ratio = r2.mean() / r1.mean()
        ratio_a = r2a_arr.mean() / r1.mean()
        results[cond_name] = {
            'fd001': r1, 'fd002': r2, 'fd002_adapt': r2a_arr,
            'ratio': ratio, 'ratio_adapt': ratio_a
        }
        print(f"  AVG: FD001={r1.mean():.2f}±{r1.std():.2f} FD002={r2.mean():.2f}±{r2.std():.2f} "
              f"FD002-adapt={r2a_arr.mean():.2f}±{r2a_arr.std():.2f}")
        print(f"  Ratio: zero={ratio:.2f} adapted={ratio_a:.2f}")
        sys.stdout.flush()

    # Summary
    print("\n" + "="*70)
    print("ABLATION SUMMARY: Grouping Assignment Matters?")
    print("="*70)
    print(f"{'Condition':<15} {'FD001':>10} {'FD002':>10} {'FD002-10%':>10} {'R(zero)':>8} {'R(10%)':>8}")
    print("-"*65)
    for name, r in results.items():
        print(f"{name:<15} {r['fd001'].mean():>10.2f} {r['fd002'].mean():>10.2f} "
              f"{r['fd002_adapt'].mean():>10.2f} {r['ratio']:>8.2f} {r['ratio_adapt']:>8.2f}")

    # Statistical test: physics vs random
    from scipy import stats
    print("\n--- Statistical Tests ---")
    phys = results['physics']
    for rname in ['random_0', 'random_1', 'random_2', 'wrong']:
        if rname in results:
            r = results[rname]
            # Compare FD002 transfer scores
            t, p = stats.ttest_ind(phys['fd002'], r['fd002'])
            print(f"  Physics vs {rname} (FD002): t={t:.2f}, p={p:.3f} "
                  f"({'*' if p < 0.05 else 'ns'})")

    # Average of random conditions
    rnd_fd002 = np.concatenate([results[f'random_{i}']['fd002'] for i in range(3)])
    t, p = stats.ttest_ind(phys['fd002'], rnd_fd002)
    print(f"\n  Physics vs ALL random (FD002): t={t:.2f}, p={p:.3f} "
          f"({'*' if p < 0.05 else 'ns'})")
    print(f"  Physics FD002 avg: {phys['fd002'].mean():.2f}")
    print(f"  Random FD002 avg: {rnd_fd002.mean():.2f}")

    wrong_fd002 = results['wrong']['fd002']
    print(f"  Wrong FD002 avg: {wrong_fd002.mean():.2f}")

    if phys['fd002'].mean() < rnd_fd002.mean():
        pct = (1 - phys['fd002'].mean() / rnd_fd002.mean()) * 100
        print(f"\n  VERDICT: Physics grouping is {pct:.1f}% better than random on transfer")
    else:
        print(f"\n  VERDICT: Random grouping matches or beats physics — grouping structure doesn't matter")

    print(f"\nAblation complete! Time: {time.strftime('%H:%M')}")
    return results


if __name__ == "__main__":
    run_ablation()
