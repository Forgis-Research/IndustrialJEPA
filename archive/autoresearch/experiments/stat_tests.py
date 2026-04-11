#!/usr/bin/env python3
"""
Statistical significance tests across all 3 tiers.
Run AFTER all tier experiments complete.
Uses more seeds (10) for tighter confidence intervals on the key comparisons.
"""
import sys, time, copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy import stats

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── Import tier scripts ───
sys.path.insert(0, '/home/sagemaker-user/IndustrialJEPA/autoresearch/experiments')

SEEDS_10 = [42, 123, 456, 789, 1234, 5678, 9012, 3456, 7890, 2468]


def run_cmapss_10seed():
    """Run CI-Trans vs Role-Trans on C-MAPSS with 10 seeds for significance."""
    from tier2_cmapss import (load_cmapss, TrainDS, TestDS, CITrans, RoleTrans,
                               train_m, ev, finetune, N_CH, SEQ_LEN, GROUP_IDX)

    print("="*70)
    print("10-SEED C-MAPSS: CI-Trans vs Role-Trans")
    print("="*70)
    sys.stdout.flush()

    fd1 = load_cmapss("FD001")
    fd2 = load_cmapss("FD002")
    nm = fd1['ts'].mean(0); ns = fd1['ts'].std(0); ns[ns < 1e-8] = 1.0

    train_ds = TrainDS(fd1['tu'], fd1['ts'], fd1['tr'], nm, ns)
    nt = int(0.8*len(train_ds)); nv = len(train_ds) - nt
    tl1 = DataLoader(TestDS(fd1, nm, ns), 256)
    tl2 = DataLoader(TestDS(fd2, nm, ns), 256)

    results = {'ci': {'fd001': [], 'fd002': []}, 'role': {'fd001': [], 'fd002': []}}

    for seed in SEEDS_10:
        t0 = time.time()
        torch.manual_seed(seed); np.random.seed(seed)
        trn, val = torch.utils.data.random_split(
            train_ds, [nt, nv], generator=torch.Generator().manual_seed(seed))
        tl = DataLoader(trn, 256, shuffle=True)
        vl = DataLoader(val, 256)

        # CI-Trans
        ci = CITrans()
        ci, _ = train_m(ci, tl, vl)
        ci1 = ev(ci, tl1); ci2 = ev(ci, tl2)
        results['ci']['fd001'].append(ci1)
        results['ci']['fd002'].append(ci2)

        # Role-Trans
        torch.manual_seed(seed); np.random.seed(seed)
        rt = RoleTrans()
        rt, _ = train_m(rt, tl, vl)
        rt1 = ev(rt, tl1); rt2 = ev(rt, tl2)
        results['role']['fd001'].append(rt1)
        results['role']['fd002'].append(rt2)

        print(f"  [{seed}] CI: {ci1:.2f}/{ci2:.2f} | Role: {rt1:.2f}/{rt2:.2f} ({time.time()-t0:.0f}s)")
        sys.stdout.flush()

    # Statistical tests
    ci_fd002 = np.array(results['ci']['fd002'])
    rt_fd002 = np.array(results['role']['fd002'])

    print(f"\n--- Results ---")
    print(f"  CI-Trans FD002: {ci_fd002.mean():.2f} ± {ci_fd002.std():.2f}")
    print(f"  Role-Trans FD002: {rt_fd002.mean():.2f} ± {rt_fd002.std():.2f}")

    # Paired t-test (same seeds)
    t_stat, p_val = stats.ttest_rel(ci_fd002, rt_fd002)
    print(f"\n  Paired t-test: t={t_stat:.3f}, p={p_val:.4f}")

    # Wilcoxon signed-rank (non-parametric)
    w_stat, w_p = stats.wilcoxon(ci_fd002, rt_fd002)
    print(f"  Wilcoxon: W={w_stat:.1f}, p={w_p:.4f}")

    # Effect size (Cohen's d)
    diff = ci_fd002 - rt_fd002
    cohens_d = diff.mean() / diff.std()
    print(f"  Cohen's d: {cohens_d:.2f}")

    # Win rate
    wins = (rt_fd002 < ci_fd002).sum()
    print(f"  Role wins: {wins}/{len(SEEDS_10)} seeds")

    # Improvement
    pct = (1 - rt_fd002.mean() / ci_fd002.mean()) * 100
    print(f"  Improvement: {pct:.1f}%")

    return results


def run_pendulum_10seed():
    """Run CI-Trans vs Physics-Grouped on Pendulum with 10 seeds."""
    from tier1_pendulum import (load_pendulum_data, TSDataset, CITransformer as PendCITrans,
                                 PhysicsGrouped, train_model as pend_train, evaluate as pend_eval,
                                 LOOKBACK, HORIZON, N_CH as PEND_NCH, PHYSICS_GROUPS as PEND_GROUPS)

    print("\n" + "="*70)
    print("10-SEED PENDULUM: CI-Trans vs Physics-Grouped")
    print("="*70)
    sys.stdout.flush()

    source, target = load_pendulum_data()

    src_ds = TSDataset(source, max_samples=8000)
    mn, sd = src_ds.mean, src_ds.std
    tgt_ds = TSDataset(target, mean=mn, std=sd, max_samples=3000)
    src_l = DataLoader(src_ds, 256)
    tgt_l = DataLoader(tgt_ds, 256)

    results = {'ci': {'src': [], 'tgt': []}, 'phys': {'src': [], 'tgt': []}}

    for seed in SEEDS_10:
        t0 = time.time()
        torch.manual_seed(seed); np.random.seed(seed)
        n = len(src_ds); nt = int(0.8*n); nv = n - nt
        trn, val = torch.utils.data.random_split(
            src_ds, [nt, nv], generator=torch.Generator().manual_seed(seed))
        tl = DataLoader(trn, 128, shuffle=True)
        vl = DataLoader(val, 256)

        # CI
        ci = PendCITrans()
        ci, _ = pend_train(ci, tl, vl)
        cs = pend_eval(ci, src_l); ct = pend_eval(ci, tgt_l)
        results['ci']['src'].append(cs); results['ci']['tgt'].append(ct)

        # Physics
        torch.manual_seed(seed); np.random.seed(seed)
        pg = PhysicsGrouped()
        pg, _ = pend_train(pg, tl, vl)
        ps = pend_eval(pg, src_l); pt = pend_eval(pg, tgt_l)
        results['phys']['src'].append(ps); results['phys']['tgt'].append(pt)

        print(f"  [{seed}] CI tgt={ct:.6f} | Phys tgt={pt:.6f} ({time.time()-t0:.0f}s)")
        sys.stdout.flush()

    ci_tgt = np.array(results['ci']['tgt'])
    ph_tgt = np.array(results['phys']['tgt'])

    print(f"\n--- Results ---")
    print(f"  CI-Trans target: {ci_tgt.mean():.6f} ± {ci_tgt.std():.6f}")
    print(f"  Physics target: {ph_tgt.mean():.6f} ± {ph_tgt.std():.6f}")

    t_stat, p_val = stats.ttest_rel(ci_tgt, ph_tgt)
    print(f"  Paired t-test: t={t_stat:.3f}, p={p_val:.4f}")

    wins = (ph_tgt < ci_tgt).sum()
    print(f"  Physics wins: {wins}/{len(SEEDS_10)} seeds")

    pct = (1 - ph_tgt.mean() / ci_tgt.mean()) * 100
    print(f"  Improvement: {pct:.1f}%")

    return results


if __name__ == "__main__":
    print(f"Statistical significance tests — {time.strftime('%H:%M')}")
    print(f"Device: {DEVICE}\n")

    cmapss_results = run_cmapss_10seed()
    # Pendulum 10-seed may fail if tier1 imports don't work — that's ok
    try:
        pend_results = run_pendulum_10seed()
    except Exception as e:
        print(f"\nPendulum 10-seed failed: {e}")
        print("Run manually if needed.")

    print(f"\nAll tests complete! Time: {time.strftime('%H:%M')}")
