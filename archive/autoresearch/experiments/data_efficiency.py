#!/usr/bin/env python3
"""
Data efficiency experiment: Does grouped architecture help MORE with less training data?

Hypothesis: With limited training data, full attention overfits while grouped architecture
provides inductive bias that prevents overfitting. This would give grouped models
a practical advantage in low-data regimes common in industry.

Tests CI-Trans, Full-Attn, Role-Trans at 10%, 25%, 50%, 100% training data on C-MAPSS.
"""
import sys, time, copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sys.path.insert(0, '/home/sagemaker-user/IndustrialJEPA/autoresearch/experiments')
from tier2_cmapss import (load_cmapss, TrainDS, TestDS, CITrans, FullAttnTrans, RoleTrans,
                           train_m, ev, N_CH, SEQ_LEN, GROUP_IDX)

SEEDS = [42, 123, 456]
DATA_FRACTIONS = [0.05, 0.10, 0.25, 0.50, 1.00]


def run_data_efficiency():
    print("="*70)
    print("DATA EFFICIENCY: Grouped vs CI at varying training sizes")
    print(f"Time: {time.strftime('%H:%M')} | Device: {DEVICE}")
    print("="*70)
    sys.stdout.flush()

    fd1 = load_cmapss("FD001")
    fd2 = load_cmapss("FD002")
    nm = fd1['ts'].mean(0); ns = fd1['ts'].std(0); ns[ns < 1e-8] = 1.0

    full_train_ds = TrainDS(fd1['tu'], fd1['ts'], fd1['tr'], nm, ns)
    tl1 = DataLoader(TestDS(fd1, nm, ns), 256)
    tl2 = DataLoader(TestDS(fd2, nm, ns), 256)

    n_full = len(full_train_ds)
    print(f"  Full training set: {n_full} samples")
    sys.stdout.flush()

    all_results = {}

    for frac in DATA_FRACTIONS:
        n_train = max(100, int(n_full * frac))  # at least 100 samples
        n_val = max(50, int(n_train * 0.2))
        n_train_actual = n_train - n_val

        print(f"\n{'='*50}")
        print(f"DATA FRACTION: {frac*100:.0f}% ({n_train} samples, train={n_train_actual}, val={n_val})")
        print(f"{'='*50}")
        sys.stdout.flush()

        frac_results = {}
        models = {
            "ci_trans": lambda: CITrans(),
            "full_attn": lambda: FullAttnTrans(),
            "role_trans": lambda: RoleTrans(),
        }

        for mname, fn in models.items():
            print(f"\n  --- {mname} ---")
            r1a, r2a = [], []
            for seed in SEEDS:
                t0 = time.time()
                torch.manual_seed(seed); np.random.seed(seed)

                # Subsample training data
                indices = torch.randperm(n_full, generator=torch.Generator().manual_seed(seed))[:n_train]
                subset = Subset(full_train_ds, indices.tolist())

                trn, val = torch.utils.data.random_split(
                    subset, [n_train_actual, n_val],
                    generator=torch.Generator().manual_seed(seed))
                tl = DataLoader(trn, min(256, n_train_actual), shuffle=True)
                vl = DataLoader(val, 256)

                m = fn()
                m, _ = train_m(m, tl, vl)
                e1 = ev(m, tl1); r1a.append(e1)
                e2 = ev(m, tl2); r2a.append(e2)
                print(f"    [{seed}] FD001={e1:.2f} FD002={e2:.2f} ({time.time()-t0:.0f}s)")
                sys.stdout.flush()

            r1 = np.array(r1a); r2 = np.array(r2a)
            frac_results[mname] = {'fd001': r1, 'fd002': r2}
            print(f"    AVG: FD001={r1.mean():.2f}±{r1.std():.2f} FD002={r2.mean():.2f}±{r2.std():.2f}")

        all_results[frac] = frac_results

    # Summary
    print("\n" + "="*70)
    print("DATA EFFICIENCY SUMMARY")
    print("="*70)

    # FD001 (in-domain)
    print("\nFD001 (in-domain) RMSE:")
    print(f"{'Fraction':<10} {'CI-Trans':>12} {'Full-Attn':>12} {'Role-Trans':>12}")
    print("-"*50)
    for frac in DATA_FRACTIONS:
        r = all_results[frac]
        ci = r['ci_trans']['fd001'].mean()
        fa = r['full_attn']['fd001'].mean()
        rt = r['role_trans']['fd001'].mean()
        print(f"{frac*100:>5.0f}%     {ci:>12.2f} {fa:>12.2f} {rt:>12.2f}")

    # FD002 (transfer)
    print("\nFD002 (zero-shot transfer) RMSE:")
    print(f"{'Fraction':<10} {'CI-Trans':>12} {'Full-Attn':>12} {'Role-Trans':>12} {'RT vs CI':>10} {'RT vs FA':>10}")
    print("-"*70)
    for frac in DATA_FRACTIONS:
        r = all_results[frac]
        ci = r['ci_trans']['fd002'].mean()
        fa = r['full_attn']['fd002'].mean()
        rt = r['role_trans']['fd002'].mean()
        rt_ci = (1 - rt/ci) * 100
        rt_fa = (1 - rt/fa) * 100
        print(f"{frac*100:>5.0f}%     {ci:>12.2f} {fa:>12.2f} {rt:>12.2f} {rt_ci:>9.1f}% {rt_fa:>9.1f}%")

    print(f"\nData efficiency complete! Time: {time.strftime('%H:%M')}")
    return all_results


if __name__ == "__main__":
    run_data_efficiency()
