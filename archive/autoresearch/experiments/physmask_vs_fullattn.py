#!/usr/bin/env python3
"""
PhysMask vs Full-Attn: 10-seed comparison on Pendulum and Weather.

PhysMask = masked attention (no pooling, physics-informed mask)
Full-Attn = unconstrained all-to-all attention

If PhysMask beats Full-Attn, physics grouping provides value beyond just
"any grouping" — the attention mask itself encodes useful structure.
"""
import sys, time
import numpy as np
import torch
from torch.utils.data import DataLoader
from scipy import stats

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sys.path.insert(0, '/home/sagemaker-user/IndustrialJEPA/autoresearch/experiments')

SEEDS_10 = [42, 123, 456, 789, 1234, 5678, 9012, 3456, 7890, 2468]


def pendulum_comparison():
    from tier1_pendulum import (load_pendulum_data, TSDataset, CITransformer,
                                 FullAttention, PhysicsGrouped, train_model, evaluate)

    print("="*70)
    print("PENDULUM: CI vs Full-Attn vs PhysMask (10 seeds)")
    print("="*70)

    source, target = load_pendulum_data()
    src_ds = TSDataset(source, max_samples=8000)
    mn, sd = src_ds.mean, src_ds.std
    tgt_ds = TSDataset(target, mean=mn, std=sd, max_samples=3000)
    src_l = DataLoader(src_ds, 256)
    tgt_l = DataLoader(tgt_ds, 256)
    n = len(src_ds); nt = int(0.8*n); nv = n - nt

    results = {m: {'src': [], 'tgt': []} for m in ['ci', 'full', 'phys']}

    for seed in SEEDS_10:
        t0 = time.time()
        torch.manual_seed(seed); np.random.seed(seed)
        trn, val = torch.utils.data.random_split(
            src_ds, [nt, nv], generator=torch.Generator().manual_seed(seed))
        tl = DataLoader(trn, 128, shuffle=True)
        vl = DataLoader(val, 256)

        for mname, ModelClass in [('ci', CITransformer), ('full', FullAttention), ('phys', PhysicsGrouped)]:
            torch.manual_seed(seed); np.random.seed(seed)
            m = ModelClass()
            m, _ = train_model(m, tl, vl)
            sm = evaluate(m, src_l); tm = evaluate(m, tgt_l)
            results[mname]['src'].append(sm)
            results[mname]['tgt'].append(tm)

        print(f"  [{seed}] CI={results['ci']['tgt'][-1]:.6f} Full={results['full']['tgt'][-1]:.6f} "
              f"Phys={results['phys']['tgt'][-1]:.6f} ({time.time()-t0:.0f}s)")
        sys.stdout.flush()

    # Summary
    for mname in ['ci', 'full', 'phys']:
        t = np.array(results[mname]['tgt'])
        print(f"\n{mname}: {t.mean():.6f} ± {t.std():.6f}")

    ci_t = np.array(results['ci']['tgt'])
    full_t = np.array(results['full']['tgt'])
    phys_t = np.array(results['phys']['tgt'])

    print(f"\n--- Statistical Tests ---")
    t1, p1 = stats.ttest_rel(full_t, phys_t)
    print(f"  Full vs Phys: t={t1:.3f}, p={p1:.4f} (better: {'phys' if phys_t.mean() < full_t.mean() else 'full'})")
    print(f"  PhysMask wins: {(phys_t < full_t).sum()}/10")
    print(f"  Improvement: {(1-phys_t.mean()/full_t.mean())*100:.1f}%")

    t2, p2 = stats.ttest_rel(ci_t, phys_t)
    print(f"  CI vs Phys: t={t2:.3f}, p={p2:.4f}")
    print(f"  Improvement: {(1-phys_t.mean()/ci_t.mean())*100:.1f}%")

    return results


def weather_comparison():
    """Same comparison on weather (H=96)."""
    import csv
    from tier3_weather import (CITrans, FullAttn, PhysicsGrouped, WeatherDS,
                                train_m, evaluate, LOOKBACK, HORIZON, N_CH, PHYSICS_GROUPS)

    print("\n" + "="*70)
    print("WEATHER: CI vs Full-Attn vs PhysMask (10 seeds)")
    print("="*70)

    # Load data
    DATA_PATH = "/home/sagemaker-user/IndustrialJEPA/data/weather/jena_climate_2009_2016.csv"
    dates, data = [], []
    with open(DATA_PATH, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for i, row in enumerate(reader):
            if i % 6 != 0: continue
            try:
                dates.append(row[0])
                data.append([float(v) for v in row[1:]])
            except (ValueError, IndexError):
                continue
    data = np.array(data)
    for col in [11, 12]:
        bad = data[:, col] < -999
        if bad.any():
            data[bad, col] = data[~bad, col].mean()

    train_idx, test_idx = [], []
    for i, d in enumerate(dates):
        year = int(d.split('.')[2].split()[0])
        if year <= 2015: train_idx.append(i)
        else: test_idx.append(i)

    train_data = data[train_idx]; test_data = data[test_idx]
    mn, sd = train_data.mean(0), np.maximum(train_data.std(0), 1e-8)
    train_ds = WeatherDS(train_data, mean=mn, std=sd, stride=6, max_samples=12000)
    test_ds = WeatherDS(test_data, mean=mn, std=sd, stride=3, max_samples=4000)
    test_l = DataLoader(test_ds, 256)

    n = len(train_ds); nt = int(0.8*n); nv = n - nt

    results = {m: [] for m in ['ci', 'full', 'phys']}

    for seed in SEEDS_10:
        t0 = time.time()
        torch.manual_seed(seed); np.random.seed(seed)
        trn, val = torch.utils.data.random_split(
            train_ds, [nt, nv], generator=torch.Generator().manual_seed(seed))
        tl = DataLoader(trn, 128, shuffle=True)
        vl = DataLoader(val, 256)

        for mname, fn in [('ci', CITrans), ('full', FullAttn), ('phys', PhysicsGrouped)]:
            torch.manual_seed(seed); np.random.seed(seed)
            m = fn()
            m, _ = train_m(m, tl, vl)
            te = evaluate(m, test_l)
            results[mname].append(te)

        print(f"  [{seed}] CI={results['ci'][-1]:.6f} Full={results['full'][-1]:.6f} "
              f"Phys={results['phys'][-1]:.6f} ({time.time()-t0:.0f}s)")
        sys.stdout.flush()

    for mname in ['ci', 'full', 'phys']:
        t = np.array(results[mname])
        print(f"\n{mname}: {t.mean():.6f} ± {t.std():.6f}")

    ci_t = np.array(results['ci'])
    full_t = np.array(results['full'])
    phys_t = np.array(results['phys'])

    print(f"\n--- Statistical Tests ---")
    t1, p1 = stats.ttest_rel(full_t, phys_t)
    print(f"  Full vs Phys: t={t1:.3f}, p={p1:.4f} (better: {'phys' if phys_t.mean() < full_t.mean() else 'full'})")
    print(f"  PhysMask wins: {(phys_t < full_t).sum()}/10")

    t2, p2 = stats.ttest_rel(ci_t, phys_t)
    print(f"  CI vs Phys: t={t2:.3f}, p={p2:.4f}")
    print(f"  Improvement over CI: {(1-phys_t.mean()/ci_t.mean())*100:.1f}%")

    return results


if __name__ == "__main__":
    print(f"PhysMask vs Full-Attn — {time.strftime('%H:%M')}")
    print(f"Device: {DEVICE}\n")
    pend = pendulum_comparison()
    try:
        weather = weather_comparison()
    except Exception as e:
        print(f"\nWeather comparison failed: {e}")
    print(f"\nAll done! Time: {time.strftime('%H:%M')}")
