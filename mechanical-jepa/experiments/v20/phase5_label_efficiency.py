"""V20 Phase 5: Label-efficiency sweep at 50%/20%/10% (extends Phase 0b).

pred_ft vs e2e vs scratch at 5 intermediate label budgets on FD001.
5 seeds each. Uses Phase 0b's 100% and 5% as end points.
"""
import sys, json, time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa')
V11 = ROOT / 'experiments' / 'v11'
V20 = ROOT / 'experiments' / 'v20'
sys.path.insert(0, str(V11)); sys.path.insert(0, str(V20)); sys.path.insert(0, str(ROOT))

from phase0_pred_ft import run_single, aggregate, fmt, BUDGETS, SEEDS, MODES
from data_utils import load_cmapss_subset

# Override: just pred_ft and e2e (drop probe_h, frozen_multi, scratch) at new budgets
MODES_SWEEP = ['pred_ft', 'e2e']
BUDGETS_SWEEP = [0.5, 0.2, 0.1]
SEEDS_SWEEP = [0, 1, 2, 3, 4]


def main():
    out_path = V20 / 'phase5_label_efficiency.json'
    data = load_cmapss_subset('FD001')
    print("=" * 80)
    print(f"V20 Phase 5: label efficiency {BUDGETS_SWEEP}, modes {MODES_SWEEP}")
    print("=" * 80, flush=True)

    t0 = time.time()
    results = {}
    for mode in MODES_SWEEP:
        for b in BUDGETS_SWEEP:
            key = f"{mode}@{b}"
            results[key] = []
            for seed in SEEDS_SWEEP:
                t1 = time.time()
                r = run_single(data, mode, b, seed)
                dt = time.time() - t1
                results[key].append(r)
                print(f"  [{mode:14s} b={b*100:4.0f}% s={seed}] "
                      f"RMSE={r['test_rmse']:.2f} "
                      f"F1w={r['per_window_f1_mean']:.3f} "
                      f"AUROCw={r['per_window_auroc_mean']:.3f} ({dt:.0f}s)",
                      flush=True)
                out = {'config': 'v20_phase5_label_efficiency',
                       'seeds': SEEDS_SWEEP, 'budgets': BUDGETS_SWEEP,
                       'modes': MODES_SWEEP,
                       'runtime_min': (time.time() - t0) / 60,
                       'results': {k: aggregate(v) for k, v in results.items()}}
                with open(out_path, 'w') as f:
                    json.dump(out, f, indent=2, default=float)
    print(f"\nRuntime: {(time.time() - t0)/60:.1f} min")


if __name__ == '__main__':
    main()
