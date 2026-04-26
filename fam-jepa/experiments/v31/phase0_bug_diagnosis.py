"""V31 Phase 0: Diagnose and verify the label_fraction bug fix.

Prints sample counts at 100% vs 10% for all 11 datasets.
Self-check: train_windows at lf10 must be < train_windows at lf100
for the fix to be valid.
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, '/home/sagemaker-user/IndustrialJEPA/fam-jepa')
sys.path.insert(0, '/home/sagemaker-user/IndustrialJEPA/fam-jepa/experiments/v24')
sys.path.insert(0, '/home/sagemaker-user/IndustrialJEPA/fam-jepa/experiments/archive/v24')
sys.path.insert(0, '/home/sagemaker-user/IndustrialJEPA/fam-jepa/experiments/archive/v11')
sys.path.insert(0, '/home/sagemaker-user/IndustrialJEPA/fam-jepa/experiments/v27')
sys.path.insert(0, '/home/sagemaker-user/IndustrialJEPA/fam-jepa/experiments/v28')
sys.path.insert(0, '/home/sagemaker-user/IndustrialJEPA/fam-jepa/experiments/v29')
sys.path.insert(0, '/home/sagemaker-user/IndustrialJEPA/fam-jepa/experiments/v30')
sys.path.insert(0, '/home/sagemaker-user/IndustrialJEPA/fam-jepa/experiments/v31')

from _runner_v29 import LOADERS, NORM_POLICY
from _runner import _build_event_concat, _global_zscore
from _runner_v31 import _apply_label_fraction

DATASETS_11 = [
    'FD001', 'FD002', 'FD003',
    'SMAP', 'PSM', 'MBA',
    'GECCO', 'BATADAL',
    'SKAB', 'ETTm1', 'SMD',
]

# For label fraction test, only datasets where 10% makes sense
# (originally only FD001, FD003, MBA, BATADAL - but with fix we can extend to ALL)
DENSE_HORIZONS = list(range(1, 151))
MAX_FT_FUTURE = 150

results = []
print("\n=== V31 Phase 0: Label Fraction Bug Diagnosis ===\n")
print(f"{'Dataset':10s} {'n_entities_orig':>15s} {'n_entities_lf10':>15s} "
      f"{'windows_lf100':>14s} {'windows_lf10':>13s} {'ratio':>8s} {'fix_ok':>8s}")
print("-" * 90)

for ds in DATASETS_11:
    try:
        bundle = LOADERS[ds]()
        norm = NORM_POLICY[ds]
        if norm == 'none':
            bundle = _global_zscore(bundle)

        ft_train = bundle['ft_train']

        # Count at 100%
        train_ft_100 = _build_event_concat(ft_train, stride=4, max_context=512,
                                            max_future=MAX_FT_FUTURE)
        n_windows_100 = len(train_ft_100)

        # Count at 10%
        ft_train_10 = _apply_label_fraction(ft_train, 0.1, 42, ds)
        train_ft_10 = _build_event_concat(ft_train_10, stride=4, max_context=512,
                                           max_future=MAX_FT_FUTURE)
        n_windows_10 = len(train_ft_10)

        n_orig = len(ft_train) if not isinstance(ft_train, dict) else len(ft_train)
        n_lf10 = len(ft_train_10) if not isinstance(ft_train_10, dict) else len(ft_train_10)

        ratio = n_windows_10 / max(n_windows_100, 1)
        fix_ok = (n_windows_10 < n_windows_100)

        print(f"{ds:10s} {n_orig:>15d} {n_lf10:>15d} "
              f"{n_windows_100:>14d} {n_windows_10:>13d} {ratio:>8.3f} "
              f"{'OK' if fix_ok else 'BUG!':>8s}")

        results.append({
            'dataset': ds, 'n_entities_orig': n_orig, 'n_entities_lf10': n_lf10,
            'windows_lf100': n_windows_100, 'windows_lf10': n_windows_10,
            'ratio': ratio, 'fix_ok': fix_ok,
        })
    except Exception as e:
        print(f"{ds:10s}  ERROR: {e}")
        results.append({'dataset': ds, 'error': str(e)})

print("\n=== Summary ===")
bugs = [r for r in results if 'fix_ok' in r and not r['fix_ok']]
oks = [r for r in results if r.get('fix_ok', False)]
print(f"  Datasets with fix OK: {len(oks)}/{len(results)}")
if bugs:
    print(f"  BUGS REMAINING: {[r['dataset'] for r in bugs]}")
else:
    print("  No remaining bugs!")

out = Path('/home/sagemaker-user/IndustrialJEPA/fam-jepa/experiments/v31/results/phase0_bug_diagnosis.json')
out.parent.mkdir(exist_ok=True)
with open(out, 'w') as f:
    json.dump({'diagnosis': results, 'bugs': [r['dataset'] for r in bugs]}, f, indent=2)
print(f"\nWrote {out}")
