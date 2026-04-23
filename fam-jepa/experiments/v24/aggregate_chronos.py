"""Aggregate Chronos-2 baseline results across seeds into a single JSON."""
import json
from pathlib import Path

import numpy as np

V24 = Path('/home/sagemaker-user/IndustrialJEPA/fam-jepa/experiments/v24')
RES = V24 / 'results'

DATASETS = ['FD001', 'FD002', 'FD003', 'SMAP', 'MSL', 'PSM', 'SMD', 'MBA', 'sepsis']
SEEDS = [42, 123, 456]

out = {}
for ds in DATASETS:
    per_seed = []
    for seed in SEEDS:
        p = RES / f'baseline_chronos2_{ds}_s{seed}.json'
        if p.exists():
            per_seed.append(json.loads(p.read_text()))
        else:
            # default path (no seed suffix if seed=42 run without out specified)
            p0 = RES / f'baseline_chronos2_{ds}.json'
            if p0.exists() and seed == 42:
                per_seed.append(json.loads(p0.read_text()))
    if not per_seed:
        continue
    auprcs = [r['primary']['auprc'] for r in per_seed]
    aurocs = [r['primary']['auroc'] for r in per_seed]
    f1s = [r['primary']['f1_best'] for r in per_seed]
    out[ds] = {
        'n_seeds': len(per_seed),
        'auprc_mean': float(np.mean(auprcs)),
        'auprc_std':  float(np.std(auprcs)),
        'auroc_mean': float(np.mean(aurocs)),
        'auroc_std':  float(np.std(aurocs)),
        'f1_best_mean': float(np.mean(f1s)),
        'f1_best_std':  float(np.std(f1s)),
        'per_seed': per_seed,
    }
    print(f"  {ds}  AUPRC {np.mean(auprcs):.4f} +/- {np.std(auprcs):.4f}  "
          f"AUROC {np.mean(aurocs):.4f} +/- {np.std(aurocs):.4f}",
          flush=True)

(RES / 'baseline_chronos2_agg.json').write_text(json.dumps(out, indent=2))
print(f"\nwrote {RES / 'baseline_chronos2_agg.json'}")
