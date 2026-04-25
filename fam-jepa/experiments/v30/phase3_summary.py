"""Pretty-print the v30 Phase 3 master table (markdown)."""
import json
import sys
from pathlib import Path

import numpy as np

V30 = Path('/home/sagemaker-user/IndustrialJEPA/fam-jepa/experiments/v30')
MT = V30 / 'results/master_table.json'

if not MT.exists():
    print(f'ERROR: {MT} missing')
    sys.exit(1)

mt = json.load(open(MT))
ds_data = mt.get('datasets', {})

# Optional: pull v29 + Phase 1 + Phase 2 baselines for comparison
v29_baselines = {
    'FD001': 0.742, 'FD002': 0.569, 'FD003': 0.819,
    'SMAP': 0.550, 'PSM': 0.559, 'MBA': 0.746,
    'GECCO': 0.859, 'BATADAL': 0.629, 'SKAB': 0.726,
    'ETTm1': 0.869, 'SMD': 0.616,
}
chr2_p1 = {}
p1 = V30 / 'results/phase1_decision.json'
if p1.exists():
    p1d = json.load(open(p1))
    for ds, v in p1d.get('chr2-probe_hauroc', {}).items():
        chr2_p1[ds] = (v.get('mean'), v.get('std'))

print(f"\n# v30 Phase 3 Master Table — {len(ds_data)} datasets, "
      f"dense K=150, 3 seeds (42/123/456)\n")
print("| Dataset | h-AUROC 100% (mean ± std) | h-AUROC 10% | v29 sparse-K=8 | Δ vs v29 | Chr2 (Phase 1) |")
print("|---------|---------------------------|-------------|----------------|----------|-----------------|")

for ds, row in ds_data.items():
    lf100 = row.get('lf100', {})
    lf10 = row.get('lf10', {})
    f_mean = lf100.get('mean_h_auroc')
    f_std = lf100.get('std_h_auroc')
    f_n = lf100.get('n_seeds', 0)
    t_mean = lf10.get('mean_h_auroc')
    t_std = lf10.get('std_h_auroc')

    f_str = f"{f_mean:.4f} ± {f_std:.4f}" if f_std else (f"{f_mean:.4f}" if f_mean else '—')
    t_str = f"{t_mean:.4f} ± {t_std:.4f}" if t_std else (f"{t_mean:.4f}" if t_mean else '—')
    v29_str = f"{v29_baselines.get(ds, '—')}" if v29_baselines.get(ds) is not None else '—'
    delta_v29 = (f"{f_mean - v29_baselines[ds]:+.4f}"
                 if f_mean and ds in v29_baselines else '—')
    chr_m, chr_s = chr2_p1.get(ds, (None, None))
    chr_str = (f"{chr_m:.4f} ± {chr_s:.4f}"
               if chr_m and chr_s else (f"{chr_m:.4f}" if chr_m else '—'))
    print(f"| {ds} | {f_str} ({f_n}s) | {t_str} | {v29_str} | {delta_v29} | {chr_str} |")

t_elapsed = mt.get('time_elapsed_s') or mt.get('time_total_s') or 0
print(f"\n(elapsed: {t_elapsed:.0f}s)")
