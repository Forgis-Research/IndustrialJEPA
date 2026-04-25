"""V28 follow-up — fill PSM and SMD coverage in the master table.

Phase 3 dense-FT was killed before reaching PSM/SMD/MBA/GECCO/BATADAL.
PSM and SMD currently have only v26 'revin' baseline numbers in the
v28 master table. This script runs the v28 baseline (no extra options,
norm_mode='revin') on PSM + SMD × 3 seeds so they have v28 ckpts.

The v28 ckpt is identical to v26 in terms of architecture and training
recipe (same FAM, same pretrain, same pred-FT) — the value-add is
producing an idiomatic v28 row + dense surface for the master table.
"""

import argparse
from runner_v28 import run_and_persist, RES_DIR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', nargs='+', default=['PSM', 'SMD'])
    ap.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 456])
    args = ap.parse_args()

    for ds in args.datasets:
        out = RES_DIR / f'phase9_{ds}_baseline.json'
        run_and_persist(
            dataset=ds, norm_mode='revin', seeds=args.seeds, out_json=out,
        )


if __name__ == '__main__':
    main()
