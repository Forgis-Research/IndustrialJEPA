"""V28 Phase 2 Try A* — lag features combined with norm_mode='none'.

Phase 2A under RevIN failed on FD001 (mean h-AUROC = 0.501, below base).
Hypothesis for the failure: RevIN normalises every lag-channel to zero
mean independently, so the cross-context drift signal that the lag
features were meant to expose gets washed out a second time.

This variant pairs lag features with norm_mode='none' (the v27 winner).
Now the lag channels carry the raw drift, and the global z-score keeps
them on a comparable scale across contexts.
"""

import argparse
from runner_v28 import run_and_persist, RES_DIR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', nargs='+', default=['FD001'])
    ap.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 456])
    ap.add_argument('--lags', type=int, nargs='+', default=[10, 50, 100])
    args = ap.parse_args()

    for ds in args.datasets:
        out = RES_DIR / f'phase2d_{ds}_lag_none.json'
        run_and_persist(
            dataset=ds, norm_mode='none', seeds=args.seeds, out_json=out,
            lag_features=args.lags,
        )


if __name__ == '__main__':
    main()
