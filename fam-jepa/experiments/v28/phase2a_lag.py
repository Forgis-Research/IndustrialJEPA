"""V28 Phase 2 Try A — lag-feature augmentation under RevIN.

Hypothesis: per Lag-Llama (Rasul+ 2024), encoding x[t-L] as additional
channels at position t makes the within-token vector contain the drift
gradient. RevIN normalizes per-context per-channel, but the LAG channels
get the same normalization as their corresponding originals — so the
relative drift signal (current minus 100-ago) survives.

Run on FD001 + MBA with norm_mode='revin' (the v26 default that erases
drift). 3 seeds each. If FD001 mean per-horizon AUROC > 0.93 we have a
universal fix that does not require dataset-family-specific norm_mode.
"""

import argparse
from pathlib import Path

from runner_v28 import run_and_persist, RES_DIR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', nargs='+', default=['FD001', 'MBA'])
    ap.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 456])
    ap.add_argument('--lags', type=int, nargs='+', default=[10, 50, 100])
    ap.add_argument('--pre-epochs', type=int, default=30)
    ap.add_argument('--ft-epochs', type=int, default=30)
    args = ap.parse_args()

    for ds in args.datasets:
        out = RES_DIR / f'phase2a_{ds}_lag.json'
        run_and_persist(
            dataset=ds, norm_mode='revin', seeds=args.seeds, out_json=out,
            lag_features=args.lags,
            pre_epochs=args.pre_epochs, ft_epochs=args.ft_epochs,
        )


if __name__ == '__main__':
    main()
