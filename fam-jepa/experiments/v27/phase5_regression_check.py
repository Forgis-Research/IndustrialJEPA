"""V27 Phase 5 — regression check on MBA + SMAP with winning variant.

Phase 2-4 established norm_mode='none' + global z-score as the winner on
FD001 (+0.12 AUROC at dt=10, +0.26 at dt=50). We now check that it does
not regress the datasets where per-instance RevIN was winning in v26:

  - MBA (cardiac, local waveform shape events): v26 dt=1 AUROC ~ 0.78
  - SMAP (spacecraft, local pattern anomalies):  v26 dt=1 AUROC ~ 0.59

If either drops meaningfully vs v26, we have a domain tradeoff rather
than a universal fix. That is itself a paper-worthy finding, per the
SESSION_PROMPT.
"""

import argparse
from pathlib import Path

from _runner import run_and_persist, RES_DIR


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', default=['MBA', 'SMAP'])
    parser.add_argument('--norm-mode', default='none',
                        choices=['none', 'last_value', 'revin_stat', 'revin'])
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 456])
    parser.add_argument('--pre-epochs', type=int, default=30)
    parser.add_argument('--ft-epochs', type=int, default=30)
    parser.add_argument('--n-cuts-train', type=int, default=40)
    parser.add_argument('--n-cuts-val', type=int, default=10)
    args = parser.parse_args()

    for ds in args.datasets:
        out = RES_DIR / f'phase5_{ds}_{args.norm_mode}.json'
        run_and_persist(
            dataset=ds, norm_mode=args.norm_mode, seeds=args.seeds,
            out_json=out,
            pre_epochs=args.pre_epochs, ft_epochs=args.ft_epochs,
            n_cuts_train=args.n_cuts_train, n_cuts_val=args.n_cuts_val,
        )


if __name__ == '__main__':
    main()
