"""V27 Phase 3 — Ablation B: NLinear-style last-value subtraction on FD001.

Model uses ``norm_mode='last_value'``. The encoder subtracts the last
valid observation per channel from the whole context (no scale). This
preserves the within-window drift while removing absolute offset,
following Zeng et al. (AAAI 2023) "Are Transformers Effective for Time
Series Forecasting?".
"""

import argparse
from pathlib import Path

from _runner import run_and_persist, RES_DIR


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 456])
    parser.add_argument('--pre-epochs', type=int, default=50)
    parser.add_argument('--ft-epochs', type=int, default=40)
    parser.add_argument('--n-cuts-train', type=int, default=60)
    parser.add_argument('--n-cuts-val', type=int, default=20)
    parser.add_argument('--dataset', default='FD001')
    args = parser.parse_args()

    out = RES_DIR / f'phase3_{args.dataset}_last_value.json'
    run_and_persist(
        dataset=args.dataset, norm_mode='last_value', seeds=args.seeds,
        out_json=out,
        pre_epochs=args.pre_epochs, ft_epochs=args.ft_epochs,
        n_cuts_train=args.n_cuts_train, n_cuts_val=args.n_cuts_val,
    )


if __name__ == '__main__':
    main()
