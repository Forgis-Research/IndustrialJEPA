"""V27 Phase 2 — Ablation A: no RevIN, global z-score on FD001.

Data is pre-normalized with train-set mean/std (per channel). Model uses
``norm_mode='none'`` — RevIN is disabled entirely. This is the most direct
test of the over-stationarization hypothesis: if the slow sensor drift is
what discriminates engines, preserving it globally should recover Δt=10
AUROC from ~0.52 toward >0.7.
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

    out = RES_DIR / f'phase2_{args.dataset}_none.json'
    run_and_persist(
        dataset=args.dataset, norm_mode='none', seeds=args.seeds,
        out_json=out,
        pre_epochs=args.pre_epochs, ft_epochs=args.ft_epochs,
        n_cuts_train=args.n_cuts_train, n_cuts_val=args.n_cuts_val,
    )


if __name__ == '__main__':
    main()
