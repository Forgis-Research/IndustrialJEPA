"""V27 Phase 4 — Ablation C: RevIN + stat token on FD001.

Model uses ``norm_mode='revin_stat'``. Context encoder still applies
RevIN so training stability is preserved, but the removed (mean, std)
pair is projected into a "stat token" prepended at position 0 of the
token sequence. Causal attention makes the stat token visible to every
later patch — the encoder can read distributional state alongside the
normalized sequence dynamics.

The target encoder uses plain RevIN (no stat token) — asymmetry by
design: the target interval is short and has no "lifecycle" for the
stat token to represent.

If this beats the baseline AND doesn't regress MBA/SMAP, it's the
permanent fix: RevIN for training stability + statistics for degradation
awareness (Liu+ NeurIPS 2022 de-stationary attention, simplified).
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

    out = RES_DIR / f'phase4_{args.dataset}_revin_stat.json'
    run_and_persist(
        dataset=args.dataset, norm_mode='revin_stat', seeds=args.seeds,
        out_json=out,
        pre_epochs=args.pre_epochs, ft_epochs=args.ft_epochs,
        n_cuts_train=args.n_cuts_train, n_cuts_val=args.n_cuts_val,
    )


if __name__ == '__main__':
    main()
