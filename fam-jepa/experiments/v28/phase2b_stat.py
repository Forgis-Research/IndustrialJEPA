"""V28 Phase 2 Try B — auxiliary stat-prediction loss during pretraining.

Inspired by LaT-PFN (arXiv:2405.10093). During pretraining we add a head
that predicts the TARGET interval's per-channel (mean, std, slope) from
h_t. The target stats are a self-supervised signal of "what regime is
the system in" — degradation, healthy, etc.

Run on FD001 + MBA with norm_mode='revin'. The hypothesis: even though
RevIN strips per-context distributional shift, asking the encoder to
predict the target's stats forces it to encode regime information from
features that survive RevIN (high-frequency residuals, cross-channel
correlations). 3 seeds each.
"""

import argparse
from runner_v28 import run_and_persist, RES_DIR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', nargs='+', default=['FD001', 'MBA'])
    ap.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 456])
    ap.add_argument('--pre-epochs', type=int, default=30)
    ap.add_argument('--ft-epochs', type=int, default=30)
    args = ap.parse_args()

    for ds in args.datasets:
        out = RES_DIR / f'phase2b_{ds}_stat.json'
        run_and_persist(
            dataset=ds, norm_mode='revin', seeds=args.seeds, out_json=out,
            aux_stat=True,
            pre_epochs=args.pre_epochs, ft_epochs=args.ft_epochs,
        )


if __name__ == '__main__':
    main()
