"""V28 follow-up — Try B v2: z-scored aux stat-prediction loss.

The v28 Phase 2B implementation had a magnitude problem (raw L1 ~700 vs
JEPA L1 ~0.04, a 17,500:1 ratio). The aux loss dominated and the encoder
learned to ignore the JEPA path.

This v2 standardises the (mean, std, slope) target stats against per-
corpus (μ, σ) constants computed once from the training loader, so the
aux L1 is in z-units and stays comparable to the JEPA L1.

Run on FD001 + MBA × 3 seeds, RevIN preserved (the v28 Phase 2B condition).
"""

import argparse
from runner_v28 import run_and_persist, RES_DIR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', nargs='+', default=['FD001', 'MBA'])
    ap.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 456])
    args = ap.parse_args()

    for ds in args.datasets:
        out = RES_DIR / f'phase8_{ds}_statz.json'
        run_and_persist(
            dataset=ds, norm_mode='revin', seeds=args.seeds, out_json=out,
            aux_stat=True, stat_normalize=True,
        )


if __name__ == '__main__':
    main()
