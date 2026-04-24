"""V27 Phase 6 — full benchmark with per-dataset normalization.

Phase 5 established a domain tradeoff:

  - Degradation datasets (C-MAPSS FD00x): need absolute sensor levels
    preserved. norm_mode='none' with train-set global z-score wins
    (Δt=50 AUROC 0.53 → 0.79 on FD001).
  - Multi-entity anomaly datasets (SMAP, MSL, SMD): entity scales
    differ so strongly that global z-score collapses to base-rate
    predictions. Per-instance RevIN (v26 default) remains optimal.

This phase populates the final paper tables:
  - FD002, FD003: run from scratch with norm_mode='none'
  - FD001:         reuse Phase 2 surfaces (also norm_mode='none')
  - SMAP, MSL, PSM, SMD, MBA, PhysioNet: reuse v26 surfaces
    (norm_mode='revin', already on disk at experiments/v26/surfaces).
"""

import argparse
from pathlib import Path

from _runner import run_and_persist, RES_DIR

DEGRADATION_DATASETS = ['FD002', 'FD003']  # FD001 already done in Phase 2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', default=DEGRADATION_DATASETS)
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 456])
    parser.add_argument('--pre-epochs', type=int, default=50)
    parser.add_argument('--ft-epochs', type=int, default=40)
    parser.add_argument('--n-cuts-train', type=int, default=60)
    parser.add_argument('--n-cuts-val', type=int, default=20)
    args = parser.parse_args()

    for ds in args.datasets:
        out = RES_DIR / f'phase6_{ds}_none.json'
        run_and_persist(
            dataset=ds, norm_mode='none', seeds=args.seeds,
            out_json=out,
            pre_epochs=args.pre_epochs, ft_epochs=args.ft_epochs,
            n_cuts_train=args.n_cuts_train, n_cuts_val=args.n_cuts_val,
        )


if __name__ == '__main__':
    main()
