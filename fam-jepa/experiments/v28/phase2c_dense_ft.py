"""V28 Phase 2 Try C — dense-horizon finetuning.

The v27 baseline finetunes on a sparse set of 7-8 horizons. The predictor
takes Δt as a continuous scalar, so during training we sample K random
integers from [1, max_horizon] each batch — supplying the predictor with
gradient signal at every horizon, not only the sparse evaluation grid.

Eval still uses the fixed sparse horizons so AUPRC/AUROC are comparable
to the v27 numbers.

Run on FD001 + MBA. FD001 uses the v27 best norm_mode ('none'); MBA uses
'revin' which won on the v27 anomaly benchmark.
"""

import argparse
from runner_v28 import run_and_persist, RES_DIR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', nargs='+', default=['FD001', 'MBA'])
    ap.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 456])
    ap.add_argument('--k-dense', type=int, default=20)
    ap.add_argument('--pre-epochs', type=int, default=30)
    ap.add_argument('--ft-epochs', type=int, default=30)
    args = ap.parse_args()

    for ds in args.datasets:
        norm = 'none' if ds.startswith('FD') else 'revin'
        out = RES_DIR / f'phase2c_{ds}_dense.json'
        run_and_persist(
            dataset=ds, norm_mode=norm, seeds=args.seeds, out_json=out,
            dense_ft=True, k_dense=args.k_dense,
            pre_epochs=args.pre_epochs, ft_epochs=args.ft_epochs,
        )


if __name__ == '__main__':
    main()
