"""V29 Phase 1: pretrain + pred-FT + eval on the three new datasets.

Runs SKAB, ETTm1, CHB-MIT × 3 seeds each with the canonical v27/v28 setup
(MLP predictor, norm_mode per the policy table). Chronos-2 baselines run
separately (phase1_chronos2.py).

Usage:
  python experiments/v29/phase1_new_datasets.py [--datasets SKAB,ETTm1]
  python experiments/v29/phase1_new_datasets.py --datasets CHBMIT --seeds 42

By default skips CHBMIT if data isn't downloaded yet.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

V29_DIR = Path('/home/sagemaker-user/IndustrialJEPA/fam-jepa/experiments/v29')
sys.path.insert(0, str(V29_DIR))

from _runner_v29 import run_and_persist, RES_DIR


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', default='SKAB,ETTm1,CHBMIT')
    parser.add_argument('--seeds', default='42,123,456')
    parser.add_argument('--predictor', default='mlp',
                        choices=['mlp', 'transformer'])
    parser.add_argument('--pre_epochs', type=int, default=30)
    parser.add_argument('--ft_epochs', type=int, default=30)
    args = parser.parse_args()

    datasets = [d.strip() for d in args.datasets.split(',') if d.strip()]
    seeds = [int(s) for s in args.seeds.split(',')]

    # Per-dataset overrides for very large streams (CHB-MIT @ 32Hz, 7.7M steps).
    overrides = {
        'CHBMIT': dict(pre_epochs=15, ft_epochs=12, n_cuts_train=80,
                       n_cuts_val=20, ft_batch=128),
    }

    for ds in datasets:
        out_json = RES_DIR / f'phase1_{ds}_{args.predictor}.json'
        kw = dict(predictor_kind=args.predictor,
                  pre_epochs=args.pre_epochs, ft_epochs=args.ft_epochs)
        kw.update(overrides.get(ds, {}))
        print(f"\n{'#'*70}\n# Phase 1: {ds}  predictor={args.predictor}  "
              f"seeds={seeds}  kw={kw}\n{'#'*70}", flush=True)
        try:
            run_and_persist(ds, seeds=seeds, out_json=out_json, **kw)
        except FileNotFoundError as e:
            print(f"  SKIP {ds}: {e}", flush=True)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"  ERROR {ds}: {e}", flush=True)


if __name__ == '__main__':
    main()
