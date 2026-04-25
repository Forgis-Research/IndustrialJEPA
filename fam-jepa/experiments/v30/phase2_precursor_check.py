"""V30 Phase 2: precursor check on MSL, SMD, PhysioNet.

Quick 1-seed check to determine which datasets have predictable precursor
signal at the chosen Phase 0 head. Skip CHB-MIT (v29 confirmed null).

Decision rules (per SESSION_PROMPT):
  h-AUROC > 0.55  → include in Phase 3
  0.50–0.55       → render surface, marginal — include if temporal structure
  < 0.50          → SKIP, document null

For MSL: 3 seeds (was n=1 in v29 with 0.438, suspicious).
For SMD: 2 more seeds (was n=1 in v29 with 0.616).
For PhysioNet: 1 seed (never run with h-AUROC). If > 0.55, run 2 more.

Save: results/phase2_precursor_check.json
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from _runner_v30 import run_v30, RES_DIR, find_pretrain_ckpt

# Use sparse horizons to keep this fast (precursor-check stage).
SPARSE_H = [1, 5, 10, 20, 50, 100, 150, 200]


def precursor_check(dataset: str, seeds: list, horizons: list,
                    n_cuts_train: int = 30) -> list:
    out = []
    for sd in seeds:
        pre_ckpt = find_pretrain_ckpt(dataset, None, sd)
        # NORM_POLICY auto-applies; pretrained ckpt path uses default norm.
        try:
            r = run_v30(dataset=dataset, seed=sd, eval_horizons=horizons,
                        event_head_kind='discrete_hazard',
                        train_horizons_dense=20,
                        tag_suffix='p2',
                        init_from_ckpt=pre_ckpt,
                        ft_epochs=30, ft_patience=8,
                        n_cuts_train=n_cuts_train)
            if r is not None:
                out.append(r)
        except Exception as e:
            print(f"  ERROR {dataset} s{sd}: {e}", flush=True)
            import traceback; traceback.print_exc()
    return out


def main():
    t0 = time.time()
    decision = {'datasets': {}, 'horizons': SPARSE_H,
                'sparse_horizons_note': 'sparse {1,5,10,20,50,100,150,200} for fast triage; Phase 3 uses dense K=150'}

    # CHB-MIT — skip per v29 confirmation
    decision['datasets']['CHBMIT'] = {
        'hauroc': [0.5002, 0.4982, 0.4938],
        'mean': 0.4974, 'decision': 'skip (null confirmed v29)',
        'reason': 'v29 3-seed result with bug-fixed onset labels: 0.497 ± 0.003 < base 0.513',
    }
    print(f"\n>>> CHBMIT: skip per v29\n", flush=True)

    # MSL — 3 seeds (v29 had n=1 with 0.438, suspicious)
    print(f"\n>>> MSL — 3 seeds <<<\n", flush=True)
    msl = precursor_check('MSL', [42, 123, 456], SPARSE_H)
    msl_h = [r['mean_h_auroc'] for r in msl]
    msl_mean = float(np.mean(msl_h)) if msl_h else None
    msl_dec = ('include' if msl_mean and msl_mean > 0.55 else
               'marginal-render' if msl_mean and msl_mean > 0.50 else
               'skip')
    decision['datasets']['MSL'] = {
        'hauroc': msl_h, 'mean': msl_mean,
        'std': float(np.std(msl_h, ddof=1)) if len(msl_h) > 1 else None,
        'tags': [r['tag'] for r in msl], 'decision': msl_dec,
    }

    # SMD — 2 more seeds (v29 had only s42)
    print(f"\n>>> SMD — 3 seeds <<<\n", flush=True)
    smd = precursor_check('SMD', [42, 123, 456], SPARSE_H)
    smd_h = [r['mean_h_auroc'] for r in smd]
    smd_mean = float(np.mean(smd_h)) if smd_h else None
    smd_dec = ('include' if smd_mean and smd_mean > 0.55 else
               'marginal-render' if smd_mean and smd_mean > 0.50 else
               'skip')
    decision['datasets']['SMD'] = {
        'hauroc': smd_h, 'mean': smd_mean,
        'std': float(np.std(smd_h, ddof=1)) if len(smd_h) > 1 else None,
        'tags': [r['tag'] for r in smd], 'decision': smd_dec,
    }

    # PhysioNet — 1 seed first; if > 0.55, run 2 more
    print(f"\n>>> PhysioNet — 1 seed first <<<\n", flush=True)
    try:
        # Skip if loader missing (PhysioNet may not be wired into LOADERS yet)
        from _runner_v29 import LOADERS as L
        physio_avail = 'PhysioNet' in L or 'sepsis' in L
    except Exception:
        physio_avail = False
    if not physio_avail:
        decision['datasets']['PhysioNet'] = {
            'hauroc': [], 'mean': None,
            'decision': 'skip (no loader registered in LOADERS)',
            'reason': 'PhysioNet/sepsis not in v29 LOADERS dict; deferred to v31',
        }
    else:
        ds_name = 'PhysioNet' if 'PhysioNet' in L else 'sepsis'
        physio = precursor_check(ds_name, [42], SPARSE_H)
        physio_h = [r['mean_h_auroc'] for r in physio] if physio else []
        if physio_h and physio_h[0] > 0.55:
            print(f"\n>>> PhysioNet seed 42 = {physio_h[0]:.4f} > 0.55 → 2 more <<<\n",
                  flush=True)
            extra = precursor_check(ds_name, [123, 456], SPARSE_H)
            physio_h.extend([r['mean_h_auroc'] for r in extra])
            physio.extend(extra)
        physio_mean = float(np.mean(physio_h)) if physio_h else None
        physio_dec = ('include' if physio_mean and physio_mean > 0.55 else
                      'marginal-render' if physio_mean and physio_mean > 0.50 else
                      'skip')
        decision['datasets']['PhysioNet'] = {
            'hauroc': physio_h, 'mean': physio_mean,
            'std': float(np.std(physio_h, ddof=1)) if len(physio_h) > 1 else None,
            'tags': [r['tag'] for r in physio], 'decision': physio_dec,
        }

    decision['time_total_s'] = time.time() - t0
    out = RES_DIR / 'phase2_precursor_check.json'
    with open(out, 'w') as f:
        json.dump(decision, f, indent=2)
    print(f"\nwrote {out}\n", flush=True)
    print(json.dumps(decision, indent=2))


if __name__ == '__main__':
    main()
