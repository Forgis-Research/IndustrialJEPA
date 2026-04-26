"""V31 MOMENT MBA-only patch: append MBA seeds to existing moment_baseline.json.

The original baseline_moment.py run skipped MBA due to a data-format bug in the
MBA branch of load_dataset(). That bug is now fixed; this script runs only MBA
seeds and appends to the existing results so the FD001/FD003/BATADAL numbers
are not retrained.

Run with: conda run -n py310 python3 run_moment_mba.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

V31_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(V31_DIR))

import baseline_moment as bm  # noqa: E402


def main():
    seeds = [42, 123, 456]
    out_path = bm.RES_DIR / 'moment_baseline.json'

    with open(out_path) as f:
        existing = json.load(f)
    existing_results = existing.get('results', [])
    existing_tags = {r['tag'] for r in existing_results}

    print(f"Existing results: {len(existing_results)} runs", flush=True)
    print(f"Adding MBA seeds: {seeds}", flush=True)

    extractor = bm.MOMENTExtractor(bm.DEVICE)

    new_results = []
    for seed in seeds:
        tag = f"MBA_moment-mlp_s{seed}"
        if tag in existing_tags:
            print(f"[{tag}] skip - already present", flush=True)
            continue
        try:
            r = bm.run_moment_baseline('MBA', seed, extractor, new_results)
        except Exception as e:
            print(f"ERROR [{tag}]: {e}", flush=True)
            import traceback
            traceback.print_exc()

    if not new_results:
        print("No new results - exiting without overwrite", flush=True)
        return

    combined = existing_results + new_results
    payload = {
        'results': combined,
        'model': 'MOMENT-1-large',
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
    }
    with open(out_path, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved {len(combined)} total results to {out_path}", flush=True)

    means = [r['mean_h_auroc'] for r in new_results]
    print(f"\n[MBA] h-AUROC: {np.mean(means):.4f} +/- {np.std(means, ddof=0):.4f} "
          f"(seeds: {[f'{m:.4f}' for m in means]})", flush=True)


if __name__ == '__main__':
    main()
