"""V30 Phase 7: post-Phase-3 finalisation.

Runs after Phase 3 + Phase 4a + Phase 8 subagents complete:
  1. Phase 4b: compute legacy metrics from stored Phase 3 surfaces.
  2. Phase 5: render figure 4 (FAM vs Chronos-2 surfaces) + figure 5
     (benchmark bar chart) + per-dataset 3-panel PNGs (already exist
     from Phase 3 runner).
  3. Render quarto notebook 30_v30_analysis.qmd → HTML.
  4. Update RESULTS.md with the final Phase 3 master table.
  5. Update SESSION_SUMMARY.md with the actual numbers.
  6. Print summary so the operator can verify before committing.

Does NOT commit/push - that's a separate explicit step.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

V30 = Path('/home/sagemaker-user/IndustrialJEPA/fam-jepa/experiments/v30')
REPO = Path('/home/sagemaker-user/IndustrialJEPA')


def run(cmd, cwd=None):
    print(f"\n$ {cmd}", flush=True)
    res = subprocess.run(cmd, shell=True, cwd=cwd or REPO,
                         capture_output=True, text=True)
    if res.stdout: print(res.stdout)
    if res.stderr: print('STDERR:', res.stderr[-2000:], file=sys.stderr)
    return res.returncode


def main():
    # 4b: legacy metrics
    rc = run('python experiments/v30/phase4_legacy_metrics.py',
             cwd=REPO / 'fam-jepa')
    if rc != 0:
        print('WARN: phase4_legacy_metrics.py failed', file=sys.stderr)

    # 5: figures
    rc = run('python experiments/v30/phase5_figures.py',
             cwd=REPO / 'fam-jepa')
    if rc != 0:
        print('WARN: phase5_figures.py failed', file=sys.stderr)

    # 5d: quarto notebook
    rc = run('quarto render notebooks/30_v30_analysis.qmd',
             cwd=REPO / 'fam-jepa')
    if rc != 0:
        print('WARN: quarto render failed (notebook still on disk)',
              file=sys.stderr)

    # Print phase3 summary to stdout for the SESSION_SUMMARY.md drafter.
    print('\n\n=== PHASE 3 SUMMARY ===\n', flush=True)
    run('python experiments/v30/phase3_summary.py', cwd=REPO / 'fam-jepa')

    print('\n\n=== Phase 4 legacy metrics ===\n', flush=True)
    p4 = V30 / 'results/phase4_legacy_metrics.json'
    if p4.exists():
        print(json.dumps(json.load(open(p4)), indent=2, default=str))

    print('\n\n=== Phase 4a SOTA findings ===\n', flush=True)
    p4a = V30 / 'results/phase4a_sota.json'
    if p4a.exists():
        print(json.dumps(json.load(open(p4a)), indent=2, default=str))
    else:
        print('(missing — Phase 4a subagent not done?)')

    print('\n\n=== Phase 8 dataset scouting ===\n', flush=True)
    p8 = V30 / 'results/phase8_dataset_scouting.json'
    if p8.exists():
        d = json.load(open(p8))
        print(f"Top 4 picks: {d.get('top_4_picks', [])}")
    else:
        print('(missing — Phase 8 subagent not done?)')


if __name__ == '__main__':
    main()
