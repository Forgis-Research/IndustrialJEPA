"""V30 Phase 0: Dense K=150 + MonotoneCDF quick ablation on FD001 (s42).

Default: dense discrete hazard CDF over horizons range(1, 151).
Quick ablation: MonotoneCDF (continuous, monotone-by-construction).

Both reuse the v29 FD001 norm=none seed=42 pretrained encoder.
Decision rule (per SESSION_PROMPT):
  IF MonotoneCDF h-AUROC >= dense discrete AND visually smoother → adopt
  ELSE use dense discrete K=150 for Phase 3.

Save: results/phase0_decision.json
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from _runner_v30 import run_v30, RES_DIR, PNG_DIR, find_pretrain_ckpt


def main():
    t_start = time.time()
    dataset = 'FD001'; seed = 42
    dense_horizons = list(range(1, 151))

    # Variant A: dense discrete K=150
    print("\n>>> VARIANT A: dense discrete CDF (K=150) <<<", flush=True)
    pre_ckpt = find_pretrain_ckpt(dataset, 'none', seed, 'mlp')
    print(f"  pretrained encoder: {pre_ckpt}", flush=True)
    res_dense = run_v30(
        dataset=dataset, seed=seed,
        eval_horizons=dense_horizons,
        event_head_kind='discrete_hazard',
        train_horizons_dense=20,        # sample 20 random horizons per batch
        tag_suffix='p0',
        init_from_ckpt=pre_ckpt,
        ft_epochs=30, ft_patience=8,
        sort_panel_by_tte=True,
    )

    # Variant B: MonotoneCDF
    print("\n>>> VARIANT B: MonotoneCDF (hidden=64, 3 layers) <<<", flush=True)
    res_mono = run_v30(
        dataset=dataset, seed=seed,
        eval_horizons=dense_horizons,
        event_head_kind='monotone_cdf',
        train_horizons_dense=20,
        tag_suffix='p0',
        init_from_ckpt=pre_ckpt,
        ft_epochs=30, ft_patience=8,
        sort_panel_by_tte=True,
    )

    # Decide
    a_h = res_dense['mean_h_auroc']
    b_h = res_mono['mean_h_auroc']
    chosen = 'monotone_cdf' if b_h >= a_h - 1e-4 else 'discrete_hazard'
    decision = {
        'fd001_seed': seed,
        'variant_a_dense_discrete': {
            'mean_h_auroc': a_h, 'pooled_auprc': res_dense['pooled_auprc'],
            'png': res_dense['png_path'], 'tag': res_dense['tag'],
            'ft_time_s': res_dense['ft_time_s'],
        },
        'variant_b_monotone_cdf': {
            'mean_h_auroc': b_h, 'pooled_auprc': res_mono['pooled_auprc'],
            'png': res_mono['png_path'], 'tag': res_mono['tag'],
            'ft_time_s': res_mono['ft_time_s'],
        },
        'chosen_head': chosen,
        'reason': (f"MonotoneCDF h-AUROC ({b_h:.4f}) "
                   f"{'≥' if b_h >= a_h else '<'} dense discrete "
                   f"({a_h:.4f}); chose {chosen}."),
        'total_time_s': time.time() - t_start,
    }
    out = RES_DIR / 'phase0_decision.json'
    with open(out, 'w') as f:
        json.dump(decision, f, indent=2)
    print(f"\nwrote {out}\n=== Phase 0 decision: {chosen} ===", flush=True)
    print(json.dumps(decision, indent=2))


if __name__ == '__main__':
    main()
