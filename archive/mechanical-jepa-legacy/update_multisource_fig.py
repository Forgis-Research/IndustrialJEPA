"""
Run this after multi_source_hf_pretrain.py completes to update the multi-source figure.
Loads multisource_pretrain.json and generates the final multi-source comparison figure.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS_DIR = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/results')
PLOTS_DIR = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/notebooks/plots')

def main():
    with open(RESULTS_DIR / 'multisource_pretrain.json') as f:
        data = json.load(f)

    # Check how many seeds we have (filter out non-list keys like _summary)
    methods_in_data = [k for k in data.keys() if isinstance(data[k], list)]
    n_seeds_available = min(len(data.get(m, [])) for m in methods_in_data if data.get(m))
    print(f"Methods: {methods_in_data}")
    print(f"Seeds available: {n_seeds_available}")
    for method in methods_in_data:
        seeds = data[method]
        if not isinstance(seeds, list) or not seeds:
            continue
        cwru = [s['cwru_f1'] for s in seeds]
        pad = [s['pad_f1'] for s in seeds]
        print(f"  {method}: CWRU={np.mean(cwru):.3f}+/-{np.std(cwru):.3f}, Pad={np.mean(pad):.3f}+/-{np.std(pad):.3f} (n={len(seeds)})")

    if n_seeds_available < 3:
        print(f"\nOnly {n_seeds_available} seed(s) complete. Showing partial results.")

    # Build comparison
    method_order = ['jepa_v2_cwru_pretrained', 'random_init', 'jepa_gear_pretrained', 'jepa_multisource']
    method_labels = ['CWRU-pretrained\n(reference)', 'Random Init', 'Gear-pretrained\nonly', 'Multi-source\nCWRU+Gear']
    colors = ['#F44336', '#9E9E9E', '#FF9800', '#2196F3']

    metrics = ['cwru_f1', 'pad_f1']
    metric_labels = ['CWRU F1', 'Paderborn F1']

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, metric, metric_label in zip(axes, metrics, metric_labels):
        x = np.arange(len(method_order))
        means, stds = [], []
        for method in method_order:
            seeds = data.get(method, [])
            if seeds:
                vals = [s[metric] for s in seeds]
                means.append(np.mean(vals))
                stds.append(np.std(vals) if len(vals) > 1 else 0)
            else:
                means.append(0)
                stds.append(0)

        bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors, alpha=0.85, width=0.6)

        for bar, mean in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                    f'{mean:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

        ax.set_xticks(x)
        ax.set_xticklabels(method_labels, fontsize=9)
        ax.set_ylabel(metric_label)
        ax.set_title(f'Cross-Component Transfer: {metric_label}', fontsize=11)
        ax.set_ylim([0.3, 1.05])
        ax.grid(axis='y', alpha=0.3)

    plt.suptitle(f'Multi-Source Pretraining Results (n={n_seeds_available} seed{"s" if n_seeds_available != 1 else ""})\n'
                 'Gear-pretrained JEPA shows negative transfer to bearings',
                 fontsize=11, y=1.02)
    plt.tight_layout()

    out = PLOTS_DIR / f'fig7_multisource_n{n_seeds_available}.png'
    plt.savefig(out, bbox_inches='tight', dpi=150)
    plt.savefig(str(out).replace('.png', '.pdf'), bbox_inches='tight')
    print(f"\nSaved to {out}")
    plt.close()

    # Print summary table
    print("\n=== Multi-Source Results Summary ===")
    print(f"{'Method':<30} {'CWRU F1':<18} {'Pad F1':<18} {'n seeds'}")
    print("-" * 75)
    for method in method_order:
        seeds = data.get(method, [])
        if seeds:
            cwru = [s['cwru_f1'] for s in seeds]
            pad = [s['pad_f1'] for s in seeds]
            label = {
                'jepa_v2_cwru_pretrained': 'CWRU pretrained (ref)',
                'random_init': 'Random init',
                'jepa_gear_pretrained': 'Gear pretrained',
                'jepa_multisource': 'Multi-source CWRU+Gear',
            }.get(method, method)
            print(f"{label:<30} {np.mean(cwru):.3f} +/- {np.std(cwru):.3f}   {np.mean(pad):.3f} +/- {np.std(pad):.3f}   {len(seeds)}")


if __name__ == '__main__':
    main()
