"""
Generate all required figures for the MTS-JEPA replication report.
"""
import os
import json
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")

PAPER_TARGETS = {
    'MSL': {'f1': 33.58, 'auc': 66.08},
    'SMAP': {'f1': 33.64, 'auc': 65.41},
    'PSM': {'f1': 61.61, 'auc': 77.85},
}


def load_results():
    """Load all per-seed results."""
    results = {}
    for f in sorted(glob.glob(os.path.join(RESULTS_DIR, "*_seed*.json"))):
        if "ablation" in f or "comparison" in f:
            continue
        with open(f) as fh:
            r = json.load(fh)
        ds = r['dataset']
        if ds not in results:
            results[ds] = []
        results[ds].append(r)
    return results


def plot_replication_comparison(results):
    """Bar chart: paper vs our replication per dataset."""
    os.makedirs(FIGURES_DIR, exist_ok=True)

    datasets = sorted(results.keys())
    if not datasets:
        print("No results to plot")
        return

    metrics = ['f1', 'auc']
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for i, metric in enumerate(metrics):
        ax = axes[i]
        x = np.arange(len(datasets))
        width = 0.35

        paper_vals = [PAPER_TARGETS.get(ds, {}).get(metric, 0) for ds in datasets]
        our_means = []
        our_stds = []
        for ds in datasets:
            vals = [r['downstream'].get(metric, 0) for r in results[ds]]
            our_means.append(np.mean(vals))
            our_stds.append(np.std(vals))

        ax.bar(x - width/2, paper_vals, width, label='Paper', color='steelblue', alpha=0.8)
        ax.bar(x + width/2, our_means, width, yerr=our_stds, label='Ours',
               color='coral', alpha=0.8, capsize=5)

        ax.set_xlabel('Dataset')
        ax.set_ylabel(metric.upper() + ' (%)')
        ax.set_title(f'{metric.upper()} Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(datasets)
        ax.legend()
        ax.set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'replication_comparison.png'), dpi=150)
    plt.close()
    print("Saved replication_comparison.png")


def plot_lead_time_breakdown():
    """Pie chart: TRUE_PREDICTION vs CONTINUATION vs BOUNDARY."""
    os.makedirs(FIGURES_DIR, exist_ok=True)

    data = {
        'PSM': {'TRUE_PREDICTION': 45, 'CONTINUATION': 246, 'BOUNDARY': 0},
        'MSL': {'TRUE_PREDICTION': 35, 'CONTINUATION': 80, 'BOUNDARY': 0},
        'SMAP': {'TRUE_PREDICTION': 67, 'CONTINUATION': 548, 'BOUNDARY': 0},
    }

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    colors = ['#2ecc71', '#e74c3c', '#f39c12']

    for i, (ds, vals) in enumerate(data.items()):
        ax = axes[i]
        labels = list(vals.keys())
        sizes = list(vals.values())

        # Remove zero entries
        non_zero = [(l, s) for l, s in zip(labels, sizes) if s > 0]
        labels, sizes = zip(*non_zero) if non_zero else ([], [])

        ax.pie(sizes, labels=labels, colors=colors[:len(sizes)],
               autopct='%1.1f%%', startangle=90, textprops={'fontsize': 10})
        ax.set_title(f'{ds}\n(n={sum(sizes)} anomalous targets)')

    plt.suptitle('Lead-Time Analysis: Prediction Type Breakdown', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'lead_time_breakdown.png'), dpi=150)
    plt.close()
    print("Saved lead_time_breakdown.png")


def plot_codebook_utilization(results):
    """Histogram of codebook utilization across seeds."""
    os.makedirs(FIGURES_DIR, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))

    all_utils = {}
    for ds, seeds in results.items():
        utils = [r['pretrain']['codebook_utilization'] for r in seeds]
        all_utils[ds] = utils

    x_pos = 0
    colors = ['steelblue', 'coral', 'seagreen']
    for i, (ds, utils) in enumerate(all_utils.items()):
        positions = [x_pos + j * 0.3 for j in range(len(utils))]
        ax.bar(positions, utils, width=0.25, color=colors[i], alpha=0.8, label=ds)
        x_pos += len(utils) * 0.3 + 0.5

    ax.set_ylabel('Codebook Utilization')
    ax.set_title('Codebook Utilization Across Seeds and Datasets')
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'codebook_utilization.png'), dpi=150)
    plt.close()
    print("Saved codebook_utilization.png")


if __name__ == '__main__':
    results = load_results()
    print(f"Loaded results for: {list(results.keys())}")

    plot_replication_comparison(results)
    plot_lead_time_breakdown()
    plot_codebook_utilization(results)
    print("\nAll figures saved to", FIGURES_DIR)
