"""
Post-run analysis: generate embedding plots and update RESULTS.md with final numbers.
Run after run_experiments.py completes.

Usage:
  python experiments/v9/post_run_analysis.py
"""

import os
import sys
import json
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, '/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/v8')
sys.path.insert(0, '/home/sagemaker-user/IndustrialJEPA/mechanical-jepa')

CKPT_DIR = '/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/checkpoints'
RESULTS_DIR = '/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/experiments/v9/results'
PLOTS_DIR = '/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/analysis/plots'
os.makedirs(PLOTS_DIR, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_json(name):
    path = os.path.join(RESULTS_DIR, f'{name}.json')
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def main():
    print("=" * 60)
    print("V9 Post-Run Analysis")
    print("=" * 60)

    # Load all result files
    result_files = [f for f in os.listdir(RESULTS_DIR) if f.endswith('.json')]
    print(f"\nFound {len(result_files)} result files:")
    for f in sorted(result_files):
        print(f"  {f}")

    # Load and display results
    print("\n=== Results Summary ===")
    all_results = {}
    for f in sorted(result_files):
        name = f.replace('.json', '')
        with open(os.path.join(RESULTS_DIR, f)) as fp:
            r = json.load(fp)
        all_results[name] = r
        if 'rmse_mean' in r:
            print(f"  {name}: RMSE={r['rmse_mean']:.4f}±{r.get('rmse_std', 0):.4f}")

    # Generate embedding analysis if checkpoints exist
    print("\n=== Embedding Analysis ===")
    from analysis.embeddings import run_all_embedding_analysis
    from experiments.v9.run_experiments import load_rul_episodes_all

    print("Loading episodes...")
    episodes = load_rul_episodes_all(verbose=False)

    models, emb_stats = run_all_embedding_analysis(CKPT_DIR, episodes)

    # Generate pretraining loss curve plot
    print("\n=== Pretraining Loss Curves ===")
    pretrain_configs = ['all_8', 'compatible_6', 'bearing_rul_3']
    colors = {'all_8': '#d62728', 'compatible_6': '#1f77b4', 'bearing_rul_3': '#2ca02c'}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for config in pretrain_configs:
        r = all_results.get(f'pretrain_{config}')
        if r and 'history' in r:
            hist = r['history']
            if 'val_loss' in hist:
                val_hist = hist['val_loss']
                epochs = list(range(1, len(val_hist) + 1))
                best_ep = r.get('best_epoch', '?')
                ax1.plot(epochs, val_hist,
                         label=f'{config} (best@ep{best_ep})',
                         color=colors.get(config, 'black'), linewidth=2)

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('JEPA Val Loss')
    ax1.set_title('JEPA Pretraining Val Loss (first 20 epochs)\nDoes compatible_6 stabilize beyond epoch 2?')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Downstream RMSE bar chart
    rmse_data = []
    for config in pretrain_configs:
        r = all_results.get(f'pretrain_{config}')
        if r:
            rmse_data.append((config, r.get('rmse_mean', 0), r.get('rmse_std', 0),
                              r.get('best_epoch', '?')))

    if rmse_data:
        names, means, stds, best_eps = zip(*rmse_data)
        bar_colors = [colors.get(n, 'steelblue') for n in names]
        bars = ax2.bar(names, means, yerr=stds, color=bar_colors, alpha=0.8, capsize=6)
        ax2.axhline(y=0.189, color='red', linestyle='--', linewidth=1.5,
                    label='V8 JEPA+LSTM (0.189)')
        ax2.axhline(y=0.224, color='orange', linestyle='--', linewidth=1.5,
                    label='Elapsed time (0.224)')
        for bar, ep in zip(bars, best_eps):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                     f'ep{ep}', ha='center', fontsize=9)
        ax2.set_ylabel('RMSE')
        ax2.set_title('Downstream RUL RMSE by Pretraining Group\n(JEPA+LSTM, 5 seeds, 31 episodes)')
        ax2.legend(fontsize=9)
        ax2.grid(axis='y', alpha=0.3)
        ax2.set_ylim([0, 0.30])

    plt.tight_layout()
    save_path = os.path.join(PLOTS_DIR, 'v9_pretrain_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")

    # Write final RESULTS.md with actual numbers
    write_final_results(all_results, emb_stats)

    print("\n=== Post-Run Analysis Complete ===")


def write_final_results(all_results, emb_stats):
    """Write RESULTS.md with actual numbers from completed experiments."""
    results_path = '/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/experiments/v9/RESULTS.md'

    lines = []
    lines.append("# V9 Results: Data-First JEPA\n")
    lines.append("Session: 2026-04-09\n")
    lines.append("Dataset: 31 episodes (16 FEMTO + 15 XJTU-SY), 75/25 episode-based split\n")
    lines.append("V8 baselines: JEPA+LSTM=0.189±0.015, Hybrid JEPA+HC=0.055±0.004\n\n")

    lines.append("## Part B: Dataset Compatibility (key findings)\n")
    lines.append("- MAFAULDA: centroid 173Hz vs FEMTO 2453Hz. KL divergence=3.04. **EXCLUDED**.")
    lines.append("- MFPT: kurtosis 12.4±17 vs FEMTO 1.0±2. High impulse variance. **MARGINAL**.")
    lines.append("- All other sources (cwru, ims, paderborn, ottawa): KL<1.5. **COMPATIBLE**.\n")

    lines.append("## Part C: Pretraining Source Comparison (JEPA+LSTM, 5 seeds)\n")
    lines.append("| Config | Windows | Best Epoch | Val Loss | Emb Corr | RMSE ± std | vs V8 |")
    lines.append("|--------|:-------:|:----------:|:--------:|:--------:|:----------:|:------:|")

    for config in ['all_8', 'compatible_6', 'bearing_rul_3']:
        r = all_results.get(f'pretrain_{config}', {})
        if r:
            vs_v8 = (0.189 - r.get('rmse_mean', 0.189)) / 0.189 * 100
            lines.append(
                f"| {config:15s} | {r.get('n_windows', '?'):6} "
                f"| {r.get('best_epoch', '?'):5} "
                f"| {r.get('best_val_loss', 0):.4f} "
                f"| {r.get('max_dim_corr', 0):.3f} "
                f"| {r.get('rmse_mean', 0):.4f}±{r.get('rmse_std', 0):.4f} "
                f"| {vs_v8:+.1f}% |")
    lines.append("")

    lines.append("## Part D: TCN-Transformer (5 seeds)\n")
    lines.append("| Method | RMSE | ±std | vs V8 JEPA+LSTM | vs Elapsed |")
    lines.append("|--------|:----:|:----:|:---------------:|:----------:|")

    d_map = {
        'TCN-Transformer+HC': 'D1_tcn_transformer_hc',
        'JEPA+TCN-Transformer': 'D2_jepa_tcn_transformer',
        'JEPA+Deviation': 'D3_jepa_deviation',
        'JEPA+HC+Deviation': 'D4_hybrid_deviation',
    }
    for label, key in d_map.items():
        r = all_results.get(key, {})
        if r:
            vs_jepa = (0.189 - r.get('rmse_mean', 0.189)) / 0.189 * 100
            vs_elapsed = (0.224 - r.get('rmse_mean', 0.224)) / 0.224 * 100
            lines.append(f"| {label:30s} | {r.get('rmse_mean', 0):.4f} "
                         f"| {r.get('rmse_std', 0):.4f} "
                         f"| {vs_jepa:+.1f}% | {vs_elapsed:+.1f}% |")
    lines.append("")

    lines.append("## Part E: Masking Strategy\n")
    r = all_results.get('E1_block_masking', {})
    if r:
        vs_v8 = (0.189 - r.get('rmse_mean', 0.189)) / 0.189 * 100
        lines.append(f"Block masking (ep{r.get('best_epoch', '?')}): "
                     f"RMSE={r.get('rmse_mean', 0):.4f}±{r.get('rmse_std', 0):.4f} "
                     f"({vs_v8:+.1f}% vs V8)\n")

    lines.append("## Part F: Probabilistic Output\n")
    r = all_results.get('F1_probabilistic_lstm', {})
    if r:
        vs_v8 = (0.189 - r.get('rmse_mean', 0.189)) / 0.189 * 100
        lines.append(f"Heteroscedastic LSTM: "
                     f"RMSE={r.get('rmse_mean', 0):.4f}±{r.get('rmse_std', 0):.4f} "
                     f"({vs_v8:+.1f}% vs V8)\n")

    lines.append("## Embedding Quality\n")
    lines.append("| Encoder | Max Dim Corr | PC1 Corr |")
    lines.append("|---------|:-----------:|:--------:|")
    for name, stats in emb_stats.items():
        lines.append(f"| {name:25s} | {stats.get('max_dim_corr', 0):.3f} "
                     f"| {stats.get('pc1_corr', 0):.3f} |")
    lines.append("")

    lines.append("## Key Claims\n")
    pretrain_c6 = all_results.get('pretrain_compatible_6', {})
    pretrain_all = all_results.get('pretrain_all_8', {})
    if pretrain_c6 and pretrain_all:
        delta = pretrain_all.get('best_epoch', 2) - pretrain_c6.get('best_epoch', 2)
        rmse_delta = (pretrain_all.get('rmse_mean', 0.189) -
                      pretrain_c6.get('rmse_mean', 0.189)) / 0.189 * 100
        lines.append(f"1. **Compatible source pretraining best epoch**: "
                     f"{pretrain_c6.get('best_epoch', '?')} "
                     f"(vs {pretrain_all.get('best_epoch', '?')} for all_8)")
        lines.append(f"2. **Compatible vs all_8 downstream RMSE**: "
                     f"{rmse_delta:+.1f}% improvement")

    with open(results_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"RESULTS.md updated: {results_path}")


if __name__ == '__main__':
    main()
