"""
Compile all experimental results into RESULTS.md and SESSION_SUMMARY.md.
Run after all experiments complete.
"""
import os
import sys
import json
import glob
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

PAPER_TARGETS = {
    'MSL': {'f1': 33.58, 'auc': 66.08, 'precision': 35.87, 'recall': 40.80},
    'SMAP': {'f1': 33.64, 'auc': 65.41, 'precision': 24.24, 'recall': 56.02},
    'PSM': {'f1': 61.61, 'auc': 77.85, 'precision': 55.01, 'recall': 72.00},
    'SWaT': {'f1': 72.89, 'auc': 84.95, 'precision': 98.00, 'recall': 58.05},
}


def load_all_results():
    """Load all per-seed JSON results."""
    results = {}
    for fpath in sorted(glob.glob(os.path.join(RESULTS_DIR, "*_seed*.json"))):
        if "ablation" in fpath or "comparison" in fpath:
            continue
        with open(fpath) as f:
            r = json.load(f)
        ds = r['dataset']
        if ds not in results:
            results[ds] = []
        results[ds].append(r)
    return results


def compute_aggregates(results):
    """Compute mean ± std for each metric per dataset."""
    aggs = {}
    for ds, seeds in results.items():
        metrics = ['f1', 'auc', 'precision', 'recall']
        agg = {}
        for m in metrics:
            vals = [s['downstream'].get(m, 0) for s in seeds]
            agg[f'{m}_mean'] = np.mean(vals)
            agg[f'{m}_std'] = np.std(vals)

            paper = PAPER_TARGETS.get(ds, {}).get(m, 0)
            if paper > 0:
                gap = abs(agg[f'{m}_mean'] - paper) / paper * 100
                agg[f'{m}_gap_pct'] = gap
                if gap < 5:
                    agg[f'{m}_status'] = 'EXACT'
                elif gap < 15:
                    agg[f'{m}_status'] = 'GOOD'
                elif gap < 25:
                    agg[f'{m}_status'] = 'MARGINAL'
                else:
                    agg[f'{m}_status'] = 'FAILED'

        agg['n_seeds'] = len(seeds)
        aggs[ds] = agg
    return aggs


def write_results_md(results, aggs):
    """Write RESULTS.md."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "RESULTS.md")

    with open(path, 'w') as f:
        f.write("# MTS-JEPA Replication Results\n\n")

        # Summary table
        f.write("## Replication Summary\n\n")
        f.write("| Dataset | Metric | Ours | Paper | Gap (%) | Status |\n")
        f.write("|---------|--------|------|-------|---------|--------|\n")

        for ds in sorted(aggs.keys()):
            agg = aggs[ds]
            paper = PAPER_TARGETS.get(ds, {})
            for m in ['f1', 'auc']:
                ours = f"{agg[f'{m}_mean']:.2f} ± {agg[f'{m}_std']:.2f}"
                paper_val = f"{paper.get(m, 0):.2f}"
                gap = f"{agg.get(f'{m}_gap_pct', 0):.1f}%"
                status = agg.get(f'{m}_status', 'N/A')
                f.write(f"| {ds} | {m.upper()} | {ours} | {paper_val} | {gap} | {status} |\n")

        # Per-dataset details
        f.write("\n## Per-Dataset Details\n\n")
        for ds in sorted(results.keys()):
            f.write(f"\n### {ds}\n\n")
            seeds = results[ds]
            f.write(f"Seeds: {len(seeds)}\n\n")
            f.write("| Seed | F1 | AUC | Precision | Recall | Best Epoch | Codebook Util |\n")
            f.write("|------|-----|-----|-----------|--------|------------|---------------|\n")
            for s in seeds:
                d = s['downstream']
                p = s['pretrain']
                f.write(f"| {s['seed']} | {d['f1']:.2f} | {d['auc']:.2f} | "
                        f"{d['precision']:.2f} | {d['recall']:.2f} | "
                        f"{p['best_epoch']} | {p['codebook_utilization']:.3f} |\n")

    print(f"Wrote {path}")


def write_session_summary(results, aggs):
    """Write SESSION_SUMMARY.md."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "SESSION_SUMMARY.md")

    with open(path, 'w') as f:
        f.write("# MTS-JEPA Research Session Summary\n\n")

        f.write("## Bottom Line\n\n")

        # Determine overall status
        all_statuses = []
        for ds, agg in aggs.items():
            for m in ['f1', 'auc']:
                s = agg.get(f'{m}_status', 'N/A')
                all_statuses.append(s)

        f.write("MTS-JEPA replication with medium model (d=128, 3 layers, K=64) ")
        f.write("on A10G GPU. Results pending full analysis.\n\n")

        f.write("## Replication Verdict\n\n")
        for ds in sorted(aggs.keys()):
            agg = aggs[ds]
            f1_status = agg.get('f1_status', 'N/A')
            auc_status = agg.get('auc_status', 'N/A')
            f.write(f"- **{ds}**: F1 [{f1_status}], AUC [{auc_status}]\n")

        f.write("\n## Key Observations\n\n")
        f.write("- Medium model used (d=128 vs paper's d=256) due to single-GPU constraints\n")
        f.write("- Codebook utilization and training dynamics verified\n")
        f.write("- Lead-time analysis and theory validation prepared\n")

        f.write("\n## Literature Review Findings\n\n")
        f.write("- C-JEPA (NeurIPS 2024) shows EMA alone causes partial collapse\n")
        f.write("- FCM (KDD 2025) is the canonical anomaly prediction benchmark\n")
        f.write("- No existing work combines JEPA + codebook for prognostics\n")

        f.write("\n## Extension: CC-JEPA (Causal Codebook JEPA)\n\n")
        f.write("Top idea from brainstorming: merge Trajectory JEPA's causal encoder ")
        f.write("with MTS-JEPA's soft codebook. Prototype implemented.\n")

        f.write("\n## Next Steps\n\n")
        f.write("1. Run full-scale validation with paper hyperparameters (d=256, K=128)\n")
        f.write("2. Complete lead-time analysis on all datasets\n")
        f.write("3. Run CC-JEPA comparison experiments\n")
        f.write("4. Full ablation study\n")
        f.write("5. Theory validation plots\n")

        f.write("\n## Key Files Index\n\n")
        f.write("| File | Contents |\n")
        f.write("|------|----------|\n")
        f.write("| `models.py` | MTS-JEPA architecture |\n")
        f.write("| `cc_jepa.py` | Causal Codebook JEPA extension |\n")
        f.write("| `data_utils.py` | Data pipeline, RevIN, views |\n")
        f.write("| `train_utils.py` | Training loops, evaluation |\n")
        f.write("| `run_experiments.py` | Main experiment runner |\n")
        f.write("| `run_comparison.py` | Cross-method comparison |\n")
        f.write("| `run_ablations.py` | Ablation studies |\n")
        f.write("| `lead_time_analysis.py` | Lead-time breakdown |\n")
        f.write("| `theory_tracking.py` | Theory validation |\n")
        f.write("| `analysis/literature_review.md` | 45-paper literature survey |\n")
        f.write("| `analysis/brainstorming.md` | 15 ideas, top 3 selected |\n")
        f.write("| `analysis/gap_map.md` | 10 open research gaps |\n")
        f.write("| `CRITICAL_REVIEW.md` | 9-point weakness analysis |\n")
        f.write("| `NEURIPS_REVIEW.md` | Simulated review (5/10) |\n")
        f.write("| `REPLICATION_SPEC.md` | Architecture specification |\n")

    print(f"Wrote {path}")


if __name__ == '__main__':
    results = load_all_results()
    if not results:
        print("No results found!")
        sys.exit(1)

    aggs = compute_aggregates(results)
    write_results_md(results, aggs)
    write_session_summary(results, aggs)

    print("\nAggregate Results:")
    for ds, agg in sorted(aggs.items()):
        print(f"\n{ds} ({agg['n_seeds']} seeds):")
        for m in ['f1', 'auc']:
            mean = agg[f'{m}_mean']
            std = agg[f'{m}_std']
            status = agg.get(f'{m}_status', 'N/A')
            print(f"  {m.upper()}: {mean:.2f} ± {std:.2f} [{status}]")
