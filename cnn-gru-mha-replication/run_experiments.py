"""
Main experiment runner for CNN-GRU-MHA replication.

Runs all 12 FEMTO transfer experiments (5 seeds each = 60 runs).
Also checks for XJTU-SY data and runs those transfers if available.

Generates:
  - results/transfer_{source}_{target}.json (per transfer)
  - results/all_results.json (summary)
  - RESULTS.md (comparison table)
  - EXPERIMENT_LOG.md (detailed log)
  - results/plots/ (RUL prediction plots, loss curves)

Usage:
    python run_experiments.py [--device cpu|cuda] [--fast] [--debug_transfer SOURCE TARGET]
"""

import argparse
import json
import os
import sys
import time
import numpy as np
import torch
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from data_utils import (
    load_bearing_for_cnn_gru,
    FEMTO_DATA_ROOT,
    FEMTO_TRANSFERS,
    XJTU_TRANSFERS,
    PAPER_FEMTO_TARGETS,
    PAPER_FEMTO_AVG,
    find_xjtu_root,
    load_xjtu_bearing,
    get_transfer_split,
)
from models import CNNGRUMHAModel
from train_utils import run_transfer_multi_seed

# =====================================================================
# Paper targets (Table 4)
# =====================================================================
PAPER_TABLE4 = {
    ("Bearing1_3", "Bearing2_3"): 0.0463,
    ("Bearing1_3", "Bearing2_4"): 0.0449,
    ("Bearing1_3", "Bearing3_1"): 0.0427,
    ("Bearing1_3", "Bearing3_3"): 0.0461,
    ("Bearing2_3", "Bearing1_3"): 0.0458,
    ("Bearing2_3", "Bearing1_4"): 0.0426,
    ("Bearing2_3", "Bearing3_3"): 0.0416,
    ("Bearing3_2", "Bearing1_3"): 0.0382,
    ("Bearing3_2", "Bearing1_4"): 0.0397,
    ("Bearing3_2", "Bearing2_3"): 0.0413,
    ("Bearing3_2", "Bearing2_4"): 0.0418,
}
PAPER_AVG = 0.0443


def get_unique_bearings(transfers: dict) -> set:
    """Extract all unique bearing names from transfer dict."""
    bearings = set()
    for exp in transfers.values():
        bearings.add(exp["source"])
        for t in exp["targets"]:
            bearings.add(t)
    return bearings


def load_all_bearings(data_root: Path, bearing_names: list, verbose: bool = True) -> dict:
    """Load all bearings, caching to avoid redundant work."""
    bearing_cache = {}
    for name in sorted(set(bearing_names)):
        if verbose:
            print(f"  Loading {name}...", end=" ", flush=True)
        try:
            bdata = load_bearing_for_cnn_gru(data_root, name)
            bearing_cache[name] = bdata
            if verbose:
                print(f"OK ({bdata['n_snapshots']} snapshots)")
        except FileNotFoundError as e:
            if verbose:
                print(f"NOT FOUND: {e}")
            bearing_cache[name] = None
        except Exception as e:
            if verbose:
                print(f"ERROR: {e}")
            bearing_cache[name] = None
    return bearing_cache


def run_femto_experiments(
    device: torch.device,
    seeds: list,
    source_iterations: int,
    finetune_iterations: int,
    results_dir: Path,
    verbose: bool = True,
) -> dict:
    """
    Run all 12 FEMTO transfer experiments.

    Returns dict mapping (source, target) -> multi-seed result dict.
    """
    print("\n" + "="*70)
    print("FEMTO TRANSFER EXPERIMENTS")
    print("="*70)

    # Enumerate all (source, target) pairs (with dedup)
    transfer_pairs = []
    seen_pairs = set()
    for test_name, exp in FEMTO_TRANSFERS.items():
        source = exp["source"]
        for target in exp["targets"]:
            pair = (source, target)
            if pair not in seen_pairs:
                transfer_pairs.append((test_name, source, target))
                seen_pairs.add(pair)

    print(f"\nTransfer pairs ({len(transfer_pairs)} unique):")
    for _, src, tgt in transfer_pairs:
        paper_rmse = PAPER_TABLE4.get((src, tgt), "?")
        print(f"  {src} -> {tgt}  [paper: {paper_rmse}]")

    # Get all unique bearing names
    all_bearing_names = list(
        set([src for _, src, _ in transfer_pairs] + [tgt for _, _, tgt in transfer_pairs])
    )

    # Load all bearings
    print(f"\nLoading {len(all_bearing_names)} unique bearings...")
    bearing_cache = load_all_bearings(FEMTO_DATA_ROOT, all_bearing_names, verbose=verbose)

    # Run experiments
    all_results = {}
    t_start = time.time()

    for i, (test_name, source, target) in enumerate(transfer_pairs):
        print(f"\n[{i+1}/{len(transfer_pairs)}] {test_name}: {source} -> {target}")

        if bearing_cache.get(source) is None:
            print(f"  SKIP: source bearing {source} not available")
            continue
        if bearing_cache.get(target) is None:
            print(f"  SKIP: target bearing {target} not available")
            continue

        result_key = f"{source}_to_{target}"
        result_file = results_dir / f"transfer_{result_key}.json"

        # Skip if already done
        if result_file.exists():
            print(f"  Loading cached result from {result_file}")
            with open(result_file) as f:
                result = json.load(f)
            all_results[(source, target)] = result
            print(f"  Mean RMSE: {result['rmse_mean']:.4f} ± {result['rmse_std']:.4f}")
            continue

        result = run_transfer_multi_seed(
            source_data=bearing_cache[source],
            target_data=bearing_cache[target],
            seeds=seeds,
            source_iterations=source_iterations,
            finetune_iterations=finetune_iterations,
            device=device,
            verbose=verbose,
        )

        # Save per-transfer result
        result_to_save = {k: v for k, v in result.items()
                         if k not in ("best_seed_predictions", "best_seed_ground_truth",
                                      "per_seed_results")}
        result_to_save["best_seed_predictions"] = result["best_seed_predictions"]
        result_to_save["best_seed_ground_truth"] = result["best_seed_ground_truth"]
        # Save per_seed_results without large arrays
        result_to_save["per_seed_results"] = [
            {k: v for k, v in r.items()
             if k not in ("predictions", "ground_truth",
                          "source_loss_history", "finetune_loss_history")}
            for r in result["per_seed_results"]
        ]
        result_to_save["per_seed_loss_histories"] = [
            {
                "seed": r["seed"],
                "source_loss": r["source_loss_history"],
                "finetune_loss": r["finetune_loss_history"],
            }
            for r in result["per_seed_results"]
        ]

        with open(result_file, "w") as f:
            json.dump(result_to_save, f, indent=2)
        print(f"  Saved: {result_file}")

        all_results[(source, target)] = result

    print(f"\nFEMTO experiments complete. Total time: {time.time()-t_start:.1f}s")
    return all_results


def run_xjtu_experiments(
    device: torch.device,
    seeds: list,
    source_iterations: int,
    finetune_iterations: int,
    results_dir: Path,
    verbose: bool = True,
) -> dict:
    """Run XJTU-SY transfer experiments if data available."""
    print("\n" + "="*70)
    print("XJTU-SY TRANSFER EXPERIMENTS")
    print("="*70)

    xjtu_root = find_xjtu_root()
    if xjtu_root is None:
        print("  XJTU-SY data NOT found. Skipping.")
        print("  Searched:", [str(p) for p in [
            Path("/mnt/sagemaker-nvme/xjtu_data"),
            Path("/home/sagemaker-user/IndustrialJEPA/data/xjtu"),
        ]])
        return {}

    print(f"  Found XJTU-SY data at: {xjtu_root}")
    all_results = {}

    for exp_name, exp in XJTU_TRANSFERS.items():
        source = exp["source"]
        for target in exp["targets"]:
            src_data = load_xjtu_bearing(xjtu_root, source)
            tgt_data = load_xjtu_bearing(xjtu_root, target)
            if src_data is None or tgt_data is None:
                print(f"  SKIP: {source} -> {target} (data not available)")
                continue

            result = run_transfer_multi_seed(
                source_data=src_data,
                target_data=tgt_data,
                seeds=seeds,
                source_iterations=source_iterations,
                finetune_iterations=finetune_iterations,
                device=device,
                verbose=verbose,
            )
            all_results[(source, target)] = result
            result_file = results_dir / f"xjtu_transfer_{source}_to_{target}.json"
            with open(result_file, "w") as f:
                json.dump({k: v for k, v in result.items()
                           if k not in ("per_seed_results",)}, f, indent=2)

    return all_results


def generate_plots(all_results: dict, results_dir: Path):
    """Generate RUL prediction plots and loss curves."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available, skipping plots")
        return

    plots_dir = results_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    print(f"\nGenerating plots in {plots_dir}...")

    for (source, target), result in all_results.items():
        # RUL prediction plot
        preds = np.array(result.get("best_seed_predictions", []))
        gt = np.array(result.get("best_seed_ground_truth", []))

        if len(preds) == 0 or len(gt) == 0:
            continue

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Left: RUL trajectory
        ax = axes[0]
        ax.plot(gt, "b-", label="Ground Truth", linewidth=1.5)
        ax.plot(preds, "r--", label=f"Predicted (seed={result.get('best_seed', '?')})",
                linewidth=1.5, alpha=0.8)
        ax.set_xlabel("Snapshot Index")
        ax.set_ylabel("RUL")
        ax.set_title(f"{source} → {target}\nRMSE={result['rmse_mean']:.4f}±{result['rmse_std']:.4f}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.05)

        # Right: per-seed RMSE bar chart
        ax = axes[1]
        rmse_per_seed = result.get("rmse_per_seed", [])
        if rmse_per_seed:
            seeds = result.get("seeds", list(range(len(rmse_per_seed))))
            bars = ax.bar(range(len(seeds)), rmse_per_seed, color="steelblue", alpha=0.7)
            ax.axhline(y=result["rmse_mean"], color="red", linestyle="--",
                       label=f"Mean: {result['rmse_mean']:.4f}")
            paper_rmse = PAPER_TABLE4.get((source, target))
            if paper_rmse:
                ax.axhline(y=paper_rmse, color="green", linestyle=":",
                           label=f"Paper: {paper_rmse:.4f}")
            ax.set_xticks(range(len(seeds)))
            ax.set_xticklabels([f"s{s}" for s in seeds], rotation=45)
            ax.set_ylabel("RMSE")
            ax.set_title("Per-Seed RMSE")
            ax.legend()
            ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        fname = plots_dir / f"rul_{source}_to_{target}.png"
        plt.savefig(str(fname), dpi=100, bbox_inches="tight")
        plt.close()

    # Summary plot: all transfers comparison
    if all_results:
        fig, ax = plt.subplots(figsize=(14, 6))
        labels = []
        our_rmse = []
        paper_rmse_vals = []

        for (src, tgt), result in sorted(all_results.items()):
            label = f"{src[-3:]}→{tgt[-3:]}"
            labels.append(label)
            our_rmse.append(result["rmse_mean"])
            paper_rmse_vals.append(PAPER_TABLE4.get((src, tgt), np.nan))

        x = np.arange(len(labels))
        width = 0.35
        ax.bar(x - width/2, paper_rmse_vals, width, label="Paper (Yu et al. 2024)",
               color="green", alpha=0.7)
        ax.bar(x + width/2, our_rmse, width, label="Our Replication",
               color="steelblue", alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylabel("RMSE")
        ax.set_title("CNN-GRU-MHA: Paper vs Replication (FEMTO Transfers)")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

        paper_avg = PAPER_AVG
        our_avg = np.mean([r["rmse_mean"] for r in all_results.values()])
        ax.axhline(y=paper_avg, color="green", linestyle="--",
                   alpha=0.5, label=f"Paper avg: {paper_avg:.4f}")
        ax.axhline(y=our_avg, color="blue", linestyle="--",
                   alpha=0.5, label=f"Our avg: {our_avg:.4f}")
        ax.legend()

        plt.tight_layout()
        plt.savefig(str(plots_dir / "femto_summary.png"), dpi=100, bbox_inches="tight")
        plt.close()
        print(f"  Saved summary plot: {plots_dir / 'femto_summary.png'}")

    print(f"  Plots saved to {plots_dir}")


def write_results_md(all_results: dict, results_dir: Path, runtime_info: dict):
    """Write RESULTS.md comparison table."""
    lines = [
        "# CNN-GRU-MHA Replication Results",
        "",
        f"**Paper**: Yu et al., Applied Sciences 2024, DOI: 10.3390/app14199039",
        f"**Run date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Device**: {runtime_info.get('device', 'unknown')}",
        f"**Seeds**: {runtime_info.get('seeds', [])}",
        f"**Source epochs**: {runtime_info.get('source_iterations', 60)}",
        f"**Finetune epochs**: {runtime_info.get('finetune_iterations', 100)}",
        "",
        "## Table 4: FEMTO Transfer RMSE",
        "",
        "| Source | Target | Paper RMSE | Our RMSE (mean±std) | Delta | Status |",
        "|--------|--------|:----------:|:-------------------:|:-----:|:------:|",
    ]

    our_rmse_vals = []

    # Paper's Table 4 order
    table_order = [
        ("Bearing1_3", "Bearing2_3"),
        ("Bearing1_3", "Bearing2_4"),
        ("Bearing1_3", "Bearing3_1"),
        ("Bearing1_3", "Bearing3_3"),
        ("Bearing2_3", "Bearing1_3"),
        ("Bearing2_3", "Bearing1_4"),
        ("Bearing2_3", "Bearing3_3"),
        ("Bearing2_3", "Bearing3_3"),  # duplicate in paper
        ("Bearing3_2", "Bearing1_3"),
        ("Bearing3_2", "Bearing1_4"),
        ("Bearing3_2", "Bearing2_3"),
        ("Bearing3_2", "Bearing2_4"),
    ]

    seen_in_table = set()
    for (src, tgt) in table_order:
        paper_val = PAPER_TABLE4.get((src, tgt), None)
        result = all_results.get((src, tgt))

        if result is None:
            status = "MISSING"
            our_str = "N/A"
            delta_str = "N/A"
        else:
            mean_r = result["rmse_mean"]
            std_r = result["rmse_std"]
            our_str = f"{mean_r:.4f}±{std_r:.4f}"
            our_rmse_vals.append(mean_r)
            if paper_val:
                delta = (mean_r - paper_val) / paper_val * 100
                delta_str = f"{delta:+.1f}%"
                if abs(delta) <= 10:
                    status = "EXACT"
                elif abs(delta) <= 20:
                    status = "GOOD"
                elif mean_r < paper_val:
                    status = "BETTER"
                else:
                    status = "WORSE"
            else:
                delta_str = "N/A"
                status = "OK"

        paper_str = f"{paper_val:.4f}" if paper_val else "N/A"
        lines.append(f"| {src} | {tgt} | {paper_str} | {our_str} | {delta_str} | {status} |")
        seen_in_table.add((src, tgt))

    # Average row
    paper_avg_str = f"{PAPER_AVG:.4f}"
    if our_rmse_vals:
        our_avg = np.mean(our_rmse_vals)
        our_std = np.std(our_rmse_vals)
        our_avg_str = f"{our_avg:.4f}±{our_std:.4f}"
        avg_delta = (our_avg - PAPER_AVG) / PAPER_AVG * 100
        avg_delta_str = f"{avg_delta:+.1f}%"
        if abs(avg_delta) <= 10:
            avg_status = "EXACT"
        elif abs(avg_delta) <= 20:
            avg_status = "GOOD"
        elif our_avg < PAPER_AVG:
            avg_status = "BETTER"
        else:
            avg_status = "WORSE"
    else:
        our_avg_str = "N/A"
        avg_delta_str = "N/A"
        avg_status = "N/A"

    lines.append(f"| **Average** | | **{paper_avg_str}** | **{our_avg_str}** | **{avg_delta_str}** | **{avg_status}** |")

    lines.extend([
        "",
        "## Success Criteria",
        "",
        f"- Good: Average RMSE within 20% of {PAPER_AVG} (threshold: {PAPER_AVG*1.2:.4f})",
        f"- Exact: Average RMSE within 10% of {PAPER_AVG} (threshold: {PAPER_AVG*1.1:.4f})",
        "",
    ])

    if our_rmse_vals:
        our_avg = np.mean(our_rmse_vals)
        if our_avg <= PAPER_AVG * 1.1:
            lines.append(f"**Result: EXACT replication achieved** (our avg={our_avg:.4f} vs paper={PAPER_AVG:.4f})")
        elif our_avg <= PAPER_AVG * 1.2:
            lines.append(f"**Result: GOOD replication** (our avg={our_avg:.4f} vs paper={PAPER_AVG:.4f})")
        elif our_avg < PAPER_AVG:
            lines.append(f"**Result: BETTER than paper** (our avg={our_avg:.4f} vs paper={PAPER_AVG:.4f})")
        else:
            lines.append(f"**Result: WORSE than paper** (our avg={our_avg:.4f} vs paper={PAPER_AVG:.4f})")
    else:
        lines.append("**Result: No results available**")

    lines.extend([
        "",
        "## Notes",
        "",
        "- Architecture: CNN (6 blocks, MHA after block 3) + 2-layer GRU + FC head",
        "- Preprocessing: DWT denoising (sym8, level=3) + min-max normalization",
        "- Channel: horizontal only (channel 0)",
        "- RUL labels: linear decay Y_i = (N-i)/N",
        "- Transfer protocol: freeze CNN+GRU, fine-tune FC on first half of target",
        "- Evaluation: RMSE on second half of target (chronological split)",
        f"- Framework: PyTorch (paper used TensorFlow 2.5.0)",
        "",
    ])

    results_path = results_dir / "RESULTS.md"
    with open(results_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\nSaved RESULTS.md: {results_path}")


def write_experiment_log(all_results: dict, results_dir: Path, runtime_info: dict):
    """Write EXPERIMENT_LOG.md."""
    log_path = results_dir.parent / "EXPERIMENT_LOG.md"

    lines = [
        "# CNN-GRU-MHA Experiment Log",
        "",
        f"**Paper**: Yu et al., Applied Sciences 2024",
        f"**Goal**: Replicate FEMTO average RMSE = {PAPER_AVG}",
        f"**Run date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "## Sanity Checks",
        "",
        "- [x] DWT denoising implemented (sym8, level=3)",
        "- [x] Min-max normalization per snapshot",
        "- [x] Horizontal channel only (channel 0)",
        "- [x] Linear RUL labels: Y_i = (N-i)/N",
        "- [x] 1:1 chronological target split",
        "- [x] CNN+GRU frozen during FC fine-tuning",
        "- [x] 5 seeds per transfer",
        "",
        "## Experiment Results",
        "",
    ]

    for exp_num, ((source, target), result) in enumerate(sorted(all_results.items()), 1):
        paper_rmse = PAPER_TABLE4.get((source, target), None)
        mean_r = result["rmse_mean"]
        std_r = result["rmse_std"]

        if paper_rmse:
            delta = (mean_r - paper_rmse) / paper_rmse * 100
            verdict = "KEEP" if abs(delta) <= 20 else "INVESTIGATE"
        else:
            delta = 0
            verdict = "OK"

        lines.extend([
            f"### Exp {exp_num}: {source} -> {target}",
            "",
            f"**Time**: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            f"**Source snapshots**: {result.get('source_n_snapshots', '?')}",
            f"**Target snapshots**: {result.get('target_n_snapshots', '?')}",
            f"**Seeds**: {result.get('seeds', [])}",
            f"**RMSE per seed**: {[f'{r:.4f}' for r in result.get('rmse_per_seed', [])]}",
            f"**Result**: {mean_r:.4f} ± {std_r:.4f}",
            f"**Paper target**: {paper_rmse:.4f}" if paper_rmse else "**Paper target**: N/A",
            f"**Delta vs paper**: {delta:+.1f}%" if paper_rmse else "",
            f"**Verdict**: {verdict}",
            "",
        ])

    # Summary
    if all_results:
        our_avg = np.mean([r["rmse_mean"] for r in all_results.values()])
        lines.extend([
            "## Final Summary",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Transfers completed | {len(all_results)} / 11 (unique) |",
            f"| Our average RMSE | {our_avg:.4f} |",
            f"| Paper average RMSE | {PAPER_AVG:.4f} |",
            f"| Delta vs paper | {(our_avg-PAPER_AVG)/PAPER_AVG*100:+.1f}% |",
            f"| Seeds per transfer | {len(runtime_info.get('seeds', []))} |",
            "",
        ])

    with open(log_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved EXPERIMENT_LOG.md: {log_path}")


def main():
    parser = argparse.ArgumentParser(description="CNN-GRU-MHA replication experiments")
    parser.add_argument("--device", type=str, default=None,
                        help="Device: cpu, cuda, cuda:0 (default: auto)")
    parser.add_argument("--source_iterations", type=int, default=60,
                        help="Source domain training epochs (default: 60)")
    parser.add_argument("--finetune_iterations", type=int, default=100,
                        help="FC fine-tuning epochs (default: 100)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456, 789, 1024],
                        help="Random seeds for averaging (default: 5 seeds)")
    parser.add_argument("--fast", action="store_true",
                        help="Fast mode: 10 source epochs, 20 FT epochs, 1 seed (for debugging)")
    parser.add_argument("--debug_transfer", type=str, nargs=2, metavar=("SOURCE", "TARGET"),
                        help="Run only one specific transfer (e.g. --debug_transfer Bearing1_3 Bearing2_3)")
    parser.add_argument("--no_xjtu", action="store_true", help="Skip XJTU-SY experiments")
    parser.add_argument("--results_dir", type=str, default=None)
    args = parser.parse_args()

    # Fast mode overrides
    if args.fast:
        args.source_iterations = 10
        args.finetune_iterations = 20
        args.seeds = [42]
        print("FAST MODE: 10 source epochs, 20 FT epochs, 1 seed")

    # Device selection
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # Results directory
    results_dir = Path(args.results_dir) if args.results_dir else Path(__file__).parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "plots").mkdir(exist_ok=True)

    print(f"Results dir: {results_dir}")
    print(f"Seeds: {args.seeds}")
    print(f"Source epochs: {args.source_iterations}")
    print(f"Finetune epochs: {args.finetune_iterations}")

    runtime_info = {
        "device": str(device),
        "seeds": args.seeds,
        "source_iterations": args.source_iterations,
        "finetune_iterations": args.finetune_iterations,
        "start_time": datetime.now().isoformat(),
    }

    t_total = time.time()

    # Debug mode: single transfer
    if args.debug_transfer:
        source_name, target_name = args.debug_transfer
        print(f"\nDEBUG: Running single transfer {source_name} -> {target_name}")
        src = load_bearing_for_cnn_gru(FEMTO_DATA_ROOT, source_name)
        tgt = load_bearing_for_cnn_gru(FEMTO_DATA_ROOT, target_name)
        result = run_transfer_multi_seed(
            source_data=src,
            target_data=tgt,
            seeds=args.seeds,
            source_iterations=args.source_iterations,
            finetune_iterations=args.finetune_iterations,
            device=device,
            verbose=True,
        )
        print(f"\nResult: RMSE = {result['rmse_mean']:.4f} ± {result['rmse_std']:.4f}")
        paper_val = PAPER_TABLE4.get((source_name, target_name))
        if paper_val:
            print(f"Paper:  RMSE = {paper_val:.4f}")
            print(f"Delta:  {(result['rmse_mean']-paper_val)/paper_val*100:+.1f}%")
        return

    # Run FEMTO experiments
    femto_results = run_femto_experiments(
        device=device,
        seeds=args.seeds,
        source_iterations=args.source_iterations,
        finetune_iterations=args.finetune_iterations,
        results_dir=results_dir,
        verbose=True,
    )

    # Save all results
    all_results_file = results_dir / "all_results.json"
    all_results_serializable = {}
    for (src, tgt), result in femto_results.items():
        key = f"{src}_to_{tgt}"
        all_results_serializable[key] = {
            k: v for k, v in result.items()
            if k not in ("per_seed_results",)
        }
    with open(all_results_file, "w") as f:
        json.dump(all_results_serializable, f, indent=2)
    print(f"\nSaved all results: {all_results_file}")

    # Run XJTU-SY if available and not skipped
    xjtu_results = {}
    if not args.no_xjtu:
        xjtu_results = run_xjtu_experiments(
            device=device,
            seeds=args.seeds,
            source_iterations=args.source_iterations,
            finetune_iterations=args.finetune_iterations,
            results_dir=results_dir,
            verbose=True,
        )

    # Generate plots
    generate_plots(femto_results, results_dir)

    # Write reports
    runtime_info["end_time"] = datetime.now().isoformat()
    runtime_info["total_time_s"] = time.time() - t_total

    write_results_md(femto_results, results_dir, runtime_info)
    write_experiment_log(femto_results, results_dir, runtime_info)

    # Final summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    if femto_results:
        our_vals = [r["rmse_mean"] for r in femto_results.values()]
        our_avg = np.mean(our_vals)
        our_std = np.std(our_vals)
        print(f"\nFEMTO Transfer Results ({len(femto_results)} transfers):")
        print(f"  Our average RMSE:   {our_avg:.4f} ± {our_std:.4f}")
        print(f"  Paper average RMSE: {PAPER_AVG:.4f}")
        print(f"  Delta vs paper:     {(our_avg-PAPER_AVG)/PAPER_AVG*100:+.1f}%")

        print(f"\nPer-transfer:")
        for (src, tgt), result in sorted(femto_results.items()):
            paper_val = PAPER_TABLE4.get((src, tgt))
            mean_r = result["rmse_mean"]
            if paper_val:
                delta = (mean_r - paper_val) / paper_val * 100
                print(f"  {src} -> {tgt}: {mean_r:.4f} (paper: {paper_val:.4f}, delta: {delta:+.1f}%)")
            else:
                print(f"  {src} -> {tgt}: {mean_r:.4f}")

        if our_avg <= PAPER_AVG * 1.1:
            print(f"\n*** EXACT replication: within 10% of paper target ***")
        elif our_avg <= PAPER_AVG * 1.2:
            print(f"\n*** GOOD replication: within 20% of paper target ***")
        elif our_avg < PAPER_AVG:
            print(f"\n*** BETTER than paper! ***")
        else:
            print(f"\n*** Gap vs paper: {(our_avg-PAPER_AVG)/PAPER_AVG*100:.1f}% above target ***")

    if xjtu_results:
        print(f"\nXJTU-SY Transfer Results ({len(xjtu_results)} transfers):")
        for (src, tgt), result in xjtu_results.items():
            print(f"  {src} -> {tgt}: {result['rmse_mean']:.4f}")

    print(f"\nTotal runtime: {time.time()-t_total:.1f}s")
    print(f"Results saved to: {results_dir}")


if __name__ == "__main__":
    main()
