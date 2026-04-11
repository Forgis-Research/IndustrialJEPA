"""
Generate final RESULTS.md from completed JSON result files.
Run this after all experiments are done.
"""

import json
import sys
from pathlib import Path
from datetime import datetime

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_MD = Path(__file__).parent / "RESULTS.md"
EXPERIMENT_LOG = Path(__file__).parent / "EXPERIMENT_LOG.md"

PAPER_TARGETS = {
    "FD001": dict(rmse=10.61, score=169),
    "FD002": dict(rmse=13.47, score=784),
    "FD003": dict(rmse=10.71, score=202),
    "FD004": dict(rmse=15.87, score=1449),
}

SOTA_CONTEXT = {
    "FD001": "STAR is claimed SOTA. Prior SOTA: CNN-GRU-MHA ~11.44 RMSE.",
    "FD002": "STAR is claimed SOTA. Prior SOTA: DLformer ~13.80 RMSE.",
    "FD003": "STAR is claimed SOTA. Prior SOTA: DLformer ~11.67 RMSE.",
    "FD004": "STAR is claimed SOTA. Prior SOTA: DLformer ~15.86 RMSE.",
}


def assessment_label(gap_pct: float) -> str:
    if abs(gap_pct) <= 10:
        return "EXACT"
    elif abs(gap_pct) <= 20:
        return "GOOD"
    elif abs(gap_pct) <= 30:
        return "MARGINAL"
    else:
        return "FAILED"


def load_ablation_results():
    ablation_dir = RESULTS_DIR / "ablations"
    results = {}
    for f in ablation_dir.glob("*.json"):
        with open(f) as fp:
            results[f.stem] = json.load(fp)
    return results


def generate_results_md():
    lines = [
        "# STAR Replication Results",
        "",
        "Paper: Fan et al., \"A Two-Stage Attention-Based Hierarchical Transformer for Turbofan Engine",
        "Remaining Useful Life Prediction\", Sensors 2024.",
        "",
        f"Replication completed: {datetime.now().strftime('%Y-%m-%d')}",
        "Hardware: NVIDIA A10G (22 GB), CUDA, PyTorch 2.6.0",
        "Seeds: [42, 123, 456, 789, 1024]",
        "",
        "---",
        "",
        "## Main Results Table",
        "",
        "| Subset | Paper RMSE | Ours RMSE (mean +/- std) | Paper Score | Ours Score (mean +/- std) | RMSE Gap | Assessment |",
        "|--------|-----------|--------------------------|-------------|---------------------------|----------|------------|",
    ]

    assessments = {}
    available_subsets = []
    for subset in ["FD001", "FD002", "FD003", "FD004"]:
        json_path = RESULTS_DIR / f"{subset}_results.json"
        if not json_path.exists():
            lines.append(f"| {subset} | {PAPER_TARGETS[subset]['rmse']} | PENDING | {PAPER_TARGETS[subset]['score']} | PENDING | - | - |")
            continue

        with open(json_path) as f:
            res = json.load(f)

        available_subsets.append(subset)
        gap_pct = res.get("rmse_gap_pct", 100.0 * (res["rmse_mean"] - PAPER_TARGETS[subset]["rmse"]) / PAPER_TARGETS[subset]["rmse"])
        label = assessment_label(gap_pct)
        assessments[subset] = label

        lines.append(
            f"| {subset} | {PAPER_TARGETS[subset]['rmse']} | {res['rmse_mean']:.2f} +/- {res['rmse_std']:.2f} "
            f"| {PAPER_TARGETS[subset]['score']} | {res['score_mean']:.0f} +/- {res['score_std']:.0f} "
            f"| {gap_pct:+.1f}% | {label} |"
        )

    lines += ["", "Assessment: EXACT (<=10% gap), GOOD (<=20%), MARGINAL (<=30%), FAILED (>30%)", ""]

    # Per-seed breakdown
    lines += ["", "## Per-Seed Breakdown", ""]
    for subset in available_subsets:
        json_path = RESULTS_DIR / f"{subset}_results.json"
        with open(json_path) as f:
            res = json.load(f)
        lines.append(f"### {subset}")
        lines.append("")
        lines.append("| Seed | RMSE | PHM Score | Best Val RMSE | Best Epoch | Epochs Run | Time (s) |")
        lines.append("|------|------|-----------|---------------|------------|------------|----------|")
        for sr in res["per_seed"]:
            lines.append(f"| {sr['seed']} | {sr['test_rmse']:.3f} | {sr['test_score']:.0f} | "
                        f"{sr['best_val_rmse']:.3f} | {sr['best_epoch']} | {sr['epochs_run']} | {sr['wall_time_s']:.0f} |")
        lines.append(f"| **Mean** | **{res['rmse_mean']:.3f}** | **{res['score_mean']:.0f}** | - | - | - | - |")
        lines.append(f"| **Std** | **{res['rmse_std']:.3f}** | **{res['score_std']:.0f}** | - | - | - | - |")
        lines.append("")

    # Ablations
    ablation_dir = RESULTS_DIR / "ablations"
    if ablation_dir.exists():
        ablation_files = list(ablation_dir.glob("*.json"))
        if ablation_files:
            lines += ["---", "", "## Ablations", ""]

            # Condition normalization
            cond_path = ablation_dir / "cond_norm_results.json"
            if cond_path.exists():
                with open(cond_path) as f:
                    cond = json.load(f)
                lines += ["### Ablation 1: Per-Condition Normalization (FD002, FD004)", ""]
                lines.append("| Subset | RMSE (mean +/- std) | Score (mean +/- std) | vs Paper RMSE |")
                lines.append("|--------|---------------------|----------------------|---------------|")
                for subset in ["FD002", "FD004"]:
                    if subset in cond:
                        r = cond[subset]
                        gap = 100 * (r["rmse_mean"] - PAPER_TARGETS[subset]["rmse"]) / PAPER_TARGETS[subset]["rmse"]
                        lines.append(f"| {subset} (cond-norm) | {r['rmse_mean']:.3f} +/- {r['rmse_std']:.3f} | "
                                    f"{r['score_mean']:.0f} +/- {r['score_std']:.0f} | {gap:+.1f}% |")
                lines.append("")

            # RUL cap sweep
            cap_path = ablation_dir / "rul_cap_results.json"
            if cap_path.exists():
                with open(cap_path) as f:
                    cap = json.load(f)
                lines += ["### Ablation 2: RUL Cap Sweep (FD001)", ""]
                lines.append("| RUL Cap | RMSE (mean +/- std) | Score (mean +/- std) |")
                lines.append("|---------|---------------------|----------------------|")
                for c in ["100", "110", "125", "140"]:
                    if c in cap:
                        r = cap[c]
                        lines.append(f"| {c} | {r['rmse_mean']:.3f} +/- {r['rmse_std']:.3f} | "
                                    f"{r['score_mean']:.0f} +/- {r['score_std']:.0f} |")
                lines.append("")

            # Patch length sweep
            pl_path = ablation_dir / "patch_length_results.json"
            if pl_path.exists():
                with open(pl_path) as f:
                    pl = json.load(f)
                lines += ["### Ablation 3: Patch Length Sweep (FD001)", ""]
                lines.append("| Patch Length | RMSE (mean +/- std) | Score (mean +/- std) | Params |")
                lines.append("|-------------|---------------------|----------------------|--------|")
                for L in ["2", "4", "8"]:
                    if L in pl:
                        r = pl[L]
                        lines.append(f"| L={L} | {r['rmse_mean']:.3f} +/- {r['rmse_std']:.3f} | "
                                    f"{r['score_mean']:.0f} +/- {r['score_std']:.0f} | "
                                    f"{r.get('n_params', 'N/A'):,} |")
                lines.append("")

            # n_heads sweep
            nh_path = ablation_dir / "nheads_results.json"
            if nh_path.exists():
                with open(nh_path) as f:
                    nh = json.load(f)
                lines += ["### Ablation 4: n_heads Sweep (FD001, FD003)", ""]
                lines.append("| Subset | n_heads | RMSE (mean +/- std) | Score (mean +/- std) |")
                lines.append("|--------|---------|---------------------|----------------------|")
                for subset in ["FD001", "FD003"]:
                    for h in [1, 2, 4]:
                        key = f"{subset}_nh{h}"
                        if key in nh:
                            r = nh[key]
                            lines.append(f"| {subset} | {h} | {r['rmse_mean']:.3f} +/- {r['rmse_std']:.3f} | "
                                        f"{r['score_mean']:.0f} +/- {r['score_std']:.0f} |")
                lines.append("")

    # Honest assessment
    lines += [
        "---",
        "",
        "## Honest Assessment",
        "",
    ]

    for subset in available_subsets:
        label = assessments.get(subset, "PENDING")
        json_path = RESULTS_DIR / f"{subset}_results.json"
        with open(json_path) as f:
            res = json.load(f)
        gap_pct = res.get("rmse_gap_pct", 0.0)
        lines.append(f"**{subset}**: {label} ({gap_pct:+.1f}% RMSE gap)")
        lines.append("")
        lines.append(f"Context: {SOTA_CONTEXT.get(subset, '')}")
        lines.append("")

    lines += [
        "---",
        "",
        "## Deviations from Paper",
        "",
        "1. Per-scale prediction head uses mean pooling over K and D dims before MLP. The paper",
        "   is not explicit about the exact operation; full flatten would create extremely large",
        "   parameter counts (292M for FD004 at d_model=256).",
        "2. PatchMerging with odd K: last patch is truncated (not padded). Paper not explicit.",
        "3. Sinusoidal PE for first decoder input: Vaswani-style sin/cos, not learned.",
        "4. Train/val split: 15% of training engines held out for validation.",
        "5. Early stopping on val RMSE with patience=20 (paper does not specify early stopping).",
        "",
    ]

    content = "\n".join(lines)
    with open(RESULTS_MD, "w") as f:
        f.write(content)
    print(f"RESULTS.md written to {RESULTS_MD}")
    return content


if __name__ == "__main__":
    content = generate_results_md()
    print("\nPreview (first 60 lines):")
    for line in content.split("\n")[:60]:
        print(line)
