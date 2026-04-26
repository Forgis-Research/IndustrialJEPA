"""
Compute PA-F1 (and non-PA F1) from stored probability surfaces.

Run on the GPU VM where the .npz surfaces live:
    cd IndustrialJEPA/fam-jepa
    python experiments/v22/compute_pa_f1_from_surfaces.py

Outputs a JSON with per-seed and aggregate PA-F1 for each dataset.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

# --- paths ---------------------------------------------------------------
ROOT = Path(__file__).resolve().parent          # experiments/v22/
SURF_DIR = ROOT / "surfaces"
OUT_PATH = ROOT / "pa_f1_from_surfaces.json"

sys.path.insert(0, str(ROOT.parent.parent))     # fam-jepa/
from evaluation.grey_swan_metrics import anomaly_metrics  # noqa: E402

# --- config --------------------------------------------------------------
DATASETS = ["smap", "msl", "psm", "smd", "mba"]
SEEDS = [42, 123, 456]
# Threshold sweep: best F1 over percentiles (on normal points)
PERCENTILES = [90, 92, 94, 95, 96, 98, 99]


def load_surface(path: Path) -> dict:
    d = np.load(path, allow_pickle=True)
    return {
        "p_surface": d["p_surface"],   # (N, K)
        "y_surface": d["y_surface"],   # (N, K)
    }


def surface_to_anomaly_score(p_surface: np.ndarray) -> np.ndarray:
    """Collapse (N, K) surface to (N,) anomaly score: max prob over horizons."""
    return p_surface.max(axis=1)


def surface_to_label(y_surface: np.ndarray) -> np.ndarray:
    """Collapse (N, K) labels to (N,): anomaly if any horizon is positive."""
    return (y_surface.max(axis=1) > 0).astype(int)


def best_pa_f1(scores: np.ndarray, y_true: np.ndarray) -> dict:
    """Sweep threshold percentiles, return metrics at best PA-F1 threshold."""
    best = None
    for pct in PERCENTILES:
        m = anomaly_metrics(scores, y_true, threshold_percentile=pct)
        m["threshold_percentile"] = pct
        if best is None or m["f1_pa"] > best["f1_pa"]:
            best = m
    return best


def main():
    results = {}

    for ds in DATASETS:
        seed_results = []
        for seed in SEEDS:
            surf_path = SURF_DIR / f"{ds}_pred_ft_seed{seed}.npz"
            if not surf_path.exists():
                print(f"  SKIP {surf_path} (not found)")
                continue

            surf = load_surface(surf_path)
            scores = surface_to_anomaly_score(surf["p_surface"])
            y_true = surface_to_label(surf["y_surface"])

            m = best_pa_f1(scores, y_true)
            seed_results.append({
                "seed": seed,
                "f1_pa": m["f1_pa"],
                "f1_non_pa": m["f1_non_pa"],
                "precision_pa": m["precision_pa"],
                "recall_pa": m["recall_pa"],
                "precision_non_pa": m["precision_non_pa"],
                "recall_non_pa": m["recall_non_pa"],
                "auroc": m["auroc"],
                "auc_pr": m["auc_pr"],
                "threshold_pct": m["threshold_percentile"],
                "threshold_val": m["threshold_used"],
                "prevalence": m["prevalence"],
            })
            print(f"  {ds} seed={seed}: PA-F1={m['f1_pa']:.3f}  "
                  f"non-PA-F1={m['f1_non_pa']:.3f}  "
                  f"(pct={m['threshold_percentile']})")

        if seed_results:
            pa_f1s = [r["f1_pa"] for r in seed_results]
            non_pa_f1s = [r["f1_non_pa"] for r in seed_results]
            results[ds.upper()] = {
                "per_seed": seed_results,
                "agg": {
                    "f1_pa_mean": float(np.mean(pa_f1s)),
                    "f1_pa_std": float(np.std(pa_f1s)),
                    "f1_non_pa_mean": float(np.mean(non_pa_f1s)),
                    "f1_non_pa_std": float(np.std(non_pa_f1s)),
                    "n_seeds": len(seed_results),
                },
            }
            print(f"  {ds.upper()} aggregate: "
                  f"PA-F1 = {np.mean(pa_f1s):.3f}±{np.std(pa_f1s):.3f}, "
                  f"non-PA-F1 = {np.mean(non_pa_f1s):.3f}±{np.std(non_pa_f1s):.3f}")

    OUT_PATH.write_text(json.dumps(results, indent=2))
    print(f"\nSaved to {OUT_PATH}")


if __name__ == "__main__":
    main()
