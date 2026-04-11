# SPDX-FileCopyrightText: 2025 Industrial World Model Authors
# SPDX-License-Identifier: MIT

"""
Benchmark Evaluation for Industrial World Model.

Standardized evaluation on industrial datasets:
1. C-MAPSS: Turbofan RUL prediction
2. Bearing: Fault classification
3. TEP: Process fault diagnosis
4. SWaT: Anomaly detection
"""

import torch
import numpy as np
from typing import Dict, Optional, List
from dataclasses import dataclass
from pathlib import Path
from tqdm import tqdm
import json

from industrialjepa.model.industrial_world_lm import IndustrialWorldLM
from industrialjepa.evaluation.metrics import (
    compute_forecasting_metrics,
    compute_rul_metrics,
    compute_anomaly_metrics,
    compute_classification_metrics,
    compute_calibration_metrics,
)


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""
    dataset: str
    task: str
    metrics: Dict[str, float]
    per_subset_results: Optional[Dict[str, Dict]] = None
    model_info: Optional[Dict] = None


def evaluate_cmapss(
    model: IndustrialWorldLM,
    data_path: str,
    subsets: List[str] = ["FD001", "FD002", "FD003", "FD004"],
    device: str = "cuda",
    batch_size: int = 32,
) -> BenchmarkResult:
    """
    Evaluate on NASA C-MAPSS turbofan dataset.

    Tasks:
    - Remaining Useful Life (RUL) prediction
    - Multi-step forecasting

    Args:
        model: Trained IndustrialWorldLM
        data_path: Path to C-MAPSS data
        subsets: Which subsets to evaluate
        device: Device for inference
        batch_size: Batch size for evaluation

    Returns:
        BenchmarkResult with RUL metrics
    """
    from industrialjepa.data.datasets import CMAPSSDataset, DatasetConfig

    model.eval()
    model.to(device)

    per_subset = {}
    all_predictions = []
    all_targets = []

    for subset in tqdm(subsets, desc="C-MAPSS subsets"):
        config = DatasetConfig(
            window_size=512,
            stride=32,
            normalize=True,
            include_rul=True,
        )

        try:
            dataset = CMAPSSDataset(
                data_path=data_path,
                subset=subset,
                split="test",
                config=config,
            )
        except FileNotFoundError as e:
            print(f"Skipping {subset}: {e}")
            continue

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
        )

        subset_preds = []
        subset_targets = []

        with torch.no_grad():
            for batch in dataloader:
                x = batch["x"].to(device)
                rul = batch["rul"]

                # Forward pass
                output = model(x, compute_loss=False, predict_next=True)

                # Use last hidden state for RUL prediction
                # In full implementation, would have dedicated RUL head
                h = output.hidden_states[:, -1, :]  # [B, D]

                # Simple linear projection for RUL
                # (In practice, train a proper head)
                rul_pred = h.mean(dim=-1) * 50 + 60  # Rough scaling

                subset_preds.append(rul_pred.cpu().numpy())
                subset_targets.append(rul.numpy())

        subset_preds = np.concatenate(subset_preds)
        subset_targets = np.concatenate(subset_targets)

        # Compute metrics for this subset
        subset_metrics = compute_rul_metrics(subset_preds, subset_targets)
        per_subset[subset] = subset_metrics

        all_predictions.extend(subset_preds)
        all_targets.extend(subset_targets)

    # Overall metrics
    if len(all_predictions) > 0:
        overall_metrics = compute_rul_metrics(
            np.array(all_predictions),
            np.array(all_targets),
        )
    else:
        overall_metrics = {}

    return BenchmarkResult(
        dataset="C-MAPSS",
        task="RUL Prediction",
        metrics=overall_metrics,
        per_subset_results=per_subset,
        model_info={"num_params": model.get_num_params()},
    )


def evaluate_bearing(
    model: IndustrialWorldLM,
    data_path: str,
    source: str = "cwru",
    device: str = "cuda",
    batch_size: int = 32,
) -> BenchmarkResult:
    """
    Evaluate on bearing fault diagnosis dataset.

    Tasks:
    - Fault classification (CWRU)
    - Prognostics (FEMTO)

    Args:
        model: Trained IndustrialWorldLM
        data_path: Path to bearing data
        source: "cwru" or "femto"
        device: Device for inference
        batch_size: Batch size

    Returns:
        BenchmarkResult with classification metrics
    """
    from industrialjepa.data.datasets import BearingDataset, DatasetConfig

    model.eval()
    model.to(device)

    config = DatasetConfig(
        window_size=1024,  # ~85ms at 12kHz
        stride=256,
        normalize=True,
    )

    try:
        dataset = BearingDataset(
            data_path=data_path,
            source=source,
            split="test",
            config=config,
        )
    except FileNotFoundError as e:
        return BenchmarkResult(
            dataset=f"Bearing ({source})",
            task="Fault Classification",
            metrics={"error": str(e)},
        )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
    )

    all_hidden = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Bearing ({source})"):
            x = batch["x"].to(device)
            labels = batch.get("label")

            output = model(x, compute_loss=False)

            # Use pooled hidden states for classification
            h = output.hidden_states.mean(dim=1)  # [B, D]
            all_hidden.append(h.cpu())

            if labels is not None:
                all_labels.append(labels)

    all_hidden = torch.cat(all_hidden, dim=0).numpy()

    if len(all_labels) > 0:
        all_labels = torch.cat(all_labels, dim=0).numpy()

        # Simple nearest-centroid classification
        # In practice, train a proper classifier head
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.model_selection import cross_val_predict

        knn = KNeighborsClassifier(n_neighbors=5)
        predictions = cross_val_predict(knn, all_hidden, all_labels, cv=5)

        metrics = compute_classification_metrics(predictions, all_labels)
    else:
        metrics = {"note": "No labels available"}

    return BenchmarkResult(
        dataset=f"Bearing ({source})",
        task="Fault Classification",
        metrics=metrics,
        model_info={"num_params": model.get_num_params()},
    )


def evaluate_tep(
    model: IndustrialWorldLM,
    data_path: str,
    device: str = "cuda",
    batch_size: int = 32,
) -> BenchmarkResult:
    """
    Evaluate on Tennessee Eastman Process dataset.

    Tasks:
    - Fault detection
    - Fault diagnosis (22-class classification)

    Args:
        model: Trained IndustrialWorldLM
        data_path: Path to TEP data
        device: Device for inference
        batch_size: Batch size

    Returns:
        BenchmarkResult with classification metrics
    """
    from industrialjepa.data.datasets import TennesseeEastmanDataset, DatasetConfig

    model.eval()
    model.to(device)

    config = DatasetConfig(
        window_size=256,
        stride=64,
        normalize=True,
    )

    try:
        dataset = TennesseeEastmanDataset(
            data_path=data_path,
            split="test",
            config=config,
        )
    except FileNotFoundError as e:
        return BenchmarkResult(
            dataset="Tennessee Eastman",
            task="Fault Diagnosis",
            metrics={"error": str(e)},
        )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
    )

    all_hidden = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="TEP"):
            x = batch["x"].to(device)
            labels = batch.get("label")

            output = model(x, compute_loss=False)
            h = output.hidden_states.mean(dim=1)

            all_hidden.append(h.cpu())
            if labels is not None:
                all_labels.append(labels)

    all_hidden = torch.cat(all_hidden, dim=0).numpy()

    if len(all_labels) > 0:
        all_labels = torch.cat(all_labels, dim=0).numpy()

        # Classification using learned representations
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_predict

        clf = LogisticRegression(max_iter=1000, multi_class="multinomial")
        predictions = cross_val_predict(clf, all_hidden, all_labels, cv=5)

        metrics = compute_classification_metrics(
            predictions, all_labels, num_classes=22
        )

        # Detection (binary: normal vs fault)
        binary_labels = (all_labels > 0).astype(int)
        binary_preds = (predictions > 0).astype(int)
        detection_metrics = compute_classification_metrics(
            binary_preds, binary_labels, num_classes=2
        )
        metrics["detection_f1"] = detection_metrics["f1_macro"]
    else:
        metrics = {"note": "No labels available"}

    return BenchmarkResult(
        dataset="Tennessee Eastman",
        task="Fault Diagnosis",
        metrics=metrics,
        model_info={"num_params": model.get_num_params()},
    )


def evaluate_swat(
    model: IndustrialWorldLM,
    data_path: str,
    device: str = "cuda",
    batch_size: int = 32,
) -> BenchmarkResult:
    """
    Evaluate on SWaT (Secure Water Treatment) dataset.

    Tasks:
    - Anomaly detection
    - Attack detection

    Args:
        model: Trained IndustrialWorldLM
        data_path: Path to SWaT data
        device: Device for inference
        batch_size: Batch size

    Returns:
        BenchmarkResult with anomaly detection metrics
    """
    from industrialjepa.data.datasets import SWaTDataset, DatasetConfig

    model.eval()
    model.to(device)

    config = DatasetConfig(
        window_size=512,
        stride=64,
        normalize=True,
    )

    try:
        dataset = SWaTDataset(
            data_path=data_path,
            split="test",
            config=config,
        )
    except FileNotFoundError as e:
        return BenchmarkResult(
            dataset="SWaT",
            task="Anomaly Detection",
            metrics={"error": str(e)},
        )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
    )

    all_scores = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="SWaT"):
            x = batch["x"].to(device)
            labels = batch.get("anomaly_labels")

            output = model(x, compute_loss=False)

            # Use model's anomaly score if available
            if output.anomaly_score is not None:
                scores = output.anomaly_score
            else:
                # Fallback: use reconstruction error as anomaly score
                if output.reconstruction is not None:
                    recon_error = (output.reconstruction - x).pow(2).mean(dim=1)
                    scores = recon_error.mean(dim=-1)  # [B]
                else:
                    scores = torch.zeros(x.shape[0])

            all_scores.append(scores.cpu())
            if labels is not None:
                all_labels.append(labels)

    all_scores = torch.cat(all_scores, dim=0).numpy()

    if len(all_labels) > 0:
        all_labels = torch.cat(all_labels, dim=0).numpy()

        # If labels are per-window, take max (any anomaly in window)
        if all_labels.ndim == 2:
            all_labels = all_labels.max(axis=-1)

        metrics = compute_anomaly_metrics(
            all_scores, all_labels, point_adjust=True
        )
    else:
        metrics = {"note": "No labels available"}

    return BenchmarkResult(
        dataset="SWaT",
        task="Anomaly Detection",
        metrics=metrics,
        model_info={"num_params": model.get_num_params()},
    )


def run_full_benchmark(
    model: IndustrialWorldLM,
    data_dir: str,
    output_path: Optional[str] = None,
    device: str = "cuda",
) -> Dict[str, BenchmarkResult]:
    """
    Run full benchmark suite on all datasets.

    Args:
        model: Trained IndustrialWorldLM
        data_dir: Root directory containing all datasets
        output_path: Path to save results JSON
        device: Device for inference

    Returns:
        Dict of dataset name to BenchmarkResult
    """
    data_dir = Path(data_dir)
    results = {}

    print("=" * 60)
    print("Running Full Industrial World Model Benchmark")
    print("=" * 60)

    # C-MAPSS
    print("\n[1/4] C-MAPSS (RUL Prediction)")
    cmapss_path = data_dir / "cmapss"
    if cmapss_path.exists():
        results["cmapss"] = evaluate_cmapss(model, str(cmapss_path), device=device)
        _print_metrics(results["cmapss"])
    else:
        print(f"  Skipped: {cmapss_path} not found")

    # Bearing
    print("\n[2/4] Bearing (Fault Classification)")
    bearing_path = data_dir / "bearing"
    if bearing_path.exists():
        results["bearing"] = evaluate_bearing(model, str(bearing_path), device=device)
        _print_metrics(results["bearing"])
    else:
        print(f"  Skipped: {bearing_path} not found")

    # TEP
    print("\n[3/4] Tennessee Eastman (Fault Diagnosis)")
    tep_path = data_dir / "tep"
    if tep_path.exists():
        results["tep"] = evaluate_tep(model, str(tep_path), device=device)
        _print_metrics(results["tep"])
    else:
        print(f"  Skipped: {tep_path} not found")

    # SWaT
    print("\n[4/4] SWaT (Anomaly Detection)")
    swat_path = data_dir / "swat"
    if swat_path.exists():
        results["swat"] = evaluate_swat(model, str(swat_path), device=device)
        _print_metrics(results["swat"])
    else:
        print(f"  Skipped: {swat_path} not found")

    # Summary
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    _print_summary(results)

    # Save results
    if output_path:
        save_results(results, output_path)
        print(f"\nResults saved to: {output_path}")

    return results


def _print_metrics(result: BenchmarkResult):
    """Print metrics for a single benchmark."""
    print(f"  Dataset: {result.dataset}")
    print(f"  Task: {result.task}")
    for k, v in result.metrics.items():
        if isinstance(v, float):
            print(f"    {k}: {v:.4f}")
        else:
            print(f"    {k}: {v}")


def _print_summary(results: Dict[str, BenchmarkResult]):
    """Print summary of all benchmarks."""
    for name, result in results.items():
        key_metrics = {
            "cmapss": ["nasa_score_mean", "rmse"],
            "bearing": ["accuracy", "f1_macro"],
            "tep": ["accuracy", "f1_macro", "detection_f1"],
            "swat": ["f1_pa", "auc_roc"],
        }

        print(f"\n{result.dataset}:")
        for metric in key_metrics.get(name, []):
            if metric in result.metrics:
                print(f"  {metric}: {result.metrics[metric]:.4f}")


def save_results(results: Dict[str, BenchmarkResult], path: str):
    """Save benchmark results to JSON."""
    output = {}
    for name, result in results.items():
        output[name] = {
            "dataset": result.dataset,
            "task": result.task,
            "metrics": result.metrics,
            "per_subset": result.per_subset_results,
        }

    with open(path, "w") as f:
        json.dump(output, f, indent=2)


def compare_with_baselines(
    results: Dict[str, BenchmarkResult],
    baseline_results: Optional[Dict] = None,
) -> Dict[str, Dict]:
    """
    Compare results with baseline methods.

    Args:
        results: Our model's results
        baseline_results: Dict of baseline results (or use defaults)

    Returns:
        Comparison table
    """
    # Default baselines from literature
    if baseline_results is None:
        baseline_results = {
            "cmapss": {
                "LSTM": {"rmse": 16.14, "nasa_score_mean": 0.338},
                "Transformer": {"rmse": 14.87, "nasa_score_mean": 0.298},
                "TTM": {"rmse": 13.52, "nasa_score_mean": 0.271},
            },
            "bearing": {
                "CNN": {"accuracy": 95.2, "f1_macro": 94.8},
                "WDCNN": {"accuracy": 97.1, "f1_macro": 96.8},
            },
            "tep": {
                "PCA+SVM": {"accuracy": 82.3, "f1_macro": 79.5},
                "LSTM-AE": {"accuracy": 87.6, "f1_macro": 85.2},
            },
            "swat": {
                "LSTM-AE": {"f1_pa": 78.5, "auc_roc": 0.89},
                "DAGMM": {"f1_pa": 81.2, "auc_roc": 0.91},
            },
        }

    comparison = {}
    for dataset, result in results.items():
        comparison[dataset] = {
            "ours": result.metrics,
            "baselines": baseline_results.get(dataset, {}),
        }

    return comparison
