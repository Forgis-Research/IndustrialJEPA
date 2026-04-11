"""
Ablation experiments for STAR replication.

Ablation 1: Per-condition normalization for FD002 and FD004
Ablation 2: RUL cap sweep on FD001 (cap in {100, 110, 125, 140})
Ablation 3: Patch length sweep on FD001 (L in {2, 4, 8})
Ablation 4: n_heads sweep on FD001/FD003 ({1, 2, 4})
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

from data_utils import prepare_data
from models import build_model, STAR, count_parameters
from train_utils import train, evaluate

RESULTS_DIR = Path(__file__).parent / "results"
ABLATION_DIR = RESULTS_DIR / "ablations"
CHECKPOINT_DIR = RESULTS_DIR / "checkpoints"
EXPERIMENT_LOG = Path(__file__).parent / "EXPERIMENT_LOG.md"

SEEDS = [42, 123, 456]  # 3 seeds for ablations (save time)
RUL_CAP = 125.0
MAX_EPOCHS = 150  # shorter for ablations
PATIENCE = 15


def log_ablation(name, config_str, rmse_mean, rmse_std, score_mean, score_std, notes=""):
    ts = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    entry = f"""
### {ts} | ABLATION: {name}

- **Config**: {config_str}
- **RMSE**: {rmse_mean:.3f} +/- {rmse_std:.3f}
- **Score**: {score_mean:.1f} +/- {score_std:.1f}
- **Notes**: {notes}

"""
    with open(EXPERIMENT_LOG, "a") as f:
        f.write(entry)


def run_ablation_condition_norm():
    """Ablation 1: Per-condition normalization for FD002 and FD004."""
    print("\n" + "="*60)
    print("ABLATION 1: Per-condition normalization (FD002, FD004)")
    print("="*60)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    results = {}

    for subset in ["FD002", "FD004"]:
        configs = {
            "FD002": dict(lr=0.0002, batch_size=64, window_length=64, n_scales=4, d_model=64, n_heads=4),
            "FD004": dict(lr=0.0002, batch_size=64, window_length=64, n_scales=4, d_model=256, n_heads=4),
        }
        paper_targets = {"FD002": dict(rmse=13.47), "FD004": dict(rmse=15.87)}
        cfg = configs[subset]

        all_rmse, all_score = [], []
        for seed in SEEDS:
            torch.manual_seed(seed)
            np.random.seed(seed)

            data = prepare_data(
                subset,
                window_length=cfg["window_length"],
                batch_size=cfg["batch_size"],
                seed=seed,
                rul_cap=int(RUL_CAP),
                use_cond_norm=True,  # KEY: per-condition normalization
            )

            model = build_model(subset, device)
            chk_dir = CHECKPOINT_DIR / f"{subset}_cond_norm"
            chk_dir.mkdir(parents=True, exist_ok=True)

            train_info = train(
                model=model,
                train_loader=data["train_loader"],
                val_loader=data["val_loader"],
                lr=cfg["lr"],
                max_epochs=MAX_EPOCHS,
                patience=PATIENCE,
                device=device,
                checkpoint_path=str(chk_dir / f"seed{seed}.pt"),
                verbose=True,
            )

            rmse, score, _, _ = evaluate(model, data["test_loader"], device)
            all_rmse.append(rmse)
            all_score.append(score)
            print(f"  {subset} seed={seed}: RMSE={rmse:.3f}, Score={score:.1f}")

        rmse_mean = float(np.mean(all_rmse))
        rmse_std = float(np.std(all_rmse))
        score_mean = float(np.mean(all_score))
        score_std = float(np.std(all_score))

        print(f"\n{subset} per-condition norm: RMSE={rmse_mean:.3f}+/-{rmse_std:.3f} "
              f"| Paper: {paper_targets[subset]['rmse']}")

        results[subset] = dict(rmse_mean=rmse_mean, rmse_std=rmse_std,
                               score_mean=score_mean, score_std=score_std)
        log_ablation(f"per_cond_norm_{subset}", f"use_cond_norm=True",
                     rmse_mean, rmse_std, score_mean, score_std)

    out = ABLATION_DIR / "cond_norm_results.json"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {out}")
    return results


def run_ablation_rul_cap():
    """Ablation 2: RUL cap sweep on FD001."""
    print("\n" + "="*60)
    print("ABLATION 2: RUL cap sweep on FD001 (caps: 100, 110, 125, 140)")
    print("="*60)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cfg = dict(lr=0.0002, batch_size=32, window_length=32)
    results = {}

    for cap in [100, 110, 125, 140]:
        print(f"\n--- cap={cap} ---")
        all_rmse, all_score = [], []

        for seed in SEEDS:
            torch.manual_seed(seed)
            np.random.seed(seed)

            data = prepare_data(
                "FD001",
                window_length=cfg["window_length"],
                batch_size=cfg["batch_size"],
                seed=seed,
                rul_cap=cap,
            )

            model = build_model("FD001", device)
            chk_dir = CHECKPOINT_DIR / f"FD001_cap{cap}"
            chk_dir.mkdir(parents=True, exist_ok=True)

            train_info = train(
                model=model,
                train_loader=data["train_loader"],
                val_loader=data["val_loader"],
                lr=cfg["lr"],
                max_epochs=MAX_EPOCHS,
                patience=PATIENCE,
                device=device,
                checkpoint_path=str(chk_dir / f"seed{seed}.pt"),
                verbose=False,
            )

            # Evaluate with cap-normalized predictions
            model.eval()
            preds, trues = [], []
            with torch.no_grad():
                for X, y in data["test_loader"]:
                    X = X.to(device)
                    out = model(X).cpu().numpy() * cap
                    preds.append(np.clip(out, 0, cap))
                    trues.append(y.numpy())
            preds = np.concatenate(preds)
            trues = np.concatenate(trues)
            rmse = float(np.sqrt(np.mean((preds - trues) ** 2)))
            d = preds - trues
            score = float(np.where(d < 0, np.exp(-d/13) - 1, np.exp(d/10) - 1).sum())

            all_rmse.append(rmse)
            all_score.append(score)
            print(f"  cap={cap} seed={seed}: RMSE={rmse:.3f}, Score={score:.1f}")

        rmse_mean = float(np.mean(all_rmse))
        rmse_std = float(np.std(all_rmse))
        score_mean = float(np.mean(all_score))
        score_std = float(np.std(all_score))
        print(f"cap={cap}: RMSE={rmse_mean:.3f}+/-{rmse_std:.3f}")
        results[str(cap)] = dict(rmse_mean=rmse_mean, rmse_std=rmse_std,
                                  score_mean=score_mean, score_std=score_std)
        log_ablation(f"rul_cap_FD001_cap{cap}", f"cap={cap}",
                     rmse_mean, rmse_std, score_mean, score_std)

    out = ABLATION_DIR / "rul_cap_results.json"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {out}")
    return results


def run_ablation_patch_length():
    """Ablation 3: Patch length sweep on FD001 (L in {2, 4, 8})."""
    print("\n" + "="*60)
    print("ABLATION 3: Patch length sweep on FD001 (L: 2, 4, 8)")
    print("="*60)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    results = {}

    for L in [2, 4, 8]:
        print(f"\n--- patch_length={L} ---")
        # Window=32, n_scales=3: K=32/L
        # L=2: K=16, merging: 16->8->4
        # L=4: K=8, merging: 8->4->2
        # L=8: K=4, merging: 4->2->1
        K0 = 32 // L
        print(f"  K0={K0}")

        all_rmse, all_score = [], []

        for seed in SEEDS:
            torch.manual_seed(seed)
            np.random.seed(seed)

            data = prepare_data("FD001", window_length=32, batch_size=32, seed=seed, rul_cap=125)

            model = STAR(T=32, D=14, patch_length=L, n_scales=3, d_model=128, n_heads=1).to(device)
            n_params = count_parameters(model)

            chk_dir = CHECKPOINT_DIR / f"FD001_L{L}"
            chk_dir.mkdir(parents=True, exist_ok=True)

            train_info = train(
                model=model,
                train_loader=data["train_loader"],
                val_loader=data["val_loader"],
                lr=0.0002,
                max_epochs=MAX_EPOCHS,
                patience=PATIENCE,
                device=device,
                checkpoint_path=str(chk_dir / f"seed{seed}.pt"),
                verbose=False,
            )

            rmse, score, _, _ = evaluate(model, data["test_loader"], device)
            all_rmse.append(rmse)
            all_score.append(score)
            print(f"  L={L} seed={seed}: RMSE={rmse:.3f}, Score={score:.1f}, params={n_params:,}")

        rmse_mean = float(np.mean(all_rmse))
        rmse_std = float(np.std(all_rmse))
        score_mean = float(np.mean(all_score))
        score_std = float(np.std(all_score))
        print(f"L={L}: RMSE={rmse_mean:.3f}+/-{rmse_std:.3f}")
        results[str(L)] = dict(rmse_mean=rmse_mean, rmse_std=rmse_std,
                                score_mean=score_mean, score_std=score_std,
                                n_params=n_params)
        log_ablation(f"patch_length_FD001_L{L}", f"patch_length={L}, K0={K0}",
                     rmse_mean, rmse_std, score_mean, score_std)

    out = ABLATION_DIR / "patch_length_results.json"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {out}")
    return results


def run_ablation_nheads():
    """Ablation 4: n_heads sweep on FD001 and FD003."""
    print("\n" + "="*60)
    print("ABLATION 4: n_heads sweep on FD001/FD003 (heads: 1, 2, 4)")
    print("="*60)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    results = {}

    for subset in ["FD001", "FD003"]:
        cfg = {
            "FD001": dict(T=32, window=32, bs=32, n_scales=3, d_model=128),
            "FD003": dict(T=48, window=48, bs=32, n_scales=1, d_model=128),
        }[subset]

        for nh in [1, 2, 4]:
            print(f"\n--- {subset} n_heads={nh} ---")
            all_rmse, all_score = [], []

            for seed in SEEDS:
                torch.manual_seed(seed)
                np.random.seed(seed)

                data = prepare_data(subset, window_length=cfg["window"], batch_size=cfg["bs"],
                                   seed=seed, rul_cap=125)

                model = STAR(T=cfg["T"], D=14, patch_length=4, n_scales=cfg["n_scales"],
                             d_model=cfg["d_model"], n_heads=nh).to(device)

                chk_dir = CHECKPOINT_DIR / f"{subset}_nh{nh}"
                chk_dir.mkdir(parents=True, exist_ok=True)

                train_info = train(
                    model=model,
                    train_loader=data["train_loader"],
                    val_loader=data["val_loader"],
                    lr=0.0002,
                    max_epochs=MAX_EPOCHS,
                    patience=PATIENCE,
                    device=device,
                    checkpoint_path=str(chk_dir / f"seed{seed}.pt"),
                    verbose=False,
                )

                rmse, score, _, _ = evaluate(model, data["test_loader"], device)
                all_rmse.append(rmse)
                all_score.append(score)
                print(f"  {subset} nh={nh} seed={seed}: RMSE={rmse:.3f}, Score={score:.1f}")

            rmse_mean = float(np.mean(all_rmse))
            rmse_std = float(np.std(all_rmse))
            score_mean = float(np.mean(all_score))
            score_std = float(np.std(all_score))
            print(f"{subset} nh={nh}: RMSE={rmse_mean:.3f}+/-{rmse_std:.3f}")
            key = f"{subset}_nh{nh}"
            results[key] = dict(rmse_mean=rmse_mean, rmse_std=rmse_std,
                                score_mean=score_mean, score_std=score_std)
            log_ablation(f"nheads_{subset}_nh{nh}", f"n_heads={nh}",
                         rmse_mean, rmse_std, score_mean, score_std)

    out = ABLATION_DIR / "nheads_results.json"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {out}")
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("ablation", choices=["cond_norm", "rul_cap", "patch_length", "nheads", "all"])
    args = parser.parse_args()

    ABLATION_DIR.mkdir(parents=True, exist_ok=True)

    if args.ablation in ("cond_norm", "all"):
        run_ablation_condition_norm()
    if args.ablation in ("rul_cap", "all"):
        run_ablation_rul_cap()
    if args.ablation in ("patch_length", "all"):
        run_ablation_patch_length()
    if args.ablation in ("nheads", "all"):
        run_ablation_nheads()
