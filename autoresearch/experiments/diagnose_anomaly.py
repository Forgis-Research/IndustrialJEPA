#!/usr/bin/env python
"""Diagnose why anomaly detection fails: examine signal differences."""

import sys
import numpy as np
import torch
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

FIGURES_DIR = PROJECT_ROOT / "autoresearch" / "figures"
FIGURES_DIR.mkdir(exist_ok=True)


def main():
    from industrialjepa.data.factorynet import FactoryNetDataset, FactoryNetConfig

    # Load both datasets with episode normalization
    for norm_mode in ["episode", "global"]:
        for ds_name, ds_source in [("AURSAD", "aursad"), ("Voraus", "voraus")]:
            print(f"\n{'='*60}")
            print(f"Analyzing {ds_name} ({norm_mode} normalization)")
            print(f"{'='*60}")

            config = FactoryNetConfig(
                data_source=ds_source,
                max_episodes=200,
                window_size=256, stride=256,
                normalize=True, norm_mode=norm_mode,
                effort_signals=["voltage"],
                setpoint_signals=["position", "velocity"],
                unified_setpoint_dim=12, unified_effort_dim=6,
            )
            ds = FactoryNetDataset(config, split='test')

            # Collect stats per class
            normal_efforts = []
            anomaly_efforts = []
            normal_setpoints = []
            anomaly_setpoints = []

            n = min(500, len(ds))
            for i in range(n):
                sp, eff, meta = ds[i]
                if meta['is_anomaly']:
                    anomaly_efforts.append(eff.numpy())
                    anomaly_setpoints.append(sp.numpy())
                else:
                    normal_efforts.append(eff.numpy())
                    normal_setpoints.append(sp.numpy())

            print(f"  Normal windows: {len(normal_efforts)}")
            print(f"  Anomaly windows: {len(anomaly_efforts)}")

            if not normal_efforts or not anomaly_efforts:
                print("  Skipping (no samples in one class)")
                continue

            normal_efforts = np.stack(normal_efforts)  # (N, T, D)
            anomaly_efforts = np.stack(anomaly_efforts)
            normal_setpoints = np.stack(normal_setpoints)
            anomaly_setpoints = np.stack(anomaly_setpoints)

            # Stats
            print(f"\n  Effort statistics ({norm_mode}):")
            for d in range(min(6, normal_efforts.shape[-1])):
                nm = normal_efforts[:, :, d].flatten()
                am = anomaly_efforts[:, :, d].flatten()
                print(f"    Dim {d}: Normal mean={nm.mean():.4f} std={nm.std():.4f} | "
                      f"Anomaly mean={am.mean():.4f} std={am.std():.4f} | "
                      f"Diff={abs(nm.mean()-am.mean()):.4f}")

            # Temporal stats
            print(f"\n  Temporal statistics:")
            # Compute mean absolute temporal difference (how much signals change)
            normal_diff = np.abs(np.diff(normal_efforts, axis=1)).mean(axis=(0, 1))
            anomaly_diff = np.abs(np.diff(anomaly_efforts, axis=1)).mean(axis=(0, 1))
            for d in range(min(6, len(normal_diff))):
                print(f"    Dim {d}: Normal Δ={normal_diff[d]:.4f} | Anomaly Δ={anomaly_diff[d]:.4f}")

            # Variance within window
            normal_var = normal_efforts.var(axis=1).mean(axis=0)
            anomaly_var = anomaly_efforts.var(axis=1).mean(axis=0)
            print(f"\n  Within-window variance:")
            for d in range(min(6, len(normal_var))):
                print(f"    Dim {d}: Normal var={normal_var[d]:.4f} | Anomaly var={anomaly_var[d]:.4f}")

            # Check correlation between setpoint and effort
            # If anomalies break the setpoint→effort relationship, we should see lower correlation
            print(f"\n  Setpoint-Effort correlation:")
            for d in range(min(6, normal_efforts.shape[-1])):
                # Use first setpoint dim (position_0)
                sp_d = 0
                nm_corr = np.corrcoef(
                    normal_setpoints[:, :, sp_d].flatten(),
                    normal_efforts[:, :, d].flatten()
                )[0, 1]
                am_corr = np.corrcoef(
                    anomaly_setpoints[:, :, sp_d].flatten(),
                    anomaly_efforts[:, :, d].flatten()
                )[0, 1]
                print(f"    Effort dim {d} vs Setpoint pos_0: Normal r={nm_corr:.4f} | Anomaly r={am_corr:.4f}")

    # Plot signal examples
    print("\n\nGenerating diagnostic plots...")
    config = FactoryNetConfig(
        data_source='voraus', max_episodes=200,
        window_size=256, stride=256,
        normalize=True, norm_mode="episode",
        effort_signals=["voltage"],
        setpoint_signals=["position", "velocity"],
        unified_setpoint_dim=12, unified_effort_dim=6,
    )
    ds = FactoryNetDataset(config, split='test')

    fig, axes = plt.subplots(3, 2, figsize=(16, 12))

    # Find one normal and one anomalous example
    normal_idx = anomaly_idx = None
    for i in range(len(ds)):
        _, _, m = ds[i]
        if not m['is_anomaly'] and normal_idx is None:
            normal_idx = i
        if m['is_anomaly'] and anomaly_idx is None:
            anomaly_idx = i
        if normal_idx is not None and anomaly_idx is not None:
            break

    for col, (idx, label) in enumerate([(normal_idx, "Normal"), (anomaly_idx, "Anomaly")]):
        if idx is None:
            continue
        sp, eff, meta = ds[idx]
        sp, eff = sp.numpy(), eff.numpy()

        # Effort signals
        axes[0, col].set_title(f"{label} - Effort (voltage)")
        for d in range(min(6, eff.shape[1])):
            axes[0, col].plot(eff[:, d], alpha=0.7, label=f'V{d}')
        axes[0, col].legend(fontsize=6)

        # Setpoint signals
        axes[1, col].set_title(f"{label} - Setpoint (position)")
        for d in range(min(6, sp.shape[1])):
            axes[1, col].plot(sp[:, d], alpha=0.7, label=f'P{d}')
        axes[1, col].legend(fontsize=6)

        # Effort - Setpoint relationship
        axes[2, col].set_title(f"{label} - Effort[0] vs Setpoint[0]")
        axes[2, col].scatter(sp[:, 0], eff[:, 0], alpha=0.3, s=1)
        axes[2, col].set_xlabel('Setpoint pos_0')
        axes[2, col].set_ylabel('Effort V_0')

    plt.suptitle("Voraus: Normal vs Anomalous Windows (Episode Normalized)", fontsize=14)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "diagnose_anomaly_signals.png", dpi=150)
    plt.close()
    print(f"Saved: {FIGURES_DIR / 'diagnose_anomaly_signals.png'}")


if __name__ == "__main__":
    main()
