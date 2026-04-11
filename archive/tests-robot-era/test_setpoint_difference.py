# SPDX-FileCopyrightText: 2025 Industrial JEPA Authors
# SPDX-License-Identifier: MIT

"""
Check if setpoints differ between healthy and missing_screw.

If setpoints are THE SAME but effort differs, then setpoint->effort
prediction CANNOT detect the anomaly - the model has no way to know
a screw is missing just from the setpoint!
"""

import numpy as np
from industrialjepa.data.factorynet import FactoryNetDataset, FactoryNetConfig


def main():
    config = FactoryNetConfig(
        dataset_name="Forgis/factorynet-hackathon",
        subset="aursad",
        window_size=256,
        stride=256,
        normalize=False,
        train_healthy_only=False,
        aursad_phase_handling="tightening_only",
    )

    print("Loading data...")
    ds = FactoryNetDataset(config, split="test")

    healthy_setpoints = []
    missing_setpoints = []

    for i in range(len(ds)):
        setpoint, _, meta = ds[i]
        fault = meta["fault_type"]

        # Take mean of setpoint as summary
        sp_mean = setpoint.mean(dim=0).numpy()

        if fault == "normal":
            healthy_setpoints.append(sp_mean)
        elif fault == "missing_screw":
            missing_setpoints.append(sp_mean)

    healthy_setpoints = np.array(healthy_setpoints)
    missing_setpoints = np.array(missing_setpoints)

    print(f"\nHealthy: {len(healthy_setpoints)} samples")
    print(f"Missing: {len(missing_setpoints)} samples")

    print("\n" + "="*60)
    print("SETPOINT COMPARISON")
    print("="*60)

    for i, col in enumerate(ds.setpoint_cols):
        h_mean = healthy_setpoints[:, i].mean()
        h_std = healthy_setpoints[:, i].std()
        m_mean = missing_setpoints[:, i].mean()
        m_std = missing_setpoints[:, i].std()

        if h_std > 0:
            effect = (m_mean - h_mean) / h_std
        else:
            effect = 0

        print(f"{col}: healthy={h_mean:.3f}+/-{h_std:.3f}, missing={m_mean:.3f}+/-{m_std:.3f}, effect={effect:+.2f} std")

    # Overall setpoint difference
    h_flat = healthy_setpoints.reshape(len(healthy_setpoints), -1)
    m_flat = missing_setpoints.reshape(len(missing_setpoints), -1)

    overall_effect = np.abs(m_flat.mean(axis=0) - h_flat.mean(axis=0)) / (h_flat.std(axis=0) + 1e-8)
    print(f"\nOverall mean absolute effect: {overall_effect.mean():.3f} std")

    if overall_effect.mean() < 0.1:
        print("\n" + "="*60)
        print("CRITICAL FINDING:")
        print("Setpoints are NEARLY IDENTICAL for healthy vs missing_screw!")
        print("The robot controller sends the same commands regardless of screw presence.")
        print("")
        print("IMPLICATION:")
        print("Setpoint->Effort prediction CANNOT detect this anomaly because")
        print("the model has no information about the screw state in the input.")
        print("The anomaly is in the EFFORT response, not the SETPOINT command.")
        print("="*60)


if __name__ == "__main__":
    main()
