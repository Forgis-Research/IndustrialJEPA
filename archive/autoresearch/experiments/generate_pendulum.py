#!/usr/bin/env python3
"""
Generate Double Pendulum Dataset (One-Time)

Creates synthetic trajectories with known physics for Tier 1 validation.
Run once, save as CSV, use forever.

Usage:
    python generate_pendulum.py --output data/pendulum.csv

Physics:
    Double pendulum with masses m1, m2 and lengths l1, l2.
    State: [theta1, omega1, theta2, omega2]
    Groups: mass_1 = [theta1, omega1], mass_2 = [theta2, omega2]

Transfer test:
    Source: m1/m2 = 1.0 (balanced)
    Target: m1/m2 = 0.5 (unbalanced)
"""

import argparse
import numpy as np
import pandas as pd
from scipy.integrate import odeint
from pathlib import Path


def double_pendulum_derivatives(state, t, m1, m2, l1, l2, g=9.81):
    """
    Compute derivatives for double pendulum using Lagrangian mechanics.

    State: [theta1, omega1, theta2, omega2]
    Returns: [dtheta1, domega1, dtheta2, domega2]
    """
    theta1, omega1, theta2, omega2 = state
    delta = theta2 - theta1

    # Denominators (avoid division by zero)
    denom1 = (m1 + m2) * l1 - m2 * l1 * np.cos(delta) ** 2
    denom2 = (l2 / l1) * denom1

    # Angular accelerations from Lagrangian equations
    domega1 = (
        m2 * l1 * omega1 ** 2 * np.sin(delta) * np.cos(delta)
        + m2 * g * np.sin(theta2) * np.cos(delta)
        + m2 * l2 * omega2 ** 2 * np.sin(delta)
        - (m1 + m2) * g * np.sin(theta1)
    ) / denom1

    domega2 = (
        -m2 * l2 * omega2 ** 2 * np.sin(delta) * np.cos(delta)
        + (m1 + m2) * g * np.sin(theta1) * np.cos(delta)
        - (m1 + m2) * l1 * omega1 ** 2 * np.sin(delta)
        - (m1 + m2) * g * np.sin(theta2)
    ) / denom2

    return [omega1, domega1, omega2, domega2]


def generate_trajectory(
    m1: float,
    m2: float,
    l1: float = 1.0,
    l2: float = 1.0,
    g: float = 9.81,
    timesteps: int = 1000,
    dt: float = 0.01,
    initial_state: np.ndarray = None,
    seed: int = None,
) -> np.ndarray:
    """
    Generate a single double pendulum trajectory.

    Returns: array of shape [timesteps, 4] with [theta1, omega1, theta2, omega2]
    """
    if seed is not None:
        np.random.seed(seed)

    # Random initial conditions if not provided
    if initial_state is None:
        # Small angles (non-chaotic regime) for stable simulation
        theta1_0 = np.random.uniform(-np.pi / 6, np.pi / 6)
        theta2_0 = np.random.uniform(-np.pi / 6, np.pi / 6)
        omega1_0 = np.random.uniform(-1.0, 1.0)
        omega2_0 = np.random.uniform(-1.0, 1.0)
        initial_state = [theta1_0, omega1_0, theta2_0, omega2_0]

    t = np.linspace(0, timesteps * dt, timesteps)

    trajectory = odeint(
        double_pendulum_derivatives,
        initial_state,
        t,
        args=(m1, m2, l1, l2, g),
    )

    return trajectory


def generate_dataset(
    n_trajectories: int = 10000,
    timesteps: int = 1000,
    dt: float = 0.01,
    source_ratio: float = 0.7,  # 70% source (m_ratio=1.0), 30% target (m_ratio=0.5)
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate full dataset with source and target domains.

    Source: m1/m2 = 1.0 (balanced masses)
    Target: m1/m2 = 0.5 (unbalanced masses)
    """
    np.random.seed(seed)

    n_source = int(n_trajectories * source_ratio)
    n_target = n_trajectories - n_source

    rows = []

    # Source domain: m1 = m2 = 1.0
    print(f"Generating {n_source} source trajectories (m_ratio=1.0)...")
    for traj_id in range(n_source):
        m1, m2 = 1.0, 1.0
        traj = generate_trajectory(m1, m2, timesteps=timesteps, dt=dt, seed=seed + traj_id)

        for t, state in enumerate(traj):
            rows.append({
                "trajectory_id": traj_id,
                "timestep": t,
                "theta1": state[0],
                "omega1": state[1],
                "theta2": state[2],
                "omega2": state[3],
                "m1": m1,
                "m2": m2,
                "m_ratio": m1 / m2,
                "domain": "source",
            })

        if (traj_id + 1) % 1000 == 0:
            print(f"  {traj_id + 1}/{n_source} done")

    # Target domain: m1 = 0.5, m2 = 1.0 (ratio = 0.5)
    print(f"Generating {n_target} target trajectories (m_ratio=0.5)...")
    for i, traj_id in enumerate(range(n_source, n_trajectories)):
        m1, m2 = 0.5, 1.0
        traj = generate_trajectory(m1, m2, timesteps=timesteps, dt=dt, seed=seed + traj_id)

        for t, state in enumerate(traj):
            rows.append({
                "trajectory_id": traj_id,
                "timestep": t,
                "theta1": state[0],
                "omega1": state[1],
                "theta2": state[2],
                "omega2": state[3],
                "m1": m1,
                "m2": m2,
                "m_ratio": m1 / m2,
                "domain": "target",
            })

        if (i + 1) % 1000 == 0:
            print(f"  {i + 1}/{n_target} done")

    df = pd.DataFrame(rows)
    return df


def main():
    parser = argparse.ArgumentParser(description="Generate Double Pendulum Dataset")
    parser.add_argument("--n_trajectories", type=int, default=10000, help="Total trajectories")
    parser.add_argument("--timesteps", type=int, default=1000, help="Steps per trajectory")
    parser.add_argument("--dt", type=float, default=0.01, help="Time step (seconds)")
    parser.add_argument("--source_ratio", type=float, default=0.7, help="Fraction for source domain")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default="data/pendulum.csv", help="Output path")
    args = parser.parse_args()

    # Generate dataset
    df = generate_dataset(
        n_trajectories=args.n_trajectories,
        timesteps=args.timesteps,
        dt=args.dt,
        source_ratio=args.source_ratio,
        seed=args.seed,
    )

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    # Print summary
    print(f"\nDataset saved to {output_path}")
    print(f"Total rows: {len(df):,}")
    print(f"Trajectories: {df['trajectory_id'].nunique():,}")
    print(f"Source (m_ratio=1.0): {len(df[df['domain']=='source']):,} rows")
    print(f"Target (m_ratio=0.5): {len(df[df['domain']=='target']):,} rows")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nPhysics groups for Role-Transformer:")
    print("  mass_1: ['theta1', 'omega1']")
    print("  mass_2: ['theta2', 'omega2']")


if __name__ == "__main__":
    main()
