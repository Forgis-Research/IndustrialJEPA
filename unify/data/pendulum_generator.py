"""
N-Link Pendulum Dataset Generator for Unify Project
=====================================================

Generates synthetic N-link pendulum trajectories using Lagrangian mechanics.
This implementation derives the equations of motion from the Lagrangian:

    L = T - V

where T is kinetic energy and V is potential energy.

For an N-link pendulum with generalized coordinates q = [theta_1, ..., theta_N]:
- Each theta_i is the angle of link i relative to vertical
- The equations of motion are: M(q) * q_ddot + C(q, q_dot) * q_dot + G(q) = tau - B * q_dot

where:
- M(q) is the mass matrix (N x N)
- C(q, q_dot) contains Coriolis and centrifugal terms
- G(q) is the gravity vector
- B is the damping coefficient matrix
- tau is external torque (zero for free motion)

References:
- Murray, Li, Sastry. "A Mathematical Introduction to Robotic Manipulation" (1994)
- Spong, Hutchinson, Vidyasagar. "Robot Modeling and Control" (2006)

Author: Unify Research Project
"""

import numpy as np
from scipy.integrate import solve_ivp
import json
import os
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional
import csv
from pathlib import Path
from tqdm import tqdm
import warnings

# Physical constants
G = 9.81  # Gravitational acceleration (m/s^2)


@dataclass
class PendulumParameters:
    """Physical parameters for an N-link pendulum."""
    n_links: int
    masses: List[float]      # Mass of each link [kg]
    lengths: List[float]     # Length of each link [m]
    damping: List[float]     # Damping coefficient per joint [N*m*s/rad]

    def __post_init__(self):
        assert len(self.masses) == self.n_links
        assert len(self.lengths) == self.n_links
        assert len(self.damping) == self.n_links


@dataclass
class TrajectoryMetadata:
    """Metadata for a single trajectory."""
    trajectory_id: int
    n_links: int
    masses: List[float]
    lengths: List[float]
    damping: List[float]
    initial_angles: List[float]
    initial_velocities: List[float]
    dt: float
    n_timesteps: int
    total_time: float
    integration_method: str
    max_integration_step: float


class NLinkPendulumDynamics:
    """
    Computes dynamics for an N-link pendulum using Lagrangian mechanics.

    The position of the center of mass of link i is:
        x_i = sum_{j=1}^{i-1} l_j * sin(theta_j) + (l_i/2) * sin(theta_i)
        y_i = -sum_{j=1}^{i-1} l_j * cos(theta_j) - (l_i/2) * cos(theta_i)

    The kinetic energy is:
        T = (1/2) * sum_i (m_i * v_i^2 + I_i * omega_i^2)

    For a thin rod, I_i = (1/12) * m_i * l_i^2

    The potential energy is:
        V = sum_i (m_i * g * y_cm_i)
    """

    def __init__(self, params: PendulumParameters):
        self.params = params
        self.n = params.n_links
        self.m = np.array(params.masses)
        self.l = np.array(params.lengths)
        self.b = np.array(params.damping)

        # Moment of inertia for uniform rod about center: I = (1/12) * m * l^2
        self.I = (1.0 / 12.0) * self.m * self.l**2

    def compute_mass_matrix(self, theta: np.ndarray) -> np.ndarray:
        """
        Compute the mass matrix M(q) for the N-link pendulum.

        The mass matrix elements M_ij depend on the coupling between joints.
        Using the formula from Spong et al.:

        M_ij = sum_{k=max(i,j)}^{n} [ m_k * l_i * l_j * cos(theta_i - theta_j)
               + ... (center of mass terms) ]

        For computational efficiency, we compute this element by element.
        """
        n = self.n
        M = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                # Contribution from links k >= max(i, j)
                for k in range(max(i, j), n):
                    # Distance from pivot to CoM of link k
                    if k == i == j:
                        # Diagonal contribution from link k's own rotation
                        # Uses parallel axis theorem
                        d_ik = self.l[i] / 2 if k == i else self.l[i]
                        d_jk = self.l[j] / 2 if k == j else self.l[j]
                        M[i, j] += self.m[k] * d_ik * d_jk * np.cos(theta[i] - theta[j])
                        if k == i:
                            M[i, j] += self.I[k]
                    else:
                        # Full length for links before link k, half for link k
                        d_ik = self.l[i] if i < k else self.l[i] / 2
                        d_jk = self.l[j] if j < k else self.l[j] / 2
                        M[i, j] += self.m[k] * d_ik * d_jk * np.cos(theta[i] - theta[j])

        return M

    def compute_coriolis_matrix(self, theta: np.ndarray, theta_dot: np.ndarray) -> np.ndarray:
        """
        Compute the Coriolis/centrifugal matrix C(q, q_dot).

        Using Christoffel symbols of the first kind:
        C_ij = sum_k c_ijk * theta_dot_k
        c_ijk = (1/2) * (dM_ij/dq_k + dM_ik/dq_j - dM_jk/dq_i)
        """
        n = self.n
        C = np.zeros((n, n))
        eps = 1e-8

        # Compute M at current configuration
        M = self.compute_mass_matrix(theta)

        # Compute partial derivatives numerically for robustness
        dM_dq = np.zeros((n, n, n))  # dM_dq[i, j, k] = dM_ij / dtheta_k

        for k in range(n):
            theta_plus = theta.copy()
            theta_minus = theta.copy()
            theta_plus[k] += eps
            theta_minus[k] -= eps

            M_plus = self.compute_mass_matrix(theta_plus)
            M_minus = self.compute_mass_matrix(theta_minus)

            dM_dq[:, :, k] = (M_plus - M_minus) / (2 * eps)

        # Christoffel symbols and Coriolis matrix
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    c_ijk = 0.5 * (dM_dq[i, j, k] + dM_dq[i, k, j] - dM_dq[j, k, i])
                    C[i, j] += c_ijk * theta_dot[k]

        return C

    def compute_gravity_vector(self, theta: np.ndarray) -> np.ndarray:
        """
        Compute the gravity vector G(q) = dV/dq.

        V = g * sum_i m_i * y_cm_i

        where y_cm_i = -sum_{j<i} l_j * cos(theta_j) - (l_i/2) * cos(theta_i)
        """
        n = self.n
        G_vec = np.zeros(n)

        for i in range(n):
            # dV/dtheta_i has contributions from all links k >= i
            for k in range(i, n):
                if k == i:
                    # Link k rotates about its own pivot
                    G_vec[i] += self.m[k] * G * (self.l[i] / 2) * np.sin(theta[i])
                else:
                    # Link k is affected by rotation of joint i (full length)
                    G_vec[i] += self.m[k] * G * self.l[i] * np.sin(theta[i])

        return G_vec

    def equations_of_motion(self, t: float, state: np.ndarray) -> np.ndarray:
        """
        Compute the state derivative for integration.

        state = [theta_1, ..., theta_n, theta_dot_1, ..., theta_dot_n]

        Returns d(state)/dt = [theta_dot, theta_ddot]
        """
        n = self.n
        theta = state[:n]
        theta_dot = state[n:]

        # Compute dynamics matrices
        M = self.compute_mass_matrix(theta)
        C = self.compute_coriolis_matrix(theta, theta_dot)
        G_vec = self.compute_gravity_vector(theta)

        # Damping torque: tau_damping = -B * theta_dot
        tau_damping = -self.b * theta_dot

        # Equation of motion: M * theta_ddot + C * theta_dot + G = tau_damping
        # Solve for theta_ddot: theta_ddot = M^{-1} * (tau_damping - C * theta_dot - G)
        rhs = tau_damping - C @ theta_dot - G_vec

        try:
            theta_ddot = np.linalg.solve(M, rhs)
        except np.linalg.LinAlgError:
            # Fallback to pseudoinverse if singular
            theta_ddot = np.linalg.lstsq(M, rhs, rcond=None)[0]

        return np.concatenate([theta_dot, theta_ddot])


def generate_random_parameters(
    n_links: int,
    mass_range: Tuple[float, float] = (0.5, 2.0),
    length_range: Tuple[float, float] = (0.5, 1.5),
    damping_range: Tuple[float, float] = (0.01, 0.1),
    rng: Optional[np.random.Generator] = None
) -> PendulumParameters:
    """Generate random physical parameters for an N-link pendulum."""
    if rng is None:
        rng = np.random.default_rng()

    masses = rng.uniform(mass_range[0], mass_range[1], size=n_links).tolist()
    lengths = rng.uniform(length_range[0], length_range[1], size=n_links).tolist()
    damping = rng.uniform(damping_range[0], damping_range[1], size=n_links).tolist()

    return PendulumParameters(
        n_links=n_links,
        masses=masses,
        lengths=lengths,
        damping=damping
    )


def generate_random_initial_conditions(
    n_links: int,
    angle_range: Tuple[float, float] = (-np.pi/3, np.pi/3),
    velocity_range: Tuple[float, float] = (-1.0, 1.0),
    rng: Optional[np.random.Generator] = None
) -> Tuple[List[float], List[float]]:
    """
    Generate random initial conditions.

    Angles are relative to vertical (hanging down).
    theta = 0 means hanging straight down.
    """
    if rng is None:
        rng = np.random.default_rng()

    initial_angles = rng.uniform(angle_range[0], angle_range[1], size=n_links).tolist()
    initial_velocities = rng.uniform(velocity_range[0], velocity_range[1], size=n_links).tolist()

    return initial_angles, initial_velocities


def simulate_trajectory(
    params: PendulumParameters,
    initial_angles: List[float],
    initial_velocities: List[float],
    dt: float = 0.01,
    n_timesteps: int = 1000,
    integration_method: str = 'RK45',
    max_step: float = 0.001
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate an N-link pendulum trajectory.

    Args:
        params: Physical parameters
        initial_angles: Initial joint angles [rad]
        initial_velocities: Initial angular velocities [rad/s]
        dt: Timestep for output [s]
        n_timesteps: Number of timesteps to simulate
        integration_method: scipy solve_ivp method ('RK45', 'DOP853', 'Radau', etc.)
        max_step: Maximum step size for integrator

    Returns:
        times: Array of timestamps [s]
        angles: Array of joint angles (n_timesteps x n_links)
        velocities: Array of angular velocities (n_timesteps x n_links)
    """
    n = params.n_links
    dynamics = NLinkPendulumDynamics(params)

    # Initial state
    initial_state = np.array(initial_angles + initial_velocities)

    # Time span
    t_span = (0.0, dt * n_timesteps)
    t_eval = np.linspace(0.0, dt * n_timesteps, n_timesteps + 1)

    # Integrate with error handling
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning)

        solution = solve_ivp(
            dynamics.equations_of_motion,
            t_span,
            initial_state,
            method=integration_method,
            t_eval=t_eval,
            max_step=max_step,
            rtol=1e-8,
            atol=1e-10
        )

    if not solution.success:
        warnings.warn(f"Integration failed: {solution.message}")

    times = solution.t
    angles = solution.y[:n, :].T  # (n_timesteps x n_links)
    velocities = solution.y[n:, :].T

    # Normalize angles to [-pi, pi] for consistency
    angles = np.arctan2(np.sin(angles), np.cos(angles))

    return times, angles, velocities


def save_trajectory_csv(
    filepath: str,
    times: np.ndarray,
    angles: np.ndarray,
    velocities: np.ndarray,
    n_links: int
):
    """Save trajectory data to CSV file."""
    header = ['time'] + [f'theta_{i+1}' for i in range(n_links)] + [f'theta_dot_{i+1}' for i in range(n_links)]

    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for t_idx in range(len(times)):
            row = [times[t_idx]] + list(angles[t_idx]) + list(velocities[t_idx])
            writer.writerow(row)


def generate_dataset(
    n_links: int,
    n_trajectories: int = 100,
    n_timesteps: int = 1000,
    dt: float = 0.01,
    output_dir: str = 'pendulum_data',
    mass_range: Tuple[float, float] = (0.5, 2.0),
    length_range: Tuple[float, float] = (0.5, 1.5),
    damping_range: Tuple[float, float] = (0.01, 0.1),
    angle_range: Tuple[float, float] = (-np.pi/3, np.pi/3),
    velocity_range: Tuple[float, float] = (-1.0, 1.0),
    seed: int = 42,
    integration_method: str = 'DOP853',
    max_step: float = 0.001,
    verbose: bool = True
) -> List[TrajectoryMetadata]:
    """
    Generate a complete dataset of N-link pendulum trajectories.

    Args:
        n_links: Number of links in the pendulum
        n_trajectories: Number of trajectories to generate
        n_timesteps: Timesteps per trajectory
        dt: Time step [s]
        output_dir: Directory to save data
        mass_range: (min, max) mass for each link [kg]
        length_range: (min, max) length for each link [m]
        damping_range: (min, max) damping coefficient
        angle_range: (min, max) initial angle [rad]
        velocity_range: (min, max) initial angular velocity [rad/s]
        seed: Random seed for reproducibility
        integration_method: scipy integrator method
        max_step: Maximum integration step size
        verbose: Whether to show progress bar

    Returns:
        List of TrajectoryMetadata objects
    """
    rng = np.random.default_rng(seed)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    metadata_list = []

    iterator = range(n_trajectories)
    if verbose:
        iterator = tqdm(iterator, desc=f'Generating {n_links}-link pendulums')

    for traj_id in iterator:
        # Generate random parameters
        params = generate_random_parameters(
            n_links=n_links,
            mass_range=mass_range,
            length_range=length_range,
            damping_range=damping_range,
            rng=rng
        )

        # Generate random initial conditions
        initial_angles, initial_velocities = generate_random_initial_conditions(
            n_links=n_links,
            angle_range=angle_range,
            velocity_range=velocity_range,
            rng=rng
        )

        # Simulate trajectory
        times, angles, velocities = simulate_trajectory(
            params=params,
            initial_angles=initial_angles,
            initial_velocities=initial_velocities,
            dt=dt,
            n_timesteps=n_timesteps,
            integration_method=integration_method,
            max_step=max_step
        )

        # Save trajectory
        filename = f'trajectory_{traj_id:04d}.csv'
        filepath = os.path.join(output_dir, filename)
        save_trajectory_csv(filepath, times, angles, velocities, n_links)

        # Create metadata
        metadata = TrajectoryMetadata(
            trajectory_id=traj_id,
            n_links=n_links,
            masses=params.masses,
            lengths=params.lengths,
            damping=params.damping,
            initial_angles=initial_angles,
            initial_velocities=initial_velocities,
            dt=dt,
            n_timesteps=n_timesteps + 1,  # +1 for initial condition
            total_time=dt * n_timesteps,
            integration_method=integration_method,
            max_integration_step=max_step
        )
        metadata_list.append(metadata)

    # Save metadata
    metadata_filepath = os.path.join(output_dir, 'metadata.json')
    with open(metadata_filepath, 'w') as f:
        json.dump(
            {
                'dataset_info': {
                    'n_links': n_links,
                    'n_trajectories': n_trajectories,
                    'n_timesteps': n_timesteps + 1,
                    'dt': dt,
                    'total_time_per_trajectory': dt * n_timesteps,
                    'parameter_ranges': {
                        'mass_kg': list(mass_range),
                        'length_m': list(length_range),
                        'damping': list(damping_range),
                        'initial_angle_rad': list(angle_range),
                        'initial_velocity_rad_s': list(velocity_range)
                    },
                    'physics': {
                        'gravity_m_s2': G,
                        'link_model': 'uniform_rod',
                        'moment_of_inertia': 'I = (1/12) * m * l^2',
                        'dynamics_formulation': 'Lagrangian_mechanics'
                    },
                    'integration': {
                        'method': integration_method,
                        'max_step': max_step,
                        'rtol': 1e-8,
                        'atol': 1e-10
                    },
                    'seed': seed
                },
                'trajectories': [asdict(m) for m in metadata_list]
            },
            f,
            indent=2
        )

    if verbose:
        print(f"Saved {n_trajectories} trajectories to {output_dir}/")
        print(f"Metadata saved to {metadata_filepath}")

    return metadata_list


def compute_total_energy(
    params: PendulumParameters,
    angles: np.ndarray,
    velocities: np.ndarray
) -> np.ndarray:
    """
    Compute the total mechanical energy at each timestep.

    Useful for validating the integration accuracy (energy should decrease
    monotonically due to damping, or stay constant if damping is zero).

    Args:
        params: Pendulum parameters
        angles: Joint angles (n_timesteps x n_links)
        velocities: Angular velocities (n_timesteps x n_links)

    Returns:
        Array of total energies (n_timesteps,)
    """
    n = params.n_links
    m = np.array(params.masses)
    l = np.array(params.lengths)
    I_link = (1.0 / 12.0) * m * l**2

    n_timesteps = angles.shape[0]
    energies = np.zeros(n_timesteps)

    for t_idx in range(n_timesteps):
        theta = angles[t_idx]
        theta_dot = velocities[t_idx]

        T = 0.0  # Kinetic energy
        V = 0.0  # Potential energy

        # Compute position and velocity of each link's center of mass
        for i in range(n):
            # Position of link i's pivot
            x_pivot = sum(l[j] * np.sin(theta[j]) for j in range(i))
            y_pivot = -sum(l[j] * np.cos(theta[j]) for j in range(i))

            # Position of link i's center of mass
            x_cm = x_pivot + (l[i] / 2) * np.sin(theta[i])
            y_cm = y_pivot - (l[i] / 2) * np.cos(theta[i])

            # Velocity of link i's pivot
            x_dot_pivot = sum(l[j] * np.cos(theta[j]) * theta_dot[j] for j in range(i))
            y_dot_pivot = sum(l[j] * np.sin(theta[j]) * theta_dot[j] for j in range(i))

            # Velocity of link i's center of mass
            x_dot_cm = x_dot_pivot + (l[i] / 2) * np.cos(theta[i]) * theta_dot[i]
            y_dot_cm = y_dot_pivot + (l[i] / 2) * np.sin(theta[i]) * theta_dot[i]

            # Translational kinetic energy: (1/2) * m * v^2
            T += 0.5 * m[i] * (x_dot_cm**2 + y_dot_cm**2)

            # Rotational kinetic energy about CoM: (1/2) * I * omega^2
            T += 0.5 * I_link[i] * theta_dot[i]**2

            # Potential energy: m * g * y
            V += m[i] * G * y_cm

        energies[t_idx] = T + V

    return energies


def validate_trajectory(
    params: PendulumParameters,
    times: np.ndarray,
    angles: np.ndarray,
    velocities: np.ndarray,
    tolerance: float = 0.1
) -> dict:
    """
    Validate a trajectory by checking energy dissipation.

    With damping, energy should monotonically decrease.
    Without damping, energy should be conserved.

    Args:
        params: Pendulum parameters
        times: Time array
        angles: Joint angles
        velocities: Angular velocities
        tolerance: Maximum allowed energy increase (as fraction of initial)

    Returns:
        Dictionary with validation results
    """
    energies = compute_total_energy(params, angles, velocities)

    initial_energy = energies[0]
    final_energy = energies[-1]
    max_energy = np.max(energies)
    min_energy = np.min(energies)

    # Check for energy conservation violations
    energy_increases = np.diff(energies)
    max_increase = np.max(energy_increases)

    has_damping = any(b > 1e-10 for b in params.damping)

    if has_damping:
        # Energy should generally decrease
        valid = (max_energy - initial_energy) / abs(initial_energy) < tolerance
    else:
        # Energy should be conserved (within numerical precision)
        energy_variation = (max_energy - min_energy) / abs(initial_energy)
        valid = energy_variation < tolerance

    return {
        'valid': valid,
        'initial_energy': initial_energy,
        'final_energy': final_energy,
        'energy_dissipated': initial_energy - final_energy,
        'max_energy': max_energy,
        'min_energy': min_energy,
        'max_energy_increase': max_increase,
        'has_damping': has_damping
    }


def generate_all_datasets(
    base_dir: str = 'C:/Users/Jonaspetersen/dev/IndustrialJEPA/unify/data',
    n_values: List[int] = [1, 2, 3, 5],
    n_trajectories: int = 100,
    n_timesteps: int = 1000,
    dt: float = 0.01,
    seed: int = 42,
    verbose: bool = True
):
    """
    Generate datasets for all specified N-link pendulums.

    Args:
        base_dir: Base directory for datasets
        n_values: List of N values (number of links)
        n_trajectories: Trajectories per N value
        n_timesteps: Timesteps per trajectory
        dt: Time step [s]
        seed: Base random seed
        verbose: Whether to show progress
    """
    all_metadata = {}

    for n_links in n_values:
        if verbose:
            print(f"\n{'='*60}")
            print(f"Generating {n_links}-link pendulum dataset")
            print(f"{'='*60}")

        output_dir = os.path.join(base_dir, f'pendulum_n{n_links}')

        # Use different seed for each N to ensure independence
        dataset_seed = seed + n_links * 1000

        metadata = generate_dataset(
            n_links=n_links,
            n_trajectories=n_trajectories,
            n_timesteps=n_timesteps,
            dt=dt,
            output_dir=output_dir,
            seed=dataset_seed,
            verbose=verbose
        )

        all_metadata[f'n{n_links}'] = metadata

        if verbose:
            # Validate a few trajectories
            print(f"\nValidating sample trajectories...")
            sample_indices = [0, n_trajectories // 2, n_trajectories - 1]

            for idx in sample_indices:
                traj_path = os.path.join(output_dir, f'trajectory_{idx:04d}.csv')

                # Load trajectory
                data = np.genfromtxt(traj_path, delimiter=',', skip_header=1)
                times = data[:, 0]
                angles = data[:, 1:n_links+1]
                velocities = data[:, n_links+1:]

                # Get parameters
                m = metadata[idx]
                params = PendulumParameters(
                    n_links=m.n_links,
                    masses=m.masses,
                    lengths=m.lengths,
                    damping=m.damping
                )

                # Validate
                validation = validate_trajectory(params, times, angles, velocities)
                status = "PASS" if validation['valid'] else "FAIL"
                print(f"  Trajectory {idx}: {status} | "
                      f"E_initial={validation['initial_energy']:.4f} J | "
                      f"E_dissipated={validation['energy_dissipated']:.4f} J")

    # Save combined metadata
    combined_metadata_path = os.path.join(base_dir, 'dataset_summary.json')
    summary = {
        'description': 'N-Link Pendulum Dataset for Unify Project',
        'purpose': 'Cross-morphological latent alignment research',
        'n_values': n_values,
        'trajectories_per_n': n_trajectories,
        'timesteps_per_trajectory': n_timesteps + 1,
        'dt_seconds': dt,
        'total_time_per_trajectory': n_timesteps * dt,
        'total_trajectories': n_trajectories * len(n_values),
        'base_seed': seed,
        'directories': {f'n{n}': f'pendulum_n{n}/' for n in n_values},
        'physics_model': {
            'formulation': 'Lagrangian mechanics',
            'link_model': 'Uniform rod with I = (1/12)*m*l^2',
            'gravity': f'{G} m/s^2',
            'includes_damping': True
        }
    }

    with open(combined_metadata_path, 'w') as f:
        json.dump(summary, f, indent=2)

    if verbose:
        print(f"\n{'='*60}")
        print(f"Dataset generation complete!")
        print(f"Summary saved to: {combined_metadata_path}")
        print(f"{'='*60}")

    return all_metadata


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Generate N-Link Pendulum datasets for Unify project'
    )
    parser.add_argument(
        '--base-dir',
        type=str,
        default='C:/Users/Jonaspetersen/dev/IndustrialJEPA/unify/data',
        help='Base directory for output'
    )
    parser.add_argument(
        '--n-values',
        type=int,
        nargs='+',
        default=[1, 2, 3, 5],
        help='Number of links to generate (default: 1 2 3 5)'
    )
    parser.add_argument(
        '--n-trajectories',
        type=int,
        default=100,
        help='Number of trajectories per N value (default: 100)'
    )
    parser.add_argument(
        '--n-timesteps',
        type=int,
        default=1000,
        help='Timesteps per trajectory (default: 1000)'
    )
    parser.add_argument(
        '--dt',
        type=float,
        default=0.01,
        help='Time step in seconds (default: 0.01)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress progress output'
    )

    args = parser.parse_args()

    generate_all_datasets(
        base_dir=args.base_dir,
        n_values=args.n_values,
        n_trajectories=args.n_trajectories,
        n_timesteps=args.n_timesteps,
        dt=args.dt,
        seed=args.seed,
        verbose=not args.quiet
    )
