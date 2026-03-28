# N-Link Pendulum Dataset

Synthetic dataset for the Unify research project on cross-morphological latent alignment.

## Overview

This dataset contains trajectories of N-link pendulums with N in {1, 2, 3, 5}, designed to test the hypothesis that diverse mechanical systems can be mapped to a shared latent physics space.

## Dataset Structure

```
unify/data/
├── pendulum_generator.py    # Dataset generation script
├── dataset_summary.json     # Combined metadata for all datasets
├── README.md               # This file
├── pendulum_n1/            # 1-link (simple) pendulum
│   ├── trajectory_0000.csv
│   ├── trajectory_0001.csv
│   ├── ...
│   └── metadata.json
├── pendulum_n2/            # 2-link (double) pendulum
│   ├── trajectory_0000.csv
│   ├── ...
│   └── metadata.json
├── pendulum_n3/            # 3-link pendulum
│   ├── trajectory_0000.csv
│   ├── ...
│   └── metadata.json
└── pendulum_n5/            # 5-link pendulum (zero-shot test)
    ├── trajectory_0000.csv
    ├── ...
    └── metadata.json
```

## Physics Model

### Lagrangian Mechanics Formulation

The equations of motion are derived from the Lagrangian:

```
L = T - V
```

where:
- **T** = Kinetic energy (translational + rotational)
- **V** = Potential energy (gravitational)

For an N-link pendulum with generalized coordinates **q** = [theta_1, ..., theta_N]:

```
M(q) * q_ddot + C(q, q_dot) * q_dot + G(q) = -B * q_dot
```

where:
- **M(q)**: Mass matrix (N x N) - configuration-dependent inertia
- **C(q, q_dot)**: Coriolis/centrifugal matrix - velocity-dependent terms
- **G(q)**: Gravity vector - gravitational torques
- **B**: Damping matrix - energy dissipation

### Link Model

Each link is modeled as a uniform rod:
- Mass concentrated at center of mass (l/2 from pivot)
- Moment of inertia: I = (1/12) * m * l^2 (about center)
- Full rigid-body dynamics with parallel axis theorem

### Coordinate Convention

- **theta = 0**: Link hanging straight down (stable equilibrium)
- **theta > 0**: Counter-clockwise rotation from vertical
- **theta < 0**: Clockwise rotation from vertical
- Angles are normalized to [-pi, pi]

## Data Format

### CSV Files

Each trajectory file contains:

| Column | Description | Unit |
|--------|-------------|------|
| `time` | Simulation time | seconds |
| `theta_1` | Angle of link 1 | radians |
| `theta_2` | Angle of link 2 (if N >= 2) | radians |
| ... | ... | ... |
| `theta_N` | Angle of link N | radians |
| `theta_dot_1` | Angular velocity of link 1 | rad/s |
| `theta_dot_2` | Angular velocity of link 2 (if N >= 2) | rad/s |
| ... | ... | ... |
| `theta_dot_N` | Angular velocity of link N | rad/s |

### Metadata JSON

Each `metadata.json` contains:

```json
{
  "dataset_info": {
    "n_links": 2,
    "n_trajectories": 100,
    "n_timesteps": 1001,
    "dt": 0.01,
    "total_time_per_trajectory": 10.0,
    "parameter_ranges": {
      "mass_kg": [0.5, 2.0],
      "length_m": [0.5, 1.5],
      "damping": [0.01, 0.1],
      "initial_angle_rad": [-1.047, 1.047],
      "initial_velocity_rad_s": [-1.0, 1.0]
    },
    "physics": {
      "gravity_m_s2": 9.81,
      "link_model": "uniform_rod",
      "moment_of_inertia": "I = (1/12) * m * l^2",
      "dynamics_formulation": "Lagrangian_mechanics"
    }
  },
  "trajectories": [
    {
      "trajectory_id": 0,
      "n_links": 2,
      "masses": [1.23, 0.87],
      "lengths": [0.95, 1.12],
      "damping": [0.045, 0.032],
      "initial_angles": [0.52, -0.31],
      "initial_velocities": [0.12, -0.45],
      ...
    },
    ...
  ]
}
```

## Generation Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| N values | {1, 2, 3, 5} | Number of links |
| Trajectories per N | 100 | Dataset size |
| Timesteps | 1001 | Including initial condition |
| dt | 0.01 s | Sampling interval |
| Total time | 10.0 s | Per trajectory |
| Mass range | [0.5, 2.0] kg | Per link |
| Length range | [0.5, 1.5] m | Per link |
| Damping range | [0.01, 0.1] | Per joint |
| Initial angle range | [-pi/3, pi/3] rad | Moderate displacements |
| Initial velocity range | [-1.0, 1.0] rad/s | Moderate velocities |
| Gravity | 9.81 m/s^2 | Earth standard |
| Integration method | DOP853 | 8th order Dormand-Prince |
| Integration rtol | 1e-8 | Relative tolerance |
| Integration atol | 1e-10 | Absolute tolerance |

## Usage

### Loading Data (Python)

```python
import numpy as np
import json
import os

def load_trajectory(data_dir, n_links, trajectory_id):
    """Load a single trajectory and its parameters."""
    traj_path = os.path.join(data_dir, f'pendulum_n{n_links}',
                             f'trajectory_{trajectory_id:04d}.csv')
    meta_path = os.path.join(data_dir, f'pendulum_n{n_links}', 'metadata.json')

    # Load trajectory data
    data = np.genfromtxt(traj_path, delimiter=',', skip_header=1)
    times = data[:, 0]
    angles = data[:, 1:n_links+1]
    velocities = data[:, n_links+1:]

    # Load metadata
    with open(meta_path) as f:
        metadata = json.load(f)
    params = metadata['trajectories'][trajectory_id]

    return {
        'times': times,
        'angles': angles,
        'velocities': velocities,
        'masses': params['masses'],
        'lengths': params['lengths'],
        'damping': params['damping']
    }

# Example: Load trajectory 42 from 3-link pendulum dataset
traj = load_trajectory('unify/data', n_links=3, trajectory_id=42)
print(f"Shape: {traj['angles'].shape}")  # (1001, 3)
```

### Loading for PyTorch

```python
import torch
from torch.utils.data import Dataset

class PendulumDataset(Dataset):
    """Dataset for N-link pendulum trajectories."""

    def __init__(self, data_dir, n_links, seq_len=100, stride=10):
        self.data_dir = data_dir
        self.n_links = n_links
        self.seq_len = seq_len
        self.stride = stride

        # Load all trajectories
        meta_path = os.path.join(data_dir, f'pendulum_n{n_links}', 'metadata.json')
        with open(meta_path) as f:
            self.metadata = json.load(f)

        n_traj = self.metadata['dataset_info']['n_trajectories']
        n_timesteps = self.metadata['dataset_info']['n_timesteps']

        # Create index mapping
        self.samples = []
        for traj_id in range(n_traj):
            for start in range(0, n_timesteps - seq_len, stride):
                self.samples.append((traj_id, start))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        traj_id, start = self.samples[idx]

        # Load trajectory
        traj_path = os.path.join(
            self.data_dir, f'pendulum_n{self.n_links}',
            f'trajectory_{traj_id:04d}.csv'
        )
        data = np.genfromtxt(traj_path, delimiter=',', skip_header=1)

        # Extract sequence
        seq = data[start:start+self.seq_len, 1:]  # Exclude time column

        # Get parameters
        params = self.metadata['trajectories'][traj_id]

        return {
            'sequence': torch.tensor(seq, dtype=torch.float32),
            'n_links': self.n_links,
            'masses': torch.tensor(params['masses'], dtype=torch.float32),
            'lengths': torch.tensor(params['lengths'], dtype=torch.float32),
        }
```

## Research Protocol

### Training Set
- N = 1, 2, 3 pendulums
- 100 trajectories each
- Total: 300 trajectories

### Validation Set
- Held-out instances from N = 1, 2, 3
- Use train/val split on trajectory IDs

### Zero-Shot Test Set
- N = 5 pendulums (never seen during training)
- 100 trajectories
- Tests cross-morphology generalization

## Physical Interpretation

### Why N-Link Pendulums?

1. **Shared physics**: All configurations obey Lagrangian mechanics
2. **Scalable complexity**: N=1 is simple harmonic motion; N=5 is chaotic
3. **Natural hierarchy**: N-link contains the dynamics of (N-1)-link as a special case
4. **Dimensional variation**: State dimension scales as 2N (angles + velocities)

### Energy Conservation Validation

The generator validates each trajectory by checking:
- Energy monotonically decreases (with damping)
- No spurious energy increases from numerical errors
- Total dissipation matches expected damping losses

### Chaotic Behavior (N >= 2)

For N >= 2, the system exhibits deterministic chaos:
- Sensitive dependence on initial conditions
- Positive Lyapunov exponents
- Unpredictable long-term behavior

This makes forecasting challenging and tests the model's ability to capture nonlinear dynamics.

## Re-generating the Dataset

```bash
cd unify/data

# Generate all datasets (default parameters)
python pendulum_generator.py

# Custom generation
python pendulum_generator.py \
    --n-values 1 2 3 5 \
    --n-trajectories 100 \
    --n-timesteps 1000 \
    --dt 0.01 \
    --seed 42
```

## References

1. Murray, R. M., Li, Z., & Sastry, S. S. (1994). *A Mathematical Introduction to Robotic Manipulation*. CRC Press.

2. Spong, M. W., Hutchinson, S., & Vidyasagar, M. (2006). *Robot Modeling and Control*. Wiley.

3. Strogatz, S. H. (2015). *Nonlinear Dynamics and Chaos*. CRC Press.

## Citation

If you use this dataset, please cite the Unify project:

```bibtex
@misc{unify2024pendulum,
  title={N-Link Pendulum Dataset for Cross-Morphological Physics Learning},
  author={Unify Research Project},
  year={2024},
  howpublished={\url{https://github.com/...}}
}
```
