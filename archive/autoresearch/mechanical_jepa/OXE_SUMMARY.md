# OXE Dataset Summary for Mechanical-JEPA

Quick reference for the datasets we're using. Full details in `datasets/catalog/open_x_embodiment.md`.

---

## Pretraining Datasets (7-DOF)

### DROID (Primary)
```python
tfds_name = 'droid_100'
robot = 'Franka Panda'
episodes = 76_000
hz = 15

state_fields = {
    'joint_position': (7,),      # Joint angles
    'cartesian_position': (6,),  # EE pose
    'gripper_position': (1,),    # Gripper
}
# Total: 14 dims

action = (7,)  # EE space: [cartesian(6), gripper(1)]
```

### ManiSkill (Cleanest)
```python
tfds_name = 'maniskill_dataset_converted_externally_to_rlds'
robot = 'Franka Panda (sim)'
episodes = 30_213
hz = 20

state_fields = {
    'state': (18,),  # joints(7) + gripper(2) + velocities(9)
}

action = (7,)  # [EE delta pos(3), EE delta orient(3), gripper(1)]
action_state_corr = 0.473  # Highest — best for action-conditioned
```

### Stanford KUKA (Force Data)
```python
tfds_name = 'stanford_kuka_multimodal_dataset_converted_externally_to_rlds'
robot = 'KUKA iiwa'
episodes = 3_000
hz = 20

state_fields = {
    'joint_pos': (7,),
    'joint_vel': (7,),
    'ee_position': (3,),
    'ee_vel': (3,),
    'ee_orientation': (4,),
    'ee_forces_continuous': (50, 6),  # Unique! Contact forces
    'contact': (50,),
}
# Total: 27+ dims (without force history)

action = (4,)  # [EE pos delta(3), gripper(1)]
```

---

## Transfer Target Datasets (6-DOF)

### Berkeley UR5
```python
tfds_name = 'berkeley_autolab_ur5'
robot = 'UR5'
dof = 6
episodes = 896
state = 'robot_state': (15,)  # Likely: joints(6) + vels(6) + EE(3)
```

### Berkeley FANUC
```python
tfds_name = 'berkeley_fanuc_manipulation'
robot = 'FANUC Mate 200iD'
dof = 6
episodes = 415
state = 'state': (13,)
```

### JACO Play
```python
tfds_name = 'jaco_play'
robot = 'Kinova JACO'
dof = 6
episodes = 976
state_fields = {
    'joint_pos': (8,),  # 6 joints + 2 fingers
    'end_effector_cartesian_pos': (7,),
    'end_effector_cartesian_velocity': (6,),
}
```

### TOTO (Same Robot, Different Task)
```python
tfds_name = 'toto'
robot = 'Franka Panda'
dof = 7
episodes = 902
state = 'state': (7,)  # Joint angles only
```

---

## Data Loading

```python
import tensorflow_datasets as tfds

# Load dataset
ds = tfds.load(
    'maniskill_dataset_converted_externally_to_rlds',
    data_dir='gs://gresearch/robotics',
    split='train[:1000]'  # First 1000 episodes
)

# Extract proprioception
for episode in ds:
    for step in episode['steps']:
        state = step['observation']['state'].numpy()
        action = step['action'].numpy()
```

---

## State Normalization

```python
# Recommended: per-channel z-score
# Compute on training set only

def normalize(data, mean, std):
    return (data - mean) / (std + 1e-8)

# Clip outliers first
data = np.clip(data, mean - 5*std, mean + 5*std)
data = normalize(data, mean, std)
```

---

## Action Handling

**All actions are EE-space deltas.** Options:

| Strategy | Code |
|----------|------|
| Raw EE | `action_embed = Linear(action_dim, d_model)(action)` |
| Normalized | `action = action / action_std` |
| With state | `combined = concat(state_embed, action_embed)` |

Start with raw EE, ablate later.

---

## DOF Alignment

For 6-DOF robots:

```python
def align_to_7dof(state_6dof):
    # Zero-pad to 7 joints
    return np.pad(state_6dof, (0, 1), mode='constant')

# In attention: mask the padding position
```

---

## Episode Lengths

| Dataset | Min | Max | Mean | Std |
|---------|-----|-----|------|-----|
| TOTO | 229 | 1160 | 361 | 210 |
| ManiSkill | 124 | 187 | 154 | 17 |
| FANUC | 27 | 235 | 123 | 60 |
| UR5 | 70 | 123 | 95 | 17 |
| JACO | 35 | 115 | 74 | 22 |
| KUKA | 50 | 50 | 50 | 0 |
| DROID | ~166 | -- | -- | -- |

Use window_size=128 for training, shorter for sanity checks.
