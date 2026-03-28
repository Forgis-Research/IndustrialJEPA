# Dataset Analysis for Unify Project

## Selection Criteria

For cross-morphological latent alignment, we need datasets that:
1. **Multiple instances** sharing underlying physics
2. **Varying feature dimensions** across instances (or at least configurations)
3. **Real mechanical systems** (preferred over synthetic)
4. **Sufficient data** for training dynamics models

---

## Tier 1: BEST FIT — Robotics Cluster (Multi-Embodiment)

This is the **direct analog to the N-link pendulum POC** — different robot morphologies with shared rigid-body dynamics.

| Dataset | Robot | DOF | Channels | Episodes | Sampling |
|---------|-------|-----|----------|----------|----------|
| **AURSAD** | UR3e | 6 | 20 | 2,045 | 500 Hz |
| **Voraus-AD** | Yu-Cobot | 6 | 66 | ~2,000 | 100 Hz |
| **OXE-KUKA** | KUKA iiwa | 7 | 27+ | 3,000 | varies |
| **OXE-ManiSkill** | Panda | 7 | 18 | 30,213 | varies |
| **OXE-UR5** | UR5 | 6 | 15 | 896 | varies |

### Why This Works for Unify

```
           AURSAD (6-DOF, 20ch)     Voraus-AD (6-DOF, 66ch)     KUKA (7-DOF, 27ch)
                    \                        |                        /
                     \                       |                       /
                      ↘                      ↓                      ↙
                    ┌─────────────────────────────────────────────────┐
                    │         SET-BASED ENCODER                       │
                    │   (handles 15-66 channels via token embedding)  │
                    └─────────────────────────────────────────────────┘
                                         │
                                         ↓
                    ┌─────────────────────────────────────────────────┐
                    │      SHARED LATENT SPACE (Fixed D=128?)         │
                    │         "Universal Robot Dynamics"              │
                    └─────────────────────────────────────────────────┘
                                         │
                                         ↓
                    ┌─────────────────────────────────────────────────┐
                    │           LATENT DYNAMICS MODEL                 │
                    │    (learns multi-body mechanics, not stats)     │
                    └─────────────────────────────────────────────────┘
```

### Shared Physics Structure

All robots share:
- **Joint kinematics**: position → velocity → acceleration
- **Effort dynamics**: torque/current → motion
- **Cartesian coupling**: joint space ↔ end-effector space
- **Inertial properties**: mass, damping, friction

### Test Protocol

| Training | Zero-Shot Test | Expected Outcome |
|----------|----------------|------------------|
| UR3e (6-DOF) + Yu-Cobot (6-DOF) | KUKA (7-DOF) | Transfer to unseen DOF count |
| AURSAD (20ch) + Voraus (66ch) | OXE-UR5 (15ch) | Transfer to unseen sensor config |
| All 6-DOF robots | 7-DOF robots | Extrapolate joint dynamics |

### Data Availability
- ✅ AURSAD: Public (HuggingFace)
- ✅ Voraus-AD: Public (GitHub)
- ✅ OXE: Public (TensorFlow Datasets)

**Verdict: START HERE** — closest to N-link pendulum concept with real data

---

## Tier 2: GOOD FIT — Bearing Cluster (Rotational Dynamics)

Different bearing configurations sharing rotational/vibrational physics.

| Dataset | Channels | Instances | Modalities | Access |
|---------|----------|-----------|------------|--------|
| **Paderborn** | 8 | 33 conditions | Vib + Current + Mech + Thermal | ✅ Public |
| **CWRU** | 2-4 | ~500 files | Vibration only | ✅ Public |
| **IMS** | 4-8 | 3 runs | Vibration only | ✅ Public |
| **XJTU-SY** | 2 | 15 bearings | Vibration only | ✅ Public |

### Why It Works

```
        Paderborn (8ch)        CWRU (4ch)         IMS (8ch)
        Vib+Current+Mech       Vib only           Vib only
                \                  |                 /
                 ↘                 ↓                ↙
              ┌─────────────────────────────────────────┐
              │   SET ENCODER (2-8 sensors as tokens)   │
              └─────────────────────────────────────────┘
                               │
                               ↓
              ┌─────────────────────────────────────────┐
              │   SHARED LATENT: Rotational Dynamics    │
              │   (bearing fault physics are universal) │
              └─────────────────────────────────────────┘
```

### Shared Physics
- Rotational kinematics (speed, position)
- Vibrational signatures (characteristic frequencies)
- Fault progression dynamics (wear → failure)

### Limitation
- Less morphological variation (all are "rotating machines")
- Channel variation is more about sensor types than system structure
- Primarily useful for condition monitoring, not trajectory forecasting

**Verdict: GOOD SECONDARY OPTION** — useful for proving sensor-agnostic encoding

---

## Tier 3: INTERESTING BUT WEAKER — Process Systems

| Dataset | Channels | Physics | Challenge |
|---------|----------|---------|-----------|
| **Hydraulic** | 17 | Fluid dynamics | Single system type |
| **SWaT** | 51 | Water treatment | Complex chemical + flow |
| **WADI** | 127 | Water distribution | Network dynamics |
| **TEP** | 52 | Chemical process | Synthetic |

### Why Weaker for Unify
- Less clear "morphological" variation
- Physics is more heterogeneous (chemical + flow + thermal)
- Harder to argue "shared underlying dynamics"

**Verdict: SKIP FOR NOW** — may revisit for "unified process control" variant

---

## Recommended Experimental Plan

### Phase 1: Sanity Check with Synthetic (N-Link Pendulum)
- Generate N=1,2,3,5 pendulum data
- Prove the architecture works with controlled variation
- Establish baselines

### Phase 2: Real-World Validation with Robotics
```
Step 1: AURSAD only (baseline)
        ↓
Step 2: AURSAD + Voraus-AD (cross-robot pretraining)
        ↓
Step 3: Test zero-shot on OXE-KUKA (7-DOF, never seen)
        ↓
Step 4: Compare vs supervised baseline on each dataset
```

### Phase 3: Extension to Bearings (Optional)
- Train on Paderborn (rich sensors)
- Test on CWRU (fewer sensors) zero-shot

---

## Summary: Dataset Recommendations

| Priority | Cluster | Datasets | Why |
|----------|---------|----------|-----|
| **1 (Start)** | Robotics | AURSAD, Voraus-AD, OXE | Direct analog to N-link; varying DOF + channels |
| **2 (Extend)** | Bearings | Paderborn, CWRU | Prove sensor-agnostic encoding |
| **3 (Later)** | Synthetic | N-link pendulums | Controlled ablation studies |
| **Skip** | Process | SWaT, TEP, Hydraulic | Weaker morphological argument |

---

---

## Concrete Validation Protocol for Robotics

### Training Setup
```
Dataset 1: AURSAD    → 2,045 episodes × 20 channels  (UR3e, 6-DOF)
Dataset 2: Voraus-AD → 2,000 episodes × 66 channels  (Yu-Cobot, 6-DOF)
           ─────────────────────────────────────────
           Total:     ~4,000 episodes, mixed channel counts
```

### Test Matrix

| Test | Data | What It Proves |
|------|------|----------------|
| **In-Dist AURSAD** | Held-out UR3e | No regression vs per-dataset model |
| **In-Dist Voraus** | Held-out Yu-Cobot | Handles 66ch as well as 20ch |
| **Cross-Robot** | Train UR3e → Test Yu-Cobot | Same DOF, different sensors |
| **Zero-Shot 7-DOF** | Train 6-DOF → Test KUKA | **THE KEY TEST** |

### Success Criteria (Normalized MSE)

```
                        In-Distribution    Zero-Shot 7-DOF
                        ───────────────    ───────────────
Per-Dataset Baseline:        1.0               FAILS (wrong input dim)
Concat-Padded Baseline:      1.1               >3.0 (poor)
Our Unified Model:           ≤1.05             <2.5 (meaningful)
```

### Why This Is Hard (And Interesting)

1. **Input shape varies**: 20 vs 66 vs 27 channels — standard models can't handle this
2. **DOF extrapolation**: Model must "imagine" 7th joint it never saw
3. **Sensor semantics differ**: AURSAD has current, Voraus has temperature — encoder must learn invariances

---

## Key Insight

The **robotics cluster** is ideal because:

1. **Clear morphological variation**: 6-DOF vs 7-DOF is analogous to 3-link vs 4-link pendulum
2. **Varying sensor counts**: 15 to 66 channels across datasets
3. **Shared physics**: All robots obey Lagrangian mechanics
4. **Existing infrastructure**: FactoryNet already unifies AURSAD + Voraus-AD
5. **Zero-shot test is meaningful**: Can a model trained on UR3e predict KUKA dynamics?

This directly maps to the research question:
> *"Train on 2-joint and 3-joint systems to achieve superior performance on a never-seen 4-joint system"*

Becomes:
> *"Train on 6-DOF robots (UR3e, Yu-Cobot) to achieve superior performance on never-seen 7-DOF robot (KUKA, Panda)"*
