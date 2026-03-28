# Dataset Screening: Awesome-Industrial-Datasets

Source: [jonathanwvd/awesome-industrial-datasets](https://github.com/jonathanwvd/awesome-industrial-datasets) (176 datasets)

## Selection Criteria for Unify

✅ **Must have:**
- Multivariate time-series (not images/tabular)
- Mechanical/physical system with sensors
- Multiple instances or operating conditions
- Publicly accessible

🎯 **Ideal for cross-morphological alignment:**
- Varying number of features across instances
- Shared underlying physics
- Real data preferred (synthetic OK for ablation)

---

## GROUP A: Rotating Machinery / Bearings

**Shared Physics:** Rotational dynamics, vibration signatures, fault frequency patterns

| Dataset | Features | Instances | Sampling | Real? | Notes |
|---------|----------|-----------|----------|-------|-------|
| **CWRU Bearing** | 2-4 (vib) | ~500 files | 12-48 kHz | ✅ | Gold standard, DE/FE accelerometers |
| **IMS/NASA Bearing** | 4-8 (vib) | 3 runs | 20 kHz | ✅ | Run-to-failure, 4 bearings |
| **FEMTO/PRONOSTIA** | 2 (vib) | 17 bearings | 25.6 kHz | ✅ | IEEE PHM challenge |
| **Paderborn Bearing** | 8 (vib+current+mech) | 33 conditions | 64 kHz | ✅ | **BEST:** multi-modal |
| **Gearbox Fault** | 4+ (accel) | multiple | varies | ✅ | Transmission faults |

**Unify Potential:** ⭐⭐⭐ Good — can train on 2-4 ch (CWRU), test zero-shot on 8 ch (Paderborn)

---

## GROUP B: Turbomachinery / Engines

**Shared Physics:** Thermodynamic cycles, compressor/turbine dynamics, degradation

| Dataset | Features | Instances | Sampling | Real? | Notes |
|---------|----------|-----------|----------|-------|-------|
| **C-MAPSS (FD001-4)** | 21 sensors | 100-260 units | per-cycle | ❌ Synth | NASA turbofan, RUL benchmark |
| **N-CMAPSS** | 20+ sensors | ~100 units | 1 Hz | ❌ Synth | New CMAPSS, flight-level data |
| **PHM 2008 Challenge** | 21 sensors | ~200 units | per-cycle | ❌ Synth | Same as C-MAPSS |
| **Naval Propulsion** | 16 features | 11,934 | - | ❌ Synth | Gas turbine decay |
| **Diesel Engine Faults** | pressure+vib | multiple | varies | ❌ Synth | Injection faults |

**Unify Potential:** ⭐⭐ Moderate — synthetic but well-studied; good for controlled ablation

---

## GROUP C: CNC / Manufacturing

**Shared Physics:** Cutting mechanics, tool wear, spindle dynamics

| Dataset | Features | Instances | Sampling | Real? | Notes |
|---------|----------|-----------|----------|-------|-------|
| **CNC Mill Tool Wear** | 48 sensors | 18 cases | 100 Hz | ✅ | UC Berkeley, force+vib+acoustic |
| **Milling Wear** | 6-9 sensors | multiple | varies | ✅ | Acoustic+force+vib |
| **One Year Degradation** | varies | 1 blade | daily | ✅ | Cutting blade wear |
| **Laser Welding** | process params | 50+ | - | ✅ | Steel-copper joints |

**Unify Potential:** ⭐⭐⭐ Good — CNC shares cutting physics; varying sensor configs

---

## GROUP D: Hydraulic / Fluid Systems

**Shared Physics:** Pressure dynamics, flow conservation, valve behavior

| Dataset | Features | Instances | Sampling | Real? | Notes |
|---------|----------|-----------|----------|-------|-------|
| **Hydraulic System (UCI)** | 17 sensors | 2,205 cycles | 100/10/1 Hz | ✅ | Cooler+valve+pump+accumulator |
| **DAMADICS Actuator** | 32 sensors | multiple | 1 Hz | ✅ | Sugar factory control valves |
| **3W Oil Wells** | 8 sensors | 1,984 instances | varies | ✅ | Petrobras offshore wells |

**Unify Potential:** ⭐⭐⭐ Good — hydraulics share pressure/flow physics

---

## GROUP E: Process Control / Chemical

**Shared Physics:** Mass/energy balance, reaction kinetics, control loops

| Dataset | Features | Instances | Sampling | Real? | Notes |
|---------|----------|-----------|----------|-------|-------|
| **Tennessee Eastman** | 52 sensors | 22 faults | varies | ❌ Synth | Chemical process benchmark |
| **IndPenSim** | 15+ sensors | 100 batches | varies | ❌ Synth | Penicillin fermentation |
| **ISDB (Stiction)** | varies | 100+ loops | varies | ✅ | Control loop oscillations |

**Unify Potential:** ⭐⭐ Moderate — process systems are complex but well-structured

---

## GROUP F: Robotics / Automation

**Shared Physics:** Rigid-body dynamics, joint kinematics, motor control

| Dataset | Features | Instances | Sampling | Real? | Notes |
|---------|----------|-----------|----------|-------|-------|
| **AURSAD** | 20 ch | 2,045 episodes | 500 Hz | ✅ | UR3e screwdriving |
| **Voraus-AD** | 66 ch | ~2,000 episodes | 100 Hz | ✅ | Yu-Cobot pick-place |
| **Degradation Robot Arm** | varies | multiple | varies | ✅ | UR5 position accuracy |
| **Genesis Pick-Place** | pneumatic | varies | varies | ✅ | Pneumatic drive anomalies |
| **OXE (KUKA/Panda)** | 18-27 ch | 3,000+ | varies | ✅ | Multi-embodiment |

**Unify Potential:** ⭐⭐⭐⭐⭐ **BEST** — direct analog to N-link pendulum concept

---

## GROUP G: Energy / Power Systems

**Shared Physics:** Electrical dynamics, load balancing, thermal coupling

| Dataset | Features | Instances | Sampling | Real? | Notes |
|---------|----------|-----------|----------|-------|-------|
| **ETT (Transformer)** | 7 ch | 4 datasets | 15min/1hr | ✅ | Oil temperature forecasting |
| **Appliances Energy** | 29 sensors | 4.5 months | 10 min | ✅ | Building energy |
| **GREEND** | per-device | 8 households | 1 Hz | ✅ | Austria/Italy |
| **Electrical Grid Stability** | 12 features | 10,000 | - | ❌ Synth | Decentralized control |

**Unify Potential:** ⭐⭐ Moderate — electrical dynamics differ from mechanical

---

## GROUP H: Water / Infrastructure (SCADA)

**Shared Physics:** Fluid networks, pump dynamics, pressure propagation

| Dataset | Features | Instances | Sampling | Real? | Notes |
|---------|----------|-----------|----------|-------|-------|
| **SWaT** | 51 sensors | 11 days | 1 Hz | ✅ | Water treatment (gated) |
| **WADI** | 127 sensors | 16 days | 1 Hz | ✅ | Water distribution (gated) |
| **BATADAL** | 43 sensors | 1 year | hourly | ❌ Synth | Cyber-attack detection |

**Unify Potential:** ⭐⭐⭐ Good if access granted — high channel count, real physics

---

## Summary: Priority Datasets for Unify

### Tier 1: Start Here (Cross-Morphological Robotics)

| Dataset | Ch | DOF | Why |
|---------|-----|-----|-----|
| AURSAD | 20 | 6 | Real robot, good size |
| Voraus-AD | 66 | 6 | Same DOF, 3x channels |
| OXE-KUKA | 27 | **7** | Zero-shot DOF test |

### Tier 2: Extend (Varying Sensor Configs)

| Dataset | Ch | Why |
|---------|-----|-----|
| Paderborn | 8 | Multi-modal bearing (vib+current+mech) |
| CWRU | 2-4 | Vib-only bearing (simpler) |
| Hydraulic | 17 | 4 physics groups (pressure/flow/thermal/mech) |
| CNC Mill | 48 | Rich sensor config |

### Tier 3: Ablation (Synthetic Control)

| Dataset | Ch | Why |
|---------|-----|-----|
| C-MAPSS | 21 | Well-studied, controlled |
| TEP | 52 | Complex process, many channels |
| Naval Propulsion | 16 | Thermodynamic system |

---

## Cross-Group Transfer Tests

| Train On | Test On (Zero-Shot) | Shared Physics |
|----------|---------------------|----------------|
| CWRU (2ch) + Paderborn (8ch) | IMS (4ch) | Rotational/vibration |
| AURSAD (20ch) + Voraus (66ch) | OXE-KUKA (27ch, 7-DOF) | Rigid-body dynamics |
| Hydraulic (17ch) | DAMADICS (32ch) | Pressure/flow |
| C-MAPSS (21ch) | Naval (16ch) | Thermodynamics |

---

## Quick Links

| Resource | URL |
|----------|-----|
| awesome-industrial-datasets | https://github.com/jonathanwvd/awesome-industrial-datasets |
| GIFT-Eval Benchmark | https://github.com/SalesforceAIResearch/gift-eval |
| TabPFN-TS | https://github.com/PriorLabs/tabpfn-time-series |
| Chronos-2 | https://github.com/amazon-science/chronos-forecasting |
| GluonTS | https://ts.gluon.ai/ |
