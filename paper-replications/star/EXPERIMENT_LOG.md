# STAR Replication Experiment Log

Paper: "STAR: Spatio-Temporal Attention-based Regression for Turbofan RUL Prediction"
Fan et al., Sensors 2024
Replication started: 2026-04-11

---

## Context

- Hardware: NVIDIA A10G (23 GB), but shared with 6-7 other concurrent processes
- Effective training speed: ~33 min/seed for FD001 (3.7M params, 15K windows)
- Estimated total runtime: FD001 ~2.75h, FD002 ~2.5h, FD003 ~1.7h, FD004 ~5-7.5h
- Session started: 2026-04-11T12:06 UTC

---

## Log


### 2026-04-11T12:39:12 | FD001 seed=42

- **Hyperparams**: lr=0.0002,bs=32,w=32,scales=3,dm=128,nh=1
- **Parameters**: 3,676,992
- **Epochs run**: 39 (best epoch 19)
- **Best val RMSE**: 13.150
- **Test RMSE**: 12.301 (paper target: 10.61)
- **Test Score**: 253.2 (paper target: 169)
- **Wall time**: 1980s
- **Notes**: none


### 2026-04-11T13:55:52 | FD001 seed=42

- **Hyperparams**: lr=0.0002,bs=32,w=32,scales=3,dm=128,nh=1
- **Parameters**: 3,676,992
- **Epochs run**: 39 (best epoch 19)
- **Best val RMSE**: 13.150
- **Test RMSE**: 12.301 (paper target: 10.61)
- **Test Score**: 253.2 (paper target: 169)
- **Wall time**: 1832s
- **Notes**: none


### 2026-04-11T14:29:30 | FD001 seed=123

- **Hyperparams**: lr=0.0002,bs=32,w=32,scales=3,dm=128,nh=1
- **Parameters**: 3,676,992
- **Epochs run**: 54 (best epoch 34)
- **Best val RMSE**: 11.396
- **Test RMSE**: 12.969 (paper target: 10.61)
- **Test Score**: 286.7 (paper target: 169)
- **Wall time**: 2018s
- **Notes**: none


### 2026-04-11T15:00:44 | FD001 seed=456

- **Hyperparams**: lr=0.0002,bs=32,w=32,scales=3,dm=128,nh=1
- **Parameters**: 3,676,992
- **Epochs run**: 41 (best epoch 21)
- **Best val RMSE**: 11.316
- **Test RMSE**: 11.250 (paper target: 10.61)
- **Test Score**: 196.2 (paper target: 169)
- **Wall time**: 1874s
- **Notes**: none


### 2026-04-11T15:27:45 | FD001 seed=789

- **Hyperparams**: lr=0.0002,bs=32,w=32,scales=3,dm=128,nh=1
- **Parameters**: 3,676,992
- **Epochs run**: 61 (best epoch 41)
- **Best val RMSE**: 12.140
- **Test RMSE**: 12.095 (paper target: 10.61)
- **Test Score**: 252.8 (paper target: 169)
- **Wall time**: 1621s
- **Notes**: none


### 2026-04-11T15:44:17 | FD001 seed=1024

- **Hyperparams**: lr=0.0002,bs=32,w=32,scales=3,dm=128,nh=1
- **Parameters**: 3,676,992
- **Epochs run**: 33 (best epoch 13)
- **Best val RMSE**: 11.002
- **Test RMSE**: 12.315 (paper target: 10.61)
- **Test Score**: 278.4 (paper target: 169)
- **Wall time**: 992s
- **Notes**: none

---

## FD001 Summary (5-seed, COMPLETE)

| Seed | Test RMSE | Test Score | Best Val RMSE | Epochs | Status |
|------|-----------|------------|---------------|--------|--------|
| 42 | 12.301 | 253.2 | 13.150 | 39 | GOOD |
| 123 | 12.969 | 286.7 | 11.396 | 54 | ABOVE |
| 456 | 11.250 | 196.2 | 11.316 | 41 | EXACT |
| 789 | 12.095 | 252.8 | 12.140 | 61 | GOOD |
| 1024 | 12.315 | 278.4 | 11.002 | 33 | GOOD |
| **Mean** | **12.186 +/- 0.553** | **253.5 +/- 31.6** | - | - | **GOOD** |
| Paper | 10.61 | 169 | - | - | - |
| Gap | +14.9% | +50% | - | - | GOOD |

**Assessment**: GOOD replication. 4/5 seeds in GOOD range (<=12.7), 1/5 EXACT (<=11.7). Mean 14.9% above paper. Score gap 50% (score is exponentially scaled, so larger gap is expected with RMSE gap). FD002 started 2026-04-11T15:44:20.

---


### 2026-04-11T17:13:31 | FD002 seed=42

- **Hyperparams**: lr=0.0002,bs=64,w=64,scales=4,dm=64,nh=4
- **Parameters**: 1,249,125
- **Epochs run**: 67 (best epoch 47)
- **Best val RMSE**: 20.539
- **Test RMSE**: 18.309 (paper target: 13.47)
- **Test Score**: 2107.7 (paper target: 784)
- **Wall time**: 5347s
- **Notes**: none


### 2026-04-11T18:48:04 | FD002 seed=123

- **Hyperparams**: lr=0.0002,bs=64,w=64,scales=4,dm=64,nh=4
- **Parameters**: 1,249,125
- **Epochs run**: 56 (best epoch 36)
- **Best val RMSE**: 20.075
- **Test RMSE**: 19.307 (paper target: 13.47)
- **Test Score**: 3555.8 (paper target: 784)
- **Wall time**: 5673s
- **Notes**: none


### 2026-04-11T21:03:29 | FD002 seed=456

- **Hyperparams**: lr=0.0002,bs=64,w=64,scales=4,dm=64,nh=4
- **Parameters**: 1,249,125
- **Epochs run**: 90 (best epoch 70)
- **Best val RMSE**: 19.015
- **Test RMSE**: 22.050 (paper target: 13.47)
- **Test Score**: 3278.7 (paper target: 784)
- **Wall time**: 8126s
- **Notes**: none


### 2026-04-11T22:36:16 | FD002 seed=789

- **Hyperparams**: lr=0.0002,bs=64,w=64,scales=4,dm=64,nh=4
- **Parameters**: 1,249,125
- **Epochs run**: 70 (best epoch 50)
- **Best val RMSE**: 19.296
- **Test RMSE**: 18.871 (paper target: 13.47)
- **Test Score**: 2320.2 (paper target: 784)
- **Wall time**: 5567s
- **Notes**: none


### 2026-04-11T23:04:35 | FD002 seed=1024

- **Hyperparams**: lr=0.0002,bs=64,w=64,scales=4,dm=64,nh=4
- **Parameters**: 1,249,125
- **Epochs run**: 33 (best epoch 13)
- **Best val RMSE**: 22.762
- **Test RMSE**: 21.589 (paper target: 13.47)
- **Test Score**: 3729.3 (paper target: 784)
- **Wall time**: 1699s
- **Notes**: none


### 2026-04-11T23:18:21 | FD003 seed=42

- **Hyperparams**: lr=0.0002,bs=32,w=48,scales=1,dm=128,nh=1
- **Parameters**: 1,220,890
- **Epochs run**: 53 (best epoch 33)
- **Best val RMSE**: 10.978
- **Test RMSE**: 12.832 (paper target: 10.71)
- **Test Score**: 296.0 (paper target: 202)
- **Wall time**: 814s
- **Notes**: none


### 2026-04-11T23:38:36 | FD003 seed=123

- **Hyperparams**: lr=0.0002,bs=32,w=48,scales=1,dm=128,nh=1
- **Parameters**: 1,220,890
- **Epochs run**: 101 (best epoch 81)
- **Best val RMSE**: 10.292
- **Test RMSE**: 12.621 (paper target: 10.71)
- **Test Score**: 270.2 (paper target: 202)
- **Wall time**: 1215s
- **Notes**: none


### 2026-04-11T23:53:57 | FD003 seed=456

- **Hyperparams**: lr=0.0002,bs=32,w=48,scales=1,dm=128,nh=1
- **Parameters**: 1,220,890
- **Epochs run**: 60 (best epoch 40)
- **Best val RMSE**: 10.956
- **Test RMSE**: 12.728 (paper target: 10.71)
- **Test Score**: 319.2 (paper target: 202)
- **Wall time**: 921s
- **Notes**: none


### 2026-04-12T00:10:58 | FD003 seed=789

- **Hyperparams**: lr=0.0002,bs=32,w=48,scales=1,dm=128,nh=1
- **Parameters**: 1,220,890
- **Epochs run**: 55 (best epoch 35)
- **Best val RMSE**: 10.300
- **Test RMSE**: 12.865 (paper target: 10.71)
- **Test Score**: 308.7 (paper target: 202)
- **Wall time**: 1022s
- **Notes**: none


### 2026-04-12T00:27:11 | FD003 seed=1024

- **Hyperparams**: lr=0.0002,bs=32,w=48,scales=1,dm=128,nh=1
- **Parameters**: 1,220,890
- **Epochs run**: 54 (best epoch 34)
- **Best val RMSE**: 9.577
- **Test RMSE**: 12.635 (paper target: 10.71)
- **Test Score**: 281.5 (paper target: 202)
- **Wall time**: 972s
- **Notes**: none


### 2026-04-12T05:03:20 | FD004 seed=42

- **Hyperparams**: lr=0.0002,bs=64,w=64,scales=4,dm=256,nh=4
- **Parameters**: 19,545,189
- **Epochs run**: 65 (best epoch 45)
- **Best val RMSE**: 24.115
- **Test RMSE**: 24.204 (paper target: 15.87)
- **Test Score**: 10067.7 (paper target: 1449)
- **Wall time**: 16554s
- **Notes**: none

