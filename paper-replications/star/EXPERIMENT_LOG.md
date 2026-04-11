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

