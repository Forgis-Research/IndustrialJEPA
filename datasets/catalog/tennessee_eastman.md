# Tennessee Eastman Process (TEP)

## Executive Summary
- **Domain**: Chemical Process Industry
- **Task**: Fault detection / process monitoring; adaptable to forecasting
- **Size**: 22 fault types × 22 train simulations + 22 test simulations × 500 timesteps × 52 channels
- **Sampling Rate**: 1 sample per 3 minutes (process simulation)
- **Real vs Synthetic**: Synthetic — FORTRAN simulation of a chemical process
- **License**: Public domain (widely reproduced; original by Downs & Vogel, 1993)
- **Download URL**: Multiple mirrors available:
  - https://github.com/camaramm/tennessee-eastman-profBraatz (simulation code)
  - https://www.kaggle.com/datasets/averkij/tennessee-eastman-process-simulation-dataset
- **Published SOTA**: Extensive — classic process control benchmark, hundreds of papers

## Detailed Description

TEP (Tennessee Eastman Process) is a simulated chemical plant benchmark introduced in 1993 by Downs and Vogel. Despite being synthetic, it remains the most widely used benchmark in industrial process monitoring because it models realistic multivariate coupled dynamics with 21 fault types.

### Physical System (Simulated)
A chemical process with 5 unit operations:
1. Reactor (exothermic reaction: A+C+D → G; A+C+E → H)
2. Condenser
3. Vapor-liquid separator
4. Stripper (removes remaining liquid)
5. Compressor (recycle)

### Channels (52 total)
| Category | Count | Description |
|---|---|---|
| Process measurements (XMEAS) | 41 | Temperatures, pressures, flows, levels, compositions |
| Manipulated variables (XMV) | 11 | Valve positions, compressor speed, agitation speed |

### Key Measurements
| Variable | Description |
|---|---|
| XMEAS(1) | A feed flow (stream 1) |
| XMEAS(7) | Reactor feed rate |
| XMEAS(9) | Reactor temperature |
| XMEAS(13) | Production separator temperature |
| XMEAS(17) | Stripper temperature |
| XMEAS(22) | Reactor cooling water outlet temp |
| XMV(1)–(11) | Valve positions + compressor speed |

### Fault Scenarios
| Fault | Description | Type |
|---|---|---|
| IDV(0) | Normal operation | — |
| IDV(1) | A/C feed ratio step | Step change |
| IDV(2) | B composition step | Step change |
| IDV(4) | Reactor cooling water inlet temp step | Step change |
| IDV(5) | Condenser cooling water inlet temp step | Step change |
| IDV(6) | A feed loss (large) | Step change |
| IDV(7) | C header pressure loss | Step change |
| IDV(10) | C feed temp random variation | Random variation |
| IDV(13) | Reaction kinetics slow drift | Slow drift |
| IDV(16–20) | Unknown faults | — |

## Physics Groups
```python
TEP_GROUPS = {
    "reactor":       [6, 8, 9, 10, 11, 12],  # Feed, temp, pressure
    "separator":     [12, 13, 14, 15, 16],   # Separator conditions
    "stripper":      [16, 17, 18, 19],       # Stripper conditions
    "recycle":       [0, 1, 2, 3, 4, 5],     # Feed streams
    "compositions":  [22..40],               # Component analyzers
    "manipulated":   [41..51],               # Valve positions
}
```

## Published Benchmarks / SOTA (Fault Detection)
| Method | FDR (avg) | FAR | Paper | Year |
|---|---|---|---|---|
| PCA-based | ~0.79 | ~0.05 | Many | 1990s |
| SVM | ~0.90 | ~0.03 | Various | 2010s |
| CNN | ~0.94 | ~0.02 | Wu & Zhao | 2018 |
| LSTM + PCA | ~0.95 | ~0.02 | Various | 2019 |
| Transformer | ~0.97 | ~0.01 | Multiple | 2021–2023 |

FDR = Fault Detection Rate (recall), FAR = False Alarm Rate.

## Relevance to IndustrialJEPA

### Physics Grouping Potential
**Strong** — 52 channels with clear process-unit grouping (reactor/separator/stripper/recycle). Causal structure: feed → reactor → separator → stripper mimics physical flow.

### Transfer Learning Scenarios
- Cross-fault-type: Train on IDV(1–7), test on IDV(10–21)
- Cross-operating-mode: Different steady-state operating points (TEP has 6 modes)
- Fault-to-fault: Can the model generalize to novel faults?

### Scale Adequacy
**Moderate** — Standard dataset is small (500 × 52 × 44 runs). Extended simulations available (longer runs, more conditions) and can be regenerated via the FORTRAN/Python simulation code.

### Key Issue: Synthetic
Same limitation as C-MAPSS — simulation, not real sensor data. But TEP's coupling dynamics are much richer than C-MAPSS, and it is used in industrial control system research as a proxy for real plants.

### Verdict for Tier 2
**Possible Tier 2-synthetic upgrade over C-MAPSS** — 52 channels vs 21, richer physics, well-studied fault modes. But still synthetic. If the paper story requires only one synthetic Tier 2, TEP is superior to C-MAPSS for demonstrating physics-informed attention on a complex multi-subsystem process.

## Download Notes
- Kaggle version: https://www.kaggle.com/datasets/averkij/tennessee-eastman-process-simulation-dataset
  (CSV format, no account needed via direct download API)
- Simulation code: Can regenerate with python-control or MATLAB
- Downloader: `datasets/downloaders/download_tep.py`
