# V20 Multi-Domain Benchmark (FAM across N datasets)

All FAM results use the 1.26M-parameter (V17 arch, d_model=256, 2L, 4H) model. Legacy metrics (PA-F1, RMSE, macro-F1) reported for literature comparability. Per-window F1 (W=16) reported where pred-FT is run.


## FAM primary results

| Dataset | Domain | FAM primary | FAM legacy | SOTA (primary) | SOTA ref |
|---------|--------|-------------|------------|----------------|----------|
| C-MAPSS FD001 | Turbofan RUL | F1w 0.391 ± 0.085 (5s) (pred-FT) | RMSE 16.903 ± 1.711 (5s) | RMSE 10.61 | STAR (Fan 2024) |
| PSM | Server metrics | (pred-FT pending) | PA-F1 0.813 ± 0.048 (3s) (Mahal k=100) | PA-F1 0.616 | MTS-JEPA (He 2026) |
| SMAP | Spacecraft telemetry | PA-F1 0.793 ± 0.014 (3s) | non-PA F1 0.038 | PA-F1 0.336 | MTS-JEPA (He 2026) |
| MSL | Spacecraft telemetry | PA-F1 0.707 ± 0.050 (3s) | non-PA F1 - | PA-F1 0.336 | MTS-JEPA (He 2026) |
| SMD | Server machine | PA-F1 0.252 ± 0.017 (3s) | non-PA F1 0.14429674455635588 | PA-F1 0.925 | Anomaly Transformer (Xu 2022) |
| MBA ECG | Cardiac | PA-F1 0.551 ± 0.054 (3s) | non-PA F1 0.25072007032622523 | - | Schmidt & Simic (2020) |
| Paderborn bearing | Bearing fault | macro-F1 0.781 ± 0.035 (3s) | acc 0.783 | - | STAR (Fan 2024) |

## Pred-FT (FD001) vs baselines detailed (Phase 0)

| Mode | Labels | F1w | RMSE |
|------|--------|-----|------|
| probe_h | 100% | 0.299 ± 0.061 (5s) | 15.997 ± 1.481 (5s) |
| pred_ft | 100% | 0.391 ± 0.085 (5s) | 16.903 ± 1.711 (5s) |
| pred_ft | 5% | 0.260 ± 0.165 (5s) | 24.334 ± 6.835 (5s) |
| e2e | 100% | 0.408 ± 0.119 (5s) | 14.956 ± 1.157 (5s) |
| e2e | 5% | 0.177 ± 0.242 (5s) | 20.085 ± 1.885 (5s) |
