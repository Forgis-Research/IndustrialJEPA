# CNN-GRU-MHA Replication Results

**Paper**: Yu et al., Applied Sciences 2024, DOI: 10.3390/app14199039
**Run date**: 2026-04-10 01:55
**Device**: cuda
**Seeds**: [42, 123, 456, 789, 1024]
**Source epochs**: 200
**Finetune epochs**: 200

## Table 4: FEMTO Transfer RMSE

| Source | Target | Paper RMSE | Our RMSE (mean±std) | Delta | Status |
|--------|--------|:----------:|:-------------------:|:-----:|:------:|
| Bearing1_3 | Bearing2_3 | 0.0463 | 0.0435±0.0105 | -6.0% | EXACT |
| Bearing1_3 | Bearing2_4 | 0.0449 | 0.0487±0.0151 | +8.5% | EXACT |
| Bearing1_3 | Bearing3_1 | 0.0427 | 0.0444±0.0141 | +3.9% | EXACT |
| Bearing1_3 | Bearing3_3 | 0.0461 | 0.0544±0.0152 | +18.1% | GOOD |
| Bearing2_3 | Bearing1_3 | 0.0458 | 0.0252±0.0029 | -45.0% | BETTER |
| Bearing2_3 | Bearing1_4 | 0.0426 | 0.0376±0.0108 | -11.7% | GOOD |
| Bearing2_3 | Bearing3_3 | 0.0416 | 0.0514±0.0135 | +23.6% | WORSE |
| Bearing2_3 | Bearing3_3 | 0.0416 | 0.0514±0.0135 | +23.6% | WORSE |
| Bearing3_2 | Bearing1_3 | 0.0382 | 0.0328±0.0101 | -14.2% | GOOD |
| Bearing3_2 | Bearing1_4 | 0.0397 | 0.0355±0.0090 | -10.6% | GOOD |
| Bearing3_2 | Bearing2_3 | 0.0413 | 0.0336±0.0112 | -18.6% | GOOD |
| Bearing3_2 | Bearing2_4 | 0.0418 | 0.0504±0.0054 | +20.5% | WORSE |
| **Average** | | **0.0443** | **0.0424±0.0089** | **-4.3%** | **EXACT** |

## Success Criteria

- Good: Average RMSE within 20% of 0.0443 (threshold: 0.0532)
- Exact: Average RMSE within 10% of 0.0443 (threshold: 0.0487)

**Result: EXACT replication achieved** (our avg=0.0424 vs paper=0.0443)

## Notes

- Architecture: CNN (6 blocks, MHA after block 3) + 2-layer GRU + FC head
- Preprocessing: DWT denoising (sym8, level=3) + min-max normalization
- Channel: horizontal only (channel 0)
- RUL labels: linear decay Y_i = (N-i)/N
- Transfer protocol: freeze CNN+GRU, fine-tune FC on first half of target
- Evaluation: RMSE on second half of target (chronological split)
- Framework: PyTorch (paper used TensorFlow 2.5.0)
