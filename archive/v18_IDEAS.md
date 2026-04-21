# V18 Ideas (session after next)

## 1. Shift to F1-based evaluation (MTS-JEPA comparable)

Currently we evaluate C-MAPSS with RMSE and SMAP with non-PA F1 + PA-F1. To become directly comparable against MTS-JEPA and all other anomaly detection work:

**Proposal**: frame ALL grey swan tasks as binary event detection with F1:
- **Anomaly**: F1 on per-point or per-window binary labels (already doing this)
- **RUL as event detection**: "will the system fail within k cycles?" → binary → F1
- **TTE as event detection**: "will sensor j cross threshold within k cycles?" → binary → F1

The trajectory probe already gives us s(k) = probe(γ(k)). Threshold at τ → binary prediction → F1.

**Window size decisions**: align with MTS-JEPA protocol for direct comparison:
- SMAP/MSL: window=100, stride=1 (standard)
- SWaT: window=100, stride=1 (standard)
- C-MAPSS: window = engine lifetime (variable), one prediction per cycle

**Benefit**: one metric family (F1, precision, recall, AUC-PR) across all tasks and datasets. Directly comparable to MTS-JEPA numbers (SMAP PA-F1=33.6%, SWaT PA-F1=72.9%).

## 2. Gaussian Process over event timing

**Goal**: construct a probability distribution p(event at time t+k | x_{1:t}) rather than a point estimate.

**Approach**: replace the linear probe with a GP head:
- Input: [γ(1), γ(2), ..., γ(K)] — the predicted latent trajectory
- Kernel: learned on top of the frozen features
- Output: mean μ(k), variance σ²(k) for event probability at each horizon k
- The GP gives calibrated uncertainty bands around the event-time prediction

**What this enables**:
- AUC-style reliability charts: predicted probability vs actual event rate
- Risk-aware maintenance scheduling: "90% probability of failure within 50 cycles"
- Calibrated uncertainty: different from point-estimate RMSE, this quantifies HOW CONFIDENT the prediction is
- Natural connection to survival analysis (hazard functions, Kaplan-Meier)

**Lightweight alternative**: ensemble of linear probes (5 probes with different random seeds) → mean + std as uncertainty estimate. Cheap proxy for GP.

**Literature**: Deep Kernel Learning (Wilson et al. 2016), GP layers on neural features (Bradshaw et al. 2017), natural gradient for GP (Salimbeni & Deisenroth 2017).

## 3. iTransformer for cross-sensor correlations

**Goal**: understand and leverage cross-sensor correlations for multivariate prediction.

**Current state**: V14 cross-sensor (sensor-as-token, iTransformer-style) achieved frozen 14.98 ± 0.22 (best frozen result). But the learnable sensor ID embeddings act as training stabilizers — without them, 2/3 seeds fail.

**V18 proposal**: properly integrate iTransformer insights:
- **Alternating attention layers**: temporal attention (attend across time) → sensor attention (attend across sensors) → temporal → sensor → ...
- **Cross-sensor attention reveals which sensor pairs matter**: attention maps shift during degradation (V14 found s3→s14 attention increases during FD001 degradation)
- **Sensor-group masking**: mask entire sensor groups during pretraining (physics-aware: group by subsystem). Forces the model to learn cross-sensor dependencies.
- **Sensor dropout**: randomly drop sensors during training → robustness to missing channels at inference

**Key question**: does cross-sensor structure improve the trajectory probe? If γ(k) from a cross-sensor encoder is richer, the trajectory probe should be better.

**Connection to GP**: cross-sensor correlations could inform the GP kernel — sensors that are correlated should have correlated event probabilities.

## 4. Combined vision

The full V18 system would be:
```
Pretrain:  cross-sensor encoder + LogUniform predictor
Probe:     GP head on predicted trajectory γ(k)
Output:    p(event | x_{1:t}, k) with calibrated uncertainty
Evaluate:  F1 at optimal threshold, AUC-PR, calibration curves
```

This gives us:
- Direct MTS-JEPA comparability (F1/AUC-PR on same datasets)
- Calibrated uncertainty (GP or ensemble)
- Cross-sensor structure (iTransformer attention)
- Multi-scale dynamics (LogUniform k from V17)
