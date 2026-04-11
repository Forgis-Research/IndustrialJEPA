# TabPFN-TS Exercises

Practice problems to build intuition for time series forecasting with TabPFN.

## Beginner Exercises

### Ex1: Signal Types
Explore how TabPFN-TS handles different signal characteristics:
1. Pure sine wave (easy)
2. Trend + noise (medium)
3. Step change / regime shift (hard)

### Ex2: Horizon Sensitivity
Plot RMSE vs forecast horizon for:
- Short (5 steps)
- Medium (20 steps)
- Long (50+ steps)

### Ex3: Baseline Comparison
Implement and compare:
- Last value naive
- Seasonal naive
- Moving average
- TabPFN-TS

## Intermediate Exercises

### Ex4: Covariate Ablation
Given multi-covariate data:
1. Test all covariates
2. Remove one at a time
3. Find the most important covariate

### Ex5: Cross-Validation
Implement rolling-window cross-validation for TabPFN-TS on mechanical data.

### Ex6: Multi-Step Forecasting
Compare:
- Direct multi-step (predict all at once)
- Recursive (predict one, feed back, repeat)

## Advanced Exercises

### Ex7: Anomaly Detection
Can TabPFN-TS prediction residuals detect faults?
1. Train on healthy data
2. Test on fault data
3. Plot residuals and detect anomalies

### Ex8: Transfer Learning
Train on one hydraulic cycle, test on different cycles:
- Same operating conditions
- Different operating conditions

### Ex9: Ensemble Methods
Combine TabPFN-TS with other forecasters (ARIMA, Prophet) for improved predictions.

---

## Solutions

Solutions are provided in `solutions/` after you've attempted the exercises.
