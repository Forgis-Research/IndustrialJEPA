# Lessons Learned (March 2026 Exploration)

## What We Tried
5 experiments on cross-machine transfer using AURSAD + Voraus robot datasets via FactoryNet.

## What Actually Happened (Honest Assessment)

### "Cross-machine forecasting transfer" (exp01-exp05) — MISLEADING
- Claimed transfer ratio ~1.07 with PatchTST channel-independent architecture
- **Reality**: Channel-independent processing means each sensor is predicted in isolation. Noisy, near-constant signals (joint positions, setpoints) are trivially predictable. A predict-last-value baseline would likely match this.
- This is NOT cross-machine transfer. It's per-channel autoregression on easy signals.

### Anomaly detection (exp01, exp03) — FAILED
- Best AUC: 0.53 (random chance)
- Anomaly signatures differ fundamentally across robots
- Setpoint-to-effort prediction with any normalization strategy doesn't work

### Many-to-1 framing (exp02, exp04) — INCONCLUSIVE
- Leave-one-out across AURSAD/Voraus/CNC
- Results not meaningfully better than single-source
- Small dataset (CNC: only 18 episodes) made comparisons noisy

## Genuine Insights Worth Keeping
1. **FactoryNet data loading works** — iloc-based windowing, episode normalization, source aliases all functional
2. **Shared signal space is achievable** — setpoint_pos, setpoint_vel, effort_voltage mapping across robots
3. **Cross-channel correlations don't transfer** — different robots have different kinematic coupling (this IS a real finding)
4. **RevIN helps with distribution shift** but insufficient alone for meaningful transfer

## Failed Approaches (DO NOT REPEAT)
| Approach | Why |
|----------|-----|
| Setpoint→effort prediction for anomaly detection | Anomaly signatures are robot-specific |
| Episode normalization for anomaly detection | Erases the anomaly signal |
| Channel-independent forecasting claimed as "transfer" | Not actually transfer |
| Scaling up model/data on fundamentally flawed setup | Won't fix representation problems |
