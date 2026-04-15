# Sensor Correlation Analysis (V15)

## C-MAPSS FD001 Findings

### Correlation Structure

- Available sensors: 21
- High-correlation pairs (|r| > 0.7): 39
- Natural sensor clusters (Ward hierarchical): 4

Top correlated pairs:
  s5-s16: r=1.000
  s9-s14: r=0.963
  s11-s12: r=-0.847
  s4-s11: r=0.830
  s8-s13: r=0.826

### Degradation-Phase Correlation Shifts

Largest shifts in pairwise correlation (degraded - healthy phase):
  sel_s3-sel_s6: delta=-0.637
  sel_s2-sel_s6: delta=0.630
  sel_s6-sel_s7: delta=-0.618
  sel_s2-sel_s7: delta=-0.605
  sel_s3-sel_s7: delta=0.605

### Cluster Assignment

Sensors naturally group into 4 clusters:
  Cluster 1: s2, s3, s4, s7, s8, s11, s12, s13, s15, s17, s20, s21
  Cluster 2: s5, s16
  Cluster 3: s9, s14
  Cluster 4: s1, s6, s10, s18, s19

## Permutation Invariance

Linear input projection is position-dependent

Recommendation: Add sensor ID embeddings to make architecture permutation-equivariant

## Architecture Recommendation

**Based on this analysis:**

1. **Sensors are STRONGLY correlated** (many pairs with |r| > 0.7):
   This means simple channel-fusion (V2, treating all sensors as one input)
   can capture most of the shared variance. This explains why V2 is competitive.

2. **Correlations SHIFT during degradation**: The cross-sensor attention maps
   found in V14 (attention concentrating on s14 during degradation) are consistent
   with correlation shifts. Cross-sensor attention explicitly models this.

3. **Permutation equivariance**: Current architecture is NOT permutation-equivariant.
   Adding sensor ID embeddings (Phase 2) would fix this and make the model
   more principled. However, sensor ordering is fixed in C-MAPSS, so it's not
   a correctness issue, just an architectural principle.

**Recommendation:**

- **Default (channel-fusion)**: V2 is sufficient when sensor correlations are
  stable. Use for robust low-label regime (5% labels).
- **Cross-sensor attention**: When correlations shift during events (degradation,
  anomalies) - use V14 Phase 3 architecture with sensor ID embeddings.
- **Group attention**: If sensors form known groups (e.g., temperature cluster,
  pressure cluster) - apply attention within groups. C-MAPSS clusters don't have
  strong physical interpretation beyond redundancy.
- **iTransformer approach**: Treat each sensor as a token for the full time series.
  This is what we do in Phase 2; good for datasets where sensor identity matters
  more than temporal dynamics.

**Verdict:** The correlation shift during degradation (not the static correlation
structure) is the key signal. Cross-sensor attention + learnable sensor ID embeddings
is the principled choice for grey swan detection.
