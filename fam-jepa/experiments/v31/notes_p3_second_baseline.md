# P3: Second Foundation Model Baseline - FAILED

Attempted: MOMENT (momentfm), TimesFM 2.0, Moirai (uni2ts)

## MOMENT
Error: `pkgutil.ImpImporter` removed in Python 3.12. Package requires numpy<2.0
which in turn triggers a numpy wheel rebuild that fails on py3.12.

## TimesFM
Error: requires `lingvo==0.12.7` (Google JAX framework), not available on PyPI
for Python 3.12. All timesfm versions require Python <3.12.

## Moirai (uni2ts)
Installed but import fails: `lightning` framework has circular imports in py3.12.

## Conclusion
All three foundation model baselines blocked by Python 3.12 dependency conflicts.
The v30 notes documented this as needing a containerized environment or conda env
with Python 3.10. This is still the case in v31.

**Workaround options (for future session)**:
1. Create conda env: `conda create -n py310 python=3.10 && conda activate py310`
2. Docker container with Python 3.10
3. Use existing Chronos-2 from v24/v27 (already computed, cached in `experiments/v24/chronos_features/`)

**Decision**: Skip P3 as per session rules (cap at 1h). Chronos-2 alone is
sufficient for the main paper. Document in SESSION_SUMMARY.md.

Timestamp: 2026-04-26
