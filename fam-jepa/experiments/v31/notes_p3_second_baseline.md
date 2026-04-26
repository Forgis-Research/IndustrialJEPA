# P3: Second Foundation Model Baseline - PARTIAL (MOMENT works under py310)

Attempted: MOMENT (momentfm), TimesFM 2.0, Moirai (uni2ts)

## MOMENT - WORKS under py310 conda env
Initial failure under base Python 3.12: `pkgutil.ImpImporter` removed; numpy<2.0
required by momentfm. Resolved by creating `py310` conda env and running with
`conda run -n py310 python3 baseline_moment.py`. Full 4-dataset / 3-seed sweep
completed. Results in `results/moment_baseline.json` (12 runs).

## TimesFM
Error: requires `lingvo==0.12.7` (Google JAX framework), not available on PyPI
for Python 3.12. All timesfm versions require Python <3.12.

## Moirai (uni2ts)
Installed but import fails: `lightning` framework has circular imports in py3.12.

## Conclusion
MOMENT-1-large works under the `py310` conda env and is now the second foundation
model baseline in the paper alongside Chronos-2. TimesFM and Moirai remain blocked
even under py310 due to `lingvo` / `lightning` import failures; they are documented
as deferred.

Timestamp: 2026-04-26
