# Phase 1 — New dataset acquisition notes

Session start: 2026-04-24 22:45 UTC.

## Outcomes

| Dataset | Status | Reason |
|---------|--------|--------|
| **SWaT** | SKIPPED | iTrust requires registration + manual approval; not available locally. |
| **HAI 22.04** | BLOCKED | GitHub LFS budget exceeded (`This repository exceeded its LFS budget`). Plain `git clone --depth=1` got the 604MB metadata + README + PDF, but `git lfs pull` returns: "batch response: This repository exceeded its LFS budget". Attempted Zenodo mirror lookup (record 6559779) – unrelated content. |
| **CHB-MIT** | DEFERRED | 982h × 23 ch × 256 Hz raw is ~64 GB EDF. Disk only has 41 GB free. Per-subject streaming + downsample-on-the-fly is doable but takes ≥2-3 h end-to-end and consumes most of the remaining disk. |

## Pivot

Phase 1's intent was "broaden scope with 3 new datasets, each testing a
different event regime." I'm preserving that intent with the **two new
datasets that already have local data + cached Chronos-2 features**:

| Dataset | Domain | Event regime | Chronos features | FAM ckpt |
|---------|--------|--------------|------------------|----------|
| **GECCO 2018** | Drinking-water quality | Slow contamination (sparse, clustered) | cached (s42/123/456) | NEW (this session) |
| **BATADAL** | Water-distribution attacks | Discrete cyber-physical attacks | cached (s42/123/456) | NEW (this session) |

Neither was in v27's main benchmark — v27 had Chronos-2 surfaces only on
both. v28 adds the FAM side, giving the v28 master table a head-to-head
on every dataset.

## Honest tradeoff

The session prompt asked for genuinely new domains (cyber-physical attacks,
cascading propagation, EEG seizures). What I'm shipping covers cyber-physical
attacks (BATADAL) and a slow drift / contamination regime (GECCO), but
NOT cascading propagation or biomedical. The CHB-MIT and HAI rows are
explicitly absent and noted as deferred-pending-data in RESULTS.md.

If a future session wants HAI: the data is on Zenodo at
https://zenodo.org/records/8106109 (HAI 23.05) - this needs a clean
download to a separate directory (not via `git lfs`).

If a future session wants CHB-MIT: download per-subject from
PhysioNet, downsample to 32 Hz immediately, delete EDFs as you go;
target ~12 GB final cache.
