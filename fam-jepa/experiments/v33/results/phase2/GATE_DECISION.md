# Phase 2: Channel Dropout Gate Decision

## Sweep Results

| Dataset | Rate=0.0 | Rate=0.1 | Rate=0.3 | Rate=0.5 | Best Rate | 3-seed AUROC | Delta | Decision |
|---------|----------|----------|----------|----------|-----------|--------------|-------|----------|
| PSM | 0.5787 | 0.5607 | 0.5584 | 0.5834 | 0.5 | 0.5678 +/- 0.0030 | +0.013 | PASS |
| SMAP | 0.6301 | 0.5652 | 0.5242 | 0.4048 | 0.0 | 0.5767 +/- 0.0293 | +0.044 | PASS |
| FD001 | 0.7216 | 0.7369 | 0.7289 | 0.7307 | 0.1 | 0.7322 +/- 0.0135 | +0.011 | PASS |

## Gate Decision: PROCEED to Phase 3 (ST-JEPA)

- PSM: PASS (delta=+0.013 > +0.01)
- SMAP: PASS (delta=+0.044 > -0.02)
- FD001: PASS (delta=+0.011 > -0.02)

## Next Steps
Channel dropout PASSED gate. Proceed with Phase 3 (Full ST-JEPA).
Best dropout rates per dataset:
  - PSM: 0.5
  - SMAP: 0.0
  - FD001: 0.1