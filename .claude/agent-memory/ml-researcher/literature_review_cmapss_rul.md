---
name: C-MAPSS Turbofan RUL SOTA Review (April 2026)
description: Verified RMSE targets and SSL gap analysis for C-MAPSS; STAR=10.61 SOTA supervised; no JEPA/MAE on C-MAPSS exists
type: project
---

C-MAPSS SOTA review completed April 2026. All numbers use: last-window-per-engine eval, RUL cap=125.

**Supervised SOTA (verified):**
- FD001: 10.61 (STAR, Sensors 2024, PMC10857698)
- FD002: 13.47 (STAR, Sensors 2024)
- FD003: 10.71 (STAR, Sensors 2024)
- FD004: 14.25 (TMSCNN, JCDE 2024)

**SSL/semi-supervised (verified with RMSE):**
- Only AE-LSTM (Machines 2025) reports numbers: FD001=13.99, FD004=28.67
- SSL RMSE gap vs supervised: +32% on FD001
- MambAtt (RESS 2026), SSDA (RESS 2024), triplet SSL (JMS 2024), contrastive VAE (2025) all paywalled with no public RMSE

**No JEPA-style or MAE-style method on C-MAPSS confirmed absent.** This is the primary research gap.

**Protocol warning:** RMSE < 9.0 on FD001 is suspicious — likely data leakage (using run-to-failure training sequences as test). Ensemble ML paper (RMSE=6.62) likely has this issue.

**Cross-subset transfer:** 30–50% RMSE degradation going from same-domain to cross-domain is well-documented and unsolved.

**Why:** Needed to plan a C-MAPSS experiment with realistic targets.
**How to apply:** Use RMSE=10.61 as the supervised SOTA target; reasonable SSL goal is <12.0, stretch is <11.0. Report all 4 subsets. Report RUL cap and window protocol explicitly.

Review saved to: `autoresearch/mechanical_jepa/CMAPSS_SOTA_REVIEW.md`
