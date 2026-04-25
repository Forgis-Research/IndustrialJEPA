# Phase 4a SOTA Research — agent prompt

You are the v30 Phase 4a SOTA research agent for the FAM (Forecast-Anything
Model) NeurIPS 2026 paper. Your job: for EACH dataset in the v30 benchmark
table, find the current published SOTA result, document its evaluation
protocol, and write a structured JSON file the rest of v30 will use to
populate the paper's "Comparison vs SOTA" column.

INPUTS:
- `/home/sagemaker-user/IndustrialJEPA/fam-jepa/experiments/v30/results/master_table.json`
  (the v30 Phase 3 numbers — gives you the dataset list)
- `/home/sagemaker-user/IndustrialJEPA/fam-jepa/experiments/RESULTS.md`
  (existing v29 SOTA references for some datasets — verify and update)

DATASETS TO COVER (from master_table.json):
- C-MAPSS: FD001, FD002, FD003 (RMSE)
- Server anomaly: SMAP, MSL (if included), PSM, SMD (if included)
- Clinical ECG: MBA (AUROC)
- Water: GECCO, BATADAL (F1, no point-adjust)
- New v30: SKAB, ETTm1
- (CHB-MIT was confirmed null in v29 — skip)

FOR EACH DATASET, find:
1. The current published SOTA paper (use WebSearch — search for "{dataset
   name} SOTA 2024 2025" and similar). Prefer 2024-2025 results from
   NeurIPS / ICML / ICLR / KDD.
2. The metric value they report (e.g. "RMSE = 11.91 on FD001").
3. The exact evaluation protocol:
   - Train/val/test split (matches our usage?)
   - Preprocessing (RUL cap, normalisation, etc.)
   - Scoring function (asymmetric? PA-F1 vs raw F1? threshold protocol?)
   - Whether they use point-adjust (PA) — many anomaly papers DO and FAM does NOT.
4. Citation: author, year, venue, paper title, URL/arXiv ID.

OUTPUT: write `/home/sagemaker-user/IndustrialJEPA/fam-jepa/experiments/v30/results/phase4a_sota.json`:
```json
{
  "FD001": {
    "sota_method": "...",
    "sota_metric": "RMSE",
    "sota_value": 11.91,
    "sota_paper": "Author et al. (Year) Title",
    "venue_year": "NeurIPS 2024",
    "url": "https://arxiv.org/abs/...",
    "their_protocol": {
      "train_split": "...",
      "test_split": "...",
      "rul_cap": 125,
      "preprocessing": "...",
      "scoring": "..."
    },
    "our_protocol_uses": {...},
    "protocol_match": "yes" | "no" | "partial",
    "our_value_our_protocol": null,  // FAM team fills from phase4_legacy_metrics.json
    "our_value_their_protocol": null, // FAM team fills if protocols differ
    "notes": "..."
  },
  ...
}
```

Use WebSearch to find:
- C-MAPSS RMSE SOTA: STAR 2024, ManInstrum 2024, etc.
- SMAP/MSL: TranAD, AnomalyTransformer, DCdetector, PatchAD — note that these are detection (not prediction); the cleaner FAM comparison is to forecasting/anomaly-prediction baselines.
- PSM/SMD: Same as above. Note PA vs non-PA difference is a famous trap (Kim et al. 2022 "Towards a Rigorous Evaluation of Time-Series Anomaly Detection").
- MBA (MIT-BIH Arrhythmia): typically reported as binary classification AUROC; SOTA depends on whether the protocol is patient-specific or cross-patient. Cite the specific MBA setup.
- GECCO: water-quality anomaly competition; some recent works exist.
- BATADAL: BATtle of Attack Detection Algorithms in Water Distribution Systems; CTOWN dataset specifically. Older SOTA but still cited.
- SKAB: Skoltech Anomaly Benchmark — published by Katser & Kozitsin 2020-2022; check for newer results.
- ETTm1: this is normally a FORECASTING benchmark (Informer / PatchTST / TimesNet); we use it as a derived event-prediction task with a causal rolling-window threshold. There is NO standard SOTA event-prediction number on ETTm1 — note this honestly. Compare against the forecasting RMSE/MAE if useful, but flag the framing change.

WRITING:
- For each dataset, write 3-5 sentences in plain prose alongside the JSON entry, describing the protocol gap (if any) between SOTA and our setup.
- For famous traps (PA-F1, RUL cap, etc.), be EXPLICIT: state which assumptions might inflate or deflate our number relative to SOTA.
- Cite paper IDs (arXiv) where possible. If you cannot find a reliable SOTA, write "no clear SOTA found" rather than guess.

REPORT BACK: when done, return a short text summary (≤300 words) listing
which datasets had clear SOTA, which were ambiguous, and any protocol
mismatches the FAM team should be aware of for the paper.
