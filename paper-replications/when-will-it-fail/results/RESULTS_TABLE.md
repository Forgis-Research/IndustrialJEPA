# Results Table - A2P Replication vs Paper

Paper: Park et al. "When Will It Fail?", ICML 2025, Table 1.  
Our run: ml-researcher agent, 2026-04-10. **Status: AWAITING EXECUTION.**

Bash tool was not available in this session. The table below shows paper numbers and will be populated once `python run_replication.py --dataset mba` is executed.

---

## Table 1 Replication (F1-with-tolerance, no point adjustment)

| L_out | Dataset | Paper F1 | Paper Std | Our F1 | Our Std | Delta |
|:-----:|---------|:--------:|:---------:|:------:|:-------:|:-----:|
| 100 | MBA | 67.55 | 5.62 | NOT RUN | - | - |
| 100 | Exathlon | 18.64 | 0.16 | NOT RUN | - | - |
| 100 | SMD | 36.29 | 0.18 | NOT RUN | - | - |
| 100 | WADI | 64.91 | 0.47 | NOT RUN | - | - |
| 200 | MBA | 74.63 | 5.92 | NOT RUN | - | - |
| 200 | Exathlon | 28.71 | 0.54 | NOT RUN | - | - |
| 200 | SMD | 42.36 | 0.80 | NOT RUN | - | - |
| 200 | WADI | 66.65 | 1.93 | NOT RUN | - | - |
| 400 | MBA | 69.35 | 7.15 | NOT RUN | - | - |
| 400 | Exathlon | 43.57 | 1.10 | NOT RUN | - | - |
| 400 | SMD | 48.10 | 2.55 | NOT RUN | - | - |
| 400 | WADI | 74.57 | 6.37 | NOT RUN | - | - |

---

## Ablation Results

### Table 2: AAF and SAP ablation (L_in = L_out = 100)

| Condition | MBA | Exathlon | SMD | WADI | Avg F1 | Status |
|-----------|:---:|:--------:|:---:|:----:|:------:|--------|
| No AAF, No SAP | 36.26 | 17.65 | 34.74 | 58.66 | 36.82 | PAPER ONLY |
| AAF only | 50.57 | 17.76 | 34.87 | 62.31 | 38.89 | PAPER ONLY |
| SAP only | 55.95 | 18.29 | 36.05 | 59.36 | 42.41 | PAPER ONLY |
| Full A2P | 67.55 | 18.64 | 36.29 | 64.91 | 46.84 | PAPER ONLY |

### Table 4: Shared backbone ablation

| Condition | MBA | Exathlon | SMD | WADI | Avg F1 | Status |
|-----------|:---:|:--------:|:---:|:----:|:------:|--------|
| Not shared | 51.53 | 18.00 | 35.60 | 60.70 | 41.45 | PAPER ONLY |
| Shared (A2P) | 67.55 | 18.64 | 36.29 | 64.91 | 46.84 | PAPER ONLY |

---

## How to Execute

```bash
# Prerequisites
cd /path/to/IndustrialJEPA/paper-replications/when-will-it-fail
git clone https://github.com/KU-VGI/AP.git AP/
pip install -r AP/requirements.txt

# Download data (from AnomalyTransformer repo - thuml/Anomaly-Transformer)
# Place .npy files at DATA_ROOT/MBA/, DATA_ROOT/SMD/, etc.

# Run MBA replication (all 3 seeds, all 3 L_out values)
export A2P_DATA_ROOT=/path/to/data
python run_replication.py --dataset mba

# Run SMD
python run_replication.py --dataset smd

# Run ablations on MBA
python run_replication.py --dataset mba --pred_lens 100 --ablation aaf_off
python run_replication.py --dataset mba --pred_lens 100 --ablation shared_off
```

After running, re-execute `python run_replication.py` to regenerate this table with actual numbers.
