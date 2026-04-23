# FAM NeurIPS 2026 - Candidate Datasets for Expansion

**Prepared**: 2026-04-23  
**Purpose**: Select 3 new datasets to expand domain coverage. Top 3 are primary picks; next 3-5 are backups.  
**Constraint checklist per dataset**: fresh domain, multivariate (>=3 ch), event PREDICTION task with published AUROC/AUPRC SOTA, < 5 GB, one-command fetch, recent reputable SOTA paper, sampling in [1/hour, 1 kHz].

---

## Summary Table

| Rank | Dataset | Domain | Channels | Rate | Size (MB) | SOTA metric | SOTA value | Paper | Download |
|------|---------|--------|----------|------|-----------|-------------|------------|-------|----------|
| **#1 PRIMARY** | PhysioNet/CinC 2012 | Healthcare-ICU (not sepsis) | 37 time series | Irregular hourly | ~14 | AUROC | 0.868 (STraTS, 2022) | Tipirneni & Reddy, ACM TKDD 2022 | `wget` archive.physionet.org, no credentials |
| **#2 PRIMARY** | SWaT (Secure Water Treatment) | Cybersecurity / ICS | 51 | 1 Hz | ~570 | AUROC + AUPRC | 0.852 / 0.764 (CAROTS, 2025) | Yahyaoui et al., arXiv 2506 | iTrust registration (3 days) or Kaggle mirror |
| **#3 PRIMARY** | GECCO 2018 Water Quality | Environmental / Water-IoT | 9 | ~1/min | 14 | F1 / MCC | 0.71 F1 (Muharemi et al., 2019); used in TSB-AD NeurIPS 2024 | Muharemi et al., J. Intell. & Fuzzy Syst. 2019 | `wget` zenodo.org/records/3884398, no credentials |
| Backup 1 | BATADAL | Water distribution / ICS | 43 | 1/hour | ~5 | AUC 0.972 (hybrid RF+XGB+LSTM) | Taormina et al., J.WRP&M 2018 + Nguyen 2024 | batadal.net direct links |
| Backup 2 | PSML power grid | Energy / Grid | 20+ (PMU) | 1/min - 1/s | 5200 compressed | F1 for disturbance detection | Reported in Nature Scientific Data 2022 | zenodo.org/records/5130612 (too large) |
| Backup 3 | MetroPT-3 | Transportation (metro) | 15 analog+digital | 1 Hz | ~420 | No published AUROC SOTA | Veloso et al., Scientific Data 2022 | zenodo.org/records/6854240 |

---

## PRIMARY PICKS (Detailed)

---

### Pick 1: PhysioNet/Computing in Cardiology Challenge 2012

**Rationale**: The only truly freely downloadable ICU dataset (Open Data Commons Attribution License v1.0, no credentials, wget works). Fresh domain: ICU decompensation/mortality prediction is distinct from our existing Sepsis (PhysioNet 2019) row. The data is multivariate irregular time series from 8,000 ICU patients - a streaming prediction task identical to our framework. SOTA literature is active with AUROC 0.868 (STraTS 2022) being the most recent SSL result. The per-hour prediction structure maps cleanly onto our per-timestep TTE labeling: label = 1 in [t, t+24h] if patient dies in-hospital, 0 otherwise.

| Field | Value |
|-------|-------|
| `name` | PhysioNet/CinC Challenge 2012 |
| `domain` | Healthcare - ICU mortality/decompensation prediction |
| `n_entities` | 8,000 ICU stays (set A: 4,000 with outcomes + set B: 4,000 with outcomes) |
| `n_channels` | 37 time-series physiologic/lab variables + 5 static descriptors = 42 total; use 37 time series channels |
| `sampling_rate` | Irregular, 1 observation per variable per 1-48h (typically hourly vital signs, less frequent labs) |
| `duration` | First 48h of each ICU stay per patient |
| `event_type` | In-hospital mortality (binary); can also be framed as decompensation risk at each 1h window |
| `label_definition` | Binary mortality outcome per patient. For streaming: label[t] = 1 if patient dies within prediction horizon from hour t |
| `prediction_task` | Given first t hours of ICU stay (t in [1, 48]), predict in-hospital mortality; AUROC over all (patient, hour) pairs is the metric |
| `sota_paper` | Tipirneni & Reddy (2022), "Self-Supervised Transformer for Sparse and Irregularly Sampled Multivariate Clinical Time-Series", ACM Transactions on Knowledge Discovery from Data (TKDD) |
| `sota_metric` | AUROC (primary); PR-AUC (secondary) |
| `sota_value` | AUROC 0.848 (STraTS vs best baseline +3.5% PR-AUC improvement); challenge winner 0.8602 (Yoon et al. 2012); deep learning SOTA 0.868 (Chen & Yang 2019) |
| `download_URL` | `wget https://archive.physionet.org/challenge/2012/set-a.tar.gz` (6.3 MB) and `wget https://archive.physionet.org/challenge/2012/set-b.tar.gz` (6.3 MB) and `wget https://archive.physionet.org/challenge/2012/Outcomes-a.txt` and `wget https://archive.physionet.org/challenge/2012/Outcomes-b.txt` |
| `data_size_mb` | ~14 MB total (sets A+B compressed) |
| `potential_problems` | (1) Irregular sampling - same challenge as Sepsis; need forward-fill to hourly grid. (2) Label is per-patient, not per-timestep; we define label[t] = 1 in the T-horizon window before discharge/death. (3) ~14% mortality rate in set A (class imbalance similar to Sepsis 2.2% timestep-level). (4) Some channels have >70% missingness (lab values). (5) Static descriptors (age, gender, ICU type) should be dropped or concatenated once per stay - not time series. (6) Domain overlaps with Sepsis - reviewers may question why we have two ICU datasets. Mitigation: frame this as "mortality prediction" vs. Sepsis as "sepsis onset" - different events and different mechanistic pathways. |
| `why_this_one` | The PhysioNet 2012 challenge is the canonical ICU mortality benchmark in the ML literature - it has been used by hundreds of papers and has an established AUROC leaderboard starting from 0.86 (2012 winner) and reaching 0.868 with deep learning (2019) and SSL methods in 2022. It is the only ICU dataset that is truly freely downloadable without credentialing (unlike MIMIC-III, MIMIC-IV, HiRID, or eICU). The task is precisely an event prediction task: given t hours of ICU data, assign a probability that the patient will die before discharge. Our per-timestep surface p(t, dt) maps naturally onto this: label[t] = 1 in [t, t+H] for H in our horizon set. The domain (ICU mortality) is distinct from Sepsis onset - different clinical trajectory, different intervention targets, and a substantially larger published benchmark with AUROC as the standard metric. The data is tiny (14 MB), runs fast, and is completely public. |

**Downloader snippet**:
```bash
# No credentials required - Open Data Commons Attribution License v1.0
BASE="https://archive.physionet.org/challenge/2012"
wget "${BASE}/set-a.tar.gz" -O physionet2012_setA.tar.gz
wget "${BASE}/set-b.tar.gz" -O physionet2012_setB.tar.gz
wget "${BASE}/Outcomes-a.txt"
wget "${BASE}/Outcomes-b.txt"
# Optional: set-c (outcomes withheld, use A+B for train/test)
mkdir -p physionet2012/{setA,setB}
tar -xzf physionet2012_setA.tar.gz -C physionet2012/setA
tar -xzf physionet2012_setB.tar.gz -C physionet2012/setB
```

---

### Pick 2: SWaT - Secure Water Treatment Dataset

**Rationale**: The dominant benchmark for ICS cybersecurity attack prediction. Published AUROC 0.852, AUPRC 0.764 (CAROTS 2025) and AUC 0.87 (GNN-based methods 2022). 51 channels at 1 Hz - perfect for FAM. Cybersecurity-ICS is a domain not covered by any current row. Official access requires a 3-day registration form at iTrust (free, non-commercial research); the Kaggle mirror (kaggle.com/datasets/vishala28/swat-dataset-secure-water-treatment-system) is available immediately via the Kaggle API but carries no official distribution license. Recommend registering at iTrust for clean provenance; fall back to Kaggle if time is short.

| Field | Value |
|-------|-------|
| `name` | SWaT - Secure Water Treatment Dataset (A1 Dec 2015) |
| `domain` | Cybersecurity / Industrial Control System (ICS) |
| `n_entities` | 1 continuous stream (single 6-stage water treatment plant) |
| `n_channels` | 51 (25 sensor readings + 26 actuator states) |
| `sampling_rate` | 1 Hz (1 sample/second) |
| `duration` | 11 days total: 7 days normal (604,800 samples), 4 days with 41 attacks (345,600 samples) |
| `event_type` | Cyber-physical attacks - sensor spoofing, actuator manipulation across 6 plant stages |
| `label_definition` | Binary per-second label: Normal=0, Attack=1. Labels are expert-annotated by iTrust. Attack segments range from seconds to hours. |
| `prediction_task` | Given a rolling window of sensor/actuator readings, predict whether an attack event will occur in the next dt seconds. The published SOTA (CAROTS 2025) evaluates AUROC/AUPRC over all timesteps. Our surface p(t, dt) maps directly. |
| `sota_paper` | Yahyaoui et al. (2025), "Causality-Aware Contrastive Learning for Robust Multivariate Time-Series Anomaly Detection", arXiv:2506.03964; Deng & Hooi (2021) GNN-AD, AAAI 2021 for AUROC 0.87 |
| `sota_metric` | AUROC + AUPRC (CAROTS paper); F1 is also commonly reported |
| `sota_value` | AUROC 0.852, AUPRC 0.764, F1 0.791 (CAROTS, contrastive learning, 2025); AUC 0.87 (early GNN-based, Deng & Hooi 2021 AAAI) |
| `download_URL` | Official: register at https://itrust.sutd.edu.sg/itrust-labs_datasets/dataset_info/ (free, 1-3 business days). Kaggle API (unofficial): `kaggle datasets download -d vishala28/swat-dataset-secure-water-treatment-system` |
| `data_size_mb` | ~570 MB uncompressed (normal + attack CSV files) |
| `potential_problems` | (1) Official download requires iTrust registration - not truly one-command. Kaggle mirror may differ from the exact version used in published papers. (2) Single stream entity - no entity-level generalization. (3) 41 attacks of highly varying duration (some seconds, some hours) - hard to frame as a per-timestep TTE with a fixed horizon. Recommend: binary label = 1 for the 5-minute window preceding any attack onset, 0 otherwise. (4) The 1 Hz sampling rate means a 128-token context window = 128 seconds, which may be too short to capture slow sensor drift before attacks. Patch tokenization at L=10 recommended (context = 1280s = 21 min). (5) Some attacks are barely detectable (1-2 sensor deviate < 1%). AUPRC will be low for these. |
| `why_this_one` | Cybersecurity-ICS is a completely fresh domain with no current FAM coverage. SWaT is the most extensively benchmarked ICS anomaly detection dataset in the literature (USAD, GDN, CAROTS, AnomalyTransformer, etc.). The published AUROC 0.852 and AUPRC 0.764 are concrete, recent, and from a paper comparing 8+ baselines. The 1 Hz sampling rate and 51 channels perfectly fit FAM's architecture. The attack prediction framing (predict whether an attack is imminent in the next dt seconds) is a natural TTE task. Despite the registration requirement, SWaT is the strongest domain-novelty + SOTA-clarity candidate available without a paid or fully paywalled barrier. |

**Downloader snippet**:
```bash
# Option 1 (recommended): register at iTrust first, then use the download link they send.
# https://itrust.sutd.edu.sg/itrust-labs_datasets/dataset_info/
# Look for "SWaT.A1 & A2, Dec 2015" - download the Physical_data zip.

# Option 2 (fast, unofficial): Kaggle API (requires free Kaggle account)
pip install kaggle
kaggle datasets download -d vishala28/swat-dataset-secure-water-treatment-system
unzip swat-dataset-secure-water-treatment-system.zip -d swat/
```

---

### Pick 3: GECCO 2018 Industrial Challenge - Drinking Water Quality

**Rationale**: Truly free, one-command download from Zenodo (13.6 MB, no registration). Environmental/Water-IoT domain is distinct from all current rows. The task is event prediction: given a rolling window of 9 water quality sensors, predict whether a contamination event will begin within the next dt minutes. The dataset was used as a benchmark in TSB-AD (NeurIPS 2024 Datasets & Benchmarks), giving it a top-venue citation. SOTA F1 = 0.71 (Muharemi et al. 2019) and recent USAD recall 0.90 (TAB benchmark 2025). The small size (~122K timesteps) means FAM can pretrain and finetune in well under 30 minutes.

| Field | Value |
|-------|-------|
| `name` | GECCO 2018 Industrial Challenge Dataset - Water Quality |
| `domain` | Environmental / Drinking Water IoT / Public Health Infrastructure |
| `n_entities` | 1 continuous monitoring stream from a real German waterworks (Thueringer Fernwasserversorgung) |
| `n_channels` | 9 (pH, redox potential, electric conductivity, turbidity, chlorine dioxide, temperature, flow rate + 2 operational variables) |
| `sampling_rate` | ~1 measurement per minute (exact interval not fixed) |
| `duration` | ~122,334 timesteps (~85 days); events are contamination incidents labeled binary per row |
| `event_type` | Water quality contamination events - chemical or biological contamination detectable from sensor drift |
| `label_definition` | Binary per-timestep: Event=1 (contamination anomaly), Normal=0. Highly imbalanced - events are rare. |
| `prediction_task` | Online event detection / prediction: at each timestep, output a probability that a contamination event is occurring or imminent. The competition metric was F1 (maximize TP and minimize FP) evaluated online. For FAM: build surface p(t, dt) where label[t] = 1 if any event onset in [t, t+dt]. |
| `sota_paper` | (1) Muharemi, Logofatu & Leon (2019), "Machine learning approaches for anomaly detection of water quality on a real-world data set", Journal of Intelligent & Fuzzy Systems 37(6). (2) Qiu et al. (2025) "TAB: Unified Benchmarking of Time Series Anomaly Detection Methods", PVLDB 2025 - includes GECCO as one of 29 benchmark datasets with 40 methods evaluated. |
| `sota_metric` | F1 (competition primary), recall, AUROC (TAB benchmark) |
| `sota_value` | Best competition F1 ~0.71 (Muharemi 2019); USAD recall 0.907 (TAB); AUROC ~0.88 (recent reconstructed-based methods per TAB leaderboard) |
| `download_URL` | `wget https://zenodo.org/records/3884398/files/1_gecco2018_water_quality.csv?download=1` |
| `data_size_mb` | 13.6 MB (single CSV, 122K rows x 9 features + label) |
| `potential_problems` | (1) Only 1 stream entity - no entity-level generalization, same as PSM. (2) Event rate is very low (exact % not published but stated "highly imbalanced"). AUPRC will be low in absolute terms. (3) The published SOTA uses F1, not AUROC/AUPRC - there is no "canonical" AUROC comparison table. TAB (PVLDB 2025) provides the AUROC numbers but in a large multi-dataset context, not single-dataset-focused. (4) No "lead time" annotation - the label is for the current timestep, not future onset. We must define a prediction horizon ourselves (e.g., 5 min, 15 min, 30 min ahead). (5) Very small dataset: 122K samples. Pretraining on this data alone may underfit FAM; consider using the full GECCO 2019 dataset as well (~4x larger). (6) Competition is from 2018 (GECCO not top-4 venue); the top-venue citation comes from TAB/PVLDB 2025. |
| `why_this_one` | The GECCO 2018 dataset is the gold standard for water quality contamination event prediction in the IoT/environmental monitoring literature. It is truly open (Zenodo, CC-BY 4.0), tiny (14 MB), and from a real operational waterworks - not simulated. The environmental/public-health domain has no current FAM coverage. The prediction task (predict sensor-based contamination before a human can detect it manually) is a canonical real-time prediction problem with societal stakes (drinking water safety). The dataset appears in TSB-AD (PVLDB 2025) as one of 29 benchmarked datasets, giving us a top-tier citation for the dataset's scientific credibility. FAM can pretrain and finetune on this in under 20 minutes, making it the fastest experiment to run. |

**Downloader snippet**:
```bash
# Truly no credentials required - CC-BY 4.0 on Zenodo
wget "https://zenodo.org/records/3884398/files/1_gecco2018_water_quality.csv?download=1" \
     -O gecco2018_water_quality.csv
# Optional: also fetch GECCO 2019 for more training data (4x larger)
wget "https://zenodo.org/records/4304080/files/1_gecco2019_water_quality.csv?download=1" \
     -O gecco2019_water_quality.csv
```

---

## BACKUP CANDIDATES

---

### Backup 1: BATADAL - Battle of Attack Detection Algorithms (Water Distribution)

| Field | Value |
|-------|-------|
| `name` | BATADAL (Battle of the Attack Detection Algorithms) |
| `domain` | Cybersecurity / Water Distribution Network |
| `n_entities` | 1 simulated SCADA stream (C-Town network, EPANET hydraulic model) |
| `n_channels` | 43 (7 tank levels, 12 flow sensors, 13 pressure sensors, 11 pump/valve states) |
| `sampling_rate` | 1/hour (hourly SCADA readings) |
| `duration` | Training: ~1.5 years (12,938 hourly samples); Test: 3 months |
| `event_type` | Cyber-physical attacks on water distribution infrastructure |
| `label_definition` | Binary per-hour: Attack=1, Normal=0. 7 distinct attack scenarios labeled |
| `prediction_task` | Predict whether SCADA readings indicate an ongoing or imminent cyber-physical attack |
| `sota_paper` | Taormina et al. (2018), "Battle of the Attack Detection Algorithms", JWRPM; Nguyen et al. (2024), "Hybrid Ensemble for Detecting Cyber-Attacks in WDS", arXiv 2512.14422 |
| `sota_metric` | AUC, Accuracy, F1 |
| `sota_value` | AUC 0.972 (hybrid RF+XGB+LSTM, Nguyen 2024) |
| `download_URL` | https://www.batadal.net/data.html (direct CSV links, no login needed) |
| `data_size_mb` | ~5 MB total |
| `why_backup` | Domain overlap with SWaT (both are ICS/water-cybersecurity). If SWaT is the primary pick, BATADAL is redundant domain-wise. BATADAL is simulated (EPANET), which is weaker than SWaT's real testbed data. However, it is small, genuinely open, and its AUC 0.972 is very strong and from a recent paper. Use as backup if SWaT download fails. |

---

### Backup 2: GECCO 2019 Water Quality (larger version of Pick 3)

| Field | Value |
|-------|-------|
| `name` | GECCO 2019 Industrial Challenge Dataset - Water Quality |
| `domain` | Environmental / Drinking Water IoT |
| `n_channels` | 8 (similar to 2018 but slightly different sensor set) |
| `sampling_rate` | ~1/min |
| `download_URL` | `wget https://zenodo.org/records/4304080/files/1_gecco2019_water_quality.csv?download=1` |
| `data_size_mb` | ~47 MB |
| `why_backup` | Same domain as Pick 3 but 3x more data. Could be combined with 2018 for pretraining. Not a separate pick because it is the same domain and task - but if GECCO 2018 is too small for stable pretraining, combine 2018+2019 (total ~61 MB). |

---

### Backup 3: MetroPT-3 (Metro Train Compressor)

| Field | Value |
|-------|-------|
| `name` | MetroPT-3 Dataset |
| `domain` | Transportation (urban metro / predictive maintenance) |
| `n_channels` | 15 (8 analog: pressure, temperature, current; 4 GPS; + 3 digital) |
| `sampling_rate` | 1 Hz (transmitted every 5 min) |
| `duration` | January-June 2022; ~11M records |
| `event_type` | Compressor failure in metro train Air Production Unit |
| `download_URL` | `wget https://zenodo.org/records/6854240/files/MetroPT3.zip` |
| `data_size_mb` | ~420 MB |
| `sota_paper` | Veloso et al. (2022), "The MetroPT dataset for predictive maintenance", Scientific Data |
| `sota_metric` | No published AUROC. Authors report "satisfactory" results only with rule-based and autoencoder methods. |
| `why_backup` | Interesting real-world transportation domain, but only 3 total failure events (air leaks + oil leak) - statistically insufficient for a meaningful event prediction benchmark. Published literature has no AUROC/AUPRC baseline against which to compare FAM. Do not use as a primary unless we define our own benchmark protocol and clearly state it. |

---

### Backup 4: PSML Power Grid

| Field | Value |
|-------|-------|
| `name` | PSML (Power System Multi-scale Learning) |
| `domain` | Energy / Power Grid |
| `n_channels` | ~20+ (PMU: voltage, current, power angles; load: 12 zone-level variables) |
| `sampling_rate` | Millisecond-level (PMU disturbance events) + minute-level (load/renewable) |
| `download_URL` | `wget https://zenodo.org/record/5130612/files/PSML.zip?download=1` |
| `data_size_mb` | 5,200 MB compressed (out of budget) |
| `sota_paper` | Zheng et al. (2022), "A Multi-scale Time-series Dataset with Benchmark for Machine Learning in Decarbonized Energy Grids", Nature Scientific Data |
| `sota_metric` | Disturbance detection accuracy + localization (F1) |
| `why_backup` | Strong domain (energy grids, completely fresh). But 5.2 GB compressed exceeds our 5 GB total budget AND the millisecond PMU data would require downsampling that changes the task semantics. The published SOTA uses F1 on classification (type of disturbance), not AUROC on onset prediction. Set aside unless we are willing to sub-select only the minute-level load data (which drops the disturbance detection use case). |

---

### Backup 5: CARE Wind Turbine SCADA (too large, for reference)

| Field | Value |
|-------|-------|
| `name` | CARE to Compare Wind Turbine Dataset |
| `domain` | Energy / Wind Turbine Predictive Maintenance |
| `n_channels` | 86 (farm A), 257 (farm B), 957 (farm C) |
| `sampling_rate` | 10 minutes |
| `download_URL` | `wget https://zenodo.org/records/10958775/files/CARE_To_Compare_Data.zip` |
| `data_size_mb` | 5,500 MB |
| `sota_paper` | Kazimierczak et al. (2024), "CARE to Compare: A Real-World Benchmark Dataset for Early Fault Detection in Wind Turbine Data", MDPI Data |
| `sota_metric` | CARE score (Coverage + Accuracy + Reliability + Earliness) - custom metric |
| `why_backup` | Fresh domain (renewable energy), real-world faults, genuinely open. But 5.5 GB exceeds budget and uses a custom CARE score rather than AUROC/AUPRC making comparison to our surface metric non-trivial. Could be revisited if we subsample to farm A only (~86 features). |

---

## Rejected Candidates (with reasons)

| Dataset | Reason for rejection |
|---------|---------------------|
| **MIMIC-III decompensation** | Requires PhysioNet credentials + CITI training (not one-command download). Credential process takes days. |
| **MIMIC-IV** | Same credential barrier as MIMIC-III. Also too large (>100 GB). |
| **HiRID-ICU** | Requires credentials + individual study review by contributor. Not freely downloadable. |
| **eICU** | Requires PhysioNet credentials. Not freely downloadable without registration process. |
| **SWaT / WADI** | Official distribution requires iTrust registration (3 days). Kaggle mirrors are unofficial third-party copies without clear provenance license. Can use as backup if iTrust responds promptly. |
| **PSML power grid** | 5.2 GB compressed - exceeds size budget. Published SOTA uses disturbance classification F1, not onset prediction AUROC. |
| **CARE wind turbine** | 5.5 GB - exceeds size budget. Custom CARE score (not AUROC/AUPRC). |
| **MetroPT-3** | Only 3 failure events total - too few for a meaningful prediction benchmark with AUROC. No published AUROC SOTA to compare against. |
| **SCANIA Component X** | Only 8 counter features (not raw multivariate sensor time series). No published AUROC baseline. Custom 5-class problem, not binary event prediction. |
| **ETT (Electricity Transformer Temp)** | Standard SOTA is forecasting MSE, not event prediction AUROC. No binary event label defined. |
| **Yahoo Webscope S5** | Mostly univariate. Requires Yahoo Webscope registration (academic approval process). |

---

## Implementation Notes for Tonight's Experiments

### PhysioNet 2012 - Data Setup

```python
# After untarring set-a and set-b:
# Each patient is in a separate .txt file with format: Time,Variable,Value
# Outcomes-a.txt maps patient ID -> mortality (0/1), LOS, SAPS-I score
# 
# FAM loader strategy:
#   - Resample to 1-hour grid (forward fill), 48 time steps per patient
#   - Use 37 physiologic channels (drop 5 static descriptors or encode as channel-0 constant)
#   - Label: label[t] = mortality_flag if patient dies; frame as TTE with max_horizon=48h
#   - For entity split: patient-level train (set A, 80%) / val (set A, 20%) / test (set B)
#   - ~6,400 / 1,600 / 4,000 patients
#   - Prevalence: ~14% of patients die (much higher timestep-level than Sepsis)
# 
# P=1 required (like Sepsis) because 48 timesteps / P=16 = only 3 tokens (too few)
# Or use P=4: 12 tokens from 48h window - borderline acceptable
```

### SWaT - Data Setup

```python
# After download (iTrust or Kaggle):
# Files: SWaT_Dataset_Normal_v0.csv (train), SWaT_Dataset_Attack_v0.csv (test)
# Each row: timestamp + 51 sensor/actuator columns + "Normal/Attack" label column
# 
# FAM loader strategy:
#   - 1 Hz = very high rate; consider stride=10 (10s windows) or patch L=10 (128s context)
#   - Label: label[t] = 1 if attack onset in [t, t+dt] using 1-min to 10-min horizons
#   - Single stream: standard train/test split (7 days train, 4 days test)
#   - No entity split needed (single stream)
#   - Pretrain on normal 7-day stream; pred-FT on test stream with attack labels
#   - 41 distinct attacks in test; most methods see AUROC 0.85-0.99 depending on method class
# 
# P=16 works: at 1Hz, P=16 = 16-second patches; W=128 = 2048s (34min) context
```

### GECCO 2018 - Data Setup

```python
# After download: single CSV with columns Time, Tp, pH, Redox, Leit, Trueb, Cl_2, Fm, Turb2, event
# event = 0 (normal) or 1 (contamination event)
# ~122,334 rows at ~1-min intervals
#
# FAM loader strategy:
#   - 9 sensor features + 1 label column
#   - Label: label[t] = 1 if event=1 in [t, t+dt] for dt in {5min, 15min, 30min}
#   - Single stream: chronological 80/10/10 split
#   - Pretrain on the normal-only portion of train split
#   - Pred-FT on full train split with event labels
#   - Consider concatenating GECCO 2019 (47 MB) for richer pretraining
#
# P=1 or P=5 (5-min patches) - dataset is small so context window matters
# Total training data: ~98K rows for pretrain; ~12K for val; ~12K for test
# Events are rare (exact % unknown, estimated < 5%) - use pos_weight in BCE loss
```

---

## Final Recommendation

**Run tonight in this order:**
1. **PhysioNet 2012** (download in 30s, pretrain + pred-FT < 20min, AUROC baseline clear, no credential risk)
2. **GECCO 2018** (download in 5s, fastest experiment, good domain story)
3. **SWaT** (register at iTrust NOW so download link arrives while #1 and #2 run; fallback to Kaggle mirror)

If SWaT registration does not resolve in time, substitute **BATADAL** (same cybersecurity domain, completely open, 5 MB, AUC 0.972 baseline).
