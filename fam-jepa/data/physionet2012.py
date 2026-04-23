"""PhysioNet / CinC Challenge 2012: in-hospital mortality prediction.

8,000 ICU patients with 48 hours of multivariate clinical recordings.
Distinct from PhysioNet 2019 Sepsis: the event is in-hospital death, not
sepsis onset; the prediction window is the full 48h ICU stay rather than
pre-onset dynamics.

File layout on disk:
  datasets/data/physionet2012/set-a/p*.txt      (4000 patients, train)
  datasets/data/physionet2012/set-b/p*.txt      (4000 patients, test)
  datasets/data/physionet2012/Outcomes-a.txt
  datasets/data/physionet2012/Outcomes-b.txt

Each patient file is CSV with columns: Time,Parameter,Value. Time is HH:MM
since admission. We resample to a 1-hour grid (forward-fill) to get a 48-step
multivariate series per patient.

Labels: In-hospital_death in {0, 1}. For the streaming event-prediction
framing, label[t] = 1 for all t (for deceased patients) - the "event" is
the mortality flag for this stay, and our surface p(t, Δt) answers "given
history up to hour t, will this stay end in death?"

Unlike sepsis, there is no per-hour onset time; the label is constant across
the stay for a given patient. TTE framing: for deceased patients, TTE at
hour t = discharge_hour - t (>= 0); for survivors, TTE = inf.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

try:
    from .config import _ROOT as CONFIG_ROOT
    P2012_DIR = CONFIG_ROOT / 'datasets' / 'data' / 'physionet2012'
except Exception:
    P2012_DIR = Path('/home/sagemaker-user/IndustrialJEPA/datasets/data/physionet2012')

# Time-series clinical variables (dropping 5 static descriptors:
# RecordID, Age, Gender, Height, ICUType, Weight)
CLINICAL_COLS = [
    'GCS', 'HR', 'NIDiasABP', 'NIMAP', 'NISysABP', 'RespRate', 'Temp',
    'Urine', 'SysABP', 'DiasABP', 'MAP', 'Albumin', 'ALP', 'ALT', 'AST',
    'Bilirubin', 'BUN', 'Cholesterol', 'Creatinine', 'FiO2', 'Glucose',
    'HCO3', 'HCT', 'K', 'Lactate', 'Mg', 'MechVent', 'Na', 'PaCO2',
    'PaO2', 'pH', 'Platelets', 'SaO2', 'TroponinI', 'TroponinT', 'WBC',
    'WBC',
]
# de-dup
CLINICAL_COLS = list(dict.fromkeys(CLINICAL_COLS))
N_CHANNELS = len(CLINICAL_COLS)

MAX_HOURS = 48
STATIC_COLS = {'RecordID', 'Age', 'Gender', 'Height', 'ICUType', 'Weight'}


def _parse_patient_file(path: Path) -> pd.DataFrame:
    """Parse one PhysioNet 2012 patient .txt file.

    Returns a (MAX_HOURS, n_channels) DataFrame, NaN where missing. Time is
    quantized to integer hour (0-47). Static rows (Age, Gender, etc.) are
    dropped. Multiple observations within the same hour are averaged.
    """
    raw = pd.read_csv(path)
    if raw.empty:
        return pd.DataFrame(columns=CLINICAL_COLS)
    raw = raw[~raw['Parameter'].isin(STATIC_COLS)].copy()
    if raw.empty:
        return pd.DataFrame(columns=CLINICAL_COLS)
    # Replace sentinel missing-values (PhysioNet uses -1 for missing in some
    # static fields; clinical values with -1 on weight/temp are OK for us
    # since we dropped static above). No transformation needed here.
    raw['hour'] = raw['Time'].str.split(':').str[0].astype(int)
    raw = raw[raw['hour'] < MAX_HOURS]
    # Pivot to (hour, parameter) wide
    wide = raw.pivot_table(index='hour', columns='Parameter', values='Value',
                            aggfunc='mean')
    # Reindex to 0..MAX_HOURS-1 and to our canonical column set
    wide = wide.reindex(index=range(MAX_HOURS), columns=CLINICAL_COLS)
    return wide


def _load_split(split_dir: Path, outcomes_path: Path,
                verbose: bool = False) -> List[Dict]:
    outcomes = pd.read_csv(outcomes_path)
    outcomes = outcomes.set_index('RecordID')
    patients = []
    all_files = sorted(split_dir.glob('p*.txt')) + sorted(split_dir.glob('[0-9]*.txt'))
    for i, pf in enumerate(all_files):
        rid = int(pf.stem.lstrip('p'))
        if rid not in outcomes.index:
            continue
        row = outcomes.loc[rid]
        # In-hospital death is the downstream label. Length_of_stay is used to
        # derive TTE: for deceased patients, label all 48 hours as 1.
        death = int(row['In-hospital_death'])
        los = int(row['Length_of_stay'])
        wide = _parse_patient_file(pf)
        if wide.empty:
            continue
        X = wide.to_numpy(dtype=np.float32)  # (48, n_channels), NaN-riddled
        # Forward-fill then zero-fill (standard for PhysioNet 2012)
        Xdf = pd.DataFrame(X, columns=CLINICAL_COLS).ffill().fillna(0.0)
        X = Xdf.to_numpy(dtype=np.float32)
        # Label: for deceased, event within the 48h window (label[t] = 1 for
        # all t = 0..47). For survivors, label[t] = 0 for all t.
        y = np.full(MAX_HOURS, int(death), dtype=np.int32)
        patients.append({
            'entity_id': f'p{rid}',
            'x': X,
            'labels': y,
            'death': death,
            'los_days': los,
        })
        if verbose and (i + 1) % 1000 == 0:
            print(f'    loaded {i+1}/{len(all_files)}', flush=True)
    return patients


def _normalize(patients: List[Dict], mu: np.ndarray, std: np.ndarray) -> List[Dict]:
    out = []
    for p in patients:
        x = (p['x'] - mu) / std
        out.append({**p, 'x': x.astype(np.float32)})
    return out


def load_physionet2012(val_frac: float = 0.2, val_seed: int = 42,
                        verbose: bool = True,
                        pretrain_from: str = 'survivors_setA') -> Dict:
    """Load PhysioNet 2012 with patient-level splits.

    pretrain_from:
      'survivors_setA': use non-deceased set-A ft_train patients only.
      'all_setA':       use all set-A ft_train patients (label-agnostic).
    """
    if not (P2012_DIR / 'set-a').exists():
        raise FileNotFoundError(
            f'PhysioNet 2012 set-a not found at {P2012_DIR}/set-a. '
            f"Run wget + tar to populate {P2012_DIR}.")

    if verbose:
        print('  loading set-a patient files...', flush=True)
    pa = _load_split(P2012_DIR / 'set-a', P2012_DIR / 'Outcomes-a.txt',
                      verbose=verbose)
    if verbose:
        print(f'  set-a: {len(pa)} patients', flush=True)
        print('  loading set-b patient files...', flush=True)
    pb = _load_split(P2012_DIR / 'set-b', P2012_DIR / 'Outcomes-b.txt',
                      verbose=verbose)
    if verbose:
        print(f'  set-b: {len(pb)} patients', flush=True)

    # Patient-level train/val split on set A, fixed seed so splits are stable
    ids = sorted(p['entity_id'] for p in pa)
    rng = np.random.default_rng(val_seed)
    n_val = int(round(val_frac * len(ids)))
    perm = rng.permutation(len(ids))
    val_ids = {ids[i] for i in perm[:n_val]}
    ft_train_raw = [p for p in pa if p['entity_id'] not in val_ids]
    ft_val_raw   = [p for p in pa if p['entity_id'] in val_ids]

    # Pretrain cohort
    if pretrain_from == 'survivors_setA':
        pretrain_raw = [p for p in ft_train_raw if p['death'] == 0]
    elif pretrain_from == 'all_setA':
        pretrain_raw = ft_train_raw
    else:
        raise ValueError(pretrain_from)

    # Normalization stats from pretrain cohort
    X_stack = np.concatenate([p['x'] for p in pretrain_raw], axis=0)
    mu = X_stack.mean(axis=0).astype(np.float32)
    std = X_stack.std(axis=0).astype(np.float32)
    std = np.where(std < 1e-4, 1.0, std)
    if verbose:
        print(f'  pretrain cohort ({pretrain_from}): {len(pretrain_raw)} patients, '
              f'{len(X_stack)} timesteps', flush=True)

    pretrain_patients = _normalize(pretrain_raw, mu, std)
    ft_train = _normalize(ft_train_raw, mu, std)
    ft_val   = _normalize(ft_val_raw,   mu, std)
    ft_test  = _normalize(pb,           mu, std)

    # Per-patient mortality prevalence sanity
    for split, ps in [('ft_train', ft_train), ('ft_val', ft_val), ('ft_test', ft_test)]:
        rate = sum(p['death'] for p in ps) / max(len(ps), 1)
        if verbose:
            print(f'  {split}: {len(ps)} patients, mortality {rate:.4f}',
                  flush=True)

    return {
        'pretrain_patients': pretrain_patients,
        'ft_train': ft_train, 'ft_val': ft_val, 'ft_test': ft_test,
        'n_channels': N_CHANNELS, 'mu': mu, 'std': std,
        'channel_names': CLINICAL_COLS,
        'name': 'physionet2012',
    }


if __name__ == '__main__':
    d = load_physionet2012()
    print('channels:', d['n_channels'])
    print('pretrain:', len(d['pretrain_patients']))
    print('ft_train/val/test:', len(d['ft_train']), len(d['ft_val']),
          len(d['ft_test']))
    Ts = [len(p['x']) for p in d['ft_test']]
    print(f'stay length (test): min={min(Ts)} max={max(Ts)} '
          f'mean={np.mean(Ts):.1f} median={np.median(Ts):.1f}')
