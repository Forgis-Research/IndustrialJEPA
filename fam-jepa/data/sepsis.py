"""PhysioNet Computing in Cardiology 2019 Sepsis Challenge loader.

Early prediction of sepsis in ICU patients.  Each patient has a variable-
length hourly stream (6 - 336 hours) of 40 clinical variables, with a
binary SepsisLabel that flips 0 -> 1 at the sepsis onset time (if any).

Data layout on disk:
  DATASETS/data/sepsis/training/training_setA/p{NNNNNN}.psv    (~20k stays)
  DATASETS/data/sepsis/training/training_setB/p{NNNNNN}.psv    (~20k stays)

Each .psv is pipe-separated with a header row:
  HR|O2Sat|Temp|...|Platelets|Age|Gender|Unit1|Unit2|HospAdmTime|ICULOS|SepsisLabel
  = 34 clinical variables + 6 static/admin + SepsisLabel

Splits per the session prompt:
  pretrain     : set A patients with no sepsis (pre-onset only kept when a
                 patient does develop sepsis).
  ft_train     : 80% of set A (patient-level split, shuffled by seed)
  ft_val       : 20% of set A
  ft_test      : all of set B

Preprocessing:
  - Drop 6 static/admin columns (Age, Gender, Unit1, Unit2, HospAdmTime,
    ICULOS) -> 34 channels.
  - Forward-fill within each patient, then fill remaining NaN with 0.
  - Per-channel z-score normalize using set A pretrain normals.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from .config import _ROOT as CONFIG_ROOT
    SEPSIS_DIR = CONFIG_ROOT / 'datasets' / 'data' / 'sepsis'
except Exception:
    SEPSIS_DIR = Path('/home/sagemaker-user/IndustrialJEPA'
                      '/datasets/data/sepsis')

SETA_DIR = SEPSIS_DIR / 'training' / 'training_setA'
SETB_DIR = SEPSIS_DIR / 'training' / 'training_setB'

DEMOGRAPHIC_COLS = ['Age', 'Gender', 'Unit1', 'Unit2', 'HospAdmTime', 'ICULOS']
LABEL_COL = 'SepsisLabel'

# The 34 clinical variables that remain after dropping demographics/label.
CLINICAL_COLS = [
    'HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'EtCO2',
    'BaseExcess', 'HCO3', 'FiO2', 'pH', 'PaCO2', 'SaO2',
    'AST', 'BUN', 'Alkalinephos', 'Calcium', 'Chloride', 'Creatinine',
    'Bilirubin_direct', 'Glucose', 'Lactate', 'Magnesium', 'Phosphate',
    'Potassium', 'Bilirubin_total', 'TroponinI', 'Hct', 'Hgb',
    'PTT', 'WBC', 'Fibrinogen', 'Platelets',
]
N_CHANNELS = len(CLINICAL_COLS)


def _load_patient_file(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load one .psv and return (X, y).

    X : (T, 34)  float32, forward-filled then zero-filled clinical values.
    y : (T,)     int32    SepsisLabel.
    """
    df = pd.read_csv(path, sep='|')
    # Keep only clinical columns + label
    X = df[CLINICAL_COLS].to_numpy(dtype=np.float32)
    y = df[LABEL_COL].to_numpy(dtype=np.int32)
    # Forward-fill then zero-fill (standard for sepsis challenge).
    # pandas ffill on the DataFrame subset is cleanest:
    X_df = pd.DataFrame(X, columns=CLINICAL_COLS).ffill().fillna(0.0)
    X = X_df.to_numpy(dtype=np.float32)
    return X, y


def _list_patients(d: Path) -> List[Path]:
    return sorted(d.glob('p*.psv'))


def _split_patients_A(paths: List[Path], train_frac: float,
                      seed: int) -> Tuple[List[Path], List[Path]]:
    rng = np.random.default_rng(seed)
    idx = np.arange(len(paths))
    rng.shuffle(idx)
    n_tr = int(round(train_frac * len(paths)))
    return [paths[i] for i in idx[:n_tr]], [paths[i] for i in idx[n_tr:]]


def _load_many(paths: List[Path], verbose: bool = False) -> List[Dict]:
    """Load a list of patient files into a list of dicts."""
    patients = []
    for i, p in enumerate(paths):
        X, y = _load_patient_file(p)
        if len(X) < 2:                 # skip empty / degenerate stays
            continue
        patients.append({
            'entity_id': p.stem,       # 'pNNNNNN'
            'x': X,                    # (T, 34)
            'labels': y,               # (T,)
            'has_sepsis': bool(y.any()),
            'onset_t': int(y.argmax()) if y.any() else -1,
        })
        if verbose and (i + 1) % 2000 == 0:
            print(f'    loaded {i+1}/{len(paths)}', flush=True)
    return patients


def _normalize(patients: List[Dict], mu: np.ndarray, std: np.ndarray
               ) -> List[Dict]:
    out = []
    for p in patients:
        x = (p['x'] - mu) / std
        out.append({**p, 'x': x.astype(np.float32)})
    return out


def load_sepsis(train_frac: float = 0.8, seed: int = 42,
                verbose: bool = True,
                pretrain_from: str = 'nonseptic_setA',
                ) -> Dict[str, object]:
    """Load PhysioNet 2019 Sepsis Challenge with patient-level splits.

    pretrain_from:
      'nonseptic_setA'  - use only non-septic set-A ft_train patients for
                          unsupervised pretraining (prevents leaking sepsis
                          into JEPA representations).
      'pre_onset_setA'  - use all set-A ft_train patients, but for septic
                          ones truncate to [0, onset) so the context never
                          sees sepsis hours.
      'all_setA'        - raw set-A ft_train patients (may include sepsis
                          periods in context).

    Returns:
      {
        'pretrain_patients' : [ {'entity_id', 'x'(T,34)} ]  — normalize-ready
        'ft_train': [ {entity_id, x, labels} ],     (set A, 80%)
        'ft_val'  : [ {entity_id, x, labels} ],     (set A, 20%)
        'ft_test' : [ {entity_id, x, labels} ],     (set B)
        'n_channels': 34, 'mu': (34,), 'std': (34,),
      }
    """
    if not SETA_DIR.exists():
        raise FileNotFoundError(
            f'Sepsis set A not found at {SETA_DIR}. '
            f'Run: aws s3 sync s3://physionet-open/challenge-2019/1.0.0/'
            f'training/training_setA/ {SETA_DIR}/ --no-sign-request')
    if not SETB_DIR.exists():
        raise FileNotFoundError(f'Sepsis set B not found at {SETB_DIR}')

    pa_paths = _list_patients(SETA_DIR)
    pb_paths = _list_patients(SETB_DIR)
    if verbose:
        print(f'  set A: {len(pa_paths)} patients, '
              f'set B: {len(pb_paths)} patients', flush=True)

    # Patient-level 80/20 split on set A
    pa_tr, pa_va = _split_patients_A(pa_paths, train_frac, seed)
    if verbose:
        print(f'  ft_train={len(pa_tr)} ft_val={len(pa_va)} '
              f'ft_test={len(pb_paths)}', flush=True)

    # Load (no normalization yet)
    ft_train = _load_many(pa_tr, verbose=verbose)
    ft_val   = _load_many(pa_va, verbose=verbose)
    ft_test  = _load_many(pb_paths, verbose=verbose)

    # Pretrain cohort selection (normals-only for cleanest pretraining).
    if pretrain_from == 'nonseptic_setA':
        pretrain_raw = [p for p in ft_train if not p['has_sepsis']]
    elif pretrain_from == 'pre_onset_setA':
        pretrain_raw = []
        for p in ft_train:
            if p['has_sepsis']:
                t_end = p['onset_t']
                if t_end < 2:
                    continue
                pretrain_raw.append({
                    **p, 'x': p['x'][:t_end], 'labels': p['labels'][:t_end],
                })
            else:
                pretrain_raw.append(p)
    elif pretrain_from == 'all_setA':
        pretrain_raw = ft_train
    else:
        raise ValueError(f'pretrain_from={pretrain_from!r}')
    if verbose:
        print(f'  pretrain cohort ({pretrain_from}): {len(pretrain_raw)} '
              f'patients', flush=True)

    # Compute normalization stats on the pretrain cohort (no label leakage).
    all_x = np.concatenate([p['x'] for p in pretrain_raw], axis=0)
    mu = all_x.mean(axis=0).astype(np.float32)
    std = all_x.std(axis=0).astype(np.float32)
    std = np.where(std < 1e-4, 1.0, std)
    if verbose:
        print(f'  normalization: mu/std computed on {len(all_x)} '
              f'pretrain timesteps', flush=True)

    # Normalize all splits
    pretrain_patients = _normalize(pretrain_raw, mu, std)
    ft_train = _normalize(ft_train, mu, std)
    ft_val   = _normalize(ft_val,   mu, std)
    ft_test  = _normalize(ft_test,  mu, std)

    # Sanity rates
    def prev(ps):
        total = sum(len(p['labels']) for p in ps)
        pos = sum(int(p['labels'].sum()) for p in ps)
        return pos / max(total, 1), total, pos
    pr_tr, n_tr, pos_tr = prev(ft_train)
    pr_va, n_va, pos_va = prev(ft_val)
    pr_te, n_te, pos_te = prev(ft_test)
    if verbose:
        print(f'  timestep-level prevalence: '
              f'ft_train={pr_tr:.4f} ({pos_tr}/{n_tr})  '
              f'ft_val={pr_va:.4f} ({pos_va}/{n_va})  '
              f'ft_test={pr_te:.4f} ({pos_te}/{n_te})', flush=True)

    return {
        'pretrain_patients': pretrain_patients,
        'ft_train': ft_train, 'ft_val': ft_val, 'ft_test': ft_test,
        'n_channels': N_CHANNELS, 'mu': mu, 'std': std,
        'channel_names': CLINICAL_COLS,
        'name': 'sepsis',
    }


if __name__ == '__main__':
    # Sanity check
    d = load_sepsis()
    print('channels:', d['n_channels'])
    print('pretrain:', len(d['pretrain_patients']))
    print('ft_train/val/test:',
          len(d['ft_train']), len(d['ft_val']), len(d['ft_test']))
    ps = d['ft_test']
    Ts = [len(p['x']) for p in ps]
    print(f'stay length (test): min={min(Ts)} max={max(Ts)} '
          f'mean={np.mean(Ts):.1f} median={np.median(Ts):.1f}')
    n_sept = sum(1 for p in d['ft_test'] if p['has_sepsis'])
    print(f'septic patients in test: {n_sept}/{len(d["ft_test"])}')
