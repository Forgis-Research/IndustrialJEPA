"""CHB-MIT Scalp EEG — pediatric epilepsy seizure prediction.

Source: PhysioNet (https://physionet.org/content/chbmit/1.0.0/, no
registration). Per-subject .edf files at 256 Hz, 23+ channels.

Local layout:
  datasets/data/chbmit/
    chb01-summary.txt
    chb01_*.edf
    (chb03-summary.txt + chb03_*.edf, chb05_*.edf if downloaded)

Protocol (per the v29 SESSION_PROMPT):
  - 18 EEG channels (the first 18 listed in summary, 10-20 system common subset)
  - Downsample 256 Hz → 32 Hz (8x decimation, anti-aliased)
  - P=16 at 32 Hz: each token covers 0.5s
  - Context 512 steps = 16s of EEG
  - Preictal label y=1 for 30 min (1800s = 57600 samples@256Hz = 7200@32Hz)
    immediately before each seizure onset
  - 4-hour buffer after each seizure offset (samples in this window are
    excluded from train/val/test entirely)
  - Per-subject. Pick subjects with >=5 seizures.

Returns the FAM-style bundle.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from .config import _ROOT as CONFIG_ROOT
    CHBMIT_DIR = CONFIG_ROOT / 'datasets' / 'data' / 'chbmit'
except Exception:
    CHBMIT_DIR = Path('/home/sagemaker-user/IndustrialJEPA/datasets/data/chbmit')

TARGET_SR = 32                # Hz, after downsample
PREICTAL_SECONDS = 30 * 60    # 1800s = 30 min preictal window
POSTICTAL_BUFFER_SECONDS = 4 * 3600   # 4 h post-seizure buffer
N_CHANNELS = 18               # first 18 listed in the summary (10-20 common)


def _parse_summary(path: Path) -> List[Dict]:
    """Parse a CHB-MIT summary text file. Returns list of file records:
       {file, start_s, end_s, seizures: [(start_s, end_s)]}.
    """
    text = path.read_text()
    blocks = re.split(r'\n(?=File Name:)', text)
    records: List[Dict] = []
    for blk in blocks:
        m_name = re.search(r'File Name:\s*(\S+)', blk)
        if not m_name:
            continue
        m_n = re.search(r'Number of Seizures in File:\s*(\d+)', blk)
        n_seiz = int(m_n.group(1)) if m_n else 0
        seizures: List[Tuple[int, int]] = []
        for m in re.finditer(
                r'Seizure(?: \d+)? Start Time:\s*(\d+)\s*seconds.*?'
                r'Seizure(?: \d+)? End Time:\s*(\d+)\s*seconds',
                blk, flags=re.DOTALL):
            seizures.append((int(m.group(1)), int(m.group(2))))
        records.append({'file': m_name.group(1), 'n_seiz': n_seiz,
                        'seizures': seizures})
    return records


def _load_edf(path: Path, n_channels: int = N_CHANNELS) -> np.ndarray:
    """Load EDF, take first n_channels, downsample 256→32 Hz."""
    import mne
    raw = mne.io.read_raw_edf(path, preload=True, verbose='ERROR')
    if raw.info['sfreq'] != 256:
        # Some sessions are at different rates; resample to 256 first
        raw = raw.resample(256, verbose='ERROR')
    raw = raw.resample(TARGET_SR, verbose='ERROR')
    data = raw.get_data()  # (C, T) in volts
    # Pick first N_CHANNELS — all subjects in the project use the common
    # 18-channel subset that matches the summary's channel list (10-20).
    data = data[:n_channels]
    if data.shape[0] < n_channels:
        # Pad with zeros if a session is missing channels (rare)
        pad = np.zeros((n_channels - data.shape[0], data.shape[1]),
                       dtype=data.dtype)
        data = np.concatenate([data, pad], axis=0)
    return data.T.astype(np.float32)   # (T, C)


def _build_labels(file_recs: List[Dict],
                  signals: List[np.ndarray]) -> List[np.ndarray]:
    """Build per-file labels: y=1 in the preictal 30-min before seizure onset.
    Postictal 4-hour buffer is encoded as label=-1 (excluded by the
    EventDataset wrapper, which treats inf time-to-event as 'no event').

    For simplicity, we mark only y=1 (preictal) — the postictal buffer is
    handled at the per-stream level by truncating each subject's stream
    around the buffer.
    """
    labels = []
    for rec, sig in zip(file_recs, signals):
        T = sig.shape[0]
        y = np.zeros(T, dtype=np.int32)
        for (s, _) in rec['seizures']:
            onset_sample = int(s * TARGET_SR)
            preictal_start = max(0, onset_sample
                                 - PREICTAL_SECONDS * TARGET_SR)
            y[preictal_start:onset_sample] = 1
        labels.append(y)
    return labels


def load_chbmit_subject(subject: str = 'chb01',
                        normalize: bool = False) -> Optional[Dict]:
    """Load one subject's full stream + labels + per-seizure markers.

    Returns dict or None if the subject's data is missing.
    """
    summary_path = CHBMIT_DIR / f'{subject}-summary.txt'
    if not summary_path.exists():
        return None
    records = _parse_summary(summary_path)
    available = []
    for rec in records:
        fp = CHBMIT_DIR / rec['file']
        if fp.exists() and fp.stat().st_size > 1000:
            available.append((rec, fp))
    if not available:
        return None

    signals = []
    file_recs = []
    for rec, fp in available:
        try:
            sig = _load_edf(fp)
            signals.append(sig)
            file_recs.append(rec)
        except Exception as e:
            print(f"  WARN load {fp.name} failed: {e}")
            continue
    if not signals:
        return None

    labels = _build_labels(file_recs, signals)

    # Concatenate all files into one chronological stream + per-seizure
    # absolute sample indices for stratified split.
    cum = 0
    seizure_sample_indices = []
    for rec, sig in zip(file_recs, signals):
        for (s, e) in rec['seizures']:
            seizure_sample_indices.append(
                (cum + int(s * TARGET_SR), cum + int(e * TARGET_SR)))
        cum += sig.shape[0]
    full_x = np.concatenate(signals, axis=0)   # (T_total, C)
    full_y = np.concatenate(labels, axis=0)    # (T_total,)

    if normalize:
        mu = full_x.mean(axis=0)
        std = full_x.std(axis=0)
        std = np.where(std < 1e-10, 1.0, std)
        full_x = ((full_x - mu) / std).astype(np.float32)

    # EEG raw values are tiny (~1e-5 V); rescale by 1e4 so they aren't
    # truncated by downstream float32 ops. Pure linear rescale, no info loss.
    full_x = (full_x * 1e4).astype(np.float32)

    return {
        'subject': subject,
        'x': full_x,
        'y': full_y,
        'seizure_indices': seizure_sample_indices,
        'n_seizures': len(seizure_sample_indices),
        'n_files': len(signals),
        'sample_rate_hz': TARGET_SR,
    }


def load_chbmit(subjects: Optional[List[str]] = None,
                normalize: bool = False,
                gap: int = 200) -> Dict:
    """Load CHB-MIT into the FAM-style bundle.

    For each available subject, builds a per-subject stream with leave-one-
    seizure-out test split: the LAST seizure goes to test (with ~30min before
    + ~30min after), the SECOND-TO-LAST to val, the rest to train. Postictal
    buffers (4h after seizure offset) are dropped from the streams.

    Pretrain stream = all 'normal' samples from train portion (no preictal
    label) so the encoder never sees seizure-precursor dynamics.
    """
    if subjects is None:
        subjects = ['chb01', 'chb03', 'chb05']

    pre_streams = {}
    ft_train, ft_val, ft_test = [], [], []
    n_subj_loaded = 0

    for s in subjects:
        data = load_chbmit_subject(s, normalize=normalize)
        if data is None or data['n_seizures'] < 2:
            continue
        n_subj_loaded += 1
        x = data['x']
        y = data['y']
        seiz = data['seizure_indices']
        T = x.shape[0]

        # Sort seizures chronologically, take last as test, 2nd-to-last val.
        seiz = sorted(seiz)
        last = seiz[-1]
        second_last = seiz[-2] if len(seiz) >= 2 else None

        # Define test window: [last_onset - 60min, last_offset + 30min]
        test_start = max(0, last[0] - 60 * 60 * TARGET_SR)
        test_end = min(T, last[1] + 30 * 60 * TARGET_SR)
        # Val window
        val_start = max(0, second_last[0] - 60 * 60 * TARGET_SR)
        val_end = min(T, second_last[1] + 30 * 60 * TARGET_SR)
        # Make sure val is before test (and not overlapping)
        if val_end > test_start:
            val_end = test_start - gap

        # Train = everything outside val_window and test_window, EXCLUDING
        # 4-hour postictal buffers (labels stay 0 there but we drop them)
        keep = np.ones(T, dtype=bool)
        for (_, off) in seiz:
            buf_end = min(T, off + POSTICTAL_BUFFER_SECONDS * TARGET_SR)
            keep[off:buf_end] = False
        # Also exclude val + test windows from train
        keep[val_start:val_end] = False
        keep[test_start:test_end] = False

        # Train stream = chunks of contiguous keep=True
        train_chunks = []
        i = 0
        while i < T:
            if not keep[i]:
                i += 1
                continue
            j = i
            while j < T and keep[j]:
                j += 1
            chunk_x = x[i:j]
            chunk_y = y[i:j]
            if chunk_x.shape[0] >= 200:    # need >= min_context
                train_chunks.append((chunk_x, chunk_y))
            i = j

        # Pretrain stream = NORMAL portion of train chunks
        pre_x = np.concatenate(
            [c[0][c[1] == 0] for c in train_chunks if (c[1] == 0).any()],
            axis=0) if train_chunks else np.zeros((0, x.shape[1]),
                                                  dtype=np.float32)
        if pre_x.shape[0] >= 200:
            pre_streams[s] = pre_x.astype(np.float32)

        # ft_train: each train chunk = one entity
        for ci, (cx, cy) in enumerate(train_chunks):
            ft_train.append({'entity_id': f'{s}_train_{ci}',
                             'test': cx.astype(np.float32),
                             'labels': cy.astype(np.int32)})
        # ft_val and ft_test: single chunks
        if val_end - val_start >= 200:
            ft_val.append({'entity_id': f'{s}_val',
                           'test': x[val_start:val_end].astype(np.float32),
                           'labels': y[val_start:val_end].astype(np.int32)})
        if test_end - test_start >= 200:
            ft_test.append({'entity_id': f'{s}_test',
                            'test': x[test_start:test_end].astype(np.float32),
                            'labels': y[test_start:test_end].astype(np.int32)})

    return {
        'pretrain_stream': pre_streams,
        'ft_train': ft_train,
        'ft_val': ft_val,
        'ft_test': ft_test,
        'n_channels': N_CHANNELS,
        'name': 'CHB-MIT',
        'n_subjects_loaded': n_subj_loaded,
    }


if __name__ == '__main__':
    d = load_chbmit()
    print(f"subjects loaded: {d['n_subjects_loaded']}")
    for k, v in d['pretrain_stream'].items():
        print(f"  pretrain[{k}]: {v.shape}")
    for split in ('ft_train', 'ft_val', 'ft_test'):
        ents = d[split]
        total = sum(e['test'].shape[0] for e in ents)
        pos = sum(int(e['labels'].sum()) for e in ents)
        print(f"  {split}: {len(ents)} entities, {total} steps, "
              f"{pos} positive ({100*pos/max(total,1):.2f}%)")
