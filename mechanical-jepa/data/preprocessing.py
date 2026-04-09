"""
Unified preprocessing pipeline for bearing vibration signals.
Supports: resample, bandpass filter, spectral whitening, instance/source normalization.
"""

import numpy as np
from scipy.signal import resample_poly, butter, sosfilt
from math import gcd
from typing import Optional, Tuple

TARGET_SR = 12800
WINDOW_LEN = 1024


def resample_to_target(ch: np.ndarray, native_sr: int,
                        target_sr: int = TARGET_SR) -> np.ndarray:
    """Polyphase resampling to target SR."""
    if native_sr == target_sr:
        return ch.astype(np.float32)
    g = gcd(int(native_sr), int(target_sr))
    up = target_sr // g
    down = native_sr // g
    return resample_poly(ch, up, down).astype(np.float32)


def bandpass_filter(ch: np.ndarray, sr: int,
                    low_hz: float = 100.0, high_hz: float = 5000.0) -> np.ndarray:
    """Butterworth bandpass filter."""
    nyq = sr / 2.0
    low = low_hz / nyq
    high = min(high_hz / nyq, 0.99)
    sos = butter(4, [low, high], btype='band', output='sos')
    return sosfilt(sos, ch).astype(np.float32)


def instance_norm(x: np.ndarray) -> Optional[np.ndarray]:
    """Zero-mean, unit-std normalization per window."""
    std = x.std()
    if std < 1e-10:
        return None
    return ((x - x.mean()) / std).astype(np.float32)


def spectral_whitening(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Flatten the power spectrum (spectral whitening).
    Divides by the magnitude spectrum so all frequencies have equal energy.
    Preserves phase structure (waveform texture) while removing spectral tilt.
    """
    X = np.fft.rfft(x)
    magnitude = np.abs(X)
    magnitude_smoothed = np.maximum(magnitude, eps)
    X_whitened = X / magnitude_smoothed
    return np.fft.irfft(X_whitened, n=len(x)).astype(np.float32)


def extract_windows(signal: np.ndarray, win_len: int = WINDOW_LEN,
                    max_windows: int = 20,
                    do_bandpass: bool = False,
                    sr: int = TARGET_SR,
                    do_whitening: bool = False) -> list:
    """
    Extract non-overlapping windows from a signal.
    Applies preprocessing options in order: bandpass → whiten → instance_norm.
    """
    if do_bandpass:
        signal = bandpass_filter(signal, sr)

    n = len(signal)
    if n < win_len:
        return []
    n_wins = min(n // win_len, max_windows)
    windows = []
    for i in range(n_wins):
        w = signal[i * win_len:(i + 1) * win_len].copy()
        if do_whitening:
            w = spectral_whitening(w)
        w_norm = instance_norm(w)
        if w_norm is not None:
            windows.append(w_norm)
    return windows


def compute_spectral_features(x: np.ndarray, sr: int = TARGET_SR) -> dict:
    """
    Compute per-window spectral features for compatibility analysis.
    Returns dict with: centroid, bandwidth, kurtosis, rms, crest_factor, band_energies.
    """
    from scipy.stats import kurtosis as scipy_kurtosis, skew as scipy_skew

    n = len(x)
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)
    psd = np.abs(np.fft.rfft(x)) ** 2
    psd_norm = psd / (psd.sum() + 1e-10)

    centroid = float(np.sum(freqs * psd_norm))
    bandwidth = float(np.sqrt(np.sum((freqs - centroid) ** 2 * psd_norm)))

    rms = float(np.sqrt(np.mean(x ** 2)))
    peak = float(np.max(np.abs(x)))
    crest_factor = peak / (rms + 1e-10)

    kurt = float(scipy_kurtosis(x, fisher=True))
    skewness = float(scipy_skew(x))

    # Band energies
    bands = [(0, 500), (500, 2000), (2000, 5000), (5000, sr // 2)]
    band_energies = {}
    total_energy = psd.sum() + 1e-10
    for lo, hi in bands:
        mask = (freqs >= lo) & (freqs < hi)
        band_energies[f'{lo}-{hi}Hz'] = float(psd[mask].sum() / total_energy)

    return {
        'centroid': centroid,
        'bandwidth': bandwidth,
        'rms': rms,
        'crest_factor': crest_factor,
        'kurtosis': kurt,
        'skewness': skewness,
        'band_energies': band_energies,
        'psd': psd,
        'freqs': freqs,
    }
