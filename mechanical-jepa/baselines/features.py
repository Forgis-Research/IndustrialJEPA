"""
Feature extraction toolkit for mechanical vibration baselines.

All features operate on 1D numpy arrays (single channel, single window).
Window size: 16384 samples at 12800 Hz.

Feature families:
- Time domain: 8 statistical features
- Frequency domain: spectral centroid, spread, entropy, band energies
- Envelope analysis: Hilbert transform RMS, kurtosis of envelope
- Aggregate: combined feature vector
"""

import numpy as np
from scipy.signal import hilbert
from scipy.stats import kurtosis as scipy_kurtosis, skew as scipy_skew
from scipy.fft import rfft, rfftfreq
from typing import Optional, Dict, List

TARGET_SR = 12800
WINDOW_LEN = 16384


# ============================================================
# BASIC TIME DOMAIN FEATURES
# ============================================================

def compute_rms(x: np.ndarray) -> float:
    """Root mean square."""
    return float(np.sqrt(np.mean(x ** 2)))


def compute_peak(x: np.ndarray) -> float:
    """Peak (max absolute value)."""
    return float(np.max(np.abs(x)))


def compute_crest_factor(x: np.ndarray) -> float:
    """Crest factor = peak / RMS."""
    rms = compute_rms(x)
    if rms < 1e-12:
        return 0.0
    return float(compute_peak(x) / rms)


def compute_kurtosis(x: np.ndarray) -> float:
    """Statistical kurtosis (excess kurtosis, Fisher definition)."""
    return float(scipy_kurtosis(x, fisher=True))


def compute_skewness(x: np.ndarray) -> float:
    """Statistical skewness."""
    return float(scipy_skew(x))


def compute_shape_factor(x: np.ndarray) -> float:
    """Shape factor = RMS / mean absolute value."""
    mean_abs = np.mean(np.abs(x))
    if mean_abs < 1e-12:
        return 0.0
    return float(compute_rms(x) / mean_abs)


def compute_impulse_factor(x: np.ndarray) -> float:
    """Impulse factor = peak / mean absolute value."""
    mean_abs = np.mean(np.abs(x))
    if mean_abs < 1e-12:
        return 0.0
    return float(compute_peak(x) / mean_abs)


def compute_clearance_factor(x: np.ndarray) -> float:
    """Clearance factor = peak / (mean sqrt absolute)^2."""
    mean_sqrt_abs = np.mean(np.sqrt(np.abs(x)))
    if mean_sqrt_abs < 1e-12:
        return 0.0
    return float(compute_peak(x) / (mean_sqrt_abs ** 2))


def time_domain_features(x: np.ndarray) -> np.ndarray:
    """Return 8 time-domain features as a vector."""
    return np.array([
        compute_rms(x),
        compute_peak(x),
        compute_crest_factor(x),
        compute_kurtosis(x),
        compute_skewness(x),
        compute_shape_factor(x),
        compute_impulse_factor(x),
        compute_clearance_factor(x),
    ], dtype=np.float32)


# ============================================================
# FREQUENCY DOMAIN FEATURES
# ============================================================

def compute_fft_features(x: np.ndarray, sr: int = TARGET_SR) -> np.ndarray:
    """
    Compute frequency domain features:
    - Spectral centroid
    - Spectral spread (RMS bandwidth)
    - Spectral entropy (normalized)
    - Band energies: [0-1kHz, 1-3kHz, 3-5kHz, 5-6.4kHz] as fraction of total
    Returns 7-element vector.
    """
    n = len(x)
    fft_mag = np.abs(rfft(x)) / n
    fft_pow = fft_mag ** 2
    freqs = rfftfreq(n, 1.0 / sr)

    total_power = fft_pow.sum()
    if total_power < 1e-20:
        return np.zeros(7, dtype=np.float32)

    # Spectral centroid
    centroid = float(np.sum(freqs * fft_pow) / total_power)

    # Spectral spread
    spread = float(np.sqrt(np.sum(((freqs - centroid) ** 2) * fft_pow) / total_power))

    # Spectral entropy
    p = fft_pow / total_power
    p = p[p > 0]
    entropy = float(-np.sum(p * np.log(p + 1e-20)) / np.log(len(fft_pow)))

    # Band energies as fraction of total
    nyq = sr / 2
    band_limits = [(0, 1000), (1000, 3000), (3000, 5000), (5000, nyq)]
    band_energies = []
    for flo, fhi in band_limits:
        mask = (freqs >= flo) & (freqs < fhi)
        band_e = float(fft_pow[mask].sum() / total_power)
        band_energies.append(band_e)

    return np.array([centroid, spread, entropy] + band_energies, dtype=np.float32)


# ============================================================
# ENVELOPE ANALYSIS
# ============================================================

def compute_envelope_features(x: np.ndarray) -> np.ndarray:
    """
    Hilbert transform envelope analysis:
    - Envelope RMS
    - Envelope kurtosis
    - Envelope peak
    Returns 3-element vector.
    """
    try:
        analytic = hilbert(x)
        envelope = np.abs(analytic)
        return np.array([
            compute_rms(envelope),
            compute_kurtosis(envelope),
            compute_peak(envelope),
        ], dtype=np.float32)
    except Exception:
        return np.zeros(3, dtype=np.float32)


# ============================================================
# COMBINED FEATURE VECTOR
# ============================================================

def extract_features(x: np.ndarray, sr: int = TARGET_SR) -> np.ndarray:
    """
    Extract full feature vector from a 1D vibration signal window.
    Total: 8 + 7 + 3 = 18 features.

    Input: 1D float array (typically WINDOW_LEN = 16384 samples)
    Output: 18-element float32 vector
    """
    td = time_domain_features(x)
    fd = compute_fft_features(x, sr)
    env = compute_envelope_features(x)
    feats = np.concatenate([td, fd, env])

    # Replace NaN/Inf with 0
    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
    return feats.astype(np.float32)


FEATURE_NAMES = [
    # Time domain (8)
    'rms', 'peak', 'crest_factor', 'kurtosis', 'skewness',
    'shape_factor', 'impulse_factor', 'clearance_factor',
    # Frequency domain (7)
    'spectral_centroid', 'spectral_spread', 'spectral_entropy',
    'band_energy_0_1kHz', 'band_energy_1_3kHz', 'band_energy_3_5kHz', 'band_energy_5_nyq',
    # Envelope (3)
    'envelope_rms', 'envelope_kurtosis', 'envelope_peak',
]

N_FEATURES = len(FEATURE_NAMES)  # 18


def extract_features_batch(X, sr: int = TARGET_SR, verbose: bool = False) -> np.ndarray:
    """
    Extract features for a batch of signals (fixed or variable length).
    Input: list of 1D arrays OR (N, L) ndarray
    Output: (N, N_FEATURES) float32
    """
    if isinstance(X, np.ndarray) and X.ndim == 2:
        signals = [X[i] for i in range(X.shape[0])]
    else:
        signals = X
    N = len(signals)
    feats = np.zeros((N, N_FEATURES), dtype=np.float32)
    for i, sig in enumerate(signals):
        if verbose and i % 100 == 0:
            print(f"  Features: {i}/{N}")
        feats[i] = extract_features(np.asarray(sig, dtype=np.float32), sr)
    return feats


if __name__ == '__main__':
    # Quick test
    np.random.seed(42)
    x = np.random.randn(WINDOW_LEN).astype(np.float32)

    print("=== Feature Extraction Test ===")
    print(f"Input shape: {x.shape}")

    td = time_domain_features(x)
    print(f"\nTime domain ({len(td)}): {dict(zip(FEATURE_NAMES[:8], td.round(4)))}")

    fd = compute_fft_features(x)
    print(f"\nFreq domain ({len(fd)}): {dict(zip(FEATURE_NAMES[8:15], fd.round(4)))}")

    env = compute_envelope_features(x)
    print(f"\nEnvelope ({len(env)}): {dict(zip(FEATURE_NAMES[15:], env.round(4)))}")

    full = extract_features(x)
    print(f"\nFull feature vector ({len(full)}): shape OK = {len(full) == N_FEATURES}")
    print("Feature names:", FEATURE_NAMES)
