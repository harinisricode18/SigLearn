"""
classifier.py
-------------
Automatic signal type classification.

Strategy: hand-crafted features in the frequency + time domain fed into
a Random Forest.  No deep learning needed here — the spectral fingerprint
of ECG / EEG / EMG / PPG / Respiration is very distinctive.

Features extracted
──────────────────
  Time domain   : mean, std, skewness, kurtosis, zero-crossing rate,
                  peak-to-peak amplitude, RMS
  Frequency     : dominant frequency, spectral centroid, spectral spread,
                  band-power ratios (delta/theta/alpha/beta/gamma),
                  spectral entropy
  Autocorrelation: lag-1 autocorrelation, autocorr peak prominence

The classifier is pre-trained with synthetic reference signals so it works
out of the box without any dataset.  For production quality, call
`train_classifier(X, y)` with real examples (see below).
"""

import numpy as np
from scipy import signal as sp_signal
from scipy.stats import skew, kurtosis
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pickle
import os

LABELS = ["ECG", "EEG", "EMG", "PPG", "Respiration"]
MODEL_PATH = os.path.join(os.path.dirname(__file__), "clf_model.pkl")


# ──────────────────────────────────────────────
# Feature Extraction
# ──────────────────────────────────────────────

def extract_features(sig: np.ndarray, fs: int) -> np.ndarray:
    """Extract a fixed-length feature vector from a 1-D signal."""
    sig = sig.astype(float)
    n   = len(sig)

    # ── Time domain ──────────────────────────────
    mean_v  = np.mean(sig)
    std_v   = np.std(sig)
    skew_v  = float(skew(sig))
    kurt_v  = float(kurtosis(sig))
    rms_v   = np.sqrt(np.mean(sig**2))
    p2p_v   = np.ptp(sig)                         # peak-to-peak
    zcr_v   = np.sum(np.diff(np.sign(sig)) != 0) / n

    # ── Frequency domain ─────────────────────────
    freqs, psd = sp_signal.welch(sig, fs=fs, nperseg=min(256, n // 2))
    total_power = np.sum(psd) + 1e-12

    def band_power(f_lo, f_hi):
        mask = (freqs >= f_lo) & (freqs < f_hi)
        return np.sum(psd[mask]) / total_power

    # Physiological bands
    delta = band_power(0.5,  4.0)   # EEG delta
    theta = band_power(4.0,  8.0)   # EEG theta
    alpha = band_power(8.0, 13.0)   # EEG alpha
    beta  = band_power(13.0,30.0)   # EEG beta
    low   = band_power(0.1,  2.0)   # Respiration / PPG
    mid   = band_power(0.5, 10.0)   # ECG / PPG
    high  = band_power(20.0,150.0)  # EMG

    dom_freq = float(freqs[np.argmax(psd)])
    centroid = float(np.sum(freqs * psd) / (total_power))

    # Spectral entropy
    psd_norm = psd / (total_power)
    spec_ent = float(-np.sum(psd_norm * np.log2(psd_norm + 1e-12)))

    # ── Autocorrelation ───────────────────────────
    ac = np.correlate(sig - mean_v, sig - mean_v, mode="full")
    ac = ac[n - 1:]
    ac /= (ac[0] + 1e-12)
    lag1_ac = float(ac[1]) if len(ac) > 1 else 0.0

    # Strongest autocorrelation peak (beyond lag 0)
    if len(ac) > 2:
        ac_peaks, _ = sp_signal.find_peaks(ac[1:], height=0.1)
        ac_peak_val = float(ac[ac_peaks[0] + 1]) if len(ac_peaks) > 0 else 0.0
    else:
        ac_peak_val = 0.0

    feat = np.array([
        mean_v, std_v, skew_v, kurt_v, rms_v, p2p_v, zcr_v,
        delta, theta, alpha, beta, low, mid, high,
        dom_freq, centroid, spec_ent,
        lag1_ac, ac_peak_val,
    ], dtype=float)

    # Replace NaN / Inf with 0
    feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
    return feat


# ──────────────────────────────────────────────
# Synthetic signal generators (for default model)
# ──────────────────────────────────────────────

def _gen_ecg(fs=360, seconds=10):
    t = np.arange(fs * seconds) / fs
    hr_hz = 1.2
    sig  = 0.1 * np.sin(2 * np.pi * hr_hz * t)
    for beat_t in np.arange(0, seconds, 1 / hr_hz):
        i = int(beat_t * fs)
        for offset, amp, width in [(-5, -0.1, 8), (0, 1.0, 3), (5, -0.2, 6), (12, 0.15, 10)]:
            idx = i + offset
            if 0 < idx < len(t):
                s = max(0, idx - width)
                e = min(len(sig), idx + width)
                seg_len = e - s
                if seg_len > 0:
                    sig[s:e] += amp * np.exp(-np.linspace(-2, 2, seg_len)**2)
    sig += np.random.normal(0, 0.02, len(sig))
    return sig / (np.ptp(sig) + 1e-12)


def _gen_eeg(fs=256, seconds=10):
    t = np.arange(fs * seconds) / fs
    sig = (0.4 * np.sin(2 * np.pi * 10 * t) +    # alpha 10 Hz
           0.3 * np.sin(2 * np.pi * 6  * t) +    # theta 6 Hz
           0.2 * np.sin(2 * np.pi * 2  * t) +    # delta 2 Hz
           0.1 * np.sin(2 * np.pi * 20 * t))      # beta 20 Hz
    sig += np.random.normal(0, 0.05, len(sig))
    return sig / (np.ptp(sig) + 1e-12)


def _gen_emg(fs=100, seconds=10):
    t = np.arange(fs * seconds) / fs
    sig = np.random.normal(0, 0.05, len(t))
    # Add muscle activation bursts
    for burst_t in [1, 3, 5, 7, 9]:
        i = int(burst_t * fs)
        burst_len = int(0.3 * fs)
        if i + burst_len < len(sig):
            sig[i:i+burst_len] += np.random.normal(0, 0.6, burst_len)
    return sig / (np.ptp(sig) + 1e-12)


def _gen_ppg(fs=125, seconds=10):
    t = np.arange(fs * seconds) / fs
    hr_hz = 1.1
    sig = (np.sin(2 * np.pi * hr_hz * t) ** 2 *
           np.maximum(np.sin(2 * np.pi * hr_hz * t), 0))
    sig += 0.05 * np.sin(2 * np.pi * 2 * hr_hz * t)
    sig += np.random.normal(0, 0.02, len(sig))
    return sig / (np.ptp(sig) + 1e-12)


def _gen_resp(fs=62, seconds=10):
    t = np.arange(fs * seconds) / fs
    sig = (np.sin(2 * np.pi * 0.25 * t) +
           0.15 * np.sin(2 * np.pi * 0.5 * t))
    sig += np.random.normal(0, 0.01, len(sig))
    return sig / (np.ptp(sig) + 1e-12)


# ──────────────────────────────────────────────
# Train / Load Model
# ──────────────────────────────────────────────

def _build_default_model():
    """Build and return a model trained on synthetic signals."""
    generators = [_gen_ecg, _gen_eeg, _gen_emg, _gen_ppg, _gen_resp]
    fs_list    = [360, 256, 100, 125, 62]
    X, y = [], []

    for label_idx, (gen, fs) in enumerate(zip(generators, fs_list)):
        for _ in range(60):                          # 60 samples per class
            sig = gen(fs=fs, seconds=10)
            X.append(extract_features(sig, fs))
            y.append(label_idx)

    X = np.array(X)
    y = np.array(y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_scaled, y)
    return clf, scaler


def load_model():
    """Load saved model or build default."""
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)
    clf, scaler = _build_default_model()
    # Save for next run
    with open(MODEL_PATH, "wb") as f:
        pickle.dump((clf, scaler), f)
    return clf, scaler


def train_classifier(X: np.ndarray, y: np.ndarray):
    """
    (Optional) Train the classifier on your own real data.

    Parameters
    ----------
    X : (N_samples, N_features)  — call extract_features() for each sample
    y : (N_samples,)             — integer labels matching LABELS list
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    clf = RandomForestClassifier(n_estimators=200, max_depth=12,
                                 random_state=42, n_jobs=-1)
    clf.fit(X_scaled, y)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump((clf, scaler), f)
    return clf, scaler


# ──────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────

_model_cache = None

def classify(sig: np.ndarray, fs: int) -> dict:
    """
    Classify a biosignal.

    Returns
    -------
    {
        "label"        : str   top predicted class
        "confidence"   : float 0–1
        "all_probs"    : dict  {label: prob} for all classes
    }
    """
    global _model_cache
    if _model_cache is None:
        _model_cache = load_model()

    clf, scaler = _model_cache
    feat = extract_features(sig, fs).reshape(1, -1)
    feat_scaled = scaler.transform(feat)

    probs = clf.predict_proba(feat_scaled)[0]
    top   = int(np.argmax(probs))

    return {
        "label"     : LABELS[top],
        "confidence": float(probs[top]),
        "all_probs" : {LABELS[i]: float(probs[i]) for i in range(len(LABELS))},
    }
