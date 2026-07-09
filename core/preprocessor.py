"""
preprocessor.py
---------------
Signal-type-specific filtering pipelines.

Each function returns:
{
    "signal"  : np.ndarray  filtered signal
    "steps"   : list[str]   human-readable processing log (for Feature 12)
}
"""

import numpy as np
from scipy import signal as sp


def _butter(sig, cutoff, fs, btype, order=4):
    nyq = fs / 2.0
    if isinstance(cutoff, (list, tuple)):
        wn = [c / nyq for c in cutoff]
    else:
        wn = cutoff / nyq
    b, a = sp.butter(order, wn, btype=btype)
    return sp.filtfilt(b, a, sig)


def preprocess(raw: np.ndarray, fs: int, signal_type: str) -> dict:
    """
    Apply the standard filtering pipeline for the given signal type.

    Parameters
    ----------
    raw         : raw 1-D signal (already loaded / normalised)
    fs          : sampling frequency
    signal_type : one of ECG / EEG / EMG / PPG / Respiration

    Returns
    -------
    dict with keys:
        signal  : np.ndarray  — filtered signal
        steps   : list[str]   — processing log
    """
    sig   = raw.astype(float).copy()
    steps = [f"Signal loaded — {len(sig)} samples at {fs} Hz"]

    t = signal_type.upper()

    # ── ECG ──────────────────────────────────────────────────────
    if t == "ECG":
        # 1. Bandpass 0.5–40 Hz  (removes baseline wander + HF noise)
        sig   = _butter(sig, [0.5, 40.0], fs, "bandpass")
        steps.append("Bandpass filter applied (0.5–40 Hz) — removes baseline wander and high-frequency noise")

        # 2. Notch at 50/60 Hz  (power-line interference)
        for notch_f in [50, 60]:
            if notch_f < fs / 2:
                b, a = sp.iirnotch(notch_f, Q=30, fs=fs)
                sig  = sp.filtfilt(b, a, sig)
        steps.append("Notch filter applied (50 Hz & 60 Hz) — removes power-line interference")

    # ── EEG ──────────────────────────────────────────────────────
    elif t == "EEG":
        # 1. Bandpass 1–40 Hz
        sig   = _butter(sig, [1.0, 40.0], fs, "bandpass")
        steps.append("Bandpass filter applied (1–40 Hz) — preserves delta through beta bands")

        # 2. Notch 50/60 Hz
        for notch_f in [50, 60]:
            if notch_f < fs / 2:
                b, a = sp.iirnotch(notch_f, Q=30, fs=fs)
                sig  = sp.filtfilt(b, a, sig)
        steps.append("Notch filter applied (50 Hz & 60 Hz) — removes power-line interference")

        # 3. Detrend
        sig   = sp.detrend(sig)
        steps.append("Linear detrend applied — removes slow drift from electrode movement")

    # ── EMG ──────────────────────────────────────────────────────
    elif t == "EMG":
        # 1. Highpass 20 Hz  (removes motion artifact + ECG contamination)
        sig   = _butter(sig, 20.0, fs, "highpass")
        steps.append("High-pass filter applied (20 Hz) — removes motion artifact and ECG contamination")

        # 2. Notch 50/60 Hz
        for notch_f in [50, 60]:
            if notch_f < fs / 2:
                b, a = sp.iirnotch(notch_f, Q=30, fs=fs)
                sig  = sp.filtfilt(b, a, sig)
        steps.append("Notch filter applied (50 Hz & 60 Hz) — removes power-line interference")

    # ── PPG ──────────────────────────────────────────────────────
    elif t == "PPG":
        # 1. Bandpass 0.5–8 Hz  (0.5 Hz = 30 BPM min, 8 Hz = well above 2nd harmonic)
        sig   = _butter(sig, [0.5, 8.0], fs, "bandpass")
        steps.append("Bandpass filter applied (0.5–8 Hz) — isolates the cardiac pulse waveform")

        # 2. Detrend
        sig   = sp.detrend(sig)
        steps.append("Linear detrend applied — removes slow baseline drift")

    # ── RESPIRATION ──────────────────────────────────────────────
    elif t in ("RESPIRATION", "RESP"):
        # 1. Bandpass 0.05–1.0 Hz  (3–60 breaths/min)
        sig   = _butter(sig, [0.05, 1.0], fs, "bandpass")
        steps.append("Bandpass filter applied (0.05–1.0 Hz) — isolates respiratory frequency band")

        # 2. Detrend
        sig   = sp.detrend(sig)
        steps.append("Linear detrend applied — removes movement-related drift")

    else:
        steps.append("No domain-specific filter applied (unknown signal type)")

    # Final normalisation
    rng = sig.max() - sig.min()
    if rng > 0:
        sig = 2.0 * (sig - sig.min()) / rng - 1.0
    steps.append("Signal normalised to [−1, 1] for display")

    return {"signal": sig, "steps": steps}
