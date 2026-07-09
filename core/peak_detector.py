"""
peak_detector.py
----------------
Detects physiologically meaningful peaks / events in each signal type.

Returns a list of annotation dicts:
[
    {
        "name"   : str   annotation label  (e.g. "R-peak", "P-wave")
        "index"  : int   sample index in the signal array
        "time"   : float time in seconds
        "color"  : str   hex color for the plot marker
        "symbol" : str   plotly marker symbol
    },
    ...
]
"""

import numpy as np
from scipy import signal as sp


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _find_peaks_safe(sig, **kwargs):
    peaks, props = sp.find_peaks(sig, **kwargs)
    return peaks, props


def _median_rr(r_peaks, fs):
    if len(r_peaks) < 2:
        return None
    rr = np.diff(r_peaks) / fs          # in seconds
    return float(np.median(rr))


# ──────────────────────────────────────────────
# ECG Peak Detection
# P-wave · QRS complex · T-wave
# ──────────────────────────────────────────────

def detect_ecg_peaks(sig: np.ndarray, fs: int) -> list:
    annotations = []
    n = len(sig)

    # ── R-peaks (QRS) via Pan-Tompkins-lite ───────────────────
    # 1. Derivative
    diff_sig = np.diff(sig, prepend=sig[0])
    # 2. Square
    sq = diff_sig ** 2
    # 3. Moving average (≈ 150 ms window)
    win = max(1, int(0.15 * fs))
    ma = np.convolve(sq, np.ones(win) / win, mode="same")

    min_dist = int(0.4 * fs)            # min 400 ms between beats (150 BPM max)
    r_peaks, _ = _find_peaks_safe(
        ma,
        distance=min_dist,
        height=np.mean(ma) * 0.5,
    )

    for idx in r_peaks:
        annotations.append({
            "name"  : "R-peak (QRS)",
            "index" : int(idx),
            "time"  : idx / fs,
            "color" : "#FF4B4B",
            "symbol": "triangle-up",
        })

    # ── P-waves (≈ 150–200 ms before R) ───────────────────────
    p_window = int(0.18 * fs)
    p_offset = int(0.20 * fs)
    for r in r_peaks:
        start = max(0, r - p_offset - p_window)
        end   = max(0, r - p_offset)
        if end > start + 2:
            seg   = sig[start:end]
            local = np.argmax(seg)
            idx   = start + local
            annotations.append({
                "name"  : "P-wave",
                "index" : int(idx),
                "time"  : idx / fs,
                "color" : "#4B8BFF",
                "symbol": "circle",
            })

    # ── T-waves (≈ 200–400 ms after R) ────────────────────────
    t_start_off = int(0.15 * fs)
    t_end_off   = int(0.40 * fs)
    for r in r_peaks:
        start = min(n - 1, r + t_start_off)
        end   = min(n,     r + t_end_off)
        if end > start + 2:
            seg   = sig[start:end]
            local = np.argmax(np.abs(seg))
            idx   = start + local
            annotations.append({
                "name"  : "T-wave",
                "index" : int(idx),
                "time"  : idx / fs,
                "color" : "#4BFF91",
                "symbol": "diamond",
            })

    return annotations


# ──────────────────────────────────────────────
# PPG Peak Detection
# Systolic peak · Diastolic peak · Dicrotic notch
# ──────────────────────────────────────────────

def detect_ppg_peaks(sig: np.ndarray, fs: int) -> list:
    annotations = []

    min_dist = int(0.4 * fs)
    systolic, _ = _find_peaks_safe(
        sig,
        distance=min_dist,
        height=np.mean(sig),
    )

    for idx in systolic:
        annotations.append({
            "name"  : "Systolic Peak",
            "index" : int(idx),
            "time"  : idx / fs,
            "color" : "#FF4B4B",
            "symbol": "triangle-up",
        })

    # Dicrotic notch + diastolic peak (between systolic peaks)
    for i in range(len(systolic) - 1):
        seg_start = systolic[i]
        seg_end   = systolic[i + 1]
        seg       = sig[seg_start:seg_end]
        if len(seg) < 6:
            continue

        # Notch = minimum in the descending portion (first 60% of seg)
        desc_end  = int(0.6 * len(seg))
        notch_rel = np.argmin(seg[:desc_end])
        notch_idx = seg_start + notch_rel
        annotations.append({
            "name"  : "Dicrotic Notch",
            "index" : int(notch_idx),
            "time"  : notch_idx / fs,
            "color" : "#FFB84B",
            "symbol": "x",
        })

        # Diastolic = local max after notch
        after_notch = seg[notch_rel:]
        if len(after_notch) > 3:
            diast_rel = np.argmax(after_notch)
            diast_idx = notch_idx + diast_rel
            annotations.append({
                "name"  : "Diastolic Peak",
                "index" : int(diast_idx),
                "time"  : diast_idx / fs,
                "color" : "#C04BFF",
                "symbol": "triangle-down",
            })

    return annotations


# ──────────────────────────────────────────────
# EMG — Muscle Activation Bursts
# ──────────────────────────────────────────────

def detect_emg_bursts(sig: np.ndarray, fs: int) -> list:
    annotations = []

    # Envelope via full-wave rectification + moving average
    rectified = np.abs(sig)
    win       = max(1, int(0.05 * fs))           # 50 ms smoothing
    envelope  = np.convolve(rectified, np.ones(win) / win, mode="same")

    threshold = np.mean(envelope) + 1.5 * np.std(envelope)
    min_dist  = int(0.2 * fs)                    # min 200 ms between bursts

    burst_peaks, _ = _find_peaks_safe(
        envelope,
        height=threshold,
        distance=min_dist,
    )

    for idx in burst_peaks:
        annotations.append({
            "name"  : "Muscle Burst",
            "index" : int(idx),
            "time"  : idx / fs,
            "color" : "#FF6B35",
            "symbol": "triangle-up",
        })

    return annotations


# ──────────────────────────────────────────────
# EEG — Alpha / Theta / Beta band events
# (simple power burst detection per band)
# ──────────────────────────────────────────────

def detect_eeg_events(sig: np.ndarray, fs: int) -> list:
    annotations = []

    bands = {
        "Alpha (8–13 Hz)" : (8,  13,  "#4BFF91"),
        "Beta (13–30 Hz)" : (13, 30,  "#4B8BFF"),
        "Theta (4–8 Hz)"  : (4,  8,   "#FFB84B"),
    }

    for band_name, (lo, hi, color) in bands.items():
        nyq = fs / 2.0
        if hi >= nyq:
            continue
        b, a     = sp.butter(4, [lo / nyq, hi / nyq], btype="bandpass")
        filtered = sp.filtfilt(b, a, sig)
        power    = filtered ** 2

        win     = max(1, int(0.5 * fs))           # 500 ms window
        smooth  = np.convolve(power, np.ones(win) / win, mode="same")

        thresh  = np.mean(smooth) + 1.0 * np.std(smooth)
        peaks, _ = _find_peaks_safe(smooth, height=thresh,
                                    distance=int(0.5 * fs))

        for idx in peaks[:5]:                     # limit to 5 per band
            annotations.append({
                "name"  : f"{band_name} burst",
                "index" : int(idx),
                "time"  : idx / fs,
                "color" : color,
                "symbol": "circle",
            })

    return annotations


# ──────────────────────────────────────────────
# Respiration — Breath Peaks
# ──────────────────────────────────────────────

def detect_resp_peaks(sig: np.ndarray, fs: int) -> list:
    annotations = []

    min_dist  = int(1.5 * fs)                    # min 1.5 s between breaths (40 brpm max)
    peaks, _  = _find_peaks_safe(sig, distance=min_dist,
                                  height=np.mean(sig))
    troughs,_ = _find_peaks_safe(-sig, distance=min_dist)

    for idx in peaks:
        annotations.append({
            "name"  : "Inhalation Peak",
            "index" : int(idx),
            "time"  : idx / fs,
            "color" : "#4B8BFF",
            "symbol": "triangle-up",
        })

    for idx in troughs:
        annotations.append({
            "name"  : "Exhalation Trough",
            "index" : int(idx),
            "time"  : idx / fs,
            "color" : "#FF4B4B",
            "symbol": "triangle-down",
        })

    return annotations


# ──────────────────────────────────────────────
# Dispatcher
# ──────────────────────────────────────────────

def detect_peaks(sig: np.ndarray, fs: int, signal_type: str) -> list:
    t = signal_type.upper()
    if t == "ECG":
        return detect_ecg_peaks(sig, fs)
    elif t == "PPG":
        return detect_ppg_peaks(sig, fs)
    elif t == "EMG":
        return detect_emg_bursts(sig, fs)
    elif t == "EEG":
        return detect_eeg_events(sig, fs)
    elif t in ("RESPIRATION", "RESP"):
        return detect_resp_peaks(sig, fs)
    return []
