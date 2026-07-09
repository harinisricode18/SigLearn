"""
feature_extractor.py
--------------------
Computes derived biomedical parameters for each signal type.

ECG   : Heart rate, RR interval, QT interval, PR interval, HRV
PPG   : Heart rate, pulse variability, SpO2 concepts
EEG   : Band powers (delta/theta/alpha/beta/gamma), dominant frequency
EMG   : RMS, mean absolute value, fatigue index, burst count
Resp  : Respiratory rate, inspiration/expiration ratio, breath depth

Returns a list of parameter dicts:
[
    {
        "name"  : str   parameter name
        "value" : float numeric value
        "unit"  : str   unit string
        "normal": str   normal range text (for education)
        "status": str   "normal" | "high" | "low" | "info"
    },
    ...
]
"""

import numpy as np
from scipy import signal as sp


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _r_peaks(sig, fs):
    """Quick R-peak detector (same logic as peak_detector.py)."""
    diff_sig = np.diff(sig, prepend=sig[0])
    sq       = diff_sig ** 2
    win      = max(1, int(0.15 * fs))
    ma       = np.convolve(sq, np.ones(win) / win, mode="same")
    peaks, _ = sp.find_peaks(ma, distance=int(0.4 * fs),
                              height=np.mean(ma) * 0.5)
    return peaks


def _band_power(sig, fs, lo, hi):
    nyq = fs / 2.0
    if hi >= nyq:
        hi = nyq * 0.99
    b, a     = sp.butter(4, [lo / nyq, hi / nyq], btype="bandpass")
    filtered = sp.filtfilt(b, a, sig)
    return float(np.mean(filtered ** 2))


def _param(name, value, unit, normal, status="info"):
    return {
        "name"  : name,
        "value" : round(float(value), 2) if not np.isnan(value) else 0.0,
        "unit"  : unit,
        "normal": normal,
        "status": status,
    }


def _hr_status(hr):
    if hr < 60:  return "low"
    if hr > 100: return "high"
    return "normal"


# ──────────────────────────────────────────────
# ECG
# ──────────────────────────────────────────────

def extract_ecg_features(sig, fs):
    params = []
    r_idx  = _r_peaks(sig, fs)

    if len(r_idx) < 2:
        return [_param("R-peaks detected", len(r_idx), "peaks",
                        "≥ 2 needed for HR calculation", "info")]

    rr_sec   = np.diff(r_idx) / fs                    # RR intervals in seconds
    rr_ms    = rr_sec * 1000                           # in ms

    hr       = 60.0 / np.mean(rr_sec)
    rr_mean  = float(np.mean(rr_ms))
    rr_std   = float(np.std(rr_ms))                   # SDNN — HRV index

    params.append(_param("Heart Rate",    hr,      "BPM", "60–100 BPM",  _hr_status(hr)))
    params.append(_param("Mean RR Interval", rr_mean, "ms", "600–1000 ms",
                          "normal" if 600 <= rr_mean <= 1000 else "high"))
    params.append(_param("SDNN (HRV)",    rr_std,  "ms",  "20–50 ms (rest)",
                          "normal" if 20 <= rr_std <= 100 else "info"))

    # QT interval estimate: R to end of T ≈ 0.38 * RR (Bazett's formula range)
    qt_est  = 0.38 * np.mean(rr_sec) * 1000
    qtc     = qt_est / np.sqrt(np.mean(rr_sec))      # Bazett corrected QTc
    params.append(_param("QT Interval (est.)",  qt_est, "ms",  "350–440 ms",
                          "normal" if 350 <= qt_est <= 440 else "high"))
    params.append(_param("QTc (Bazett)",        qtc,    "ms",  "< 450 ms",
                          "normal" if qtc < 450 else "high"))

    # PR interval estimate: P to R ≈ 0.16 * RR
    pr_est  = 0.16 * np.mean(rr_sec) * 1000
    params.append(_param("PR Interval (est.)",  pr_est, "ms",  "120–200 ms",
                          "normal" if 120 <= pr_est <= 200 else "info"))

    return params


# ──────────────────────────────────────────────
# PPG
# ──────────────────────────────────────────────

def extract_ppg_features(sig, fs):
    params = []

    # Systolic peaks
    min_dist  = int(0.4 * fs)
    peaks, _  = sp.find_peaks(sig, distance=min_dist,
                               height=np.mean(sig))

    if len(peaks) < 2:
        return [_param("Peaks detected", len(peaks), "peaks",
                        "≥ 2 needed", "info")]

    ipi_sec  = np.diff(peaks) / fs                    # inter-pulse intervals
    hr_ppg   = 60.0 / np.mean(ipi_sec)
    ipi_ms   = ipi_sec * 1000
    ipi_std  = float(np.std(ipi_ms))

    params.append(_param("Heart Rate (PPG)", hr_ppg,  "BPM",  "60–100 BPM",
                          _hr_status(hr_ppg)))
    params.append(_param("Mean IPI",         float(np.mean(ipi_ms)), "ms",
                          "600–1000 ms",
                          "normal" if 600 <= np.mean(ipi_ms) <= 1000 else "info"))
    params.append(_param("Pulse Variability (std IPI)", ipi_std, "ms",
                          "< 50 ms normal", "info"))

    # AC/DC ratio — rough SpO2 concept
    ac_val  = float(np.std(sig))
    dc_val  = float(np.mean(sig) + 1.0)              # shift to positive
    ac_dc   = ac_val / (dc_val + 1e-9)
    params.append(_param("AC/DC Ratio (SpO₂ proxy)", ac_dc, "",
                          "Higher ratio → better perfusion", "info"))

    # Perfusion index
    pi      = ac_dc * 100
    params.append(_param("Perfusion Index", pi, "%",
                          "Normal > 0.2%", "normal" if pi > 0.2 else "low"))

    return params


# ──────────────────────────────────────────────
# EEG
# ──────────────────────────────────────────────

def extract_eeg_features(sig, fs):
    params = []

    total_power = np.mean(sig ** 2) + 1e-12

    bands = {
        "Delta (0.5–4 Hz)" : (0.5, 4.0,  "Dominant in deep sleep"),
        "Theta (4–8 Hz)"   : (4.0, 8.0,  "Drowsiness / memory encoding"),
        "Alpha (8–13 Hz)"  : (8.0, 13.0, "Relaxed wakefulness"),
        "Beta (13–30 Hz)"  : (13.0,30.0, "Active thinking / alertness"),
        "Gamma (30–45 Hz)" : (30.0,45.0, "High-level cognition"),
    }

    for band_name, (lo, hi, note) in bands.items():
        bp       = _band_power(sig, fs, lo, hi)
        rel_pwr  = (bp / total_power) * 100
        params.append(_param(f"{band_name} Power", rel_pwr, "%", note, "info"))

    # Dominant frequency
    freqs, psd = sp.welch(sig, fs=fs, nperseg=min(256, len(sig) // 2))
    dom_f      = float(freqs[np.argmax(psd)])
    params.append(_param("Dominant Frequency", dom_f, "Hz",
                          "Alpha peak: 8–13 Hz at rest", "info"))

    # Alpha/theta ratio (attentiveness index)
    alpha_p = _band_power(sig, fs, 8.0, 13.0)
    theta_p = _band_power(sig, fs, 4.0, 8.0)
    at_ratio = alpha_p / (theta_p + 1e-12)
    params.append(_param("Alpha/Theta Ratio", at_ratio, "",
                          "> 1 → more alert/relaxed", "info"))

    return params


# ──────────────────────────────────────────────
# EMG
# ──────────────────────────────────────────────

def extract_emg_features(sig, fs):
    params = []

    # RMS amplitude
    rms = float(np.sqrt(np.mean(sig ** 2)))
    params.append(_param("RMS Amplitude",  rms,  "a.u.", "Depends on contraction level", "info"))

    # Mean Absolute Value
    mav = float(np.mean(np.abs(sig)))
    params.append(_param("Mean Absolute Value (MAV)", mav, "a.u.",
                          "Proportional to muscle force", "info"))

    # Median frequency (fatigue index — shifts down with fatigue)
    freqs, psd = sp.welch(sig, fs=fs, nperseg=min(256, len(sig) // 2))
    cumulative  = np.cumsum(psd)
    med_f_idx   = np.searchsorted(cumulative, cumulative[-1] / 2)
    med_f       = float(freqs[min(med_f_idx, len(freqs) - 1)])
    params.append(_param("Median Frequency", med_f, "Hz",
                          "60–100 Hz fresh; drops with fatigue", "info"))

    # Zero-crossing rate
    zcr = float(np.sum(np.diff(np.sign(sig)) != 0)) / len(sig)
    params.append(_param("Zero-Crossing Rate", zcr * 100, "%/sample",
                          "Increases with co-contraction", "info"))

    # Burst count
    rectified = np.abs(sig)
    win       = max(1, int(0.05 * fs))
    envelope  = np.convolve(rectified, np.ones(win) / win, mode="same")
    thresh    = np.mean(envelope) + 1.5 * np.std(envelope)
    bursts, _ = sp.find_peaks(envelope, height=thresh,
                               distance=int(0.2 * fs))
    params.append(_param("Muscle Burst Count", len(bursts), "bursts",
                          "Reflects activation events", "info"))

    return params


# ──────────────────────────────────────────────
# Respiration
# ──────────────────────────────────────────────

def extract_resp_features(sig, fs):
    params = []

    min_dist = int(1.5 * fs)
    peaks, _ = sp.find_peaks(sig, distance=min_dist, height=np.mean(sig))

    if len(peaks) < 2:
        return [_param("Breath peaks detected", len(peaks), "",
                        "≥ 2 needed", "info")]

    ibi_sec = np.diff(peaks) / fs
    rr_rate = 60.0 / np.mean(ibi_sec)
    params.append(_param("Respiratory Rate", rr_rate, "breaths/min",
                          "12–20 breaths/min",
                          "normal" if 12 <= rr_rate <= 20 else
                          "low" if rr_rate < 12 else "high"))

    breath_depth = float(np.mean(sig[peaks]) - np.min(sig))
    params.append(_param("Breath Depth (amplitude)", breath_depth, "a.u.",
                          "Relative — higher = deeper breath", "info"))

    # Inspiration/expiration ratio estimate
    troughs, _ = sp.find_peaks(-sig, distance=min_dist)
    if len(troughs) > 0 and len(peaks) > 1:
        try:
            # IE ratio: time from trough to next peak / peak to next trough
            ie_ratios = []
            for p_idx in range(len(peaks) - 1):
                t_after = troughs[(troughs > peaks[p_idx]) &
                                   (troughs < peaks[p_idx + 1])]
                if len(t_after) > 0:
                    t_idx = t_after[0]
                    insp  = (t_idx - peaks[p_idx]) / fs
                    exp   = (peaks[p_idx + 1] - t_idx) / fs
                    if exp > 0:
                        ie_ratios.append(insp / exp)
            if ie_ratios:
                ie = float(np.mean(ie_ratios))
                params.append(_param("I:E Ratio", ie, "",
                                      "Normal ≈ 1:2", "info"))
        except Exception:
            pass

    return params


# ──────────────────────────────────────────────
# Dispatcher
# ──────────────────────────────────────────────

def extract_features(sig: np.ndarray, fs: int, signal_type: str) -> list:
    t = signal_type.upper()
    if t == "ECG":
        return extract_ecg_features(sig, fs)
    elif t == "PPG":
        return extract_ppg_features(sig, fs)
    elif t == "EEG":
        return extract_eeg_features(sig, fs)
    elif t == "EMG":
        return extract_emg_features(sig, fs)
    elif t in ("RESPIRATION", "RESP"):
        return extract_resp_features(sig, fs)
    return []
