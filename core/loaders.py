"""
loaders.py
----------
Loads signals from all four datasets:
  - ECG  : MIT-BIH Arrhythmia Database  (wfdb)
  - EMG  : Ninapro DB1                  (numpy .mat via scipy)
  - PPG/Resp: BIDMC Dataset             (wfdb)
  - EEG  : MNE EEGBCI                   (mne)

Each loader returns a dict:
{
    "signal"  : np.ndarray  (1-D, already normalised to [-1, 1])
    "fs"      : int          sampling frequency (Hz)
    "label"   : str          signal type tag
    "meta"    : dict         anything extra (record name, subject id …)
}
"""

import os
from pathlib import Path
from huggingface_hub import snapshot_download

import numpy as np
import wfdb
import mne
from mne.datasets import eegbci
from scipy.io import loadmat


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _normalise(sig: np.ndarray) -> np.ndarray:
    """Scale signal to [-1, 1]."""
    rng = sig.max() - sig.min()
    if rng == 0:
        return sig.astype(float)
    return 2.0 * (sig - sig.min()) / rng - 1.0


def _trim(sig: np.ndarray, fs: int, seconds: int = 10) -> np.ndarray:
    """Return first `seconds` seconds of a signal."""
    return sig[: fs * seconds]

# --------------------------------------------------
# Dataset location
# --------------------------------------------------

if os.path.exists("./data"):
    # Running locally
    DATA_ROOT = "./data"

else:
    # Running on Hugging Face Space
    DATA_ROOT = snapshot_download(
        repo_id="Harinisri18/siglearn-datasets",
        repo_type="dataset",
        token=os.getenv("HF_TOKEN"),
    )


MIT_BIH_PATH = os.path.join(
    DATA_ROOT,
    "ECG",
    "mit-bih-arrhythmia-database-1.0.0",
)

NINAPRO_PATH = os.path.join(
    DATA_ROOT,
    "EMG",
)

BIDMC_PATH = os.path.join(
    DATA_ROOT,
    "BIDMC",
    "bidmc-ppg-and-respiration-dataset-1.0.0",
)

#--------------------------------------------------------------


# ──────────────────────────────────────────────
# ECG — MIT-BIH Arrhythmia Database
# ──────────────────────────────────────────────


# Records available in the MIT-BIH dataset
MIT_BIH_RECORDS = [
    "100", "101", "102", "103", "104", "105", "106", "107",
    "108", "109", "111", "112", "113", "114", "115", "116",
    "117", "118", "119", "121", "122", "123", "124", "200",
    "201", "202", "203", "205", "207", "208", "209", "210",
    "212", "213", "214", "215", "217", "219", "220", "221",
    "222", "223", "228", "230", "231", "232", "233", "234",
]


def load_ecg(record: str = "100", channel: int = 0,
             seconds: int = 10) -> dict:
    """
    Load an ECG record from the MIT-BIH Arrhythmia Database.

    Parameters
    ----------
    record  : record name, e.g. "100"
    channel : 0 = MLII, 1 = V1 (most records)
    seconds : how many seconds to return
    """
    record_path = os.path.join(MIT_BIH_PATH, record)
    rec = wfdb.rdrecord(record_path)

    sig = rec.p_signal[:, channel].astype(float)
    fs  = rec.fs

    sig = _trim(sig, fs, seconds)
    return {
        "signal": _normalise(sig),
        "fs"    : fs,
        "label" : "ECG",
        "meta"  : {
            "record" : record,
            "channel": rec.sig_name[channel],
            "units"  : rec.units[channel],
        },
    }


# ──────────────────────────────────────────────
# EMG — Ninapro DB1
# ──────────────────────────────────────────────


def load_emg(subject: int = 1, channel: int = 0,
             exercise: int = 1, seconds: int = 10) -> dict:
    """
    Load an EMG record from Ninapro DB1.

    Ninapro DB1 structure (per subject):
        s{n}/S{n}_A1_E1.mat  – Exercise 1  (basic finger movements)
        s{n}/S{n}_A1_E2.mat  – Exercise 2  (isometric/isotonic hand)
        s{n}/S{n}_A1_E3.mat  – Exercise 3  (grasping)

    .mat fields:
        emg   : (N_samples × 10)  raw EMG from 10 electrodes
        restimulus : stimulus labels
        rerepetition: repetition labels

    Parameters
    ----------
    subject  : 1–27
    channel  : 0–9  (electrode index)
    exercise : 1–3
    seconds  : seconds to return
    """
    # Ninapro DB1 structure: data/EMG/s{n}/S{n}_A1_E{e}.mat
    mat_path = os.path.join(
        NINAPRO_PATH,
        f"s{subject}",
        f"S{subject}_A1_E{exercise}.mat"
    )
    data = loadmat(mat_path)

    emg = data["emg"].astype(float)          # shape (N, 10)
    sig = emg[:, channel]
    fs  = 100                                 # Ninapro DB1 = 100 Hz

    sig = _trim(sig, fs, seconds)
    return {
        "signal": _normalise(sig),
        "fs"    : fs,
        "label" : "EMG",
        "meta"  : {
            "subject" : subject,
            "channel" : channel,
            "exercise": f"E{exercise}",
        },
    }


# ──────────────────────────────────────────────
# PPG + Respiration — BIDMC Dataset
# ──────────────────────────────────────────────

BIDMC_RECORDS = [f"{i:02d}" for i in range(1, 54)]   # 01–53


def load_ppg(record: str = "01", seconds: int = 10) -> dict:
    """
    Load PPG from BIDMC dataset.
    Channel 0 in BIDMC wfdb records = PPG (PLETH).
    """
    record_path = os.path.join(BIDMC_PATH, f"bidmc{record}")
    rec = wfdb.rdrecord(record_path)

    # Identify PPG channel
    names = [n.upper() for n in rec.sig_name]
    ch = names.index("PLETH") if "PLETH" in names else 0

    sig = rec.p_signal[:, ch].astype(float)
    fs  = rec.fs

    sig = _trim(sig, fs, seconds)
    return {
        "signal": _normalise(sig),
        "fs"    : fs,
        "label" : "PPG",
        "meta"  : {
            "record" : record,
            "channel": rec.sig_name[ch],
        },
    }


def load_respiration(record: str = "01", seconds: int = 10) -> dict:
    """
    Load Respiration signal from BIDMC dataset.
    Channel named 'RESP' in BIDMC wfdb records.
    """
    record_path = os.path.join(BIDMC_PATH, f"bidmc{record}")
    rec = wfdb.rdrecord(record_path)

    names = [n.upper() for n in rec.sig_name]
    ch = names.index("RESP") if "RESP" in names else 1

    sig = rec.p_signal[:, ch].astype(float)
    fs  = rec.fs

    sig = _trim(sig, fs, seconds)
    return {
        "signal": _normalise(sig),
        "fs"    : fs,
        "label" : "Respiration",
        "meta"  : {
            "record" : record,
            "channel": rec.sig_name[ch],
        },
    }


# ──────────────────────────────────────────────
# EEG — MNE EEGBCI (PhysioNet Motor Imagery)
# ──────────────────────────────────────────────

def load_eeg(subject: int = 1, run: int = 1,
             channel: str = "C3", seconds: int = 10) -> dict:
    """
    Load EEG from MNE's built-in EEGBCI dataset.
    Downloads ~2 MB per run on FIRST call only, then cached locally.
    Subsequent loads from the same subject/run are instant.

    Parameters
    ----------
    subject : 1–109
    run     : 1 = rest open, 2 = rest closed, 3–14 = motor tasks
    channel : electrode name, e.g. 'C3', 'Cz', 'C4', 'Fz'
    seconds : how many seconds to return
    """
    # Step 1: Download (or load from cache) — ~2 MB per run
    fnames = eegbci.load_data(subject, runs=[run], update_path=True, verbose=False)

    # Step 2: Read EDF WITHOUT preloading all data into RAM first
    raw = mne.io.read_raw_edf(fnames[0], preload=False, verbose=False)
    mne.datasets.eegbci.standardize(raw)   # normalise channel names to 10-20 system

    fs = int(raw.info["sfreq"])

    # Step 3: Pick channel BEFORE loading data (avoids loading all channels)
    if channel not in raw.ch_names:
        channel = raw.ch_names[0]
    raw.pick([channel])

    # Step 4: Crop to only the seconds we need BEFORE loading into RAM
    # This is the key speedup — we never load the full recording
    crop_end = min(seconds + 1.0, raw.times[-1])
    raw.crop(tmin=0.0, tmax=crop_end)

    # Step 5: NOW load into RAM (only the small cropped segment)
    raw.load_data(verbose=False)

    # Step 6: Filter only the short segment (fast)
    raw.filter(1.0, 40.0, fir_window="hamming", verbose=False)

    sig = raw.get_data()[0]   # single channel, shape (N_samples,)
    sig = _trim(sig, fs, seconds)

    return {
        "signal": _normalise(sig),
        "fs"    : fs,
        "label" : "EEG",
        "meta"  : {
            "subject": subject,
            "run"    : run,
            "channel": channel,
        },
    }


# ──────────────────────────────────────────────
# Load from uploaded file (CSV / txt / npy)
# ──────────────────────────────────────────────

def load_from_upload(path: str, fs: int = 250) -> dict:
    """
    Load a signal from a user-uploaded file.
    Supports: .npy, .csv, .txt

    CSV/txt: first numeric column is used as signal.
    """
    ext = os.path.splitext(path)[-1].lower()

    if ext == ".npy":
        sig = np.load(path).astype(float).flatten()
    elif ext in (".csv", ".txt"):
        import pandas as pd
        df  = pd.read_csv(path, header=None)
        sig = df.iloc[:, 0].values.astype(float)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    return {
        "signal": _normalise(sig),
        "fs"    : fs,
        "label" : "Unknown",
        "meta"  : {"source": os.path.basename(path)},
    }
