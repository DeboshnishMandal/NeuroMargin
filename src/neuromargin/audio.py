from __future__ import annotations
import io
import numpy as np
import librosa
import soundfile as sf

SR = 16000
N_MFCC = 20

def load_audio(source, max_seconds=1800):
    """Load WAV/compatible audio, convert to mono 16 kHz; allow long research recordings.

    The 30-minute ceiling is a safety guard, not a 120-second task limit.
    Training and inference call this same loader so feature definitions remain aligned.
    Pass max_seconds=None to disable the ceiling explicitly.
    """
    y, sr = sf.read(source, dtype="float32", always_2d=False)
    if y.ndim == 2:
        y = np.mean(y, axis=1)
    if y.ndim != 1 or len(y) == 0:
        raise ValueError("Audio is empty or has an unsupported channel layout.")
    if not np.isfinite(y).all():
        raise ValueError("Audio contains NaN or infinite samples.")
    if sr <= 0:
        raise ValueError("Invalid sample rate.")
    duration_before_resample = len(y) / sr
    if max_seconds is not None and duration_before_resample > max_seconds:
        raise ValueError(f"Audio is too long. Maximum supported duration is {max_seconds // 60} minutes.")
    if sr != SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)
    y = np.asarray(y, dtype=np.float32)
    peak = float(np.max(np.abs(y)))
    if peak < 1e-5:
        raise ValueError("Recording is silent or too quiet.")
    y = y / max(peak, 1e-8) * 0.95
    duration = len(y) / SR
    if duration < 2.0:
        raise ValueError("Please upload at least 2 seconds of speech.")
    return y, {"original_sample_rate": int(sr), "model_sample_rate": SR,
               "duration_seconds": round(duration, 3), "peak_amplitude": round(peak, 5)}

def extract_features(y):
    """Fixed-size acoustic representation used identically for training and inference."""
    y = np.asarray(y, dtype=np.float32)
    if len(y) < SR:
        raise ValueError("Audio must contain at least one second of usable signal.")
    hop, n_fft = 256, 1024
    feats = {}
    def add_stats(prefix, arr):
        arr = np.asarray(arr, dtype=float)
        if arr.ndim == 1: arr = arr[None, :]
        for i, row in enumerate(arr):
            feats[f"{prefix}_{i+1:02d}_mean"] = float(np.mean(row))
            feats[f"{prefix}_{i+1:02d}_std"] = float(np.std(row))
    mfcc = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=N_MFCC, n_fft=n_fft, hop_length=hop)
    delta = librosa.feature.delta(mfcc, order=1)
    delta2 = librosa.feature.delta(mfcc, order=2)
    add_stats("mfcc", mfcc)
    add_stats("delta", delta)
    add_stats("delta2", delta2)
    for name, fn in [
        ("rms", lambda: librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop)[0]),
        ("zcr", lambda: librosa.feature.zero_crossing_rate(y, frame_length=n_fft, hop_length=hop)[0]),
        ("centroid", lambda: librosa.feature.spectral_centroid(y=y, sr=SR, n_fft=n_fft, hop_length=hop)[0]),
        ("bandwidth", lambda: librosa.feature.spectral_bandwidth(y=y, sr=SR, n_fft=n_fft, hop_length=hop)[0]),
        ("rolloff", lambda: librosa.feature.spectral_rolloff(y=y, sr=SR, n_fft=n_fft, hop_length=hop)[0]),
        ("flatness", lambda: librosa.feature.spectral_flatness(y=y, n_fft=n_fft, hop_length=hop)[0]),
    ]:
        a = np.asarray(fn())
        feats[name+"_mean"] = float(np.mean(a))
        feats[name+"_std"] = float(np.std(a))
        feats[name+"_median"] = float(np.median(a))
    f0 = librosa.yin(y, fmin=65, fmax=500, sr=SR, frame_length=2048, hop_length=hop)
    f0 = f0[np.isfinite(f0)]
    feats["f0_mean"] = float(np.mean(f0)) if len(f0) else 0.0
    feats["f0_std"] = float(np.std(f0)) if len(f0) else 0.0
    feats["f0_median"] = float(np.median(f0)) if len(f0) else 0.0
    feats["duration_seconds"] = float(len(y)/SR)
    feats["silence_fraction"] = float(np.mean(np.abs(y) < 0.008))
    vals = np.asarray(list(feats.values()), dtype=np.float64)
    if not np.isfinite(vals).all():
        raise ValueError("Feature extraction produced non-finite values.")
    return feats

def inspect_audio(source):
    y, meta = load_audio(source)
    return y, meta, extract_features(y)
