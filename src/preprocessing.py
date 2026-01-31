"""
preprocessing.py
Shared utilities for audio processing and Mel-spectrogram computation.
"""
import hashlib
import importlib.util
from pathlib import Path
from typing import Tuple, Optional, List
import librosa
import numpy as np
import soundfile as sf

# Prefer resampy-backed kaiser_fast when available; otherwise fall back to scipy polyphase.
_RESAMPLE_TYPE = "kaiser_fast" if importlib.util.find_spec("resampy") else "polyphase"

def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path

def _hash_path(p: str) -> str:
    """Generate a short hash for a file path to prevent filename collisions."""
    return hashlib.md5(p.encode("utf-8")).hexdigest()[:10]

def _next_pow2(n: int) -> int:
    return 1 << (n - 1).bit_length()

def calc_fft_hop(sr: int, win_ms: float, hop_ms: float) -> Tuple[int, int, int]:
    """Convert ms parameters to sample counts."""
    win_length = int(round(sr * (win_ms / 1000.0)))
    hop = int(round(sr * (hop_ms / 1000.0)))
    n_fft = _next_pow2(win_length)
    return n_fft, hop, win_length

def load_audio_stereo(path: Path, target_sr: int) -> np.ndarray:
    """
    Load audio, force stereo (2 channels), and resample if necessary.
    Returns: (2, Time) numpy array.
    """
    # Load with soundfile (much faster than librosa for raw reads)
    try:
        x, sr_in = sf.read(str(path), always_2d=True) # Returns (Time, Channels)
    except Exception as e:
        raise ValueError(f"Could not read {path}: {e}")

    # Transpose to (Channels, Time)
    x = x.T 

    # Resample if needed
    if sr_in != target_sr:
        # librosa.resample works on (C, T) or (T,)
        x = librosa.resample(x, orig_sr=sr_in, target_sr=target_sr, res_type=_RESAMPLE_TYPE)

    # Force Stereo
    C, T = x.shape
    if C == 1:
        # Duplicate mono to stereo
        stereo = np.vstack([x, x])
    elif C == 2:
        stereo = x
    else:
        # Take first two channels if > 2
        stereo = x[:2, :]

    return stereo.astype(np.float32)

def ensure_duration(stereo: np.ndarray, sr: int, duration_s: float) -> np.ndarray:
    """Pad or crop audio to exact duration."""
    C, T = stereo.shape
    target = int(round(sr * duration_s))
    
    if T >= target:
        return stereo[:, :target]
    else:
        padding = target - T
        return np.pad(stereo, ((0, 0), (0, padding)), mode='constant')

def mel_stereo2_from_stereo(stereo: np.ndarray, sr: int, n_fft: int, hop: int, win_length: int,
                            n_mels: int, fmin: float = 20.0, fmax: float | None = None) -> np.ndarray:
    """Compute Mel spectrogram for both channels."""
    fmax = fmax or (sr / 2)
    feats = []
    for ch in range(2):
        S = librosa.feature.melspectrogram(
            y=stereo[ch], sr=sr, n_fft=n_fft,
            hop_length=hop, win_length=win_length, window="hann",
            n_mels=n_mels, fmin=fmin, fmax=fmax, power=2.0, center=True
        )
        # Convert to dB
        S_db = librosa.power_to_db(S, ref=np.max).astype(np.float32)
        feats.append(S_db)
    
    return np.stack(feats, axis=0)  # Shape: (2, n_mels, Time)

def precache_one(wav_path: Path, label: str, cache_root: Path,
                 sr: int, dur: float, n_mels: int, win_ms: float, hop_ms: float,
                 fmin: float, fmax: Optional[float]) -> Path:
    """
    Main pipeline function: Load WAV -> Compute Mel -> Save .npy
    Returns the path to the saved .npy file.
    """
    n_fft, hop, win_length = calc_fft_hop(sr, win_ms, hop_ms)
    
    # 1. Load
    stereo = load_audio_stereo(wav_path, target_sr=sr)
    
    # 2. Fix Length
    stereo = ensure_duration(stereo, sr, dur)
    
    # 3. Compute Mel
    mel = mel_stereo2_from_stereo(
        stereo, sr,
        n_fft=n_fft, hop=hop, win_length=win_length,
        n_mels=n_mels, fmin=fmin, fmax=fmax
    )  # (2, n_mels, T)

    # 4. Save
    stem = wav_path.stem
    # Include params in filename to invalidate cache if params change
    tag  = f"sr{sr}_dur{dur}_m{n_mels}_w{int(win_ms)}_h{int(hop_ms)}"
    fn   = f"{stem}__{_hash_path(str(wav_path))}__{tag}.npy"
    
    out_path = cache_root / label / fn
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, mel.astype(np.float32))
    
    return out_path


