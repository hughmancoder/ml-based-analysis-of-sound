from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from src.models import CNN
from src.data_loader import normalise_spectrograms
from src.preprocessing import (
    calc_fft_hop,
    ensure_duration,
    load_audio_stereo,
    mel_stereo2_from_stereo,
)

def _select_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@dataclass(frozen=True)
class AudioConfig:
    sample_rate: int = 44100
    clip_duration: float = 3.0
    n_mels: int = 128
    win_ms: float = 30.0
    hop_ms: float = 10.0
    fmin: float = 20.0
    fmax: Optional[float] = None


class InstrumentClassifier:
    """
    Thin inference wrapper around the fine-tuned CNNVarTime classifier.

    Handles:
    Loading checkpoint weights and label mappings
    Converting raw audio clips to the (2, n_mels, T) mel tensors expected by the model
    Running a forward pass and returning class probabilities
    """

    def __init__(
        self,
        default_weights: Optional[Path] = None,
        audio_config: AudioConfig = AudioConfig(),
        device: Optional[torch.device] = None,
        cache_dir: Optional[Path] = Path(".cache/gui_mels"),
    ):
        self.audio_config = audio_config
        self.device = device or _select_device()
        self.default_weights = Path(default_weights) if default_weights else None
        self.cache_dir = Path(cache_dir) if cache_dir else None

        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Pre-compute FFT parameters
        self._n_fft, self._hop_length, self._win_length = calc_fft_hop(
            audio_config.sample_rate, audio_config.win_ms, audio_config.hop_ms
        )

        self._model: Optional[torch.nn.Module] = None
        self._loaded_weights: Optional[Path] = None
        self.label_to_idx: Optional[Dict[str, int]] = None
        self.idx_to_label: Optional[Dict[int, str]] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def predict(
        self,
        audio_path: Path | str,
        weights_path: Optional[Path | str] = None,
        save_mel: bool = True,
    ) -> Tuple[List[Tuple[str, float]], np.ndarray, Optional[Path], Path]:
        """
        Run a forward pass on a 3 second audio clip.

        Returns:
            predictions : List of (label, probability) sorted descending
            mel         : np.ndarray of shape (2, n_mels, T) before z-score normalisation
            mel_path    : Path where the mel .npy cache was stored (or None)
            weights     : Resolved checkpoint path used for inference
        """
        weights = self._resolve_weights(weights_path)
        model = self._ensure_model(weights)

        mel, mel_cache_path = self._audio_to_mel(Path(audio_path), save=save_mel)

        mel_norm = normalise_spectrograms(mel).astype(np.float32, copy=False)
        mel_tensor = torch.from_numpy(mel_norm).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            logits = model(mel_tensor)
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        predictions = self._format_predictions(probs)
        return predictions, mel, mel_cache_path, weights

    def predict_long_audio(
        self,
        audio_path: Path | str,
        weights_path: Optional[Path | str] = None,
        chunk_duration: Optional[float] = None,
        stride: Optional[float] = None,
    ) -> Tuple[List[Dict[str, float | str]], Counter, Path]:
        """
        Slice a long-form audio file into fixed-length windows and run the
        classifier on each chunk.

        Returns:
            chunks   : list of dicts containing chunk metadata + top-k predictions
            counts   : Counter of top-1 labels
            weights  : Resolved checkpoint path
        """
        weights = self._resolve_weights(weights_path)
        model = self._ensure_model(weights)

        cfg = self.audio_config
        chunk_len = float(chunk_duration) if chunk_duration and chunk_duration > 0 else cfg.clip_duration
        stride_len = float(stride) if stride and stride > 0 else chunk_len

        chunk_samples = int(round(chunk_len * cfg.sample_rate))
        stride_samples = int(round(stride_len * cfg.sample_rate))
        if chunk_samples <= 0 or stride_samples <= 0:
            raise ValueError("chunk_duration and stride must be positive.")

        stereo = load_audio_stereo(Path(audio_path), cfg.sample_rate)
        total_samples = stereo.shape[1]
        if total_samples == 0:
            raise ValueError("Input audio appears to be empty.")

        results: List[Dict[str, float | str]] = []
        counts: Counter = Counter()

        for idx, start in enumerate(range(0, total_samples, stride_samples)):
            end = start + chunk_samples
            segment = stereo[:, start:end]
            if segment.shape[1] < chunk_samples:
                segment = ensure_duration(segment, cfg.sample_rate, chunk_len)

            mel = mel_stereo2_from_stereo(
                segment,
                cfg.sample_rate,
                n_fft=self._n_fft,
                hop=self._hop_length,
                win_length=self._win_length,
                n_mels=cfg.n_mels,
                fmin=cfg.fmin,
                fmax=cfg.fmax,
            ).astype(np.float32, copy=False)

            mel_norm = normalise_spectrograms(mel).astype(np.float32, copy=False)
            mel_tensor = torch.from_numpy(mel_norm).unsqueeze(0).to(self.device)

            with torch.inference_mode():
                logits = model(mel_tensor)
                probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

            predictions = self._format_predictions(probs)
            top_label, top_prob = predictions[0]
            counts[top_label] += 1

            chunk_info: Dict[str, float | str] = {
                "chunk": idx,
                "start_s": start / cfg.sample_rate,
                "end_s": min(end, total_samples) / cfg.sample_rate,
                "top_label": top_label,
                "top_prob": float(top_prob),
            }
            for rank, (label, prob) in enumerate(predictions[:3], start=1):
                chunk_info[f"rank{rank}_label"] = label
                chunk_info[f"rank{rank}_prob"] = float(prob)

            results.append(chunk_info)

        return results, counts, weights


    def _resolve_weights(self, weights_path: Optional[Path | str]) -> Path:
        path = weights_path or self.default_weights
        if path is None:
            raise ValueError("No weights path provided and no default configured.")
        resolved = Path(path).expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Checkpoint not found: {resolved}")
        return resolved

    def _ensure_model(self, weights_path: Path) -> torch.nn.Module:
        if self._model is not None and self._loaded_weights == weights_path:
            return self._model

        ckpt = torch.load(weights_path, map_location=self.device)
        label_to_idx = ckpt.get("label_to_idx")
        if label_to_idx is None:
            raise KeyError("Checkpoint is missing 'label_to_idx'.")

        num_classes = len(label_to_idx)
        model = CNN(in_ch=2, num_classes=num_classes)
        model.load_state_dict(ckpt["model_state"], strict=True)
        model.to(self.device)
        model.eval()

        self._model = model
        self._loaded_weights = weights_path
        self.label_to_idx = {str(k): int(v) for k, v in label_to_idx.items()}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
        return model

    def _audio_to_mel(self, audio_path: Path, save: bool) -> Tuple[np.ndarray, Optional[Path]]:
        cfg = self.audio_config
        stereo = load_audio_stereo(audio_path, cfg.sample_rate)
        stereo = ensure_duration(stereo, cfg.sample_rate, cfg.clip_duration)
        mel = mel_stereo2_from_stereo(
            stereo,
            cfg.sample_rate,
            n_fft=self._n_fft,
            hop=self._hop_length,
            win_length=self._win_length,
            n_mels=cfg.n_mels,
            fmin=cfg.fmin,
            fmax=cfg.fmax,
        ).astype(np.float32, copy=False)

        mel_path = None
        if save and self.cache_dir is not None:
            mel_path = self._write_mel_cache(audio_path, mel)
        return mel, mel_path

    def _write_mel_cache(self, audio_path: Path, mel: np.ndarray) -> Path:
        digest = self._hash_audio(audio_path)
        tag = (
            f"sr{self.audio_config.sample_rate}"
            f"_dur{int(round(self.audio_config.clip_duration * 1000))}ms"
            f"_m{self.audio_config.n_mels}"
            f"_w{int(self.audio_config.win_ms)}"
            f"_h{int(self.audio_config.hop_ms)}"
        )
        filename = f"{audio_path.stem}_{digest[:10]}__{tag}.npy"
        cache_path = self.cache_dir / filename
        np.save(cache_path, mel.astype(np.float32))
        return cache_path

    def _hash_audio(self, audio_path: Path) -> str:
        hasher = hashlib.md5()
        with audio_path.open("rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def _format_predictions(self, probs: np.ndarray) -> List[Tuple[str, float]]:
        if self.idx_to_label is None:
            raise RuntimeError("Model labels are not loaded.")
        items = [(self.idx_to_label[idx], float(prob)) for idx, prob in enumerate(probs)]
        return sorted(items, key=lambda kv: kv[1], reverse=True)
