import torch
import pandas as pd
import numpy as np
import re
import math
from pathlib import Path
from typing import Optional
from tqdm import tqdm
import librosa

from src.preprocessing import calc_fft_hop, ensure_duration, load_audio_stereo, mel_stereo2_from_stereo


def _read_csv_with_fallback(path: Path) -> pd.DataFrame:
    encodings = ("utf-8", "utf-8-sig", "gbk", "cp936")
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path)


def _read_text_with_fallback(path: Path) -> str:
    encodings = ("utf-8", "utf-8-sig", "gbk", "cp936")
    for enc in encodings:
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="ignore")


def parse_ground_truth(txt_path, label_to_idx):
    """
    Parses text files. Handles newlines, tabs, and commas.
    Only keeps labels present in the training set.
    """
    path = Path(txt_path)
    gt_vector = np.zeros(len(label_to_idx))
    if not path.exists():
        return gt_vector

    content = _read_text_with_fallback(path)

    # Regex split handles \n, \t, and commas simultaneously
    raw_labels = re.split(r'[\n,\t]', content)

    for label in raw_labels:
        clean = label.strip().lower()
        if clean in label_to_idx:
            gt_vector[label_to_idx[clean]] = 1.0

    return gt_vector


def _parse_labels_field(raw_value, label_to_idx):
    if raw_value is None:
        return np.zeros(len(label_to_idx))
    raw = str(raw_value).strip().lower()
    if not raw:
        return np.zeros(len(label_to_idx))

    # Accept common delimiters: ; | , whitespace, tabs, newlines
    raw_labels = re.split(r'[;|,\s]+', raw)
    gt_vector = np.zeros(len(label_to_idx))
    for label in raw_labels:
        clean = label.strip().lower()
        if clean in label_to_idx:
            gt_vector[label_to_idx[clean]] = 1.0
    return gt_vector


def _cqt_stereo2_from_stereo(
    stereo: np.ndarray,
    sr: int,
    n_bins: int,
    bins_per_octave: int,
    hop_length: int,
    fmin: float,
) -> np.ndarray:
    feats = []
    for ch in range(2):
        C = librosa.cqt(
            y=stereo[ch],
            sr=sr,
            hop_length=hop_length,
            fmin=fmin,
            n_bins=n_bins,
            bins_per_octave=bins_per_octave,
        )
        C_mag = np.abs(C)
        C_db = librosa.amplitude_to_db(C_mag, ref=np.max).astype(np.float32)
        feats.append(C_db)
    return np.stack(feats, axis=0)


def _cap_cqt_bins(sr: int, fmin: float, n_bins: int, bins_per_octave: int, fmax=None) -> int:
    if fmin <= 0:
        raise ValueError(f"fmin must be > 0 for CQT, got {fmin}")
    max_freq = float(fmax) if fmax else (sr / 2.0)
    if max_freq <= fmin:
        raise ValueError(f"Invalid CQT range: fmin={fmin} >= max_freq={max_freq}")
    max_bins = int(math.floor(bins_per_octave * math.log2(max_freq / fmin)))
    return min(n_bins, max_bins)


def load_and_preprocess_mel(path, cfg):
    """Load audio and convert to 2-channel Log-Mel Spectrogram."""
    p = Path(path)

    stereo = load_audio_stereo(p, target_sr=cfg['sr'])
    stereo = ensure_duration(stereo, cfg['sr'], cfg['duration'])
    n_fft, hop, win_length = calc_fft_hop(cfg['sr'], cfg['win_ms'], cfg['hop_ms'])

    mel = mel_stereo2_from_stereo(
        stereo,
        cfg['sr'],
        n_fft=n_fft,
        hop=hop,
        win_length=win_length,
        n_mels=cfg['n_mels'],
        fmin=cfg['fmin'],
        fmax=cfg.get('fmax'),
    )
    return mel


def load_and_preprocess_cqt(path, cfg, n_bins: Optional[int] = None, bins_per_octave: int = 12):
    """Load audio and convert to 2-channel CQT (log amplitude)."""
    p = Path(path)
    stereo = load_audio_stereo(p, target_sr=cfg['sr'])
    stereo = ensure_duration(stereo, cfg['sr'], cfg['duration'])

    hop_length = int(round(cfg['sr'] * (cfg['hop_ms'] / 1000.0)))
    n_bins = int(n_bins or cfg.get("n_mels", 128))
    n_bins = _cap_cqt_bins(cfg['sr'], cfg['fmin'], n_bins, bins_per_octave, cfg.get('fmax'))

    cqt = _cqt_stereo2_from_stereo(
        stereo,
        sr=cfg['sr'],
        n_bins=n_bins,
        bins_per_octave=bins_per_octave,
        hop_length=hop_length,
        fmin=cfg['fmin'],
    )
    return cqt


def load_and_preprocess_mel_cqt(path, cfg, n_bins: Optional[int] = None, bins_per_octave: int = 12):
    """Load audio and convert to 2ch mel + 2ch CQT."""
    p = Path(path)

    stereo = load_audio_stereo(p, target_sr=cfg['sr'])
    stereo = ensure_duration(stereo, cfg['sr'], cfg['duration'])

    n_fft, hop, win_length = calc_fft_hop(cfg['sr'], cfg['win_ms'], cfg['hop_ms'])
    mel = mel_stereo2_from_stereo(
        stereo,
        cfg['sr'],
        n_fft=n_fft,
        hop=hop,
        win_length=win_length,
        n_mels=cfg['n_mels'],
        fmin=cfg['fmin'],
        fmax=cfg.get('fmax'),
    )

    hop_length = int(round(cfg['sr'] * (cfg['hop_ms'] / 1000.0)))
    n_bins = int(n_bins or cfg.get("n_mels", 128))
    n_bins = _cap_cqt_bins(cfg['sr'], cfg['fmin'], n_bins, bins_per_octave, cfg.get('fmax'))
    cqt = _cqt_stereo2_from_stereo(
        stereo,
        sr=cfg['sr'],
        n_bins=n_bins,
        bins_per_octave=bins_per_octave,
        hop_length=hop_length,
        fmin=cfg['fmin'],
    )

    return mel, cqt


def _merge_mel_cqt(mel: np.ndarray, cqt: np.ndarray) -> torch.Tensor:
    mel_tensor = torch.from_numpy(mel).float()
    cqt_tensor = torch.from_numpy(cqt).float()

    # Align freq/time dimensions (crop to common size)
    min_h = min(mel_tensor.shape[1], cqt_tensor.shape[1])
    min_w = min(mel_tensor.shape[2], cqt_tensor.shape[2])
    mel_tensor = mel_tensor[:, :min_h, :min_w]
    cqt_tensor = cqt_tensor[:, :min_h, :min_w]

    # Per-example Z-score
    mel_tensor = (mel_tensor - mel_tensor.mean()) / (mel_tensor.std() + 1e-6)
    cqt_tensor = (cqt_tensor - cqt_tensor.mean()) / (cqt_tensor.std() + 1e-6)

    return torch.cat([mel_tensor, cqt_tensor], dim=0)


def get_prediction(model, mel_cqt, device):
    """Returns prediction vector from a preprocessed mel+cqt tensor."""
    model.eval()
    with torch.no_grad():
        x = mel_cqt.unsqueeze(0).to(device)
        logits = model(x)
        probs = torch.sigmoid(logits)
    return probs.cpu().numpy()[0]


def run_inference(
    *,
    model_cls,
    model_kwargs: dict,
    model_weights_path,
    device,
    test_manifest_csv,
    root: Path,
    state_key: str = "model_state",
    audio_cfg_key: str = "audio_config",
    classes_key: str = "classes",
    strict_load: bool = True,
    show_progress: bool = True,
    cqt_bins: Optional[int] = None,
    bins_per_octave: int = 12,
):
    """
    Loads checkpoint + model, runs inference over a manifest CSV.

    Returns:
        preds_arr: (N, C) float array of predicted probabilities
        gts_arr:   (N, C) int/bool array of ground-truth multi-hot vectors
        sample_ids: list[str] stems of wav filenames
        audio_cfg: dict-like audio config from checkpoint
        valid_labels: list[str] normalised class names
        label_to_idx: dict[str, int]
    """
    ckpt = torch.load(model_weights_path, map_location=device)

    audio_cfg = ckpt[audio_cfg_key]
    valid_labels = [c.strip().lower() for c in ckpt[classes_key]]
    label_to_idx = {name: i for i, name in enumerate(valid_labels)}

    model = model_cls(**model_kwargs, num_classes=len(valid_labels)).to(device)
    model.load_state_dict(ckpt[state_key], strict=strict_load)
    model.eval()

    df = _read_csv_with_fallback(Path(test_manifest_csv))

    def _resolve_path(p):
        p = Path(p)
        return p if p.is_absolute() else (root / p).resolve()

    def _maybe_resolve(p):
        if p is None or pd.isna(p):
            return ""
        s = str(p).strip()
        if not s:
            return ""
        return str(_resolve_path(s))

    if "wav_path" in df.columns:
        df["wav_path"] = df["wav_path"].apply(_maybe_resolve)
    if "txt_path" in df.columns:
        df["txt_path"] = df["txt_path"].apply(_maybe_resolve)
    if "filepath" in df.columns:
        df["filepath"] = df["filepath"].apply(_maybe_resolve)
    if "cqt_path" in df.columns:
        df["cqt_path"] = df["cqt_path"].apply(_maybe_resolve)

    all_preds, all_gt, sample_ids = [], [], []

    print(f"Running inference on {len(df)} samples against {len(valid_labels)} classes...")

    iterator = df.iterrows()
    if show_progress:
        iterator = tqdm(iterator, total=len(df))

    n_fail = 0
    with torch.no_grad():
        for _, row in iterator:
            try:
                # Ground truth
                if "txt_path" in row and str(row.get("txt_path", "")).strip():
                    gt_vec = parse_ground_truth(row["txt_path"], label_to_idx)
                elif "labels" in row or "label" in row:
                    labels_val = row.get("labels", None) if "labels" in row else row.get("label", None)
                    gt_vec = _parse_labels_field(labels_val, label_to_idx)
                else:
                    gt_vec = np.zeros(len(label_to_idx))

                mel = None
                cqt = None

                if "filepath" in row and str(row.get("filepath", "")).strip():
                    mel = np.load(row["filepath"])

                if "cqt_path" in row and str(row.get("cqt_path", "")).strip():
                    cqt = np.load(row["cqt_path"])

                need_mel = mel is None
                need_cqt = cqt is None

                wav_path = row.get("wav_path", None) if "wav_path" in row else None
                if (need_mel or need_cqt) and not wav_path:
                    raise ValueError("Missing wav_path for on-the-fly mel/cqt computation.")

                if need_mel and need_cqt:
                    mel, cqt = load_and_preprocess_mel_cqt(
                        wav_path, audio_cfg, n_bins=cqt_bins, bins_per_octave=bins_per_octave
                    )
                elif need_mel:
                    mel = load_and_preprocess_mel(wav_path, audio_cfg)
                elif need_cqt:
                    cqt = load_and_preprocess_cqt(
                        wav_path, audio_cfg, n_bins=cqt_bins, bins_per_octave=bins_per_octave
                    )

                mel_cqt = _merge_mel_cqt(mel, cqt)
                probs = get_prediction(model, mel_cqt, device)

            except Exception as exc:
                n_fail += 1
                if n_fail <= 10:
                    sample_ref = row.get("wav_path", row.get("filepath", "unknown"))
                    print(f"[WARN] Skipping unreadable sample: {sample_ref} ({exc})")
                continue

            all_preds.append(probs)
            all_gt.append(gt_vec)

            if "wav_path" in row:
                sample_ids.append(Path(row["wav_path"]).stem)
            elif "filepath" in row:
                sample_ids.append(Path(row["filepath"]).stem)
            elif "filename" in row:
                sample_ids.append(str(row["filename"]))
            else:
                sample_ids.append(f"sample_{len(sample_ids)}")

    preds_arr = np.asarray(all_preds)
    gts_arr = np.asarray(all_gt)

    if n_fail:
        print(f"[WARN] Skipped {n_fail} samples due to read/preprocess errors.")

    return preds_arr, gts_arr, sample_ids, audio_cfg, valid_labels, label_to_idx
