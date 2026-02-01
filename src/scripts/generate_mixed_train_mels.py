#!/usr/bin/env python3
import argparse
import csv
import random
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import yaml
from tqdm import tqdm
import soundfile as sf  # <-- needed for saving wavs

# Reuse your existing preprocessing utilities
from preprocessing import (
    calc_fft_hop,
    ensure_duration,
    load_audio_stereo,
    mel_stereo2_from_stereo,
    ensure_dir,
)


def load_yaml(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def get_train_labels(cfg: dict) -> Optional[Set[str]]:
    labels = cfg.get("train_labels")
    if labels is None:
        labels = (cfg.get("dataset") or {}).get("train_labels")

    if not labels:
        return None

    out: Set[str] = set()
    for x in labels:
        if x is None:
            continue
        s = str(x).strip().lower()
        if s:
            out.add(s)
    return out or None


def iter_wavs_by_label(root: Path) -> Dict[str, List[Path]]:
    """
    Assumes folder structure: root/label/*.wav
    Returns mapping label -> wav paths.
    """
    by_label: Dict[str, List[Path]] = {}
    for wav in root.rglob("*.wav"):
        if wav.is_file():
            label = wav.parent.name.strip().lower()
            by_label.setdefault(label, []).append(wav)
    return by_label


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x * x) + 1e-12))


def mix_stereo_waveforms(
    stereos: List[np.ndarray],
    snr_db_range: Tuple[float, float],
) -> np.ndarray:
    """
    Mixes a list of stereo waveforms (each shape: (2, T)).

    We treat the first waveform as 'base', then scale additional sources to
    a random SNR relative to the base RMS, and add them.

    Returns:
        mixed stereo waveform (2, T), peak-normalised to avoid clipping.
    """
    assert len(stereos) >= 2
    base = stereos[0].astype(np.float32, copy=True)

    # Use combined-channel RMS for scaling (simple + robust)
    base_rms = rms(base)

    mix = base
    for s in stereos[1:]:
        s = s.astype(np.float32, copy=False)
        s_rms = rms(s)

        snr_db = random.uniform(*snr_db_range)
        # scale so that 20log10(base_rms / (gain*s_rms)) = snr_db
        gain = (base_rms / (s_rms + 1e-12)) * (10 ** (-snr_db / 20.0))
        mix = mix + s * float(gain)

    # Peak normalise with a tiny headroom
    peak = float(np.max(np.abs(mix)) + 1e-12)
    mix = (0.99 * mix / peak).astype(np.float32)
    return mix


def save_mixed_npy(
    cache_root: Path,
    labels: List[str],
    idx: int,
    sr: int,
    dur: float,
    n_mels: int,
    win_ms: float,
    hop_ms: float,
    mel: np.ndarray,
) -> Path:
    """
    Saves mixed mel into cache_root/<first_label>/[lab1_lab2...]mix_000001__tag.npy
    """
    labels_norm = [l.strip().lower() for l in labels]
    labels_tag = "_".join(labels_norm)

    tag = f"sr{sr}_dur{dur}_m{n_mels}_w{int(win_ms)}_h{int(hop_ms)}"
    fn = f"[{labels_tag}]mix_{idx:06d}__{tag}.npy"

    out_dir = ensure_dir(cache_root / labels_norm[0])
    out_path = out_dir / fn
    np.save(out_path, mel.astype(np.float32))
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate synthetic multilabel mixed mels from wavs")
    ap.add_argument("--config", default="configs/audio_params.yaml")
    ap.add_argument("--labels_file", help="Optional YAML containing train_labels allow-list")
    ap.add_argument("--train_dir", default=None)
    ap.add_argument("--out_cache_root", default="data/processed/log_mels_mixed")
    ap.add_argument("--out_manifest", default="data/processed/train_mels_mixed.csv")
    ap.add_argument("--num_mixes", type=int, default=20000)
    ap.add_argument("--min_sources", type=int, default=2)
    ap.add_argument("--max_sources", type=int, default=3)
    ap.add_argument("--snr_db_min", type=float, default=-5.0)
    ap.add_argument("--snr_db_max", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=1337)

    # Debug wav dumping
    ap.add_argument("--save_wavs", action="store_true")
    ap.add_argument("--wav_out_dir", default="data/debug/mixed_wavs")
    ap.add_argument("--max_wavs", type=int, default=200)

    args = ap.parse_args()

    cfg = load_yaml(Path(args.config))
    audio_cfg = cfg["audio"]
    path_cfg = cfg["paths"]

    random.seed(args.seed)
    np.random.seed(args.seed)

    sr = int(audio_cfg["sr"])
    dur = float(audio_cfg["duration"])
    n_mels = int(audio_cfg["n_mels"])
    win_ms = float(audio_cfg["win_ms"])
    hop_ms = float(audio_cfg["hop_ms"])
    fmin = float(audio_cfg["fmin"])
    fmax = audio_cfg.get("fmax")

    train_dir = Path(args.train_dir or path_cfg["train_dir"])
    cache_root = Path(args.out_cache_root)
    out_csv = Path(args.out_manifest)

    ensure_dir(cache_root)
    ensure_dir(out_csv.parent)

    wav_out_dir = Path(args.wav_out_dir)
    if args.save_wavs:
        wav_out_dir.mkdir(parents=True, exist_ok=True)

    allowed_labels = get_train_labels(cfg)
    if args.labels_file:
        labels_cfg = load_yaml(Path(args.labels_file))
        allowed_labels = get_train_labels(labels_cfg)

    by_label = iter_wavs_by_label(train_dir)
    labels = sorted(by_label.keys())

    if allowed_labels is not None:
        disk_labels = set(labels)
        missing = sorted(allowed_labels - disk_labels)
        extra = sorted(disk_labels - allowed_labels)
        if missing:
            print(f"WARNING: These train_labels were not found on disk: {missing}")
        if extra:
            print(f"INFO: These labels exist on disk but will be skipped: {extra}")

        labels = [lab for lab in labels if lab in allowed_labels]
        by_label = {lab: by_label[lab] for lab in labels}

    if len(labels) < args.min_sources:
        raise ValueError(f"Need at least {args.min_sources} labels to mix; found {len(labels)} in {train_dir}")

    for lab in labels:
        if not by_label[lab]:
            raise ValueError(f"No wavs found for label '{lab}'")

    # Precompute FFT/hop parameters once, reuse for all samples
    n_fft, hop, win_length = calc_fft_hop(sr, win_ms, hop_ms)
    snr_range = (args.snr_db_min, args.snr_db_max)

    print("--- Generating Mixed Mels ---")
    print(f"Source: {train_dir}")
    print(f"Cache:  {cache_root}")
    print(f"Manifest: {out_csv}")
    print(f"Mixes: {args.num_mixes} | sources: {args.min_sources}-{args.max_sources} | snr_db: {snr_range}")
    print(f"Params: SR={sr}, Dur={dur}s, Mels={n_mels}, Win={win_ms}ms, Hop={hop_ms}ms")
    if args.save_wavs:
        print(f"Debug WAVs: saving first {args.max_wavs} mixes to {wav_out_dir}")

    rows_out: List[List[str]] = []

    for i in tqdm(range(args.num_mixes), desc="Mixing"):
        k = random.randint(args.min_sources, args.max_sources)
        chosen_labels = random.sample(labels, k)
        chosen_paths = [random.choice(by_label[lab]) for lab in chosen_labels]

        # Load + duration-fix each source (reusing your functions)
        stereos = []
        for p in chosen_paths:
            stereo = load_audio_stereo(p, target_sr=sr)  # (2, T)
            stereo = ensure_duration(stereo, sr, dur)     # (2, target_T)
            stereos.append(stereo)

        mixed_stereo = mix_stereo_waveforms(stereos, snr_db_range=snr_range)

        # Save debug WAV + label sidecar (INSIDE THE LOOP)
        if args.save_wavs and i < args.max_wavs:
            wav_path = wav_out_dir / f"mix_{i:06d}__[{'_'.join(chosen_labels)}]__sr{sr}.wav"
            sf.write(str(wav_path), mixed_stereo.T, sr, subtype="PCM_16")

            (wav_out_dir / f"mix_{i:06d}__[{'_'.join(chosen_labels)}].txt").write_text(
                "\n".join(chosen_labels) + "\n",
                encoding="utf-8",
            )

        # Compute mel using your existing function
        mel = mel_stereo2_from_stereo(
            mixed_stereo,
            sr=sr,
            n_fft=n_fft,
            hop=hop,
            win_length=win_length,
            n_mels=n_mels,
            fmin=fmin,
            fmax=fmax,
        )  # (2, n_mels, T')

        npy_path = save_mixed_npy(
            cache_root=cache_root,
            labels=chosen_labels,
            idx=i,
            sr=sr,
            dur=dur,
            n_mels=n_mels,
            win_ms=win_ms,
            hop_ms=hop_ms,
            mel=mel,
        )

        # multilabel manifest format: labels joined by |
        rows_out.append([npy_path.resolve().as_posix(), "|".join(chosen_labels)])

    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filepath", "labels"])
        w.writerows(rows_out)

    print("Done.")
    print(f"Wrote {len(rows_out)} rows to: {out_csv}")


if __name__ == "__main__":
    main()
