#!/usr/bin/env python3
import argparse
import csv
import random
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import librosa
import numpy as np
import yaml
from tqdm import tqdm
import soundfile as sf

from preprocessing import (
    calc_fft_hop,
    ensure_duration,
    load_audio_stereo,
    mel_stereo2_from_stereo,
    ensure_dir,
)


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
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
    Mixing is done in the time domain; the same mixed waveform is used
    to compute both Mel and CQT features.
    """
    assert len(stereos) >= 2
    base = stereos[0].astype(np.float32, copy=True)
    base_rms = rms(base)

    mix = base
    for s in stereos[1:]:
        s = s.astype(np.float32, copy=False)
        s_rms = rms(s)
        snr_db = random.uniform(*snr_db_range)
        gain = (base_rms / (s_rms + 1e-12)) * (10 ** (-snr_db / 20.0))
        mix = mix + s * float(gain)

    peak = float(np.max(np.abs(mix)) + 1e-12)
    mix = (0.99 * mix / peak).astype(np.float32)
    return mix


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


def save_mixed_npy(
    cache_root: Path,
    labels: List[str],
    idx: int,
    tag: str,
    array: np.ndarray,
) -> Path:
    labels_norm = [l.strip().lower() for l in labels]
    labels_tag = "_".join(labels_norm)
    fn = f"[{labels_tag}]mix_{idx:06d}__{tag}.npy"
    out_dir = ensure_dir(cache_root / labels_norm[0])
    out_path = out_dir / fn
    np.save(out_path, array.astype(np.float32))
    return out_path


_G_BY_LABEL: Dict[str, List[Path]] = {}
_G_LABELS: List[str] = []
_G_AUDIO: Dict[str, object] = {}
_G_SNR_RANGE: Tuple[float, float] = (0.0, 0.0)
_G_SAVE_WAVS: bool = False
_G_WAV_OUT_DIR: Optional[Path] = None
_G_MAX_WAVS: int = 0
_G_MIN_SOURCES: int = 2
_G_MAX_SOURCES: int = 2
_G_SEED: int = 1337


def _init_worker(
    by_label: Dict[str, List[Path]],
    labels: List[str],
    audio_cfg: Dict[str, object],
    snr_range: Tuple[float, float],
    save_wavs: bool,
    wav_out_dir: Optional[Path],
    max_wavs: int,
    min_sources: int,
    max_sources: int,
    seed: int,
):
    global _G_BY_LABEL, _G_LABELS, _G_AUDIO, _G_SNR_RANGE, _G_SAVE_WAVS, _G_WAV_OUT_DIR
    global _G_MAX_WAVS, _G_MIN_SOURCES, _G_MAX_SOURCES, _G_SEED
    _G_BY_LABEL = by_label
    _G_LABELS = labels
    _G_AUDIO = audio_cfg
    _G_SNR_RANGE = snr_range
    _G_SAVE_WAVS = save_wavs
    _G_WAV_OUT_DIR = wav_out_dir
    _G_MAX_WAVS = max_wavs
    _G_MIN_SOURCES = min_sources
    _G_MAX_SOURCES = max_sources
    _G_SEED = seed


def _mix_one(idx: int):
    try:
        random.seed(_G_SEED + idx)
        np.random.seed(_G_SEED + idx)

        k = random.randint(_G_MIN_SOURCES, _G_MAX_SOURCES)
        chosen_labels = random.sample(_G_LABELS, k)
        chosen_paths = [random.choice(_G_BY_LABEL[lab]) for lab in chosen_labels]

        stereos = []
        for p in chosen_paths:
            stereo = load_audio_stereo(p, target_sr=int(_G_AUDIO["sr"]))
            stereo = ensure_duration(stereo, int(_G_AUDIO["sr"]), float(_G_AUDIO["dur"]))
            stereos.append(stereo)

        mixed_stereo = mix_stereo_waveforms(stereos, snr_db_range=_G_SNR_RANGE)

        if _G_SAVE_WAVS and _G_WAV_OUT_DIR is not None and idx < _G_MAX_WAVS:
            labels_tag = "_".join(chosen_labels)
            wav_path = _G_WAV_OUT_DIR / f"mix_{idx:06d}__[{labels_tag}]__sr{int(_G_AUDIO['sr'])}.wav"
            sf.write(str(wav_path), mixed_stereo.T, int(_G_AUDIO["sr"]), subtype="PCM_16")
            (_G_WAV_OUT_DIR / f"mix_{idx:06d}__[{labels_tag}].txt").write_text(
                "\n".join(chosen_labels) + "\n",
                encoding="utf-8",
            )

        mel = mel_stereo2_from_stereo(
            mixed_stereo,
            sr=int(_G_AUDIO["sr"]),
            n_fft=int(_G_AUDIO["n_fft"]),
            hop=int(_G_AUDIO["hop"]),
            win_length=int(_G_AUDIO["win_length"]),
            n_mels=int(_G_AUDIO["n_mels"]),
            fmin=float(_G_AUDIO["fmin"]),
            fmax=_G_AUDIO.get("fmax"),
        )

        cqt = _cqt_stereo2_from_stereo(
            mixed_stereo,
            sr=int(_G_AUDIO["sr"]),
            n_bins=int(_G_AUDIO["n_bins"]),
            bins_per_octave=int(_G_AUDIO["bins_per_octave"]),
            hop_length=int(_G_AUDIO["hop"]),
            fmin=float(_G_AUDIO["fmin"]),
        )

        mel_tag = f"sr{int(_G_AUDIO['sr'])}_dur{float(_G_AUDIO['dur'])}_m{int(_G_AUDIO['n_mels'])}_w{int(_G_AUDIO['win_ms'])}_h{int(_G_AUDIO['hop_ms'])}"
        cqt_tag = f"sr{int(_G_AUDIO['sr'])}_dur{float(_G_AUDIO['dur'])}_b{int(_G_AUDIO['n_bins'])}_w{int(_G_AUDIO['win_ms'])}_h{int(_G_AUDIO['hop_ms'])}"

        mel_path = save_mixed_npy(
            cache_root=Path(_G_AUDIO["mel_cache_root"]),
            labels=chosen_labels,
            idx=idx,
            tag=mel_tag,
            array=mel,
        )
        cqt_path = save_mixed_npy(
            cache_root=Path(_G_AUDIO["cqt_cache_root"]),
            labels=chosen_labels,
            idx=idx,
            tag=cqt_tag,
            array=cqt,
        )

        return True, idx, mel_path, cqt_path, "|".join(chosen_labels), None
    except Exception as e:
        return False, idx, None, None, None, str(e)


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate synthetic multilabel mixed Mel+CQT from wavs")
    ap.add_argument("--config", default="configs/audio_params.yaml")
    ap.add_argument("--labels_file", help="Optional YAML containing train_labels allow-list")
    ap.add_argument("--train_dir", default=None)
    ap.add_argument("--out_cache_root", default="data/processed/log_mels_mixed")
    ap.add_argument("--out_cqt_root", default="data/processed/log_cqt_mixed")
    ap.add_argument("--out_manifest", default="data/processed/train_mels_mixed.csv")
    ap.add_argument("--num_mixes", type=int, default=20000)
    ap.add_argument("--min_sources", type=int, default=2)
    ap.add_argument("--max_sources", type=int, default=2)
    ap.add_argument("--snr_db_min", type=float, default=-5.0)
    ap.add_argument("--snr_db_max", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--num_workers", type=int, default=19)
    ap.add_argument("--n_bins", type=int, default=None)
    ap.add_argument("--bins_per_octave", type=int, default=12)

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
    n_bins = int(args.n_bins or n_mels)
    if fmin <= 0:
        raise ValueError(f"fmin must be > 0 for CQT, got {fmin}")
    max_freq = float(fmax) if fmax else (sr / 2.0)
    if max_freq <= fmin:
        raise ValueError(f"Invalid CQT range: fmin={fmin} >= max_freq={max_freq}")
    max_bins = int(math.floor(args.bins_per_octave * math.log2(max_freq / fmin)))
    if n_bins > max_bins:
        print(f"[WARN] CQT n_bins={n_bins} exceeds Nyquist; capping to {max_bins}")
        n_bins = max_bins

    train_dir = Path(args.train_dir or path_cfg["train_dir"])
    mel_cache_root = Path(args.out_cache_root)
    cqt_cache_root = Path(args.out_cqt_root)
    out_csv = Path(args.out_manifest)

    ensure_dir(mel_cache_root)
    ensure_dir(cqt_cache_root)
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

    n_fft, hop, win_length = calc_fft_hop(sr, win_ms, hop_ms)
    snr_range = (args.snr_db_min, args.snr_db_max)

    print("--- Generating Mixed Mel+CQT ---")
    print(f"Source: {train_dir}")
    print(f"Mel Cache:  {mel_cache_root}")
    print(f"CQT Cache:  {cqt_cache_root}")
    print(f"Manifest: {out_csv}")
    print(f"Mixes: {args.num_mixes} | sources: {args.min_sources}-{args.max_sources} | snr_db: {snr_range}")
    print(f"Params: SR={sr}, Dur={dur}s, Mels={n_mels}, CQT bins={n_bins}, Win={win_ms}ms, Hop={hop_ms}ms")
    if args.save_wavs:
        print(f"Debug WAVs: saving first {args.max_wavs} mixes to {wav_out_dir}")

    audio_worker_cfg = {
        "sr": sr,
        "dur": dur,
        "n_mels": n_mels,
        "n_bins": n_bins,
        "bins_per_octave": int(args.bins_per_octave),
        "win_ms": win_ms,
        "hop_ms": hop_ms,
        "fmin": fmin,
        "fmax": fmax,
        "n_fft": n_fft,
        "hop": hop,
        "win_length": win_length,
        "mel_cache_root": mel_cache_root,
        "cqt_cache_root": cqt_cache_root,
    }

    num_workers = max(1, int(args.num_workers or 1))
    rows_out_by_idx: Dict[int, List[str]] = {}
    n_ok, n_fail = 0, 0

    init_args = (
        by_label,
        labels,
        audio_worker_cfg,
        snr_range,
        args.save_wavs,
        wav_out_dir if args.save_wavs else None,
        args.max_wavs,
        args.min_sources,
        args.max_sources,
        args.seed,
    )

    if num_workers == 1:
        _init_worker(*init_args)
        for i in tqdm(range(args.num_mixes), desc="Mixing"):
            ok, idx, mel_path, cqt_path, labels_joined, err = _mix_one(i)
            if ok:
                rows_out_by_idx[idx] = [
                    mel_path.resolve().as_posix(),
                    cqt_path.resolve().as_posix(),
                    labels_joined,
                ]
                n_ok += 1
            else:
                n_fail += 1
                print(f"[WARN] Failed mix {idx}: {err}")
    else:
        with ProcessPoolExecutor(max_workers=num_workers, initializer=_init_worker, initargs=init_args) as executor:
            futures = [executor.submit(_mix_one, i) for i in range(args.num_mixes)]
            try:
                for fut in tqdm(as_completed(futures), total=len(futures), desc="Mixing"):
                    ok, idx, mel_path, cqt_path, labels_joined, err = fut.result()
                    if ok:
                        rows_out_by_idx[idx] = [
                            mel_path.resolve().as_posix(),
                            cqt_path.resolve().as_posix(),
                            labels_joined,
                        ]
                        n_ok += 1
                    else:
                        n_fail += 1
                        print(f"[WARN] Failed mix {idx}: {err}")
            except KeyboardInterrupt:
                print("\nStopped by user.")
                for fut in futures:
                    fut.cancel()

    rows_out = [rows_out_by_idx[i] for i in sorted(rows_out_by_idx)]

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["filepath", "cqt_path", "labels"])
        w.writerows(rows_out)

    print("Done.")
    print(f"Wrote {len(rows_out)} rows to: {out_csv}")


if __name__ == "__main__":
    main()
