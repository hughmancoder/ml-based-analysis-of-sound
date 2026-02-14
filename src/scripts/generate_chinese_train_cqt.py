#!/usr/bin/env python3
import argparse
import hashlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, Set, Tuple
import math

import librosa
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

from preprocessing import load_audio_stereo, ensure_duration
from utils.safe_paths import guard_path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def get_train_labels(cfg: dict) -> Optional[Set[str]]:
    """
    Supports either:
      - top-level: train_labels: [...]
      - nested: dataset: { train_labels: [...] }

    Returns a lowercased set or None if not present.
    """
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


def _hash_path(p: str) -> str:
    return hashlib.md5(p.encode("utf-8")).hexdigest()[:10]


def _read_csv_with_fallback(path: Path) -> pd.DataFrame:
    encodings = ("utf-8", "utf-8-sig", "gbk", "cp936")
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path)


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


def _process_one(
    wav_path: Path,
    label: str,
    cache_root: Path,
    sr: int,
    dur: float,
    n_bins: int,
    bins_per_octave: int,
    win_ms: float,
    hop_ms: float,
    fmin: float,
):
    try:
        stereo = load_audio_stereo(wav_path, target_sr=sr)
        stereo = ensure_duration(stereo, sr, dur)
        hop_length = int(round(sr * (hop_ms / 1000.0)))

        cqt = _cqt_stereo2_from_stereo(
            stereo,
            sr=sr,
            n_bins=n_bins,
            bins_per_octave=bins_per_octave,
            hop_length=hop_length,
            fmin=fmin,
        )

        tag = f"sr{sr}_dur{dur}_b{n_bins}_w{int(win_ms)}_h{int(hop_ms)}"
        fn = f"{wav_path.stem}__{_hash_path(str(wav_path))}__{tag}.npy"
        out_path = (cache_root / label / fn).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, cqt.astype(np.float32))
        return True, out_path, label, wav_path, None
    except Exception as e:
        return False, None, label, wav_path, str(e)


def main() -> None:
    ap = argparse.ArgumentParser(description="Precompute CQT Spectrograms into .npy files")
    ap.add_argument("--config", default="configs/audio_params.yaml", help="Path to YAML config")
    ap.add_argument("--labels_file", help="Optional YAML containing train_labels allow-list")
    ap.add_argument("--train_dir", help="Override the raw data source directory")
    ap.add_argument("--cqt_cache_root", default="data/processed/log_cqt", help="Output root for CQT .npy files")
    ap.add_argument("--num_workers", type=int, default=19)
    ap.add_argument(
        "--no_enforce_one_level",
        action="store_true",
        help="Allow nested subfolders (root/label/subdir/*.wav). Default enforces root/label/*.wav only.",
    )
    ap.add_argument("--n_bins", type=int, default=None, help="Number of CQT frequency bins (default: n_mels)")
    ap.add_argument("--bins_per_octave", type=int, default=12)
    args = ap.parse_args()

    cfg = load_yaml(Path(args.config))
    audio_cfg = cfg["audio"]
    path_cfg = cfg["paths"]

    raw_dir = Path(args.train_dir or path_cfg["train_dir"])
    cache_root = guard_path(Path(args.cqt_cache_root), PROJECT_ROOT, "cqt_cache_root")

    if "manifest_path" in path_cfg:
        manifest_path = path_cfg["manifest_path"]
    elif "train_manifest" in path_cfg:
        manifest_path = path_cfg["train_manifest"]
    else:
        raise KeyError(
            "Missing manifest path in config. Expected 'manifest_path' or 'train_manifest' under 'paths'."
        )

    out_csv = guard_path(Path(manifest_path), PROJECT_ROOT, "manifest_path")
    if not out_csv.exists():
        raise FileNotFoundError(f"Mel manifest not found: {out_csv} (run mel generation first)")

    # Load allowed labels
    allowed_labels = get_train_labels(cfg)
    if args.labels_file:
        labels_cfg = load_yaml(Path(args.labels_file))
        allowed_labels = get_train_labels(labels_cfg)

    df = _read_csv_with_fallback(out_csv)
    if "wavpath" not in df.columns:
        raise ValueError("Manifest must contain a 'wavpath' column to generate CQT features.")

    cache_root.mkdir(parents=True, exist_ok=True)

    sr = int(audio_cfg["sr"])
    dur = float(audio_cfg["duration"])
    win_ms = float(audio_cfg["win_ms"])
    hop_ms = float(audio_cfg["hop_ms"])
    fmin = float(audio_cfg["fmin"])
    fmax = audio_cfg.get("fmax")
    n_bins = int(args.n_bins or audio_cfg.get("n_mels", 128))
    if fmin <= 0:
        raise ValueError(f"fmin must be > 0 for CQT, got {fmin}")
    max_freq = float(fmax) if fmax else (sr / 2.0)
    if max_freq <= fmin:
        raise ValueError(f"Invalid CQT range: fmin={fmin} >= max_freq={max_freq}")
    max_bins = int(math.floor(args.bins_per_octave * math.log2(max_freq / fmin)))
    if n_bins > max_bins:
        print(f"[WARN] CQT n_bins={n_bins} exceeds Nyquist; capping to {max_bins}")
        n_bins = max_bins

    print("--- Processing Audio (CQT) ---")
    print(f"Source: {raw_dir}")
    print(f"Cache:  {cache_root}")
    print(f"Manifest: {out_csv}")
    print(f"Params: SR={sr}, Bins={n_bins}, Dur={dur}s")
    if allowed_labels is None:
        print("Labels: (no train_labels provided) -> processing ALL labels discovered on disk")
    else:
        print(f"Labels: {sorted(allowed_labels)}")

    rows_out = [None] * len(df)
    n_ok, n_fail = 0, 0

    num_workers = max(1, int(args.num_workers or 1))

    def _row_iter():
        for idx, row in df.iterrows():
            label = str(row.get("label", "")).strip().lower()
            if not label and "labels" in row:
                label = str(row["labels"]).split("|")[0].strip().lower()

            if allowed_labels is not None and label not in allowed_labels:
                continue

            wav_path = Path(str(row["wavpath"]))
            if not wav_path.is_absolute():
                wav_path = (PROJECT_ROOT / wav_path).resolve()
            yield idx, wav_path, label

    row_items = list(_row_iter())

    if num_workers == 1:
        for idx, wav_path, label in tqdm(row_items, desc="Generating CQTs"):
            ok, out_path, label, wav_path, err = _process_one(
                wav_path, label, cache_root, sr, dur, n_bins, args.bins_per_octave, win_ms, hop_ms, fmin
            )
            if ok:
                rows_out[idx] = out_path.as_posix()
                n_ok += 1
            else:
                print("INFO: Failed to process:", wav_path, "Error:", err)
                n_fail += 1
    else:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(
                    _process_one,
                    wav_path,
                    label,
                    cache_root,
                    sr,
                    dur,
                    n_bins,
                    args.bins_per_octave,
                    win_ms,
                    hop_ms,
                    fmin,
                ): idx
                for idx, wav_path, label in row_items
            }
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Generating CQTs"):
                idx = futures[fut]
                ok, out_path, label, wav_path, err = fut.result()
                if ok:
                    rows_out[idx] = out_path.as_posix()
                    n_ok += 1
                else:
                    print("INFO: Failed to process:", wav_path, "Error:", err)
                    n_fail += 1

    df["cqt_path"] = rows_out
    df.to_csv(out_csv, index=False, encoding="utf-8")

    print("\nProcessing Complete.")
    print(f"Successfully processed: {n_ok}")
    print(f"Failed: {n_fail}")
    print(f"Manifest updated: {out_csv}")


if __name__ == "__main__":
    main()
