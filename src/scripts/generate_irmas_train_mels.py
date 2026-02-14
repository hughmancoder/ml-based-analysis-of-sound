#!/usr/bin/env python3
"""
Generate 2-channel mel-dB .npy tensors from IRMAS train audio and emit a manifest CSV.

Two input modes:
  A) --irmas_train_dir  (expected by Makefile target `irmas_train_mels`)
     Assumes structure: <irmas_train_dir>/<label>/**/*.wav
  B) --manifest_csv (legacy)
     CSV with columns: filepath,label pointing to raw WAVs.

Output CSV columns: filepath,label  (filepath points to generated .npy)
"""
from __future__ import annotations
import argparse, csv, sys, traceback
from pathlib import Path
from typing import Optional, List, Iterable, Tuple

import numpy as np
import pandas as pd

# Import the EXACT helpers you already use
from utils.mel_utils import (
    # _hash_path,
    # load_audio_stereo,
    # ensure_duration,
    # calc_fft_hop,
    # mel_stereo2_from_stereo,
    _safe_relpath,
    precache_one,
)
from utils.safe_paths import guard_path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kwargs): return x

def _iter_wavs_from_train_dir(root: Path) -> Iterable[Tuple[Path, str]]:
    """
    Yield (wav_path, label) by walking <root>/<label>/**/*.wav.
    Label is taken from the immediate parent directory name.
    """
    if not root.exists():
        raise FileNotFoundError(f"--irmas_train_dir not found: {root}")
    # Accept both shallow and nested layouts: label is the direct parent folder
    for wav in root.rglob("*"):
        if wav.is_file():
            label = wav.parent.name.strip().lower()
            yield wav, label


def _iter_wavs_from_manifest_csv(manifest_csv: Path) -> Iterable[Tuple[Path, str]]:
    if not manifest_csv.exists():
        raise FileNotFoundError(f"--manifest_csv not found: {manifest_csv}")
    df = _read_csv_with_fallback(manifest_csv)
    df = df.rename(columns={c: c.lower() for c in df.columns})
    if not {"filepath", "label"}.issubset(df.columns):
        raise ValueError("manifest must have columns: filepath,label")
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    for _, row in df.iterrows():
        yield Path(row["filepath"]), str(row["label"])


def _read_csv_with_fallback(path: Path) -> pd.DataFrame:
    encodings = ("utf-8", "utf-8-sig", "gbk", "cp936")
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path)

def main():
    ap = argparse.ArgumentParser(
        description="Precompute 2ch mel-dB tensors (.npy) from IRMAS train audio and emit a new manifest."
    )
    # Mode A (Makefile)
    ap.add_argument("--irmas_train_dir", type=str, help="Root of IRMAS train split; expects <label>/**/*.wav")
    # Mode B (legacy)
    ap.add_argument("--manifest_csv", type=str, help="CSV: filepath,label (paths to RAW WAVs).")

    ap.add_argument("--cache_root", required=True, type=str, help="Root folder for cached .npy.")
    ap.add_argument("--mel_manifest_out", required=True, type=str, help="Output CSV listing .npy paths and labels.")

    # Kept for CLI compatibility (not used directly here)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=19)

    # Mel params (must match your training pipeline)
    ap.add_argument("--sr", type=int, default=44100)
    ap.add_argument("--dur", type=float, default=3.0)
    ap.add_argument("--n_mels", type=int, default=128)
    ap.add_argument("--win_ms", type=float, default=30.0)
    ap.add_argument("--hop_ms", type=float, default=10.0)
    ap.add_argument("--fmin", type=float, default=20.0)
    ap.add_argument("--fmax", type=float, default=None)

    args = ap.parse_args()

    cache_root = guard_path(Path(args.cache_root), PROJECT_ROOT, "cache_root")
    out_csv = guard_path(Path(args.mel_manifest_out), PROJECT_ROOT, "mel_manifest_out")

    # Decide input mode
    mode = None
    if args.irmas_train_dir:
        mode = "train_dir"
        pairs_iter = list(_iter_wavs_from_train_dir(Path(args.irmas_train_dir)))
    elif args.manifest_csv:
        mode = "manifest_csv"
        pairs_iter = list(_iter_wavs_from_manifest_csv(Path(args.manifest_csv)))
    else:
        print("ERROR: provide either --irmas_train_dir or --manifest_csv", file=sys.stderr)
        sys.exit(2)

    rows_out: List[List[str]] = []
    n_ok, n_fail = 0, 0

    for wav_path, label in tqdm(pairs_iter, total=len(pairs_iter), desc="Pre-caching mels"):
        try:
            if not wav_path.exists():
                raise FileNotFoundError(str(wav_path))
            npy_path = precache_one(
                wav_path, label, cache_root,
                sr=args.sr, dur=args.dur, n_mels=args.n_mels,
                win_ms=args.win_ms, hop_ms=args.hop_ms,
                fmin=args.fmin, fmax=args.fmax
            )
            rows_out.append([_safe_relpath(npy_path, PROJECT_ROOT), label])
            n_ok += 1
        except Exception as e:
            n_fail += 1
            print(f"[WARN] Failed: {wav_path} ({e})", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["filepath", "label"])
        w.writerows(rows_out)

    if mode == "train_dir":
        print(f"IRMAS train dir: {Path(args.irmas_train_dir).resolve()}")
    else:
        print(f"Input manifest: {Path(args.manifest_csv).resolve()}")
    print(f"Cache root: {cache_root.resolve()}")
    print(f"Manifest written: {out_csv.resolve()}")
    print(f"Success: {n_ok} | Failed: {n_fail}")


if __name__ == "__main__":
    main()
