#!/usr/bin/env python3
"""
Generate 2-channel CQT .npy tensors from IRMAS train audio and update a mel manifest CSV.

This script is coupled to the mel manifest and will add/update a `cqt_path` column
in the same CSV (no separate manifest is created).
"""
from __future__ import annotations
import argparse, csv, sys, hashlib
from concurrent.futures import ProcessPoolExecutor, as_completed
import math
from pathlib import Path
from typing import Optional, List, Iterable, Tuple

import librosa
import numpy as np
import pandas as pd

from preprocessing import load_audio_stereo, ensure_duration

PROJECT_ROOT = Path(__file__).resolve().parents[2]

try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kwargs): return x


def _read_csv_with_fallback(path: Path) -> pd.DataFrame:
    encodings = ("utf-8", "utf-8-sig", "gbk", "cp936")
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path)


def _hash_path(p: str) -> str:
    return hashlib.md5(p.encode("utf-8")).hexdigest()[:10]


def _safe_relpath(path: Path, base: Path) -> str:
    try:
        return path.relative_to(base).as_posix()
    except ValueError:
        return path.as_posix()


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
) -> Tuple[bool, Optional[Path], str, Path, Optional[str]]:
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


def _extract_mel_stem_and_hash(mel_path: Path) -> Tuple[str, Optional[str]]:
    stem = mel_path.stem
    parts = stem.split("__")
    if len(parts) >= 2:
        return parts[0], parts[1]
    return stem, None


def _match_wav_by_hash(candidates: List[Path], target_hash: str) -> Optional[Path]:
    for p in candidates:
        tests = [
            str(p),
            p.as_posix(),
            str(p.resolve()),
            p.resolve().as_posix(),
        ]
        if any(_hash_path(s) == target_hash for s in tests):
            return p
    return None


def _resolve_wav_path(train_dir: Path, label: str, mel_path: Path) -> Optional[Path]:
    wav_stem, wav_hash = _extract_mel_stem_and_hash(mel_path)
    label_dir = train_dir / label

    candidate = label_dir / f"{wav_stem}.wav"
    if candidate.exists():
        return candidate

    matches = list(label_dir.rglob(f"{wav_stem}.wav"))
    if matches:
        return matches[0]

    if wav_hash:
        all_wavs = list(label_dir.rglob("*.wav"))
        match = _match_wav_by_hash(all_wavs, wav_hash)
        if match:
            return match

    return None


def main():
    ap = argparse.ArgumentParser(
        description="Precompute 2ch CQT tensors (.npy) from IRMAS train audio and update a mel manifest."
    )
    ap.add_argument("--irmas_train_dir", type=str, required=True, help="Root of IRMAS train split; expects <label>/**/*.wav")
    ap.add_argument("--cache_root", required=True, type=str, help="Root folder for cached .npy.")
    ap.add_argument("--mel_manifest_out", required=True, type=str, help="Existing mel manifest CSV to update.")

    # CQT params
    ap.add_argument("--sr", type=int, default=44100)
    ap.add_argument("--dur", type=float, default=3.0)
    ap.add_argument("--n_bins", type=int, default=128)
    ap.add_argument("--bins_per_octave", type=int, default=12)
    ap.add_argument("--win_ms", type=float, default=30.0)
    ap.add_argument("--hop_ms", type=float, default=10.0)
    ap.add_argument("--fmin", type=float, default=20.0)
    ap.add_argument("--fmax", type=float, default=None)
    ap.add_argument("--num_workers", type=int, default=19)

    args = ap.parse_args()

    train_dir = Path(args.irmas_train_dir)
    cache_root = Path(args.cache_root)
    out_csv = Path(args.mel_manifest_out)

    if not out_csv.exists():
        print(f"ERROR: mel manifest not found: {out_csv}", file=sys.stderr)
        sys.exit(2)

    df = _read_csv_with_fallback(out_csv)
    if "filepath" not in df.columns:
        raise ValueError("Manifest must have a 'filepath' column.")
    col_map = {c.lower(): c for c in df.columns}
    wavpath_col = col_map.get("wavpath") or col_map.get("wav_path")

    cache_root.mkdir(parents=True, exist_ok=True)

    n_bins = int(args.n_bins)
    if args.fmin <= 0:
        raise ValueError(f"fmin must be > 0 for CQT, got {args.fmin}")
    max_freq = float(args.fmax) if args.fmax else (args.sr / 2.0)
    if max_freq <= args.fmin:
        raise ValueError(f"Invalid CQT range: fmin={args.fmin} >= max_freq={max_freq}")
    max_bins = int(math.floor(args.bins_per_octave * math.log2(max_freq / args.fmin)))
    if n_bins > max_bins:
        print(f"[WARN] CQT n_bins={n_bins} exceeds Nyquist; capping to {max_bins}")
        n_bins = max_bins

    print("--- Processing Audio (CQT) ---")
    print(f"Source: {train_dir}")
    print(f"Cache:  {cache_root}")
    print(f"Manifest: {out_csv}")
    print(f"Params: SR={args.sr}, Bins={n_bins}, Dur={args.dur}s")

    rows_out: List[Optional[str]] = [None] * len(df)
    n_ok = n_fail = 0

    row_items: List[Tuple[int, Path, str]] = []
    for idx, row in df.iterrows():
        mel_path = Path(str(row["filepath"]))
        if not mel_path.is_absolute():
            mel_path = (PROJECT_ROOT / mel_path).resolve()

        label = str(row.get("label", "")).strip().lower()
        if not label:
            label = mel_path.parent.name.strip().lower()

        if wavpath_col and str(row.get(wavpath_col, "")).strip():
            wav_path = Path(str(row[wavpath_col]))
            if not wav_path.is_absolute():
                wav_path = (PROJECT_ROOT / wav_path).resolve()
        else:
            wav_path = _resolve_wav_path(train_dir, label, mel_path)

        if wav_path is None or not wav_path.exists():
            n_fail += 1
            print(f"[WARN] Failed: {row.get('filepath')} (Could not resolve wav for {mel_path})", file=sys.stderr)
            continue

        row_items.append((idx, wav_path, label))

    num_workers = max(1, int(args.num_workers or 1))

    if num_workers == 1:
        for idx, wav_path, label in tqdm(row_items, total=len(row_items), desc="Pre-caching CQTs"):
            ok, out_path, _label, _wav_path, err = _process_one(
                wav_path,
                label,
                cache_root,
                args.sr,
                args.dur,
                n_bins,
                args.bins_per_octave,
                args.win_ms,
                args.hop_ms,
                args.fmin,
            )
            if ok:
                rows_out[idx] = _safe_relpath(out_path, PROJECT_ROOT)
                n_ok += 1
            else:
                n_fail += 1
                print(f"[WARN] Failed: {wav_path} ({err})", file=sys.stderr)
    else:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(
                    _process_one,
                    wav_path,
                    label,
                    cache_root,
                    args.sr,
                    args.dur,
                    n_bins,
                    args.bins_per_octave,
                    args.win_ms,
                    args.hop_ms,
                    args.fmin,
                ): idx
                for idx, wav_path, label in row_items
            }
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Pre-caching CQTs"):
                idx = futures[fut]
                try:
                    ok, out_path, _label, _wav_path, err = fut.result()
                    if ok:
                        rows_out[idx] = _safe_relpath(out_path, PROJECT_ROOT)
                        n_ok += 1
                    else:
                        n_fail += 1
                        print(f"[WARN] Failed: {_wav_path} ({err})", file=sys.stderr)
                except Exception as e:
                    n_fail += 1
                    print(f"[WARN] Failed task: {e}", file=sys.stderr)

    df["cqt_path"] = rows_out
    df.to_csv(out_csv, index=False, encoding="utf-8")

    print(f"\nManifest updated: {out_csv}")
    print(f"Cache root: {cache_root.resolve()}")
    print(f"Success: {n_ok} | Failed: {n_fail}")


if __name__ == "__main__":
    main()
