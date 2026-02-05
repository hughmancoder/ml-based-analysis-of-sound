#!/usr/bin/env python3
"""
Generate CQT windows and update an existing test manifest CSV.

Each row in the manifest will get a new `cqt_path` column.
"""
from __future__ import annotations
import argparse, csv, sys, traceback, hashlib
from concurrent.futures import ProcessPoolExecutor, as_completed
import math
from pathlib import Path
from typing import List

import librosa
import numpy as np
import pandas as pd
import soundfile as sf

from preprocessing import load_audio_stereo, ensure_duration

try:
    from tqdm import tqdm
except Exception:
    def tqdm(x=None, **kwargs):
        return x


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


def _process_full_wav(
    idx: int,
    wav_path_str: str,
    cache_root_str: str,
    sr: int,
    dur: float,
    n_bins: int,
    bins_per_octave: int,
    win_ms: float,
    hop_ms: float,
    fmin: float,
) -> tuple[bool, int, str | None, str | None]:
    try:
        wav_path = Path(wav_path_str)
        cache_root = Path(cache_root_str)

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
        out_path = (cache_root / fn).resolve()
        np.save(out_path, cqt.astype(np.float32))
        return True, idx, str(out_path), None
    except Exception as e:
        return False, idx, None, str(e)


def _process_segment(
    idx: int,
    wav_path_str: str,
    cache_root_str: str,
    sr: int,
    dur: float,
    n_bins: int,
    bins_per_octave: int,
    win_ms: float,
    hop_ms: float,
    fmin: float,
    start_ms: int,
) -> tuple[bool, int, str | None, str | None]:
    try:
        wav_path = Path(wav_path_str)
        cache_root = Path(cache_root_str)

        start_s = start_ms / 1000.0
        stereo = _load_segment_stereo(wav_path, sr, start_s, dur)
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

        tag = (
            f"sr{sr}_dur{dur}_b{n_bins}"
            f"_w{int(win_ms)}_h{int(hop_ms)}_s{start_ms}"
        )
        out_path = (cache_root / f"{wav_path.stem}__{_hash_path(str(wav_path))}__{tag}.npy").resolve()
        np.save(out_path, cqt.astype(np.float32))
        return True, idx, str(out_path), None
    except Exception as e:
        return False, idx, None, str(e)


def _load_segment_stereo(path: Path, sr: int, start_s: float, dur_s: float) -> np.ndarray:
    start_frame = int(round(start_s * sr))
    frames = int(round(dur_s * sr))
    x, sr_in = sf.read(str(path), always_2d=True, start=start_frame, frames=frames)
    x = x.T
    if sr_in != sr:
        x = librosa.resample(x, orig_sr=sr_in, target_sr=sr, res_type="kaiser_fast")
    if x.shape[0] == 1:
        x = np.vstack([x, x])
    elif x.shape[0] > 2:
        x = x[:2, :]
    return x.astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description="Generate CQT windows + update manifest")
    ap.add_argument("--input_dir", required=True, help="Root directory containing .wav files")
    ap.add_argument("--cache_root", required=True, help="Where to save generated CQT .npy files")
    ap.add_argument("--manifest_out", required=True, help="Existing CSV manifest to update")
    ap.add_argument("--project_root", type=str, default=".", help="Project root for relative paths")
    ap.add_argument("--dataset_name", type=str, default="IRMAS")

    # Audio config
    ap.add_argument("--sr", type=int, default=44100)
    ap.add_argument("--dur", type=float, default=3.0)
    ap.add_argument("--n_bins", type=int, default=128)
    ap.add_argument("--bins_per_octave", type=int, default=12)
    ap.add_argument("--win_ms", type=float, default=30.0)
    ap.add_argument("--hop_ms", type=float, default=10.0)
    ap.add_argument("--fmin", type=float, default=20.0)
    ap.add_argument("--fmax", type=float, default=None)
    ap.add_argument("--stride_s", type=float, default=1.5)
    ap.add_argument("--num_workers", type=int, default=19)
    args = ap.parse_args()

    input_root = Path(args.input_dir)
    cache_root = Path(args.cache_root).resolve()
    manifest_out = Path(args.manifest_out)
    project_root = Path(args.project_root).resolve()

    if not manifest_out.exists():
        print(f"[ERROR] Manifest not found: {manifest_out}", file=sys.stderr)
        sys.exit(2)

    df = _read_csv_with_fallback(manifest_out)
    cache_root.mkdir(parents=True, exist_ok=True)

    hop_length = int(round(args.sr * (args.hop_ms / 1000.0)))
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

    rows_out: List[str] = [""] * len(df)
    n_ok = n_fail = 0
    num_workers = max(1, int(args.num_workers or 1))

    if "wav_path" in df.columns:
        print("--- Processing Audio (CQT) ---")
        print(f"Source: {input_root}")
        print(f"Cache:  {cache_root}")
        print(f"Manifest: {manifest_out}")
        print(f"Params: SR={args.sr}, Bins={n_bins}, Dur={args.dur}s")

        row_items = []
        for idx, row in df.iterrows():
            wav_path = Path(str(row["wav_path"]))
            if not wav_path.is_absolute():
                wav_path = (project_root / wav_path).resolve()
            row_items.append((idx, str(wav_path)))

        if num_workers == 1:
            for idx, wav_path_str in tqdm(
                row_items,
                total=len(row_items),
                desc="Pre-caching CQTs",
                unit="file",
                dynamic_ncols=True,
                file=sys.stdout,
            ):
                ok, out_idx, out_path_str, err = _process_full_wav(
                    idx,
                    wav_path_str,
                    str(cache_root),
                    args.sr,
                    args.dur,
                    n_bins,
                    args.bins_per_octave,
                    args.win_ms,
                    args.hop_ms,
                    args.fmin,
                )
                if ok and out_path_str:
                    rows_out[out_idx] = _safe_relpath(Path(out_path_str), project_root)
                    n_ok += 1
                else:
                    n_fail += 1
                    print(f"[WARN] Failed {wav_path_str}: {err}", file=sys.stderr)
        else:
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = {
                    executor.submit(
                        _process_full_wav,
                        idx,
                        wav_path_str,
                        str(cache_root),
                        args.sr,
                        args.dur,
                        n_bins,
                        args.bins_per_octave,
                        args.win_ms,
                        args.hop_ms,
                        args.fmin,
                    ): wav_path_str
                    for idx, wav_path_str in row_items
                }
                pbar = tqdm(
                    total=len(futures),
                    desc="Pre-caching CQTs",
                    unit="file",
                    dynamic_ncols=True,
                    file=sys.stdout,
                )
                for fut in as_completed(futures):
                    wav_path_str = futures[fut]
                    try:
                        ok, out_idx, out_path_str, err = fut.result()
                        if ok and out_path_str:
                            rows_out[out_idx] = _safe_relpath(Path(out_path_str), project_root)
                            n_ok += 1
                        else:
                            n_fail += 1
                            print(f"[WARN] Failed {wav_path_str}: {err}", file=sys.stderr)
                    except Exception as e:
                        n_fail += 1
                        print(f"[WARN] Failed {wav_path_str}: {e}", file=sys.stderr)
                    finally:
                        pbar.update(1)
                pbar.close()

    elif "filename" in df.columns and "start_ms" in df.columns:
        print("--- Processing Audio (CQT windows) ---")
        print(f"Source: {input_root}")
        print(f"Cache:  {cache_root}")
        print(f"Manifest: {manifest_out}")
        print(f"Params: SR={args.sr}, Bins={n_bins}, Dur={args.dur}s")

        row_items = []
        for idx, row in df.iterrows():
            wav_path = (input_root / str(row["filename"])).resolve()
            start_ms = int(row["start_ms"])
            row_items.append((idx, str(wav_path), start_ms))

        if num_workers == 1:
            for idx, wav_path_str, start_ms in tqdm(
                row_items,
                total=len(row_items),
                desc="Pre-caching CQTs",
                unit="file",
                dynamic_ncols=True,
                file=sys.stdout,
            ):
                ok, out_idx, out_path_str, err = _process_segment(
                    idx,
                    wav_path_str,
                    str(cache_root),
                    args.sr,
                    args.dur,
                    n_bins,
                    args.bins_per_octave,
                    args.win_ms,
                    args.hop_ms,
                    args.fmin,
                    start_ms,
                )
                if ok and out_path_str:
                    rows_out[out_idx] = _safe_relpath(Path(out_path_str), project_root)
                    n_ok += 1
                else:
                    n_fail += 1
                    print(f"[WARN] Failed {wav_path_str}: {err}", file=sys.stderr)
        else:
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = {
                    executor.submit(
                        _process_segment,
                        idx,
                        wav_path_str,
                        str(cache_root),
                        args.sr,
                        args.dur,
                        n_bins,
                        args.bins_per_octave,
                        args.win_ms,
                        args.hop_ms,
                        args.fmin,
                        start_ms,
                    ): (wav_path_str, start_ms)
                    for idx, wav_path_str, start_ms in row_items
                }
                pbar = tqdm(
                    total=len(futures),
                    desc="Pre-caching CQTs",
                    unit="file",
                    dynamic_ncols=True,
                    file=sys.stdout,
                )
                for fut in as_completed(futures):
                    wav_path_str, _start_ms = futures[fut]
                    try:
                        ok, out_idx, out_path_str, err = fut.result()
                        if ok and out_path_str:
                            rows_out[out_idx] = _safe_relpath(Path(out_path_str), project_root)
                            n_ok += 1
                        else:
                            n_fail += 1
                            print(f"[WARN] Failed {wav_path_str}: {err}", file=sys.stderr)
                    except Exception as e:
                        n_fail += 1
                        print(f"[WARN] Failed {wav_path_str}: {e}", file=sys.stderr)
                    finally:
                        pbar.update(1)
                pbar.close()
    else:
        print("[ERROR] Manifest must contain either wav_path or (filename, start_ms).", file=sys.stderr)
        sys.exit(2)

    df["cqt_path"] = rows_out
    df.to_csv(manifest_out, index=False, encoding="utf-8")

    print(f"\nManifest updated: {manifest_out}")
    print(f"Cache root: {cache_root}")
    print(f"Clips OK: {n_ok} | Failed: {n_fail}")


if __name__ == "__main__":
    main()
