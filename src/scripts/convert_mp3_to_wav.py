#!/usr/bin/env python3
import argparse
import subprocess
import sys
from pathlib import Path
from utils.safe_paths import guard_path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert .mp3 files to .wav (in place)")
    ap.add_argument("--root", default="data/audio/chinese_instruments", help="Root directory to scan")
    ap.add_argument("--sr", type=int, default=44100)
    ap.add_argument("--channels", type=int, default=2)
    ap.add_argument("--ext", default=".mp3")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    root = guard_path(Path(args.root), PROJECT_ROOT, "root")
    if not root.exists():
        print(f"[ERROR] Root not found: {root}", file=sys.stderr)
        sys.exit(2)

    ext = args.ext.lower()
    files = [p for p in root.rglob(f"*{ext}") if p.is_file()]
    if not files:
        print(f"No {ext} files found under {root}")
        return

    n_ok = n_skip = n_fail = 0
    for src in files:
        dst = src.with_suffix(".wav")
        if dst.exists() and not args.overwrite:
            n_skip += 1
            continue

        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(src),
            "-ac",
            str(args.channels),
            "-ar",
            str(args.sr),
            str(dst),
        ]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            n_ok += 1
        except subprocess.CalledProcessError:
            n_fail += 1
            print(f"[WARN] Failed: {src}", file=sys.stderr)

    print(f"Converted: {n_ok} | Skipped: {n_skip} | Failed: {n_fail}")


if __name__ == "__main__":
    main()
