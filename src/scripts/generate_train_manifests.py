#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import argparse, csv, os
from utils.safe_paths import guard_path

# TODO: this should be generated from mel spectrogram file rather than irmas
def walk_audio(root: Path):
    for p in root.rglob("*.wav"):
        yield p

def class_from_irmas(p: Path) -> str:
    # IRMAS-TrainingData/<cls>/[cls]xxx.wav  -> folder is authoritative
    return p.parent.name.lower()

def class_from_folder(p: Path) -> str:
    return p.parent.name.lower()

PROJECT_ROOT = Path(__file__).resolve().parents[2]

def to_relative_path(path: Path, base: Path = PROJECT_ROOT) -> str:
    try:
        return path.relative_to(base).as_posix()
    except ValueError:
        # Fall back to os.path.relpath for cases outside the base tree.
        return Path(os.path.relpath(path, start=base)).as_posix()


def write_manifest(rows, out_csv: Path):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["filepath","label"])
        w.writerows(rows)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--irmas_dir", type=str, required=True)
    ap.add_argument("--chinese_dir", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    args = ap.parse_args()

    out_dir = guard_path(Path(args.out_dir), PROJECT_ROOT, "out_dir")

    # IRMAS
    irmas = Path(args.irmas_dir)
    if irmas.exists():
        rows = []
        for p in walk_audio(irmas):
            rows.append([to_relative_path(p), class_from_irmas(p)])
        write_manifest(rows, out_dir / "irmas_train.csv")
    else:
        print(f"Warning: irmas_dir {irmas} does not exist")

    # Chinese
    chin = Path(args.chinese_dir)
    if chin.exists():
        rows = []
        for p in walk_audio(chin):
            rows.append([to_relative_path(p), class_from_folder(p)])
        write_manifest(rows, out_dir / "chinese_instruments.csv")
    else:
        print(f"Warning: chinese_dir {chin} does not exist")

if __name__ == "__main__":
    main()
