#!/usr/bin/env python3
import argparse
import yaml
import csv
from pathlib import Path
from typing import Iterable, Optional, Set, Tuple

from tqdm import tqdm
from preprocessing import precache_one

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def load_yaml(path: Path) -> dict:
    with open(path, "r") as f:
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


def _iter_wavs_from_train_dir(
    root: Path,
    allowed_labels: Optional[Set[str]] = None,
    enforce_one_level: bool = True,
) -> Iterable[Tuple[Path, str]]:
    """
    Finds all .wav files in the directory.

    If enforce_one_level=True, only accepts files under:
        root/label/*.wav
    so that deeper nesting (root/label/subdir/file.wav) does not create junk labels.

    If allowed_labels is provided, yields only those labels.
    """
    if not root.exists():
        raise FileNotFoundError(f"Directory not found: {root}")

    root_resolved = root.resolve()

    for wav in root.rglob("*.wav"):
        if not wav.is_file():
            continue

        label_dir = wav.parent

        if enforce_one_level:
            # require root/label/file.wav exactly
            if label_dir.parent.resolve() != root_resolved:
                continue

        label = label_dir.name.strip().lower()

        if allowed_labels is not None and label not in allowed_labels:
            continue

        yield wav, label


def main() -> None:
    ap = argparse.ArgumentParser(description="Precompute Mel Spectrograms into .npy files")
    ap.add_argument("--config", default="configs/audio_params.yaml", help="Path to YAML config")
    ap.add_argument("--labels_file", help="Optional YAML containing train_labels allow-list")
    ap.add_argument("--train_dir", help="Override the raw data source directory")
    ap.add_argument(
        "--no_enforce_one_level",
        action="store_true",
        help="Allow nested subfolders (root/label/subdir/*.wav). Default enforces root/label/*.wav only.",
    )
    args = ap.parse_args()

    cfg = load_yaml(Path(args.config))
    audio_cfg = cfg["audio"]
    path_cfg = cfg["paths"]

    raw_dir = Path(args.train_dir or path_cfg["train_dir"])
    cache_root = Path(path_cfg["cache_root"])

    if "manifest_path" in path_cfg:
        manifest_path = path_cfg["manifest_path"]
    elif "train_manifest" in path_cfg:
        manifest_path = path_cfg["train_manifest"]
    else:
        raise KeyError(
            "Missing manifest path in config. Expected 'manifest_path' or 'train_manifest' under 'paths'."
        )

    out_csv = Path(manifest_path)

    # Load allowed labels
    allowed_labels = get_train_labels(cfg)
    if args.labels_file:
        labels_cfg = load_yaml(Path(args.labels_file))
        allowed_labels = get_train_labels(labels_cfg)

    cache_root.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    print("--- Processing Audio ---")
    print(f"Source: {raw_dir}")
    print(f"Cache:  {cache_root}")
    print(f"Params: SR={audio_cfg['sr']}, Mels={audio_cfg['n_mels']}, Dur={audio_cfg['duration']}s")
    if allowed_labels is None:
        print("Labels: (no train_labels provided) -> processing ALL labels discovered on disk")
    else:
        print(f"Labels: {sorted(allowed_labels)}")

    rows_out = []
    n_ok, n_fail = 0, 0

    enforce_one_level = not args.no_enforce_one_level

    pairs = list(_iter_wavs_from_train_dir(raw_dir, allowed_labels, enforce_one_level=enforce_one_level))

    # Report missing/extra labels (helpful sanity check)
    if allowed_labels is not None:
        disk_labels = set()
        root_resolved = raw_dir.resolve()
        for wav in raw_dir.rglob("*.wav"):
            if not wav.is_file():
                continue
            label_dir = wav.parent
            if enforce_one_level and label_dir.parent.resolve() != root_resolved:
                continue
            disk_labels.add(label_dir.name.strip().lower())

        missing = sorted(allowed_labels - disk_labels)
        extra = sorted(disk_labels - allowed_labels)
        if missing:
            print(f"WARNING: These train_labels were not found on disk: {missing}")
        if extra:
            print(f"INFO: These labels exist on disk but will be skipped: {extra}")

    for wav_path, label in tqdm(pairs, desc="Generating Mels"):
        try:
            npy_path = precache_one(
                wav_path,
                label,
                cache_root,
                sr=audio_cfg["sr"],
                dur=audio_cfg["duration"],
                n_mels=audio_cfg["n_mels"],
                win_ms=audio_cfg["win_ms"],
                hop_ms=audio_cfg["hop_ms"],
                fmin=audio_cfg["fmin"],
                fmax=audio_cfg.get("fmax"),
            )

            rel_path = npy_path.resolve().as_posix()
            rows_out.append([rel_path, label, wav_path.resolve().as_posix()])
            n_ok += 1

        except KeyboardInterrupt:
            print("\nStopped by user.")
            break
        except Exception as e:
            print("INFO: Failed to process:", wav_path, "Error:", e)
            n_fail += 1

    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filepath", "label", "wavpath"])
        w.writerows(rows_out)

    print("\nProcessing Complete.")
    print(f"Successfully processed: {n_ok}")
    print(f"Failed: {n_fail}")
    print(f"Manifest written to: {out_csv}")


if __name__ == "__main__":
    main()