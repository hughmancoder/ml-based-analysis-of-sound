

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

CLIP_SECONDS = 3.0

def human_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    parts = []
    if h: parts.append(f"{h}h")
    if m or (h and s): parts.append(f"{m}m")
    if s and not h: parts.append(f"{s}s")
    return " ".join(parts) if parts else "0s"

def count_wavs(root: Path) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    if not root.exists():
        print(f"[warn] root does not exist: {root}")
        return counts
    # labels are immediate subdirectories
    for label_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        n = 0
        for wav in label_dir.rglob("*.wav"):
            n += 1
        counts[label_dir.name] = n
    return counts

def main() -> None:
    ap = argparse.ArgumentParser(description="Summarise Chinese instruments dataset.")
    ap.add_argument(
        "--root",
        type=Path,
        default=Path("data/train"),
        help="Root dataset directory (default: data/train)",
    )
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    root = args.root
    if not root.is_absolute():
        root = (repo_root / root).resolve()

    counts = count_wavs(root)
    total = sum(counts.values())
    total_sec = total * CLIP_SECONDS

    display_root = root
    try:
        display_root = root.relative_to(repo_root)
    except ValueError:
        pass
    print(f"Root: {display_root}")
    print(f"Total clips: {total}  (~{human_time(total_sec)})")
    if not counts:
        return
    print("\nPer-label clip counts:")
    width = max(len(k) for k in counts.keys())
    for label in sorted(counts.keys()):
        n = counts[label]
        dur = human_time(n * CLIP_SECONDS)
        print(f"  {label.ljust(width)}  {str(n).rjust(6)}  (~{dur})")

if __name__ == "__main__":
    main()

# python3 data/scripts/summarise_data.py --root data/processed/train
