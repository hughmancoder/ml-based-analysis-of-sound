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
    if h:
        parts.append(f"{h}h")
    if m or (h and s):
        parts.append(f"{m}m")
    if s and not h:
        parts.append(f"{s}s")
    return " ".join(parts) if parts else "0s"


def count_wavs(root: Path) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    if not root.exists():
        print(f"[warn] root does not exist: {root}")
        return counts

    # labels are immediate subdirectories
    label_dirs = [p for p in root.iterdir() if p.is_dir()]

    # Sort label dirs by folder name (case-insensitive)
    for label_dir in sorted(label_dirs, key=lambda p: p.name.lower()):
        label = label_dir.name.strip().lower()  # normalise label names
        n = 0
        for _wav in label_dir.rglob("*.wav"):
            n += 1
        counts[label] = counts.get(label, 0) + n  # merge if duplicates by case
    return counts


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarise dataset.")
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

    print("\nPer-label clip counts:\n")
    width = max(len(k) for k in counts.keys())
    
    # Sort by count descending
    sorted_items = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)

    for label, n in sorted_items:
        # Calculate percentage
        pct = (n / total) * 100 if total > 0 else 0
        dur = human_time(n * CLIP_SECONDS)
        
        # Display label, count, percentage, and duration
        print(f"  {label.ljust(width)}  {str(n)}  ({pct:.1f}%)  ({dur})")

    print(f"\nTotal clip counts: {total}")


if __name__ == "__main__":
    main()