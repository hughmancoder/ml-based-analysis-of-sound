from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Iterable


def read_labels(path: Path, *, dedupe_per_file: bool) -> list[str]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    labels = [line.strip() for line in lines if line.strip()]
    if dedupe_per_file:
        labels = list(dict.fromkeys(labels))
    return labels


def collect_labels(
    root: Path,
    *,
    pattern: str,
    dedupe_per_file: bool,
) -> tuple[Counter[str], int, list[Path], int]:
    counts: Counter[str] = Counter()
    total_files = 0
    empty_files: list[Path] = []  # Changed from int to list of Paths
    total_labels = 0

    for txt_path in sorted(p for p in root.rglob(pattern) if p.is_file()):
        total_files += 1
        labels = read_labels(txt_path, dedupe_per_file=dedupe_per_file)
        if not labels:
            empty_files.append(txt_path)
            continue
        counts.update(labels)
        total_labels += len(labels)

    return counts, total_files, empty_files, total_labels


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Summarise label counts from .txt label files.",
    )
    ap.add_argument(
        "--root",
        type=Path,
        default=Path("data/test/a-touch-of-zen"),
        help="Root directory containing label files (default: data/test/a-touch-of-zen)",
    )
    ap.add_argument(
        "--pattern",
        default="*.txt",
        help="Glob pattern for label files (default: *.txt)",
    )
    ap.add_argument(
        "--dedupe-per-file",
        action="store_true",
        help="Count each label at most once per file.",
    )
    ap.add_argument(
        "--sort",
        choices=("count", "label"),
        default="count",
        help="Sort output by label name or count (default: count).",
    )
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    root = args.root
    if not root.is_absolute():
        root = (repo_root / root).resolve()

    if not root.exists():
        print(f"[warn] root does not exist: {root}")
        return

    counts, total_files, empty_files, total_labels = collect_labels(
        root,
        pattern=args.pattern,
        dedupe_per_file=args.dedupe_per_file,
    )

    display_root = root
    try:
        display_root = root.relative_to(repo_root)
    except ValueError:
        pass

    print(f"Root: {display_root}")
    print(f"Label files: {total_files}")
    print(f"Files with no labels: {len(empty_files)}")
    
    # New section: List the empty files
    if empty_files:
        print("\nEmpty files found:")
        for p in empty_files:
            try:
                print(f"  - {p.relative_to(root)}")
            except ValueError:
                print(f"  - {p}")

    print(f"\nTotal label entries: {total_labels}")
    print(f"Unique labels: {len(counts)}")

    if not counts:
        return

    if args.sort == "count":
        items = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    else:
        items = sorted(counts.items(), key=lambda item: item[0])

    width = max(len(label) for label in counts)
    print("\nPer-label counts:\n")
    for label, count in items:
        pct = (count / total_labels) * 100 if total_labels > 0 else 0
        print(f"  {label.ljust(width)}  {str(count).ljust(3)}  ({pct:.1f}%)")


if __name__ == "__main__":
    main()