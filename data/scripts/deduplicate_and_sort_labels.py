#!/usr/bin/env python3
"""
python dedupe_and_sort.py labels.json labels.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Iterable


def load_all_json_arrays(path: Path) -> list[dict[str, Any]]:
    """
    Load one or more top-level JSON arrays from a file.
    Handles the common mistake of concatenated arrays.
    """
    text = path.read_text(encoding="utf-8").strip()
    decoder = json.JSONDecoder()
    idx = 0
    items: list[dict[str, Any]] = []

    while idx < len(text):
        obj, end = decoder.raw_decode(text[idx:])
        if isinstance(obj, list):
            items.extend(obj)
        idx += end
        # skip whitespace between concatenated JSON values
        while idx < len(text) and text[idx].isspace():
            idx += 1

    return items


def basename(file_path: str) -> str:
    return Path(file_path).name


def dedupe_by_basename(
    items: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []

    for item in items:
        file = item.get("file")
        if not isinstance(file, str):
            continue

        key = basename(file)
        if key in seen:
            continue

        seen.add(key)
        out.append(item)

    return out


def sort_by_label_then_file(
    items: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    def key(it: dict[str, Any]):
        labels = it.get("labels") or []
        first_label = labels[0] if labels else ""
        return (first_label, it.get("file", ""))

    return sorted(items, key=key)


def main() -> None:
    if len(sys.argv) not in (2, 3):
        print("Usage: dedupe_and_sort.py input.json [output.json]", file=sys.stderr)
        sys.exit(1)

    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2]) if len(sys.argv) == 3 else None

    items = load_all_json_arrays(input_path)
    items = dedupe_by_basename(items)
    items = sort_by_label_then_file(items)

    output = json.dumps(items, indent=2, ensure_ascii=False)

    if output_path:
        output_path.write_text(output + "\n", encoding="utf-8")
    else:
        print(output)


if __name__ == "__main__":
    main()

