# #!/usr/bin/env python3
# """
# Merge IRMAS and Chinese instrument manifests into a unified mel manifest.

# The script expects that each input manifest already contains a `mel_filepath`
# column (as produced by `generate_irmas_train_mels.py`). It concatenates rows,
# filters to the canonical `ALL_CLASSES` list, and writes the merged CSV.
# """
# from __future__ import annotations

# import argparse
# from pathlib import Path

# import pandas as pd

# from development.classes import ALL_CLASSES, IRMAS_CLASS_MAP
# from src.datasets import load_manifest


# def main() -> None:
#     parser = argparse.ArgumentParser(
#         description="Combine IRMAS and Chinese mel manifests into a single CSV."
#     )
#     parser.add_argument("--irmas_manifest", required=True, type=str, help="Path to IRMAS manifest CSV.")
#     parser.add_argument("--chinese_manifest", required=True, type=str, help="Path to Chinese manifest CSV.")
#     parser.add_argument("--output_csv", required=True, type=str, help="Destination for merged manifest.")
#     args = parser.parse_args()

#     irmas_path = Path(args.irmas_manifest)
#     chinese_path = Path(args.chinese_manifest)
#     out_path = Path(args.output_csv)

#     irmas_df = load_manifest(irmas_path)
#     irmas_df["labels"] = irmas_df["labels"].map(
#         lambda label: IRMAS_CLASS_MAP.get(label, label)
#     )
#     chinese_df = load_manifest(chinese_path)

#     combined = pd.concat([irmas_df, chinese_df], ignore_index=True)
#     allowed = {label.strip().lower() for label in ALL_CLASSES}
#     combined = combined[combined["labels"].isin(allowed)].copy()

#     # Drop duplicate audio references, keeping the first occurrence.
#     combined = combined.drop_duplicates(subset=["filepath"]).reset_index(drop=True)

#     out_path.parent.mkdir(parents=True, exist_ok=True)
#     combined.to_csv(out_path, index=False)

#     print(f"Merged manifest written to {out_path.resolve()} ({len(combined)} rows)")


# if __name__ == "__main__":
#     main()
