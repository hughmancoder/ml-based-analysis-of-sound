"""
Usage
python src/test/test_irmas_mel_cqt.py \
  --weights_run IRMAS_mel_cqt_v1 \
  --test_manifest data/test/IRMAS/IRMAS-TestingData-Part1.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from src.models.CNN import CNN
from src.test.utils import find_best_threshold, evaluate_multilabel_performance
from src.test.utils_mel_cqt import run_inference


def find_repo_root() -> Path:
    root = Path.cwd().resolve()
    while root != root.parent and not (root / "src").exists():
        root = root.parent
    return root


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights_run", default="IRMAS_mel_cqt_v1")
    ap.add_argument("--weights_path", default=None, help="Override exact model weights path")
    ap.add_argument("--test_manifest", default="data/test/IRMAS/IRMAS-TestingData-Part1.csv")
    ap.add_argument("--threshold", type=float, default=None, help="If omitted, auto-tunes best threshold")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--no_progress", action="store_true")
    ap.add_argument("--cqt_bins", type=int, default=None)
    ap.add_argument("--bins_per_octave", type=int, default=12)
    args = ap.parse_args()

    repo_root = find_repo_root()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    print("Repo root:", repo_root)

    if args.weights_path:
        weights_path = Path(args.weights_path)
        if not weights_path.is_absolute():
            weights_path = (repo_root / weights_path).resolve()
    else:
        weights_dir = (repo_root / "src" / "models" / "saved_weights" / args.weights_run).resolve()
        weights_path = weights_dir / "best_val.pt"
        if not weights_path.exists():
            weights_path = weights_dir / "last.pt"

    if not weights_path.exists():
        raise FileNotFoundError(f"Model weights not found: {weights_path}")

    test_manifest = Path(args.test_manifest)
    if not test_manifest.is_absolute():
        test_manifest = (repo_root / test_manifest).resolve()
    if not test_manifest.exists():
        raise FileNotFoundError(f"Test manifest not found: {test_manifest}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    preds_arr, gts_arr, sample_ids, audio_cfg, valid_labels, label_to_idx = run_inference(
        model_cls=CNN,
        model_kwargs={"in_ch": 4},
        model_weights_path=weights_path,
        device=device,
        test_manifest_csv=test_manifest,
        root=repo_root,
        show_progress=not args.no_progress,
        cqt_bins=args.cqt_bins,
        bins_per_octave=args.bins_per_octave,
    )

    if args.threshold is None:
        best_t = find_best_threshold(preds_arr, gts_arr, valid_labels)
        print(f"Best threshold found: {best_t:.2f}")
        threshold = best_t
    else:
        threshold = args.threshold

    _ = evaluate_multilabel_performance(
        all_preds=preds_arr,
        all_gt=gts_arr,
        class_list=valid_labels,
        sample_ids=sample_ids,
        threshold=threshold,
        debug=args.debug,
    )


if __name__ == "__main__":
    main()
