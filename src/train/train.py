""" Usage
python src/train/train.py \
  --run_name CNN_v2 \
  --manifests ../../data/processed/train_mels.csv \
  --labels_yaml src/configs/labels.yaml \
  --audio_yaml src/configs/audio_params.yaml \
  --resume
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import yaml

from utils import multi_label_train_loop

def find_repo_root() -> Path:
    root = Path.cwd().resolve()
    while root != root.parent and not (root / "src").exists():
        root = root.parent
    return root

def load_yaml(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_name", default="CNN_v1")
    ap.add_argument("--weights_dir", default=None, help="Override weights dir; default uses ../models/saved_weights/<run_name>")
    ap.add_argument("--manifests", nargs="+", required=True, help="One or more manifest CSVs")
    ap.add_argument("--labels_yaml", required=True)
    ap.add_argument("--audio_yaml", required=True)

    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--num_workers", type=int, default=2)

    ap.add_argument("--resume", action="store_true", help="Resume from last.pt if present")
    ap.add_argument("--warm_start", action="store_true", help="Load weights only (do not restore optimiser/scheduler)")
    args = ap.parse_args()

    repo_root = find_repo_root()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    print("Repo root:", repo_root)

    labels_cfg = load_yaml(Path(args.labels_yaml))
    classes = [c.strip().lower() for c in labels_cfg.get("train_labels", [])]
    if not classes:
        raise ValueError(f"No train_labels found in {args.labels_yaml}")
    print(f"Loaded {len(classes)} classes: {', '.join(classes)}")

    audio_cfg_all = load_yaml(Path(args.audio_yaml))
    audio_cfg = audio_cfg_all.get("audio", audio_cfg_all)

    if args.weights_dir:
        ckpt_dir = Path(args.weights_dir)
    else:
        ckpt_dir = (repo_root / "models" / "saved_weights" / args.run_name).resolve()

    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Save run config for reproducibility
    run_cfg = {
        "manifests": args.manifests,
        "labels_yaml": args.labels_yaml,
        "audio_yaml": args.audio_yaml,
        "classes": classes,
        "train": {
            "batch_size": args.batch_size,
            "lr": args.lr,
            "epochs": args.epochs,
            "patience": args.patience,
            "weight_decay": args.weight_decay,
            "dropout": args.dropout,
            "val_frac": args.val_frac,
            "seed": args.seed,
            "threshold": args.threshold,
            "num_workers": args.num_workers,
        },
        "resume": args.resume,
        "warm_start": args.warm_start,
    }
    with open(ckpt_dir / "run_config.yaml", "w") as f:
        yaml.safe_dump(run_cfg, f, sort_keys=False)

    resume_ckpt = ckpt_dir / "last.pt"
    if not args.resume or not resume_ckpt.exists():
        resume_ckpt = None
        print("Starting fresh (no resume).")
    else:
        print(f"Resuming from: {resume_ckpt}")

    results = multi_label_train_loop(
        manifest_csv=args.manifests,
        classes=classes,
        ckpt_dir=ckpt_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        val_frac=args.val_frac,
        dropout=args.dropout,
        patience=args.patience,
        num_workers=args.num_workers,
        threshold=args.threshold,
        seed=args.seed,
        audio_cfg=audio_cfg,
        resume_from=resume_ckpt,
        save_best_stamped=False,
        # you'll add warm_start support inside multi_label_train_loop
    )

    _ = results["history"]
    print("Training complete.")

if __name__ == "__main__":
    main()