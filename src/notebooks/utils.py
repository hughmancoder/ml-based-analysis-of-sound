from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Mapping, Optional, Sequence

import math
import random

import matplotlib.pyplot as plt
import numpy as np


def find_repo_root(start: Optional[Path] = None) -> Path:
    root = Path.cwd() if start is None else Path(start).resolve()
    while True:
        if (root / "data").exists() or (root / "pyproject.toml").exists():
            return root
        if root == root.parent:
            return root
        root = root.parent


def load_npy(path: str | Path) -> np.ndarray:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"mel file not found: {path}")
    return np.load(path)


def db_to_uint8(img_db: np.ndarray, db_min: float, db_max: float) -> np.ndarray:
    x = np.clip(img_db, db_min, db_max)
    x = (x - db_min) / (db_max - db_min + 1e-12)
    return (x * 255.0).round().astype(np.uint8)


def make_display_image(
    mel_db: np.ndarray,
    *,
    tile: str = "v",
    db_min: float = -80.0,
    db_max: float = 0.0,
) -> np.ndarray:
    if mel_db.ndim == 2:
        return db_to_uint8(mel_db, db_min, db_max)
    if mel_db.ndim != 3:
        raise ValueError(f"Expected mel shape (C, F, T) or (F, T); got {mel_db.shape}")
    if mel_db.shape[0] == 1:
        return db_to_uint8(mel_db[0], db_min, db_max)
    if mel_db.shape[0] != 2:
        raise ValueError(f"Expected channel dimension 1 or 2; got {mel_db.shape}")

    left_u8 = db_to_uint8(mel_db[0], db_min, db_max)
    right_u8 = db_to_uint8(mel_db[1], db_min, db_max)
    if tile.lower().startswith("h"):
        return np.concatenate([left_u8, right_u8], axis=1)
    return np.concatenate([left_u8, right_u8], axis=0)


def plot_mel_npy(
    path: str | Path,
    *,
    tile: str = "v",
    db_min: float = -80.0,
    db_max: float = 0.0,
    title: Optional[str] = None,
    single_channel: bool = False,
) -> tuple[plt.Figure, plt.Axes]:
    mel = load_npy(path)
    if single_channel and mel.ndim == 3 and mel.shape[0] > 1:
        mel = mel.mean(axis=0, keepdims=True)
    display = make_display_image(mel, tile=tile, db_min=db_min, db_max=db_max)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(display, aspect="auto", origin="lower")
    ax.set_title(title or f"Mel spectrogram: {Path(path).name}")
    ax.set_xlabel("Frames")
    ax.set_ylabel("Mel bins (stacked L/R)")
    fig.tight_layout()
    plt.show()
    return fig, ax


def list_label_dirs(root: str | Path) -> list[str]:
    root = Path(root)
    if not root.exists():
        return []
    labels: list[str] = []
    for label_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        if any(label_dir.rglob("*.npy")):
            labels.append(label_dir.name)
    return labels


def _collect_mel_paths(root: Path, labels: Optional[Sequence[str]]) -> list[Path]:
    if not root.exists():
        return []
    if labels:
        dirs = [root / label for label in labels]
    else:
        dirs = [p for p in root.iterdir() if p.is_dir()]

    paths: list[Path] = []
    for label_dir in dirs:
        if not label_dir.exists():
            continue
        paths.extend(sorted(label_dir.rglob("*.npy")))
    return paths


def sample_mel_paths(
    root: str | Path,
    *,
    n: int = 5,
    seed: int = 1337,
    labels: Optional[Sequence[str]] = None,
) -> list[Path]:
    root = Path(root)
    paths = _collect_mel_paths(root, labels)
    if not paths:
        return []
    if n >= len(paths):
        return paths
    rng = random.Random(seed)
    return rng.sample(paths, k=n)


def sample_mel_paths_by_label(
    root: str | Path,
    *,
    labels: Optional[Sequence[str]] = None,
    seed: int = 1337,
) -> OrderedDict[str, Path]:
    root = Path(root)
    rng = random.Random(seed)
    if labels is None:
        labels = list_label_dirs(root)

    picked: OrderedDict[str, Path] = OrderedDict()
    for label in labels:
        label_dir = root / label
        if not label_dir.exists():
            continue
        files = sorted(label_dir.rglob("*.npy"))
        if not files:
            continue
        picked[label] = rng.choice(files)
    return picked


def plot_mel_grid(
    samples: Mapping[str, Path],
    *,
    tile: str = "v",
    db_min: float = -80.0,
    db_max: float = 0.0,
    cols: int = 3,
    show_colorbar: bool = False,
) -> None:
    if not samples:
        print("No mel samples available for plotting.")
        return

    items = list(samples.items())
    n_items = len(items)
    cols = max(1, min(cols, n_items))
    rows = math.ceil(n_items / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 3.5 * rows), squeeze=False)
    for idx, (label, path) in enumerate(items):
        r, c = divmod(idx, cols)
        ax = axes[r, c]
        mel = load_npy(path)
        display = make_display_image(mel, tile=tile, db_min=db_min, db_max=db_max)
        im = ax.imshow(display, aspect="auto", origin="lower")
        ax.set_title(label, fontsize=11)
        ax.set_xlabel("Frames")
        ax.set_ylabel("Mel bins")
        if show_colorbar:
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for idx in range(n_items, rows * cols):
        r, c = divmod(idx, cols)
        axes[r, c].axis("off")

    fig.tight_layout()
    plt.show()
