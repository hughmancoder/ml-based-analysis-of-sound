import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional
from torch.utils.data import Dataset


def _read_csv_with_fallback(path: Path) -> pd.DataFrame:
    encodings = ("utf-8", "utf-8-sig", "gbk", "cp936")
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path)


class MultiLabelMelCqtDataset(Dataset):
    def __init__(
        self,
        manifest_csv,
        class_names: list,
        project_root: str = ".",
        transform=None,
        max_zero_label_warnings: int = 10,
        infer_label_from_parent: bool = True,
    ):
        if isinstance(manifest_csv, (list, tuple, set)):
            if not manifest_csv:
                raise ValueError("manifest_csv list is empty")
            frames = [_read_csv_with_fallback(Path(path)) for path in manifest_csv]
            df = pd.concat(frames, ignore_index=True)
        else:
            df = _read_csv_with_fallback(Path(manifest_csv))

        if "filepath" not in df.columns:
            raise ValueError("Manifest must contain a 'filepath' column for mel paths")
        if "cqt_path" not in df.columns:
            raise ValueError("Manifest must contain a 'cqt_path' column for CQT paths")

        # ---- Canonicalise label column ----
        if "labels" not in df.columns and "label" not in df.columns:
            raise ValueError("Manifest must contain a 'label' or 'labels' column")

        labels_series = None
        if "labels" in df.columns:
            labels_series = df["labels"]
        if "label" in df.columns:
            labels_series = labels_series.combine_first(df["label"]) if labels_series is not None else df["label"]

        df["labels_raw"] = labels_series.fillna("").astype(str).str.lower().str.replace(" ", "", regex=False)
        df["labels_raw"] = df["labels_raw"].str.replace(",", "|", regex=False)

        self.df = df[["filepath", "cqt_path", "labels_raw"]].copy()

        self.root = Path(project_root)
        self.class_names = [c.strip().lower() for c in class_names]
        self.label_to_idx = {name: i for i, name in enumerate(self.class_names)}
        self.num_classes = len(self.class_names)

        self.transform = transform
        self.infer_label_from_parent = infer_label_from_parent

        self._zero_label_count = 0
        self._seen = 0
        self._max_zero_label_warnings = max_zero_label_warnings

    def __len__(self):
        return len(self.df)

    def _load_npy(self, path_str: str) -> np.ndarray:
        npy_path = Path(path_str) if Path(path_str).is_absolute() else self.root / path_str
        return np.load(npy_path)

    def __getitem__(self, idx):
        self._seen += 1
        row = self.df.iloc[idx]

        mel = self._load_npy(str(row["filepath"]))
        cqt = self._load_npy(str(row["cqt_path"]))

        mel_tensor = torch.from_numpy(mel).float()
        cqt_tensor = torch.from_numpy(cqt).float()

        # Align freq/time dimensions (crop to common size)
        min_h = min(mel_tensor.shape[1], cqt_tensor.shape[1])
        min_w = min(mel_tensor.shape[2], cqt_tensor.shape[2])
        mel_tensor = mel_tensor[:, :min_h, :min_w]
        cqt_tensor = cqt_tensor[:, :min_h, :min_w]

        # Per-example Z-score
        mel_tensor = (mel_tensor - mel_tensor.mean()) / (mel_tensor.std() + 1e-6)
        cqt_tensor = (cqt_tensor - cqt_tensor.mean()) / (cqt_tensor.std() + 1e-6)

        if self.transform is not None:
            mel_tensor = self.transform(mel_tensor)
            cqt_tensor = self.transform(cqt_tensor)

        x = torch.cat([mel_tensor, cqt_tensor], dim=0)

        # ---- Parse labels ----
        raw = row["labels_raw"]
        label_list = [x for x in raw.split("|") if x]

        if not label_list and self.infer_label_from_parent:
            inferred = Path(str(row["filepath"])).parent.name.strip().lower()
            if inferred in self.label_to_idx:
                label_list = [inferred]

        target = torch.zeros(self.num_classes, dtype=torch.float32)
        for label in label_list:
            j = self.label_to_idx.get(label)
            if j is not None:
                target[j] = 1.0

        if target.sum().item() == 0:
            self._zero_label_count += 1
            if self._zero_label_count <= self._max_zero_label_warnings:
                print(
                    "[WARN] All-zero target produced\n"
                    f"  idx={idx}\n"
                    f"  mel={row['filepath']}\n"
                    f"  cqt={row['cqt_path']}\n"
                    f"  labels_raw='{row['labels_raw']}'\n"
                )

        return x, target

    def zero_label_rate(self) -> float:
        if self._seen == 0:
            return 0.0
        return self._zero_label_count / self._seen
