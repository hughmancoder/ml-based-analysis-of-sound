import torch
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset

class MultiLabelMelDataset(Dataset):
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
            frames = [pd.read_csv(path) for path in manifest_csv]
            df = pd.concat(frames, ignore_index=True)
        else:
            df = pd.read_csv(manifest_csv)

        # ---- Canonicalise label column ----
        # Prefer 'labels' (multilabel) but fall back to 'label' (single-label)
        # This is row-wise: handles concatenated manifests cleanly.
        if "labels" not in df.columns and "label" not in df.columns:
            raise ValueError("Manifest must contain a 'label' or 'labels' column")

        labels_series = None
        if "labels" in df.columns:
            labels_series = df["labels"]
        if "label" in df.columns:
            labels_series = labels_series.combine_first(df["label"]) if labels_series is not None else df["label"]

        # Normalise to string, strip spaces; keep NaN as empty string
        df["labels_raw"] = labels_series.fillna("").astype(str).str.lower().str.replace(" ", "", regex=False)

        # Standardise separator: allow either comma or | in the manifests
        # Convert commas to pipes so parsing is consistent.
        df["labels_raw"] = df["labels_raw"].str.replace(",", "|", regex=False)

        # Keep only what we need
        self.df = df[["filepath", "labels_raw"]].copy()

        self.root = Path(project_root)
        self.class_names = [c.strip().lower() for c in class_names]
        self.label_to_idx = {name: i for i, name in enumerate(self.class_names)}
        self.num_classes = len(self.class_names)

        self.transform = transform
        self.infer_label_from_parent = infer_label_from_parent

        # ---- Safety tracking ----
        self._zero_label_count = 0
        self._seen = 0
        self._max_zero_label_warnings = max_zero_label_warnings

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        self._seen += 1
        row = self.df.iloc[idx]

        # ---- Load mel ----
        path_str = str(row["filepath"])
        npy_path = Path(path_str) if Path(path_str).is_absolute() else self.root / path_str

        mel = np.load(npy_path)
        mel_tensor = torch.from_numpy(mel).float()

        # Per-example Z-score
        mean = mel_tensor.mean()
        std = mel_tensor.std() + 1e-6
        mel_tensor = (mel_tensor - mean) / std

        if self.transform is not None:
            mel_tensor = self.transform(mel_tensor)

        # ---- Parse labels ----
        raw = row["labels_raw"]
        label_list = [x for x in raw.split("|") if x]  # canonical separator

        # Optional fallback: infer single label from parent dir if label is missing
        if not label_list and self.infer_label_from_parent:
            inferred = npy_path.parent.name.strip().lower()
            if inferred in self.label_to_idx:
                label_list = [inferred]

        target = torch.zeros(self.num_classes, dtype=torch.float32)
        for label in label_list:
            j = self.label_to_idx.get(label)
            if j is not None:
                target[j] = 1.0

        # ---- Safety check ----
        if target.sum().item() == 0:
            self._zero_label_count += 1
            if self._zero_label_count <= self._max_zero_label_warnings:
                print(
                    "[WARN] All-zero target produced\n"
                    f"  idx={idx}\n"
                    f"  file={npy_path}\n"
                    f"  labels_raw='{row['labels_raw']}'\n"
                )

        return mel_tensor, target

    def zero_label_rate(self) -> float:
        if self._seen == 0:
            return 0.0
        return self._zero_label_count / self._seen