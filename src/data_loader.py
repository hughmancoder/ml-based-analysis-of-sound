"""
PyTorch Dataset for loading pre-computed Mel spectrograms
"""
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset
import yaml

class MultiLabelMelDataset(Dataset):
    def __init__(self, manifest_csv: str, class_names: list, project_root: str = ".", transform=None):
        """
        Args:
            manifest_csv: Path to the manifest file.
            class_names: A list of strings (loaded from your YAML in the main script).
            project_root: Root directory to resolve relative paths.
        """
        self.df = pd.read_csv(manifest_csv)
        self.root = Path(project_root)
        
        # We use the list passed from the training loop
        self.class_names = [c.strip().lower() for c in class_names]
        self.label_to_idx = {name: i for i, name in enumerate(self.class_names)}
        self.num_classes = len(self.class_names)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # 1. Load Mel Spectrogram
        path_str = str(row['filepath'])
        npy_path = Path(path_str) if Path(path_str).is_absolute() else self.root / path_str
        
        # Load and convert to Tensor
        mel = np.load(npy_path) 
        mel_tensor = torch.from_numpy(mel).float()

        # Per-Example Normalisation (Z-score)
        # Standardising inputs helps the CNN converge significantly faster
        mean = mel_tensor.mean()
        std = mel_tensor.std() + 1e-6
        mel_tensor = (mel_tensor - mean) / std

        # 2. Process Multi-Label
        # We handle multi-label strings like "pipa,erhu"
        label_list = str(row['label']).lower().replace(" ", "").split(',')
        
        # Create multi-hot target vector
        target = torch.zeros(self.num_classes, dtype=torch.float32)
        for label in label_list:
            if label in self.label_to_idx:
                target[self.label_to_idx[label]] = 1.0

        return mel_tensor, target