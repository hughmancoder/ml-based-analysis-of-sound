import os
import sys
import time
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, random_split

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data_loader import MultiLabelMelDataset

# --- 1. Core Utilities ---

def seed_everything(seed: int = 1337):
    """Sets seeds for reproducibility across numpy, random, and torch."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device():
    """Detects the best available device (CUDA, MPS, or CPU)."""
    use_cuda_amp = False
    use_mps_amp = False
    scaler = torch.cuda.amp.GradScaler(enabled=False)
    pin_mem = False

    if torch.cuda.is_available():
        device = "cuda"
        use_cuda_amp = True
        scaler = torch.cuda.amp.GradScaler(enabled=True)
        pin_mem = True
    elif torch.backends.mps.is_available():
        device = "mps"
        use_mps_amp = True 
    else:
        device = "cpu"

    return device, use_cuda_amp, use_mps_amp, scaler, pin_mem

def build_model(num_classes: int, dropout: float, in_ch: int = 2, device: str = "cpu"):
    """Instantiates the CNNVarTime model and moves it to the target device."""
    # Importing inside function to avoid circular imports if CNNVarTime is in a separate file
    from src.models.CNN import CNN 
    model = CNN(in_ch=in_ch, num_classes=num_classes, p_drop=dropout)
    return model.to(device)

# --- 2. Checkpointing & Data Handling ---
def save_checkpoint(payload: Dict[str, Any], filepath: Path):
    """Saves the training state to a file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, filepath)

def load_checkpoint(
    path: Path, 
    device: str, 
    model: nn.Module, 
    optimizer: torch.optim.Optimizer, 
    scheduler: Optional[Any] = None, 
    scaler: Optional[Any] = None,
    current_label_to_idx: Optional[Dict[str, int]] = None
):
    """Loads a saved checkpoint and restores the model and optimiser states."""
    ckpt = torch.load(path, map_location=device)
    
    if current_label_to_idx and "label_to_idx" in ckpt:
        if len(current_label_to_idx) != len(ckpt["label_to_idx"]):
            print("Warning: Number of classes in checkpoint doesn't match current config!")

    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["opt_state"])
    
    if scheduler and "sched_state" in ckpt:
        scheduler.load_state_dict(ckpt["sched_state"])
    
    if scaler and ckpt.get("scaler_state") and scaler.is_enabled():
        scaler.load_state_dict(ckpt["scaler_state"])
        
    return ckpt

def collate_fn_padd(batch):
    """Pads variable length mel spectrograms to the maximum width in the batch."""
    tensors = [item[0] for item in batch]
    targets = torch.stack([item[1] for item in batch])
    
    # Pad the width (dim 2) -> Shape: (2, 128, W)
    tensors = [t.permute(2, 0, 1) for t in tensors] # Time to dim 0
    tensors_padded = torch.nn.utils.rnn.pad_sequence(tensors, batch_first=True)
    tensors_padded = tensors_padded.permute(0, 2, 3, 1) # Back to (B, C, H, W)
    
    return tensors_padded, targets

# --- 3. Multi-Label Training Logic ---

def compute_f1_metrics(prob_chunks: List[torch.Tensor], target_chunks: List[torch.Tensor], threshold: float) -> Dict[str, float]:
    """Calculates multi-label metrics using sklearn."""
    if not prob_chunks:
        return {"micro_f1": 0.0, "macro_f1": 0.0, "exact_match": 0.0}
    
    y_true = torch.cat(target_chunks).cpu().numpy()
    y_pred = torch.cat(prob_chunks).cpu().numpy()
    y_bin = (y_pred >= threshold).astype(np.int32)
    
    micro = f1_score(y_true, y_bin, average="micro", zero_division=0)
    macro = f1_score(y_true, y_bin, average="macro", zero_division=0)
    exact = (y_bin == y_true).all(axis=1).mean()
    
    return {"micro_f1": float(micro), "macro_f1": float(macro), "exact_match": float(exact)}

def train_one_epoch_multi(model, loader, criterion, optimizer, scaler, device, use_cuda_amp, use_mps_amp, pin_mem, threshold):
    model.train()
    loss_sum, total = 0.0, 0
    prob_chunks, target_chunks = [], []

    for X, y in loader:
        X, y = X.to(device, non_blocking=pin_mem), y.to(device, non_blocking=pin_mem)
        optimizer.zero_grad(set_to_none=True)

        # Handle Mixed Precision
        context = torch.cuda.amp.autocast() if use_cuda_amp else \
                  (torch.autocast(device_type="mps", dtype=torch.float16) if use_mps_amp else torch.enable_grad())
        
        with context:
            logits = model(X)
            loss = criterion(logits, y)

        if use_cuda_amp:
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        prob_chunks.append(torch.sigmoid(logits.detach()).cpu())
        target_chunks.append(y.detach().cpu())
        loss_sum += float(loss.item()) * y.size(0)
        total += int(y.size(0))

    metrics = compute_f1_metrics(prob_chunks, target_chunks, threshold)
    return loss_sum / max(1, total), metrics

def evaluate_multi(model, loader, criterion, device, use_cuda_amp, use_mps_amp, pin_mem, threshold):
    model.eval()
    loss_sum, total = 0.0, 0
    prob_chunks, target_chunks = [], []

    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device, non_blocking=pin_mem), y.to(device, non_blocking=pin_mem)
            
            context = torch.cuda.amp.autocast() if use_cuda_amp else \
                      (torch.autocast(device_type="mps", dtype=torch.float16) if use_mps_amp else torch.no_grad())
            
            with context:
                logits = model(X)
                loss = criterion(logits, y)

            prob_chunks.append(torch.sigmoid(logits.detach()).cpu())
            target_chunks.append(y.detach().cpu())
            loss_sum += float(loss.item()) * y.size(0)
            total += int(y.size(0))

    metrics = compute_f1_metrics(prob_chunks, target_chunks, threshold)
    return loss_sum / max(1, total), metrics

def multi_label_train_loop(
    manifest_csv: str,
    classes: Sequence[str],
    ckpt_dir: Path,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    val_frac: float,
    dropout: float,
    patience: int,
    num_workers: int,
    threshold: float,
    seed: int,
    audio_cfg: dict,
    resume_from: Optional[Path] = None,
    save_best_stamped: bool = False,
):

    seed_everything(seed)
    device, use_cuda_amp, use_mps_amp, scaler, pin_mem = get_device()
    
    # Load Dataset
    dataset = MultiLabelMelDataset(
        manifest_csv=manifest_csv,
        class_names=classes,
        project_root=REPO_ROOT,
    )
    val_size = int(round(len(dataset) * val_frac))
    train_ds, val_ds = random_split(dataset, [len(dataset)-val_size, val_size], generator=torch.Generator().manual_seed(seed))
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_mem, collate_fn=collate_fn_padd)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_mem, collate_fn=collate_fn_padd)

    model = build_model(num_classes=len(classes), dropout=dropout, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.BCEWithLogitsLoss() # This treats each instrument as an independent binary classification task.

    history = {k: [] for k in ["train_loss", "val_loss", "train_micro_f1", "val_micro_f1", "train_macro_f1", "val_macro_f1"]}
    start_epoch, best_val_f1, no_improve = 1, 0.0, 0

    if resume_from and resume_from.exists():
        ckpt = load_checkpoint(resume_from, device, model, optimizer, scheduler, scaler)
        start_epoch = ckpt["epoch"] + 1
        history = ckpt["history"]
        best_val_f1 = ckpt.get("best_val_micro_f1", 0.0)

    for epoch in range(start_epoch, epochs + 1):
        t0 = time.time()
        train_loss, train_m = train_one_epoch_multi(model, train_loader, criterion, optimizer, scaler, device, use_cuda_amp, use_mps_amp, pin_mem, threshold)
        val_loss, val_m = evaluate_multi(model, val_loader, criterion, device, use_cuda_amp, use_mps_amp, pin_mem, threshold)
        
        scheduler.step()
        
        # Update History
        history["train_loss"].append(train_loss); history["val_loss"].append(val_loss)
        history["train_micro_f1"].append(train_m["micro_f1"]); history["val_micro_f1"].append(val_m["micro_f1"])
        history["train_macro_f1"].append(train_m["macro_f1"]); history["val_macro_f1"].append(val_m["macro_f1"])

        print(f"[{epoch}/{epochs}] Loss: {train_loss:.4f}/{val_loss:.4f} | Val MicroF1: {val_m['micro_f1']:.4f} | Time: {time.time()-t0:.1f}s")

        payload = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "opt_state": optimizer.state_dict(),
            "history": history,
            "best_val_micro_f1": best_val_f1,
            "classes": classes,            # List of instrument names in order
            "audio_config": audio_cfg,     # sr, n_mels, hop_ms, etc.
            "label_to_idx": dataset.label_to_idx
        }
        save_checkpoint(payload, ckpt_dir / "last.pt")

        if val_m["micro_f1"] > best_val_f1:
            best_val_f1 = val_m["micro_f1"]
            no_improve = 0
            save_checkpoint(payload, ckpt_dir / "best_val.pt")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch}"); break

    return {"model": model, "history": history, "classes": classes, "audio_config": audio_cfg}
    
def plot_metrics(history):
    epochs = range(1, len(history["train_loss"]) + 1)
    
    plt.figure(figsize=(15, 5))

    # Plot Loss
    plt.subplot(1, 3, 1)
    plt.plot(epochs, history["train_loss"], label='Train Loss')
    plt.plot(epochs, history["val_loss"], label='Val Loss')
    plt.title('BCE Loss')
    plt.xlabel('Epochs')
    plt.legend()

    # Plot Micro F1
    plt.subplot(1, 3, 2)
    plt.plot(epochs, history["train_micro_f1"], label='Train Micro F1')
    plt.plot(epochs, history["val_micro_f1"], label='Val Micro F1')
    plt.title('Micro F1 Score')
    plt.xlabel('Epochs')
    plt.legend()

    # Plot Macro F1
    plt.subplot(1, 3, 3)
    plt.plot(epochs, history["train_macro_f1"], label='Train Macro F1')
    plt.plot(epochs, history["val_macro_f1"], label='Val Macro F1')
    plt.title('Macro F1 Score')
    plt.xlabel('Epochs')
    plt.legend()

    plt.tight_layout()
    plt.show()