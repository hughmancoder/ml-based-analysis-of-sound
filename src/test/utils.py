import torch
import pandas as pd
import numpy as np
import re
from pathlib import Path
from sklearn.metrics import classification_report, accuracy_score, hamming_loss
import matplotlib.pyplot as plt
from tqdm import tqdm
from src.preprocessing import calc_fft_hop, ensure_duration, load_audio_stereo, mel_stereo2_from_stereo


def _read_csv_with_fallback(path: Path) -> pd.DataFrame:
    encodings = ("utf-8", "utf-8-sig", "gbk", "cp936")
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path)


def _read_text_with_fallback(path: Path) -> str:
    encodings = ("utf-8", "utf-8-sig", "gbk", "cp936")
    for enc in encodings:
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="ignore")

def parse_ground_truth(txt_path, label_to_idx):
    """
    Parses text files. Handles newlines, tabs, and commas. 
    Only keeps labels present in the training set.
    """
    path = Path(txt_path)
    gt_vector = np.zeros(len(label_to_idx))
    if not path.exists():
        return gt_vector

    content = _read_text_with_fallback(path)
    
    # Regex split handles \n, \t, and commas simultaneously
    raw_labels = re.split(r'[\n,\t]', content)
    
    for label in raw_labels:
        clean = label.strip().lower()
        if clean in label_to_idx:
            gt_vector[label_to_idx[clean]] = 1.0
            
    return gt_vector

def load_and_preprocess(path, cfg):
    """Load audio and convert to 2-channel Log-Mel Spectrogram."""
    p = Path(path)
  
    stereo = load_audio_stereo(p, target_sr=cfg['sr'])
    stereo = ensure_duration(stereo, cfg['sr'], cfg['duration'])
    n_fft, hop, win_length = calc_fft_hop(cfg['sr'], cfg['win_ms'], cfg['hop_ms'])
    
    mel = mel_stereo2_from_stereo(
        stereo, cfg['sr'], n_fft=n_fft, hop=hop, 
        win_length=win_length, n_mels=cfg['n_mels'],
        fmin=cfg['fmin'], fmax=cfg.get('fmax')
    )
    return torch.from_numpy(mel).float()

def get_prediction(model, mel, device):
    """Standardises input and returns prediction vector."""
    model.eval()
    with torch.no_grad():
        # Add batch dimension and move to device
        x = mel.unsqueeze(0).to(device)
        # Match training normalisation
        x = (x - x.mean()) / (x.std() + 1e-6)
        logits = model(x)
        probs = torch.sigmoid(logits)
    return probs.cpu().numpy()[0]

def find_best_threshold(preds_probs, gts, labels, show_plot=False):
    thresholds = np.arange(0.05, 1, 0.05)
    micro_f1s = []
    macro_f1s = []
    
    print(f"{'Threshold':<10} | {'Micro F1':<10} | {'Macro F1':<10} | {'Subset Acc':<10}")
    print("-" * 50)

    for t in thresholds:
        # Apply current threshold
        current_preds = (preds_probs > t).astype(int)
        
        # Calculate metrics
        rep = classification_report(gts, current_preds, target_names=labels, output_dict=True, zero_division=0)
        sub_acc = accuracy_score(gts, current_preds)
        
        micro_f1 = rep['micro avg']['f1-score']
        macro_f1 = rep['macro avg']['f1-score']
        
        micro_f1s.append(micro_f1)
        macro_f1s.append(macro_f1)
        
        print(f"{t:<10.2f} | {micro_f1:<10.4f} | {macro_f1:<10.4f} | {sub_acc:<10.2%}")

    if show_plot:
        plt.figure(figsize=(10, 5))
        plt.plot(thresholds, micro_f1s, label='Micro F1', marker='o')
        plt.plot(thresholds, macro_f1s, label='Macro F1', marker='s')
        plt.xlabel('Detection Threshold')
        plt.ylabel('F1 Score')
        plt.title('Threshold vs. Model Performance')
        plt.legend()
        plt.grid(True)
        plt.show()

    # Identify best thresholds
    best_micro_t = thresholds[np.argmax(micro_f1s)]
    best_macro_t = thresholds[np.argmax(macro_f1s)]
    
    print(f"\nBest Threshold (Micro F1): {best_micro_t:.2f} (Score: {max(micro_f1s):.4f})")
    print(f"Best Threshold (Macro F1): {best_macro_t:.2f} (Score: {max(macro_f1s):.4f})")
    
    return best_micro_t

def display_formatted_results(results_dict):
    """
    Converts the classification report dictionary into a styled Pandas DataFrame
    for clear visualisation in Jupyter Notebooks.
    """
    # 1. Convert dictionary to DataFrame and Transpose
    df = pd.DataFrame(results_dict).transpose()

    # 2. Remove the global 'accuracy' row to keep the table focus on per-class metrics
    if 'accuracy' in df.index:
        df = df.drop('accuracy')

    # 3. Apply styling: 4 decimal places and a colour gradient for the F1-Score
    # This helps identify underperforming instruments (like pipa or sheng) at a glance.
    styled_df = df.style.format({
        "precision": "{:.4f}",
        "recall": "{:.4f}",
        "f1-score": "{:.4f}",
        "support": "{:.0f}"
    }).background_gradient(cmap='YlGnBu', subset=['f1-score'])

    print("\n--- Detailed Classification Report ---")
    return styled_df

def run_inference(
    *,
    model_cls,
    model_kwargs: dict,
    model_weights_path,
    device,
    test_manifest_csv,
    root: Path,
    state_key: str = "model_state",
    audio_cfg_key: str = "audio_config",
    classes_key: str = "classes",
    strict_load: bool = True,
    show_progress: bool = True,
):
    """
    Loads checkpoint + model, runs inference over a manifest CSV.

    Returns:
        preds_arr: (N, C) float array of predicted probabilities
        gts_arr:   (N, C) int/bool array of ground-truth multi-hot vectors
        sample_ids: list[str] stems of wav filenames
        audio_cfg: dict-like audio config from checkpoint
        valid_labels: list[str] normalised class names
        label_to_idx: dict[str, int]
    """
    ckpt = torch.load(model_weights_path, map_location=device)

    audio_cfg = ckpt[audio_cfg_key]
    valid_labels = [c.strip().lower() for c in ckpt[classes_key]]
    label_to_idx = {name: i for i, name in enumerate(valid_labels)}

    model = model_cls(**model_kwargs, num_classes=len(valid_labels)).to(device)
    model.load_state_dict(ckpt[state_key], strict=strict_load)
    model.eval()

    df = _read_csv_with_fallback(Path(test_manifest_csv))

    def _resolve_path(p):
        p = Path(p)
        return p if p.is_absolute() else (root / p).resolve()

    df["wav_path"] = df["wav_path"].apply(lambda p: str(_resolve_path(p)))
    df["txt_path"] = df["txt_path"].apply(lambda p: str(_resolve_path(p)))

    all_preds, all_gt, sample_ids = [], [], []

    print(f"Running inference on {len(df)} samples against {len(valid_labels)} classes...")

    iterator = df.iterrows()
    if show_progress:
        iterator = tqdm(iterator, total=len(df))

    n_fail = 0
    with torch.no_grad():
        for _, row in iterator:
            try:
                gt_vec = parse_ground_truth(row["txt_path"], label_to_idx)
                mel = load_and_preprocess(row["wav_path"], audio_cfg)
                probs = get_prediction(model, mel, device)
            except Exception as exc:
                n_fail += 1
                if n_fail <= 10:
                    print(f"[WARN] Skipping unreadable sample: {row['wav_path']} ({exc})")
                continue

            all_preds.append(probs)
            all_gt.append(gt_vec)
            sample_ids.append(Path(row["wav_path"]).stem)

    preds_arr = np.asarray(all_preds)
    gts_arr = np.asarray(all_gt)

    if n_fail:
        print(f"[WARN] Skipped {n_fail} samples due to read/preprocess errors.")

    return preds_arr, gts_arr, sample_ids, audio_cfg, valid_labels, label_to_idx


def evaluate_multilabel_performance(
    all_preds,
    all_gt,
    class_list,
    sample_ids=None,
    threshold=0.5,
    debug=False,
    zero_division=0,
):
    classes = [c.strip().lower() for c in class_list]
    probs = np.asarray(all_preds)
    gts = np.asarray(all_gt).astype(int)

    if probs.shape != gts.shape:
        raise ValueError(f"Shape mismatch: preds {probs.shape} vs gts {gts.shape}")

    preds = (probs >= threshold).astype(int)

    num_samples, num_classes = preds.shape

    # ---- Key counts ----
    total_pos_gt = int(gts.sum())
    total_pos_pred = int(preds.sum())
    total_entries = int(num_samples * num_classes)

    # Confusion totals across ALL labels (micro)
    tp = int(((preds == 1) & (gts == 1)).sum())
    fp = int(((preds == 1) & (gts == 0)).sum())
    fn = int(((preds == 0) & (gts == 1)).sum())
    tn = int(((preds == 0) & (gts == 0)).sum())

    # ---- None prediction rate ----
    none_pred_mask = preds.sum(axis=1) == 0
    num_none = int(none_pred_mask.sum())

    # ---- Metrics ----
    subset_acc = accuracy_score(gts, preds)                      # exact match
    hamming_acc = 1.0 - hamming_loss(gts, preds)                 # label-wise accuracy

    report = classification_report(
        gts,
        preds,
        target_names=classes,
        output_dict=True,
        zero_division=zero_division,
    )

    # ---- Print summary ----
    print(f"Classification threshold probability: {threshold}")
    print(f"Samples: {num_samples} | Classes: {num_classes} | Decisions: {total_entries}")
    print(f"GT positives: {total_pos_gt} ({total_pos_gt/total_entries:.2%} of all decisions)")
    print(f"Pred positives: {total_pos_pred} ({total_pos_pred/total_entries:.2%} of all decisions)")
    print(f"Predicted 'None' (all-zero): {num_none} ({num_none/num_samples:.2%})")
    print("")
    print(f"Hamming accuracy (label-wise): {hamming_acc:.2%}")
    print(f"Subset accuracy (exact match): {subset_acc:.2%}")
    print(f"Micro F1:  {report['micro avg']['f1-score']:.4f}")
    print(f"Macro F1:  {report['macro avg']['f1-score']:.4f}")
    print("")

    print(f"TP={tp} FP={fp} FN={fn} TN={tn}")
    if total_pos_pred < max(5, 0.02 * total_pos_gt):
        print("WARNING: Very few positive predictions relative to GT positives.")
        print("         Your threshold is likely too high, or logits are miscalibrated.\n")

    # ---- Per-class table ----
    pos_per_class = gts.sum(axis=0)
    print(f"{'Instrument':<15} | {'Prec':>6} | {'Recall':>6} | {'F1':>6} | {'Support':>7} | {'Pred':>5}")
    print("-" * 70)

    for i, name in enumerate(classes):
        support = int(pos_per_class[i])
        pred_count = int(preds[:, i].sum())

        if support == 0:
            print(f"{name:<15} | {'  n/a':>6} | {'  n/a':>6} | {'  n/a':>6} | {support:>7} | {pred_count:>5}")
            continue

        prec = report[name]["precision"]
        rec = report[name]["recall"]
        f1 = report[name]["f1-score"]
        print(f"{name:<15} | {prec:6.2f} | {rec:6.2f} | {f1:6.2f} | {support:>7} | {pred_count:>5}")

    # ---- Debug examples where GT had positives ----
    if debug and sample_ids is not None:
        print("\n--- DEBUG: Examples where GT has at least one label ---")
        gt_nonzero = np.where(gts.sum(axis=1) > 0)[0]
        for idx in gt_nonzero[:]:
            pred_names = [classes[j] for j, v in enumerate(preds[idx]) if v]
            gt_names = [classes[j] for j, v in enumerate(gts[idx]) if v]
            print(f"ID: {sample_ids[idx]}")
            print(f"  Predicted: {pred_names if pred_names else '(none)'}")
            print(f"  Actual:    {gt_names if gt_names else '(none)'}")
            print("-" * 30)

    return report
