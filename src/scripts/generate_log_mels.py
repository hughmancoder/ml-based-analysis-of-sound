#!/usr/bin/env python3
import argparse
import yaml
import csv
from pathlib import Path
from tqdm import tqdm
from preprocessing import precache_one


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def _iter_wavs_from_train_dir(root: Path):
    """
    Finds all .wav files in the directory.
    Assumes IRMAS/Folder structure: root/instrument_name/file.wav
    """
    if not root.exists():
        raise FileNotFoundError(f"Directory not found: {root}")
    
    # We yield (file_path, instrument_label)
    for wav in root.rglob("*.wav"):
        if wav.is_file():
            # Parent folder name is the label (e.g., 'vio', 'cel')
            label = wav.parent.name.strip().lower()
            yield wav, label

def main():
    # 1. Setup Argument Parser
    ap = argparse.ArgumentParser(description="Precompute Mel Spectrograms into .npy files")
    ap.add_argument("--config", default="configs/audio_params.yaml", help="Path to audio_params.yaml")
    ap.add_argument("--train_dir", help="Override the raw data source directory")
    args = ap.parse_args()

    # 2. Load and Resolve Configuration
    cfg = load_config(args.config)
    audio_cfg = cfg['audio']
    path_cfg = cfg['paths']

    # Resolve paths (Priority: CLI Argument > Config File)
    raw_dir = Path(args.train_dir or path_cfg['train_dir'])
    cache_root = Path(path_cfg['cache_root'])
    if "manifest_path" in path_cfg:
        manifest_path = path_cfg["manifest_path"]
    elif "train_manifest" in path_cfg:
        manifest_path = path_cfg["train_manifest"]
    else:
        raise KeyError(
            "Missing manifest path in config. Expected 'manifest_path' or 'train_manifest' under 'paths'."
        )

    out_csv = Path(manifest_path)

    # Ensure output directories exist
    cache_root.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    print(f"--- Processing Audio ---")
    print(f"Source: {raw_dir}")
    print(f"Cache:  {cache_root}")
    print(f"Params: SR={audio_cfg['sr']}, Mels={audio_cfg['n_mels']}, Dur={audio_cfg['duration']}s")

    # 3. Processing Loop
    rows_out = []
    n_ok, n_fail = 0, 0
    
    # Gather all files first for the progress bar
    pairs = list(_iter_wavs_from_train_dir(raw_dir))
    
    for wav_path, label in tqdm(pairs, desc="Generating Mels"):
        try:
            # Generate the .npy file using your existing precache_one logic
            # This function should save the file to cache_root/label/original_name.npy
            npy_path = precache_one(
                wav_path, label, cache_root,
                sr=audio_cfg['sr'], 
                dur=audio_cfg['duration'], 
                n_mels=audio_cfg['n_mels'],
                win_ms=audio_cfg['win_ms'], 
                hop_ms=audio_cfg['hop_ms'],
                fmin=audio_cfg['fmin'], 
                fmax=audio_cfg.get('fmax')
            )
            
            # rel_path = _safe_relpath(npy_path, PROJECT_ROOT)
            rel_path = npy_path.resolve().as_posix()
            rows_out.append([rel_path, label])
            n_ok += 1
            
        except KeyboardInterrupt:
            print("\nStopped by user.")
            break
        except Exception as e:
            print("INFO: Failed to process:", wav_path, "Error:", e)
            n_fail += 1
            # Uncomment for debugging specific file failures
            # print(f"Error on {wav_path.name}: {e}")

    # 4. Save Manifest
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filepath", "label"]) # Header
        w.writerows(rows_out)

    print(f"\nProcessing Complete.")
    print(f"Successfully processed: {n_ok}")
    print(f"Failed: {n_fail}")
    print(f"Manifest written to: {out_csv}")

if __name__ == "__main__":
    main()
