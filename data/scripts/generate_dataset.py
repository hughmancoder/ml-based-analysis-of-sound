import os
import json
from pathlib import Path
from pydub import AudioSegment

REPO_ROOT = Path(__file__).resolve().parents[2]

def guard_output_root(output_root: Path) -> Path:
    allow_unsafe = os.environ.get("ALLOW_UNSAFE_PATHS") == "1"
    p = Path(output_root).expanduser().resolve()
    if allow_unsafe:
        return p
    if p.parent == p:
        raise SystemExit(f"[fatal] Refusing output_root at drive root: {p}")
    if REPO_ROOT not in p.parents and p != REPO_ROOT:
        raise SystemExit(f"[fatal] Refusing output_root outside repo: {p} (set ALLOW_UNSAFE_PATHS=1 to override)")
    return p

def generate_dataset(json_path, output_root):
    # Configuration
    SAMPLE_RATE = 44100
    CLIP_DURATION_MS = 3000  # 3 seconds
    
    # 1. Get the directory where labels.json is located
    json_file_path = Path(json_path).resolve()
    base_dir = json_file_path.parent 
    
    # Load the labels data
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Base directory for audio search: {base_dir}")
    print(f"Starting processing: {len(data)} source files identified.")

    output_root = guard_output_root(output_root)
    for entry in data:
        relative_file_path = entry.get('file')
        # 2. Join the json's folder with the relative path in the json
        full_audio_path = (base_dir / relative_file_path).resolve()
        
        label = entry.get('labels')[0] if entry.get('labels') else "unlabelled"
        time_stamps = entry.get('time_stamps_to_extract', [])

        # Create the sub-folder for the label
        target_dir = Path(output_root) / label
        target_dir.mkdir(parents=True, exist_ok=True)

        if not full_audio_path.exists():
            print(f"Warning: File not found - {full_audio_path}")
            continue

        try:
            # Load audio
            audio = AudioSegment.from_file(str(full_audio_path))
            audio = audio.set_frame_rate(SAMPLE_RATE)
            
            # Extract specific segment if timestamps provided
            if time_stamps:
                def to_ms(time_str):
                    parts = list(map(int, time_str.split(':')))
                    return (parts[0] * 60 + parts[1]) * 1000 if len(parts) == 2 else parts[0] * 1000
                
                start_ms = to_ms(time_stamps[0][0])
                end_ms = to_ms(time_stamps[0][1])
                working_audio = audio[start_ms:end_ms]
            else:
                working_audio = audio

            # Slice into 3-second segments
            segment_count = 0
            for i in range(0, len(working_audio), CLIP_DURATION_MS):
                chunk = working_audio[i : i + CLIP_DURATION_MS]
                
                if len(chunk) < CLIP_DURATION_MS:
                    continue
                
                segment_count += 1
                base_name = full_audio_path.stem
                output_filename = f"[{label}]{base_name}_seg{segment_count}.wav"
                output_path = target_dir / output_filename
                
                chunk.export(output_path, format="wav")
            
            print(f"Processed {label}: {full_audio_path.name} -> {segment_count} clips.")

        except Exception as e:
            print(f"Error processing {full_audio_path}: {e}")

if __name__ == "__main__":
    # Ensure these paths are correct relative to where you run the script
    # input_json = "data/raw_sources/chinese_instrument_labels.json"
    input_json = "data/raw_sources/western_orchestral_labels.json"
    output_directory = "data/train"
    
    generate_dataset(input_json, output_directory)
    print("\nProcessing complete.")
