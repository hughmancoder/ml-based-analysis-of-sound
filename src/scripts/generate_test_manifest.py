import csv
import argparse
from pathlib import Path
from utils.safe_paths import guard_path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_dir", required=True)
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    test_path = Path(args.test_dir)
    rows = []

    # Find all wav files in the test directory
    for wav_file in test_path.rglob("*.wav"):
        txt_file = wav_file.with_suffix(".txt")
        if txt_file.exists():
            # Store paths relative to the project root
            rows.append([str(wav_file), str(txt_file)])

    out_csv = guard_path(Path(args.out_csv), PROJECT_ROOT, "out_csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["wav_path", "txt_path"])
        writer.writerows(rows)
    
    print(f"Test manifest created with {len(rows)} files.")

if __name__ == "__main__":
    main()
