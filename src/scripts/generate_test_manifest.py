import csv
import argparse
from pathlib import Path

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

    with open(args.out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["wav_path", "txt_path"])
        writer.writerows(rows)
    
    print(f"Test manifest created with {len(rows)} files.")

if __name__ == "__main__":
    main()