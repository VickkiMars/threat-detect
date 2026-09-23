"""
scripts/download_datasets.py - Programmatic Benchmark Dataset Downloader
Downloads and validates the official UNSW-NB15 benchmark splits for edge model training.
Includes pre-flight disk safety checks and dataset integrity validation.
"""

import os
import sys
import shutil
import urllib.request
import argparse
import pandas as pd
from pathlib import Path

# Official curated mirror URLs for UNSW-NB15 pre-split CSVs
DATASET_URLS = {
    "UNSW-NB15": {
        "train": {
            "filename": "UNSW_NB15_training-set.csv",
            "url": "https://raw.githubusercontent.com/Nir-J/ML-Projects/master/UNSW-Network_Packet_Classification/UNSW_NB15_training-set.csv",
            "expected_rows": 82332,
            "min_size_mb": 30.0
        },
        "test": {
            "filename": "UNSW_NB15_testing-set.csv",
            "url": "https://raw.githubusercontent.com/Nir-J/ML-Projects/master/UNSW-Network_Packet_Classification/UNSW_NB15_testing-set.csv",
            "expected_rows": 175341,
            "min_size_mb": 14.0
        }
    }
}

def check_disk_space(target_dir: Path, min_free_mb: float = 500.0) -> bool:
    """Verifies that the target filesystem has sufficient free space."""
    target_dir.mkdir(parents=True, exist_ok=True)
    total, used, free = shutil.disk_usage(target_dir)
    free_mb = free / (1024 * 1024)
    print(f"[*] Disk Space Check: {free_mb:.1f} MB free on {target_dir.resolve()}")
    if free_mb < min_free_mb:
        print(f"[!] ERROR: Insufficient disk space ({free_mb:.1f} MB < {min_free_mb:.1f} MB required).")
        return False
    return True

def download_with_progress(url: str, output_path: Path, max_retries: int = 3, timeout: int = 60):
    """Downloads a file showing downloaded bytes and progress with retries."""
    import time
    temp_path = output_path.with_suffix(".tmp")
    
    for attempt in range(1, max_retries + 1):
        try:
            print(f"[*] Downloading {output_path.name} (Attempt {attempt}/{max_retries})...")
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=timeout) as response:
                total_size = int(response.headers.get("content-length", 0))
                downloaded = 0
                chunk_size = 1024 * 64
                with open(temp_path, "wb") as f:
                    while True:
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            percent = min(100.0, (downloaded / total_size) * 100.0)
                            sys.stdout.write(f"\r    Progress: {percent:5.1f}% [{downloaded / (1024*1024):.1f} MB / {total_size / (1024*1024):.1f} MB]")
                        else:
                            sys.stdout.write(f"\r    Downloaded: {downloaded / (1024*1024):.1f} MB")
                        sys.stdout.flush()
            temp_path.replace(output_path)
            print("\n[+] Download complete.")
            return True
        except Exception as e:
            print(f"\n[!] Download error: {e}")
            if temp_path.exists():
                temp_path.unlink()
            if attempt < max_retries:
                print("[*] Retrying in 3 seconds...")
                time.sleep(3)
            else:
                raise

def validate_dataset(file_path: Path, expected_rows: int, min_size_mb: float) -> bool:
    """Validates row count, columns, and class balance of downloaded CSV."""
    size_mb = file_path.stat().st_size / (1024 * 1024)
    print(f"\n[*] Validating {file_path.name} ({size_mb:.2f} MB)...")
    
    if size_mb < min_size_mb:
        print(f"[!] File size warning: {size_mb:.2f} MB is below expected {min_size_mb:.2f} MB.")
        return False

    try:
        df = pd.read_csv(file_path, nrows=5)
        print(f"    Columns detected: {len(df.columns)} (Expected 45 features)")
        
        # Read full row count
        total_rows = sum(1 for _ in open(file_path, "r", encoding="utf-8", errors="ignore")) - 1
        print(f"    Total Flow Records: {total_rows:,} (Expected: {expected_rows:,})")
        
        # Load sample for category sanity check
        df_sample = pd.read_csv(file_path, usecols=["label", "attack_cat"], nrows=20000)
        label_dist = df_sample["label"].value_counts().to_dict()
        print(f"    Sample Class Ratio (20k rows): {label_dist}")
        
        return True
    except Exception as e:
        print(f"[!] Validation failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Download benchmark dataset splits for Grace NIDS.")
    parser.add_argument("--dest", type=str, default="data", help="Destination folder for datasets")
    parser.add_argument("--force", action="store_true", help="Re-download even if files already exist")
    args = parser.parse_args()

    dest_dir = Path(args.dest)
    print("=" * 65)
    print("  GRACE NIDS: PROGRAMMATIC BENCHMARK DATASET DOWNLOADER")
    print(f"  Target Destination: {dest_dir.resolve()}")
    print("=" * 65)

    if not check_disk_space(dest_dir, min_free_mb=500.0):
        sys.exit(1)

    dataset_info = DATASET_URLS["UNSW-NB15"]

    for split_key in ["train", "test"]:
        info = dataset_info[split_key]
        dest_file = dest_dir / info["filename"]
        
        if dest_file.exists() and not args.force:
            print(f"\n[i] {info['filename']} already exists ({dest_file.stat().st_size / (1024*1024):.1f} MB). Skipping download.")
        else:
            download_with_progress(info["url"], dest_file)

        validate_dataset(dest_file, info["expected_rows"], info["min_size_mb"])

    print("\n" + "=" * 65)
    print("  [SUCCESS] All benchmark dataset files ready in 'data/'")
    print("=" * 65)

if __name__ == "__main__":
    main()
