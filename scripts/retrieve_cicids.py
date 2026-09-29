"""
scripts/retrieve_cicids.py - Programmatic CICIDS2017 200k Subset Extractor
Downloads the official MachineLearningCSV archive, extracts a stratified
200,000 class-capped subset (100k benign / 100k attack across all attack families),
sanitizes column headers, and cleans up temporary archives to preserve disk space.
"""

import os
import sys
import shutil
import zipfile
import urllib.request
import io
import time
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CICIDS_ZIP_URL = "https://huggingface.co/datasets/bencorn/CICIDS2017/resolve/main/csvs/MachineLearningCSV.zip"
TARGET_CSV_PATH = DATA_DIR / "CICIDS2017_subset_200k.csv"
TEMP_ZIP_PATH = DATA_DIR / "MachineLearningCSV.zip"

def download_file(url: str, dest_path: Path, max_retries: int = 3, timeout: int = 120):
    """Downloads a file with streaming progress display and automatic retries."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = dest_path.with_suffix(".tmp")
    
    for attempt in range(1, max_retries + 1):
        try:
            print(f"[*] Downloading {dest_path.name} from mirror (Attempt {attempt}/{max_retries})...")
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=timeout) as response:
                total_size = int(response.headers.get("content-length", 0))
                downloaded = 0
                chunk_size = 1024 * 128
                t0 = time.time()
                
                with open(temp_path, "wb") as f:
                    while True:
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)
                        elapsed = time.time() - t0
                        speed_mb = (downloaded / (1024 * 1024)) / max(elapsed, 0.001)
                        if total_size > 0:
                            pct = (downloaded / total_size) * 100.0
                            sys.stdout.write(f"\r    Progress: {pct:5.1f}% [{downloaded/(1024*1024):.1f}/{total_size/(1024*1024):.1f} MB] ({speed_mb:.2f} MB/s)")
                        else:
                            sys.stdout.write(f"\r    Downloaded: {downloaded/(1024*1024):.1f} MB ({speed_mb:.2f} MB/s)")
                        sys.stdout.flush()
            temp_path.replace(dest_path)
            print("\n[+] Download completed successfully.")
            return True
        except Exception as e:
            print(f"\n[!] Download error: {e}")
            if temp_path.exists():
                temp_path.unlink()
            if attempt < max_retries:
                time.sleep(3)
            else:
                raise

def extract_200k_subset(zip_path: Path, output_csv: Path, target_total: int = 200000) -> Path:
    """
    Streams CSV files directly from the zip archive without extracting the full 3GB dataset,
    sampling benign flows and all attack flows to construct a balanced, class-capped 200,000-row subset.
    """
    print(f"[*] Opening archive {zip_path.name} for streaming extraction...")
    
    target_attacks = target_total // 2
    target_benign = target_total // 2
    
    attack_dfs = []
    benign_dfs = []
    total_attacks_collected = 0
    total_benign_collected = 0
    
    with zipfile.ZipFile(zip_path, "r") as z:
        csv_files = [f for f in z.namelist() if f.endswith(".csv") and not f.startswith("__MACOSX")]
        print(f"    Discovered {len(csv_files)} CSV files inside archive:")
        for cf in csv_files:
            print(f"      - {cf}")
            
        # Target samples per file for benign to ensure temporal diversity across all days
        benign_per_file = target_benign // len(csv_files) + 1000
        
        for cf in csv_files:
            print(f"\n[*] Processing flows from: {cf} ...")
            with z.open(cf) as f_in:
                # Read chunks
                chunk_iter = pd.read_csv(f_in, chunksize=25000, encoding="utf-8", low_memory=False)
                file_benign = []
                file_attacks = []
                
                for chunk in chunk_iter:
                    # Clean and standardize column names
                    chunk.columns = [c.strip() for c in chunk.columns]
                    
                    label_cols = [c for c in chunk.columns if "label" in c.lower()]
                    if not label_cols:
                        continue
                    l_col = label_cols[0]
                    
                    # Split into benign and attack
                    is_benign = chunk[l_col].astype(str).str.strip().str.upper() == "BENIGN"
                    benign_chunk = chunk[is_benign]
                    attack_chunk = chunk[~is_benign]
                    
                    if not attack_chunk.empty:
                        file_attacks.append(attack_chunk)
                        
                    if len(file_benign) * 25000 < benign_per_file and not benign_chunk.empty:
                        # Sample fraction of benign to distribute across the whole day
                        sampled_benign = benign_chunk.sample(n=min(len(benign_chunk), 5000), random_state=42)
                        file_benign.append(sampled_benign)
                        
                if file_attacks:
                    df_file_attacks = pd.concat(file_attacks, ignore_index=True)
                    attack_dfs.append(df_file_attacks)
                    total_attacks_collected += len(df_file_attacks)
                    print(f"    -> Extracted {len(df_file_attacks):,} attack flows (Current total: {total_attacks_collected:,})")
                    
                if file_benign:
                    df_file_benign = pd.concat(file_benign, ignore_index=True)
                    benign_dfs.append(df_file_benign)
                    total_benign_collected += len(df_file_benign)
                    print(f"    -> Extracted {len(df_file_benign):,} benign flows (Current total: {total_benign_collected:,})")

    # Combine and calibrate to exact 200,000 class-capped subset
    print("\n[*] Consolidating and class-capping extracted flows...")
    all_attacks = pd.concat(attack_dfs, ignore_index=True) if attack_dfs else pd.DataFrame()
    all_benign = pd.concat(benign_dfs, ignore_index=True) if benign_dfs else pd.DataFrame()
    
    print(f"    Raw extracted attack count: {len(all_attacks):,}")
    print(f"    Raw extracted benign count: {len(all_benign):,}")
    
    # Cap attacks and benign to target
    if len(all_attacks) > target_attacks:
        # Stratified sampling by attack category if possible
        label_col = [c for c in all_attacks.columns if "label" in c.lower()][0]
        # Cap high-volume attacks (DDoS/PortScan) to ensure rare attacks (Infiltration, Web, Patator, Heartbleed) are preserved
        capped_attacks = all_attacks.groupby(label_col, group_keys=False).apply(
            lambda g: g.sample(min(len(g), max(500, int(target_attacks * len(g) / len(all_attacks)))), random_state=42)
        )
        if len(capped_attacks) > target_attacks:
            all_attacks = capped_attacks.sample(n=target_attacks, random_state=42)
        else:
            remaining = target_attacks - len(capped_attacks)
            extra = all_attacks.drop(capped_attacks.index).sample(n=remaining, random_state=42)
            all_attacks = pd.concat([capped_attacks, extra], ignore_index=True)
    elif len(all_attacks) < target_attacks:
        target_benign = target_total - len(all_attacks)
        
    if len(all_benign) > target_benign:
        all_benign = all_benign.sample(n=target_benign, random_state=42)
        
    combined_df = pd.concat([all_attacks, all_benign], ignore_index=True)
    # Shuffle
    combined_df = combined_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    
    # Ensure binary label column is explicitly created: 0 for Benign, 1 for Attack
    label_col = [c for c in combined_df.columns if "label" in c.lower()][0]
    combined_df["label"] = (combined_df[label_col].astype(str).str.strip().str.upper() != "BENIGN").astype(int)
    combined_df["attack_cat"] = combined_df[label_col].astype(str).str.strip()
    
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    combined_df.to_csv(output_csv, index=False)
    
    size_mb = output_csv.stat().st_size / (1024 * 1024)
    print(f"\n[+] Successfully saved {len(combined_df):,} records to {output_csv} ({size_mb:.2f} MB)")
    print(f"    Label distribution: {combined_df['label'].value_counts().to_dict()}")
    print(f"    Attack category distribution:\n{combined_df['attack_cat'].value_counts()}")
    
    return output_csv

def main():
    print("=" * 75)
    print(" CICIDS2017 BENCHMARK SUBSET RETRIEVER (200,000 ROWS CLASS-CAPPED)")
    print("=" * 75)
    
    if TARGET_CSV_PATH.exists():
        print(f"[i] Target subset already exists at {TARGET_CSV_PATH} ({TARGET_CSV_PATH.stat().st_size / (1024*1024):.2f} MB).")
        df = pd.read_csv(TARGET_CSV_PATH, nrows=5)
        print(f"    Detected {len(df.columns)} columns.")
        return
        
    # Download archive
    if not TEMP_ZIP_PATH.exists():
        download_file(CICIDS_ZIP_URL, TEMP_ZIP_PATH)
    else:
        print(f"[i] Archive {TEMP_ZIP_PATH.name} already exists. Skipping download.")
        
    # Extract subset
    try:
        extract_200k_subset(TEMP_ZIP_PATH, TARGET_CSV_PATH, target_total=200000)
    finally:
        # Always remove temp zip to maintain disk hygiene
        if TEMP_ZIP_PATH.exists():
            print(f"[*] Removing temporary archive {TEMP_ZIP_PATH.name} to free disk space...")
            TEMP_ZIP_PATH.unlink()
            print("[+] Temporary archive cleaned up.")

if __name__ == "__main__":
    main()
