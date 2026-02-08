"""
Prepare Chapman-Shaoxing ECG Database for Dual Memory LLCS

Convert Chapman-Shaoxing to HardOOD3-compatible format:
- Extract Lead II ECG waveforms
- Detect R-peaks
- Create CSV files with Time, Voltage, Peak columns
- Generate Poincaré NumpyData (RR intervals)
- Split into 4 environments by age groups
- Create OOD unknown samples

Usage:
    python phase3/chapman_experiment/02_prepare_chapman.py
"""

import argparse
import os
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks

try:
    import wfdb
except ImportError:
    print("ERROR: wfdb package not found!")
    print("Install with: pip install wfdb")
    import sys
    sys.exit(1)


# Rhythm mapping to simplified labels
RHYTHM_MAPPING = {
    'SB': 0,    # Sinus Bradycardia → Label 0 (Normal-like)
    'SR': 0,    # Sinus Rhythm → Label 0 (Normal)
    'AFIB': 1,  # Atrial Fibrillation → Label 1
    'ST': 2,    # Sinus Tachycardia → Label 2
    'AF': 1,    # Atrial Fibrillation → Label 1 (alternative code)
    'AFLT': 3,  # Atrial Flutter → Label 3
    'SA': 0,    # Sinus Arrhythmia → Label 0
}

# Unknown rhythms (will be label 4)
UNKNOWN_RHYTHMS = ['SVT', 'SI', 'AT', 'AVNRT', 'AVRT', 'SAAWR', 'JR']


def bandpass_filter(signal: np.ndarray, fs: int, low: float = 0.5, high: float = 40.0) -> np.ndarray:
    """Apply bandpass filter to ECG signal"""
    if len(signal) == 0:
        return signal
    nyq = 0.5 * fs
    b, a = butter(2, [low / nyq, high / nyq], btype='band')
    return filtfilt(b, a, signal)


def detect_r_peaks(signal: np.ndarray, fs: int) -> np.ndarray:
    """Detect R-peaks in ECG signal"""
    if len(signal) == 0:
        return np.array([], dtype=int)
    
    # Bandpass filter
    filtered = bandpass_filter(signal, fs)
    
    # Normalize
    filtered = (filtered - np.mean(filtered)) / (np.std(filtered) + 1e-9)
    
    # Pan-Tompkins-like detection
    diff = np.diff(filtered, prepend=filtered[0])
    squared = diff ** 2
    
    # Moving average
    window = max(1, int(0.15 * fs))
    kernel = np.ones(window) / window
    integrated = np.convolve(squared, kernel, mode='same')
    
    # Find peaks
    distance = int(0.25 * fs)  # Min 0.25s between peaks (max 240 bpm)
    height = np.percentile(integrated, 85)
    peaks, _ = find_peaks(integrated, distance=distance, height=height)
    
    if len(peaks) < 3:
        return peaks
    
    # Filter physiologically impossible peaks
    rr = np.diff(peaks)
    median_rr = np.median(rr)
    if median_rr <= 0:
        return peaks
    
    # Keep peaks with reasonable RR intervals
    keep = [peaks[0]]
    for p in peaks[1:]:
        if 0.6 * median_rr <= (p - keep[-1]) <= 1.6 * median_rr:
            keep.append(p)
    
    return np.array(keep, dtype=int)


def create_poincare_matrix(rr_intervals: np.ndarray) -> np.ndarray:
    """
    Create Poincaré plot matrix (30x30 grid) from RR intervals
    Similar to HardOOD3 format
    """
    if len(rr_intervals) < 2:
        return np.zeros((31, 30), dtype=float)
    
    # Convert to milliseconds
    rr_ms = rr_intervals * 1000
    rr_n = rr_ms[:-1]
    rr_n1 = rr_ms[1:]
    
    # Create 30x30 grid
    grid_size = 30
    x_min, x_max = 400, 1400
    y_min, y_max = 400, 1400
    
    matrix = np.zeros((grid_size, grid_size), dtype=float)
    
    x_step = (x_max - x_min) / grid_size
    y_step = (y_max - y_min) / grid_size
    
    for x, y in zip(rr_n, rr_n1):
        if x_min <= x < x_max and y_min <= y < y_max:
            x_idx = int((x - x_min) / x_step)
            y_idx = int((y - y_min) / y_step)
            x_idx = min(x_idx, grid_size - 1)
            y_idx = min(y_idx, grid_size - 1)
            matrix[y_idx, x_idx] = 1
    
    # Add activity flag row (always 0 to avoid leakage)
    activity_row = np.zeros((1, grid_size), dtype=float)
    result = np.vstack([matrix, activity_row])
    
    return result


def map_rhythm_to_label(rhythm: str) -> int:
    """Map rhythm string to label (0-4)"""
    rhythm = rhythm.strip().upper()
    
    # Check if unknown
    if any(unk in rhythm for unk in UNKNOWN_RHYTHMS):
        return 4
    
    # Check known rhythms
    for key, label in RHYTHM_MAPPING.items():
        if key in rhythm:
            return label
    
    # Default to unknown
    return 4


def age_to_env(age: float) -> str:
    """Map age to environment"""
    if age < 40:
        return 'Env1'
    elif age < 60:
        return 'Env2'
    elif age < 80:
        return 'Env3'
    else:
        return 'Env4'


def process_chapman_database(
    input_root: Path,
    output_root: Path,
    csv_root: Path,
    lead_name: str = 'II',
    n_per_env: int = 1000,
    seed: int = 42
):
    """Process Chapman-Shaoxing database"""
    
    np.random.seed(seed)
    
    print("=" * 80)
    print("Chapman-Shaoxing Database Preparation")
    print("=" * 80)
    print(f"Input: {input_root}")
    print(f"Output (NumpyData): {output_root}")
    print(f"Output (CSV): {csv_root}")
    print(f"Lead: {lead_name}")
    print(f"Samples per env: {n_per_env}")
    print()
    
    # Load metadata - SNOMED-CT version
    metadata_path = input_root / "ConditionNames_SNOMED-CT.csv"
    if not metadata_path.exists():
        # Try old filename
        metadata_path = input_root / "ConditionNames.csv"
        if not metadata_path.exists():
            print(f"ERROR: Metadata file not found!")
            print("Make sure you've downloaded the database first.")
            return False
    
    # Load SNOMED mapping
    df_conditions = pd.read_csv(metadata_path)
    snomed_map = dict(zip(df_conditions['Snomed_CT'].astype(str), 
                         df_conditions['Acronym Name']))
    
    # Get all record files
    records_file = input_root / "RECORDS"
    if not records_file.exists():
        print(f"ERROR: {records_file} not found!")
        return False
    
    record_dirs = []
    with open(records_file) as f:
        for line in f:
            line = line.strip()
            if line:
                record_dirs.append(line)
    print(f"Total record directories: {len(record_dirs)}")
    print(f"Total SNOMED-CT codes: {len(snomed_map)}")
    print()
    
    # Lead mapping
    lead_map = {'I': 0, 'II': 1, 'III': 2, 'AVR': 3, 'AVL': 4, 'AVF': 5,
                'V1': 6, 'V2': 7, 'V3': 8, 'V4': 9, 'V5': 10, 'V6': 11}
    
    if lead_name.upper() not in lead_map:
        print(f"ERROR: Unknown lead '{lead_name}'")
        print(f"Available: {list(lead_map.keys())}")
        return False
    
    lead_idx = lead_map[lead_name.upper()]
    
    # Process records from WFDB files
    env_data = defaultdict(list)  # env -> [(record_path, label, age, rhythm)]
    stats = defaultdict(int)
    
    print("Processing records...")
    processed = 0
    for record_dir in record_dirs:
        record_path = input_root / record_dir
        
        # Find all .hea files in this directory
        hea_files = list(record_path.glob("*.hea"))
        
        for hea_file in hea_files:
            if processed % 1000 == 0:
                print(f"  Processed {processed} records...")
            
            try:
                # Parse .hea file
                with open(hea_file) as f:
                    lines = f.readlines()
                
                # Extract metadata
                age = 50.0  # default
                dx_codes = []
                
                for line in lines:
                    if line.startswith('#Age:'):
                        age_str = line.split(':')[1].strip()
                        try:
                            age = float(age_str)
                        except:
                            age = 50.0
                    elif line.startswith('#Dx:'):
                        dx_str = line.split(':')[1].strip()
                        dx_codes = [c.strip() for c in dx_str.split(',')]
                
                # Map SNOMED codes to rhythms
                rhythms = []
                for code in dx_codes:
                    if code in snomed_map:
                        rhythms.append(snomed_map[code])
                
                if not rhythms:
                    continue
                
                # Use first rhythm for simplicity
                rhythm = rhythms[0]
                label = map_rhythm_to_label(rhythm)
                
                record_id = hea_file.stem  # JS00001, etc
                
                # Filter: only keep labels 0-3 for training envs
                if label in [0, 1, 2, 3]:
                    env = age_to_env(age)
                    env_data[env].append((str(record_path / record_id), label, age, rhythm))
                    stats[f"{env}_label_{label}"] += 1
                elif label == 4:
                    env_data['Unknown'].append((str(record_path / record_id), label, age, rhythm))
                    stats['Unknown'] += 1
                
                processed += 1
                
            except Exception as e:
                continue
    
    print()
    print("Distribution:")
    for key, count in sorted(stats.items()):
        print(f"  {key}: {count}")
    print()
    
    # Subsample each environment
    print("Subsampling environments...")
    selected_data = {}
    for env in ['Env1', 'Env2', 'Env3', 'Env4']:
        if env not in env_data or len(env_data[env]) == 0:
            print(f"  WARNING: {env} has no data!")
            continue
        
        records = env_data[env]
        
        # Stratified sampling by label
        label_records = defaultdict(list)
        for rec in records:
            label_records[rec[1]].append(rec)
        
        # Sample equally from each label
        n_labels = len(label_records)
        n_per_label = n_per_env // n_labels
        
        selected = []
        for label, recs in label_records.items():
            if len(recs) <= n_per_label:
                selected.extend(recs)
            else:
                # Convert list of tuples to indices, then select
                indices = np.random.choice(len(recs), n_per_label, replace=False)
                selected.extend([recs[i] for i in indices])
        
        selected_data[env] = selected
        print(f"  {env}: {len(selected)} records")
    
    # Unknown samples (OOD)
    unknown_records = env_data.get('Unknown', [])
    if len(unknown_records) > 200:
        indices = np.random.choice(len(unknown_records), 200, replace=False)
        unknown_records = [unknown_records[i] for i in indices]
    selected_data['Eval'] = unknown_records
    print(f"  Eval (Unknown): {len(unknown_records)} records")
    print()
    
    # Process and save
    print("Creating output files...")
    total_success = 0
    total_failed = 0
    
    for env, records in selected_data.items():
        print(f"\nProcessing {env}...")
        
        # Create directories
        npy_dir = output_root / env / "NumpyData"
        csv_dir = csv_root / env / "csv"
        npy_dir.mkdir(parents=True, exist_ok=True)
        csv_dir.mkdir(parents=True, exist_ok=True)
        
        for record_path, label, age, rhythm in records:
            try:
                # Load ECG signal - record_path is already full path
                record = wfdb.rdrecord(record_path)
                
                # Extract lead
                if lead_idx >= record.n_sig:
                    total_failed += 1
                    continue
                
                signal = record.p_signal[:, lead_idx]
                fs = record.fs
                
                # Detect R-peaks
                peaks = detect_r_peaks(signal, fs)
                
                if len(peaks) < 3:
                    total_failed += 1
                    continue
                
                # Create time vector
                time = np.arange(len(signal)) / fs
                
                # Create Peak column
                peak_col = np.zeros(len(signal), dtype=int)
                peak_col[peaks] = 3
                
                # Save CSV
                record_id = Path(record_path).name
                filename_base = f"chap_{record_id}_{label}_{env}"
                csv_path = csv_dir / f"{filename_base}.csv"
                
                df_out = pd.DataFrame({
                    'Time': time,
                    'Voltage': signal,
                    'Peak': peak_col
                })
                df_out.to_csv(csv_path, index=False)
                
                # Create Poincaré matrix
                rr_intervals = np.diff(time[peaks])
                poincare = create_poincare_matrix(rr_intervals)
                
                # Save numpy
                npy_path = npy_dir / f"{filename_base}.npy"
                np.save(npy_path, poincare)
                
                total_success += 1
                
            except Exception as e:
                total_failed += 1
                if total_failed < 10:  # Only show first 10 errors
                    print(f"  Error processing {record_id}: {e}")
    
    print()
    print("=" * 80)
    print("Summary:")
    print(f"  Successfully processed: {total_success}")
    print(f"  Failed: {total_failed}")
    print()
    print("Output directories:")
    print(f"  NumpyData: {output_root}")
    print(f"  CSV: {csv_root}")
    print()
    print("Next step:")
    print("  python phase3/chapman_experiment/03_run_dual_memory.py")
    print("=" * 80)
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Prepare Chapman-Shaoxing for Dual Memory LLCS"
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("Data") / "Chapman_Raw",
        help="Chapman-Shaoxing raw data directory"
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("Data") / "Disease_dataset_chapman",
        help="Output directory for NumpyData"
    )
    parser.add_argument(
        "--csv-root",
        type=Path,
        default=Path("Data") / "Generated_Chapman",
        help="Output directory for CSV files"
    )
    parser.add_argument(
        "--lead",
        type=str,
        default="II",
        choices=['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'],
        help="ECG lead to extract"
    )
    parser.add_argument(
        "--n-per-env",
        type=int,
        default=1000,
        help="Number of samples per environment"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    
    args = parser.parse_args()
    
    success = process_chapman_database(
        input_root=args.input_root,
        output_root=args.output_root,
        csv_root=args.csv_root,
        lead_name=args.lead,
        n_per_env=args.n_per_env,
        seed=args.seed
    )
    
    import sys
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

