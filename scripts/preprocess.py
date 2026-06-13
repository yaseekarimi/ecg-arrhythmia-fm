"""
ECG Preprocessing Script
Phase 0 - Month 2
Harmonises all datasets to:
- 100 Hz sampling rate
- 10 second duration (1000 samples)
- Z-score normalisation
- Label mapping: AF, PVC, Other
"""

import os
import numpy as np
import wfdb
import json
from scipy import signal as scipy_signal

# ============================================================
# PATHS
# ============================================================
DATASETS = {
    "ptbxl": "data/ptbxl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3",
    "cpsc2018": "data/classification-of-12-lead-ecgs-the-physionetcomputing-in-cardiology-challenge-2020-1.0.2/training/cpsc_2018",
    "georgia": "data/classification-of-12-lead-ecgs-the-physionetcomputing-in-cardiology-challenge-2020-1.0.2/training/georgia",
    "chapman": "data/chapman",
    "mitbih": "data/mitbih",
}

OUTPUT_DIR = "data/processed"
TARGET_FS = 100       # Target sampling rate (Hz)
TARGET_LEN = 1000     # 10 seconds x 100 Hz

# ============================================================
# LABEL MAPPING
# ============================================================
AF_LABELS = ["AFIB", "AF", "Atrial fibrillation", "164889003"]
PVC_LABELS = ["PVC", "Premature ventricular", "164884008", "17338001"]

def map_label(labels):
    """Map raw labels to AF, PVC, or Other"""
    for l in labels:
        if any(af in str(l) for af in AF_LABELS):
            return "AF"
    for l in labels:
        if any(pvc in str(l) for pvc in PVC_LABELS):
            return "PVC"
    return "Other"

# ============================================================
# SIGNAL PROCESSING
# ============================================================
def resample_signal(sig, orig_fs, target_fs=TARGET_FS, target_len=TARGET_LEN):
    """Resample signal to target frequency and length"""
    if orig_fs != target_fs:
        num_samples = int(len(sig) * target_fs / orig_fs)
        sig = scipy_signal.resample(sig, num_samples)
    
    # Trim or pad to target length
    if len(sig) >= target_len:
        sig = sig[:target_len]
    else:
        pad = target_len - len(sig)
        sig = np.pad(sig, ((0, pad), (0, 0)), mode='constant')
    
    return sig

def zscore_normalise(sig):
    """Z-score normalisation per lead"""
    mean = np.mean(sig, axis=0)
    std = np.std(sig, axis=0)
    std[std == 0] = 1  # Avoid division by zero
    return (sig - mean) / std

# ============================================================
# PROCESS EACH DATASET
# ============================================================
def get_header_labels(header):
    """Extract labels from wfdb header comments"""
    labels = []
    for comment in header.comments:
        if "Dx:" in comment or "dx:" in comment:
            dx = comment.split(":")[-1].strip()
            labels.extend([l.strip() for l in dx.split(",")])
    return labels

def process_record(record_path):
    """Process a single ECG record"""
    try:
        record = wfdb.rdrecord(record_path)
        header = wfdb.rdheader(record_path)
        
        sig = record.p_signal  # Shape: (samples, leads)
        orig_fs = record.fs
        
        # Resample
        sig = resample_signal(sig, orig_fs)
        
        # Normalise
        sig = zscore_normalise(sig)
        
        # Get label
        labels = get_header_labels(header)
        label = map_label(labels)
        
        return sig, label, orig_fs
    
    except Exception as e:
        return None, None, None

def process_dataset(name, path):
    """Process all records in a dataset"""
    print(f"\n{'='*50}")
    print(f"Processing: {name}")
    print(f"{'='*50}")
    
    records = []
    
    # Find all .hea files
    hea_files = []
    for root, dirs, files in os.walk(path):
        for f in files:
            if f.endswith('.hea'):
                hea_files.append(os.path.join(root, f[:-4]))
    
    print(f"Found {len(hea_files)} records")
    
    label_counts = {"AF": 0, "PVC": 0, "Other": 0, "Error": 0}
    
    for i, record_path in enumerate(hea_files[:10]):  # Process first 10 as test
        sig, label, orig_fs = process_record(record_path)
        
        if sig is not None:
            label_counts[label] += 1
            records.append({
                "path": record_path,
                "label": label,
                "shape": sig.shape,
                "orig_fs": orig_fs
            })
        else:
            label_counts["Error"] += 1
        
        if (i+1) % 5 == 0:
            print(f"  Processed {i+1}/{min(10, len(hea_files))} records...")
    
    print(f"\nLabel distribution (first 10 records):")
    for label, count in label_counts.items():
        print(f"  {label}: {count}")
    
    return records

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("ECG Preprocessing Pipeline")
    print(f"Target: {TARGET_FS} Hz, {TARGET_LEN} samples (10 seconds)")
    print(f"Labels: AF, PVC, Other")
    
    all_results = {}
    
    for name, path in DATASETS.items():
        if os.path.exists(path):
            results = process_dataset(name, path)
            all_results[name] = results
            print(f"✓ {name}: {len(results)} records processed successfully")
        else:
            print(f"✗ {name}: path not found - skipping")
    
    print("\n" + "="*50)
    print("Preprocessing test complete!")
    print(f"Results saved to: {OUTPUT_DIR}")
    print("="*50)
