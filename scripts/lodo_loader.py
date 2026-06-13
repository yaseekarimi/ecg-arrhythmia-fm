"""
LODO (Leave-One-Dataset-Out) Data Loader
Phase 1 - Month 4
Prepares train/test splits for all 5 LODO configurations
"""

import os
import numpy as np
import wfdb
import pandas as pd
import ast
from scipy import signal as scipy_signal
from torch.utils.data import Dataset
import torch

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

TARGET_FS = 100
TARGET_LEN = 1000
MAX_PER_CLASS = 200  # Max records per class per dataset (for testing)

AF_CODES = {"ptbxl": ["AFIB", "AFLT"], "other": ["AF", "AFIB", "164889003"]}
PVC_CODES = {"ptbxl": ["PVC", "SVPB"], "other": ["PVC", "164884008", "17338001"]}

# ============================================================
# SIGNAL PROCESSING
# ============================================================
def process_signal(sig, fs, n_leads=12):
    """Resample, fix length, z-score normalise"""
    if fs != TARGET_FS:
        num = int(len(sig) * TARGET_FS / fs)
        sig = scipy_signal.resample(sig, num)
    if len(sig) >= TARGET_LEN:
        sig = sig[:TARGET_LEN]
    else:
        sig = np.pad(sig, ((0, TARGET_LEN - len(sig)), (0, 0)))
    
    # Handle different number of leads
    if sig.shape[1] >= n_leads:
        sig = sig[:, :n_leads]
    else:
        # Pad leads with zeros if less than n_leads
        pad = np.zeros((TARGET_LEN, n_leads - sig.shape[1]))
        sig = np.concatenate([sig, pad], axis=1)
    
    # Z-score per lead
    mean = np.mean(sig, axis=0)
    std = np.std(sig, axis=0)
    std[std == 0] = 1
    return (sig - mean) / std

# ============================================================
# LABEL EXTRACTION
# ============================================================
def get_label_ptbxl(path, target_class="AF"):
    """Load labels from PTB-XL CSV"""
    csv_path = os.path.join(path, "ptbxl_database.csv")
    df = pd.read_csv(csv_path)
    label_map = {}
    for _, row in df.iterrows():
        filename = row['filename_lr']
        scp = ast.literal_eval(row['scp_codes'])
        if target_class == "AF":
            label = 1 if any(k in AF_CODES["ptbxl"] for k in scp.keys()) else 0
        else:
            label = 1 if any(k in PVC_CODES["ptbxl"] for k in scp.keys()) else 0
        label_map[filename] = label
    return label_map

def get_label_from_header(header, target_class="AF"):
    """Extract label from wfdb header comments"""
    codes = AF_CODES["other"] if target_class == "AF" else PVC_CODES["other"]
    for comment in header.comments:
        if any(code in str(comment) for code in codes):
            return 1
    return 0

# ============================================================
# DATASET LOADERS
# ============================================================
def load_dataset(name, path, target_class="AF", max_per_class=MAX_PER_CLASS):
    """Load a single dataset"""
    print(f"  Loading {name} ({target_class})...")
    signals, labels = [], []

    if name == "ptbxl":
        label_map = get_label_ptbxl(path, target_class)
        pos_keys = [k for k, v in label_map.items() if v == 1][:max_per_class]
        neg_keys = [k for k, v in label_map.items() if v == 0][:max_per_class]
        selected = pos_keys + neg_keys
        for filename in selected:
            rec_path = os.path.join(path, filename)
            try:
                record = wfdb.rdrecord(rec_path)
                sig = process_signal(record.p_signal, record.fs)
                signals.append(sig.T.astype(np.float32))
                labels.append(label_map[filename])
            except:
                continue
    else:
        hea_files = []
        for root, dirs, files in os.walk(path):
            for f in files:
                if f.endswith('.hea'):
                    hea_files.append(os.path.join(root, f[:-4]))
        
        pos_files, neg_files = [], []
        for rec_path in hea_files:
            try:
                header = wfdb.rdheader(rec_path)
                label = get_label_from_header(header, target_class)
                if label == 1:
                    pos_files.append(rec_path)
                else:
                    neg_files.append(rec_path)
            except:
                continue
        
        selected = pos_files[:max_per_class] + neg_files[:max_per_class]
        for rec_path in selected:
            try:
                record = wfdb.rdrecord(rec_path)
                header = wfdb.rdheader(rec_path)
                sig = process_signal(record.p_signal, record.fs)
                label = get_label_from_header(header, target_class)
                signals.append(sig.T.astype(np.float32))
                labels.append(label)
            except:
                continue

    signals = np.array(signals)
    labels = np.array(labels)
    pos = (labels == 1).sum()
    neg = (labels == 0).sum()
    print(f"    {name}: {len(signals)} records — Positive: {pos}, Negative: {neg}")
    return signals, labels

# ============================================================
# LODO SPLITS
# ============================================================
def get_lodo_split(held_out_dataset, target_class="AF", max_per_class=MAX_PER_CLASS):
    """
    Returns train and test splits for LODO evaluation.
    held_out_dataset: name of dataset to hold out for testing
    """
    print(f"\nLODO Split — Hold out: {held_out_dataset} | Target: {target_class}")
    print("-" * 50)

    train_signals, train_labels = [], []
    test_signals, test_labels = None, None

    for name, path in DATASETS.items():
        if not os.path.exists(path):
            print(f"  Skipping {name} — path not found")
            continue

        signals, labels = load_dataset(name, path, target_class, max_per_class)

        if name == held_out_dataset:
            test_signals = signals
            test_labels = labels
        else:
            train_signals.append(signals)
            train_labels.append(labels)

    train_signals = np.concatenate(train_signals, axis=0)
    train_labels = np.concatenate(train_labels, axis=0)

    print(f"\nTrain: {len(train_signals)} records")
    print(f"Test:  {len(test_signals)} records (held out: {held_out_dataset})")

    return train_signals, train_labels, test_signals, test_labels

# ============================================================
# PYTORCH DATASET
# ============================================================
class ECGDataset(Dataset):
    def __init__(self, signals, labels):
        self.X = torch.tensor(signals, dtype=torch.float32)
        self.y = torch.tensor(labels, dtype=torch.long)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]

# ============================================================
# TEST
# ============================================================
if __name__ == "__main__":
    print("="*50)
    print("LODO Loader Test")
    print("="*50)

    # Test one LODO split
    X_train, y_train, X_test, y_test = get_lodo_split(
        held_out_dataset="ptbxl",
        target_class="AF",
        max_per_class=50  # Small for quick test
    )

    print(f"\nTrain shape: {X_train.shape}")
    print(f"Test shape:  {X_test.shape}")
    print(f"Train positive: {(y_train==1).sum()}, negative: {(y_train==0).sum()}")
    print(f"Test positive:  {(y_test==1).sum()}, negative:  {(y_test==0).sum()}")

    print("\n" + "="*50)
    print("LODO Loader ready!")
    print("="*50)
