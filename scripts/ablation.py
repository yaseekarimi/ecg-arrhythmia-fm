"""
Ablation Study
Phase 1 - Month 7
Test impact of key pipeline components on ResNet1D performance
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import wfdb
import pandas as pd
import ast
from scipy import signal as scipy_signal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
sys.path.append('scripts')
from baseline3 import ResNet1D

PTBXL_PATH = "data/ptbxl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
TARGET_FS = 100
TARGET_LEN = 1000
BATCH_SIZE = 32
EPOCHS = 10
LR = 0.001
MAX_PER_CLASS = 150

os.makedirs("outputs", exist_ok=True)

print("="*55)
print("Ablation Study — ResNet1D on PTB-XL")
print("="*55)

class ECGDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]

def load_ptbxl(path, max_per_class=MAX_PER_CLASS,
               do_resample=True, do_normalize=True, balanced=True):
    csv_path = os.path.join(path, "ptbxl_database.csv")
    df = pd.read_csv(csv_path)
    label_map = {}
    for _, row in df.iterrows():
        scp = ast.literal_eval(row['scp_codes'])
        if any(k in ['AFIB', 'AFLT'] for k in scp.keys()):
            label_map[row['filename_lr']] = 1
        else:
            label_map[row['filename_lr']] = 0

    pos_keys = [k for k, v in label_map.items() if v == 1][:max_per_class]
    neg_keys = [k for k, v in label_map.items() if v == 0]

    if balanced:
        neg_keys = neg_keys[:max_per_class]
    else:
        neg_keys = neg_keys[:max_per_class * 5]  # Imbalanced

    selected = pos_keys + neg_keys
    signals, labels = [], []

    for filename in selected:
        rec_path = os.path.join(path, filename)
        try:
            record = wfdb.rdrecord(rec_path)
            sig = record.p_signal
            fs = record.fs

            if do_resample and fs != TARGET_FS:
                num = int(len(sig) * TARGET_FS / fs)
                sig = scipy_signal.resample(sig, num)

            if len(sig) >= TARGET_LEN:
                sig = sig[:TARGET_LEN]
            else:
                sig = np.pad(sig, ((0, TARGET_LEN - len(sig)), (0, 0)))

            sig = sig[:, :12]

            if do_normalize:
                mean = np.mean(sig, axis=0)
                std = np.std(sig, axis=0)
                std[std == 0] = 1
                sig = (sig - mean) / std

            signals.append(sig.T.astype(np.float32))
            labels.append(label_map[filename])
        except:
            continue

    return np.array(signals), np.array(labels)

def train_eval(X_train, y_train, X_test, y_test, device):
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42)
    tr_loader = DataLoader(ECGDataset(X_tr, y_tr), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(ECGDataset(X_test, y_test), batch_size=BATCH_SIZE)

    model = ResNet1D(num_classes=2).to(device)
    opt = optim.Adam(model.parameters(), lr=LR)
    crit = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(EPOCHS):
        for x, y in tr_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()

    model.eval()
    probs, labs = [], []
    with torch.no_grad():
        for x, y in test_loader:
            out = torch.softmax(model(x.to(device)), dim=1)
            probs.extend(out.cpu().numpy())
            labs.extend(y.numpy())
    probs = np.array(probs)
    labs = np.array(labs)
    try:
        return roc_auc_score(labs, probs[:, 1])
    except:
        return 0.0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

results = []

# ============================================================
# 1. Full Pipeline (Baseline)
# ============================================================
print("\n[1/4] Full Pipeline (Baseline)...")
X, y = load_ptbxl(PTBXL_PATH, do_resample=True, do_normalize=True, balanced=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
auroc = train_eval(X_tr, y_tr, X_te, y_te, device)
print(f"  AUROC: {auroc:.4f}")
results.append({"Condition": "Full Pipeline (Baseline)", "AUROC": round(auroc, 4), "Change": "—"})

baseline_auroc = auroc

# ============================================================
# 2. Without Z-score Normalisation
# ============================================================
print("\n[2/4] Without Z-score Normalisation...")
X2, y2 = load_ptbxl(PTBXL_PATH, do_resample=True, do_normalize=False, balanced=True)
X_tr2, X_te2, y_tr2, y_te2 = train_test_split(X2, y2, test_size=0.2, random_state=42, stratify=y2)
auroc2 = train_eval(X_tr2, y_tr2, X_te2, y_te2, device)
print(f"  AUROC: {auroc2:.4f}")
results.append({"Condition": "Without Z-score Normalisation", "AUROC": round(auroc2, 4),
                "Change": f"{auroc2 - baseline_auroc:+.4f}"})

# ============================================================
# 3. Without Balanced Sampling
# ============================================================
print("\n[3/4] Without Balanced Sampling (Imbalanced)...")
X3, y3 = load_ptbxl(PTBXL_PATH, do_resample=True, do_normalize=True, balanced=False)
X_tr3, X_te3, y_tr3, y_te3 = train_test_split(X3, y3, test_size=0.2, random_state=42)
auroc3 = train_eval(X_tr3, y_tr3, X_te3, y_te3, device)
print(f"  AUROC: {auroc3:.4f}")
results.append({"Condition": "Without Balanced Sampling", "AUROC": round(auroc3, 4),
                "Change": f"{auroc3 - baseline_auroc:+.4f}"})

# ============================================================
# 4. Single Dataset (PTB-XL only — no LODO)
# ============================================================
print("\n[4/4] Single Dataset Only (PTB-XL, no LODO)...")
auroc4 = train_eval(X_tr, y_tr, X_te, y_te, device)
print(f"  AUROC: {auroc4:.4f}")
results.append({"Condition": "Single Dataset (PTB-XL only)", "AUROC": round(auroc4, 4),
                "Change": f"{auroc4 - baseline_auroc:+.4f}"})

# ============================================================
# PLOT
# ============================================================
df = pd.DataFrame(results)
df.to_csv("outputs/ablation_results.csv", index=False)

fig, ax = plt.subplots(figsize=(10, 5))
conditions = df['Condition']
aurocs = df['AUROC']
colors = ['#2E4057', '#E84855', '#F4A261', '#2E75B6']

bars = ax.barh(conditions, aurocs, color=colors, height=0.5, edgecolor='white')
ax.set_xlim([0.5, 1.05])
ax.set_xlabel('Test AUROC', fontsize=12)
ax.set_title('Figure 5.9. Ablation Study — Impact of Pipeline Components\non ResNet1D AF Detection Performance', 
             fontsize=12, fontweight='bold')
ax.axvline(x=baseline_auroc, color='#2E4057', linestyle='--', alpha=0.5, label=f'Baseline ({baseline_auroc:.4f})')
ax.legend(fontsize=10)
ax.grid(axis='x', alpha=0.3)

for bar, auroc, change in zip(bars, aurocs, df['Change']):
    ax.text(auroc + 0.005, bar.get_y() + bar.get_height()/2,
            f'{auroc:.4f} ({change})', va='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('outputs/Figure5_9_Ablation.png', dpi=200, bbox_inches='tight')
print("\nFigure saved!")

print("\n" + "="*55)
print(df.to_string(index=False))
