"""
Subgroup Analysis
Phase 1 - Month 5
Evaluate ResNet1D performance stratified by age and sex
Using PTB-XL metadata
"""

import os
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import ast
import wfdb
from scipy import signal as scipy_signal
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
sys.path.append('scripts')
from baseline3 import ResNet1D

# ============================================================
# CONFIG
# ============================================================
PTBXL_PATH = "data/ptbxl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
TARGET_FS = 100
TARGET_LEN = 1000
BATCH_SIZE = 32
MAX_PER_GROUP = 100  # Max records per subgroup

os.makedirs("outputs", exist_ok=True)

print("="*55)
print("Subgroup Analysis — ResNet1D on PTB-XL")
print("Stratified by Age Group and Sex")
print("="*55)

# ============================================================
# LOAD DATA WITH METADATA
# ============================================================
def load_ptbxl_with_metadata(path, max_records=1000):
    csv_path = os.path.join(path, "ptbxl_database.csv")
    df = pd.read_csv(csv_path)

    signals, labels, ages, sexes = [], [], [], []

    for _, row in df.iterrows():
        if len(signals) >= max_records:
            break
        try:
            scp = ast.literal_eval(row['scp_codes'])
            label = 1 if any(k in ['AFIB', 'AFLT'] for k in scp.keys()) else 0

            rec_path = os.path.join(path, row['filename_lr'])
            record = wfdb.rdrecord(rec_path)
            sig = record.p_signal

            # Resample
            if record.fs != TARGET_FS:
                num = int(len(sig) * TARGET_FS / record.fs)
                sig = scipy_signal.resample(sig, num)

            # Fix length
            if len(sig) >= TARGET_LEN:
                sig = sig[:TARGET_LEN]
            else:
                sig = np.pad(sig, ((0, TARGET_LEN - len(sig)), (0, 0)))

            sig = sig[:, :12]
            mean = np.mean(sig, axis=0)
            std = np.std(sig, axis=0)
            std[std == 0] = 1
            sig = (sig - mean) / std

            signals.append(sig.T.astype(np.float32))
            labels.append(label)
            ages.append(row['age'] if pd.notna(row['age']) else 50)
            sexes.append(int(row['sex']) if pd.notna(row['sex']) else 0)

        except:
            continue

    return np.array(signals), np.array(labels), np.array(ages), np.array(sexes)

class ECGDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]

def evaluate_auroc(model, X, y, device):
    if len(X) == 0 or len(np.unique(y)) < 2:
        return None
    loader = DataLoader(ECGDataset(X, y), batch_size=BATCH_SIZE)
    model.eval()
    probs, labs = [], []
    with torch.no_grad():
        for x, lbl in loader:
            out = torch.softmax(model(x.to(device)), dim=1)
            probs.extend(out.cpu().numpy())
            labs.extend(lbl.numpy())
    probs = np.array(probs)
    labs = np.array(labs)
    try:
        oh = np.zeros((len(labs), 3))
        for i, l in enumerate(labs): oh[i, l] = 1
        return roc_auc_score(oh, probs, average="macro")
    except:
        return None

# ============================================================
# MAIN
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Load model
print("\nLoading ResNet1D model...")
model = ResNet1D(num_classes=3).to(device)
if os.path.exists("scripts/best_resnet.pth"):
    model.load_state_dict(torch.load("scripts/best_resnet.pth", map_location=device))
    print("Model loaded!")

# Load data with metadata
print("\nLoading PTB-XL with metadata...")
X, y, ages, sexes = load_ptbxl_with_metadata(PTBXL_PATH, max_records=800)
print(f"Loaded {len(X)} records")
print(f"AF positive: {y.sum()} | Negative: {(y==0).sum()}")
print(f"Male: {(sexes==0).sum()} | Female: {(sexes==1).sum()}")

results = []

# ============================================================
# 1. BY SEX
# ============================================================
print("\n--- Analysis by Sex ---")
for sex_val, sex_name in [(0, "Male"), (1, "Female")]:
    idx = np.where(sexes == sex_val)[0]
    X_sub, y_sub = X[idx], y[idx]
    auroc = evaluate_auroc(model, X_sub, y_sub, device)
    n_pos = y_sub.sum()
    auroc_str = f"{auroc:.4f}" if auroc is not None else "N/A"
    print(f"  {sex_name}: n={len(X_sub)}, AF={n_pos}, AUROC={auroc_str}")
    results.append({"group_type": "Sex", "group": sex_name, "n": len(X_sub), "n_af": int(n_pos), "auroc": round(auroc, 4) if auroc else None})

# ============================================================
# 2. BY AGE GROUP
# ============================================================
print("\n--- Analysis by Age Group ---")
age_groups = [
    ("Young (<40)", ages < 40),
    ("Middle (40-60)", (ages >= 40) & (ages < 60)),
    ("Elderly (>60)", ages >= 60)
]

for group_name, mask in age_groups:
    idx = np.where(mask)[0]
    X_sub, y_sub = X[idx], y[idx]
    auroc = evaluate_auroc(model, X_sub, y_sub, device)
    n_pos = y_sub.sum()
    auroc_str = f"{auroc:.4f}" if auroc is not None else "N/A"
    print(f"  {group_name}: n={len(X_sub)}, AF={n_pos}, AUROC={auroc_str}")
    results.append({"group_type": "Age", "group": group_name, "n": len(X_sub), "n_af": int(n_pos), "auroc": round(auroc, 4) if auroc else None})

# ============================================================
# 3. OVERALL
# ============================================================
overall_auroc = evaluate_auroc(model, X, y, device)
print(f"\n  Overall: n={len(X)}, AF={y.sum()}, AUROC={overall_auroc:.4f if overall_auroc else 'N/A'}")
results.append({"group_type": "Overall", "group": "All", "n": len(X), "n_af": int(y.sum()), "auroc": round(overall_auroc, 4) if overall_auroc else None})

# Save results
df_results = pd.DataFrame(results)
df_results.to_csv("outputs/subgroup_results.csv", index=False)

# ============================================================
# PLOT
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# By Sex
sex_data = df_results[df_results['group_type'] == 'Sex']
axes[0].bar(sex_data['group'], sex_data['auroc'], color=['#2E4057', '#E84855'], width=0.5, edgecolor='white')
axes[0].axhline(y=overall_auroc, color='gray', linestyle='--', alpha=0.7, label=f'Overall AUROC ({overall_auroc:.3f})')
axes[0].set_title('AUROC by Sex', fontsize=13, fontweight='bold')
axes[0].set_ylabel('Test Macro AUROC', fontsize=11)
axes[0].set_ylim([0.5, 1.0])
axes[0].legend(fontsize=10)
axes[0].grid(axis='y', alpha=0.3)
for i, (_, row) in enumerate(sex_data.iterrows()):
    axes[0].text(i, row['auroc'] + 0.01, f"{row['auroc']:.3f}\n(n={row['n']})", ha='center', fontsize=10)

# By Age
age_data = df_results[df_results['group_type'] == 'Age']
axes[1].bar(range(len(age_data)), age_data['auroc'], color=['#2E75B6', '#2E4057', '#1F4E79'], width=0.5, edgecolor='white')
axes[1].axhline(y=overall_auroc, color='gray', linestyle='--', alpha=0.7, label=f'Overall AUROC ({overall_auroc:.3f})')
axes[1].set_xticks(range(len(age_data)))
axes[1].set_xticklabels(age_data['group'], fontsize=9)
axes[1].set_title('AUROC by Age Group', fontsize=13, fontweight='bold')
axes[1].set_ylabel('Test Macro AUROC', fontsize=11)
axes[1].set_ylim([0.5, 1.0])
axes[1].legend(fontsize=10)
axes[1].grid(axis='y', alpha=0.3)
for i, (_, row) in enumerate(age_data.iterrows()):
    axes[1].text(i, row['auroc'] + 0.01, f"{row['auroc']:.3f}\n(n={row['n']})", ha='center', fontsize=10)

plt.suptitle('ResNet1D Subgroup Analysis — AF Detection (PTB-XL)', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('outputs/Figure_SubgroupAnalysis.png', dpi=200, bbox_inches='tight')
print("\nFigure saved to outputs/Figure_SubgroupAnalysis.png")

print("\n" + "="*55)
print("SUMMARY")
print("="*55)
print(df_results.to_string(index=False))
