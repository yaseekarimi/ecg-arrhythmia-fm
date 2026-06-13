"""
Subgroup Analysis - Fixed Version
Phase 1 - Month 5
"""

import os
import numpy as np
import torch
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

PTBXL_PATH = "data/ptbxl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
TARGET_FS = 100
TARGET_LEN = 1000
BATCH_SIZE = 32

os.makedirs("outputs", exist_ok=True)

print("="*55)
print("Subgroup Analysis — ResNet1D on PTB-XL")
print("Stratified by Age Group and Sex")
print("="*55)

def load_ptbxl_with_metadata(path, max_records=1000):
    csv_path = os.path.join(path, "ptbxl_database.csv")
    df = pd.read_csv(csv_path)
    signals, labels, ages, sexes = [], [], [], []

    for _, row in df.iterrows():
        if len(signals) >= max_records:
            break
        try:
            scp = ast.literal_eval(row['scp_codes'])
            # AF=0, PVC=1, Other=2 (3 classes matching saved model)
            if any(k in ['AFIB', 'AFLT'] for k in scp.keys()):
                label = 0
            elif any(k in ['PVC', 'SVPB'] for k in scp.keys()):
                label = 1
            else:
                label = 2

            rec_path = os.path.join(path, row['filename_lr'])
            record = wfdb.rdrecord(rec_path)
            sig = record.p_signal

            if record.fs != TARGET_FS:
                num = int(len(sig) * TARGET_FS / record.fs)
                sig = scipy_signal.resample(sig, num)

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
            ages.append(float(row['age']) if pd.notna(row['age']) else 50.0)
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
    if len(X) < 10:
        return None
    unique = np.unique(y)
    if len(unique) < 2:
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
    # AF vs rest (binary)
    af_probs = probs[:, 0]
    af_labs = (labs == 0).astype(int)
    try:
        return roc_auc_score(af_labs, af_probs)
    except:
        return None

def fmt(val):
    return f"{val:.4f}" if val is not None else "N/A"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

print("\nLoading ResNet1D model...")
model = ResNet1D(num_classes=3).to(device)
if os.path.exists("scripts/best_resnet.pth"):
    model.load_state_dict(torch.load("scripts/best_resnet.pth", map_location=device))
    print("Model loaded!")

print("\nLoading PTB-XL with metadata...")
X, y, ages, sexes = load_ptbxl_with_metadata(PTBXL_PATH, max_records=1000)
print(f"Loaded {len(X)} records")
print(f"AF: {(y==0).sum()} | PVC: {(y==1).sum()} | Other: {(y==2).sum()}")
print(f"Male: {(sexes==0).sum()} | Female: {(sexes==1).sum()}")

results = []

# By Sex
print("\n--- Analysis by Sex ---")
for sex_val, sex_name in [(0, "Male"), (1, "Female")]:
    idx = np.where(sexes == sex_val)[0]
    X_sub, y_sub = X[idx], y[idx]
    auroc = evaluate_auroc(model, X_sub, y_sub, device)
    n_af = (y_sub == 0).sum()
    print(f"  {sex_name}: n={len(X_sub)}, AF={n_af}, AUROC={fmt(auroc)}")
    results.append({"group_type": "Sex", "group": sex_name, "n": len(X_sub), "n_af": int(n_af), "auroc": auroc})

# By Age
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
    n_af = (y_sub == 0).sum()
    print(f"  {group_name}: n={len(X_sub)}, AF={n_af}, AUROC={fmt(auroc)}")
    results.append({"group_type": "Age", "group": group_name, "n": len(X_sub), "n_af": int(n_af), "auroc": auroc})

# Overall
overall_auroc = evaluate_auroc(model, X, y, device)
print(f"\n  Overall: n={len(X)}, AF={(y==0).sum()}, AUROC={fmt(overall_auroc)}")
results.append({"group_type": "Overall", "group": "All", "n": len(X), "n_af": int((y==0).sum()), "auroc": overall_auroc})

df_results = pd.DataFrame(results)
df_results.to_csv("outputs/subgroup_results.csv", index=False)

# Plot
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sex_data = df_results[df_results['group_type'] == 'Sex'].dropna(subset=['auroc'])
if len(sex_data) > 0:
    axes[0].bar(sex_data['group'], sex_data['auroc'], color=['#2E4057', '#E84855'], width=0.5)
    if overall_auroc:
        axes[0].axhline(y=overall_auroc, color='gray', linestyle='--', alpha=0.7, label=f'Overall ({overall_auroc:.3f})')
    axes[0].set_title('AUROC by Sex — AF Detection', fontsize=13, fontweight='bold')
    axes[0].set_ylabel('AUROC (AF vs Rest)', fontsize=11)
    axes[0].set_ylim([0.5, 1.0])
    axes[0].legend(fontsize=10)
    axes[0].grid(axis='y', alpha=0.3)
    for i, (_, row) in enumerate(sex_data.iterrows()):
        axes[0].text(i, row['auroc'] + 0.01, f"{row['auroc']:.3f}\n(n={row['n']})", ha='center', fontsize=10)

age_data = df_results[df_results['group_type'] == 'Age'].dropna(subset=['auroc'])
if len(age_data) > 0:
    axes[1].bar(range(len(age_data)), age_data['auroc'], color=['#2E75B6', '#2E4057', '#1F4E79'], width=0.5)
    if overall_auroc:
        axes[1].axhline(y=overall_auroc, color='gray', linestyle='--', alpha=0.7, label=f'Overall ({overall_auroc:.3f})')
    axes[1].set_xticks(range(len(age_data)))
    axes[1].set_xticklabels(age_data['group'], fontsize=9)
    axes[1].set_title('AUROC by Age Group — AF Detection', fontsize=13, fontweight='bold')
    axes[1].set_ylabel('AUROC (AF vs Rest)', fontsize=11)
    axes[1].set_ylim([0.5, 1.0])
    axes[1].legend(fontsize=10)
    axes[1].grid(axis='y', alpha=0.3)
    for i, (_, row) in enumerate(age_data.iterrows()):
        axes[1].text(i, row['auroc'] + 0.01, f"{row['auroc']:.3f}\n(n={row['n']})", ha='center', fontsize=10)

plt.suptitle('ResNet1D Subgroup Analysis — PTB-XL', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/Figure_SubgroupAnalysis.png', dpi=200, bbox_inches='tight')
print("\nFigure saved to outputs/Figure_SubgroupAnalysis.png")
print("\n" + "="*55)
print(df_results.to_string(index=False))
