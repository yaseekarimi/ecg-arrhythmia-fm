"""
Ensemble Evaluation - Fixed Version
Phase 1 - Month 6
"""

import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
sys.path.append('scripts')
from lodo_loader import get_lodo_split
from baseline3 import ResNet1D
from cnn_lstm import CNNLSTM

os.makedirs("outputs", exist_ok=True)

print("="*55)
print("Ensemble Evaluation v2 — Fixed")
print("="*55)

class ECGDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]

def compute_metrics(probs, labs):
    preds = (probs >= 0.5).astype(int)
    try:
        auroc = roc_auc_score(labs, probs)
        f1 = f1_score(labs, preds, zero_division=0)
        sens = recall_score(labs, preds, zero_division=0)
        prec = precision_score(labs, preds, zero_division=0)
    except:
        auroc = f1 = sens = prec = 0.0
    return auroc, f1, sens, prec

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Load data — binary AF classification
print("\nLoading test data...")
X_train, y_train, X_test, y_test = get_lodo_split(
    held_out_dataset="ptbxl", target_class="AF", max_per_class=150)

# y_test is binary (0=negative, 1=positive) from lodo_loader
test_loader = DataLoader(ECGDataset(X_test, y_test), batch_size=32)
print(f"Test: {len(X_test)} — Positive: {y_test.sum()}, Negative: {(y_test==0).sum()}")

# ============================================================
# ResNet1D — retrain binary for fair comparison
# ============================================================
print("\nTraining ResNet1D (binary) for ensemble...")
import torch.optim as optim

# Train binary ResNet on same split
X_tr, y_tr = X_train, y_train
tr_loader = DataLoader(ECGDataset(X_tr, y_tr), batch_size=32, shuffle=True)

resnet_bin = ResNet1D(num_classes=2).to(device)
opt = optim.Adam(resnet_bin.parameters(), lr=0.001)
crit = torch.nn.CrossEntropyLoss()
resnet_bin.train()
for epoch in range(10):
    for x, y in tr_loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        loss = crit(resnet_bin(x), y)
        loss.backward()
        opt.step()
print("ResNet1D (binary) trained!")

# Get predictions
resnet_bin.eval()
resnet_probs, resnet_labs = [], []
with torch.no_grad():
    for x, y in test_loader:
        out = torch.softmax(resnet_bin(x.to(device)), dim=1)
        resnet_probs.extend(out.cpu().numpy())
        resnet_labs.extend(y.numpy())
resnet_probs = np.array(resnet_probs)
resnet_labs = np.array(resnet_labs)
resnet_af = resnet_probs[:, 1]

# ============================================================
# CNN-LSTM
# ============================================================
print("\nLoading CNN-LSTM...")
cnnlstm = CNNLSTM(n_leads=12, num_classes=2).to(device)
cnnlstm.load_state_dict(torch.load("scripts/best_cnn_lstm.pth", map_location=device))
cnnlstm.eval()
cnnlstm_probs, cnnlstm_labs = [], []
with torch.no_grad():
    for x, y in test_loader:
        out = torch.softmax(cnnlstm(x.to(device)), dim=1)
        cnnlstm_probs.extend(out.cpu().numpy())
        cnnlstm_labs.extend(y.numpy())
cnnlstm_probs = np.array(cnnlstm_probs)
cnnlstm_af = cnnlstm_probs[:, 1]
labs = np.array(cnnlstm_labs)

# ============================================================
# METRICS
# ============================================================
resnet_auroc, resnet_f1, resnet_sens, resnet_prec = compute_metrics(resnet_af, labs)
cnnlstm_auroc, cnnlstm_f1, cnnlstm_sens, cnnlstm_prec = compute_metrics(cnnlstm_af, labs)

# Ensemble
ensemble_af = (resnet_af + cnnlstm_af) / 2
ens_auroc, ens_f1, ens_sens, ens_prec = compute_metrics(ensemble_af, labs)

print(f"\nResNet1D     — AUROC: {resnet_auroc:.4f} | F1: {resnet_f1:.4f} | Sensitivity: {resnet_sens:.4f}")
print(f"CNN-LSTM     — AUROC: {cnnlstm_auroc:.4f} | F1: {cnnlstm_f1:.4f} | Sensitivity: {cnnlstm_sens:.4f}")
print(f"Ensemble     — AUROC: {ens_auroc:.4f} | F1: {ens_f1:.4f} | Sensitivity: {ens_sens:.4f}")

results = [
    {"Model": "ResNet1D", "AUROC": round(resnet_auroc,4), "F1": round(resnet_f1,4), "Sensitivity": round(resnet_sens,4), "Precision": round(resnet_prec,4)},
    {"Model": "CNN-LSTM", "AUROC": round(cnnlstm_auroc,4), "F1": round(cnnlstm_f1,4), "Sensitivity": round(cnnlstm_sens,4), "Precision": round(cnnlstm_prec,4)},
    {"Model": "Ensemble", "AUROC": round(ens_auroc,4), "F1": round(ens_f1,4), "Sensitivity": round(ens_sens,4), "Precision": round(ens_prec,4)},
]
df = pd.DataFrame(results)
df.to_csv("outputs/ensemble_results.csv", index=False)

# Plot
fig, ax = plt.subplots(figsize=(9, 5))
models = ["ResNet1D", "CNN-LSTM", "Ensemble"]
aurocs = [resnet_auroc, cnnlstm_auroc, ens_auroc]
colors = ['#2E4057', '#E84855', '#2E75B6']

bars = ax.bar(models, aurocs, color=colors, width=0.4, edgecolor='white')
ax.set_ylim([0.5, 1.05])
ax.set_ylabel('Test Macro AUROC', fontsize=12)
ax.set_title('Ensemble vs Individual Models\nAF Detection — PTB-XL Hold-out', fontsize=13, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
for bar, auroc in zip(bars, aurocs):
    ax.text(bar.get_x() + bar.get_width()/2, auroc + 0.008,
            f'{auroc:.4f}', ha='center', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('outputs/Figure_Ensemble.png', dpi=200, bbox_inches='tight')
print("\nFigure saved!")

print("\n" + "="*55)
print(df.to_string(index=False))
