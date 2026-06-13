"""
Ensemble Evaluation
Phase 1 - Month 6
Combine ResNet1D and CNN-LSTM predictions
"""

import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
sys.path.append('scripts')
from lodo_loader import get_lodo_split, ECGDataset
from baseline3 import ResNet1D
from cnn_lstm import CNNLSTM

os.makedirs("outputs", exist_ok=True)

print("="*55)
print("Ensemble Evaluation")
print("ResNet1D + CNN-LSTM")
print("="*55)

def get_predictions(model, loader, device):
    model.eval()
    probs, labs = [], []
    with torch.no_grad():
        for x, y in loader:
            out = torch.softmax(model(x.to(device)), dim=1)
            probs.extend(out.cpu().numpy())
            labs.extend(y.numpy())
    return np.array(probs), np.array(labs)

def compute_metrics(probs_binary, labs_binary):
    preds = (probs_binary >= 0.5).astype(int)
    try:
        auroc = roc_auc_score(labs_binary, probs_binary)
        f1 = f1_score(labs_binary, preds, zero_division=0)
        sensitivity = recall_score(labs_binary, preds, zero_division=0)
        precision = precision_score(labs_binary, preds, zero_division=0)
    except:
        auroc = f1 = sensitivity = precision = 0.0
    return auroc, f1, sensitivity, precision

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Load test data
print("\nLoading test data (LODO-1: Hold out PTB-XL, AF)...")
X_train, y_train, X_test, y_test = get_lodo_split(
    held_out_dataset="ptbxl", target_class="AF", max_per_class=150)
test_loader = DataLoader(ECGDataset(X_test, y_test), batch_size=32)
print(f"Test: {len(X_test)} records")

# Binary labels for evaluation
y_binary = (y_test == 1).astype(int) if y_test.max() > 1 else y_test

# Load ResNet1D
print("\nLoading ResNet1D...")
resnet = ResNet1D(num_classes=3).to(device)
resnet.load_state_dict(torch.load("scripts/best_resnet.pth", map_location=device))
resnet_probs, resnet_labs = get_predictions(resnet, test_loader, device)
# AF probability from 3-class model
resnet_af_probs = resnet_probs[:, 0]
resnet_af_labs = (resnet_labs == 0).astype(int)

# Load CNN-LSTM
print("Loading CNN-LSTM...")
cnnlstm = CNNLSTM(n_leads=12, num_classes=2).to(device)
cnnlstm.load_state_dict(torch.load("scripts/best_cnn_lstm.pth", map_location=device))
cnnlstm_probs, cnnlstm_labs = get_predictions(cnnlstm, test_loader, device)
cnnlstm_af_probs = cnnlstm_probs[:, 1]
cnnlstm_af_labs = cnnlstm_labs.astype(int)

# Individual results
resnet_auroc, resnet_f1, resnet_sens, resnet_prec = compute_metrics(resnet_af_probs, resnet_af_labs)
cnnlstm_auroc, cnnlstm_f1, cnnlstm_sens, cnnlstm_prec = compute_metrics(cnnlstm_af_probs, cnnlstm_af_labs)

print(f"\nResNet1D  — AUROC: {resnet_auroc:.4f} | F1: {resnet_f1:.4f} | Sensitivity: {resnet_sens:.4f}")
print(f"CNN-LSTM  — AUROC: {cnnlstm_auroc:.4f} | F1: {cnnlstm_f1:.4f} | Sensitivity: {cnnlstm_sens:.4f}")

# ============================================================
# ENSEMBLE — Average Probabilities
# ============================================================
print("\nComputing Ensemble (Average)...")
ensemble_probs = (resnet_af_probs + cnnlstm_af_probs) / 2
# Use ResNet labs as reference (same test set)
ensemble_auroc, ensemble_f1, ensemble_sens, ensemble_prec = compute_metrics(ensemble_probs, resnet_af_labs)
print(f"Ensemble  — AUROC: {ensemble_auroc:.4f} | F1: {ensemble_f1:.4f} | Sensitivity: {ensemble_sens:.4f}")

# ============================================================
# RESULTS TABLE
# ============================================================
results = [
    {"Model": "ResNet1D", "AUROC": round(resnet_auroc, 4), "F1": round(resnet_f1, 4), "Sensitivity": round(resnet_sens, 4), "Precision": round(resnet_prec, 4)},
    {"Model": "CNN-LSTM", "AUROC": round(cnnlstm_auroc, 4), "F1": round(cnnlstm_f1, 4), "Sensitivity": round(cnnlstm_sens, 4), "Precision": round(cnnlstm_prec, 4)},
    {"Model": "Ensemble (Avg)", "AUROC": round(ensemble_auroc, 4), "F1": round(ensemble_f1, 4), "Sensitivity": round(ensemble_sens, 4), "Precision": round(ensemble_prec, 4)},
]
df = pd.DataFrame(results)
df.to_csv("outputs/ensemble_results.csv", index=False)

# ============================================================
# PLOT
# ============================================================
fig, ax = plt.subplots(figsize=(9, 5))
models = ["ResNet1D", "CNN-LSTM", "Ensemble"]
aurocs = [resnet_auroc, cnnlstm_auroc, ensemble_auroc]
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
print("\nFigure saved to outputs/Figure_Ensemble.png")

print("\n" + "="*55)
print("SUMMARY")
print("="*55)
print(df.to_string(index=False))
