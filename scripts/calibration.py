"""
Calibration Analysis
Phase 1 - Month 6
Evaluate probability calibration of ResNet1D and CNN-LSTM
Using Expected Calibration Error (ECE) and reliability diagrams
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import sys
sys.path.append('scripts')
from lodo_loader import get_lodo_split, ECGDataset
from baseline3 import ResNet1D
from cnn_lstm import CNNLSTM

os.makedirs("outputs", exist_ok=True)

print("="*55)
print("Calibration Analysis")
print("ResNet1D and CNN-LSTM")
print("="*55)

# ============================================================
# ECE CALCULATION
# ============================================================
def compute_ece(probs, labels, n_bins=10):
    """Expected Calibration Error"""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    bin_data = []

    for i in range(n_bins):
        low, high = bin_boundaries[i], bin_boundaries[i+1]
        mask = (probs >= low) & (probs < high)
        if mask.sum() == 0:
            bin_data.append((low + high) / 2, 0, 0, 0)
            continue
        bin_probs = probs[mask]
        bin_labels = labels[mask]
        bin_conf = bin_probs.mean()
        bin_acc = bin_labels.mean()
        bin_size = mask.sum()
        ece += (bin_size / len(probs)) * abs(bin_conf - bin_acc)
        bin_data.append(((low + high) / 2, bin_conf, bin_acc, bin_size))

    return ece, bin_data

def get_predictions(model, loader, device):
    model.eval()
    probs, labs = [], []
    with torch.no_grad():
        for x, y in loader:
            out = torch.softmax(model(x.to(device)), dim=1)
            probs.extend(out.cpu().numpy())
            labs.extend(y.numpy())
    return np.array(probs), np.array(labs)

# ============================================================
# MAIN
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Load test data
print("\nLoading test data (LODO-1: Hold out PTB-XL, AF)...")
X_train, y_train, X_test, y_test = get_lodo_split(
    held_out_dataset="ptbxl",
    target_class="AF",
    max_per_class=150
)
test_loader = DataLoader(ECGDataset(X_test, y_test), batch_size=32)
print(f"Test set: {len(X_test)} records")

results = []

# ============================================================
# ResNet1D
# ============================================================
print("\nLoading ResNet1D...")
resnet = ResNet1D(num_classes=3).to(device)
if os.path.exists("scripts/best_resnet.pth"):
    resnet.load_state_dict(torch.load("scripts/best_resnet.pth", map_location=device))
    print("ResNet1D loaded!")

resnet_probs, resnet_labs = get_predictions(resnet, test_loader, device)
# AF probability (class 0) vs rest
resnet_af_probs = resnet_probs[:, 0]
resnet_af_labs = (resnet_labs == 0).astype(int)
resnet_ece, resnet_bins = compute_ece(resnet_af_probs, resnet_af_labs)
print(f"ResNet1D ECE: {resnet_ece:.4f}")
results.append({"model": "ResNet1D", "ece": round(resnet_ece, 4)})

# ============================================================
# CNN-LSTM
# ============================================================
print("\nLoading CNN-LSTM...")
cnnlstm = CNNLSTM(n_leads=12, num_classes=2).to(device)
if os.path.exists("scripts/best_cnn_lstm.pth"):
    cnnlstm.load_state_dict(torch.load("scripts/best_cnn_lstm.pth", map_location=device))
    print("CNN-LSTM loaded!")

cnnlstm_probs, cnnlstm_labs = get_predictions(cnnlstm, test_loader, device)
cnnlstm_af_probs = cnnlstm_probs[:, 1]
cnnlstm_af_labs = cnnlstm_labs.astype(int)
cnnlstm_ece, cnnlstm_bins = compute_ece(cnnlstm_af_probs, cnnlstm_af_labs)
print(f"CNN-LSTM ECE: {cnnlstm_ece:.4f}")
results.append({"model": "CNN-LSTM", "ece": round(cnnlstm_ece, 4)})

# ============================================================
# PLOT — Reliability Diagrams
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, bins, ece, model_name, color in [
    (axes[0], resnet_bins, resnet_ece, "ResNet1D", "#2E4057"),
    (axes[1], cnnlstm_bins, cnnlstm_ece, "CNN-LSTM", "#E84855")
]:
    bin_centers = [b[0] for b in bins if b[3] > 0]
    bin_accs = [b[2] for b in bins if b[3] > 0]
    bin_confs = [b[1] for b in bins if b[3] > 0]

    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect calibration')
    ax.bar(bin_centers, bin_accs, width=0.08, alpha=0.7, color=color, label='Actual accuracy')
    ax.plot(bin_centers, bin_confs, 'o-', color='gray', markersize=6, label='Mean confidence')
    ax.set_xlabel('Confidence', fontsize=11)
    ax.set_ylabel('Accuracy', fontsize=11)
    ax.set_title(f'{model_name} — Reliability Diagram\nECE = {ece:.4f}', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.grid(alpha=0.3)

plt.suptitle('Calibration Analysis — AF Detection (PTB-XL Hold-out)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/Figure_Calibration.png', dpi=200, bbox_inches='tight')
print("\nFigure saved to outputs/Figure_Calibration.png")

# Save results
df = pd.DataFrame(results)
df.to_csv("outputs/calibration_results.csv", index=False)

print("\n" + "="*55)
print("SUMMARY — Calibration Results")
print("="*55)
print(df.to_string(index=False))
print("\nNote: Lower ECE = better calibrated")
