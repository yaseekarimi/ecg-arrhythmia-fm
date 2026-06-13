"""
Calibration Analysis - Fixed Version
Phase 1 - Month 6
"""

import os
import numpy as np
import torch
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
print("Calibration Analysis v2 — Fixed")
print("="*55)

def compute_ece(probs, labels, n_bins=10):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    bin_data = []
    for i in range(n_bins):
        low, high = bin_boundaries[i], bin_boundaries[i+1]
        mask = (probs >= low) & (probs < high)
        if mask.sum() == 0:
            bin_data.append(((low+high)/2, 0, 0, 0))
            continue
        bin_conf = probs[mask].mean()
        bin_acc = labels[mask].mean()
        bin_size = mask.sum()
        ece += (bin_size / len(probs)) * abs(bin_conf - bin_acc)
        bin_data.append(((low+high)/2, bin_conf, bin_acc, bin_size))
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

print("\nLoading test data...")
X_train, y_train, X_test, y_test = get_lodo_split(
    held_out_dataset="ptbxl", target_class="AF", max_per_class=150)
test_loader = DataLoader(ECGDataset(X_test, y_test), batch_size=32)
print(f"Test: {len(X_test)} records")

results = []

# ResNet1D — binary AF vs rest
print("\nResNet1D...")
resnet = ResNet1D(num_classes=3).to(device)
resnet.load_state_dict(torch.load("scripts/best_resnet.pth", map_location=device))
resnet_probs, resnet_labs = get_predictions(resnet, test_loader, device)
# Max confidence (most confident class)
resnet_max_conf = resnet_probs.max(axis=1)
resnet_correct = (resnet_probs.argmax(axis=1) == resnet_labs).astype(int)
resnet_ece, resnet_bins = compute_ece(resnet_max_conf, resnet_correct)
print(f"ResNet1D ECE: {resnet_ece:.4f}")
results.append({"model": "ResNet1D", "ece": round(resnet_ece, 4)})

# CNN-LSTM — binary
print("\nCNN-LSTM...")
cnnlstm = CNNLSTM(n_leads=12, num_classes=2).to(device)
cnnlstm.load_state_dict(torch.load("scripts/best_cnn_lstm.pth", map_location=device))
cnnlstm_probs, cnnlstm_labs = get_predictions(cnnlstm, test_loader, device)
cnnlstm_max_conf = cnnlstm_probs.max(axis=1)
cnnlstm_correct = (cnnlstm_probs.argmax(axis=1) == cnnlstm_labs).astype(int)
cnnlstm_ece, cnnlstm_bins = compute_ece(cnnlstm_max_conf, cnnlstm_correct)
print(f"CNN-LSTM ECE: {cnnlstm_ece:.4f}")
results.append({"model": "CNN-LSTM", "ece": round(cnnlstm_ece, 4)})

# Plot
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, bins, ece, model_name, color in [
    (axes[0], resnet_bins, resnet_ece, "ResNet1D", "#2E4057"),
    (axes[1], cnnlstm_bins, cnnlstm_ece, "CNN-LSTM", "#E84855")
]:
    valid = [(b[0], b[1], b[2], b[3]) for b in bins if b[3] > 0]
    if valid:
        centers = [b[0] for b in valid]
        accs = [b[2] for b in valid]
        confs = [b[1] for b in valid]
        ax.plot([0,1],[0,1],'k--',alpha=0.5,label='Perfect calibration')
        ax.bar(centers, accs, width=0.08, alpha=0.7, color=color, label='Actual accuracy')
        ax.plot(centers, confs, 'o-', color='gray', markersize=6, label='Mean confidence')
    ax.set_xlabel('Confidence', fontsize=11)
    ax.set_ylabel('Accuracy', fontsize=11)
    ax.set_title(f'{model_name}\nECE = {ece:.4f}', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.set_xlim([0,1]); ax.set_ylim([0,1])
    ax.grid(alpha=0.3)

plt.suptitle('Calibration Analysis — AF Detection (PTB-XL Hold-out)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/Figure_Calibration_v2.png', dpi=200, bbox_inches='tight')
print("\nFigure saved!")

df = pd.DataFrame(results)
df.to_csv("outputs/calibration_results.csv", index=False)
print("\n" + "="*55)
print(df.to_string(index=False))
print("Lower ECE = better calibrated")
