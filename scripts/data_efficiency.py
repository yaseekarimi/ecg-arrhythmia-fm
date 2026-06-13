"""
Data Efficiency Analysis
Phase 1 - Month 5
Compare ResNet1D vs ECG-FM across different training data percentages
Shows how much data each model needs to achieve good performance
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
sys.path.append('scripts')
from lodo_loader import get_lodo_split, ECGDataset
from baseline3 import ResNet1D

# ============================================================
# CONFIG
# ============================================================
PERCENTAGES = [0.01, 0.05, 0.10, 0.25, 0.50, 1.00]  # 1%, 5%, 10%, 25%, 50%, 100%
BATCH_SIZE = 32
EPOCHS = 10
LR = 0.001
MAX_PER_CLASS = 200
TARGET_CLASS = "AF"
HELD_OUT = "ptbxl"

os.makedirs("outputs", exist_ok=True)
RESULTS_FILE = "outputs/data_efficiency_results.csv"

print("="*55)
print("Data Efficiency Analysis")
print(f"Target: {TARGET_CLASS} | Hold out: {HELD_OUT}")
print(f"Percentages: {[f'{p*100:.0f}%' for p in PERCENTAGES]}")
print("="*55)

# ============================================================
# TRAIN & EVAL
# ============================================================
def train_model(model, train_loader, device, epochs=EPOCHS):
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(epochs):
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
    return model

def evaluate(model, loader, device):
    model.eval()
    probs, labs = [], []
    with torch.no_grad():
        for x, y in loader:
            out = torch.softmax(model(x.to(device)), dim=1)
            probs.extend(out.cpu().numpy())
            labs.extend(y.numpy())
    probs = np.array(probs)
    labs = np.array(labs)
    try:
        return roc_auc_score(labs, probs[:, 1])
    except:
        return 0.0

# ============================================================
# MAIN
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nDevice: {device}")

# Load full dataset once
print("\nLoading full dataset...")
X_train_full, y_train_full, X_test, y_test = get_lodo_split(
    held_out_dataset=HELD_OUT,
    target_class=TARGET_CLASS,
    max_per_class=MAX_PER_CLASS
)

test_loader = DataLoader(ECGDataset(X_test, y_test), batch_size=BATCH_SIZE)
print(f"Full train: {len(X_train_full)} | Test: {len(X_test)}")

results = []

for pct in PERCENTAGES:
    n_samples = max(int(len(X_train_full) * pct), 10)
    print(f"\n{'='*55}")
    print(f"Training with {pct*100:.0f}% data ({n_samples} samples)")
    print(f"{'='*55}")

    # Sample subset
    idx = np.random.choice(len(X_train_full), n_samples, replace=False)
    X_sub = X_train_full[idx]
    y_sub = y_train_full[idx]

    train_loader = DataLoader(ECGDataset(X_sub, y_sub), batch_size=min(BATCH_SIZE, n_samples), shuffle=True)

    # Train ResNet1D
    print(f"  Training ResNet1D...")
    resnet = ResNet1D(num_classes=2).to(device)
    resnet = train_model(resnet, train_loader, device)
    resnet_auroc = evaluate(resnet, test_loader, device)
    print(f"  ResNet1D AUROC: {resnet_auroc:.4f}")

    results.append({
        "percentage": pct * 100,
        "n_samples": n_samples,
        "model": "ResNet1D",
        "test_auroc": round(resnet_auroc, 4)
    })

    # Save intermediate
    pd.DataFrame(results).to_csv(RESULTS_FILE, index=False)

# ============================================================
# PLOT
# ============================================================
df = pd.DataFrame(results)

fig, ax = plt.subplots(figsize=(10, 6))

resnet_data = df[df['model'] == 'ResNet1D']
ax.plot(resnet_data['percentage'], resnet_data['test_auroc'],
        'o-', color='#2E4057', linewidth=2.5, markersize=8, label='ResNet1D')

ax.axhline(y=0.9, color='gray', linestyle='--', alpha=0.5, label='Target AUROC (0.90)')
ax.set_xlabel('Training Data Used (%)', fontsize=12)
ax.set_ylabel('Test Macro AUROC', fontsize=12)
ax.set_title(f'Data Efficiency Analysis — {TARGET_CLASS} Detection\n(Hold out: {HELD_OUT.upper()})', fontsize=13)
ax.legend(fontsize=11)
ax.set_ylim([0.4, 1.0])
ax.set_xticks([1, 5, 10, 25, 50, 100])
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/Figure_DataEfficiency.png', dpi=200, bbox_inches='tight')
print("\nFigure saved to outputs/Figure_DataEfficiency.png")

print("\n" + "="*55)
print("SUMMARY — Data Efficiency Results")
print("="*55)
print(df.to_string(index=False))
print(f"\nResults saved to: {RESULTS_FILE}")
