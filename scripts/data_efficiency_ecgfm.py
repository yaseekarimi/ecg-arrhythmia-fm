"""
Data Efficiency Analysis - ECG-FM
Phase 1 - Month 5
Compare ECG-FM vs ResNet1D across different training data percentages
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score
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
PERCENTAGES = [0.01, 0.05, 0.10, 0.25, 0.50, 1.00]
BATCH_SIZE = 16
EPOCHS = 10
LR = 0.001
MAX_PER_CLASS = 200
TARGET_CLASS = "AF"
HELD_OUT = "ptbxl"

os.makedirs("outputs", exist_ok=True)
RESULTS_FILE = "outputs/data_efficiency_ecgfm_results.csv"

print("="*55)
print("Data Efficiency — ECG-FM vs ResNet1D")
print(f"Target: {TARGET_CLASS} | Hold out: {HELD_OUT}")
print("="*55)

# ============================================================
# ECG-FM Linear Probe Model
# ============================================================
class ECGFMProbe(nn.Module):
    def __init__(self, backbone, hidden_size, num_classes=2):
        super().__init__()
        self.backbone = backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # x: (batch, 12, 1000) — use mean of all leads
        x_mean = x.mean(dim=1)  # (batch, 1000)
        with torch.no_grad():
            out = self.backbone(x_mean)
        features = out.last_hidden_state.mean(dim=1)
        return self.classifier(features)

# ============================================================
# TRAIN & EVAL
# ============================================================
def train_model(model, train_loader, device, epochs=EPOCHS, freeze_backbone=False):
    params = model.classifier.parameters() if freeze_backbone else model.parameters()
    optimizer = optim.Adam(params, lr=LR)
    criterion = nn.CrossEntropyLoss()
    model.train()
    if freeze_backbone and hasattr(model, 'backbone'):
        model.backbone.eval()
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

# Load ECG-FM
print("\nLoading ECG-FM backbone...")
from transformers import AutoModel
backbone = AutoModel.from_pretrained("Edoardo-BS/hubert-ecg-base", trust_remote_code=True)
hidden_size = backbone.config.hidden_size
print(f"Hidden size: {hidden_size}")

# Load full dataset
print("\nLoading full dataset...")
X_train_full, y_train_full, X_test, y_test = get_lodo_split(
    held_out_dataset=HELD_OUT,
    target_class=TARGET_CLASS,
    max_per_class=MAX_PER_CLASS
)
test_loader = DataLoader(ECGDataset(X_test, y_test), batch_size=BATCH_SIZE)
print(f"Full train: {len(X_train_full)} | Test: {len(X_test)}")

# Load previous ResNet results
prev_results = []
if os.path.exists("outputs/data_efficiency_results.csv"):
    df_prev = pd.read_csv("outputs/data_efficiency_results.csv")
    prev_results = df_prev.to_dict('records')
    print(f"\nLoaded {len(prev_results)} previous ResNet results")

results = list(prev_results)

for pct in PERCENTAGES:
    n_samples = max(int(len(X_train_full) * pct), 10)
    print(f"\n{'='*55}")
    print(f"Training ECG-FM with {pct*100:.0f}% data ({n_samples} samples)")
    print(f"{'='*55}")

    # Sample subset
    idx = np.random.choice(len(X_train_full), n_samples, replace=False)
    X_sub = X_train_full[idx]
    y_sub = y_train_full[idx]

    train_loader = DataLoader(
        ECGDataset(X_sub, y_sub),
        batch_size=min(BATCH_SIZE, n_samples),
        shuffle=True
    )

    # Train ECG-FM probe
    print(f"  Training ECG-FM (frozen backbone + classifier)...")
    ecgfm_model = ECGFMProbe(backbone, hidden_size, num_classes=2).to(device)
    ecgfm_model = train_model(ecgfm_model, train_loader, device, freeze_backbone=True)
    ecgfm_auroc = evaluate(ecgfm_model, test_loader, device)
    print(f"  ECG-FM AUROC: {ecgfm_auroc:.4f}")

    results.append({
        "percentage": pct * 100,
        "n_samples": n_samples,
        "model": "ECG-FM",
        "test_auroc": round(ecgfm_auroc, 4)
    })

    pd.DataFrame(results).to_csv(RESULTS_FILE, index=False)

# ============================================================
# PLOT COMPARISON
# ============================================================
df = pd.DataFrame(results)

fig, ax = plt.subplots(figsize=(10, 6))

resnet_data = df[df['model'] == 'ResNet1D'].sort_values('percentage')
ecgfm_data = df[df['model'] == 'ECG-FM'].sort_values('percentage')

if len(resnet_data) > 0:
    ax.plot(resnet_data['percentage'], resnet_data['test_auroc'],
            'o-', color='#2E4057', linewidth=2.5, markersize=8, label='ResNet1D (Baseline)')

if len(ecgfm_data) > 0:
    ax.plot(ecgfm_data['percentage'], ecgfm_data['test_auroc'],
            's--', color='#E84855', linewidth=2.5, markersize=8, label='ECG-FM (Foundation Model)')

ax.axhline(y=0.9, color='gray', linestyle=':', alpha=0.5, label='Target AUROC (0.90)')
ax.set_xlabel('Training Data Used (%)', fontsize=12)
ax.set_ylabel('Test Macro AUROC', fontsize=12)
ax.set_title(f'Data Efficiency Analysis — {TARGET_CLASS} Detection\nResNet1D vs ECG-FM (Hold out: {HELD_OUT.upper()})', fontsize=13)
ax.legend(fontsize=11)
ax.set_ylim([0.4, 1.0])
ax.set_xticks([1, 5, 10, 25, 50, 100])
ax.grid(alpha=0.3)

plt.tight_layout()
fig_path = 'outputs/Figure_DataEfficiency_Comparison.png'
plt.savefig(fig_path, dpi=200, bbox_inches='tight')
print(f"\nFigure saved to {fig_path}")

print("\n" + "="*55)
print("SUMMARY — Data Efficiency Comparison")
print("="*55)
print(df.sort_values(['model', 'percentage']).to_string(index=False))
print(f"\nResults saved to: {RESULTS_FILE}")
