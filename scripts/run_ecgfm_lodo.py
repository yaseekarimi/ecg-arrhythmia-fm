"""
Phase 1 - M6
ECG-FM Foundation Model LODO Experiments
16 runs: 2 modes x 2 classes x 4 LODO folds
Results saved to outputs/results_ecgfm.csv
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score
from sklearn.model_selection import train_test_split
import pandas as pd
import sys
sys.path.append('scripts')
from lodo_loader import get_lodo_split

# ============================================================
# CONFIG
# ============================================================
BATCH_SIZE = 8
EPOCHS_PROBE = 10
EPOCHS_FINETUNE = 5
LR = 0.001
MAX_PER_CLASS = 150

LODO_DATASETS = ["ptbxl", "cpsc2018", "georgia", "chapman"]
TARGET_CLASSES = ["AF", "PVC"]

os.makedirs("outputs", exist_ok=True)
RESULTS_FILE = "outputs/results_ecgfm.csv"

print("="*60)
print("Phase 1 - M6: ECG-FM LODO Experiments")
print(f"Classes: {TARGET_CLASSES}")
print(f"LODO datasets: {LODO_DATASETS}")
print(f"Total runs: {len(TARGET_CLASSES) * len(LODO_DATASETS) * 2}")
print("="*60)

# ============================================================
# DATASET
# ============================================================
class ECGDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]

# ============================================================
# ECG-FM MODEL
# ============================================================
class ECGFMProbe(nn.Module):
    def __init__(self, backbone, hidden_size, num_classes=2, freeze=True):
        super().__init__()
        self.backbone = backbone
        self.freeze = freeze
        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # x: (batch, 12, 1000)
        # ECG-FM uses mean of leads as input
        x_mean = x.mean(dim=1)  # (batch, 1000)
        if self.freeze:
            with torch.no_grad():
                out = self.backbone(x_mean)
        else:
            out = self.backbone(x_mean)
        features = out.last_hidden_state.mean(dim=1)
        return self.classifier(features)

# ============================================================
# TRAIN & EVAL
# ============================================================
def train_model(model, loader, device, epochs, freeze=True):
    params = model.classifier.parameters() if freeze else model.parameters()
    optimizer = optim.Adam(params, lr=LR)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        model.train()
        if freeze:
            model.backbone.eval()
        total = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            total += loss.item()
        if (epoch+1) % 5 == 0:
            print(f"    Epoch {epoch+1}/{epochs} | Loss: {total/len(loader):.4f}")
    return model

def evaluate_full(model, loader, device):
    model.eval()
    probs, labs, preds = [], [], []
    with torch.no_grad():
        for x, y in loader:
            out = torch.softmax(model(x.to(device)), dim=1)
            pred = out.argmax(dim=1)
            probs.extend(out.cpu().numpy())
            labs.extend(y.numpy())
            preds.extend(pred.cpu().numpy())
    probs = np.array(probs)
    labs = np.array(labs)
    preds = np.array(preds)
    try:
        auroc = roc_auc_score(labs, probs[:, 1])
        f1 = f1_score(labs, preds, zero_division=0)
        sensitivity = recall_score(labs, preds, zero_division=0)
        precision = precision_score(labs, preds, zero_division=0)
    except:
        auroc = f1 = sensitivity = precision = 0.0
    return auroc, f1, sensitivity, precision

# ============================================================
# MAIN LOOP
# ============================================================
results = []
run_num = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nDevice: {device}\n")

# Load ECG-FM backbone
print("Loading ECG-FM backbone...")
from transformers import AutoModel
backbone = AutoModel.from_pretrained("wanglab/ecg-fm", trust_remote_code=True)
hidden_size = backbone.config.hidden_size
print(f"ECG-FM hidden size: {hidden_size}\n")

total_runs = len(TARGET_CLASSES) * len(LODO_DATASETS) * 2

for target_class in TARGET_CLASSES:
    for held_out in LODO_DATASETS:
        for mode in ["linear_probe", "fine_tune"]:
            run_num += 1
            print(f"\n{'='*60}")
            print(f"Run {run_num}/{total_runs}: ECG-FM | {mode} | {target_class} | Hold out: {held_out}")
            print(f"{'='*60}")

            X_train, y_train, X_test, y_test = get_lodo_split(
                held_out_dataset=held_out,
                target_class=target_class,
                max_per_class=MAX_PER_CLASS
            )

            X_tr, X_val, y_tr, y_val = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)

            tr_loader = DataLoader(ECGDataset(X_tr, y_tr), batch_size=BATCH_SIZE, shuffle=True)
            val_loader = DataLoader(ECGDataset(X_val, y_val), batch_size=BATCH_SIZE)
            test_loader = DataLoader(ECGDataset(X_test, y_test), batch_size=BATCH_SIZE)

            freeze = (mode == "linear_probe")
            epochs = EPOCHS_PROBE if freeze else EPOCHS_FINETUNE

            model = ECGFMProbe(backbone, hidden_size, num_classes=2, freeze=freeze).to(device)
            model = train_model(model, tr_loader, device, epochs=epochs, freeze=freeze)

            test_auroc, test_f1, test_sens, test_prec = evaluate_full(model, test_loader, device)
            val_auroc, _, _, _ = evaluate_full(model, val_loader, device)

            print(f"\n  Val AUROC:  {val_auroc:.4f}")
            print(f"  Test AUROC: {test_auroc:.4f}")

            results.append({
                "run": run_num,
                "model": "ECG-FM",
                "mode": mode,
                "target_class": target_class,
                "held_out": held_out,
                "val_auroc": round(val_auroc, 4),
                "test_auroc": round(test_auroc, 4),
                "f1": round(test_f1, 4),
                "sensitivity": round(test_sens, 4),
                "precision": round(test_prec, 4),
            })

            pd.DataFrame(results).to_csv(RESULTS_FILE, index=False)
            print(f"  Saved to {RESULTS_FILE}")

df = pd.DataFrame(results)
print("\n" + "="*60)
print("SUMMARY — ECG-FM LODO Results")
print("="*60)
print(df.to_string(index=False))
print(f"\nFull results saved to: {RESULTS_FILE}")
