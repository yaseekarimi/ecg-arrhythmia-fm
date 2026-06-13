"""
Phase 1 - M5
Run all LODO experiments for ResNet1D and CNN-LSTM baselines
16 runs total: 2 models x 2 classes x 4 LODO folds (excluding MIT-BIH for baselines)
Results saved to outputs/results_baseline.csv
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score
from sklearn.model_selection import train_test_split
import pandas as pd
import sys
sys.path.append('scripts')
from lodo_loader import get_lodo_split, ECGDataset
from cnn_lstm import CNNLSTM
from baseline3 import ResNet1D

# ============================================================
# CONFIG
# ============================================================
BATCH_SIZE = 32
EPOCHS = 10
LR = 0.001
MAX_PER_CLASS = 150  # Increase when GPU available

LODO_DATASETS = ["ptbxl", "cpsc2018", "georgia", "chapman"]  # MIT-BIH excluded (too small)
TARGET_CLASSES = ["AF", "PVC"]
MODELS = ["ResNet1D", "CNN-LSTM"]

os.makedirs("outputs", exist_ok=True)
RESULTS_FILE = "outputs/results_baseline.csv"

print("="*60)
print("Phase 1 - M5: Baseline LODO Experiments")
print(f"Models: {MODELS}")
print(f"Classes: {TARGET_CLASSES}")
print(f"LODO datasets: {LODO_DATASETS}")
print(f"Total runs: {len(MODELS) * len(TARGET_CLASSES) * len(LODO_DATASETS)}")
print("="*60)

# ============================================================
# TRAIN & EVAL
# ============================================================
def train_model(model, train_loader, val_loader, device, epochs=EPOCHS, lr=LR):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    best_val_auroc = 0
    best_state = None

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        val_auroc = evaluate_auroc(model, val_loader, device)
        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 5 == 0:
            print(f"    Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(train_loader):.4f} | Val AUROC: {val_auroc:.4f}")

    model.load_state_dict(best_state)
    return model, best_val_auroc

def evaluate_auroc(model, loader, device):
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

for target_class in TARGET_CLASSES:
    for held_out in LODO_DATASETS:
        for model_name in MODELS:
            run_num += 1
            print(f"\n{'='*60}")
            print(f"Run {run_num}/{len(MODELS)*len(TARGET_CLASSES)*len(LODO_DATASETS)}: {model_name} | {target_class} | Hold out: {held_out}")
            print(f"{'='*60}")

            # Load data
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

            # Build model
            if model_name == "ResNet1D":
                model = ResNet1D(num_classes=2).to(device)
            else:
                model = CNNLSTM(n_leads=12, num_classes=2).to(device)

            # Train
            model, best_val = train_model(model, tr_loader, val_loader, device)

            # Evaluate on test
            test_auroc, test_f1, test_sens, test_prec = evaluate_full(model, test_loader, device)

            print(f"\n  Results:")
            print(f"  Val AUROC:  {best_val:.4f}")
            print(f"  Test AUROC: {test_auroc:.4f}")
            print(f"  F1:         {test_f1:.4f}")
            print(f"  Sensitivity:{test_sens:.4f}")

            # Save result
            results.append({
                "run": run_num,
                "model": model_name,
                "target_class": target_class,
                "held_out": held_out,
                "val_auroc": round(best_val, 4),
                "test_auroc": round(test_auroc, 4),
                "f1": round(test_f1, 4),
                "sensitivity": round(test_sens, 4),
                "precision": round(test_prec, 4),
            })

            # Save intermediate results
            pd.DataFrame(results).to_csv(RESULTS_FILE, index=False)
            print(f"  Saved to {RESULTS_FILE}")

# ============================================================
# SUMMARY
# ============================================================
df = pd.DataFrame(results)
print("\n" + "="*60)
print("SUMMARY — Baseline LODO Results")
print("="*60)
print(df.to_string(index=False))
print(f"\nFull results saved to: {RESULTS_FILE}")
