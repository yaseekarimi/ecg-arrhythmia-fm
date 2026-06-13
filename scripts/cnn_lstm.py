"""
CNN-LSTM Baseline Model
Phase 1 - Month 4
Combines CNN for local feature extraction with LSTM for temporal patterns
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import sys
sys.path.append('scripts')
from lodo_loader import get_lodo_split, ECGDataset

BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001
MAX_PER_CLASS = 150

print("="*50)
print("CNN-LSTM Baseline")
print("="*50)

# ============================================================
# CNN-LSTM MODEL
# ============================================================
class CNNLSTM(nn.Module):
    def __init__(self, n_leads=12, num_classes=2):
        super().__init__()
        
        # CNN part — extract local features
        self.cnn = nn.Sequential(
            nn.Conv1d(n_leads, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),  # 1000 -> 500
            
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),  # 500 -> 250
            
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2),  # 250 -> 125
        )
        
        # LSTM part — capture temporal patterns
        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
            bidirectional=True
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(128 * 2, 64),  # *2 for bidirectional
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        # x: (batch, leads, samples)
        x = self.cnn(x)           # (batch, 256, 125)
        x = x.permute(0, 2, 1)   # (batch, 125, 256) for LSTM
        x, _ = self.lstm(x)       # (batch, 125, 256)
        x = x[:, -1, :]           # Last timestep
        return self.classifier(x)

# ============================================================
# TRAIN & EVAL
# ============================================================
def train_epoch(model, loader, opt, crit, device):
    model.train()
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        loss = crit(model(x), y)
        loss.backward()
        opt.step()
        total += loss.item()
    return total / len(loader)

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
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Test with one LODO split
    print("\nLoading LODO split (Hold out: ptbxl, Target: AF)...")
    X_train, y_train, X_test, y_test = get_lodo_split(
        held_out_dataset="ptbxl",
        target_class="AF",
        max_per_class=MAX_PER_CLASS
    )

    # Split train into train/val
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)

    tr_loader = DataLoader(ECGDataset(X_tr, y_tr), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(ECGDataset(X_val, y_val), batch_size=BATCH_SIZE)
    test_loader = DataLoader(ECGDataset(X_test, y_test), batch_size=BATCH_SIZE)

    # Model
    model = CNNLSTM(n_leads=12, num_classes=2).to(device)
    opt = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    crit = nn.CrossEntropyLoss()

    print(f"\nTraining CNN-LSTM for {EPOCHS} epochs...")
    print("-"*40)

    best_val = 0
    for ep in range(EPOCHS):
        loss = train_epoch(model, tr_loader, opt, crit, device)
        val_auroc = evaluate(model, val_loader, device)
        if val_auroc > best_val:
            best_val = val_auroc
            torch.save(model.state_dict(), "scripts/best_cnn_lstm.pth")
        print(f"Epoch {ep+1}/{EPOCHS} | Loss: {loss:.4f} | Val AUROC: {val_auroc:.4f}")

    # Test on held-out dataset
    model.load_state_dict(torch.load("scripts/best_cnn_lstm.pth"))
    test_auroc = evaluate(model, test_loader, device)

    print("\n" + "="*50)
    print(f"Val AUROC:  {best_val:.4f}")
    print(f"Test AUROC: {test_auroc:.4f} (held out: ptbxl)")
    print(f"ResNet AUROC: 0.9105 (for comparison)")
    print("="*50)
