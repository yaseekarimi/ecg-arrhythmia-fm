"""
Strodthoff 2021 ResNet Baseline Reproduction
Phase 0 - Month 2
Train a simple ResNet on PTB-XL and evaluate macro AUROC
Target: reproduce within 1% of reported macro AUROC
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import wfdb
from scipy import signal as scipy_signal

# ============================================================
# CONFIG
# ============================================================
PTBXL_PATH = "data/ptbxl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
TARGET_FS = 100
TARGET_LEN = 1000
NUM_LEADS = 12
BATCH_SIZE = 32
EPOCHS = 5          # Small for testing — increase later
LEARNING_RATE = 0.001
MAX_RECORDS = 500   # Use 500 records for quick test

AF_LABELS = ["AFIB", "AF", "164889003"]
PVC_LABELS = ["PVC", "164884008", "17338001"]

print("="*50)
print("Strodthoff 2021 ResNet Baseline")
print("="*50)

# ============================================================
# DATA LOADING
# ============================================================
def map_label(comments):
    for c in comments:
        if any(af in str(c) for af in AF_LABELS):
            return 0  # AF
    for c in comments:
        if any(pvc in str(c) for pvc in PVC_LABELS):
            return 1  # PVC
    return 2  # Other

def load_ptbxl(path, max_records=MAX_RECORDS):
    print(f"\nLoading PTB-XL (max {max_records} records)...")
    signals, labels = [], []
    
    hea_files = []
    for root, dirs, files in os.walk(path):
        for f in files:
            if f.endswith('.hea'):
                hea_files.append(os.path.join(root, f[:-4]))
    
    hea_files = hea_files[:max_records]
    
    for i, rec_path in enumerate(hea_files):
        try:
            record = wfdb.rdrecord(rec_path)
            header = wfdb.rdheader(rec_path)
            
            sig = record.p_signal
            fs = record.fs
            
            # Resample to 100 Hz
            if fs != TARGET_FS:
                num = int(len(sig) * TARGET_FS / fs)
                sig = scipy_signal.resample(sig, num)
            
            # Fix length
            if len(sig) >= TARGET_LEN:
                sig = sig[:TARGET_LEN]
            else:
                sig = np.pad(sig, ((0, TARGET_LEN - len(sig)), (0, 0)))
            
            # Only use 12-lead
            if sig.shape[1] < NUM_LEADS:
                continue
            sig = sig[:, :NUM_LEADS]
            
            # Z-score normalise
            mean = np.mean(sig, axis=0)
            std = np.std(sig, axis=0)
            std[std == 0] = 1
            sig = (sig - mean) / std
            
            label = map_label(header.comments)
            signals.append(sig.T.astype(np.float32))  # (leads, samples)
            labels.append(label)
            
            if (i+1) % 100 == 0:
                print(f"  Loaded {i+1}/{len(hea_files)} records...")
        
        except Exception as e:
            continue
    
    print(f"  Total loaded: {len(signals)} records")
    return np.array(signals), np.array(labels)

# ============================================================
# DATASET
# ============================================================
class ECGDataset(Dataset):
    def __init__(self, signals, labels):
        self.signals = torch.tensor(signals, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
    
    def __len__(self):
        return len(self.signals)
    
    def __getitem__(self, idx):
        return self.signals[idx], self.labels[idx]

# ============================================================
# RESNET MODEL
# ============================================================
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, 15, stride=stride, padding=7, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 15, padding=7, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )
    
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class ResNet1D(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.conv1 = nn.Conv1d(12, 64, 15, padding=7, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
        
        self.layer1 = ResidualBlock(64, 64)
        self.layer2 = ResidualBlock(64, 128, stride=2)
        self.layer3 = ResidualBlock(128, 256, stride=2)
        self.layer4 = ResidualBlock(256, 512, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512, num_classes)
    
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# ============================================================
# TRAINING
# ============================================================
def train(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for signals, labels in loader:
        signals, labels = signals.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(signals)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for signals, labels in loader:
            signals = signals.to(device)
            outputs = torch.softmax(model(signals), dim=1)
            all_probs.extend(outputs.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    # One-hot encode labels
    n_classes = all_probs.shape[1]
    labels_onehot = np.zeros((len(all_labels), n_classes))
    for i, l in enumerate(all_labels):
        labels_onehot[i, l] = 1
    
    try:
        auroc = roc_auc_score(labels_onehot, all_probs, average='macro')
    except Exception:
        auroc = 0.0
    
    return auroc

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    
    # Load data
    signals, labels = load_ptbxl(PTBXL_PATH, max_records=MAX_RECORDS)
    
    print(f"\nLabel distribution:")
    print(f"  AF: {(labels==0).sum()}")
    print(f"  PVC: {(labels==1).sum()}")
    print(f"  Other: {(labels==2).sum()}")
    
    # Split
    X_train, X_val, y_train, y_val = train_test_split(
        signals, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    train_dataset = ECGDataset(X_train, y_train)
    val_dataset = ECGDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    # Model
    model = ResNet1D(num_classes=3).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    
    print(f"\nTraining ResNet1D for {EPOCHS} epochs...")
    print("-"*40)
    
    best_auroc = 0
    for epoch in range(EPOCHS):
        train_loss = train(model, train_loader, optimizer, criterion, device)
        val_auroc = evaluate(model, val_loader, device)
        
        if val_auroc > best_auroc:
            best_auroc = val_auroc
            torch.save(model.state_dict(), "scripts/best_resnet.pth")
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {train_loss:.4f} | Val AUROC: {val_auroc:.4f}")
    
    print("\n" + "="*50)
    print(f"Best Macro AUROC: {best_auroc:.4f}")
    print(f"Target (Strodthoff 2021): ~0.93")
    print(f"Model saved to: scripts/best_resnet.pth")
    print("="*50)
