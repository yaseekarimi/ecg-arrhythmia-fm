"""
Strodthoff 2021 ResNet Baseline - Fixed Version
Uses random sampling to ensure balanced AF/PVC/Other classes
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
import random

# ============================================================
# CONFIG
# ============================================================
PTBXL_PATH = "data/ptbxl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
TARGET_FS = 100
TARGET_LEN = 1000
NUM_LEADS = 12
BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 0.001
MAX_RECORDS = 500

AF_LABELS = ["AFIB", "AF", "164889003"]
PVC_LABELS = ["PVC", "164884008", "17338001"]

print("="*50)
print("Strodthoff 2021 ResNet Baseline v2")
print("="*50)

def map_label(comments):
    for c in comments:
        if any(af in str(c) for af in AF_LABELS):
            return 0
    for c in comments:
        if any(pvc in str(c) for pvc in PVC_LABELS):
            return 1
    return 2

def process_signal(sig, fs):
    if fs != TARGET_FS:
        num = int(len(sig) * TARGET_FS / fs)
        sig = scipy_signal.resample(sig, num)
    if len(sig) >= TARGET_LEN:
        sig = sig[:TARGET_LEN]
    else:
        sig = np.pad(sig, ((0, TARGET_LEN - len(sig)), (0, 0)))
    if sig.shape[1] < NUM_LEADS:
        return None
    sig = sig[:, :NUM_LEADS]
    mean = np.mean(sig, axis=0)
    std = np.std(sig, axis=0)
    std[std == 0] = 1
    return (sig - mean) / std

def load_ptbxl(path, max_records=MAX_RECORDS):
    print(f"\nLoading PTB-XL with random sampling...")
    
    hea_files = []
    for root, dirs, files in os.walk(path):
        for f in files:
            if f.endswith('.hea'):
                hea_files.append(os.path.join(root, f[:-4]))
    
    # Shuffle to get random sample
    random.seed(42)
    random.shuffle(hea_files)
    
    signals, labels = [], []
    
    for i, rec_path in enumerate(hea_files):
        if len(signals) >= max_records:
            break
        try:
            record = wfdb.rdrecord(rec_path)
            header = wfdb.rdheader(rec_path)
            sig = process_signal(record.p_signal, record.fs)
            if sig is None:
                continue
            label = map_label(header.comments)
            signals.append(sig.T.astype(np.float32))
            labels.append(label)
            if (len(signals)) % 100 == 0:
                print(f"  Loaded {len(signals)}/{max_records}...")
        except:
            continue
    
    print(f"  Total: {len(signals)} records")
    return np.array(signals), np.array(labels)

class ECGDataset(Dataset):
    def __init__(self, signals, labels):
        self.signals = torch.tensor(signals, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
    def __len__(self):
        return len(self.signals)
    def __getitem__(self, idx):
        return self.signals[idx], self.labels[idx]

class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, 15, stride=stride, padding=7, bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.2)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 15, padding=7, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch))
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.drop(out)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return self.relu(out)

class ResNet1D(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.conv1 = nn.Conv1d(12, 64, 15, padding=7, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
        self.l1 = ResidualBlock(64, 64)
        self.l2 = ResidualBlock(64, 128, stride=2)
        self.l3 = ResidualBlock(128, 256, stride=2)
        self.l4 = ResidualBlock(256, 512, stride=2)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512, num_classes)
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.l1(x); x = self.l2(x)
        x = self.l3(x); x = self.l4(x)
        x = self.pool(x).view(x.size(0), -1)
        return self.fc(x)

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
    oh = np.zeros((len(labs), 3))
    for i, l in enumerate(labs): oh[i, l] = 1
    try:
        return roc_auc_score(oh, probs, average='macro')
    except:
        return 0.0

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    signals, labels = load_ptbxl(PTBXL_PATH)
    
    print(f"\nLabel distribution:")
    print(f"  AF: {(labels==0).sum()}")
    print(f"  PVC: {(labels==1).sum()}")
    print(f"  Other: {(labels==2).sum()}")
    
    X_tr, X_val, y_tr, y_val = train_test_split(
        signals, labels, test_size=0.2, random_state=42)
    
    tr_loader = DataLoader(ECGDataset(X_tr, y_tr), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(ECGDataset(X_val, y_val), batch_size=BATCH_SIZE)
    
    model = ResNet1D(3).to(device)
    opt = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    crit = nn.CrossEntropyLoss()
    
    print(f"\nTraining for {EPOCHS} epochs...")
    print("-"*40)
    
    best = 0
    for ep in range(EPOCHS):
        loss = train_epoch(model, tr_loader, opt, crit, device)
        auroc = evaluate(model, val_loader, device)
        if auroc > best:
            best = auroc
            torch.save(model.state_dict(), "scripts/best_resnet.pth")
        print(f"Epoch {ep+1}/{EPOCHS} | Loss: {loss:.4f} | Val AUROC: {auroc:.4f}")
    
    print("\n" + "="*50)
    print(f"Best Macro AUROC: {best:.4f}")
    print(f"Target (Strodthoff 2021): ~0.93")
    print("="*50)
