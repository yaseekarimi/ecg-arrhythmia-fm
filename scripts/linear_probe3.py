"""
Linear Probe - v3
HuBERT-ECG expects (batch, samples) — no channel dim
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from transformers import AutoModel
import wfdb
from scipy import signal as scipy_signal
import pandas as pd
import ast

PTBXL_PATH = "data/ptbxl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
TARGET_FS = 500
TARGET_LEN = 5000
BATCH_SIZE = 8
EPOCHS = 5
LEARNING_RATE = 0.001
MAX_RECORDS = 300

print("="*50)
print("Linear Probe — HuBERT-ECG v3")
print("="*50)

def load_labels_csv(path):
    csv_path = os.path.join(path, "ptbxl_database.csv")
    df = pd.read_csv(csv_path)
    label_map = {}
    for _, row in df.iterrows():
        filename = row['filename_lr']
        scp = ast.literal_eval(row['scp_codes'])
        label = 2
        if any(k in ['AFIB', 'AFLT'] for k in scp.keys()):
            label = 0
        elif any(k in ['PVC', 'SVPB'] for k in scp.keys()):
            label = 1
        label_map[filename] = label
    return label_map

def process_signal(sig, fs):
    if fs != TARGET_FS:
        num = int(len(sig) * TARGET_FS / fs)
        sig = scipy_signal.resample(sig, num)
    if len(sig) >= TARGET_LEN:
        sig = sig[:TARGET_LEN]
    else:
        sig = np.pad(sig, ((0, TARGET_LEN - len(sig)), (0, 0)))
    lead = sig[:, 0]  # Lead I only — shape: (5000,)
    mean, std = lead.mean(), lead.std()
    if std == 0: std = 1
    return (lead - mean) / std

def load_ptbxl(path, max_records=MAX_RECORDS):
    print(f"\nLoading labels from CSV...")
    label_map = load_labels_csv(path)
    af_keys = [k for k, v in label_map.items() if v == 0]
    pvc_keys = [k for k, v in label_map.items() if v == 1]
    other_keys = [k for k, v in label_map.items() if v == 2]
    n_each = max_records // 3
    selected = af_keys[:n_each] + pvc_keys[:n_each] + other_keys[:n_each]
    signals, labels = [], []
    print(f"Loading {len(selected)} records...")
    for i, filename in enumerate(selected):
        filename_hr = filename.replace('_lr', '_hr').replace('records100', 'records500')
        rec_path = os.path.join(path, filename_hr)
        if not os.path.exists(rec_path + '.hea'):
            rec_path = os.path.join(path, filename)
        try:
            record = wfdb.rdrecord(rec_path)
            sig = process_signal(record.p_signal, record.fs)
            signals.append(sig.astype(np.float32))
            labels.append(label_map[filename])
            if (i+1) % 50 == 0:
                print(f"  Loaded {i+1}/{len(selected)}...")
        except:
            continue
    signals = np.array(signals)
    labels = np.array(labels)
    print(f"\nFinal: AF={(labels==0).sum()}, PVC={(labels==1).sum()}, Other={(labels==2).sum()}")
    return signals, labels

class ECGDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)  # shape: (N, 5000)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]

class LinearProbe(nn.Module):
    def __init__(self, backbone, hidden_size, num_classes=3):
        super().__init__()
        self.backbone = backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x: (batch, 5000) — HuBERT expects (batch, sequence_length)
        with torch.no_grad():
            outputs = self.backbone(x)
        features = outputs.last_hidden_state.mean(dim=1)
        return self.classifier(features)

def train_epoch(model, loader, opt, crit, device):
    model.train()
    model.backbone.eval()
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

    print("\nLoading HuBERT-ECG backbone...")
    backbone = AutoModel.from_pretrained("Edoardo-BS/hubert-ecg-base", trust_remote_code=True)
    hidden_size = backbone.config.hidden_size
    print(f"Hidden size: {hidden_size}")

    signals, labels = load_ptbxl(PTBXL_PATH)

    X_tr, X_val, y_tr, y_val = train_test_split(
        signals, labels, test_size=0.2, random_state=42, stratify=labels)

    tr_loader = DataLoader(ECGDataset(X_tr, y_tr), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(ECGDataset(X_val, y_val), batch_size=BATCH_SIZE)

    model = LinearProbe(backbone, hidden_size, num_classes=3).to(device)
    opt = optim.Adam(model.classifier.parameters(), lr=LEARNING_RATE)
    crit = nn.CrossEntropyLoss()

    print(f"\nTraining Linear Probe for {EPOCHS} epochs...")
    print("-"*40)

    best = 0
    for ep in range(EPOCHS):
        loss = train_epoch(model, tr_loader, opt, crit, device)
        auroc = evaluate(model, val_loader, device)
        if auroc > best:
            best = auroc
            torch.save(model.classifier.state_dict(), "scripts/best_linear_probe.pth")
        print(f"Epoch {ep+1}/{EPOCHS} | Loss: {loss:.4f} | Val AUROC: {auroc:.4f}")

    print("\n" + "="*50)
    print(f"Best Macro AUROC (HuBERT Linear Probe): {best:.4f}")
    print(f"ResNet Baseline AUROC:                  0.9105")
    print("="*50)
