"""
Grad-CAM Visualisation
Phase 1 - Month 4
Shows which ECG segments the model uses for classification
"""

import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
sys.path.append('scripts')
from lodo_loader import get_lodo_split, ECGDataset
from cnn_lstm import CNNLSTM
from torch.utils.data import DataLoader

print("="*50)
print("Grad-CAM Visualisation")
print("="*50)

# ============================================================
# GRAD-CAM
# ============================================================
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)
    
    def _save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate(self, x, class_idx=None):
        self.model.eval()
        x = x.unsqueeze(0) if x.dim() == 2 else x
        
        output = self.model(x)
        
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        
        self.model.zero_grad()
        output[0, class_idx].backward()
        
        # Compute weights
        weights = self.gradients.mean(dim=2, keepdim=True)  # (batch, channels, 1)
        cam = (weights * self.activations).sum(dim=1)       # (batch, time)
        cam = torch.relu(cam)
        
        # Normalise
        cam = cam.squeeze().cpu().numpy()
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        
        return cam, class_idx

# ============================================================
# PLOT
# ============================================================
def plot_gradcam(signal, cam, label, pred_class, lead_idx=0, save_path=None):
    """Plot ECG signal with Grad-CAM overlay"""
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    time = np.arange(len(signal)) / 100  # seconds at 100Hz
    lead_signal = signal[lead_idx]
    
    class_names = {0: "AF", 1: "PVC", 2: "Other", 1: "Positive"}
    true_name = "Positive" if label == 1 else "Negative"
    pred_name = "Positive" if pred_class == 1 else "Negative"
    
    # Top: ECG signal
    axes[0].plot(time, lead_signal, color='#2E4057', linewidth=1.2)
    axes[0].set_ylabel("Amplitude (mV)", fontsize=11)
    axes[0].set_title(f"ECG Signal (Lead I) — True: {true_name} | Predicted: {pred_name}", fontsize=12)
    axes[0].set_xlim([0, len(signal)/100])
    axes[0].grid(alpha=0.3)
    
    # Bottom: ECG with Grad-CAM heatmap
    # Upsample CAM to signal length
    cam_upsampled = np.interp(
        np.linspace(0, len(cam)-1, len(lead_signal)),
        np.arange(len(cam)),
        cam
    )
    
    axes[1].plot(time, lead_signal, color='#2E4057', linewidth=1.2, zorder=2)
    
    # Color background by Grad-CAM intensity
    for i in range(len(time)-1):
        alpha = float(cam_upsampled[i]) * 0.6
        axes[1].axvspan(time[i], time[i+1], alpha=alpha, color='red', zorder=1)
    
    axes[1].set_xlabel("Time (seconds)", fontsize=11)
    axes[1].set_ylabel("Amplitude (mV)", fontsize=11)
    axes[1].set_title("Grad-CAM Overlay — Red = Regions Important for Classification", fontsize=12)
    axes[1].set_xlim([0, len(signal)/100])
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.close()

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    print("\nLoading CNN-LSTM model...")
    model = CNNLSTM(n_leads=12, num_classes=2).to(device)
    
    if os.path.exists("scripts/best_cnn_lstm.pth"):
        model.load_state_dict(torch.load("scripts/best_cnn_lstm.pth", map_location=device))
        print("Model loaded!")
    else:
        print("No saved model found — using random weights")
    
    # Setup Grad-CAM on last CNN layer
    target_layer = model.cnn[6]  # Last conv layer
    gradcam = GradCAM(model, target_layer)
    
    # Load some test data
    print("\nLoading test data...")
    X_train, y_train, X_test, y_test = get_lodo_split(
        held_out_dataset="ptbxl",
        target_class="AF",
        max_per_class=20
    )
    
    # Create output directory
    os.makedirs("outputs/gradcam", exist_ok=True)
    
    # Generate Grad-CAM for 4 examples
    print("\nGenerating Grad-CAM visualisations...")
    
    # Get 2 positive and 2 negative examples
    pos_idx = np.where(y_test == 1)[0][:2]
    neg_idx = np.where(y_test == 0)[0][:2]
    examples = list(pos_idx) + list(neg_idx)
    
    for i, idx in enumerate(examples):
        x = torch.tensor(X_test[idx], dtype=torch.float32).to(device)
        label = y_test[idx]
        
        cam, pred_class = gradcam.generate(x)
        
        save_path = f"outputs/gradcam/gradcam_example_{i+1}_true{'Pos' if label==1 else 'Neg'}.png"
        plot_gradcam(X_test[idx], cam, label, pred_class, save_path=save_path)
    
    print("\n" + "="*50)
    print("Grad-CAM complete!")
    print("Images saved to: outputs/gradcam/")
    print("="*50)
