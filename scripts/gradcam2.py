"""
Grad-CAM Visualisation - Fixed Version
Phase 1 - Month 4
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
from lodo_loader import get_lodo_split
from cnn_lstm import CNNLSTM

print("="*50)
print("Grad-CAM Visualisation v2")
print("="*50)

# ============================================================
# GRAD-CAM
# ============================================================
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)
    
    def _save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate(self, x, class_idx=None):
        self.model.eval()
        if x.dim() == 2:
            x = x.unsqueeze(0)  # Add batch dim
        
        output = self.model(x)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        
        self.model.zero_grad()
        output[0, class_idx].backward()
        
        weights = self.gradients.mean(dim=2, keepdim=True)
        cam = (weights * self.activations).sum(dim=1)
        cam = torch.relu(cam)
        cam = cam.squeeze().cpu().numpy()
        
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        
        return cam, class_idx, output.softmax(dim=1)[0, class_idx].item()

# ============================================================
# PLOT
# ============================================================
def plot_gradcam(signal, cam, label, pred_class, confidence, save_path):
    """
    signal shape: (12, 1000) — leads x samples
    cam shape: (125,) — from CNN output
    """
    # Use Lead I (index 0)
    lead_signal = signal[0, :]  # shape: (1000,)
    time = np.arange(len(lead_signal)) / 100.0  # seconds

    # Upsample CAM to signal length
    cam_upsampled = np.interp(
        np.linspace(0, len(cam)-1, len(lead_signal)),
        np.arange(len(cam)),
        cam
    )

    true_name = "AF" if label == 1 else "Normal"
    pred_name = "AF" if pred_class == 1 else "Normal"

    fig, axes = plt.subplots(2, 1, figsize=(14, 7))

    # Top: raw ECG
    axes[0].plot(time, lead_signal, color='#2E4057', linewidth=1.2)
    axes[0].set_ylabel("Amplitude", fontsize=11)
    axes[0].set_title(f"ECG Signal (Lead I) — True: {true_name} | Predicted: {pred_name} ({confidence:.2f})", fontsize=12)
    axes[0].set_xlim([0, time[-1]])
    axes[0].grid(alpha=0.3)

    # Bottom: ECG + Grad-CAM heatmap
    axes[1].plot(time, lead_signal, color='#2E4057', linewidth=1.2, zorder=2)
    for i in range(len(time)-1):
        alpha = float(cam_upsampled[i]) * 0.6
        if alpha > 0.05:
            axes[1].axvspan(time[i], time[i+1], alpha=alpha, color='red', zorder=1)

    axes[1].set_xlabel("Time (seconds)", fontsize=11)
    axes[1].set_ylabel("Amplitude", fontsize=11)
    axes[1].set_title("Grad-CAM — Red regions = important for classification", fontsize=12)
    axes[1].set_xlim([0, time[-1]])
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")

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
        print("No saved model — using random weights")

    # Grad-CAM on last CNN layer
    target_layer = model.cnn[6]
    gradcam = GradCAM(model, target_layer)

    # Load test data
    print("\nLoading test data...")
    X_train, y_train, X_test, y_test = get_lodo_split(
        held_out_dataset="ptbxl",
        target_class="AF",
        max_per_class=20
    )

    os.makedirs("outputs/gradcam", exist_ok=True)

    # 2 positive + 2 negative examples
    pos_idx = np.where(y_test == 1)[0][:2]
    neg_idx = np.where(y_test == 0)[0][:2]
    examples = list(pos_idx) + list(neg_idx)

    print("\nGenerating Grad-CAM visualisations...")
    for i, idx in enumerate(examples):
        # signal shape: (12, 1000)
        x = torch.tensor(X_test[idx], dtype=torch.float32).to(device)
        label = int(y_test[idx])

        cam, pred_class, confidence = gradcam.generate(x)

        save_path = f"outputs/gradcam/gradcam_{i+1}_true{'AF' if label==1 else 'Normal'}_pred{'AF' if pred_class==1 else 'Normal'}.png"
        plot_gradcam(X_test[idx], cam, label, pred_class, confidence, save_path)

    print("\n" + "="*50)
    print("Grad-CAM complete!")
    print("Images saved to: outputs/gradcam/")
    print("="*50)
