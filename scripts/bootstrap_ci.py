"""
Bootstrap Confidence Intervals
Phase 1 - Month 7
Compute 95% CI for all model AUROC results
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.makedirs("outputs", exist_ok=True)

print("="*55)
print("Bootstrap Confidence Intervals")
print("95% CI for all LODO AUROC results")
print("="*55)

def bootstrap_ci(auroc_values, n_bootstrap=1000, ci=0.95):
    """
    Compute bootstrap CI for mean AUROC across LODO folds
    """
    bootstraps = []
    n = len(auroc_values)
    for _ in range(n_bootstrap):
        sample = np.random.choice(auroc_values, size=n, replace=True)
        bootstraps.append(np.mean(sample))
    
    alpha = (1 - ci) / 2
    lower = np.percentile(bootstraps, alpha * 100)
    upper = np.percentile(bootstraps, (1 - alpha) * 100)
    mean = np.mean(auroc_values)
    return mean, lower, upper

# ============================================================
# DATA — All LODO results
# ============================================================
results_data = {
    # Model, Class, AUROC values across 4 LODO folds
    ("ResNet1D", "AF"): [0.9672, 0.9224, 0.8256, 0.9537],
    ("ResNet1D", "PVC"): [0.9956, 0.8587, 0.7883, 0.9720],
    ("CNN-LSTM", "AF"): [0.9471, 0.8184, 0.6088, 0.8801],
    ("CNN-LSTM", "PVC"): [0.7704, 0.7157, 0.6716, 0.8201],
    ("HuBERT-ECG (LP)", "AF"): [0.9578, 0.9227, 0.7762, 0.9053],
    ("HuBERT-ECG (LP)", "PVC"): [0.9544, 0.8362, 0.7602, 0.8636],
    ("HuBERT-ECG (FT)", "AF"): [0.9262, 0.8790, 0.7089, 0.8585],
    ("HuBERT-ECG (FT)", "PVC"): [0.9191, 0.7568, 0.7460, 0.8617],
    ("ECG-FM (LP)", "AF"): [0.9533, 0.9328, 0.7564, 0.8948],
    ("ECG-FM (LP)", "PVC"): [0.9619, 0.8154, 0.7627, 0.8608],
    ("ECG-FM (FT)", "AF"): [0.9106, 0.8790, 0.6466, 0.8408],
    ("ECG-FM (FT)", "PVC"): [0.9118, 0.7192, 0.7384, 0.8692],
}

np.random.seed(42)
ci_results = []

print("\nComputing Bootstrap CIs (1000 iterations each)...")
for (model, cls), aurocs in results_data.items():
    mean, lower, upper = bootstrap_ci(aurocs)
    print(f"  {model} ({cls}): {mean:.4f} (95% CI: {lower:.4f} - {upper:.4f})")
    ci_results.append({
        "Model": model,
        "Class": cls,
        "Mean AUROC": round(mean, 4),
        "CI Lower": round(lower, 4),
        "CI Upper": round(upper, 4),
        "CI Width": round(upper - lower, 4)
    })

df = pd.DataFrame(ci_results)
df.to_csv("outputs/bootstrap_ci_results.csv", index=False)

# ============================================================
# PLOT
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 7))

for ax, cls, title in [(axes[0], "AF", "AF Detection"), (axes[1], "PVC", "PVC Detection")]:
    cls_data = df[df['Class'] == cls].reset_index(drop=True)
    y_pos = np.arange(len(cls_data))
    
    colors = ['#2E4057', '#2E4057', '#E84855', '#E84855', '#2E75B6', '#2E75B6', '#F4A261', '#F4A261']
    
    ax.barh(y_pos, cls_data['Mean AUROC'], 
            xerr=[cls_data['Mean AUROC'] - cls_data['CI Lower'],
                  cls_data['CI Upper'] - cls_data['Mean AUROC']],
            color=colors[:len(cls_data)], alpha=0.8,
            error_kw=dict(ecolor='black', capsize=5, capthick=1.5))
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(cls_data['Model'], fontsize=10)
    ax.set_xlabel('Mean AUROC (95% CI)', fontsize=11)
    ax.set_title(f'{title}\nMean AUROC with 95% Bootstrap CI', fontsize=12, fontweight='bold')
    ax.set_xlim([0.5, 1.05])
    ax.axvline(x=0.9, color='gray', linestyle='--', alpha=0.5, label='AUROC = 0.90')
    ax.legend(fontsize=9)
    ax.grid(axis='x', alpha=0.3)
    
    for i, row in cls_data.iterrows():
        ax.text(row['CI Upper'] + 0.005, i, f"{row['Mean AUROC']:.3f}", 
                va='center', fontsize=9, fontweight='bold')

plt.suptitle('Figure 5.7. Bootstrap 95% Confidence Intervals for Mean AUROC\nacross LODO Evaluation Folds', 
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/Figure5_7_BootstrapCI.png', dpi=200, bbox_inches='tight')
print("\nFigure saved to outputs/Figure5_7_BootstrapCI.png")

print("\n" + "="*55)
print("SUMMARY")
print("="*55)
print(df.to_string(index=False))
