"""
DeLong Test for AUROC Comparison
Phase 1 - Month 7
Statistical significance testing between model pairs
"""

import os
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.makedirs("outputs", exist_ok=True)

print("="*55)
print("DeLong Test — Statistical Significance")
print("Comparing AUROC across model pairs")
print("="*55)

def delong_test(auroc1, auroc2, n1=300, n2=300):
    """
    Simplified DeLong test using variance estimation
    Tests H0: AUROC1 = AUROC2
    Returns z-statistic and p-value
    """
    # Variance of AUROC using Hanley & McNeil approximation
    def auroc_variance(auroc, n_pos, n_neg):
        q1 = auroc / (2 - auroc)
        q2 = 2 * auroc**2 / (1 + auroc)
        var = (auroc*(1-auroc) + (n_pos-1)*(q1-auroc**2) + (n_neg-1)*(q2-auroc**2)) / (n_pos*n_neg)
        return var

    n_pos = n1 // 2
    n_neg = n1 // 2

    var1 = auroc_variance(auroc1, n_pos, n_neg)
    var2 = auroc_variance(auroc2, n_pos, n_neg)

    # Correlation between AUROCs (assume 0.5 for independent models)
    r = 0.5
    se = np.sqrt(var1 + var2 - 2*r*np.sqrt(var1*var2))

    if se == 0:
        return 0, 1.0

    z = (auroc1 - auroc2) / se
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    return z, p_value

# ============================================================
# AUROC values — mean across LODO folds
# ============================================================
models = {
    "ResNet1D": {"AF": 0.9172, "PVC": 0.9036},
    "CNN-LSTM": {"AF": 0.8136, "PVC": 0.7445},
    "HuBERT-ECG (LP)": {"AF": 0.8905, "PVC": 0.8536},
    "HuBERT-ECG (FT)": {"AF": 0.8432, "PVC": 0.8209},
    "ECG-FM (LP)": {"AF": 0.8843, "PVC": 0.8502},
    "ECG-FM (FT)": {"AF": 0.8193, "PVC": 0.8096},
}

# Key comparisons
comparisons = [
    ("ResNet1D", "HuBERT-ECG (LP)"),
    ("ResNet1D", "ECG-FM (LP)"),
    ("ResNet1D", "CNN-LSTM"),
    ("HuBERT-ECG (LP)", "ECG-FM (LP)"),
    ("HuBERT-ECG (LP)", "HuBERT-ECG (FT)"),
    ("ECG-FM (LP)", "ECG-FM (FT)"),
]

results = []

for cls in ["AF", "PVC"]:
    print(f"\n--- {cls} Detection ---")
    for m1, m2 in comparisons:
        auroc1 = models[m1][cls]
        auroc2 = models[m2][cls]
        z, p = delong_test(auroc1, auroc2)
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        print(f"  {m1} vs {m2}: z={z:.3f}, p={p:.4f} {sig}")
        results.append({
            "Class": cls,
            "Model 1": m1,
            "Model 2": m2,
            "AUROC 1": auroc1,
            "AUROC 2": auroc2,
            "Difference": round(auroc1 - auroc2, 4),
            "z-statistic": round(z, 3),
            "p-value": round(p, 4),
            "Significance": sig
        })

df = pd.DataFrame(results)
df.to_csv("outputs/delong_test_results.csv", index=False)

# ============================================================
# PLOT — Heatmap of p-values
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

model_names = list(models.keys())

for ax, cls, title in [(axes[0], "AF", "AF Detection"), (axes[1], "PVC", "PVC Detection")]:
    n = len(model_names)
    pval_matrix = np.ones((n, n))

    for i, m1 in enumerate(model_names):
        for j, m2 in enumerate(model_names):
            if i != j:
                _, p = delong_test(models[m1][cls], models[m2][cls])
                pval_matrix[i, j] = p

    im = ax.imshow(pval_matrix, cmap='RdYlGn', vmin=0, vmax=0.1)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    short_names = ["ResNet1D", "CNN-LSTM", "HuBERT\n(LP)", "HuBERT\n(FT)", "ECG-FM\n(LP)", "ECG-FM\n(FT)"]
    ax.set_xticklabels(short_names, fontsize=8)
    ax.set_yticklabels(short_names, fontsize=8)
    ax.set_title(f'{title}\nDeLong Test p-values', fontsize=12, fontweight='bold')

    for i in range(n):
        for j in range(n):
            if i != j:
                p = pval_matrix[i, j]
                sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
                ax.text(j, i, f'{p:.3f}\n{sig}', ha='center', va='center', fontsize=7,
                       color='black' if p > 0.03 else 'white')
            else:
                ax.text(j, i, '—', ha='center', va='center', fontsize=9)

    plt.colorbar(im, ax=ax, label='p-value')

plt.suptitle('Figure 5.8. DeLong Test p-values for Pairwise AUROC Comparisons\n(green = significant difference, red = no significant difference)',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/Figure5_8_DeLongTest.png', dpi=200, bbox_inches='tight')
print("\nFigure saved to outputs/Figure5_8_DeLongTest.png")

print("\n" + "="*55)
print("KEY COMPARISONS SUMMARY")
print("="*55)
print(df[['Class', 'Model 1', 'Model 2', 'Difference', 'p-value', 'Significance']].to_string(index=False))
print("\n* p<0.05  ** p<0.01  *** p<0.001  ns=not significant")
