# ECG Arrhythmia Detection Using Foundation Models

> **Cross-Dataset Benchmarking of ECG Foundation Models for AF and PVC Detection**  
> Yaser Karimi & Ali Sadeghzadeh | Staffordshire University | 2026

---

## Overview

This repository contains the code for the paper:

**"Cross-Dataset Generalisation of ECG Foundation Models for Arrhythmia Detection: A Systematic Benchmarking Study with Leave-One-Dataset-Out Evaluation"**

We systematically compare two ECG Foundation Models (ECG-FM, HuBERT-ECG) against two conventional deep learning baselines (ResNet1D, CNN-LSTM) for binary detection of:
- **Atrial Fibrillation (AF)**
- **Premature Ventricular Contractions (PVC)**

using a rigorous **Leave-One-Dataset-Out (LODO)** cross-validation protocol across **5 publicly available ECG datasets**.

---

## Key Results

| Model | Mode | AF AUROC (mean) | PVC AUROC (mean) |
|-------|------|-----------------|------------------|
| ResNet1D | From scratch | 0.917 | 0.904 |
| CNN-LSTM | From scratch | 0.814 | 0.745 |
| HuBERT-ECG | Linear Probe | 0.891 | 0.854 |
| HuBERT-ECG | Fine-tune | 0.843 | 0.821 |
| ECG-FM | Linear Probe | 0.884 | 0.850 |
| ECG-FM | Fine-tune | 0.819 | 0.810 |

> **Note:** All results are preliminary, obtained under constrained CPU conditions with limited training subsets (150 samples/class/dataset). Full-scale GPU experiments are ongoing.

---

## Datasets

All datasets are publicly available via [PhysioNet](https://physionet.org):

| Dataset | Records | Leads | Hz | Country |
|---------|---------|-------|----|---------|
| PTB-XL | 87,205 | 12 | 500 | Germany |
| CPSC2018 | 6,877 | 12 | 500 | China |
| Chapman-Shaoxing | 10,646 | 12 | 500 | China |
| Georgia 12-Lead | 10,344 | 12 | 500 | USA |
| MIT-BIH | 705 | 2 | 360 | USA |

---

## Installation

```bash
# Clone repository
git clone https://github.com/yaseekarimi/ecg-arrhythmia-fm.git
cd ecg-arrhythmia-fm

# Create virtual environment
python -m venv ecg_env
ecg_env\Scripts\activate  # Windows
source ecg_env/bin/activate  # Linux/Mac

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

---

## Project Structure

```
ecg-arrhythmia-fm/
├── scripts/
│   ├── lodo_loader.py          # Leave-One-Dataset-Out data loader
│   ├── baseline3.py            # ResNet1D baseline model
│   ├── cnn_lstm.py             # CNN-LSTM baseline model
│   ├── run_baseline_lodo.py    # Run all baseline LODO experiments (16 runs)
│   ├── run_fm_lodo.py          # Run HuBERT-ECG LODO experiments (16 runs)
│   ├── run_ecgfm_lodo2.py      # Run ECG-FM LODO experiments (16 runs)
│   ├── gradcam2.py             # Grad-CAM interpretability analysis
│   ├── data_efficiency.py      # Data efficiency analysis (ResNet)
│   ├── data_efficiency_ecgfm.py # Data efficiency analysis (ECG-FM vs ResNet)
│   ├── subgroup_analysis3.py   # Demographic subgroup analysis
│   ├── calibration2.py         # Probability calibration (ECE)
│   ├── ensemble2.py            # Ensemble evaluation
│   ├── bootstrap_ci.py         # Bootstrap confidence intervals
│   ├── delong_test.py          # DeLong statistical significance tests
│   └── ablation.py             # Ablation study
├── requirements.txt
└── README.md
```

---

## Usage

### 1. Prepare datasets
Download all datasets from PhysioNet and place them in `data/` directory:
```
data/
├── ptbxl/
├── classification-of-12-lead-ecgs.../training/
│   ├── cpsc_2018/
│   └── georgia/
├── chapman/
└── mitbih/
```

### 2. Run baseline experiments
```bash
python scripts/run_baseline_lodo.py
```

### 3. Run Foundation Model experiments
```bash
python scripts/run_fm_lodo.py        # HuBERT-ECG
python scripts/run_ecgfm_lodo2.py    # ECG-FM proxy
```

### 4. Run supplementary analyses
```bash
python scripts/data_efficiency_ecgfm.py   # Data efficiency
python scripts/subgroup_analysis3.py      # Subgroup analysis
python scripts/calibration2.py            # Calibration
python scripts/bootstrap_ci.py            # Bootstrap CIs
python scripts/delong_test.py             # DeLong tests
python scripts/ablation.py                # Ablation study
```

---

## Experimental Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Learning rate | 0.001 (0.0001 for fine-tune) |
| Batch size | 32 |
| Epochs (baseline) | 10 |
| Epochs (fine-tune) | 5 |
| Max samples/class | 150 (preliminary) |
| Random seed | 42 |
| Hardware | Intel Core i7-13th Gen, CPU |

---

## Supplementary Analyses

| Analysis | Key Finding |
|----------|-------------|
| Data Efficiency | ECG-FM outperforms ResNet at 5% and 25% data fractions |
| Grad-CAM | Model attends to clinically relevant ECG regions |
| Calibration | CNN-LSTM ECE: 0.046 ✅ vs ResNet ECE: 0.813 ❌ |
| Subgroup | Male: 0.944, Female: 0.959, Elderly: 0.915, Middle-aged: 0.996 |
| Ensemble | ResNet+CNN-LSTM AUROC: 0.968, F1: 0.931 |
| Bootstrap CI | ResNet AF: 0.917 (95% CI: 0.858–0.964) |
| DeLong Test | ResNet vs HuBERT AF: p=0.142 (ns) — comparable performance |

---

## Citation

```bibtex
@article{karimi2026ecg,
  title={Cross-Dataset Generalisation of ECG Foundation Models for Arrhythmia Detection},
  author={Karimi, Yaser and Sadeghzadeh, Ali},
  journal={IEEE Journal of Biomedical and Health Informatics},
  year={2026},
  note={Under review}
}
```

---

## License

This project is licensed under the MIT License.

---

## Acknowledgements

We thank the creators of PTB-XL, CPSC2018, Chapman-Shaoxing, Georgia 12-Lead, and MIT-BIH datasets for making their data publicly available through PhysioNet.
