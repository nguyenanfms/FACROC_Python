# Evaluation of Fairness in Clustering via FACROC

This repository contains the source code for our machine learning project reproducing and evaluating **FACROC** (Fairness Area under the Curve for ROC of Clustering), a metric designed to assess the trade-off between clustering quality (represented by AUCC) and demographic representation across sensitive groups.

The project is a replication study based on the paper: *Evaluating Fairness in Clustering via FACROC* (2025).

---

## 📂 Project Structure

```text
├── dataset/                  # Datasets (Ignored by Git, see Setup to download)
│   ├── adult/
│   ├── compas/
│   ├── default+of+credit+card+clients/
│   ├── statlog+german+credit+data/
│   ├── student+performance/
│   └── processed/            # Preprocessed datasets (tracked by Git)
├── notebooks/
│   ├── preprocessing.ipynb   # Data cleaning and feature scaling notebook
│   └── clustering.ipynb      # Algorithms execution and FACROC evaluation notebook
├── src/
│   ├── models/               # 5 clustering algorithms built from scratch
│   │   ├── kmeans.py            - K-Means with K-Means++ initialization
│   │   ├── hierarchical.py      - Agglomerative Hierarchical (Ward's Linkage)
│   │   ├── fairlet.py           - Classical Fairlet Decomposition (Greedy)
│   │   ├── scalable_fair.py     - Scalable Fairlet Decomposition (KD-Tree)
│   │   └── proportional_fair.py - Proportional Fair Clustering (Greedy Capture)
│   └── utils/                # Utility scripts
│       ├── data_loader.py       - Custom dataset loading and caching
│       ├── facroc.py            - AUCC and FACROC metric calculation
│       ├── facroc_experiments.py- Automated batch experiments runner
│       ├── FACROC.R             - Reference R implementation
│       └── FACROC_experiments.R - Reference R experiments script
├── reports/
│   ├── experimental_results.md  - Summary table of findings
│   └── figures/                 - ROC slice plots generated for each dataset (6x5 layout)
└── .gitignore                # Git ignore configurations (cache, data, build artifacts)
```

---

## 🛠️ Setup & Installation

### 1. Prerequisites
- Python 3.11+
- Recommended packages (install manually or via pip):
  ```bash
  pip install numpy pandas scikit-learn matplotlib seaborn notebook
  ```

### 2. Download Datasets
Place the raw data in the respective folders under `dataset/` (refer to `notebooks/preprocessing.ipynb` for links and details):
- **Adult**: UCI Machine Learning Repository
- **COMPAS**: ProPublica COMPAS dataset
- **German Credit**: Statlog German Credit Dataset (UCI)
- **Credit Card**: Default of Credit Card Clients Dataset (UCI)
- **Student Performance**: Student Performance Dataset (UCI)

---

## 🚀 How to Run

1. **Preprocessing**:
   Open and run all cells in `notebooks/preprocessing.ipynb` to clean the raw data and save them into `dataset/processed/`.
   
2. **Clustering & Evaluation**:
   Open and run `notebooks/clustering.ipynb`. This will:
   - Load the preprocessed datasets.
   - Run the 5 clustering algorithms (K-Means, Hierarchical, Fairlet, Scalable Fair, Proportional Fair).
   - Compute AUCC and FACROC scores.
   - Plot the ROC curves and save them to `reports/figures/`.

---

## 📊 Summary of Results

Detailed experimental comparisons between the original paper's reported values and our reproduced results are available in [reports/experimental_results.md](file:///d:/HK3-UIT-2025_2026/%C4%90%E1%BB%93%20%C3%A1n/reports/experimental_results.md).

The visual comparisons and the 6x5 grid showing all 30 ROC curves can be viewed directly inside the executed notebook: **[notebooks/clustering.ipynb](file:///d:/HK3-UIT-2025_2026/%C4%90%E1%BB%93%20%C3%A1n/notebooks/clustering.ipynb)**.
