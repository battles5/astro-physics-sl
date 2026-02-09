# astro-physics-sl

**Final course project — Statistical Learning (MD2SL)**
**University of Florence — A.Y. 2024-2025**

A systematic comparative analysis of supervised learning methods applied to two astrophysics datasets: predicting the critical temperature of superconductors (regression) and classifying pulsar candidates (binary classification).

---

## Overview

This project benchmarks **13 regression methods** and **5 classification methods** on real-world data, covering the full spectrum from classical linear models to Bayesian ensemble methods.

### Regression — Superconductivity ($T_c$)

| Group | Methods |
|-------|---------|
| Baselines | OLS, Polynomial Regression |
| Regularization | Ridge, Lasso, Elastic Net, Adaptive Lasso, Group Lasso |
| Subset Selection | Best Subset (ABESS) |
| Tree-based | CART, Bagging, Random Forest, XGBoost, BART |

### Classification — Pulsar (HTRU2)

CART, Bagging, Random Forest, XGBoost, BART — each tested with and without **SMOTE** to handle class imbalance (91% noise / 9% pulsar).

### Additional Analyses

- **Variable Importance** comparison across methods (Lasso, RF, XGBoost, BART)
- **Post-selection inference**: Knockoff Filter and Stability Selection after Lasso
- **DeLong test** for statistical comparison of ROC curves
- **SMOTE impact** analysis on classification metrics
- Coefficient path visualization (Ridge, Lasso, Elastic Net)
- Bias-variance tradeoff discussion on the observed results

---

## Datasets

### Superconductivity (regression)

- **Source:** [UCI ML Repository](https://archive.ics.uci.edu/dataset/464/superconductivty+data)
- **n = 21,263** superconducting materials, **p = 81** features
- **Target:** critical temperature $T_c$ (Kelvin)
- The 81 features are aggregate statistics (mean, std, range, entropy, …) of elemental atomic properties

### HTRU2 Pulsar (classification)

- **Source:** [UCI ML Repository](https://archive.ics.uci.edu/dataset/372/htru2)
- **n = 17,898** candidates, **p = 8** features
- **Target:** pulsar (1) vs. noise (0)
- The 8 features are mean, std, kurtosis and skewness of the integrated profile and DM-SNR curve

---

## Project Structure

```
astro-physics-sl/
├── data/
│   ├── raw/               # Original CSVs (downloaded from UCI)
│   └── processed/         # Preprocessed data (.rds)
├── R/
│   ├── 00_setup.R         # Libraries, seed, global config
│   ├── 01_data_loading.R  # Download and load datasets
│   ├── 02_eda.R           # Exploratory data analysis
│   ├── 03_preprocessing.R # tidymodels recipes
│   ├── 04_cv_setup.R      # Shared train/test split and CV folds
│   ├── regression/
│   │   ├── 05a_baseline.R       # OLS and polynomial
│   │   ├── 05b_regularization.R # Ridge, Lasso, EN, Adaptive, Group
│   │   ├── 05c_subset.R         # ABESS
│   │   ├── 05d_trees.R          # CART, Bagging, RF, XGBoost, BART
│   │   ├── 06_comparison.R      # Final comparison and plots
│   │   └── 07_post_selection.R  # Knockoff and Stability Selection
│   ├── classification/
│   │   └── 05_pipeline.R        # All 5 methods ± SMOTE
│   └── utils/
│       ├── helpers.R             # Helper functions
│       └── plotting.R            # Plotting functions
├── output/
│   ├── figures/           # Plots (PNG and PDF)
│   ├── models/            # Saved models (.rds)
│   └── tables/            # Results tables (.csv, .rds)
└── report/                # Final report (optional)
```

---

## Getting Started

### Prerequisites

- **R ≥ 4.3** (tested with R 4.5.2)
- Internet connection for first-time data download

### Installation

1. **Clone the repository:**

```bash
git clone https://github.com/battles5/astro-physics-sl.git
cd astro-physics-sl
```

2. **Install dependencies.** Open R in the project folder and run:

```r
# If you have renv (recommended):
renv::restore()

# Otherwise install manually:
install.packages(c(
  "tidyverse", "glmnet", "rpart", "rpart.plot",
  "ranger", "xgboost", "BART", "abess", "leaps",
  "grpreg", "knockoff", "stabs",
  "vip", "pROC", "themis", "rsample",
  "caret", "yardstick", "recipes", "parsnip",
  "ggcorrplot", "gridExtra", "scales"
))
```

### Running

Scripts must be executed **in numerical order**. Each script saves results to `output/` and subsequent scripts read from there.

```r
# 1. Setup and data loading
source("R/00_setup.R")
source("R/01_data_loading.R")    # Downloads CSVs from UCI if not present
source("R/02_eda.R")             # EDA + exploratory plots
source("R/03_preprocessing.R")   # Preprocessing recipes
source("R/04_cv_setup.R")        # 75/25 split + 10-fold CV x5

# 2. Regression (in order)
source("R/regression/05a_baseline.R")
source("R/regression/05b_regularization.R")
source("R/regression/05c_subset.R")
source("R/regression/05d_trees.R")
source("R/regression/06_comparison.R")
source("R/regression/07_post_selection.R")

# 3. Classification
source("R/classification/05_pipeline.R")
```

### Data

Raw CSV files are **not included in the repository** due to size.
The script `01_data_loading.R` downloads them automatically from UCI on first run.
Processed files (`.rds`) are generated automatically by the scripts.

### Output

All plots and tables are saved to `output/figures/` and `output/tables/`.
Trained models are saved to `output/models/`.

---

## Main Results

### Regression

| Method | RMSE | R² |
|--------|------|----|
| **Random Forest** | **9.33** | **0.925** |
| XGBoost | 9.37 | 0.924 |
| Bagging | 9.39 | 0.924 |
| BART | 11.59 | 0.884 |
| CART | 12.30 | 0.870 |
| OLS | 17.66 | 0.730 |
| Lasso | 17.72 | 0.729 |
| Ridge | 18.87 | 0.691 |

### Classification

| Method | AUC | Balanced Acc. |
|--------|-----|---------------|
| **XGBoost** | **0.976** | 0.944 |
| BART | 0.975 | 0.935 |
| RF | 0.969 | 0.938 |
| Bagging | 0.967 | 0.927 |
| CART | 0.939 | 0.913 |

---

## References

- James, G., Witten, D., Hastie, T., Tibshirani, R. (2021). *An Introduction to Statistical Learning with Applications in R* (2nd ed.). Springer.
- Hamidieh, K. (2018). A data-driven statistical model for predicting the critical temperature of a superconductor. *Computational Materials Science*, 154, 346–354.
- Lyon, R.J. et al. (2016). Fifty years of pulsar candidate selection. *MNRAS*, 459(1), 1104–1123.


