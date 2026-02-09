# astro-physics-sl

**Progetto di fine corso — Statistical Learning (MD2SL)**
**Università degli Studi di Firenze — A.A. 2024-2025**
**Prof.ssa Anna Gottard**

Analisi comparativa sistematica di metodi di apprendimento supervisionato applicati a due dataset di astrofisica: predizione della temperatura critica di superconduttori (regressione) e classificazione di candidati pulsar (classificazione binaria).

---

## Cosa si fa

Si confrontano **13 metodi di regressione** e **5 metodi di classificazione** su dati reali, coprendo l'intero spettro dei metodi visti nel corso: dai modelli lineari classici fino agli ensemble bayesiani.

### Regressione — Superconduttività ($T_c$)

| Gruppo | Metodi |
|--------|--------|
| Baseline | OLS, Regressione polinomiale |
| Regolarizzazione | Ridge, Lasso, Elastic Net, Adaptive Lasso, Group Lasso |
| Selezione | Best Subset (ABESS) |
| Alberi ed ensemble | CART, Bagging, Random Forest, XGBoost, BART |

### Classificazione — Pulsar (HTRU2)

CART, Bagging, Random Forest, XGBoost, BART — ciascuno testato con e senza **SMOTE** per gestire lo sbilanciamento delle classi (91% rumore / 9% pulsar).

### Analisi aggiuntive

- **Variable Importance** confrontata tra metodi (Lasso, RF, XGBoost, BART)
- **Inferenza post-selezione**: Knockoff Filter e Stability Selection dopo Lasso
- **Test di DeLong** per confronto statistico tra curve ROC
- **Impatto di SMOTE** sulle metriche di classificazione
- Visualizzazione dei coefficient path (Ridge, Lasso, Elastic Net)
- Discussione del trade-off bias-varianza sui risultati ottenuti

---

## Dataset

### Superconduttività (regressione)

- **Fonte:** [UCI ML Repository](https://archive.ics.uci.edu/dataset/464/superconductivty+data)
- **n = 21.263** materiali superconduttori, **p = 81** feature
- **Target:** temperatura critica $T_c$ (Kelvin)
- Le 81 feature sono statistiche aggregate (media, std, range, entropia, …) di proprietà atomiche degli elementi che compongono ciascun materiale

### HTRU2 Pulsar (classificazione)

- **Fonte:** [UCI ML Repository](https://archive.ics.uci.edu/dataset/372/htru2)
- **n = 17.898** candidati, **p = 8** feature
- **Target:** pulsar (1) vs. rumore (0)
- Le 8 feature sono media, std, curtosi e asimmetria del profilo integrato e della curva DM-SNR

---

## Struttura del progetto

```
astro-physics-sl/
├── data/
│   ├── raw/               # CSV originali (scaricati da UCI)
│   └── processed/         # Dati preprocessati (.rds)
├── R/
│   ├── 00_setup.R         # Librerie, seed, configurazione globale
│   ├── 01_data_loading.R  # Download e caricamento dati
│   ├── 02_eda.R           # Analisi esplorativa
│   ├── 03_preprocessing.R # Ricette tidymodels
│   ├── 04_cv_setup.R      # Split train/test e fold CV condivisi
│   ├── regression/
│   │   ├── 05a_baseline.R       # OLS e polinomiale
│   │   ├── 05b_regularization.R # Ridge, Lasso, EN, Adaptive, Group
│   │   ├── 05c_subset.R         # ABESS
│   │   ├── 05d_trees.R          # CART, Bagging, RF, XGBoost, BART
│   │   ├── 06_comparison.R      # Confronto finale e grafici
│   │   └── 07_post_selection.R  # Knockoff e Stability Selection
│   ├── classification/
│   │   └── 05_pipeline.R        # Tutti i 5 metodi ± SMOTE
│   └── utils/
│       ├── helpers.R             # Funzioni ausiliarie
│       └── plotting.R            # Funzioni per grafici
├── output/
│   ├── figures/           # Grafici (PNG e PDF)
│   ├── models/            # Modelli salvati (.rds)
│   └── tables/            # Tabelle risultati (.csv, .rds)
└── report/                # Report finale (opzionale)
```

---

## Come far partire il codice

### Prerequisiti

- **R ≥ 4.3** (testato con R 4.5.2)
- Connessione internet per il primo download dei dati

### Installazione

1. **Clona il repository:**

```bash
git clone https://github.com/battles5/astro-physics-sl.git
cd astro-physics-sl
```

2. **Installa le dipendenze.** Apri R nella cartella del progetto e esegui:

```r
# Se hai renv (consigliato):
renv::restore()

# Altrimenti installa manualmente:
install.packages(c(
  "tidyverse", "glmnet", "rpart", "rpart.plot",
  "ranger", "xgboost", "BART", "abess", "leaps",
  "grpreg", "knockoff", "stabs",
  "vip", "pROC", "themis", "rsample",
  "caret", "yardstick", "recipes", "parsnip",
  "ggcorrplot", "gridExtra", "scales"
))
```

### Esecuzione

Gli script vanno eseguiti **nell'ordine numerico**. Ogni script salva i risultati in `output/` e gli script successivi li leggono da lì.

```r
# 1. Setup e caricamento dati
source("R/00_setup.R")
source("R/01_data_loading.R")    # Scarica i CSV da UCI se non presenti
source("R/02_eda.R")             # EDA + grafici esplorativi
source("R/03_preprocessing.R")   # Ricette di preprocessing
source("R/04_cv_setup.R")        # Split 75/25 + 10-fold CV x5

# 2. Regressione (in ordine)
source("R/regression/05a_baseline.R")
source("R/regression/05b_regularization.R")
source("R/regression/05c_subset.R")
source("R/regression/05d_trees.R")
source("R/regression/06_comparison.R")
source("R/regression/07_post_selection.R")

# 3. Classificazione
source("R/classification/05_pipeline.R")
```

### Dati

I dati grezzi (CSV) **non sono inclusi nel repository** per ragioni di dimensione.
Lo script `01_data_loading.R` li scarica automaticamente da UCI al primo avvio.
I file processati (`.rds`) vengono generati automaticamente dagli script.

### Risultati

Tutti i grafici e le tabelle vengono salvati in `output/figures/` e `output/tables/`.
I modelli addestrati vengono salvati in `output/models/`.

---

## Risultati principali

### Regressione

| Metodo | RMSE | R² |
|--------|------|----|
| **Random Forest** | **9.33** | **0.925** |
| XGBoost | 9.37 | 0.924 |
| Bagging | 9.39 | 0.924 |
| BART | 11.59 | 0.884 |
| CART | 12.30 | 0.870 |
| OLS | 17.66 | 0.730 |
| Lasso | 17.72 | 0.729 |
| Ridge | 18.87 | 0.691 |

### Classificazione

| Metodo | AUC | Balanced Acc. |
|--------|-----|---------------|
| **XGBoost** | **0.976** | 0.944 |
| BART | 0.975 | 0.935 |
| RF | 0.969 | 0.938 |
| Bagging | 0.967 | 0.927 |
| CART | 0.939 | 0.913 |

---

## Riferimenti

- James, G., Witten, D., Hastie, T., Tibshirani, R. (2021). *An Introduction to Statistical Learning with Applications in R* (2nd ed.). Springer.
- Hamidieh, K. (2018). A data-driven statistical model for predicting the critical temperature of a superconductor. *Computational Materials Science*, 154, 346–354.
- Lyon, R.J. et al. (2016). Fifty years of pulsar candidate selection. *MNRAS*, 459(1), 1104–1123.

---

## Autore

**Orso Peruzzi** — Università degli Studi di Firenze, Master MD2SL
