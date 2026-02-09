# ============================================================================
# 00_setup.R — Global configuration, libraries, and helper loading
# ============================================================================
# Run this script first. It loads all required packages, sets the global seed,
# configures plotting defaults, and sources utility functions.
# ============================================================================

# --- Global seed for reproducibility ----------------------------------------
set.seed(42)

# --- Core tidyverse + tidymodels --------------------------------------------
library(tidyverse)      # dplyr, ggplot2, tidyr, readr, purrr, stringr, forcats
library(tidymodels)      # parsnip, recipes, rsample, tune, yardstick, workflows

# --- Model engines ----------------------------------------------------------
library(glmnet)          # Ridge, Lasso, Elastic Net, Adaptive Lasso
library(gglasso)         # Group Lasso
library(leaps)           # Best Subset Selection
library(rpart)           # CART
library(rpart.plot)      # CART visualization
library(ranger)          # Random Forest / Bagging (fast implementation)
library(xgboost)         # Gradient Boosting
library(BART)            # Bayesian Additive Regression Trees
library(abess)           # Adaptive Best-Subset Selection

# --- Evaluation & variable importance ---------------------------------------
library(vip)             # Permutation-based variable importance
library(pROC)            # ROC curves, AUC, DeLong test

# --- Class imbalance --------------------------------------------------------
library(themis)          # SMOTE inside recipes (step_smote)

# --- Post-selection inference -----------------------------------------------
library(knockoff)        # Knockoff filter for FDR-controlled selection
library(stabs)           # Stability selection

# --- EDA & visualization utilities ------------------------------------------
library(skimr)           # Quick data summaries
library(corrplot)        # Correlation heatmaps
library(patchwork)       # Combine ggplots
library(janitor)         # Clean column names

# --- Project management -----------------------------------------------------
library(here)            # Relative paths from project root

# --- ggplot2 global theme ---------------------------------------------------
theme_set(
  theme_minimal(base_size = 12) +
    theme(
      plot.title    = element_text(face = "bold", hjust = 0.5),
      plot.subtitle = element_text(hjust = 0.5, color = "grey40"),
      legend.position = "bottom"
    )
)

# Color palette for methods (consistent across all plots)
METHOD_COLORS <- c(
  "OLS"            = "#1b9e77",
  "Polynomial"     = "#d95f02",
  "Best Subset"    = "#7570b3",
  "Ridge"          = "#e7298a",
  "Lasso"          = "#66a61e",
  "Elastic Net"    = "#e6ab02",
  "Adaptive Lasso" = "#a6761d",
  "Group Lasso"    = "#666666",
  "CART"           = "#1f78b4",
  "Bagging"        = "#b2df8a",
  "Random Forest"  = "#33a02c",
  "XGBoost"        = "#fb9a99",
  "BART"           = "#e31a1c"
)

# --- Source utility functions -----------------------------------------------
source(here("R", "utils", "helpers.R"))
source(here("R", "utils", "plotting.R"))

log_msg("Setup complete. All libraries loaded.")
log_msg(paste("R version:", R.version.string))
log_msg(paste("Working directory:", here()))
