# ============================================================================
# 04_cv_setup.R — Train/test split and cross-validation folds
# ============================================================================
# Defines a single set of CV folds shared across ALL methods for fair
# comparison. Uses stratified sampling on the target.
#
# Strategy:
#   - 75/25 train/test split (stratified)
#   - 10-fold CV repeated 5 times on training set (same folds for all methods)
#   - For methods with hyperparameters: nested CV
#     (outer = the shared folds, inner = 5-fold within each outer training set)
# ============================================================================

if (!exists("recipe_reg_linear")) source(here::here("R", "03_preprocessing.R"))

# ============================================================================
# 1. SUPERCONDUCTIVITY — Regression split
# ============================================================================

log_msg("Setting up train/test split for Superconductivity...")

# Stratified split on critical_temp (stratify on quartiles for continuous target)
set.seed(42)
reg_split <- initial_split(superconductivity, prop = 0.75, strata = critical_temp)

reg_train <- training(reg_split)
reg_test  <- testing(reg_split)

log_msg(sprintf("Regression — Train: %d | Test: %d", nrow(reg_train), nrow(reg_test)))

# --- Shared 10-fold CV (repeated 5 times) on training set ---
set.seed(42)
reg_folds <- vfold_cv(reg_train, v = 10, repeats = 5, strata = critical_temp)

log_msg(sprintf("Regression — CV folds: %d-fold x %d repeats = %d total",
                10, 5, nrow(reg_folds)))


# ============================================================================
# 2. HTRU2 PULSAR — Classification split
# ============================================================================

log_msg("Setting up train/test split for Pulsar...")

# Stratified split on class (preserves 91/9 ratio in both sets)
set.seed(42)
clf_split <- initial_split(pulsar, prop = 0.75, strata = class)

clf_train <- training(clf_split)
clf_test  <- testing(clf_split)

log_msg(sprintf("Classification — Train: %d | Test: %d", nrow(clf_train), nrow(clf_test)))

# Verify stratification
cat("\n--- Class distribution in train set ---\n")
clf_train %>%
  count(class) %>%
  mutate(pct = round(n / sum(n) * 100, 1)) %>%
  print()

cat("\n--- Class distribution in test set ---\n")
clf_test %>%
  count(class) %>%
  mutate(pct = round(n / sum(n) * 100, 1)) %>%
  print()


# ============================================================================
# 3. Save splits and folds for reproducibility
# ============================================================================

splits_and_folds <- list(
  regression = list(
    split  = reg_split,
    train  = reg_train,
    test   = reg_test,
    folds  = reg_folds
  ),
  classification = list(
    split  = clf_split,
    train  = clf_train,
    test   = clf_test
  )
)

saveRDS(splits_and_folds, here("data", "processed", "splits_and_folds.rds"))

log_msg("CV setup complete. All splits and folds saved.")

# ============================================================================
# 4. Summary
# ============================================================================

cat("\n")
section_header("CV SETUP SUMMARY")

cat(sprintf("
REGRESSION (Superconductivity)
  Total observations:  %d
  Training set:        %d (75%%)
  Test set:            %d (25%%)
  CV strategy:         10-fold, repeated 5 times (50 resamples)
  Stratification:      On critical_temp quartiles

CLASSIFICATION (HTRU2 Pulsar)
  Total observations:  %d
  Training set:        %d (75%%)
  Test set:            %d (25%%)
  Stratification:      On class (preserves ~91/9 ratio)
  SMOTE:               Applied inside training folds only (recipe_clf_smote)
",
nrow(superconductivity), nrow(reg_train), nrow(reg_test),
nrow(pulsar), nrow(clf_train), nrow(clf_test)
))
