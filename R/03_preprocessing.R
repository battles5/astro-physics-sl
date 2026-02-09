# ============================================================================
# 03_preprocessing.R — Recipes for preprocessing
# ============================================================================
# Creates two recipe variants for each dataset:
#   - Linear recipe: normalization + encoding (for OLS, Ridge, Lasso, etc.)
#   - Tree recipe:   minimal preprocessing (trees handle raw features)
#
# Superconductivity: all 81 features are continuous → no dummy encoding needed
# HTRU2 Pulsar:      all 8 features are continuous → no dummy encoding needed
# ============================================================================

if (!exists("superconductivity")) source(here::here("R", "01_data_loading.R"))

# ============================================================================
# 1. SUPERCONDUCTIVITY — Regression recipes
# ============================================================================

log_msg("Creating preprocessing recipes for Superconductivity...")

# --- 1a. Linear recipe (for OLS, Ridge, Lasso, EN, Adaptive Lasso, etc.) ----
# Normalize all predictors (zero mean, unit variance) — required for
# penalized methods to treat all features on equal footing.
recipe_reg_linear <- recipe(critical_temp ~ ., data = superconductivity) %>%
  step_nzv(all_predictors()) %>%          # Remove near-zero variance features
  step_normalize(all_numeric_predictors()) # Center and scale

# --- 1b. Tree recipe (for CART, RF, Bagging, XGBoost, BART) ----------------
# Trees are invariant to monotone transformations → no scaling needed.
# Only remove constant features.
recipe_reg_tree <- recipe(critical_temp ~ ., data = superconductivity) %>%
  step_nzv(all_predictors())              # Remove near-zero variance only

# --- 1c. Check: how many features survive nzv filter? ----------------------
reg_linear_prep <- prep(recipe_reg_linear, training = superconductivity)
reg_tree_prep   <- prep(recipe_reg_tree,   training = superconductivity)

n_feats_linear <- ncol(bake(reg_linear_prep, new_data = NULL)) - 1
n_feats_tree   <- ncol(bake(reg_tree_prep,   new_data = NULL)) - 1

log_msg(sprintf("Superconductivity — Linear recipe: %d features after nzv", n_feats_linear))
log_msg(sprintf("Superconductivity — Tree recipe:   %d features after nzv", n_feats_tree))


# ============================================================================
# 2. HTRU2 PULSAR — Classification recipes
# ============================================================================

log_msg("Creating preprocessing recipes for Pulsar...")

# --- 2a. Tree recipe (for CART, Bagging, RF, XGBoost, BART) ----------------
recipe_clf_tree <- recipe(class ~ ., data = pulsar) %>%
  step_nzv(all_predictors())

# --- 2b. SMOTE recipe (for CART, Bagging, RF, XGBoost, BART with SMOTE) ----
# CRITICAL: step_smote() must be applied ONLY inside training folds,
# never to test/validation data. The skip = TRUE default in themis
# ensures this when used inside a tidymodels workflow.
recipe_clf_smote <- recipe(class ~ ., data = pulsar) %>%
  step_nzv(all_predictors()) %>%
  step_smote(class, over_ratio = 1, neighbors = 5)  # Balance to 1:1

# --- 2c. Check features ----------------------------------------------------
clf_tree_prep <- prep(recipe_clf_tree, training = pulsar)
n_feats_clf   <- ncol(bake(clf_tree_prep, new_data = NULL)) - 1

log_msg(sprintf("Pulsar — Tree recipe: %d features after nzv", n_feats_clf))

# Check SMOTE effect (on training data only — for illustration)
clf_smote_prep <- prep(recipe_clf_smote, training = pulsar)
smote_baked    <- bake(clf_smote_prep, new_data = NULL)

cat("\n--- Class distribution BEFORE SMOTE ---\n")
pulsar %>% count(class) %>% mutate(pct = round(n / sum(n) * 100, 1)) %>% print()

cat("\n--- Class distribution AFTER SMOTE (illustration only) ---\n")
smote_baked %>% count(class) %>% mutate(pct = round(n / sum(n) * 100, 1)) %>% print()


# ============================================================================
# 3. Group structure for Group Lasso (Superconductivity)
# ============================================================================
# Features follow the pattern: <statistic>_<property>
# e.g., mean_atomic_mass, std_atomic_mass, range_atomic_mass, ...
# We group by PROPERTY so that all statistics of the same elemental
# property enter/exit the model together.
# ============================================================================

log_msg("Defining Group Lasso groups for Superconductivity...")

# Extract feature names (after nzv filter)
feat_names_reg <- names(bake(reg_tree_prep, new_data = NULL)) %>%
  setdiff("critical_temp")

# Parse property names: remove the statistic prefix
# Pattern: statistic_property → extract property
# Known statistics: mean, wtd_mean, gmean, wtd_gmean, entropy, wtd_entropy,
#                   range, wtd_range, std, wtd_std
# Plus: number_of_elements (standalone)
group_lasso_groups <- tibble(feature = feat_names_reg) %>%
  mutate(
    # Extract the property part (everything after the statistic prefix)
    property = case_when(
      feature == "number_of_elements" ~ "number_of_elements",
      str_starts(feature, "wtd_gmean_")    ~ str_remove(feature, "^wtd_gmean_"),
      str_starts(feature, "wtd_mean_")     ~ str_remove(feature, "^wtd_mean_"),
      str_starts(feature, "wtd_entropy_")  ~ str_remove(feature, "^wtd_entropy_"),
      str_starts(feature, "wtd_range_")    ~ str_remove(feature, "^wtd_range_"),
      str_starts(feature, "wtd_std_")      ~ str_remove(feature, "^wtd_std_"),
      str_starts(feature, "gmean_")        ~ str_remove(feature, "^gmean_"),
      str_starts(feature, "entropy_")      ~ str_remove(feature, "^entropy_"),
      str_starts(feature, "range_")        ~ str_remove(feature, "^range_"),
      str_starts(feature, "mean_")         ~ str_remove(feature, "^mean_"),
      str_starts(feature, "std_")          ~ str_remove(feature, "^std_"),
      TRUE ~ feature
    )
  )

# Create numeric group vector for gglasso
group_levels <- unique(group_lasso_groups$property)
group_lasso_groups <- group_lasso_groups %>%
  mutate(group_id = as.integer(factor(property, levels = group_levels)))

cat("\n--- Group Lasso groups ---\n")
group_lasso_groups %>%
  count(property, group_id, name = "n_features") %>%
  arrange(group_id) %>%
  print(n = Inf)

# Save group vector for use in 05b_regularization.R
group_vector <- group_lasso_groups$group_id
names(group_vector) <- group_lasso_groups$feature

saveRDS(group_lasso_groups, here("data", "processed", "group_lasso_groups.rds"))
saveRDS(group_vector, here("data", "processed", "group_lasso_vector.rds"))

log_msg("Preprocessing complete. Recipes and group structure ready.")
