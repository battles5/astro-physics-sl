# ============================================================================
# 06_comparison.R — Compare all 13 regression methods
# ============================================================================
# Loads results from all previous scripts, builds a unified comparison
# table, creates variable importance heatmaps, and produces final plots.
# ============================================================================

if (!exists("recipe_reg_linear")) source(here::here("R", "03_preprocessing.R"))

section_header("REGRESSION: Final Comparison (13 Methods)")

# ============================================================================
# 1. Load all results
# ============================================================================

baseline_results <- readRDS(here("output", "tables", "reg_baseline_results.rds"))
reg_results      <- readRDS(here("output", "tables", "reg_regularization_results.rds"))
subset_results   <- readRDS(here("output", "tables", "reg_subset_results.rds"))
tree_results     <- readRDS(here("output", "tables", "reg_tree_results.rds"))

# Combine all results
all_results <- bind_rows(baseline_results, reg_results, subset_results, tree_results)

# Order methods logically
method_order <- c("OLS", "Polynomial", "Best Subset",
                  "Ridge", "Lasso", "Elastic Net", "Adaptive Lasso", "Group Lasso",
                  "CART", "Bagging", "Random Forest", "XGBoost", "BART")
all_results <- all_results %>%
  mutate(method = factor(method, levels = method_order))

cat("\n--- All 13 Regression Methods: Test Set Metrics ---\n")
all_results %>%
  arrange(method) %>%
  select(method, RMSE, R2, MAE, MSE) %>%
  print(n = 13)


# ============================================================================
# 2. Comparison visualizations
# ============================================================================

# --- 2a. RMSE comparison barplot -------------------------------------------
p_rmse <- all_results %>%
  ggplot(aes(x = reorder(method, RMSE), y = RMSE, fill = method)) +
  geom_col(alpha = 0.85, width = 0.7) +
  geom_text(aes(label = sprintf("%.2f", RMSE)), hjust = -0.1, size = 3) +
  coord_flip() +
  scale_fill_manual(values = METHOD_COLORS) +
  labs(title = "Regression Methods: Test RMSE Comparison",
       subtitle = "Superconductivity Critical Temperature (K)",
       x = NULL, y = "Test RMSE (K)") +
  theme(legend.position = "none") +
  scale_y_continuous(expand = expansion(mult = c(0, 0.15)))

save_plot(p_rmse, "reg_rmse_comparison", width = 10, height = 7)

# --- 2b. R² comparison barplot ---------------------------------------------
p_r2 <- all_results %>%
  ggplot(aes(x = reorder(method, R2), y = R2, fill = method)) +
  geom_col(alpha = 0.85, width = 0.7) +
  geom_text(aes(label = sprintf("%.3f", R2)), hjust = -0.1, size = 3) +
  coord_flip() +
  scale_fill_manual(values = METHOD_COLORS) +
  labs(title = "Regression Methods: Test R² Comparison",
       subtitle = "Superconductivity Critical Temperature",
       x = NULL, y = "Test R²") +
  theme(legend.position = "none") +
  scale_y_continuous(expand = expansion(mult = c(0, 0.15)))

save_plot(p_r2, "reg_r2_comparison", width = 10, height = 7)

# --- 2c. Combined RMSE + R² ------------------------------------------------
p_combined <- p_rmse + p_r2 +
  plot_annotation(title = "Regression Methods Comparison — 13 Methods on Superconductivity Data")
save_plot(p_combined, "reg_combined_comparison", width = 16, height = 7)


# ============================================================================
# 3. Variable Importance comparison across methods
# ============================================================================

log_msg("Computing variable importance comparison...")

# Load models
lasso_cv    <- load_model("lasso_cv",    subdir = "regression")
rf_fit      <- load_model("rf_fit",      subdir = "regression")
xgb_fit     <- load_model("xgb_fit",     subdir = "regression")
bart_fit    <- readRDS(here("output", "models", "regression", "bart_fit.rds"))

# Prepare data for VI extraction
reg_prep <- prep(recipe_reg_linear, training = readRDS(here("data", "processed", "splits_and_folds.rds"))$regression$train)
feat_names <- colnames(readRDS(here("data", "processed", "splits_and_folds.rds"))$regression$train %>% select(-critical_temp))

# --- Lasso: absolute coefficient values (normalized) -----------------------
lasso_coefs <- as.vector(coef(lasso_cv, s = "lambda.min"))[-1]
names(lasso_coefs) <- colnames(bake(reg_prep, new_data = NULL) %>% select(-critical_temp))
lasso_vi <- tibble(
  variable   = names(lasso_coefs),
  importance = abs(lasso_coefs) / max(abs(lasso_coefs)) * 100,
  method     = "Lasso"
)

# --- Random Forest: permutation importance ----------------------------------
rf_vi <- tibble(
  variable   = names(rf_fit$variable.importance),
  importance = rf_fit$variable.importance / max(rf_fit$variable.importance) * 100,
  method     = "Random Forest"
)

# --- XGBoost: gain-based importance -----------------------------------------
xgb_imp <- xgb.importance(model = xgb_fit)
xgb_vi <- tibble(
  variable   = xgb_imp$Feature,
  importance = xgb_imp$Gain / max(xgb_imp$Gain) * 100,
  method     = "XGBoost"
)

# --- BART: inclusion proportions --------------------------------------------
bart_varcount <- colMeans(bart_fit$varcount)
bart_vi <- tibble(
  variable   = names(bart_varcount),
  importance = bart_varcount / max(bart_varcount) * 100,
  method     = "BART"
)

# Combine
all_vi <- bind_rows(lasso_vi, rf_vi, xgb_vi, bart_vi)

# Heatmap
p_vi <- plot_vi_heatmap(all_vi, top_n = 20,
                         title = "Variable Importance: Top 20 Features Across Methods")
save_plot(p_vi, "reg_vi_heatmap", width = 10, height = 8)

# Save comparison table
save_table(
  all_results %>% select(method, RMSE, R2, MAE, MSE) %>% arrange(RMSE),
  "reg_all_comparison"
)
saveRDS(all_vi, here("output", "tables", "reg_variable_importance.rds"))


# ============================================================================
# 4. Summary table (formatted for report)
# ============================================================================

cat("\n")
section_header("FINAL REGRESSION RANKING (by Test RMSE)")
all_results %>%
  arrange(RMSE) %>%
  mutate(rank = row_number()) %>%
  select(rank, method, RMSE, R2, MAE) %>%
  print_comparison_table(caption = "All 13 Regression Methods — Test Set Performance")

log_msg("Regression comparison complete. All figures saved to output/figures/.")
