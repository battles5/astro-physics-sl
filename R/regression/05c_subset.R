# ============================================================================
# 05c_subset.R — Best Subset Selection
# ============================================================================
# With 81 features, exhaustive search is infeasible (2^81 models).
# Strategy: use abess (Adaptive Best-Subset Selection) which handles
# high-dimensional data via a splicing algorithm.
# Also compare with leaps on a reduced feature set for pedagogical value.
# ============================================================================

if (!exists("reg_train")) source(here::here("R", "04_cv_setup.R"))

section_header("REGRESSION: Best Subset Selection")

# ============================================================================
# 0. Prepare data
# ============================================================================

reg_prep <- prep(recipe_reg_linear, training = reg_train)
train_baked <- bake(reg_prep, new_data = reg_train)
test_baked  <- bake(reg_prep, new_data = reg_test)

X_train <- train_baked %>% select(-critical_temp) %>% as.matrix()
y_train <- train_baked$critical_temp
X_test  <- test_baked  %>% select(-critical_temp) %>% as.matrix()
y_test  <- test_baked$critical_temp

p <- ncol(X_train)

# ============================================================================
# 1. ABESS — Adaptive Best Subset Selection (full p = 81)
# ============================================================================

log_msg(sprintf("Running ABESS on %d features...", p))

# abess uses an efficient splicing algorithm — handles p >> 20
# Tune support size (number of selected variables) via information criterion
abess_fit <- abess(X_train, y_train,
                    family = "gaussian",
                    tune.type = "cv",       # Use cross-validation
                    nfolds = 10,
                    support.size = 1:min(p, 50))  # Test up to 50 variables

# Best model
best_size <- abess_fit$support.size[which.min(abess_fit$tune.value)]
cat(sprintf("\n--- ABESS: Best support size = %d ---\n", best_size))

# Extract selected variables
abess_coefs <- coef(abess_fit, support.size = best_size)
abess_selected <- names(which(abess_coefs[-1] != 0))  # exclude intercept

cat(sprintf("  Selected variables (%d): %s\n",
            length(abess_selected),
            paste(head(abess_selected, 20), collapse = ", ")))

# Predictions on test set
abess_pred <- predict(abess_fit, newx = X_test, support.size = best_size)
abess_metrics <- compute_regression_metrics(y_test, as.vector(abess_pred))
abess_metrics$method <- "Best Subset"

cat("\n--- ABESS Test Performance ---\n")
print(abess_metrics)

# Information criterion plot
p_abess_ic <- tibble(
  support_size = abess_fit$support.size,
  cv_error     = abess_fit$tune.value
) %>%
  ggplot(aes(x = support_size, y = cv_error)) +
  geom_line(color = "#7570b3", linewidth = 0.8) +
  geom_point(color = "#7570b3") +
  geom_vline(xintercept = best_size, linetype = "dashed", color = "red") +
  annotate("text", x = best_size + 1, y = max(abess_fit$tune.value) * 0.95,
           label = sprintf("Best: %d vars", best_size),
           hjust = 0, color = "red") +
  labs(title = "Best Subset Selection (ABESS): CV Error vs. Support Size",
       x = "Number of Selected Variables",
       y = "Cross-Validation Error")

save_plot(p_abess_ic, "reg_abess_cv", width = 8, height = 5)


# ============================================================================
# 2. LEAPS — Exhaustive search on reduced feature set (pedagogical)
# ============================================================================
# Show exhaustive search is feasible only for small p.
# Select top 20 features from Lasso, then run leaps.
# ============================================================================

log_msg("Running leaps on top-20 Lasso-selected features (pedagogical)...")

# Load Lasso selected variables (from 05b)
lasso_cv <- readRDS(here("output", "models", "regression", "lasso_cv.rds"))
lasso_coefs <- coef(lasso_cv, s = "lambda.min")
lasso_imp <- abs(as.vector(lasso_coefs[-1]))
names(lasso_imp) <- colnames(X_train)
top20_lasso <- names(sort(lasso_imp, decreasing = TRUE))[1:20]

X_train_20 <- X_train[, top20_lasso]
X_test_20  <- X_test[, top20_lasso]

# Exhaustive search
leaps_fit <- regsubsets(x = X_train_20, y = y_train,
                        nvmax = 20, method = "exhaustive")
leaps_summary <- summary(leaps_fit)

# Plot BIC, Cp, Adj R²
leaps_df <- tibble(
  n_vars = 1:20,
  BIC    = leaps_summary$bic,
  Cp     = leaps_summary$cp,
  AdjR2  = leaps_summary$adjr2
)

p_bic <- ggplot(leaps_df, aes(x = n_vars, y = BIC)) +
  geom_line() + geom_point() +
  geom_vline(xintercept = which.min(leaps_df$BIC), linetype = "dashed", color = "red") +
  labs(title = "BIC", x = "# Variables", y = "BIC")

p_cp <- ggplot(leaps_df, aes(x = n_vars, y = Cp)) +
  geom_line() + geom_point() +
  geom_abline(slope = 1, intercept = 0, linetype = "dotted", color = "grey50") +
  geom_vline(xintercept = which.min(leaps_df$Cp), linetype = "dashed", color = "red") +
  labs(title = "Mallow's Cp", x = "# Variables", y = "Cp")

p_adjr2 <- ggplot(leaps_df, aes(x = n_vars, y = AdjR2)) +
  geom_line() + geom_point() +
  geom_vline(xintercept = which.max(leaps_df$AdjR2), linetype = "dashed", color = "red") +
  labs(title = "Adjusted R²", x = "# Variables", y = "Adj R²")

p_leaps <- p_bic + p_cp + p_adjr2 +
  plot_annotation(title = "Best Subset Selection (leaps): Criteria on Top 20 Features")

save_plot(p_leaps, "reg_leaps_criteria", width = 14, height = 5)

cat("\n--- leaps: Optimal model sizes ---\n")
cat(sprintf("  BIC best:    %d variables\n", which.min(leaps_df$BIC)))
cat(sprintf("  Cp best:     %d variables\n", which.min(leaps_df$Cp)))
cat(sprintf("  Adj R² best: %d variables\n", which.max(leaps_df$AdjR2)))


# ============================================================================
# 3. Save results
# ============================================================================

save_model(abess_fit, "abess_fit", subdir = "regression")
saveRDS(abess_metrics, here("output", "tables", "reg_subset_results.rds"))
saveRDS(abess_selected, here("output", "tables", "reg_abess_selected.rds"))

log_msg("Best Subset Selection complete.")
