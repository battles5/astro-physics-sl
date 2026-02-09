# ============================================================================
# 05a_baseline.R — Baseline regression models: OLS + Polynomial
# ============================================================================
# These serve as reference points for all subsequent methods.
# No hyperparameter tuning — just fit and evaluate.
# ============================================================================

if (!exists("reg_train")) source(here::here("R", "04_cv_setup.R"))

section_header("REGRESSION BASELINES: OLS & Polynomial")

# ============================================================================
# 1. OLS — Ordinary Least Squares
# ============================================================================

log_msg("Fitting OLS regression...")

# --- 1a. Fit on training set ------------------------------------------------
ols_fit <- lm(critical_temp ~ ., data = reg_train)

# --- 1b. Training set diagnostics ------------------------------------------
ols_summary <- summary(ols_fit)
cat("\n--- OLS Summary ---\n")
cat(sprintf("  R²:          %.4f\n", ols_summary$r.squared))
cat(sprintf("  Adj R²:      %.4f\n", ols_summary$adj.r.squared))
cat(sprintf("  AIC:         %.1f\n", AIC(ols_fit)))
cat(sprintf("  BIC:         %.1f\n", BIC(ols_fit)))
cat(sprintf("  # Coefs:     %d (incl. intercept)\n", length(coef(ols_fit))))

# Mallow's Cp (Cp = p for full model, but useful to report)
n_train <- nrow(reg_train)
p_full  <- length(coef(ols_fit))
rss_ols <- sum(residuals(ols_fit)^2)
sigma2_hat <- rss_ols / (n_train - p_full)

# --- 1c. Predictions on test set -------------------------------------------
ols_pred <- predict(ols_fit, newdata = reg_test)
ols_metrics <- compute_regression_metrics(reg_test$critical_temp, ols_pred)
ols_metrics$method <- "OLS"

cat("\n--- OLS Test Set Performance ---\n")
print(ols_metrics)

# --- 1d. Cross-validation RMSE (using shared folds) -------------------------
log_msg("OLS: Running 10-fold CV (repeated 5x)...")

ols_cv_results <- map_dfr(1:nrow(reg_folds), function(i) {
  fold <- reg_folds$splits[[i]]
  train_fold <- analysis(fold)
  val_fold   <- assessment(fold)

  fit <- lm(critical_temp ~ ., data = train_fold)
  pred <- predict(fit, newdata = val_fold)

  tibble(
    fold_id = reg_folds$id[[i]],
    repeat_id = reg_folds$id2[[i]],
    RMSE = sqrt(mean((val_fold$critical_temp - pred)^2)),
    R2   = 1 - sum((val_fold$critical_temp - pred)^2) /
               sum((val_fold$critical_temp - mean(val_fold$critical_temp))^2)
  )
})

cat("\n--- OLS CV Results ---\n")
cat(sprintf("  CV RMSE: %.4f (± %.4f)\n",
            mean(ols_cv_results$RMSE), sd(ols_cv_results$RMSE)))
cat(sprintf("  CV R²:   %.4f (± %.4f)\n",
            mean(ols_cv_results$R2), sd(ols_cv_results$R2)))


# ============================================================================
# 2. POLYNOMIAL REGRESSION
# ============================================================================
# Since we have 81 features, full polynomial expansion is infeasible.
# Strategy: fit degree-2 polynomials of the top-K most correlated features
# with the target, plus all remaining features linearly.
# ============================================================================

log_msg("Fitting Polynomial regression (degree 2, top features)...")

# --- 2a. Select top features for polynomial terms --------------------------
# Use correlation with target to select features for polynomial expansion
cor_with_target <- cor(reg_train %>% select(-critical_temp),
                       reg_train$critical_temp) %>%
  as.data.frame() %>%
  rownames_to_column("feature") %>%
  rename(cor = V1) %>%
  arrange(desc(abs(cor)))

# Top 10 most correlated features get degree-2 terms
top_poly_features <- head(cor_with_target$feature, 10)
cat("\n--- Features selected for polynomial expansion ---\n")
print(top_poly_features)

# --- 2b. Create polynomial formula -----------------------------------------
# All features linearly + poly(top10, degree=2)
poly_formula <- as.formula(paste0(
  "critical_temp ~ . + ",
  paste0("I(", top_poly_features, "^2)", collapse = " + ")
))

# --- 2c. Fit on training set ------------------------------------------------
poly_fit <- lm(poly_formula, data = reg_train)

poly_summary <- summary(poly_fit)
cat("\n--- Polynomial Regression Summary ---\n")
cat(sprintf("  R²:          %.4f\n", poly_summary$r.squared))
cat(sprintf("  Adj R²:      %.4f\n", poly_summary$adj.r.squared))
cat(sprintf("  AIC:         %.1f\n", AIC(poly_fit)))
cat(sprintf("  BIC:         %.1f\n", BIC(poly_fit)))
cat(sprintf("  # Coefs:     %d (incl. intercept)\n", length(coef(poly_fit))))

# --- 2d. Predictions on test set -------------------------------------------
poly_pred <- predict(poly_fit, newdata = reg_test)
poly_metrics <- compute_regression_metrics(reg_test$critical_temp, poly_pred)
poly_metrics$method <- "Polynomial"

cat("\n--- Polynomial Test Set Performance ---\n")
print(poly_metrics)

# --- 2e. Cross-validation RMSE ----------------------------------------------
log_msg("Polynomial: Running 10-fold CV (repeated 5x)...")

poly_cv_results <- map_dfr(1:nrow(reg_folds), function(i) {
  fold <- reg_folds$splits[[i]]
  train_fold <- analysis(fold)
  val_fold   <- assessment(fold)

  fit <- lm(poly_formula, data = train_fold)
  pred <- predict(fit, newdata = val_fold)

  tibble(
    fold_id = reg_folds$id[[i]],
    repeat_id = reg_folds$id2[[i]],
    RMSE = sqrt(mean((val_fold$critical_temp - pred)^2)),
    R2   = 1 - sum((val_fold$critical_temp - pred)^2) /
               sum((val_fold$critical_temp - mean(val_fold$critical_temp))^2)
  )
})

cat("\n--- Polynomial CV Results ---\n")
cat(sprintf("  CV RMSE: %.4f (± %.4f)\n",
            mean(poly_cv_results$RMSE), sd(poly_cv_results$RMSE)))
cat(sprintf("  CV R²:   %.4f (± %.4f)\n",
            mean(poly_cv_results$R2), sd(poly_cv_results$R2)))


# ============================================================================
# 3. Save results
# ============================================================================

baseline_results <- bind_rows(ols_metrics, poly_metrics)
baseline_cv <- list(OLS = ols_cv_results, Polynomial = poly_cv_results)

save_model(ols_fit,  "ols_fit",  subdir = "regression")
save_model(poly_fit, "poly_fit", subdir = "regression")
saveRDS(baseline_results, here("output", "tables", "reg_baseline_results.rds"))
saveRDS(baseline_cv, here("output", "tables", "reg_baseline_cv.rds"))

# --- Diagnostic plots -------------------------------------------------------
# Residuals vs fitted for OLS
p_resid <- ggplot(tibble(fitted = fitted(ols_fit), residuals = residuals(ols_fit)),
                  aes(x = fitted, y = residuals)) +
  geom_point(alpha = 0.2, size = 0.5) +
  geom_hline(yintercept = 0, color = "red", linetype = "dashed") +
  geom_smooth(method = "loess", se = FALSE, color = "blue", linewidth = 0.8) +
  labs(title = "OLS: Residuals vs Fitted Values",
       x = "Fitted Values (K)", y = "Residuals (K)")

save_plot(p_resid, "reg_ols_residuals", width = 8, height = 5)

log_msg("Baseline models complete.")
