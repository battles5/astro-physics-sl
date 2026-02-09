# ============================================================================
# 05b_regularization.R — Penalized regression: Ridge, Lasso, EN, Adaptive, Group
# ============================================================================
# All methods use glmnet (except Group Lasso → gglasso).
# Lambda tuning via cv.glmnet with the SAME folds used across all methods.
# ============================================================================

if (!exists("reg_train")) source(here::here("R", "04_cv_setup.R"))

section_header("REGRESSION: Regularization Methods")

# ============================================================================
# 0. Prepare data matrices (glmnet requires matrix input)
# ============================================================================

# Prep the linear recipe (normalize + nzv removal) on training data
reg_prep <- prep(recipe_reg_linear, training = reg_train)

train_baked <- bake(reg_prep, new_data = reg_train)
test_baked  <- bake(reg_prep, new_data = reg_test)

X_train <- train_baked %>% select(-critical_temp) %>% as.matrix()
y_train <- train_baked$critical_temp

X_test <- test_baked %>% select(-critical_temp) %>% as.matrix()
y_test <- test_baked$critical_temp  # same as reg_test$critical_temp

# Create fold IDs for cv.glmnet (map from rsample folds to integer vector)
# We use simple 10-fold for cv.glmnet (it doesn't support repeats natively)
set.seed(42)
cv_fold_ids <- sample(rep(1:10, length.out = nrow(X_train)))

# ============================================================================
# 1. RIDGE REGRESSION (alpha = 0)
# ============================================================================

log_msg("Fitting Ridge regression...")

ridge_cv <- cv.glmnet(X_train, y_train, alpha = 0,
                       foldid = cv_fold_ids, nfolds = 10)

# Coefficient path plot
png(here("output", "figures", "reg_ridge_path.png"), width = 800, height = 500, res = 120)
plot_coefficient_path(ridge_cv$glmnet.fit, title = "Ridge: Coefficient Paths",
                      lambda_min = ridge_cv$lambda.min,
                      lambda_1se = ridge_cv$lambda.1se)
dev.off()
cairo_pdf(here("output", "figures", "reg_ridge_path.pdf"), width = 8, height = 5)
plot_coefficient_path(ridge_cv$glmnet.fit, title = "Ridge: Coefficient Paths",
                      lambda_min = ridge_cv$lambda.min,
                      lambda_1se = ridge_cv$lambda.1se)
dev.off()

# CV plot
png(here("output", "figures", "reg_ridge_cv.png"), width = 700, height = 500, res = 120)
plot(ridge_cv, main = "Ridge: Cross-Validation Error")
dev.off()
cairo_pdf(here("output", "figures", "reg_ridge_cv.pdf"), width = 7, height = 5)
plot(ridge_cv, main = "Ridge: Cross-Validation Error")
dev.off()

# Test predictions
ridge_pred_min <- predict(ridge_cv, newx = X_test, s = "lambda.min")
ridge_pred_1se <- predict(ridge_cv, newx = X_test, s = "lambda.1se")

ridge_metrics <- compute_regression_metrics(y_test, as.vector(ridge_pred_min))
ridge_metrics$method <- "Ridge"

cat("\n--- Ridge Results ---\n")
cat(sprintf("  lambda.min: %.6f | lambda.1se: %.6f\n",
            ridge_cv$lambda.min, ridge_cv$lambda.1se))
cat(sprintf("  # Non-zero coefs (lambda.min): %d\n",
            sum(coef(ridge_cv, s = "lambda.min") != 0) - 1))
print(ridge_metrics)


# ============================================================================
# 2. LASSO (alpha = 1)
# ============================================================================

log_msg("Fitting Lasso regression...")

lasso_cv <- cv.glmnet(X_train, y_train, alpha = 1,
                       foldid = cv_fold_ids, nfolds = 10)

# Coefficient path
png(here("output", "figures", "reg_lasso_path.png"), width = 800, height = 500, res = 120)
plot_coefficient_path(lasso_cv$glmnet.fit, title = "Lasso: Coefficient Paths",
                      lambda_min = lasso_cv$lambda.min,
                      lambda_1se = lasso_cv$lambda.1se)
dev.off()
cairo_pdf(here("output", "figures", "reg_lasso_path.pdf"), width = 8, height = 5)
plot_coefficient_path(lasso_cv$glmnet.fit, title = "Lasso: Coefficient Paths",
                      lambda_min = lasso_cv$lambda.min,
                      lambda_1se = lasso_cv$lambda.1se)
dev.off()

# CV plot
png(here("output", "figures", "reg_lasso_cv.png"), width = 700, height = 500, res = 120)
plot(lasso_cv, main = "Lasso: Cross-Validation Error")
dev.off()
cairo_pdf(here("output", "figures", "reg_lasso_cv.pdf"), width = 7, height = 5)
plot(lasso_cv, main = "Lasso: Cross-Validation Error")
dev.off()

# Test predictions
lasso_pred <- predict(lasso_cv, newx = X_test, s = "lambda.min")
lasso_metrics <- compute_regression_metrics(y_test, as.vector(lasso_pred))
lasso_metrics$method <- "Lasso"

# Selected variables
lasso_coefs <- coef(lasso_cv, s = "lambda.min")
lasso_selected <- rownames(lasso_coefs)[which(lasso_coefs != 0)]
lasso_selected <- setdiff(lasso_selected, "(Intercept)")

cat("\n--- Lasso Results ---\n")
cat(sprintf("  lambda.min: %.6f | lambda.1se: %.6f\n",
            lasso_cv$lambda.min, lasso_cv$lambda.1se))
cat(sprintf("  # Selected variables: %d / %d\n",
            length(lasso_selected), ncol(X_train)))
cat(sprintf("  Selected: %s\n", paste(head(lasso_selected, 20), collapse = ", ")))
print(lasso_metrics)


# ============================================================================
# 3. ELASTIC NET (tune both alpha and lambda)
# ============================================================================

log_msg("Fitting Elastic Net (tuning alpha and lambda)...")

# Grid of alpha values
alpha_grid <- seq(0.1, 0.9, by = 0.1)

en_results <- map_dfr(alpha_grid, function(a) {
  cv_fit <- cv.glmnet(X_train, y_train, alpha = a,
                       foldid = cv_fold_ids, nfolds = 10)
  tibble(
    alpha      = a,
    lambda_min = cv_fit$lambda.min,
    lambda_1se = cv_fit$lambda.1se,
    cv_mse_min = min(cv_fit$cvm),
    n_nonzero  = sum(coef(cv_fit, s = "lambda.min") != 0) - 1
  )
})

cat("\n--- Elastic Net: Alpha tuning results ---\n")
print(en_results)

# Select best alpha
best_alpha <- en_results %>% slice_min(cv_mse_min) %>% pull(alpha) %>% first()
log_msg(sprintf("Elastic Net: Best alpha = %.1f", best_alpha))

# Refit with best alpha
en_cv <- cv.glmnet(X_train, y_train, alpha = best_alpha,
                    foldid = cv_fold_ids, nfolds = 10)

# Coefficient path
png(here("output", "figures", "reg_en_path.png"), width = 800, height = 500, res = 120)
plot_coefficient_path(en_cv$glmnet.fit,
                      title = sprintf("Elastic Net (α=%.1f): Coefficient Paths", best_alpha),
                      lambda_min = en_cv$lambda.min,
                      lambda_1se = en_cv$lambda.1se)
dev.off()
cairo_pdf(here("output", "figures", "reg_en_path.pdf"), width = 8, height = 5)
plot_coefficient_path(en_cv$glmnet.fit,
                      title = sprintf("Elastic Net (α=%.1f): Coefficient Paths", best_alpha),
                      lambda_min = en_cv$lambda.min,
                      lambda_1se = en_cv$lambda.1se)
dev.off()

# Test predictions
en_pred <- predict(en_cv, newx = X_test, s = "lambda.min")
en_metrics <- compute_regression_metrics(y_test, as.vector(en_pred))
en_metrics$method <- "Elastic Net"

cat("\n--- Elastic Net Results ---\n")
print(en_metrics)


# ============================================================================
# 4. ADAPTIVE LASSO (two-stage)
# ============================================================================
# Stage 1: Fit Ridge to get initial coefficient estimates
# Stage 2: Weighted Lasso with penalty.factor = 1 / |beta_ridge|^gamma
# ============================================================================

log_msg("Fitting Adaptive Lasso (two-stage)...")

# Stage 1: Ridge coefficients as weights
ridge_for_weights <- glmnet(X_train, y_train, alpha = 0,
                             lambda = ridge_cv$lambda.min)
beta_ridge <- as.vector(coef(ridge_for_weights))[-1]  # exclude intercept

# Adaptive weights (gamma = 1 is standard)
gamma_adapt <- 1
adapt_weights <- 1 / (abs(beta_ridge)^gamma_adapt + 1e-6)  # avoid division by zero

# Stage 2: Weighted Lasso
adapt_cv <- cv.glmnet(X_train, y_train, alpha = 1,
                       penalty.factor = adapt_weights,
                       foldid = cv_fold_ids, nfolds = 10)

# Test predictions
adapt_pred <- predict(adapt_cv, newx = X_test, s = "lambda.min")
adapt_metrics <- compute_regression_metrics(y_test, as.vector(adapt_pred))
adapt_metrics$method <- "Adaptive Lasso"

# Selected variables
adapt_coefs <- coef(adapt_cv, s = "lambda.min")
adapt_selected <- rownames(adapt_coefs)[which(adapt_coefs != 0)]
adapt_selected <- setdiff(adapt_selected, "(Intercept)")

cat("\n--- Adaptive Lasso Results ---\n")
cat(sprintf("  # Selected variables: %d / %d\n",
            length(adapt_selected), ncol(X_train)))
print(adapt_metrics)


# ============================================================================
# 5. GROUP LASSO (gglasso)
# ============================================================================
# Groups defined in 03_preprocessing.R: features grouped by elemental property
# (all statistics of the same property enter/exit together)
# ============================================================================

log_msg("Fitting Group Lasso...")

# Load group vector
group_vector <- readRDS(here("data", "processed", "group_lasso_vector.rds"))

# Align features with group vector (after nzv filter)
feat_names <- colnames(X_train)
group_ids <- group_vector[feat_names]

# Check for any unmatched features
if (any(is.na(group_ids))) {
  unmatched <- feat_names[is.na(group_ids)]
  log_msg(sprintf("WARNING: %d features not in group vector: %s",
                  length(unmatched), paste(head(unmatched), collapse = ", ")),
          level = "WARN")
  # Assign unmatched to their own groups
  max_group <- max(group_ids, na.rm = TRUE)
  for (j in seq_along(unmatched)) {
    group_ids[unmatched[j]] <- max_group + j
  }
}

# Fit Group Lasso with CV
set.seed(42)
grplasso_cv <- cv.gglasso(X_train, y_train,
                           group = as.integer(group_ids),
                           loss = "ls",
                           nfolds = 10)

# Test predictions
grplasso_pred <- predict(grplasso_cv, newx = X_test, s = "lambda.min")
grplasso_metrics <- compute_regression_metrics(y_test, as.vector(grplasso_pred))
grplasso_metrics$method <- "Group Lasso"

# Selected groups
grplasso_coefs <- coef(grplasso_cv, s = "lambda.min")[-1]  # exclude intercept
selected_feats <- feat_names[grplasso_coefs != 0]
selected_groups <- unique(group_ids[selected_feats])

cat("\n--- Group Lasso Results ---\n")
cat(sprintf("  # Selected groups: %d / %d\n",
            length(selected_groups), length(unique(group_ids))))
cat(sprintf("  # Selected features: %d / %d\n",
            length(selected_feats), ncol(X_train)))
print(grplasso_metrics)

# Group Lasso CV plot
png(here("output", "figures", "reg_grplasso_cv.png"), width = 700, height = 500, res = 120)
plot(grplasso_cv, main = "Group Lasso: Cross-Validation Error")
dev.off()
cairo_pdf(here("output", "figures", "reg_grplasso_cv.pdf"), width = 7, height = 5)
plot(grplasso_cv, main = "Group Lasso: Cross-Validation Error")
dev.off()


# ============================================================================
# 6. Collect and save all regularization results
# ============================================================================

reg_results <- bind_rows(ridge_metrics, lasso_metrics, en_metrics,
                          adapt_metrics, grplasso_metrics)

cat("\n")
section_header("REGULARIZATION METHODS — SUMMARY")
print(reg_results %>% select(method, RMSE, R2, MAE, MSE))

# Save everything
save_model(ridge_cv,    "ridge_cv",    subdir = "regression")
save_model(lasso_cv,    "lasso_cv",    subdir = "regression")
save_model(en_cv,       "en_cv",       subdir = "regression")
save_model(adapt_cv,    "adapt_cv",    subdir = "regression")
save_model(grplasso_cv, "grplasso_cv", subdir = "regression")

saveRDS(reg_results, here("output", "tables", "reg_regularization_results.rds"))
saveRDS(en_results,  here("output", "tables", "reg_en_alpha_tuning.rds"))

# Save selected variable lists for comparison
reg_selected_vars <- list(
  Lasso          = lasso_selected,
  Adaptive_Lasso = adapt_selected,
  Group_Lasso    = selected_feats
)
saveRDS(reg_selected_vars, here("output", "tables", "reg_selected_variables.rds"))

log_msg("Regularization methods complete.")
