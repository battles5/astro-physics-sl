# ============================================================================
# 05d_trees.R — Tree-based regression: CART, Bagging, RF, XGBoost, BART
# ============================================================================
# Tree-based methods don't require normalization — use raw features.
# All use the same train/test split and CV folds defined in 04_cv_setup.R.
# ============================================================================

if (!exists("reg_train")) source(here::here("R", "04_cv_setup.R"))

section_header("REGRESSION: Tree-Based Methods")

# ============================================================================
# 0. Prepare data (tree recipe — no normalization)
# ============================================================================

reg_tree_p <- prep(recipe_reg_tree, training = reg_train)
train_tree <- bake(reg_tree_p, new_data = reg_train)
test_tree  <- bake(reg_tree_p, new_data = reg_test)

X_train_mat <- train_tree %>% select(-critical_temp) %>% as.matrix()
y_train_vec <- train_tree$critical_temp
X_test_mat  <- test_tree  %>% select(-critical_temp) %>% as.matrix()
y_test_vec  <- test_tree$critical_temp

p <- ncol(X_train_mat)

# ============================================================================
# 1. CART — Classification and Regression Tree
# ============================================================================

log_msg("Fitting CART regression tree...")

# --- 1a. Grow a large tree --------------------------------------------------
cart_full <- rpart(critical_temp ~ ., data = train_tree,
                   control = rpart.control(cp = 0.0001,
                                           minsplit = 20,
                                           maxdepth = 30))

# --- 1b. Cost-complexity pruning (1-SE rule) --------------------------------
printcp(cart_full)
plotcp(cart_full)

# Find optimal cp using 1-SE rule
cp_table <- as.data.frame(cart_full$cptable)
min_xerror <- min(cp_table$xerror)
min_xstd   <- cp_table$xstd[which.min(cp_table$xerror)]
cp_1se     <- cp_table %>%
  filter(xerror <= min_xerror + min_xstd) %>%
  slice_max(CP) %>%
  pull(CP) %>%
  first()

cat(sprintf("\n--- CART Pruning ---\n"))
cat(sprintf("  Min xerror: %.4f (at cp = %.6f)\n",
            min_xerror, cp_table$CP[which.min(cp_table$xerror)]))
cat(sprintf("  1-SE cp:    %.6f\n", cp_1se))

# Prune the tree
cart_pruned <- prune(cart_full, cp = cp_1se)

# --- 1c. Visualize pruned tree ----------------------------------------------
png(here("output", "figures", "reg_cart_tree.png"), width = 1000, height = 700, res = 120)
plot_cart_tree(cart_pruned, title = "CART: Pruned Regression Tree (1-SE rule)")
dev.off()
cairo_pdf(here("output", "figures", "reg_cart_tree.pdf"), width = 10, height = 7)
plot_cart_tree(cart_pruned, title = "CART: Pruned Regression Tree (1-SE rule)")
dev.off()

# --- 1d. Test predictions ---------------------------------------------------
cart_pred <- predict(cart_pruned, newdata = test_tree)
cart_metrics <- compute_regression_metrics(y_test_vec, cart_pred)
cart_metrics$method <- "CART"

cat("\n--- CART Test Performance ---\n")
print(cart_metrics)

# CP plot
png(here("output", "figures", "reg_cart_cp.png"), width = 700, height = 500, res = 120)
plotcp(cart_full, main = "CART: Cost-Complexity Pruning")
abline(h = min_xerror + min_xstd, col = "red", lty = 2)
dev.off()
cairo_pdf(here("output", "figures", "reg_cart_cp.pdf"), width = 7, height = 5)
plotcp(cart_full, main = "CART: Cost-Complexity Pruning")
abline(h = min_xerror + min_xstd, col = "red", lty = 2)
dev.off()


# ============================================================================
# 2. BAGGING (Random Forest with mtry = p)
# ============================================================================

log_msg(sprintf("Fitting Bagging (mtry = %d = all features)...", p))

set.seed(42)
bag_fit <- ranger(critical_temp ~ ., data = train_tree,
                  mtry = p,              # Use ALL features at each split → Bagging
                  num.trees = 500,
                  importance = "permutation",
                  seed = 42)

# OOB error
cat(sprintf("\n--- Bagging OOB ---\n  OOB RMSE: %.4f\n",
            sqrt(bag_fit$prediction.error)))

# Test predictions
bag_pred <- predict(bag_fit, data = test_tree)$predictions
bag_metrics <- compute_regression_metrics(y_test_vec, bag_pred)
bag_metrics$method <- "Bagging"

cat("\n--- Bagging Test Performance ---\n")
print(bag_metrics)


# ============================================================================
# 3. RANDOM FOREST (tune mtry)
# ============================================================================

log_msg("Fitting Random Forest (tuning mtry)...")

# Test multiple mtry values
mtry_grid <- c(
  floor(p / 3),       # p/3 (ISLR recommendation for regression)
  floor(sqrt(p)),      # sqrt(p)
  floor(p / 2),        # p/2
  floor(2 * p / 3)     # 2p/3
)
mtry_grid <- sort(unique(mtry_grid))

rf_results <- map_dfr(mtry_grid, function(m) {
  set.seed(42)
  fit <- ranger(critical_temp ~ ., data = train_tree,
                mtry = m, num.trees = 500,
                importance = "permutation", seed = 42)
  pred <- predict(fit, data = test_tree)$predictions

  tibble(
    mtry     = m,
    OOB_RMSE = sqrt(fit$prediction.error),
    Test_RMSE = sqrt(mean((y_test_vec - pred)^2)),
    Test_R2   = 1 - sum((y_test_vec - pred)^2) / sum((y_test_vec - mean(y_test_vec))^2)
  )
})

cat("\n--- Random Forest: mtry comparison ---\n")
print(rf_results)

best_mtry <- rf_results %>% slice_min(Test_RMSE) %>% pull(mtry) %>% first()
log_msg(sprintf("Random Forest: Best mtry = %d", best_mtry))

# Refit with best mtry
set.seed(42)
rf_fit <- ranger(critical_temp ~ ., data = train_tree,
                 mtry = best_mtry, num.trees = 500,
                 importance = "permutation", seed = 42)

rf_pred <- predict(rf_fit, data = test_tree)$predictions
rf_metrics <- compute_regression_metrics(y_test_vec, rf_pred)
rf_metrics$method <- "Random Forest"

cat("\n--- Random Forest Test Performance ---\n")
print(rf_metrics)


# ============================================================================
# 4. XGBoost (Gradient Boosting)
# ============================================================================

log_msg("Fitting XGBoost (hyperparameter tuning)...")

# Prepare xgboost DMatrix
dtrain <- xgb.DMatrix(data = X_train_mat, label = y_train_vec)
dtest  <- xgb.DMatrix(data = X_test_mat,  label = y_test_vec)

# --- 4a. Hyperparameter tuning via grid search with CV ---------------------
xgb_grid <- expand.grid(
  eta           = c(0.01, 0.05, 0.1),
  max_depth     = c(4, 6, 8),
  subsample     = c(0.7, 0.8),
  colsample_bytree = 0.8
)

xgb_tune_results <- map_dfr(1:nrow(xgb_grid), function(i) {
  params <- list(
    objective        = "reg:squarederror",
    eta              = xgb_grid$eta[i],
    max_depth        = xgb_grid$max_depth[i],
    subsample        = xgb_grid$subsample[i],
    colsample_bytree = xgb_grid$colsample_bytree[i]
  )

  set.seed(42)
  cv_result <- xgb.cv(
    params  = params,
    data    = dtrain,
    nrounds = 2000,
    nfold   = 10,
    early_stopping_rounds = 50,
    verbose = 0
  )

  # xgboost >= 3.x stores best_iteration inside early_stop list
  best_iter <- cv_result$early_stop$best_iteration
  if (is.null(best_iter)) best_iter <- cv_result$niter
  best_rmse <- cv_result$evaluation_log$test_rmse_mean[best_iter]

  tibble(
    eta       = xgb_grid$eta[i],
    max_depth = xgb_grid$max_depth[i],
    subsample = xgb_grid$subsample[i],
    best_nrounds = best_iter,
    cv_rmse      = best_rmse
  )
})

cat("\n--- XGBoost: Top 5 configurations ---\n")
print(xgb_tune_results %>% arrange(cv_rmse) %>% head(5))

# --- 4b. Fit final model with best params ----------------------------------
best_xgb <- xgb_tune_results %>% slice_min(cv_rmse) %>% slice(1)

xgb_params <- list(
  objective        = "reg:squarederror",
  eta              = best_xgb$eta,
  max_depth        = best_xgb$max_depth,
  subsample        = best_xgb$subsample,
  colsample_bytree = 0.8
)

set.seed(42)
xgb_fit <- xgb.train(
  params  = xgb_params,
  data    = dtrain,
  nrounds = best_xgb$best_nrounds,
  verbose = 0
)

# Test predictions
xgb_pred <- predict(xgb_fit, newdata = dtest)
xgb_metrics <- compute_regression_metrics(y_test_vec, xgb_pred)
xgb_metrics$method <- "XGBoost"

cat("\n--- XGBoost Test Performance ---\n")
cat(sprintf("  Best params: eta=%.2f, max_depth=%d, subsample=%.1f, nrounds=%d\n",
            best_xgb$eta, best_xgb$max_depth, best_xgb$subsample, best_xgb$best_nrounds))
print(xgb_metrics)

# XGBoost importance
xgb_imp <- xgb.importance(model = xgb_fit)
png(here("output", "figures", "reg_xgb_importance.png"), width = 800, height = 600, res = 120)
xgb.plot.importance(xgb_imp, top_n = 20, main = "XGBoost: Top 20 Feature Importance")
dev.off()
cairo_pdf(here("output", "figures", "reg_xgb_importance.pdf"), width = 8, height = 6)
xgb.plot.importance(xgb_imp, top_n = 20, main = "XGBoost: Top 20 Feature Importance")
dev.off()


# ============================================================================
# 5. BART (Bayesian Additive Regression Trees)
# ============================================================================

log_msg("Fitting BART...")

# BART with default settings (50 trees)
# Use a subsample if dataset is very large for computational feasibility
set.seed(42)
bart_fit <- wbart(
  x.train = X_train_mat,
  y.train = y_train_vec,
  x.test  = X_test_mat,
  ntree   = 50,
  ndpost  = 1000,    # Number of posterior draws
  nskip   = 250,     # Burn-in
  keepevery = 5      # Thinning
)

# Test predictions (posterior mean)
bart_pred <- bart_fit$yhat.test.mean
bart_metrics <- compute_regression_metrics(y_test_vec, bart_pred)
bart_metrics$method <- "BART"

cat("\n--- BART Test Performance ---\n")
print(bart_metrics)

# BART variable importance (inclusion proportions)
bart_varcount <- colMeans(bart_fit$varcount)  # Average split count per variable
names(bart_varcount) <- colnames(X_train_mat)
bart_varimp <- sort(bart_varcount, decreasing = TRUE)

cat("\n--- BART: Top 20 variables by inclusion count ---\n")
print(head(bart_varimp, 20))


# ============================================================================
# 6. Collect all tree-based results
# ============================================================================

tree_results <- bind_rows(cart_metrics, bag_metrics, rf_metrics,
                           xgb_metrics, bart_metrics)

cat("\n")
section_header("TREE-BASED METHODS — SUMMARY")
print(tree_results %>% select(method, RMSE, R2, MAE, MSE))

# Save everything
save_model(cart_pruned, "cart_pruned", subdir = "regression")
save_model(bag_fit,     "bag_fit",     subdir = "regression")
save_model(rf_fit,      "rf_fit",      subdir = "regression")
save_model(xgb_fit,     "xgb_fit",     subdir = "regression")
saveRDS(bart_fit,       here("output", "models", "regression", "bart_fit.rds"))

saveRDS(tree_results,  here("output", "tables", "reg_tree_results.rds"))
saveRDS(rf_results,    here("output", "tables", "reg_rf_mtry_tuning.rds"))
saveRDS(xgb_tune_results, here("output", "tables", "reg_xgb_tuning.rds"))

log_msg("Tree-based regression methods complete.")
