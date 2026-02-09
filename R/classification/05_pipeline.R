# ============================================================================
# 05_pipeline.R — Classification: Full Pipeline
# ============================================================================
# Trains 5 tree-based classifiers (CART, Bagging, RF, XGBoost, BART) on the
# HTRU2 Pulsar dataset, both without and with SMOTE.  Compares methods via
# DeLong's test for AUC, and computes variable importance.
#
# Sections:
#   05a  Tree-based methods (no SMOTE)
#   05b  Tree-based methods (with SMOTE)
#   06   Comparison: SMOTE impact + DeLong pairwise tests
#   07   Variable importance heatmap
# ============================================================================

t_start <- Sys.time()
cat(sprintf("[%s] STARTING CLASSIFICATION PIPELINE\n", format(t_start)))

# ============================================================================
# 1. ONE-TIME SETUP (libraries + data + preprocessing + splits)
# ============================================================================

if (!exists("clf_train")) source(here::here("R", "04_cv_setup.R"))

cat(sprintf("[%s] Setup complete in %.0f seconds\n",
            format(Sys.time()), as.numeric(Sys.time() - t_start)))


# ############################################################################
#                     05a — TREE-BASED METHODS (NO SMOTE)
# ############################################################################

section_header("CLASSIFICATION: Tree-Based Methods (No SMOTE)")

# --- Prepare data ---
clf_tree_p <- prep(recipe_clf_tree, training = clf_train)
train_clf  <- bake(clf_tree_p, new_data = clf_train)
test_clf   <- bake(clf_tree_p, new_data = clf_test)

X_train_clf <- train_clf %>% select(-class) %>% as.matrix()
y_train_clf <- train_clf$class
X_test_clf  <- test_clf  %>% select(-class) %>% as.matrix()
y_test_clf  <- test_clf$class

y_train_num <- as.integer(y_train_clf == "pulsar")
y_test_num  <- as.integer(y_test_clf  == "pulsar")
p_clf <- ncol(X_train_clf)

clf_results <- tibble()
roc_objects <- list()

# --- 1. CART ---
log_msg("CART classification...")
set.seed(42)
cart_clf_full <- rpart(class ~ ., data = train_clf,
                       control = rpart.control(cp = 0.0001, minsplit = 20))
cp_table <- as.data.frame(cart_clf_full$cptable)
min_xe <- min(cp_table$xerror)
min_xs <- cp_table$xstd[which.min(cp_table$xerror)]
cp_1se <- cp_table %>% filter(xerror <= min_xe + min_xs) %>%
  slice_max(CP) %>% pull(CP) %>% first()
cart_clf <- prune(cart_clf_full, cp = cp_1se)

png(here("output", "figures", "clf_cart_tree.png"), width = 800, height = 600, res = 120)
plot_cart_tree(cart_clf, title = "CART: Classification Tree (1-SE)")
dev.off()
cairo_pdf(here("output", "figures", "clf_cart_tree.pdf"), width = 8, height = 6)
plot_cart_tree(cart_clf, title = "CART: Classification Tree (1-SE)")
dev.off()

cart_prob <- predict(cart_clf, newdata = test_clf, type = "prob")[, "pulsar"]
cart_pred <- predict(cart_clf, newdata = test_clf, type = "class")
cart_met  <- compute_classification_metrics(y_test_clf, cart_pred, prob = cart_prob, positive_class = "pulsar")
cart_met$method <- "CART"
clf_results <- bind_rows(clf_results, cart_met)
roc_objects[["CART"]] <- pROC::roc(y_test_clf, cart_prob, levels = c("noise", "pulsar"),
                                    direction = "<", quiet = TRUE)
cat("  CART done — AUC:", round(cart_met$AUC, 4), "\n")


# --- 2. Bagging ---
log_msg("Bagging classification...")
set.seed(42)
bag_clf_fit <- ranger(class ~ ., data = train_clf, mtry = p_clf,
                      num.trees = 500, probability = TRUE,
                      importance = "permutation", seed = 42)
bag_prob <- predict(bag_clf_fit, data = test_clf)$predictions[, "pulsar"]
bag_pred <- factor(ifelse(bag_prob >= 0.5, "pulsar", "noise"), levels = c("noise", "pulsar"))
bag_met  <- compute_classification_metrics(y_test_clf, bag_pred, prob = bag_prob, positive_class = "pulsar")
bag_met$method <- "Bagging"
clf_results <- bind_rows(clf_results, bag_met)
roc_objects[["Bagging"]] <- pROC::roc(y_test_clf, bag_prob, levels = c("noise", "pulsar"),
                                       direction = "<", quiet = TRUE)
cat("  Bagging done — AUC:", round(bag_met$AUC, 4), "\n")


# --- 3. Random Forest ---
log_msg("Random Forest classification...")
mtry_grid_clf <- c(floor(sqrt(p_clf)), floor(p_clf / 2))

rf_clf_results <- map_dfr(mtry_grid_clf, function(m) {
  set.seed(42)
  fit <- ranger(class ~ ., data = train_clf, mtry = m, num.trees = 500,
                probability = TRUE, importance = "permutation", seed = 42)
  prob <- predict(fit, data = test_clf)$predictions[, "pulsar"]
  roc_obj <- pROC::roc(y_test_clf, prob, levels = c("noise", "pulsar"),
                        direction = "<", quiet = TRUE)
  tibble(mtry = m, OOB_error = fit$prediction.error, AUC = as.numeric(pROC::auc(roc_obj)))
})
cat("  RF mtry results:\n"); print(rf_clf_results)

best_mtry_clf <- rf_clf_results %>% slice_max(AUC) %>% pull(mtry) %>% first()
set.seed(42)
rf_clf_fit <- ranger(class ~ ., data = train_clf, mtry = best_mtry_clf,
                     num.trees = 500, probability = TRUE,
                     importance = "permutation", seed = 42)
rf_prob <- predict(rf_clf_fit, data = test_clf)$predictions[, "pulsar"]
rf_pred <- factor(ifelse(rf_prob >= 0.5, "pulsar", "noise"), levels = c("noise", "pulsar"))
rf_met  <- compute_classification_metrics(y_test_clf, rf_pred, prob = rf_prob, positive_class = "pulsar")
rf_met$method <- "Random Forest"
clf_results <- bind_rows(clf_results, rf_met)
roc_objects[["Random Forest"]] <- pROC::roc(y_test_clf, rf_prob, levels = c("noise", "pulsar"),
                                              direction = "<", quiet = TRUE)
cat("  RF done — AUC:", round(rf_met$AUC, 4), "\n")


# --- 4. XGBoost ---
log_msg("XGBoost classification...")
scale_pos_weight <- sum(y_train_num == 0) / sum(y_train_num == 1)
dtrain_clf <- xgb.DMatrix(data = X_train_clf, label = y_train_num)
dtest_clf  <- xgb.DMatrix(data = X_test_clf,  label = y_test_num)

xgb_clf_grid <- expand.grid(
  eta       = c(0.01, 0.05),
  max_depth = c(4, 6, 8),
  subsample = 0.8
)

xgb_clf_tune <- map_dfr(1:nrow(xgb_clf_grid), function(i) {
  params <- list(
    objective = "binary:logistic", eval_metric = "auc",
    eta = xgb_clf_grid$eta[i], max_depth = xgb_clf_grid$max_depth[i],
    subsample = xgb_clf_grid$subsample[i], colsample_bytree = 0.8,
    scale_pos_weight = scale_pos_weight
  )
  set.seed(42)
  cv_res <- xgb.cv(params = params, data = dtrain_clf,
                    nrounds = 1500, nfold = 5,
                    early_stopping_rounds = 30, verbose = 0)
  best_iter <- cv_res$early_stop$best_iteration
  if (is.null(best_iter)) best_iter <- cv_res$niter
  tibble(
    eta = xgb_clf_grid$eta[i], max_depth = xgb_clf_grid$max_depth[i],
    subsample = xgb_clf_grid$subsample[i],
    best_nrounds = best_iter,
    cv_auc = cv_res$evaluation_log$test_auc_mean[best_iter]
  )
})

best_xgb_clf <- xgb_clf_tune %>% slice_max(cv_auc) %>% slice(1)
cat("  XGBoost best: eta=", best_xgb_clf$eta, " depth=", best_xgb_clf$max_depth,
    " nrounds=", best_xgb_clf$best_nrounds, " AUC_cv=", round(best_xgb_clf$cv_auc, 4), "\n")

set.seed(42)
xgb_clf_fit <- xgb.train(
  params = list(objective = "binary:logistic",
                eta = best_xgb_clf$eta, max_depth = best_xgb_clf$max_depth,
                subsample = best_xgb_clf$subsample, colsample_bytree = 0.8,
                scale_pos_weight = scale_pos_weight),
  data = dtrain_clf, nrounds = best_xgb_clf$best_nrounds, verbose = 0
)

xgb_clf_prob <- predict(xgb_clf_fit, newdata = dtest_clf)
xgb_clf_pred <- factor(ifelse(xgb_clf_prob >= 0.5, "pulsar", "noise"), levels = c("noise", "pulsar"))
xgb_met <- compute_classification_metrics(y_test_clf, xgb_clf_pred, prob = xgb_clf_prob, positive_class = "pulsar")
xgb_met$method <- "XGBoost"
clf_results <- bind_rows(clf_results, xgb_met)
roc_objects[["XGBoost"]] <- pROC::roc(y_test_clf, xgb_clf_prob, levels = c("noise", "pulsar"),
                                       direction = "<", quiet = TRUE)
cat("  XGBoost done — AUC:", round(xgb_met$AUC, 4), "\n")


# --- 5. BART ---
log_msg("BART classification...")
set.seed(42)
bart_clf_fit <- pbart(
  x.train = X_train_clf, y.train = y_train_num,
  x.test = X_test_clf,
  ntree = 50, ndpost = 1000, nskip = 250, keepevery = 5
)

bart_clf_prob <- colMeans(pnorm(bart_clf_fit$yhat.test))
bart_clf_pred <- factor(ifelse(bart_clf_prob >= 0.5, "pulsar", "noise"), levels = c("noise", "pulsar"))
bart_met <- compute_classification_metrics(y_test_clf, bart_clf_pred, prob = bart_clf_prob, positive_class = "pulsar")
bart_met$method <- "BART"
clf_results <- bind_rows(clf_results, bart_met)
roc_objects[["BART"]] <- pROC::roc(y_test_clf, bart_clf_prob, levels = c("noise", "pulsar"),
                                    direction = "<", quiet = TRUE)
cat("  BART done — AUC:", round(bart_met$AUC, 4), "\n")


# --- Save no-SMOTE results ---
section_header("CLASSIFICATION (NO SMOTE) — SUMMARY")
clf_results %>%
  select(method, Accuracy, Balanced_Accuracy, AUC, Precision, Recall, F1) %>%
  print()

saveRDS(clf_results, here("output", "tables", "clf_no_smote_results.rds"))
saveRDS(roc_objects, here("output", "tables", "clf_no_smote_roc.rds"))

save_model(cart_clf,     "cart_clf",     subdir = "classification")
save_model(bag_clf_fit,  "bag_clf_fit",  subdir = "classification")
save_model(rf_clf_fit,   "rf_clf_fit",   subdir = "classification")
save_model(xgb_clf_fit,  "xgb_clf_fit",  subdir = "classification")
saveRDS(bart_clf_fit,    here("output", "models", "classification", "bart_clf_fit.rds"))

p_roc <- plot_roc_overlay(roc_objects, title = "ROC Curves — Classification (No SMOTE)")
save_plot(p_roc, "clf_roc_no_smote", width = 8, height = 6)

t_05a <- Sys.time()
cat(sprintf("\n[%s] 05a complete in %.0f seconds\n", format(t_05a), as.numeric(t_05a - t_start)))


# ############################################################################
#                     05b — SAME METHODS WITH SMOTE
# ############################################################################

section_header("CLASSIFICATION: Tree-Based Methods (With SMOTE)")

clf_smote_p <- prep(recipe_clf_smote, training = clf_train)
train_s     <- bake(clf_smote_p, new_data = clf_train)
test_s      <- bake(clf_smote_p, new_data = clf_test)

cat("  Train after SMOTE:", nrow(train_s), "\n")
cat("  Class balance:\n"); print(table(train_s$class))

X_train_s <- train_s %>% select(-class) %>% as.matrix()
y_train_s <- train_s$class
X_test_s  <- test_s  %>% select(-class) %>% as.matrix()
y_test_s  <- test_s$class

y_train_s_num <- as.integer(y_train_s == "pulsar")
y_test_s_num  <- as.integer(y_test_s == "pulsar")

clf_smote_results <- tibble()
roc_objects_smote <- list()

# --- 1. CART with SMOTE ---
log_msg("CART with SMOTE...")
set.seed(42)
cart_s_full <- rpart(class ~ ., data = train_s,
                     control = rpart.control(cp = 0.0001, minsplit = 20))
cp_t <- as.data.frame(cart_s_full$cptable)
min_xe_s <- min(cp_t$xerror)
cp_1se_s <- cp_t %>% filter(xerror <= min_xe_s + cp_t$xstd[which.min(cp_t$xerror)]) %>%
  slice_max(CP) %>% pull(CP) %>% first()
cart_smote <- prune(cart_s_full, cp = cp_1se_s)

cart_s_prob <- predict(cart_smote, newdata = test_s, type = "prob")[, "pulsar"]
cart_s_pred <- predict(cart_smote, newdata = test_s, type = "class")
cart_s_met  <- compute_classification_metrics(y_test_s, cart_s_pred, prob = cart_s_prob, positive_class = "pulsar")
cart_s_met$method <- "CART"
clf_smote_results <- bind_rows(clf_smote_results, cart_s_met)
roc_objects_smote[["CART"]] <- pROC::roc(y_test_s, cart_s_prob, levels = c("noise", "pulsar"),
                                          direction = "<", quiet = TRUE)
cat("  CART+SMOTE done — AUC:", round(cart_s_met$AUC, 4), "\n")

# --- 2. Bagging with SMOTE ---
log_msg("Bagging with SMOTE...")
set.seed(42)
bag_smote <- ranger(class ~ ., data = train_s, mtry = p_clf,
                    num.trees = 500, probability = TRUE,
                    importance = "permutation", seed = 42)
bag_s_prob <- predict(bag_smote, data = test_s)$predictions[, "pulsar"]
bag_s_pred <- factor(ifelse(bag_s_prob >= 0.5, "pulsar", "noise"), levels = c("noise", "pulsar"))
bag_s_met  <- compute_classification_metrics(y_test_s, bag_s_pred, prob = bag_s_prob, positive_class = "pulsar")
bag_s_met$method <- "Bagging"
clf_smote_results <- bind_rows(clf_smote_results, bag_s_met)
roc_objects_smote[["Bagging"]] <- pROC::roc(y_test_s, bag_s_prob, levels = c("noise", "pulsar"),
                                              direction = "<", quiet = TRUE)
cat("  Bagging+SMOTE done — AUC:", round(bag_s_met$AUC, 4), "\n")

# --- 3. RF with SMOTE ---
log_msg("Random Forest with SMOTE...")
set.seed(42)
rf_smote <- ranger(class ~ ., data = train_s, mtry = best_mtry_clf,
                   num.trees = 500, probability = TRUE,
                   importance = "permutation", seed = 42)
rf_s_prob <- predict(rf_smote, data = test_s)$predictions[, "pulsar"]
rf_s_pred <- factor(ifelse(rf_s_prob >= 0.5, "pulsar", "noise"), levels = c("noise", "pulsar"))
rf_s_met  <- compute_classification_metrics(y_test_s, rf_s_pred, prob = rf_s_prob, positive_class = "pulsar")
rf_s_met$method <- "Random Forest"
clf_smote_results <- bind_rows(clf_smote_results, rf_s_met)
roc_objects_smote[["Random Forest"]] <- pROC::roc(y_test_s, rf_s_prob, levels = c("noise", "pulsar"),
                                                     direction = "<", quiet = TRUE)
cat("  RF+SMOTE done — AUC:", round(rf_s_met$AUC, 4), "\n")

# --- 4. XGBoost with SMOTE ---
log_msg("XGBoost with SMOTE...")
dtrain_s <- xgb.DMatrix(data = X_train_s, label = y_train_s_num)
dtest_s  <- xgb.DMatrix(data = X_test_s,  label = y_test_s_num)

set.seed(42)
xgb_smote_cv <- xgb.cv(
  params = list(objective = "binary:logistic", eval_metric = "auc",
                eta = 0.05, max_depth = 6, subsample = 0.8, colsample_bytree = 0.8),
  data = dtrain_s, nrounds = 1500, nfold = 5,
  early_stopping_rounds = 30, verbose = 0
)
xgb_smote_best <- xgb_smote_cv$early_stop$best_iteration
if (is.null(xgb_smote_best)) xgb_smote_best <- xgb_smote_cv$niter

set.seed(42)
xgb_smote <- xgb.train(
  params = list(objective = "binary:logistic",
                eta = 0.05, max_depth = 6, subsample = 0.8, colsample_bytree = 0.8),
  data = dtrain_s, nrounds = xgb_smote_best, verbose = 0
)
xgb_s_prob <- predict(xgb_smote, newdata = dtest_s)
xgb_s_pred <- factor(ifelse(xgb_s_prob >= 0.5, "pulsar", "noise"), levels = c("noise", "pulsar"))
xgb_s_met  <- compute_classification_metrics(y_test_s, xgb_s_pred, prob = xgb_s_prob, positive_class = "pulsar")
xgb_s_met$method <- "XGBoost"
clf_smote_results <- bind_rows(clf_smote_results, xgb_s_met)
roc_objects_smote[["XGBoost"]] <- pROC::roc(y_test_s, xgb_s_prob, levels = c("noise", "pulsar"),
                                              direction = "<", quiet = TRUE)
cat("  XGBoost+SMOTE done — AUC:", round(xgb_s_met$AUC, 4), "\n")

# --- 5. BART with SMOTE ---
log_msg("BART with SMOTE...")
set.seed(42)
bart_smote <- pbart(
  x.train = X_train_s, y.train = y_train_s_num,
  x.test = X_test_s,
  ntree = 50, ndpost = 1000, nskip = 250, keepevery = 5
)
bart_s_prob <- colMeans(pnorm(bart_smote$yhat.test))
bart_s_pred <- factor(ifelse(bart_s_prob >= 0.5, "pulsar", "noise"), levels = c("noise", "pulsar"))
bart_s_met  <- compute_classification_metrics(y_test_s, bart_s_pred, prob = bart_s_prob, positive_class = "pulsar")
bart_s_met$method <- "BART"
clf_smote_results <- bind_rows(clf_smote_results, bart_s_met)
roc_objects_smote[["BART"]] <- pROC::roc(y_test_s, bart_s_prob, levels = c("noise", "pulsar"),
                                           direction = "<", quiet = TRUE)
cat("  BART+SMOTE done — AUC:", round(bart_s_met$AUC, 4), "\n")

# --- Save SMOTE results ---
section_header("CLASSIFICATION (WITH SMOTE) — SUMMARY")
clf_smote_results %>%
  select(method, Accuracy, Balanced_Accuracy, AUC, Precision, Recall, F1) %>%
  print()

saveRDS(clf_smote_results, here("output", "tables", "clf_smote_results.rds"))
saveRDS(roc_objects_smote, here("output", "tables", "clf_smote_roc.rds"))

p_roc_smote <- plot_roc_overlay(roc_objects_smote, title = "ROC Curves — Classification (With SMOTE)")
save_plot(p_roc_smote, "clf_roc_smote", width = 8, height = 6)

t_05b <- Sys.time()
cat(sprintf("\n[%s] 05b complete — elapsed %.0f seconds\n", format(t_05b), as.numeric(t_05b - t_start)))


# ############################################################################
#                     06 — COMPARISON (SMOTE IMPACT + DELONG)
# ############################################################################

section_header("CLASSIFICATION: Final Comparison")

no_smote_results <- clf_results %>% mutate(smote = "No SMOTE")
smote_results_df <- clf_smote_results %>% mutate(smote = "With SMOTE")
all_clf <- bind_rows(no_smote_results, smote_results_df)

# SMOTE impact
smote_impact <- no_smote_results %>%
  select(method, AUC_no = AUC, Recall_no = Recall,
         Specificity_no = Specificity, BalAcc_no = Balanced_Accuracy) %>%
  inner_join(
    smote_results_df %>%
      select(method, AUC_smote = AUC, Recall_smote = Recall,
             Specificity_smote = Specificity, BalAcc_smote = Balanced_Accuracy),
    by = "method"
  ) %>%
  mutate(
    AUC_change = AUC_smote - AUC_no,
    Recall_change = Recall_smote - Recall_no,
    Specificity_change = Specificity_smote - Specificity_no,
    BalAcc_change = BalAcc_smote - BalAcc_no
  )

cat("\n--- SMOTE Impact ---\n")
print(smote_impact)

smote_long <- smote_impact %>%
  select(method, AUC_change, Recall_change, Specificity_change) %>%
  pivot_longer(-method, names_to = "metric", values_to = "change") %>%
  mutate(metric = str_remove(metric, "_change"))

p_smote_impact <- ggplot(smote_long, aes(x = method, y = change, fill = metric)) +
  geom_col(position = "dodge", alpha = 0.8) +
  geom_hline(yintercept = 0, linetype = "dashed") +
  scale_fill_brewer(palette = "Set2") +
  labs(title = "SMOTE Impact on Classification Metrics",
       subtitle = "Positive = improvement, Negative = degradation",
       x = NULL, y = "Change (SMOTE - No SMOTE)", fill = "Metric") +
  theme(axis.text.x = element_text(angle = 30, hjust = 1))
save_plot(p_smote_impact, "clf_smote_impact", width = 10, height = 6)

# ROC combined
p_roc_no <- plot_roc_overlay(roc_objects, title = "Without SMOTE")
p_roc_sm <- plot_roc_overlay(roc_objects_smote, title = "With SMOTE")
p_roc_combined <- p_roc_no + p_roc_sm +
  plot_annotation(title = "ROC Curves Comparison: SMOTE Impact")
save_plot(p_roc_combined, "clf_roc_combined", width = 16, height = 7)

# DeLong tests
log_msg("DeLong's test for pairwise AUC comparison...")
methods_vec <- names(roc_objects)
delong_results <- tibble()
for (i in 1:(length(methods_vec) - 1)) {
  for (j in (i + 1):length(methods_vec)) {
    test <- pROC::roc.test(roc_objects[[methods_vec[i]]],
                            roc_objects[[methods_vec[j]]], method = "delong")
    delong_results <- bind_rows(delong_results, tibble(
      method_1 = methods_vec[i], method_2 = methods_vec[j],
      AUC_1 = as.numeric(pROC::auc(roc_objects[[methods_vec[i]]])),
      AUC_2 = as.numeric(pROC::auc(roc_objects[[methods_vec[j]]])),
      z_stat = test$statistic, p_value = test$p.value,
      significant = test$p.value < 0.05
    ))
  }
}
cat("\n--- DeLong's Test ---\n"); print(delong_results)

# AUC barplot
p_auc <- all_clf %>%
  ggplot(aes(x = reorder(method, AUC), y = AUC, fill = smote)) +
  geom_col(position = "dodge", alpha = 0.85, width = 0.7) +
  geom_text(aes(label = sprintf("%.3f", AUC)),
            position = position_dodge(width = 0.7), hjust = -0.1, size = 3) +
  coord_flip() +
  scale_fill_manual(values = c("No SMOTE" = "#636363", "With SMOTE" = "#e6550d")) +
  labs(title = "Classification: AUC Comparison",
       subtitle = "HTRU2 Pulsar (91/9 imbalance)", x = NULL, y = "AUC", fill = NULL) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.15)))
save_plot(p_auc, "clf_auc_comparison", width = 10, height = 6)

section_header("CLASSIFICATION — FINAL METRICS")
all_clf %>%
  select(method, smote, Accuracy, Balanced_Accuracy, AUC, Precision, Recall, Specificity, F1) %>%
  arrange(smote, desc(AUC)) %>%
  print_comparison_table(caption = "All Classification Results")

save_table(
  all_clf %>% select(method, smote, Accuracy, Balanced_Accuracy, AUC,
                      Precision, Recall, Specificity, F1) %>% arrange(smote, desc(AUC)),
  "clf_all_comparison"
)
saveRDS(delong_results, here("output", "tables", "clf_delong_tests.rds"))
saveRDS(smote_impact,   here("output", "tables", "clf_smote_impact.rds"))


# ############################################################################
#                     07 — VARIABLE IMPORTANCE
# ############################################################################

section_header("CLASSIFICATION: Variable Importance Comparison")

feature_names <- c("ip_mean", "ip_sd", "ip_kurtosis", "ip_skewness",
                    "dm_mean", "dm_sd", "dm_kurtosis", "dm_skewness")
feature_labels <- c(
  ip_mean = "IP Mean", ip_sd = "IP Std Dev",
  ip_kurtosis = "IP Kurtosis", ip_skewness = "IP Skewness",
  dm_mean = "DM-SNR Mean", dm_sd = "DM-SNR Std Dev",
  dm_kurtosis = "DM-SNR Kurtosis", dm_skewness = "DM-SNR Skewness"
)

# CART VI
cart_vi_raw <- cart_clf$variable.importance
cart_vi <- tibble(variable = names(cart_vi_raw),
                  importance = as.numeric(cart_vi_raw) / max(cart_vi_raw) * 100,
                  method = "CART")

# Bagging VI
bag_vi <- tibble(variable = names(bag_clf_fit$variable.importance),
                 importance = bag_clf_fit$variable.importance / max(bag_clf_fit$variable.importance) * 100,
                 method = "Bagging")

# RF VI
rf_vi <- tibble(variable = names(rf_clf_fit$variable.importance),
                importance = rf_clf_fit$variable.importance / max(rf_clf_fit$variable.importance) * 100,
                method = "Random Forest")

# XGBoost VI
xgb_imp <- xgb.importance(model = xgb_clf_fit)
xgb_vi <- tibble(variable = xgb_imp$Feature,
                 importance = xgb_imp$Gain / max(xgb_imp$Gain) * 100,
                 method = "XGBoost")

# BART VI
bart_vc <- colMeans(bart_clf_fit$varcount)
names(bart_vc) <- feature_names
bart_vi <- tibble(variable = names(bart_vc),
                  importance = bart_vc / max(bart_vc) * 100,
                  method = "BART")

all_vi <- bind_rows(cart_vi, bag_vi, rf_vi, xgb_vi, bart_vi) %>%
  mutate(label = feature_labels[variable])

# Heatmap
p_heatmap <- all_vi %>%
  ggplot(aes(x = method, y = reorder(label, importance), fill = importance)) +
  geom_tile(color = "white", linewidth = 0.5) +
  geom_text(aes(label = round(importance, 0)), color = "white", size = 3.5) +
  scale_fill_viridis_c(option = "magma", direction = -1, name = "Importance (%)") +
  labs(title = "Variable Importance: Pulsar Features Across Methods",
       subtitle = "IP = Integrated Profile | DM-SNR = Dispersion Measure Signal-to-Noise",
       x = "Method", y = NULL) +
  theme(axis.text.x = element_text(angle = 30, hjust = 1))
save_plot(p_heatmap, "clf_vi_heatmap", width = 9, height = 6)

# Barplot
p_bars <- all_vi %>%
  ggplot(aes(x = reorder(label, importance), y = importance, fill = method)) +
  geom_col(position = "dodge", alpha = 0.8) +
  coord_flip() +
  scale_fill_brewer(palette = "Set2") +
  labs(title = "Variable Importance by Method",
       subtitle = "HTRU2 Pulsar — 8 Radio Signal Features",
       x = NULL, y = "Relative Importance (%)", fill = "Method")
save_plot(p_bars, "clf_vi_bars", width = 10, height = 6)

# Summary
top_by_method <- all_vi %>%
  group_by(method) %>% slice_max(importance, n = 1) %>% ungroup()
cat("\n--- Top feature by method ---\n")
print(top_by_method %>% select(method, variable, label, importance))

avg_vi <- all_vi %>%
  group_by(variable, label) %>%
  summarise(avg_importance = mean(importance), .groups = "drop") %>%
  arrange(desc(avg_importance))
cat("\n--- Average importance across methods ---\n")
print(avg_vi)

saveRDS(all_vi, here("output", "tables", "clf_variable_importance.rds"))
save_table(avg_vi, "clf_avg_variable_importance")


# ############################################################################
#                     TIMING SUMMARY
# ############################################################################

t_end <- Sys.time()
cat(sprintf("\n\n========================================\n"))
cat(sprintf("CLASSIFICATION PIPELINE COMPLETE\n"))
cat(sprintf("Total time: %.1f minutes\n", as.numeric(t_end - t_start, units = "mins")))
cat(sprintf("========================================\n"))
