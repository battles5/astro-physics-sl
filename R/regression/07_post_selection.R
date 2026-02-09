# ============================================================================
# 07_post_selection.R — Post-selection inference after Lasso
# ============================================================================
# Classical p-values are invalid after variable selection (selection bias).
# We apply two principled approaches:
#   1. Knockoff filter — FDR-controlled variable selection (Candès et al.)
#   2. Stability selection — selection frequency across bootstrap resamples
# ============================================================================

if (!exists("reg_train")) source(here::here("R", "04_cv_setup.R"))

section_header("POST-SELECTION INFERENCE")

# ============================================================================
# 0. Prepare data
# ============================================================================

reg_prep <- prep(recipe_reg_linear, training = reg_train)
train_baked <- bake(reg_prep, new_data = reg_train)

X <- train_baked %>% select(-critical_temp) %>% as.matrix()
y <- train_baked$critical_temp
p <- ncol(X)

cat(sprintf("Post-selection inference on %d observations x %d features\n\n",
            nrow(X), p))


# ============================================================================
# 1. KNOCKOFF FILTER (Model-X)
# ============================================================================
# The Knockoff filter controls FDR (False Discovery Rate) — it guarantees
# that the proportion of false discoveries among selected variables is
# bounded by the target FDR level q.
# ============================================================================

log_msg("Running Knockoff filter (FDR control)...")

# Knockoff filter requires n > 2p. Check feasibility.
if (nrow(X) > 2 * ncol(X)) {
  set.seed(42)
  knockoff_result <- knockoff.filter(X, y,
                                      fdr = 0.10,      # Target FDR = 10%
                                      statistic = stat.glmnet_coefdiff)

  knockoff_selected <- knockoff_result$selected
  cat(sprintf("\n--- Knockoff Filter (FDR = 0.10) ---\n"))
  cat(sprintf("  # Selected variables: %d / %d\n",
              length(knockoff_selected), p))

  if (length(knockoff_selected) > 0) {
    knockoff_var_names <- colnames(X)[knockoff_selected]
    cat(sprintf("  Selected: %s\n", paste(knockoff_var_names, collapse = ", ")))
  } else {
    knockoff_var_names <- character(0)
    cat("  No variables selected at FDR = 0.10\n")
  }

  # Also try FDR = 0.20 (more liberal)
  set.seed(42)
  knockoff_20 <- knockoff.filter(X, y, fdr = 0.20,
                                  statistic = stat.glmnet_coefdiff)
  knockoff_20_names <- colnames(X)[knockoff_20$selected]
  cat(sprintf("\n  At FDR = 0.20: %d variables selected\n",
              length(knockoff_20$selected)))
  if (length(knockoff_20_names) > 0) {
    cat(sprintf("  Selected: %s\n", paste(knockoff_20_names, collapse = ", ")))
  }

  # Knockoff statistics plot
  kn_stats <- tibble(
    variable  = colnames(X),
    statistic = knockoff_result$statistic
  ) %>%
    arrange(desc(abs(statistic))) %>%
    head(30)

  p_knockoff <- ggplot(kn_stats, aes(x = reorder(variable, statistic),
                                      y = statistic)) +
    geom_col(aes(fill = statistic > 0), alpha = 0.8) +
    geom_hline(yintercept = knockoff_result$threshold, linetype = "dashed",
               color = "red") +
    annotate("text", x = 1, y = knockoff_result$threshold,
             label = sprintf("Threshold = %.2f", knockoff_result$threshold),
             vjust = -0.5, color = "red", size = 3) +
    coord_flip() +
    scale_fill_manual(values = c("TRUE" = "#33a02c", "FALSE" = "#e31a1c"),
                      labels = c("Negative", "Positive")) +
    labs(title = "Knockoff Filter: Variable Statistics (Top 30)",
         subtitle = "Variables above red threshold pass FDR = 0.10 control",
         x = NULL, y = "Knockoff Statistic", fill = "Sign") +
    theme(legend.position = "bottom")

  save_plot(p_knockoff, "reg_knockoff", width = 10, height = 8)

} else {
  log_msg(sprintf("Knockoff requires n > 2p. Have n=%d, p=%d (need n > %d).",
                  nrow(X), p, 2*p), level = "WARN")
  log_msg("Subsampling features for Knockoff demonstration...")

  # Use Lasso-selected features for Knockoff (reduced p)
  lasso_cv <- load_model("lasso_cv", subdir = "regression")
  lasso_coefs <- coef(lasso_cv, s = "lambda.min")
  lasso_sel <- which(lasso_coefs[-1] != 0)

  if (length(lasso_sel) > 0 && nrow(X) > 2 * length(lasso_sel)) {
    X_reduced <- X[, lasso_sel]
    set.seed(42)
    knockoff_result <- knockoff.filter(X_reduced, y, fdr = 0.10,
                                        statistic = stat.glmnet_coefdiff)
    knockoff_var_names <- colnames(X_reduced)[knockoff_result$selected]
    cat(sprintf("\n--- Knockoff on Lasso-selected features (%d) ---\n",
                ncol(X_reduced)))
    cat(sprintf("  # Surviving Knockoff: %d\n", length(knockoff_var_names)))
    if (length(knockoff_var_names) > 0) {
      cat(sprintf("  Selected: %s\n", paste(knockoff_var_names, collapse = ", ")))
    }
  } else {
    knockoff_var_names <- character(0)
    log_msg("Knockoff not feasible even on reduced set.", level = "WARN")
  }
}


# ============================================================================
# 2. STABILITY SELECTION
# ============================================================================
# Repeatedly subsample the data and run Lasso. Variables that are selected
# in a high proportion of subsamples (> cutoff) are deemed "stable".
# This avoids the instability of single Lasso runs.
# ============================================================================

log_msg("Running Stability Selection (100 bootstrap resamples)...")

set.seed(42)
stab_result <- stabsel(x = X, y = y,
                        fitfun = glmnet.lasso,
                        cutoff = 0.75,          # Selection probability threshold
                        PFER = 1,               # Expected # false positives ≤ 1
                        B = 100)                # Number of subsamples

cat("\n--- Stability Selection Results ---\n")
cat(sprintf("  Cutoff: %.2f | PFER bound: %.1f\n",
            stab_result$cutoff, stab_result$PFER))

stab_selected <- names(stab_result$selected)
cat(sprintf("  # Stable variables: %d / %d\n", length(stab_selected), p))
if (length(stab_selected) > 0) {
  cat(sprintf("  Selected: %s\n", paste(stab_selected, collapse = ", ")))
}

# Selection probability plot (top 30)
stab_probs <- tibble(
  variable = names(stab_result$max),
  sel_prob = stab_result$max
) %>%
  arrange(desc(sel_prob)) %>%
  head(30)

p_stab <- ggplot(stab_probs, aes(x = reorder(variable, sel_prob), y = sel_prob)) +
  geom_col(aes(fill = sel_prob >= stab_result$cutoff), alpha = 0.8) +
  geom_hline(yintercept = stab_result$cutoff, linetype = "dashed", color = "red") +
  annotate("text", x = 1, y = stab_result$cutoff,
           label = sprintf("Cutoff = %.2f", stab_result$cutoff),
           vjust = -0.5, color = "red", size = 3) +
  coord_flip() +
  scale_fill_manual(values = c("TRUE" = "#33a02c", "FALSE" = "#636363"),
                    labels = c("Below cutoff", "Above cutoff")) +
  labs(title = "Stability Selection: Selection Probabilities (Top 30)",
       subtitle = "Variables above red threshold are stably selected",
       x = NULL, y = "Selection Probability", fill = NULL) +
  theme(legend.position = "bottom")

save_plot(p_stab, "reg_stability_selection", width = 10, height = 8)


# ============================================================================
# 3. Compare: Lasso vs. Knockoff vs. Stability Selection
# ============================================================================

log_msg("Comparing variable selection across methods...")

# Load Lasso selected variables
lasso_selected <- readRDS(here("output", "tables", "reg_selected_variables.rds"))$Lasso

# Build comparison
all_methods_selected <- list(
  Lasso               = lasso_selected,
  Knockoff_FDR10      = knockoff_var_names,
  Stability_Selection = stab_selected
)

# Create a presence/absence matrix
all_vars <- sort(unique(unlist(all_methods_selected)))
selection_matrix <- map_dfc(all_methods_selected, function(sel) {
  as.integer(all_vars %in% sel)
}) %>%
  mutate(variable = all_vars, .before = 1)

cat("\n--- Variable Selection Comparison ---\n")
cat(sprintf("  Lasso:               %d variables\n", length(lasso_selected)))
cat(sprintf("  Knockoff (FDR=0.10): %d variables\n", length(knockoff_var_names)))
cat(sprintf("  Stability Selection: %d variables\n", length(stab_selected)))

# Variables selected by ALL methods
consensus <- all_vars[
  selection_matrix$Lasso == 1 &
  selection_matrix$Knockoff_FDR10 == 1 &
  selection_matrix$Stability_Selection == 1
]
cat(sprintf("\n  Consensus (all 3 methods): %d variables\n", length(consensus)))
if (length(consensus) > 0) {
  cat(sprintf("  %s\n", paste(consensus, collapse = ", ")))
}


# ============================================================================
# 4. Save results
# ============================================================================

post_selection_results <- list(
  knockoff = list(
    selected   = knockoff_var_names,
    fdr        = 0.10
  ),
  stability = list(
    selected   = stab_selected,
    cutoff     = stab_result$cutoff,
    PFER       = stab_result$PFER,
    sel_probs  = stab_result$max
  ),
  comparison = selection_matrix,
  consensus  = consensus
)

saveRDS(post_selection_results, here("output", "tables", "reg_post_selection.rds"))
save_table(selection_matrix, "reg_variable_selection_comparison")

log_msg("Post-selection inference complete.")
