# ============================================================================
# plotting.R — Standardized plotting functions
# ============================================================================

#' Plot coefficient paths for penalized regression (Ridge/Lasso/Elastic Net)
#'
#' @param fit A cv.glmnet or glmnet fit object
#' @param title Plot title
#' @param lambda_min Optimal lambda (vertical line)
#' @param lambda_1se 1-SE lambda (vertical line)
#' @return A base R plot (not ggplot — glmnet uses base)
plot_coefficient_path <- function(fit, title = "Coefficient Path",
                                  lambda_min = NULL, lambda_1se = NULL) {
  plot(fit, xvar = "lambda", label = TRUE)
  title(main = title, line = 2.5)

  if (!is.null(lambda_min)) {
    abline(v = log(lambda_min), col = "blue", lty = 2, lwd = 1.5)
  }
  if (!is.null(lambda_1se)) {
    abline(v = log(lambda_1se), col = "red", lty = 2, lwd = 1.5)
  }
}

#' Create an overlaid ROC curve plot for multiple classifiers
#'
#' @param roc_list A named list of pROC::roc objects
#' @param title Plot title
#' @return A ggplot object
plot_roc_overlay <- function(roc_list, title = "ROC Curves Comparison") {
  # Convert roc objects to data frames for ggplot
  roc_df <- map2_dfr(roc_list, names(roc_list), function(roc_obj, name) {
    tibble(
      method      = name,
      sensitivity = roc_obj$sensitivities,
      specificity = roc_obj$specificities,
      auc         = as.numeric(pROC::auc(roc_obj))
    )
  })

  # Create legend labels with AUC
  auc_labels <- roc_df %>%
    distinct(method, auc) %>%
    mutate(label = sprintf("%s (AUC = %.3f)", method, auc))

  roc_df <- roc_df %>%
    left_join(auc_labels, by = c("method", "auc"))

  ggplot(roc_df, aes(x = 1 - specificity, y = sensitivity,
                      color = label)) +
    geom_line(linewidth = 0.8) +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed",
                color = "grey50") +
    scale_color_brewer(palette = "Set1") +
    labs(
      title = title,
      x = "1 - Specificity (False Positive Rate)",
      y = "Sensitivity (True Positive Rate)",
      color = "Method"
    ) +
    coord_equal() +
    theme(legend.position = "right")
}

#' Create a variable importance heatmap across methods
#'
#' @param vi_df A data frame with columns: variable, method, importance
#'   (importance should be normalized 0-100 within each method)
#' @param top_n Number of top variables to show
#' @param title Plot title
#' @return A ggplot object
plot_vi_heatmap <- function(vi_df, top_n = 20,
                             title = "Variable Importance Across Methods") {
  # Select top variables by average importance
  top_vars <- vi_df %>%
    group_by(variable) %>%
    summarise(avg_imp = mean(importance, na.rm = TRUE), .groups = "drop") %>%
    slice_max(avg_imp, n = top_n) %>%
    pull(variable)

  vi_df %>%
    filter(variable %in% top_vars) %>%
    ggplot(aes(x = method, y = reorder(variable, importance),
               fill = importance)) +
    geom_tile(color = "white", linewidth = 0.3) +
    scale_fill_viridis_c(option = "magma", direction = -1,
                          name = "Importance\n(0-100)") +
    labs(title = title, x = "Method", y = NULL) +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
}

#' Plot a CART tree using rpart.plot with consistent styling
#'
#' @param tree An rpart object
#' @param title Plot title
plot_cart_tree <- function(tree, title = "CART") {
  rpart.plot(tree,
             type    = 4,
             extra   = 101,
             under   = TRUE,
             fallen.leaves = TRUE,
             box.palette   = "RdYlGn",
             shadow.col    = "grey80",
             main          = title,
             cex           = 0.7)
}

#' Plot a comparison table as a formatted gt or kable
#'
#' @param results_df Data frame with method as first column, metrics as rest
#' @param caption Table caption
#' @param digits Number of decimal places
#' @return Printed table
print_comparison_table <- function(results_df, caption = "Model Comparison",
                                    digits = 4) {
  results_df %>%
    mutate(across(where(is.numeric), ~round(., digits))) %>%
    knitr::kable(caption = caption, align = "lrrrr") %>%
    print()
}
