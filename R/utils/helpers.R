# ============================================================================
# helpers.R — Reusable utility functions
# ============================================================================

#' Print a timestamped log message
#'
#' @param msg Character message to log
#' @param level Log level (default "INFO")
log_msg <- function(msg, level = "INFO") {
  timestamp <- format(Sys.time(), "%Y-%m-%d %H:%M:%S")
  cat(sprintf("[%s] %s: %s\n", timestamp, level, msg))
}

#' Save a model object to output/models/ as RDS
#'
#' @param model The fitted model object
#' @param name Character string for the filename (without extension)
#' @param subdir Optional subdirectory within output/models/
save_model <- function(model, name, subdir = NULL) {
  if (!is.null(subdir)) {
    dir_path <- here::here("output", "models", subdir)
    if (!dir.exists(dir_path)) dir.create(dir_path, recursive = TRUE)
    path <- file.path(dir_path, paste0(name, ".rds"))
  } else {
    path <- here::here("output", "models", paste0(name, ".rds"))
  }
  saveRDS(model, path)
  log_msg(sprintf("Model saved: %s", path))
}

#' Load a model object from output/models/
#'
#' @param name Character string for the filename (without extension)
#' @param subdir Optional subdirectory within output/models/
load_model <- function(name, subdir = NULL) {
  if (!is.null(subdir)) {
    path <- here::here("output", "models", subdir, paste0(name, ".rds"))
  } else {
    path <- here::here("output", "models", paste0(name, ".rds"))
  }
  if (!file.exists(path)) stop(sprintf("Model file not found: %s", path))
  readRDS(path)
}

#' Save a ggplot to output/figures/ (PNG + PDF for LaTeX)
#'
#' @param plot A ggplot object
#' @param name Filename (without extension)
#' @param width Width in inches (default 10)
#' @param height Height in inches (default 6)
#' @param dpi Resolution for PNG (default 300)
save_plot <- function(plot, name, width = 10, height = 6, dpi = 300) {
  # PNG for quick viewing and README
  path_png <- here::here("output", "figures", paste0(name, ".png"))
  ggsave(path_png, plot, width = width, height = height, dpi = dpi)

  # PDF (vector) for LaTeX presentation
  path_pdf <- here::here("output", "figures", paste0(name, ".pdf"))
  ggsave(path_pdf, plot, width = width, height = height, device = cairo_pdf)

  log_msg(sprintf("Figure saved: %s (.png + .pdf)", name))
}

#' Save a data frame as CSV to output/tables/
#'
#' @param df A data frame
#' @param name Filename (without extension)
save_table <- function(df, name) {
  path <- here::here("output", "tables", paste0(name, ".csv"))
  write_csv(df, path)
  log_msg(sprintf("Table saved: %s", path))
}

#' Compute regression metrics on a test set
#'
#' @param truth Numeric vector of true values
#' @param estimate Numeric vector of predicted values
#' @param n_train Number of training observations (for PMSE)
#' @return A tibble with MSE, RMSE, PMSE, R², MAE
compute_regression_metrics <- function(truth, estimate, n_train = NULL) {
  residuals <- truth - estimate
  mse  <- mean(residuals^2)
  rmse <- sqrt(mse)
  mae  <- mean(abs(residuals))
  ss_res <- sum(residuals^2)
  ss_tot <- sum((truth - mean(truth))^2)
  r_squared <- 1 - ss_res / ss_tot

  result <- tibble(
    MSE  = mse,
    RMSE = rmse,
    MAE  = mae,
    R2   = r_squared
  )

  # PMSE = MSE / Var(Y_train) — normalized prediction error
  if (!is.null(n_train)) {
    result <- result %>%
      mutate(PMSE = MSE / var(truth))
  }

  result
}

#' Compute classification metrics from a confusion matrix
#'
#' @param truth Factor of true classes
#' @param estimate Factor of predicted classes
#' @param prob Numeric vector of predicted probabilities for positive class
#' @param positive_class The label of the positive class
#' @return A tibble with accuracy, balanced accuracy, precision, recall, etc.
compute_classification_metrics <- function(truth, estimate, prob = NULL,
                                            positive_class = "pulsar") {
  # Ensure factor levels
  truth    <- factor(truth)
  estimate <- factor(estimate, levels = levels(truth))

  cm <- table(Predicted = estimate, Actual = truth)

  # Extract TP, TN, FP, FN
  tp <- cm[positive_class, positive_class]
  tn <- sum(diag(cm)) - tp
  fp <- sum(cm[positive_class, ]) - tp
  fn <- sum(cm[, positive_class]) - tp

  accuracy    <- (tp + tn) / sum(cm)
  sensitivity <- tp / (tp + fn)       # Recall

  specificity <- tn / (tn + fp)
  precision   <- tp / (tp + fp)
  f1          <- 2 * precision * sensitivity / (precision + sensitivity)
  balanced_acc <- (sensitivity + specificity) / 2

  result <- tibble(
    Accuracy          = accuracy,
    Balanced_Accuracy = balanced_acc,
    Precision         = precision,
    Recall            = sensitivity,
    Specificity       = specificity,
    F1                = f1
  )

  # AUC if probabilities are provided
  if (!is.null(prob)) {
    roc_obj <- pROC::roc(truth, prob, levels = levels(truth),
                          direction = "<", quiet = TRUE)
    result <- result %>%
      mutate(AUC = as.numeric(pROC::auc(roc_obj)))
  }

  result
}

#' Print a section header for console output
section_header <- function(title) {
  width <- 70
  line <- paste(rep("=", width), collapse = "")
  cat("\n", line, "\n", title, "\n", line, "\n\n")
}
