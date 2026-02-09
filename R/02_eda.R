# ============================================================================
# 02_eda.R — Exploratory Data Analysis
# ============================================================================
# EDA for both datasets: distributions, missing values, correlations,
# class balance, and key relationships.
# ============================================================================

if (!exists("superconductivity")) source(here::here("R", "01_data_loading.R"))

# ============================================================================
# 1. SUPERCONDUCTIVITY — EDA
# ============================================================================

log_msg("Starting EDA for Superconductivity dataset...")

# --- 1.1 Summary statistics ------------------------------------------------
cat("\n=== SUPERCONDUCTIVITY: Summary ===\n")
skim(superconductivity) %>% print()

# --- 1.2 Missing values ----------------------------------------------------
n_missing <- sum(is.na(superconductivity))
log_msg(sprintf("Superconductivity missing values: %d", n_missing))

if (n_missing > 0) {
  missing_by_col <- superconductivity %>%
    summarise(across(everything(), ~sum(is.na(.)))) %>%
    pivot_longer(everything(), names_to = "variable", values_to = "n_missing") %>%
    filter(n_missing > 0) %>%
    arrange(desc(n_missing))
  print(missing_by_col)
}

# --- 1.3 Target distribution (critical_temp) --------------------------------
p_target_reg <- ggplot(superconductivity, aes(x = critical_temp)) +
  geom_histogram(bins = 60, fill = "#1f78b4", alpha = 0.7, color = "white") +
  geom_vline(aes(xintercept = median(critical_temp)),
             color = "red", linetype = "dashed", linewidth = 0.8) +
  labs(
    title = "Distribution of Critical Temperature (Tc)",
    subtitle = sprintf(
      "Median = %.1f K | Mean = %.1f K | Range = [%.1f, %.1f] K",
      median(superconductivity$critical_temp),
      mean(superconductivity$critical_temp),
      min(superconductivity$critical_temp),
      max(superconductivity$critical_temp)
    ),
    x = "Critical Temperature (K)",
    y = "Count"
  )

# Also check if log-transform is needed
p_target_log <- ggplot(superconductivity, aes(x = log1p(critical_temp))) +
  geom_histogram(bins = 60, fill = "#33a02c", alpha = 0.7, color = "white") +
  labs(
    title = "Distribution of log(1 + Tc)",
    x = "log(1 + Critical Temperature)",
    y = "Count"
  )

p_reg_target <- p_target_reg + p_target_log +
  plot_annotation(title = "Superconductivity — Target Distribution")

save_plot(p_reg_target, "eda_supercond_target", width = 12, height = 5)
print(p_reg_target)

# --- 1.4 Correlation analysis -----------------------------------------------
# With 81 features, full corrplot is dense — show top correlated with target
cor_with_target <- superconductivity %>%
  select(where(is.numeric)) %>%
  cor(use = "pairwise.complete.obs") %>%
  as.data.frame() %>%
  select(critical_temp) %>%
  rownames_to_column("variable") %>%
  filter(variable != "critical_temp") %>%
  arrange(desc(abs(critical_temp)))

cat("\n--- Top 20 features most correlated with critical_temp ---\n")
print(head(cor_with_target, 20))

# Correlation heatmap of top 20 features + target
top20_vars <- c(head(cor_with_target$variable, 20), "critical_temp")
cor_top20 <- superconductivity %>%
  select(all_of(top20_vars)) %>%
  cor(use = "pairwise.complete.obs")

png(here("output", "figures", "eda_supercond_corr_top20.png"),
    width = 900, height = 900, res = 120)
corrplot(cor_top20, method = "color", type = "lower",
         tl.cex = 0.7, tl.col = "black",
         title = "Superconductivity: Correlation (Top 20 features + Tc)",
         mar = c(0, 0, 2, 0))
dev.off()

cairo_pdf(here("output", "figures", "eda_supercond_corr_top20.pdf"),
          width = 8, height = 8)
corrplot(cor_top20, method = "color", type = "lower",
         tl.cex = 0.7, tl.col = "black",
         title = "Superconductivity: Correlation (Top 20 features + Tc)",
         mar = c(0, 0, 2, 0))
dev.off()

# --- 1.5 Feature group structure (for Group Lasso) --------------------------
# Features are statistics (mean, wtd_mean, gmean, entropy, range, std, ...)
# of elemental properties — identify groups
feature_names <- names(superconductivity) %>%
  setdiff("critical_temp")

cat("\n--- Feature name patterns (for Group Lasso grouping) ---\n")
# Extract the statistic type (prefix) from feature names
stat_types <- unique(gsub("_.*", "", feature_names))
cat("Statistic types found:", paste(stat_types, collapse = ", "), "\n")

# Count features per group
feature_groups <- tibble(feature = feature_names) %>%
  mutate(group = gsub("_.*", "", feature)) %>%
  count(group, name = "n_features")
print(feature_groups)


# ============================================================================
# 2. HTRU2 PULSAR — EDA
# ============================================================================

log_msg("Starting EDA for Pulsar dataset...")

# --- 2.1 Summary statistics ------------------------------------------------
cat("\n=== PULSAR: Summary ===\n")
skim(pulsar) %>% print()

# --- 2.2 Missing values ----------------------------------------------------
n_missing_p <- sum(is.na(pulsar))
log_msg(sprintf("Pulsar missing values: %d", n_missing_p))

# --- 2.3 Class distribution ------------------------------------------------
class_dist <- pulsar %>%
  count(class) %>%
  mutate(
    pct = round(n / sum(n) * 100, 1),
    label = sprintf("%s\nn = %s (%.1f%%)", class, scales::comma(n), pct)
  )

p_class <- ggplot(class_dist, aes(x = class, y = n, fill = class)) +
  geom_col(alpha = 0.8, width = 0.6) +
  geom_text(aes(label = label), vjust = -0.3, size = 3.5) +
  scale_fill_manual(values = c("noise" = "#636363", "pulsar" = "#e6550d")) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.15))) +
  labs(
    title = "HTRU2 Pulsar — Class Distribution",
    subtitle = sprintf("Imbalance ratio: %.1f:1",
                        class_dist$n[class_dist$class == "noise"] /
                        class_dist$n[class_dist$class == "pulsar"]),
    x = NULL, y = "Count"
  ) +
  theme(legend.position = "none")

save_plot(p_class, "eda_pulsar_class_dist", width = 6, height = 5)
print(p_class)

# --- 2.4 Feature distributions by class ------------------------------------
pulsar_long <- pulsar %>%
  pivot_longer(-class, names_to = "feature", values_to = "value")

p_features <- ggplot(pulsar_long, aes(x = value, fill = class)) +
  geom_density(alpha = 0.5) +
  facet_wrap(~feature, scales = "free", ncol = 4) +
  scale_fill_manual(values = c("noise" = "#636363", "pulsar" = "#e6550d")) +
  labs(
    title = "HTRU2 Pulsar — Feature Distributions by Class",
    x = "Value", y = "Density", fill = "Class"
  )

save_plot(p_features, "eda_pulsar_features", width = 14, height = 7)
print(p_features)

# --- 2.5 Correlation matrix (all 8 features) --------------------------------
cor_pulsar <- pulsar %>%
  select(-class) %>%
  cor(use = "pairwise.complete.obs")

png(here("output", "figures", "eda_pulsar_corr.png"),
    width = 600, height = 600, res = 120)
corrplot(cor_pulsar, method = "color", type = "lower",
         addCoef.col = "black", number.cex = 0.8,
         tl.cex = 0.8, tl.col = "black",
         title = "HTRU2 Pulsar: Feature Correlation Matrix",
         mar = c(0, 0, 2, 0))
dev.off()

cairo_pdf(here("output", "figures", "eda_pulsar_corr.pdf"),
          width = 6, height = 6)
corrplot(cor_pulsar, method = "color", type = "lower",
         addCoef.col = "black", number.cex = 0.8,
         tl.cex = 0.8, tl.col = "black",
         title = "HTRU2 Pulsar: Feature Correlation Matrix",
         mar = c(0, 0, 2, 0))
dev.off()

# --- 2.6 Boxplots by class (standardized for comparison) -------------------
pulsar_scaled <- pulsar %>%
  mutate(across(-class, scale)) %>%
  pivot_longer(-class, names_to = "feature", values_to = "value")

p_box <- ggplot(pulsar_scaled, aes(x = feature, y = value, fill = class)) +
  geom_boxplot(alpha = 0.6, outlier.size = 0.5) +
  scale_fill_manual(values = c("noise" = "#636363", "pulsar" = "#e6550d")) +
  coord_flip() +
  labs(
    title = "HTRU2 Pulsar — Standardized Feature Boxplots by Class",
    x = NULL, y = "Standardized Value", fill = "Class"
  )

save_plot(p_box, "eda_pulsar_boxplots", width = 10, height = 5)
print(p_box)


# ============================================================================
# 3. Save EDA summaries
# ============================================================================

# Save correlation tables for later reference
write_csv(cor_with_target,
          here("output", "tables", "eda_supercond_correlations.csv"))

eda_summary <- list(
  superconductivity = list(
    n_obs      = nrow(superconductivity),
    n_features = ncol(superconductivity) - 1,
    n_missing  = n_missing,
    target_mean   = mean(superconductivity$critical_temp),
    target_median = median(superconductivity$critical_temp),
    target_sd     = sd(superconductivity$critical_temp)
  ),
  pulsar = list(
    n_obs      = nrow(pulsar),
    n_features = ncol(pulsar) - 1,
    n_missing  = n_missing_p,
    class_counts  = table(pulsar$class),
    imbalance_ratio = sum(pulsar$class == "noise") / sum(pulsar$class == "pulsar")
  )
)

saveRDS(eda_summary, here("output", "tables", "eda_summary.rds"))

log_msg("EDA complete. Figures saved to output/figures/.")
