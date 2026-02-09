# ============================================================================
# 01_data_loading.R — Download and load datasets
# ============================================================================
# Downloads the two UCI datasets and saves them to data/raw/.
# After first run, loads from local files (skips download if already present).
# ============================================================================

if (!exists("METHOD_COLORS")) source(here::here("R", "00_setup.R"))

# ============================================================================
# 1. SUPERCONDUCTIVITY — Regression dataset
# ============================================================================
# Source: https://archive.ics.uci.edu/dataset/464/superconductivty+data
# 21,263 superconductors, 81 features, target = critical_temp (K)
# ============================================================================

supercond_csv <- here("data", "raw", "superconductivity.csv")

if (!file.exists(supercond_csv)) {
  log_msg("Downloading Superconductivity dataset from UCI...")

  supercond_url <- "https://archive.ics.uci.edu/static/public/464/superconductivty+data.zip"
  supercond_zip <- here("data", "raw", "superconductivity.zip")

  download.file(supercond_url, destfile = supercond_zip, mode = "wb")
  unzip(supercond_zip, exdir = here("data", "raw"))

  # The zip contains "train.csv" — rename for clarity
  raw_file <- here("data", "raw", "train.csv")
  if (file.exists(raw_file)) {
    file.rename(raw_file, supercond_csv)
    log_msg("Superconductivity data saved to data/raw/superconductivity.csv")
  } else {
    # If file structure differs, list extracted files for debugging
    extracted <- list.files(here("data", "raw"), recursive = TRUE)
    log_msg(paste("Extracted files:", paste(extracted, collapse = ", ")), level = "WARN")
    stop("Could not find expected CSV after unzipping. Check extracted files.")
  }

  # Clean up zip
  file.remove(supercond_zip)
} else {
  log_msg("Superconductivity data already present, skipping download.")
}

superconductivity <- read_csv(supercond_csv, show_col_types = FALSE) %>%
  janitor::clean_names()

log_msg(sprintf(
  "Superconductivity loaded: %d obs x %d vars. Target: critical_temp",
  nrow(superconductivity), ncol(superconductivity)
))

# ============================================================================
# 2. HTRU2 PULSAR STARS — Classification dataset
# ============================================================================
# Source: https://archive.ics.uci.edu/dataset/372/htru2
# 17,898 candidates, 8 features, target = class (0 = noise, 1 = pulsar)
# Class distribution: ~91% negative / ~9% positive
# ============================================================================

pulsar_csv <- here("data", "raw", "pulsar.csv")

if (!file.exists(pulsar_csv)) {
  log_msg("Downloading HTRU2 Pulsar dataset from UCI...")

  pulsar_url <- "https://archive.ics.uci.edu/static/public/372/htru2.zip"
  pulsar_zip <- here("data", "raw", "htru2.zip")

  download.file(pulsar_url, destfile = pulsar_zip, mode = "wb")
  unzip(pulsar_zip, exdir = here("data", "raw"))

  # The zip contains "HTRU_2.csv" (no header) or "HTRU_2.arff"
  raw_file <- here("data", "raw", "HTRU_2.csv")
  if (!file.exists(raw_file)) {
    # Try alternative name
    raw_file <- here("data", "raw", "HTRU2.csv")
  }

  if (file.exists(raw_file)) {
    # Add proper column names (dataset has no header)
    pulsar_names <- c(
      "ip_mean", "ip_sd", "ip_kurtosis", "ip_skewness",   # Integrated profile
      "dm_mean", "dm_sd", "dm_kurtosis", "dm_skewness",   # DM-SNR curve
      "class"                                               # 0 = noise, 1 = pulsar
    )

    pulsar_raw <- read_csv(raw_file, col_names = pulsar_names, show_col_types = FALSE)
    write_csv(pulsar_raw, pulsar_csv)
    log_msg("Pulsar data saved to data/raw/pulsar.csv")

    # Clean up original file if different name
    if (raw_file != pulsar_csv) file.remove(raw_file)
  } else {
    extracted <- list.files(here("data", "raw"), recursive = TRUE)
    log_msg(paste("Extracted files:", paste(extracted, collapse = ", ")), level = "WARN")
    stop("Could not find expected CSV after unzipping. Check extracted files.")
  }

  # Clean up zip and any arff file
  file.remove(pulsar_zip)
  arff_file <- here("data", "raw", "HTRU_2.arff")
  if (file.exists(arff_file)) file.remove(arff_file)
} else {
  log_msg("Pulsar data already present, skipping download.")
}

pulsar <- read_csv(pulsar_csv, show_col_types = FALSE) %>%
  mutate(class = factor(class, levels = c(0, 1), labels = c("noise", "pulsar")))

log_msg(sprintf(
  "Pulsar loaded: %d obs x %d vars. Target: class (%s)",
  nrow(pulsar), ncol(pulsar),
  paste(table(pulsar$class), names(table(pulsar$class)), sep = " ", collapse = " / ")
))

# ============================================================================
# 3. Quick sanity checks
# ============================================================================

cat("\n--- Superconductivity: first rows ---\n")
glimpse(superconductivity)

cat("\n--- Pulsar: first rows ---\n")
glimpse(pulsar)

cat("\n--- Pulsar class distribution ---\n")
pulsar %>%
  count(class) %>%
  mutate(pct = round(n / sum(n) * 100, 1)) %>%
  print()

log_msg("Data loading complete.")
