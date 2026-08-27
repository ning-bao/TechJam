# =====================================================================
# Track 5 - Calibration curve + Decision curve (DCA)
#
# Input : mock_predictions.csv  (real data arrives 8/30, same schema -
#         just swap the file)
# Output: figures + CSVs under out/
#
# Needs only base R (+ ggplot2 if available). Run:
#   Rscript calibration_dca.R
#   Rscript calibration_dca.R real_predictions.csv
#
# Skeleton - verified end to end on R 4.6.1. Polish (colours, labels,
# facet layout) is yours.
#
# Acceptance: auroc / ece_15bin / bacc_at_tau / fpr_at_tau must match
# expected_metrics.csv exactly.
#
# NOTE: this file is deliberately pure ASCII with no BOM.
#   - R refuses to parse a file starting with a UTF-8 BOM
#     ("unexpected input" on line 1).
#   - Un-BOMed UTF-8 Chinese renders as mojibake under a GBK Windows
#     locale.
#   Only ASCII is safe in both. Please keep it that way when editing.
# =====================================================================

args <- commandArgs(trailingOnly = TRUE)
infile <- if (length(args) >= 1) args[1] else "mock_predictions.csv"

# Fall back to base R plotting if ggplot2 is missing, instead of crashing
HAS_GG <- requireNamespace("ggplot2", quietly = TRUE)
if (HAS_GG) {
  suppressPackageStartupMessages(library(ggplot2))
} else {
  cat("[warn] ggplot2 not installed; plotting with base R (metrics unaffected).\n")
  cat("       To install: install.packages(\"ggplot2\")\n\n")
}
dir.create("out", showWarnings = FALSE)

# All in-figure text is English. Why: (1) the writeup and the final
# pitch are in English anyway; (2) CJK font names differ across
# Windows / macOS / Linux and detection is unreliable - on failure R
# does not error, it silently renders tofu boxes, and by then the
# figure is already in the document.
if (HAS_GG) theme_set(theme_minimal(base_size = 11))

N_BINS <- 15       # calibration bins; must match expected_metrics.csv
# Primary threshold: frozen ONCE on clean dev at FPR<=5%, identical for every
# condition (never refit per condition). Read from threshold.txt, written by
# json2csv.py alongside the predictions CSV - deliberately NOT hardcoded, and
# deliberately without a fallback default.
TAU <- local({
  cand <- c(if (length(args) >= 2) args[2],
            file.path(dirname(infile), "threshold.txt"),
            "threshold.txt")
  for (p in cand) {
    if (file.exists(p)) {
      v <- suppressWarnings(as.numeric(readLines(p, warn = FALSE)[1]))
      if (is.na(v) || v <= 0 || v >= 1) {
        stop(sprintf("threshold file %s does not hold a probability in (0,1)", p))
      }
      cat(sprintf("Threshold %.6f read from %s\n", v, p))
      return(v)
    }
  }
  stop(paste("No threshold.txt found. It is written by json2csv.py",
             "--threshold-out; pass its path as the 2nd argument, or place it",
             "next to the predictions CSV. Refusing to guess: a wrong operating",
             "point silently invalidates every figure below."))
})

# display order of the 15 conditions
COND_ORDER <- c("clean",
                "jpeg_90", "jpeg_70", "jpeg_50", "jpeg_30",
                "blur_05", "blur_10", "blur_20",
                "resize_050", "resize_025",
                "noise_002", "noise_005", "noise_010",
                "jitter_pm20", "crop_80")

# the five worst-case conditions - the calibration plot must show these
FOCUS <- c("clean", "jpeg_30", "blur_20", "resize_025", "noise_010")

# cost ratio k -> optimal threshold k/(k+1)
K_VALUES <- c(1, 5, 20)
K_TAUS   <- K_VALUES / (K_VALUES + 1)   # 0.500 / 0.833 / 0.952


# --------------------------------------------------------------- load

d <- read.csv(infile, stringsAsFactors = FALSE)
stopifnot(all(c("image_path", "label", "pred", "condition", "generator")
              %in% names(d)))
d$condition <- factor(d$condition, levels = COND_ORDER)

cat(sprintf("Loaded %s: %d rows, %d conditions\n",
            infile, nrow(d), length(unique(d$condition))))

# sample-size check - thin conditions must be marked skip, never
# dropped silently
n_by_cond <- table(d$condition)
thin <- names(n_by_cond)[n_by_cond > 0 & n_by_cond < 100]
if (length(thin) > 0) {
  cat("[warn] conditions with n<100 (mark those conclusions as skip):",
      paste(thin, collapse = ", "), "\n")
}


# --------------------------------------------------------------- metrics

# AUROC: Mann-Whitney U, average ranks for ties (same as Python side)
auroc <- function(label, pred) {
  n_pos <- sum(label == 1); n_neg <- sum(label == 0)
  if (n_pos == 0 || n_neg == 0) return(NA_real_)
  r <- rank(pred)                    # rank() defaults to ties.method="average"
  (sum(r[label == 1]) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
}

# Calibration bins. Convention: (lo, hi], first bin includes its left
# edge - matches the Python reference implementation.
calib_bins <- function(label, pred, n_bins = N_BINS) {
  edges <- seq(0, 1, length.out = n_bins + 1)
  idx <- findInterval(pred, edges, rightmost.closed = TRUE, left.open = TRUE)
  idx[pred <= edges[2]] <- 1L
  do.call(rbind, lapply(seq_len(n_bins), function(b) {
    m <- idx == b
    if (!any(m)) return(NULL)
    data.frame(bin = b, n = sum(m),
               conf = mean(pred[m]), acc = mean(label[m]))
  }))
}

# ECE = sum_b (n_b / N) * |acc_b - conf_b|
ece <- function(label, pred, n_bins = N_BINS) {
  b <- calib_bins(label, pred, n_bins)
  if (is.null(b)) return(NA_real_)
  sum(b$n / length(pred) * abs(b$acc - b$conf))
}

# confusion matrix at a threshold (pred >= tau => predicted AIGC)
conf_at <- function(label, pred, tau) {
  yhat <- pred >= tau
  c(tp = sum(yhat & label == 1), fp = sum(yhat & label == 0),
    tn = sum(!yhat & label == 0), fn = sum(!yhat & label == 1))
}

# Vickers net benefit: NB = TP/n - (FP/n) * pt/(1-pt), using pt itself
# as the operating threshold
net_benefit <- function(label, pred, pt) {
  n <- length(label); cm <- conf_at(label, pred, pt)
  cm[["tp"]] / n - (cm[["fp"]] / n) * (pt / (1 - pt))
}


# --------------------------------------- Table 1: metrics by condition

summary_tbl <- do.call(rbind, lapply(COND_ORDER, function(cd) {
  s <- d[d$condition == cd, ]
  if (nrow(s) == 0) return(NULL)
  cm  <- conf_at(s$label, s$pred, TAU)
  tpr <- cm[["tp"]] / (cm[["tp"]] + cm[["fn"]])
  tnr <- cm[["tn"]] / (cm[["tn"]] + cm[["fp"]])
  data.frame(
    condition      = cd,
    n              = nrow(s),
    auroc          = round(auroc(s$label, s$pred), 4),
    bacc_at_tau    = round((tpr + tnr) / 2, 4),
    fpr_at_tau     = round(1 - tnr, 4),
    ece_15bin      = round(ece(s$label, s$pred), 4),
    # column names identical to expected_metrics.csv so it can be diffed
    nb_at_pt_0.500 = round(net_benefit(s$label, s$pred, 0.500), 4),
    nb_at_pt_0.833 = round(net_benefit(s$label, s$pred, 0.833), 4),
    nb_at_pt_0.952 = round(net_benefit(s$label, s$pred, 0.952), 4),
    stringsAsFactors = FALSE
  )
}))
# keep the display order (without levels, plots would sort alphabetically)
summary_tbl$condition <- factor(summary_tbl$condition, levels = COND_ORDER)

write.csv(summary_tbl, "out/metrics_by_condition.csv", row.names = FALSE)
cat("\n=== Metrics by condition (compare with expected_metrics.csv) ===\n")
print(summary_tbl, row.names = FALSE)

# Delta vs clean - this column IS the official clean-vs-transformed table
clean_row <- summary_tbl[summary_tbl$condition == "clean", ]
summary_tbl$delta_bacc <- round(summary_tbl$bacc_at_tau - clean_row$bacc_at_tau, 4)
cat("\nWorst three conditions (by bAcc):\n")
print(head(summary_tbl[order(summary_tbl$bacc_at_tau),
                       c("condition", "bacc_at_tau", "delta_bacc", "fpr_at_tau")], 3),
      row.names = FALSE)


# ------------------------------------------- Figure 1: calibration curve

calib_df <- do.call(rbind, lapply(FOCUS, function(cd) {
  s <- d[d$condition == cd, ]
  if (nrow(s) == 0) return(NULL)
  b <- calib_bins(s$label, s$pred)
  if (is.null(b)) return(NULL)
  b$condition <- cd
  b
}))
calib_df$condition <- factor(calib_df$condition, levels = FOCUS)

if (HAS_GG) {
  p1 <- ggplot(calib_df, aes(conf, acc, colour = condition)) +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed", colour = "grey50") +
    geom_line(linewidth = 0.7) +
    geom_point(aes(size = n), alpha = 0.75) +
    scale_size_continuous(range = c(1, 4), guide = "none") +
    coord_equal(xlim = c(0, 1), ylim = c(0, 1)) +
    labs(title = "Reliability diagram - below the diagonal = overconfident",
         subtitle = sprintf("%d equal-width bins; point size = bin count", N_BINS),
         x = "Mean predicted p(AIGC)", y = "Observed AIGC fraction",
         colour = "Condition")
  ggsave("out/fig1_calibration.png", p1, width = 7, height = 6, dpi = 150)

  p1b <- ggplot(summary_tbl, aes(reorder(condition, ece_15bin), ece_15bin)) +
    geom_col(fill = "grey35") + coord_flip() +
    labs(title = "ECE (15 bins) - higher = less trustworthy",
         x = NULL, y = "ECE")
  ggsave("out/fig1b_ece.png", p1b, width = 6, height = 5, dpi = 150)

} else {
  # base R fallback
  png("out/fig1_calibration.png", width = 900, height = 800, res = 130)
  plot(NA, xlim = c(0, 1), ylim = c(0, 1), asp = 1,
       xlab = "Mean predicted p(AIGC)", ylab = "Observed AIGC fraction",
       main = "Reliability diagram\nbelow the diagonal = overconfident")
  abline(0, 1, lty = 2, col = "grey50")
  cols <- seq_along(FOCUS) + 1
  for (i in seq_along(FOCUS)) {
    s <- calib_df[calib_df$condition == FOCUS[i], ]
    if (nrow(s) == 0) next
    lines(s$conf, s$acc, col = cols[i], lwd = 2)
    points(s$conf, s$acc, col = cols[i], pch = 19, cex = 0.8)
  }
  legend("topleft", legend = FOCUS, col = cols, lwd = 2, bty = "n", cex = 0.85)
  dev.off()

  png("out/fig1b_ece.png", width = 800, height = 700, res = 130)
  o <- order(summary_tbl$ece_15bin)
  par(mar = c(4, 8, 3, 1))
  barplot(summary_tbl$ece_15bin[o],
          names.arg = as.character(summary_tbl$condition)[o],
          horiz = TRUE, las = 1, col = "grey35",
          main = "ECE (15 bins) - higher = less trustworthy",
          xlab = "ECE", cex.names = 0.8)
  dev.off()
}


# ----------------------------------------- Figure 2: decision curve (DCA)

DCA_CONDS <- c("clean", "jpeg_30")
pt_grid <- seq(0.01, 0.99, by = 0.01)

dca_df <- do.call(rbind, lapply(DCA_CONDS, function(cd) {
  s <- d[d$condition == cd, ]
  if (nrow(s) == 0) return(NULL)
  prev <- mean(s$label)              # share of AI-generated images
  do.call(rbind, lapply(pt_grid, function(pt) {
    data.frame(
      condition  = cd, pt = pt,
      model      = net_benefit(s$label, s$pred, pt),
      # baseline 1: treat everything as AI-generated
      treat_all  = prev - (1 - prev) * (pt / (1 - pt)),
      # baseline 2: treat nothing as AI - net benefit is identically 0
      treat_none = 0
    )
  }))
}))

if (HAS_GG) {
  dca_long <- reshape(dca_df,
                      varying = c("model", "treat_all", "treat_none"),
                      v.names = "nb", timevar = "strategy",
                      times = c("Our model", "Treat all as AI", "Treat none as AI"),
                      direction = "long")

  p2 <- ggplot(dca_long, aes(pt, nb, colour = strategy, linetype = strategy)) +
    geom_vline(xintercept = K_TAUS, colour = "grey70", linetype = "dotted") +
    geom_line(linewidth = 0.7) +
    facet_wrap(~ condition, ncol = 2) +
    coord_cartesian(ylim = c(-0.05, NA)) +
    labs(title = "Decision curve - net benefit vs threshold probability",
         subtitle = sprintf(paste("Dotted verticals = optimal threshold k/(k+1)",
                                  "for cost ratio k=%s;",
                                  "FP = real photo judged AI-generated"),
                            paste(K_VALUES, collapse = "/")),
         x = "Threshold probability pt", y = "Net benefit",
         colour = NULL, linetype = NULL) +
    theme(legend.position = "bottom")
  ggsave("out/fig2_dca.png", p2, width = 9, height = 5, dpi = 150)

} else {
  png("out/fig2_dca.png", width = 1300, height = 650, res = 130)
  par(mfrow = c(1, length(DCA_CONDS)), mar = c(4, 4, 3, 1))
  for (cd in DCA_CONDS) {
    s <- dca_df[dca_df$condition == cd, ]
    if (nrow(s) == 0) next
    yl <- range(c(s$model, s$treat_all, 0), na.rm = TRUE)
    yl[1] <- max(yl[1], -0.05)
    plot(s$pt, s$model, type = "l", lwd = 2, col = "steelblue", ylim = yl,
         xlab = "Threshold probability pt", ylab = "Net benefit", main = cd)
    lines(s$pt, s$treat_all, lwd = 2, col = "darkorange", lty = 2)
    abline(h = 0, col = "grey40", lty = 3)
    abline(v = K_TAUS, col = "grey70", lty = 3)
    text(K_TAUS, yl[1], sprintf("k=%d", K_VALUES), pos = 3, cex = 0.7,
         col = "grey30")
    legend("topright", c("Our model", "Treat all as AI", "Treat none as AI"),
           col = c("steelblue", "darkorange", "grey40"),
           lty = c(1, 2, 3), lwd = 2, bty = "n", cex = 0.8)
  }
  dev.off()
}

# cost ratio -> recommended threshold, with the actual error counts there
cost_tbl <- do.call(rbind, lapply(DCA_CONDS, function(cd) {
  s <- d[d$condition == cd, ]
  if (nrow(s) == 0) return(NULL)
  do.call(rbind, lapply(seq_along(K_VALUES), function(i) {
    tau_k <- K_TAUS[i]; cm <- conf_at(s$label, s$pred, tau_k)
    data.frame(condition = cd, k = K_VALUES[i],
               tau_star = round(tau_k, 4),
               fp = cm[["fp"]], fn = cm[["fn"]],
               nb = round(net_benefit(s$label, s$pred, tau_k), 4))
  }))
}))
write.csv(cost_tbl, "out/cost_ratio_thresholds.csv", row.names = FALSE)
cat("\n=== Cost ratio -> recommended threshold ===\n")
print(cost_tbl, row.names = FALSE)


# ------------------------- Figure 3: threshold drift under degradation

if (HAS_GG) {
  p3 <- ggplot(summary_tbl, aes(condition, fpr_at_tau, group = 1)) +
    geom_hline(yintercept = 0.05, linetype = "dashed", colour = "firebrick") +
    geom_line(colour = "grey40") + geom_point(size = 2) +
    labs(title = sprintf(paste("Actual FPR of one frozen threshold",
                               "(tau=%.4f) across conditions"), TAU),
         subtitle = paste("Red line = the 5% budget set on clean.",
                          "Above it = this operating point has failed"),
         x = NULL, y = "FPR") +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
  ggsave("out/fig3_threshold_drift.png", p3, width = 8, height = 4.5, dpi = 150)

} else {
  png("out/fig3_threshold_drift.png", width = 1100, height = 600, res = 130)
  par(mar = c(8, 4, 3, 1))
  plot(seq_len(nrow(summary_tbl)), summary_tbl$fpr_at_tau, type = "b",
       pch = 19, xaxt = "n", xlab = "", ylab = "FPR",
       main = sprintf("Actual FPR of frozen threshold tau=%.4f", TAU))
  axis(1, at = seq_len(nrow(summary_tbl)),
       labels = as.character(summary_tbl$condition), las = 2, cex.axis = 0.75)
  abline(h = 0.05, lty = 2, col = "firebrick")
  legend("topleft", "5% budget set on clean", lty = 2, col = "firebrick",
         bty = "n", cex = 0.8)
  dev.off()
}


cat("\nDone. Outputs in out/:\n")
cat("  metrics_by_condition.csv     metrics by condition (vs expected_metrics.csv)\n")
cat("  cost_ratio_thresholds.csv    cost ratio -> threshold\n")
cat("  fig1_calibration.png         calibration curve\n")
cat("  fig1b_ece.png                ECE bar chart\n")
cat("  fig2_dca.png                 decision curve\n")
cat("  fig3_threshold_drift.png     FPR drift of the frozen threshold\n")
