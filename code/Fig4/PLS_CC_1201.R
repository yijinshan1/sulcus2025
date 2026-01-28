
rm(list = ls())
library(readr)
library(dplyr)
library(mixOmics)
library(ggplot2)
library(gridExtra)
library(RColorBrewer)
library(readxl)

# ======== 路径设置 (保持不变) ========
data_dir <- "E:/lsy_group/7.9reorganize/7.9reorganize/6.paper/CB_revise/submit/4.cognition_analysis/change_analysis/change_data/left/ATT_C/"
meta_file <- file.path("E:/lsy_group/7.9reorganize/7.9reorganize/6.paper/CB_revise/submit/4.cognition_analysis/attention1.csv")
#meta_file <- file.path("E:/lsy_group/7.9reorganize/7.9reorganize/6.paper/CB_revise/submit/4.cognition_analysis/working_memory.xlsx")
out_dir <- file.path(data_dir, "PLS_batch_results_unified")

if (!dir.exists(out_dir)) dir.create(out_dir)

# ======== 读取受试者信息 (保持不变) ========
meta <- read_csv(meta_file) %>% dplyr::select(subj_unique, gender)
#meta <- read_excel(meta_file) %>% dplyr::select(subj_unique, gender)

# ======== 找到所有形态学变化csv文件 ========
csv_files <- list.files(data_dir, pattern="\\.csv$", full.names=TRUE)

# ==============================================================================
# 核心统计函数 (与代码1完全一致)
# ==============================================================================

# 自定义置换检验函数
pls_permutation_test <- function(X, Y, n_perm = 1000) {
  original_pls <- pls(X, Y, ncomp = 1)
  original_cor <- cor(original_pls$variates$X[,1], original_pls$variates$Y[,1])
  
  perm_cors <- numeric(n_perm)
  for (i in 1:n_perm) {
    Y_perm <- Y[sample(nrow(Y)), , drop = FALSE]
    tryCatch({
      perm_pls <- pls(X, Y_perm, ncomp = 1)
      perm_cors[i] <- cor(perm_pls$variates$X[,1], perm_pls$variates$Y[,1])
    }, error = function(e) { perm_cors[i] <- 0 })
  }
  
  p_value <- mean(abs(perm_cors) >= abs(original_cor), na.rm=TRUE)
  return(list(p_value = p_value, original_cor = original_cor))
}

# Loading显著性检验函数 (Bootstrap)
loading_significance_test <- function(X, Y, n_bootstrap = 1000) {
  n_features <- ncol(X)
  n_samples <- nrow(X)
  
  bootstrap_loadings <- matrix(NA, nrow = n_bootstrap, ncol = n_features)
  
  for (i in 1:n_bootstrap) {
    boot_indices <- sample(n_samples, replace = TRUE)
    X_boot <- X[boot_indices, , drop = FALSE]
    Y_boot <- Y[boot_indices, , drop = FALSE]
    
    tryCatch({
      if(sd(Y_boot) > 1e-10) {
        pls_boot <- pls(X_boot, Y_boot, ncomp = 1)
        bootstrap_loadings[i, ] <- pls_boot$loadings$X[, 1]
      }
    }, error = function(e) {})
  }
  
  bootstrap_loadings <- na.omit(bootstrap_loadings)
  n_valid <- nrow(bootstrap_loadings)
  
  p_values <- numeric(n_features)
  for (j in 1:n_features) {
    vals <- bootstrap_loadings[, j]
    mean_val <- mean(vals)
    sd_val <- sd(vals)
    if(sd_val > 0) {
      t_stat <- abs(mean_val) / sd_val
      p_values[j] <- 2 * (1 - pt(t_stat, df = n_valid - 1))
    } else {
      p_values[j] <- 1
    }
  }
  
  return(list(p_values = p_values))
}

# ==============================================================================
# 循环分析
# ==============================================================================
summary_table <- data.frame()
significant_results <- data.frame()

for (csv_file in csv_files) {
  message("分析文件: ", basename(csv_file))
  brain <- read_csv(csv_file)
  merged <- left_join(brain, meta, by="subj_unique")
  
  # 筛选数据列
  req_cols <- grep("merge", names(merged), value=TRUE)
  if(length(req_cols) == 0) next
  
  if (!"cognitive_change" %in% colnames(merged)) next
  if (!"subj_unique" %in% colnames(merged)) next
  
  Y <- merged$cognitive_change
  X <- merged %>% dplyr::select(all_of(req_cols))
  
  # 完整性检查
  complete_idx <- complete.cases(X, Y)
  X <- X[complete_idx, ]
  Y <- as.matrix(Y[complete_idx])
  if(nrow(X) < 10) next
  
  # 标准化
  Xs <- scale(X)
  Ys <- scale(Y)
  
  # 运行PLS
  pls_fit <- pls(Xs, Ys, ncomp = 1)
  
  # 1. 整体显著性 (置换检验)
  perm_result <- pls_permutation_test(Xs, Ys, n_perm = 1000)
  overall_p <- perm_result$p_value
  
  # 2. Loading显著性 (Bootstrap)
  loading_sig <- loading_significance_test(Xs, Ys, n_bootstrap = 1000)
  
  # 3. FDR校正
  fdr_corrected_p <- p.adjust(loading_sig$p_values, method = "fdr")
  
  # 4. 结果整理
  loading_df <- data.frame(
    BrainRegion = colnames(X),
    Loading = pls_fit$loadings$X[, 1],
    P_raw = loading_sig$p_values,
    P_FDR = fdr_corrected_p,
    Significant = fdr_corrected_p < 0.05
  )
  
  # 5. 记录显著的Case
  if (overall_p < 0.05 && any(loading_df$Significant)) {
    sig_regions <- loading_df$BrainRegion[loading_df$Significant]
    significant_results <- rbind(significant_results, data.frame(
      File = basename(csv_file),
      OverallP = overall_p,
      NumSigRegions = length(sig_regions)
    ))
  }
  
  # 6. 保存Loading表
  write.csv(loading_df, file = file.path(out_dir, paste0(basename(csv_file), "_PLS_loadings.csv")), row.names = FALSE)
  
  # 7. 绘图 (统一风格: 红色显著，灰色不显著)
  loading_df$ColorGroup <- ifelse(loading_df$Significant, "Significant", "Not Significant")
  
  g_loading <- ggplot(loading_df, aes(x = reorder(BrainRegion, Loading), y = Loading, fill = ColorGroup)) + 
    geom_bar(stat = "identity") + 
    scale_fill_manual(values = c("Significant" = "#D73027", "Not Significant" = "#CCCCCC")) +
    coord_flip() + 
    labs(title = paste0("PLS Loading: ", basename(csv_file)),
         subtitle = paste0("Overall p = ", round(overall_p, 4), 
                           ", Sig Regions: ", sum(loading_df$Significant)),
         x = "Brain Region", y = "Loading", fill = "FDR < 0.05") +
    theme_classic() +
    theme(legend.position = "bottom")
  
  # 8. 得分图
  scores_df <- data.frame(
    X_score = pls_fit$variates$X[, 1],
    Y_score = pls_fit$variates$Y[, 1]
  )
  
  g_scores <- ggplot(scores_df, aes(x = X_score, y = Y_score)) +
    geom_point(alpha = 0.7, size = 2) +
    geom_smooth(method = "lm", se = TRUE, color = "red") +
    labs(title = "PLS Scores",
         subtitle = paste0("Correlation: ", round(perm_result$original_cor, 3)),
         x = "X Scores (PC1)", y = "Y Scores (PC1)") +
    theme_classic()
  
  # 9. 保存组合图
  combined_plot <- grid.arrange(g_loading, g_scores, ncol = 2, widths = c(1.5, 1))
  ggsave(file.path(out_dir, paste0(basename(csv_file), "_PLS_Results.svg")), 
         combined_plot, width = 14, height = 8)
  
  # 10. 更新总表
  summary_table <- rbind(summary_table, data.frame(
    File = basename(csv_file),
    n = nrow(X),
    Correlation = perm_result$original_cor,
    Overall_P = overall_p,
    Significant_Overall = overall_p < 0.05,
    Num_Sig_Loadings = sum(loading_df$Significant)
  ))
}

# ======== 保存所有结果 ========
write.csv(summary_table, file = file.path(out_dir, "PLS_summary.csv"), row.names = FALSE)
if(nrow(significant_results) > 0) {
  write.csv(significant_results, file = file.path(out_dir, "PLS_significant_subset.csv"), row.names = FALSE)
}

cat("\n分析完成！结果保存至:", out_dir, "\n")
