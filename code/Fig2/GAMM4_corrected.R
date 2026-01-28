library(mgcv)
#library(dplyr)
library(tidyr)
library(readr)
library(purrr)
library(stringr)
library(tidyverse)
library(gamm4)
library(readxl)
library(ggplot2)

# ------------------------- 离群值筛选函数 -------------------------
remove_outliers <- function(df, features = c("SA","maxD","meanD", "SW","SA_tala","maxD_tala","meanD_tala","CT","SL","SL_tala")) {
  # 使用MAD法检测离群值（更适合非正态分布数据）
  df_clean <- df %>%
    group_by(Label) %>%
    mutate_at(vars(all_of(features)), ~ {
      med <- median(., na.rm = TRUE)
      mad <- mad(., na.rm = TRUE)
      lower <- med - 3 * mad  # 保留99.7%的正常数据
      upper <- med + 3 * mad
      ifelse(. < lower | . > upper, NA, .)
    }) %>%
    ungroup() %>%
    filter(complete.cases(across(all_of(features))))
  return(df_clean)
}

# ------------------------- 数据读取和预处理 -------------------------
# 0. 设置路径
output_dir <- "E:/lsy_group/7.9reorganize/7.9reorganize/6.paper/CB_revise/submit/3.gamm/"
if (!dir.exists(output_dir)) dir.create(output_dir)

# 1. 读取所有形态特征CSV文件
morph_path <- "E:/lsy_group/7.9reorganize/7.9reorganize/6.paper/CB_revise/submit/2.data_process/sulcal_data_merge_delete/"
morph_files <- list.files(morph_path, pattern = "\\.csv$", full.names = TRUE)

morph_data <- map_dfr(morph_files, ~ {
  id <- str_sub(basename(.x), 1, 9)
  read_csv(.x) %>% 
    mutate(id = id) %>%
    dplyr::select(Label, SA, maxD, meanD, SW, SA_tala, maxD_tala, meanD_tala, CT, SL, SL_tala,id)  # 确保列顺序一致
})

# 2. 读取人口学数据
demo_data <- readxl::read_excel("E:/lsy_group/7.9reorganize/7.9reorganize/6.paper/CB_revise/submit/1.data_demographic/merged_final_result.xlsx") %>% 
  select(id, Age, gender, subj_unique, TIV) %>%
  mutate(gender = as.factor(gender))

# 3. 合并并清洗数据
merged_data <- morph_data %>%
  left_join(demo_data, by = "id") %>%
  filter(!is.na(TIV)) %>%
  remove_outliers()  # 关键离群值筛选步骤

# ------------------------- TIV回归残差计算 -------------------------
residual_data <- merged_data %>%
  group_by(Label) %>%
  nest() %>%
  mutate(
    data_resid = map(data, ~ {
      # 使用稳健回归抵抗残留离群值
      robust_lm <- function(formula, data) {
        MASS::rlm(formula, data = data, method = "MM")
      }
      
      .x %>%
        mutate(
          SA_resid = residuals(robust_lm(SA ~ TIV, data = .x)),
          maxD_resid = residuals(robust_lm(maxD ~ TIV, data = .x)),
          meanD_resid = residuals(robust_lm(meanD ~ TIV, data = .x)),
          SW = SW,
          CT = CT,
          SL_resid = residuals(robust_lm(SL ~ TIV, data = .x))
          
        )
    })
  ) %>%
  select(-data) %>%
  unnest(data_resid)

# ------------------------- GAMM建模与模型选择 -------------------------
features <- c("SA_resid", "maxD_resid", "meanD_resid", "SW","CT","SL_resid")
results <- list()

for (feature in features) {
  for (label in unique(residual_data$Label)) {
    data_subset <- residual_data %>%
      filter(Label == label) %>%
      mutate(
        subj_unique = as.factor(subj_unique),
        gender = factor(gender)
      ) %>%
      drop_na(all_of(feature), Age, gender)  # 确保数据完整性
    
    if (nrow(data_subset) < 10) next  # 跳过样本量不足的脑区
    
    tryCatch({
      # 完整模型（含年龄项）
      full_model <- gamm(
        as.formula(paste(feature, "~ gender + s(Age, k=3, bs='tp')")),
        random = list(subj_unique = ~1),
        data = data_subset,
        method = "REML",
        control = lmeControl(opt = "optim", msMaxIter = 100)
      )
      
      # 空模型（不含年龄项）
      null_model <- gamm(
        as.formula(paste(feature, "~ gender")),
        random = list(subj_unique = ~1),
        data = data_subset,
        method = "REML"
      )
      
      # 检查模型收敛性
      if (any(is.null(full_model$lme$apVar), is.null(null_model$lme$apVar))) next
      
      # 计算调整R方差
      adj_r2_full <- summary(full_model$gam)$r.sq
      adj_r2_null <- summary(null_model$gam)$r.sq
      delta_adj_r2 <- adj_r2_full - adj_r2_null
      
      # 通过线性模型获取年龄效应方向
      linear_model <- lm(
        as.formula(paste(feature, "~ Age + gender")),
        data = data_subset
      )
      age_effect_sign <- sign(summary(linear_model)$coefficients["Age", "Estimate"])
      
      # 计算带符号的Delta_adj R²
      signed_delta_adj_r2 <- delta_adj_r2 * age_effect_sign
      
      # 模型比较的p值
      anova_pvalue <- anova(full_model$lme, null_model$lme)$"p-value"[2]
      
      # 存储结果
      results[[paste(label, feature, sep = "_")]] <- list(
        label = label,
        feature = feature,
        full_model = full_model,
        null_model = null_model,
        linear_model = linear_model,
        BIC = BIC(full_model$lme),
        Delta_adj_R2 = signed_delta_adj_r2,
        adj_r2_full = adj_r2_full,
        adj_r2_null = adj_r2_null,
        anova_pvalue = anova_pvalue,
        coefficients = list(
          fixed_effects = fixef(full_model$lme),
          smooth_terms = summary(full_model$gam)$s.table
        ),
        p_values = data.frame(
          term = c("gender", "s(Age)"), 
          p_value = c(summary(full_model$gam)$p.table[2,4],
                      summary(full_model$gam)$s.table[1,4])
        )
      )
    }, error = function(e) {
      message(paste("Error in", label, feature, ":", e$message))
      NULL
    })
  }
}

# ==============================================================================
# 接在您的 for 循环之后运行
# ==============================================================================

cat("建模循环结束，开始提取结果并进行 FDR 校正...\n")

# ------------------------- 1. 稳健的结果提取 -------------------------
# 将 results 列表转换为一个整齐的 Data Frame
all_models_df <- map_dfr(results, function(res) {
  
  # A. 获取模型摘要
  gam_sum <- summary(res$full_model$gam)
  
  # B. 提取固定效应 (如 Gender)
  # 使用 base R 方法提取行名，避免 dplyr 版本兼容性问题
  fixed_tab <- as.data.frame(gam_sum$p.table)
  fixed_tab$term <- rownames(fixed_tab)
  fixed_tab$type <- "fixed"
  # 统一列名
  colnames(fixed_tab)[colnames(fixed_tab) == "Pr(>|t|)"] <- "p_value"
  colnames(fixed_tab)[colnames(fixed_tab) == "t value"] <- "statistic"
  colnames(fixed_tab)[colnames(fixed_tab) == "Std. Error"] <- "std_error"
  colnames(fixed_tab)[colnames(fixed_tab) == "Estimate"] <- "estimate"
  
  # C. 提取平滑效应 (如 s(Age))
  smooth_tab <- as.data.frame(gam_sum$s.table)
  smooth_tab$term <- rownames(smooth_tab)
  smooth_tab$type <- "smooth"
  # 统一列名 (平滑项没有单一 estimate，statistic 是 F值)
  colnames(smooth_tab)[colnames(smooth_tab) == "p-value"] <- "p_value"
  colnames(smooth_tab)[colnames(smooth_tab) == "F"] <- "statistic"
  smooth_tab$std_error <- NA
  smooth_tab$estimate <- NA 
  
  # D. 合并当前模型的所有项
  cols_keep <- c("term", "type", "estimate", "std_error", "statistic", "p_value")
  
  current_model_df <- bind_rows(
    fixed_tab[, cols_keep],
    smooth_tab[, cols_keep]
  ) %>%
    filter(term != "(Intercept)") %>% # 排除截距
    mutate(
      label = res$label,
      feature = res$feature,
      anova_p = res$anova_pvalue,
      Delta_adj_R2 = res$Delta_adj_R2,
      adj_r2_full = res$adj_r2_full,
      BIC = res$BIC
    )
  
  return(current_model_df)
})

# ------------------------- 2. 分组 FDR 校正 (解决数量不一致的核心) -------------------------

# 步骤 A: 对模型项 (Terms) 进行校正
# 关键修改：按 'term' 分组。这意味着所有的 's(Age)' 在一起校正，所有的 'gender1' 在一起校正。
# 这样性别微弱的 P 值不会被年龄极强的 P 值“掩盖”。
stats_corrected <- all_models_df %>%
  group_by(term) %>% 
  mutate(adjusted_p = p.adjust(p_value, method = "fdr")) %>%
  ungroup()

# 步骤 B: 对模型整体比较 (ANOVA) 进行校正
# 每个 (label + feature) 只有一个 ANOVA P值，需要去重后校正
anova_correction <- stats_corrected %>%
  dplyr::select(label, feature, anova_p) %>%
  distinct() %>%
  mutate(anova_fdr_p = p.adjust(anova_p, method = "fdr"))

# 步骤 C: 将 ANOVA FDR 结果合并回主表
final_results <- stats_corrected %>%
  left_join(anova_correction, by = c("label", "feature", "anova_p"))

# ------------------------- 3. 结果保存 -------------------------

# 保存 1: 完整结果 (包含所有P值，供Python脚本使用)
# 这里的 adjusted_p 是项的 FDR，anova_fdr_p 是模型的 FDR
write_csv(final_results, paste0(output_dir, "all_models_details.csv"))
cat("完整结果已保存至: all_models_details.csv\n")

# 保存 2: 显著性别效应 (用于快速查看)
# 筛选逻辑：是固定效应 AND 名字里包含'gender' AND FDR < 0.05
significant_gender <- final_results %>%
  filter(type == "fixed") %>%
  filter(str_detect(term, "gender")) %>% # 自动匹配 gender1 或 genderM
  filter(adjusted_p < 0.05) %>%
  arrange(adjusted_p)

write_csv(significant_gender, paste0(output_dir, "significant_gender_effects.csv"))

cat("\n=== 分析完成 ===\n")
cat("显著性别效应数量:", nrow(significant_gender), "\n")
if(nrow(significant_gender) > 0) {
  print(head(significant_gender %>% select(label, feature, p_value, adjusted_p)))
}

# ==============================================================================
# 保存 3: 显著年龄发育效应 (Developmental Effects)
# ==============================================================================

# 筛选逻辑：严格遵循 Methods 描述的双重标准
# 1. 项类型是平滑项 (type == "smooth")
# 2. 模型比较通过 FDR 校正 (anova_fdr_p < 0.05)
# 3. 平滑项本身通过 FDR 校正 (adjusted_p < 0.05)

significant_age <- final_results %>%
  filter(type == "smooth" | str_detect(term, "s\\(Age\\)")) %>% # 确保选中 Age 项
  filter(anova_fdr_p < 0.05 & adjusted_p < 0.05) %>%
  # 按照效应量 (Delta_adj_R2) 的绝对值降序排列，这样变化最剧烈的脑区排在前面
  arrange(desc(abs(Delta_adj_R2)))

# 保存为 CSV
write_csv(significant_age, paste0(output_dir, "significant_developmental_effects.csv"))

# 打印统计信息
cat("\n------------------------------------------------\n")
cat("显著年龄效应已保存至: significant_developmental_effects.csv\n")
cat("显著年龄效应数量:", nrow(significant_age), "\n")
cat("------------------------------------------------\n")


# ------------------------- 3. 绘图部分 -------------------------
plot_dir <- paste0(output_dir, "model_plots/")
if (!dir.exists(plot_dir)) dir.create(plot_dir)

# 筛选显著模型进行趋势图绘制
sig_to_plot <- final_results %>%
  filter(term == "s(Age)", adjusted_p < 0.05, anova_fdr_p < 0.05)

gender_colors <- c("0" = "#D62728", "1" = "#1F77B4")
gender_labels <- c("0" = "Girls", "1" = "Boys")

for (i in seq_len(nrow(sig_to_plot))) {
  row <- sig_to_plot[i,]
  model_id <- paste(row$label, row$feature, sep = "_")
  res <- results[[model_id]]
  
  plot_data <- residual_data %>% filter(Label == row$label) %>% mutate(gender = factor(gender))
  age_range <- range(plot_data$Age, na.rm = TRUE)
  newdata <- expand.grid(Age = seq(age_range[1], age_range[2], length.out = 100),
                         gender = factor(levels(plot_data$gender)), subj_unique = NA)
  
  pred <- predict(res$full_model$gam, newdata = newdata, se.fit = TRUE, exclude = "s(subj_unique)")
  plot_df <- newdata %>% mutate(fit = pred$fit, lower = fit - 1.96*pred$se.fit, upper = fit + 1.96*pred$se.fit)
  
  p <- ggplot() +
    geom_point(data = plot_data, aes(x = Age, y = .data[[row$feature]], color = gender), alpha = 0.3) +
    geom_ribbon(data = plot_df, aes(x = Age, ymin = lower, ymax = upper, fill = gender), alpha = 0.2) +
    geom_line(data = plot_df, aes(x = Age, y = fit, color = gender), linewidth = 1) +
    scale_color_manual(values = gender_colors, labels = gender_labels, name = "Gender") +
    scale_fill_manual(values = gender_colors, labels = gender_labels, name = "Gender") +
    labs(title = paste("Sulcus:", row$label), 
         subtitle = paste("Feature:", row$feature, "\nFDR p (Age):", round(row$adjusted_p, 4)),
         x = "Age (years)", y = "Adjusted Residuals") +
    theme_bw()
  
  # 同时保存 PDF 和 PNG
  ggsave(paste0(plot_dir, model_id, ".pdf"), p, width = 7, height = 6)
  ggsave(paste0(plot_dir, model_id, ".png"), p, width = 7, height = 6, dpi = 300)
}
# ------------------------- 性别效应矩阵热图 -------------------------
# ------------------------- 4. 性别效应矩阵热图 -------------------------
# 查找 term 列中包含 "gender" 的行
sig_gender <- final_results %>% 
  filter(str_detect(term, "gender"), adjusted_p < 0.05)

if(nrow(sig_gender) > 0) {
  p_heat <- ggplot(sig_gender, aes(x = feature, y = reorder(label, estimate), fill = estimate)) +
    geom_tile(color = "white") +
    scale_fill_gradient2(
      low = "#D62728",  # 负值：Girls > Boys 
      mid = "white", 
      high = "#1F77B4", # 正值：Boys > Girls
      midpoint = 0,
      name = "Estimate\n(Gender Effect)"
    ) +
    labs(
      title = "Significant Gender Effects (FDR < 0.05)",
      subtitle = "Blue: Boys > Girls; Red: Girls > Boys",
      x = "Morphological Feature",
      y = "Brain Region"
    ) +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
  
  # 保存热图为 PDF
  ggsave(paste0(output_dir, "gender_effects_heatmap.pdf"), p_heat, width = 10, height = 12)
  ggsave(paste0(output_dir, "gender_effects_heatmap.png"), p_heat, width = 10, height = 12, dpi = 300)
  cat("热图已保存。\n")
} else {
  cat("未发现显著的性别效应，跳过热图绘制。\n")
}
}