rm(list = ls())
library(ggplot2)
library(ggpubr)
library(readr)
library(dplyr)
library(psych)
library(ppcor) # For partial correlations
library(readxl)

# 1. 读取gender信息
gender_info <- read_csv("E:/lsy_group/7.9reorganize/7.9reorganize/4.cognition_analysis/attention1.csv") %>%
  dplyr::select(subj_unique, gender)
# gender_info <- read_excel("E:/lsy_group/7.9reorganize/7.9reorganize/4.cognition_analysis/working_memory.xlsx") %>%
#   dplyr::select(subj_unique, gender)
# 2. 读取形态学csv列表
file_list <- list.files(path = "E:/lsy_group/7.9reorganize/7.9reorganize/4.cognition_analysis/change_analysis/change_data/right/ATT_O/", pattern = "*.csv", full.names = TRUE)


summary_results <- data.frame()
# 3. 循环每个文件
for(f in file_list) {
  # 读取
  df <- read_csv(f)
  df <- left_join(df, gender_info, by = "subj_unique")
  
  # 提取相关变量
  brain_cols <- grep("merge", names(df), value = TRUE)
  cname <- gsub(".csv", "", basename(f))
  
  # 结果存储
  cor_results <- data.frame(Region=brain_cols, Cor=NA, Cor_p=NA, Partial_cor=NA, Partial_cor_p=NA)
  
  # 循环每个脑区进行相关/偏相关分析
  for(i in seq_along(brain_cols)) {
    
    # 单纯相关
    cor_test <- cor.test(df[[brain_cols[i]]], df$cognitive_change)
    cor_results$Cor[i] <- cor_test$estimate
    cor_results$Cor_p[i] <- cor_test$p.value
    
    # 偏相关（控制baseline_age, time_interval, gender）
    subdf <- df %>% dplyr::select(cognitive_change, all_of(brain_cols[i]), baseline_age, time_interval, gender)
    subdf <- na.omit(subdf)
    # gender数值化
    if(is.character(subdf$gender) | is.factor(subdf$gender)) subdf$gender <- as.numeric(as.factor(subdf$gender))
    pc <- pcor.test(subdf[[2]], subdf$cognitive_change, subdf[,c("baseline_age", "time_interval", "gender")])
    cor_results$Partial_cor[i] <- pc$estimate
    cor_results$Partial_cor_p[i] <- pc$p.value
  }
  
  
  # FDR校正
  cor_results$Cor_p_fdr <- p.adjust(cor_results$Cor_p, method = "fdr")
  cor_results$Partial_cor_p_fdr <- p.adjust(cor_results$Partial_cor_p, method = "fdr")
  
  # 添加文件名信息
  cor_results$File <- cname
  
  # 修改后：只要相关或偏相关任意一个FDR校正后显著即可
  significant_results <- cor_results %>%
    filter(
      Cor_p_fdr < 0.05 |  # 直接相关FDR校正显著
        Partial_cor_p_fdr < 0.05  # 偏相关FDR校正显著
    ) %>%
    mutate(
      Significant_Type = case_when(
        Cor_p_fdr < 0.05 & Partial_cor_p_fdr < 0.05 ~ "Both_Significant",
        Cor_p_fdr < 0.05 ~ "Direct_Correlation_Significant",
        Partial_cor_p_fdr < 0.05 ~ "Partial_Correlation_Significant",
        TRUE ~ "None"
      ),
      # 添加效应大小标记
      Effect_Size_Direct = case_when(
        abs(Cor) >= 0.5 ~ "Large",
        abs(Cor) >= 0.3 ~ "Medium", 
        abs(Cor) >= 0.1 ~ "Small",
        TRUE ~ "Negligible"
      ),
      Effect_Size_Partial = case_when(
        abs(Partial_cor) >= 0.5 ~ "Large",
        abs(Partial_cor) >= 0.3 ~ "Medium",
        abs(Partial_cor) >= 0.1 ~ "Small", 
        TRUE ~ "Negligible"
      )
    )
  
  # 合并到总汇总表
  summary_results <- bind_rows(summary_results, significant_results)
  # 结果保存
  write.csv(cor_results, paste0("E:/lsy_group/7.9reorganize/7.9reorganize/4.cognition_analysis/change_analysis/change_data/right/results_", cname, ".csv"), row.names = FALSE)
  # 保存每个文件的完整结果（包含FDR校正）
  write.csv(cor_results, paste0("E:/lsy_group/7.9reorganize/7.9reorganize/4.cognition_analysis/change_analysis/change_data/right/results_", cname, "_with_FDR.csv"), row.names = FALSE)
  
  # 4. 可视化（举例：画Partial Correlation前20个脑区）
  plot_df <- cor_results %>% arrange(desc(abs(Partial_cor))) %>% head(20)
  p <- ggplot(plot_df, aes(x=reorder(Region,Partial_cor), y=Partial_cor, fill=Partial_cor_p<0.05)) +
    geom_bar(stat="identity") +
    coord_flip() +
    scale_fill_manual(values=c("grey","red")) +
    labs(title=paste("Partial Cor (Controlling Age, Interval & Gender):", cname),
         x="Brain Region", y="Partial Correlation") +
    theme_classic(base_size=15) +
    theme(axis.text=element_text(size=12), legend.position="none")
  
  ggsave(paste0("E:/lsy_group/7.9reorganize/7.9reorganize/4.cognition_analysis/change_analysis/change_data/right/plot_", cname, ".pdf"), plot=p, width=8, height=6)
}
# 在整个 for(f in file_list) 循环结束后添加：

# 生成最终汇总报告
if(nrow(summary_results) > 0) {
  # 按相关性大小排序
  final_summary <- summary_results %>%
    arrange(desc(pmax(abs(Cor), abs(Partial_cor), na.rm = TRUE))) %>%
    dplyr::select(File, Region, Cor, Cor_p_fdr, Partial_cor, Partial_cor_p_fdr, 
                  Significant_Type, Effect_Size_Direct, Effect_Size_Partial) %>%
    mutate(
      Max_Correlation = pmax(abs(Cor), abs(Partial_cor), na.rm = TRUE),
      Primary_Effect = ifelse(abs(Cor) > abs(Partial_cor), "Direct", "Partial"),
      # 添加综合效应大小评估
      Overall_Effect_Size = case_when(
        Max_Correlation >= 0.5 ~ "Large",
        Max_Correlation >= 0.3 ~ "Medium",
        Max_Correlation >= 0.1 ~ "Small",
        TRUE ~ "Negligible"
      )
    )
  
  # 保存汇总结果
  write.csv(final_summary, "E:/lsy_group/7.9reorganize/7.9reorganize/4.cognition_analysis/change_analysis/change_data/right/FINAL_SUMMARY_All_Significant_Correlations.csv", row.names = FALSE)
  
  # 打印汇总信息
  cat("\n=== 分析完成 ===\n")
  cat("共发现", nrow(final_summary), "个FDR校正后显著的脑区-认知相关关系\n")
  cat("其中:\n")
  cat("- 仅直接相关显著:", sum(final_summary$Significant_Type == "Direct_Correlation_Significant"), "个\n")
  cat("- 仅偏相关显著:", sum(final_summary$Significant_Type == "Partial_Correlation_Significant"), "个\n") 
  cat("- 两种相关都显著:", sum(final_summary$Significant_Type == "Both_Significant"), "个\n")
  
  # 按效应大小分类统计
  cat("\n=== 按效应大小分类 ===\n")
  effect_summary <- table(final_summary$Overall_Effect_Size)
  print(effect_summary)
  
  # 显示top 15结果（扩大显示数量）
  cat("\n=== Top 15 最强相关关系 ===\n")
  print(final_summary %>% head(15) %>% 
          dplyr::select(File, Region, Max_Correlation, Primary_Effect, Significant_Type, Overall_Effect_Size))
  
} else {
  cat("未发现FDR校正后显著的相关关系\n")
}
