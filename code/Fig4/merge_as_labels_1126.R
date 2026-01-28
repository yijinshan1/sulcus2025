
library(dplyr)
library(readr)
library(stringr)
library(purrr)
library(tidyr)
library(tidyverse)
library(readxl)
library(MASS)  # 用于稳健回归

# ------------------------- 数据读取和预处理 -------------------------
# 设置工作目录
setwd("E:/lsy_group/7.9reorganize/7.9reorganize/2.data_process/sulcal_data_merge_delete/")

# 1. 读取所有形态特征CSV文件
file_pattern <- "CBDP\\d{4}[A-Z]_FS_default_session_auto_sulcal_morphometry_split_processed\\.csv"
csv_files <- list.files(pattern = file_pattern)

# 读取并合并所有文件
all_data <- lapply(csv_files, function(file) {
  subject_id <- str_extract(file, "CBDP0(\\d{3}[A-Z])", group = 1)
  read_csv(file) %>% 
    mutate(subject = paste0(subject_id, "_FS"),
           # 修正ID格式：在CBDP后添加双0，确保格式为CBDP00xxX
           id = paste0("CBDP0", subject_id))
}) %>% bind_rows()

# 检查数据
if (nrow(all_data) == 0) stop("没有读取到任何数据，请检查文件名格式和路径")

# 2. 读取人口学数据（包含TIV）
demo_data <- readxl::read_excel("E:/lsy_group/7.9reorganize/7.9reorganize/1.data_demographic/merged_final_result.xlsx") %>% 
  dplyr::select(id, Age, gender, subj_unique, TIV) %>%
  mutate(gender = as.factor(gender))

# 3. 合并形态学数据和人口学数据（保留所有原始数据）
merged_data <- all_data %>%
  left_join(demo_data, by = "id") #%>%
  #dplyr::filter(!is.na(TIV))  # 只过滤掉没有TIV数据的样本


# ------------------------- TIV回归残差计算 -------------------------
# 计算TIV调整后的值（修正残差长度问题）
data_with_tiv_adjusted <- merged_data %>%
  group_by(Label) %>%
  nest() %>%
  mutate(
    data_processed = map(data, ~ {
      # 安全的残差计算函数
      safe_residuals <- function(formula, data) {
        # 创建一个与原数据行数相同的NA向量
        result <- rep(NA, nrow(data))
        
        # 找到完整的观测值（无缺失值的行）
        complete_cases <- complete.cases(data[, all.vars(formula)])
        
        if(sum(complete_cases) > 2) {  # 至少需要3个完整观测值
          tryCatch({
            # 使用稳健回归
            model <- MASS::rlm(formula, data = data[complete_cases, ], method = "MM")
            result[complete_cases] <- residuals(model)
          }, error = function(e) {
            tryCatch({
              # 如果稳健回归失败，使用普通线性回归
              model <- lm(formula, data = data[complete_cases, ])
              result[complete_cases] <- residuals(model)
            }, error = function(e2) {
              # 如果都失败了，保持NA
            })
          })
        }
        return(result)
      }
      
      .x %>%
        mutate(
          # 原始形态学特征保持不变
          SA = SA,
          maxD = maxD,
          meanD = meanD,
          SW = SW,
          SA_tala = SA_tala,
          maxD_tala = maxD_tala,
          meanD_tala = meanD_tala,
          CT = CT,
          SL = SL,
          SL_tala = SL_tala,
          # 添加TIV值
          TIV = TIV,
          # 添加TIV调整后的值（残差）
          SA_TIV_adjusted = safe_residuals(SA ~ TIV, .x),
          maxD_TIV_adjusted = safe_residuals(maxD ~ TIV, .x),
          meanD_TIV_adjusted = safe_residuals(meanD ~ TIV, .x),
          SW_TIV_adjusted = safe_residuals(SW ~ TIV, .x),
          SA_tala_TIV_adjusted = safe_residuals(SA_tala ~ TIV, .x),
          maxD_tala_TIV_adjusted = safe_residuals(maxD_tala ~ TIV, .x),
          meanD_tala_TIV_adjusted = safe_residuals(meanD_tala ~ TIV, .x),
          CT_TIV_adjusted = safe_residuals(CT ~ TIV, .x),
          SL_TIV_adjusted = safe_residuals(SL ~ TIV, .x),
          SL_tala_TIV_adjusted = safe_residuals(SL_tala ~ TIV, .x)
        )
    })
  ) %>%
  dplyr::select(-data) %>%
  unnest(data_processed)

# ------------------------- 按Label创建文件 -------------------------
# 创建输出目录
output_dir <- "E:/lsy_group/7.9reorganize/7.9reorganize/4.LASSO_revise/mergeaslabel/"
if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)

# 获取所有唯一的Label
unique_labels <- unique(data_with_tiv_adjusted$Label)

# 为每个Label创建合并后的文件
walk(unique_labels, ~{
  label_data <- data_with_tiv_adjusted %>%
    dplyr::filter(Label == .x) %>%
    # 重新排列列顺序，让相关信息更有组织
    dplyr::select(
      # 基本信息
      subject, id, Label, Age, gender, subj_unique, TIV,
      # 原始形态学特征
      SA, maxD, meanD, SW, SA_tala, maxD_tala, meanD_tala, CT, SL, SL_tala,
      # TIV调整后的形态学特征
      SA_TIV_adjusted, maxD_TIV_adjusted, meanD_TIV_adjusted, SW_TIV_adjusted,
      SA_tala_TIV_adjusted, maxD_tala_TIV_adjusted, meanD_tala_TIV_adjusted,
      CT_TIV_adjusted, SL_TIV_adjusted, SL_tala_TIV_adjusted,
      # 其他所有剩余列
      everything()
    )
  
  safe_label <- str_replace_all(.x, "[^[:alnum:]_]", "_")
  filename <- paste0(output_dir, safe_label, "_merged_with_TIV.csv")
  
  write_csv(label_data, filename)
  message("Created merged file with TIV adjustments: ", filename)
  message("  - Samples: ", nrow(label_data))
  message("  - Features: Original + TIV + TIV-adjusted morphological measures")
})

# ------------------------- 生成汇总报告 -------------------------
message("\n=== 处理完成汇总 ===")
message("共创建了 ", length(unique_labels), " 个按Label合并后的文件")
message("每个文件包含:")
message("  - 原始形态学特征: SA, maxD, meanD, SW, SA_tala, maxD_tala, meanD_tala, CT, SL, SL_tala")
message("  - TIV值: TIV")
message("  - TIV调整后特征: [特征名]_TIV_adjusted")
message("  - 人口学信息: Age, gender, subj_unique")
message("文件保存位置: ", output_dir)
message("注意: 保留了所有原始数据，未进行离群值筛选")

# 可选：创建一个汇总统计文件
summary_stats <- data_with_tiv_adjusted %>%
  group_by(Label) %>%
  summarise(
    n_subjects = n(),
    mean_TIV = mean(TIV, na.rm = TRUE),
    sd_TIV = sd(TIV, na.rm = TRUE),
    mean_age = mean(Age, na.rm = TRUE),
    sd_age = sd(Age, na.rm = TRUE),
    n_missing_values = sum(is.na(SA) | is.na(maxD) | is.na(meanD) | is.na(SW) | 
                             is.na(SA_tala) | is.na(maxD_tala) | is.na(meanD_tala) | 
                             is.na(CT) | is.na(SL) | is.na(SL_tala)),
    .groups = 'drop'
  )

write_csv(summary_stats, paste0(output_dir, "summary_by_label.csv"))
message("已创建汇总统计文件: ", paste0(output_dir, "summary_by_label.csv"))
