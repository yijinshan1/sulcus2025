#install.packages(c("readxl", "readr", "dplyr", "purrr", "tidyr", "writexl"))
# 加载必要的包
library(readxl)
library(readr)
library(dplyr)
library(purrr)
library(tidyr)
library(writexl)

# 设置参数
merge_path <- "C:/Users/15542/Desktop/410改文章工作流/7.9reorganize/2.data_process/merge.xlsx"    # 合并规则文件路径
input_dir <- "C:/Users/15542/Desktop/410改文章工作流/7.9reorganize/2.data_process/sulcal_data"
output_dir <- "C:/Users/15542/Desktop/410改文章工作流/7.9reorganize/2.data_process/sulcal_data_merge"

# 创建输出目录（如果不存在）
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}

# 读取合并规则
merge_table <- read_excel(merge_path, sheet = "Sheet1")
new_labels <- merge_table$Label
merge_rules <- merge_table$merge

# 获取待处理文件列表
csv_files <- list.files(path = input_dir, pattern = "\\.csv$", full.names = TRUE)

# 处理每个文件
walk(csv_files, function(input_path) {
  # 读取数据
  orig_data <- read_csv(input_path, na = c("nan", "NaN", "", "NA"),
                        col_types = cols(.default = col_character())) %>%
    mutate(across(-Column_1, as.numeric))
  
  # 初始化结果存储
  result <- tibble(
    Label = new_labels,
    SA = NA_real_,
    maxD = NA_real_,
    meanD = NA_real_,
    SW = NA_real_,
    SA_tala = NA_real_,
    maxD_tala = NA_real_,
    meanD_tala = NA_real_,
    CT = NA_real_,
    SL = NA_real_,
    SL_tala = NA_real_
  )
  
  # 处理每条规则
  for (rule_idx in seq_along(new_labels)) {
    current_rule <- merge_rules[rule_idx]
    
    if (is.na(current_rule)) {  # 直接复制的情况
      label <- new_labels[rule_idx]
      matched_row <- orig_data %>% filter(Column_1 == label)
      
      if (nrow(matched_row) > 0) {
        row <- matched_row[1, ]
        weight <- row$Column_11
        weight_tala <- row$Column_10
        
        result$SA[rule_idx] <- row$Column_5
        result$maxD[rule_idx] <- row$Column_7
        result$SA_tala[rule_idx] <- row$Column_4
        result$maxD_tala[rule_idx] <- row$Column_6
        result$SL_tala[rule_idx] <- row$Column_10
        result$SL[rule_idx] <- row$Column_11
        
        if (!is.na(weight) && weight != 0) {
          result$meanD[rule_idx] <- row$Column_9
          result$CT[rule_idx] <- row$Column_12
          result$SW[rule_idx] <- row$Column_13
        }
        if (!is.na(weight_tala) && weight_tala != 0) {
          result$meanD_tala[rule_idx] <- row$Column_8
        }
      }
    } else {  # 需要合并的情况
      components <- strsplit(current_rule, "\\+")[[1]] %>% trimws()
      matched_rows <- orig_data %>% filter(Column_1 %in% components)
      
      if (nrow(matched_rows) > 0) {
        # 提取有效数据
        valid_rows <- matched_rows %>%
          filter(!is.na(Column_4) | !is.na(Column_5) | !is.na(Column_6)| !is.na(Column_7)| !is.na(Column_8) |
                   !is.na(Column_9) | !is.na(Column_10)|!is.na(Column_11)| !is.na(Column_12)|!is.na(Column_13))
        
        # SA计算（有效值的求和）
        result$SA[rule_idx] <- sum(valid_rows$Column_5, na.rm = TRUE)
        if (is.na(result$SA[rule_idx])) result$SA[rule_idx] <- NA_real_
        
        # SA_tala计算（有效值的求和）
        result$SA_tala[rule_idx] <- sum(valid_rows$Column_4, na.rm = TRUE)
        if (is.na(result$SA_tala[rule_idx])) result$SA_tala[rule_idx] <- NA_real_
        
        # SL计算（有效值的求和）
        result$SL[rule_idx] <- sum(valid_rows$Column_11, na.rm = TRUE)
        if (is.na(result$SL[rule_idx])) result$SL[rule_idx] <- NA_real_
        # SL_tala计算（有效值的求和）
        result$SL_tala[rule_idx] <- sum(valid_rows$Column_10, na.rm = TRUE)
        if (is.na(result$SL_tala[rule_idx])) result$SL_tala[rule_idx] <- NA_real_
        
        # maxD计算（取最大值）
        result$maxD[rule_idx] <- max(valid_rows$Column_7, na.rm = TRUE)
        if (is.infinite(result$maxD[rule_idx])) result$maxD[rule_idx] <- NA_real_
        
        # maxD_tala计算（取最大值）
        result$maxD_tala[rule_idx] <- max(valid_rows$Column_6, na.rm = TRUE)
        if (is.infinite(result$maxD_tala[rule_idx])) result$maxD_tala[rule_idx] <- NA_real_
        
        # meanD计算（加权平均）
        mean_weights <- valid_rows %>%
          select(value = Column_9, weight = Column_11) %>%
          filter(!is.na(value) & !is.na(weight) & weight != 0)
        
        if (nrow(mean_weights) > 0) {
          total_weight <- sum(mean_weights$weight, na.rm = TRUE)
          if (total_weight > 0) {
            result$meanD[rule_idx] <- weighted.mean(mean_weights$value, 
                                                    mean_weights$weight, 
                                                    na.rm = TRUE)
          }
        }
        # meanD_tala计算（加权平均）
        mean_tala_weights <- valid_rows %>%
          select(value = Column_8, weight = Column_10) %>%
          filter(!is.na(value) & !is.na(weight) & weight != 0)
        
        if (nrow(mean_tala_weights) > 0) {
          total_weight <- sum(mean_tala_weights$weight, na.rm = TRUE)
          if (total_weight > 0) {
            result$meanD_tala[rule_idx] <- weighted.mean(mean_tala_weights$value, 
                                                    mean_tala_weights$weight, 
                                                    na.rm = TRUE)
          }
        }
        
        # SW计算（加权平均）
        sw_weights <- valid_rows %>%
          select(value = Column_13, weight = Column_11) %>%
          filter(!is.na(value) & !is.na(weight) & weight != 0)
        
        if (nrow(sw_weights) > 0) {
          total_weight <- sum(sw_weights$weight, na.rm = TRUE)
          if (total_weight > 0) {
            result$SW[rule_idx] <- weighted.mean(sw_weights$value, 
                                                 sw_weights$weight, 
                                                 na.rm = TRUE)
          }
        }
        # CT计算（加权平均）
        ct_weights <- valid_rows %>%
          select(value = Column_12, weight = Column_11) %>%
          filter(!is.na(value) & !is.na(weight) & weight != 0)
        
        if (nrow(sw_weights) > 0) {
          total_weight <- sum(ct_weights$weight, na.rm = TRUE)
          if (total_weight > 0) {
            result$CT[rule_idx] <- weighted.mean(ct_weights$value, 
                                                 ct_weights$weight, 
                                                 na.rm = TRUE)
          }
        }
      }
    }
  }
  
  # 保存结果
  output_filename <- paste0(tools::file_path_sans_ext(basename(input_path)), 
                            "_processed.csv")
  write_csv(result, file.path(output_dir, output_filename))
})
