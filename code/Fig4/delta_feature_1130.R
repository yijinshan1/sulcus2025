
# ==============================================================================
# 脑影像-认知纵向变化分析脚本 (列名匹配修复版)
# ==============================================================================

# 1. 清空环境与加载库
rm(list = ls())
library(dplyr)
library(openxlsx)
library(tools) 

# ================================ 2. 配置区域 ================================

# 路径设置
base_path <- "E:/lsy_group/7.9reorganize/7.9reorganize/4.cognition_analysis"
folder_path <- file.path(base_path, "mergeaslabel_std", "left")         
output_base_path <- file.path(base_path, "change_analysis", "change_data", "left")

if (!dir.exists(output_base_path)) {
  dir.create(output_base_path, recursive = TRUE)
}

# ----------------------- 认知指标配置 -----------------------
cognitive_configs <- list(
  working_memory = list(
    file_path = file.path(base_path, "working_memory.xlsx"),
    remove_cols = c(2, 4, 6, 7, 8, 10, 11),
    rename_subject = NULL,
    indicators = list(
      zscore_rx2_back = list(col = "zscore_rx2_back", out = "WM")
    )
  ),
  attention = list(
    file_path = file.path(base_path, "attention.csv"),
    remove_cols = c(2, 4, 6, 7, 8, 9),
    rename_subject = "subject_FS",
    indicators = list(
      attention_A = list(col = "zscore_Amean",  out = "ATT_A"),
      attention_O = list(col = "zscore_rOmean", out = "ATT_O"),
      attention_C = list(col = "zscore_Cmean",  out = "ATT_C")
    )
  )
)

# ----------------------- 形态学特征配置 (重要修改) -----------------------
# 不再依赖 keep_cols 的数字索引，而是依赖 target_col 的列名
morphological_configs <- list(
  SW       = list(target_col = "SW",                prefix = "SW"),
  CT       = list(target_col = "CT",                prefix = "CT"),
  SATIV    = list(target_col = "SA_TIV_adjusted",   prefix = "SA_TIV_adjusted"),
  maxDTIV  = list(target_col = "maxD_TIV_adjusted", prefix = "maxD_TIV_adjusted"),
  meanDTIV = list(target_col = "meanD_TIV_adjusted",prefix = "meanD_TIV_adjusted"),
  SLTIV    = list(target_col = "SL_TIV_adjusted",   prefix = "SL_TIV_adjusted")
)

# ================================ 3. 核心函数定义 ================================

# 函数 A: 加载认知数据
load_cognitive_data <- function(config) {
  if (!file.exists(config$file_path)) stop(paste("文件不存在:", config$file_path))
  
  if (grepl("\\.xlsx$", config$file_path)) {
    data <- read.xlsx(config$file_path)
  } else {
    data <- read.csv(config$file_path, header = TRUE)
  }
  
  if (!is.null(config$remove_cols)) {
    valid_cols <- config$remove_cols[config$remove_cols <= ncol(data)]
    if (length(valid_cols) > 0) data <- data[, -valid_cols]
  }
  
  if (!is.null(config$rename_subject)) {
    if (config$rename_subject %in% names(data)) {
      data <- data %>% rename(subject = !!sym(config$rename_subject))
    }
  }
  return(data)
}

# 函数 B: 加载形态学数据 (修复核心：按列名匹配)
load_morphological_data <- function(folder_path, morph_config) {
  file_names <- list.files(folder_path, pattern = "\\.csv$|\\.txt$")
  if (length(file_names) == 0) stop("文件夹为空或无CSV文件")
  
  result_data <- NULL
  
  for (file_name in file_names) {
    full_path <- file.path(folder_path, file_name)
    
    tryCatch({
      # 1. 使用 read.csv 读取，确保逗号分隔符处理正确
      # check.names = FALSE 确保列名中的特殊符号不被 R 自动修改 (比如空格变点)
      sulcus_data <- read.csv(full_path, header = TRUE, check.names = FALSE, stringsAsFactors = FALSE)
      
      # 2. 确定 subject 列 (通常是第1列，但也可能是名为 'subject' 的列)
      # 优先找名为 "subject" 的列，找不到就默认第1列
      subj_col_idx <- which(tolower(names(sulcus_data)) == "subject")
      if (length(subj_col_idx) > 0) {
        subject_col <- sulcus_data[[subj_col_idx[1]]]
      } else {
        subject_col <- sulcus_data[[1]]
      }
      
      # 3. 确定目标特征列 (例如 "SW" 或 "CT")
      target_col_name <- morph_config$target_col
      
      if (!target_col_name %in% names(sulcus_data)) {
        # 尝试模糊匹配或报错
        warning(sprintf("文件 %s 中找不到列名 '%s'，跳过。", file_name, target_col_name))
        next
      }
      
      # 提取数据
      temp_df <- data.frame(
        subject = as.character(subject_col),
        value = as.numeric(sulcus_data[[target_col_name]]),
        stringsAsFactors = FALSE
      )
      
      # 4. 重命名 value 列为特定脑区名
      feature_suffix <- tools::file_path_sans_ext(file_name)
      feature_name <- paste0(morph_config$prefix, "_", feature_suffix)
      names(temp_df)[2] <- feature_name
      
      # 5. 合并
      if (is.null(result_data)) {
        result_data <- temp_df
      } else {
        result_data <- full_join(result_data, temp_df, by = 'subject')
      }
      
    }, error = function(e) {
      warning(paste("读取出错:", file_name, "-", e$message))
    })
  }
  return(result_data)
}

# 函数 C: 计算纵向变化
calculate_longitudinal_changes <- function(df, cognitive_col) {
  # 必须包含的列
  if (!all(c("subj_unique", "age", cognitive_col) %in% names(df))) return(NULL)
  
  df <- df %>% 
    arrange(subj_unique, age) %>%
    filter(!is.na(subj_unique), !is.na(age)) # 预先过滤无效行
  
  # 识别形态学列：除了基础列之外的所有数值列
  basic_cols <- c("subject", "subj_unique", "age", "sex", "group", "site", cognitive_col)
  morph_cols <- setdiff(names(df), basic_cols)
  # 再次确认这些列确实存在且是数值
  morph_cols <- morph_cols[sapply(df[morph_cols], is.numeric)]
  
  if (length(morph_cols) == 0) return(NULL)
  
  result_list <- list()
  unique_ids <- unique(df$subj_unique)
  
  for (id in unique_ids) {
    child_data <- df[df$subj_unique == id, ]
    if (nrow(child_data) < 2) next
    
    for (i in 2:nrow(child_data)) {
      curr <- child_data[i, ]
      prev <- child_data[i-1, ]
      
      # 时间差
      time_diff <- as.numeric(curr$age) - as.numeric(prev$age)
      
      # 认知差
      cog_diff <- as.numeric(curr[[cognitive_col]]) - as.numeric(prev[[cognitive_col]])
      
      # 形态学差 (向量化计算)
      morph_curr <- as.numeric(curr[morph_cols])
      morph_prev <- as.numeric(prev[morph_cols])
      morph_diff <- morph_curr - morph_prev
      
      # 只有时间间隔为正才计算
      if (!is.na(time_diff) && time_diff > 0) {
        # 构造单行结果
        row_entry <- list(
          subject = as.character(prev$subject),
          baseline_age = as.numeric(prev$age),
          subj_unique = as.character(prev$subj_unique),
          cognitive_change = cog_diff,
          time_interval = time_diff
        )
        # 添加形态学变化
        row_entry[morph_cols] <- as.list(morph_diff)
        
        result_list[[length(result_list) + 1]] <- row_entry
      }
    }
  }
  
  if (length(result_list) == 0) return(NULL)
  return(bind_rows(result_list))
}

# ================================ 4. 执行逻辑 ================================

run_batch_analysis <- function() {
  cat("\n=== 开始分析 ===\n")
  
  for (cog_group in names(cognitive_configs)) {
    cog_conf <- cognitive_configs[[cog_group]]
    cat(sprintf("\n>> 加载认知组: %s\n", cog_group))
    
    cog_data <- load_cognitive_data(cog_conf)
    
    for (ind_name in names(cog_conf$indicators)) {
      ind_conf <- cog_conf$indicators[[ind_name]]
      col_name <- ind_conf$col
      
      if (!col_name %in% names(cog_data)) {
        cat(sprintf("  ! 警告: 找不到认知列 %s，跳过\n", col_name))
        next
      }
      
      for (morph_name in names(morphological_configs)) {
        morph_conf <- morphological_configs[[morph_name]]
        cat(sprintf("   -> 处理: %s + %s ... ", ind_name, morph_name))
        
        # 1. 加载脑数据
        brain_data <- load_morphological_data(folder_path, morph_conf)
        
        if (is.null(brain_data) || ncol(brain_data) < 2) {
          cat("跳过 (脑数据加载无效)\n")
          next
        }
        
        # 2. 合并
        merged <- inner_join(cog_data, brain_data, by = "subject")
        
        # 3. 计算变化
        res <- calculate_longitudinal_changes(merged, col_name)
        
        if (is.null(res) || nrow(res) < 3) {
          cat("跳过 (样本不足)\n")
          next
        }
        
        # 4. 保存
        # 计算非NA数据的相关性
        clean_res <- res %>% filter(!is.na(baseline_age), !is.na(cognitive_change))
        r_val <- 0
        if (nrow(clean_res) > 3) {
          r_val <- cor(clean_res$baseline_age, clean_res$cognitive_change)
        }
        
        out_dir <- file.path(output_base_path, ind_conf$out)
        if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)
        
        fname <- sprintf("%s_%s_r%.2f.csv", morph_name, ind_name, r_val)
        write.csv(res, file.path(out_dir, fname), row.names = FALSE)
        
        cat(sprintf("完成 (N=%d)\n", nrow(res)))
      }
    }
  }
  cat("\n=== 全部完成 ===\n")
}

run_batch_analysis()
