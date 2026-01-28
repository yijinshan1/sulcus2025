

# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 14:33:07 2025
Updated for publication-quality plots with a grayscale color scheme.

@author: 15542
"""

import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- 1. 数据加载和预处理 ---

# 检查文件是否存在
if not os.path.exists('all_models_details.xlsx') or not os.path.exists('app_time.xlsx'):
    print("错误：请确保 'all_models_details.xlsx' 和 'app_time.xlsx' 文件在当前目录下。")
else:
    # 读取数据
    df_models = pd.read_excel('all_models_details.xlsx')
    df_app_time = pd.read_excel('app_time.xlsx', sheet_name='Sheet1')

    # 合并数据
    merged_df = pd.merge(df_models, df_app_time, on='label', how='left')

    # --- 2. 相关性分析 ---

    # 定义分析列
    time_cols = ['appearance', 'T10', 'T50']
    age_cols = ['Delta_adj_R2', 'anova_fdr_p']
    features = merged_df['feature'].unique()

    # 存储结果
    results = []

    # 计算相关性
    for feature in features:
        feature_data = merged_df[merged_df['feature'] == feature]
        for time_var in time_cols:
            for age_var in age_cols:
                subset = feature_data[[time_var, age_var]].dropna()
                if len(subset) >= 2:
                    corr, p_val = pearsonr(subset[time_var], subset[age_var])
                else:
                    corr, p_val = np.nan, np.nan
                results.append({
                    'feature': feature,
                    'time_var': time_var,
                    'age_var': age_var,
                    'correlation': corr,
                    'p_value': p_val
                })

    results_df = pd.DataFrame(results)

    # 多重比较校正 (FDR - Benjamini/Hochberg)
    final_results = []
    for feature in features:
        feature_df = results_df[results_df['feature'] == feature].copy()
        p_values = feature_df['p_value'].dropna()
        if not p_values.empty:
            _, adj_p, _, _ = multipletests(p_values, method='fdr_bh')
            adj_series = pd.Series(adj_p, index=p_values.index)
            feature_df['adjusted_p'] = adj_series
        else:
            feature_df['adjusted_p'] = np.nan
        final_results.append(feature_df)

    final_df = pd.concat(final_results).reset_index(drop=True)

    # 保存相关性分析结果
    final_df.to_excel('correlation_results_adjusted.xlsx', index=False)
    
    # --- 3. 绘图：生成科研风格图表 (灰色系) ---

    # 创建一个目录来保存图表
    output_dir = "significant_plots_pdf_grayscale"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 设置科研论文风格的全局参数
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': 'Arial',
        'font.size': 14,
        'axes.labelsize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'axes.linewidth': 1.5,
        'lines.linewidth': 2
    })
    sns.set_style("whitegrid")

    # 定义坐标轴标签的映射，使其更具可读性
    label_map = {
        'appearance': 'Appearance',
        'T10': 'Time to 10% Accumulation (T10)',
        'T50': 'Time to 50% Accumulation (T50)',
        'Delta_adj_R2': 'Δ Adjusted R²',
        'anova_fdr_p': 'ANOVA (FDR p-value)'
    }

    # 筛选显著结果并绘图
    significant_df = final_df.dropna(subset=['adjusted_p'])
    significant_df = significant_df[significant_df['adjusted_p'] < 0.05]

    for _, row in significant_df.iterrows():
        # 准备绘图数据
        data = merged_df[
            (merged_df['feature'] == row['feature']) &
            merged_df[row['time_var']].notna() &
            merged_df[row['age_var']].notna()
        ]
        
        # 创建画布
        plt.figure(figsize=(6, 5))
        
        # 使用 regplot 绘制散点图和回归线 (灰色系)
        ax = sns.regplot(
            x=row['time_var'], 
            y=row['age_var'], 
            data=data,
            scatter_kws={'color': '#606060', 'alpha': 0.7, 's': 50}, # 灰色散点
            line_kws={'color': 'black', 'linewidth': 2.5}         # 黑色回归线
        )
        
        # 设置更清晰的坐标轴标签
        plt.xlabel(label_map.get(row['time_var'], row['time_var']))
        plt.ylabel(label_map.get(row['age_var'], row['age_var']))
        
        # 设置简洁的标题
        plt.title(f"Feature: {row['feature']}", fontsize=18, weight='bold')
        
        # 将统计数据添加到图表中
        stats_text = f"ρ = {row['correlation']:.3f}\np-adj = {row['adjusted_p']:.3g}"
        plt.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=12,
                 verticalalignment='top', horizontalalignment='right',
                 bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.5))
        
        plt.tight_layout()
        
        # 保存为PDF格式
        filename = f"{row['feature']}_{row['time_var']}_{row['age_var']}.pdf"
        safe_filename = filename.replace(' ', '_').replace('/', '_')
        plt.savefig(os.path.join(output_dir, safe_filename), format='pdf', dpi=300)
        plt.close()

    print(f"分析完成！结果已保存到 'correlation_results_adjusted.xlsx'。")
    print(f"所有显著相关的灰色系图表已保存为PDF格式，并存放在 '{output_dir}' 文件夹中。")

