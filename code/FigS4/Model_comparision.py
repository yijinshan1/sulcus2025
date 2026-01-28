# -*- coding: utf-8 -*-
"""
Created on Tue Jan 27 10:46:38 2026

@author: ASUS
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ================= 配置区域 =================
# 定义基础路径和功能列表
base_dir = r"E:\lsy_group\7.9reorganize\7.9reorganize\6.paper\CB_revise\submit\4.cognition_analysis\change_analysis\change_data"
functions = ["WM", "ATT_A", "ATT_O", "ATT_C"]
sides = ["left", "right"]
filename = "Model_RMSE_Comparison_Table.csv"
output_pdf = os.path.join(base_dir, "Model_Comparison_Summary_Plot.pdf")

# 设置绘图风格
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("white")
sns.set_context("paper", font_scale=1.2)

# ================= 数据读取与处理 =================
all_data = []

for func in functions:
    for side in sides:
        # 拼接路径
        path = os.path.join(base_dir, side, func, "result_new_style", filename)
        
        if os.path.exists(path):
            df = pd.read_csv(path)
            # 添加标识列
            df['Function'] = func
            df['Side'] = side.capitalize()
            # 简化文件名作为横坐标标签（例如：CT_attention_A_r -> CT）
            df['Label'] = df['File'].apply(lambda x: x.split('_')[0] + f"({side[0].upper()})")
            all_data.append(df)
        else:
            print(f"跳过不存在的文件: {path}")

if not all_data:
    print("未找到任何 RMSE 比较文件，请检查路径。")
else:
    full_df = pd.concat(all_data, ignore_index=True)

    # ================= 绘图部分 =================
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    axs = axs.flatten()
    
    # 调色盘：LASSO 和 Ridge
    colors = ["#4C72B0", "#C44E52"] # 经典的低饱和度红蓝配色

    for i, func in enumerate(functions):
        ax = axs[i]
        # 筛选当前功能的数据
        plot_df = full_df[full_df['Function'] == func].copy()
        
        # 将数据从宽表转为长表，方便 seaborn 绘图
        plot_df_melted = plot_df.melt(
            id_vars=['Label', 'Side'], 
            value_vars=['Lasso_RMSE', 'FullModel_RMSE'],
            var_name='Model', value_name='RMSE'
        )
        # 优化模型名称显示
        plot_df_melted['Model'] = plot_df_melted['Model'].replace({
            'Lasso_RMSE': 'LASSO', 
            'FullModel_RMSE': 'Ridge (Full)'
        })

        # 绘制分组柱状图
        sns.barplot(
            data=plot_df_melted, x='Label', y='RMSE', hue='Model',
            ax=ax, palette=colors, edgecolor=".2", alpha=0.8
        )

        # 细节调整
        ax.set_title(f'Performance Comparison: {func}', fontsize=16, fontweight='bold', pad=15)
        ax.set_xlabel('Regions / Features', fontsize=12)
        ax.set_ylabel('RMSE (Lower is Better)', fontsize=12)
        ax.tick_params(axis='x', rotation=45)
        ax.legend(title='Model Type', frameon=False)
        
        # 去除上方和右侧边框 (Despine)
        sns.despine(ax=ax)
        
        # 设置 Y 轴范围自适应（增加一点冗余空间）
        if not plot_df_melted.empty:
            y_max = plot_df_melted['RMSE'].max()
            ax.set_ylim(0, y_max * 1.2)

    plt.tight_layout()
    plt.savefig(output_pdf, bbox_inches='tight', format='pdf', dpi=300)
    print(f"\n批量绘图完成！文件已保存至: {output_pdf}")

    # ================= 生成汇总表格 =================
    summary_table_path = os.path.join(base_dir, "All_Models_RMSE_Summary.csv")
    full_df.to_csv(summary_table_path, index=False)
    print(f"汇总数据表已保存至: {summary_table_path}")
    from scipy.stats import ttest_rel  # 导入配对t检验工具

# ================= 修正版：含显著性标注的总体 RMSE 分布对比图 =================
plt.figure(figsize=(12, 8))

# 1. 数据准备 (长格式用于绘图)
overall_melted = full_df.melt(
    id_vars=['Function'], 
    value_vars=['Lasso_RMSE', 'FullModel_RMSE'],
    var_name='Model', value_name='RMSE'
)
overall_melted['Model'] = overall_melted['Model'].replace({
    'Lasso_RMSE': 'LASSO', 
    'FullModel_RMSE': 'Ridge (Full)'
})

# 2. 绘制拆分小提琴图
ax = sns.violinplot(
    data=overall_melted, x='Function', y='RMSE', hue='Model',
    split=True, inner="quartile", palette=colors, alpha=0.7
)

# 3. 叠加上抖动散点
sns.stripplot(
    data=overall_melted, x='Function', y='RMSE', hue='Model',
    dodge=True, color="black", alpha=0.3, jitter=True, size=4, ax=ax
)

# 4. 统计检验与显著性标注
functions_list = ["WM", "ATT_A", "ATT_O", "ATT_C"]
for i, func in enumerate(functions_list):
    # 提取该功能下的两组配对数据
    func_data = full_df[full_df['Function'] == func]
    
    if len(func_data) > 1: # 确保有足够数据进行t检验
        # 执行配对t检验：比较该功能下所有指标的 Lasso RMSE 和 Ridge RMSE
        stat, p_val = ttest_rel(func_data['Lasso_RMSE'], func_data['FullModel_RMSE'])
        
        # 定义显著性标签
        if p_val < 0.001:
            sig_label = "***"
        elif p_val < 0.01:
            sig_label = "**"
        elif p_val < 0.05:
            sig_label = "*"
        else:
            sig_label = "n.s."
            
        # 计算标注位置 (在该组小提琴图的最高点上方)
        y_max = max(func_data['Lasso_RMSE'].max(), func_data['FullModel_RMSE'].max())
        y_pos = y_max + (full_df['FullModel_RMSE'].max() * 0.05) # 向上偏移 5%
        
        # 在图上添加标注
        # i 是 x 轴的坐标索引
        ax.text(i, y_pos, sig_label, ha='center', va='bottom', fontsize=12, fontweight='bold', color='black')
        
        # 可选：画一条横线连接两边（虽然 split 形式通常直接标在中间即可）
        # ax.plot([i-0.2, i+0.2], [y_pos, y_pos], lw=1, c='black') 

# 5. 统一处理图例与美化
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[:2], labels[:2], title='Model Type', 
          bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)

plt.title('Overall Model RMSE Distribution with Significance (Paired t-test)', fontsize=16, fontweight='bold', pad=25)
plt.xlabel('Cognitive Function', fontsize=14)
plt.ylabel('RMSE', fontsize=14)

sns.despine()

# 保存
output_violin_pdf = os.path.join(base_dir, "Overall_Model_RMSE_Violin_with_Sig.pdf")
plt.savefig(output_violin_pdf, bbox_inches='tight', format='pdf', dpi=300)

print(f"\n含显著性标注的风琴图已保存至: {output_violin_pdf}")
    