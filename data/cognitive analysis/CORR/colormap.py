import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import re

# ================= 配置区域 =================
# CSV 文件列表及其对应的认知维度和半球
file_configs = [
    {"file": "FINAL_SUMMARY_All_Significant_Correlations - WM-left.csv", "domain": "WM", "side": "L"},
    {"file": "FINAL_SUMMARY_All_Significant_Correlations-WM-right.csv", "domain": "WM", "side": "R"},
    {"file": "FINAL_SUMMARY_All_Significant_Correlations-A-left.csv", "domain": "Alerting", "side": "L"},
    {"file": "FINAL_SUMMARY_All_Significant_Correlations-A-right.csv", "domain": "Alerting", "side": "R"},
    {"file": "FINAL_SUMMARY_All_Significant_Correlations -O_left.csv", "domain": "Orienting", "side": "L"},
    {"file": "FINAL_SUMMARY_All_Significant_Correlations-O-right.csv", "domain": "Orienting", "side": "R"},
    {"file": "FINAL_SUMMARY_All_Significant_Correlations-C-left.csv", "domain": "Conflict", "side": "L"},
    {"file": "FINAL_SUMMARY_All_Significant_Correlations-C-right.csv", "domain": "Conflict", "side": "R"},
]

# 精确的指标类型缩写映射
metric_map = {
    'SW_': 'SW',
    'CT_': 'CT',
    'SA_TIV_adjusted_': 'SA(adj)',
    'SL_TIV_adjusted_': 'SL(adj)',
    'meanD_TIV_adjusted_': 'meanD(adj)',
    'maxD_TIV_adjusted_': 'maxD(adj)'
}

# 字体与导出设置 (确保 SVG 文本可编辑)
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['svg.fonttype'] = 'none'

# ================= 数据整合 =================
all_data = []

def parse_region(region_str):
    metric = "Unknown"
    # 按长度排序 key 确保匹配最长的字符串
    for key in sorted(metric_map.keys(), key=len, reverse=True):
        if region_str.startswith(key):
            metric = metric_map[key]
            break
    
    # 提取脑沟名称 (移除所有可能的指标前缀和末尾的 TIV 标识)
    clean_name = re.sub(r'^(SW_|CT_|SA_TIV_adjusted_|SL_TIV_adjusted_|meanD_TIV_adjusted_|maxD_TIV_adjusted_)', '', region_str)
    clean_name = re.sub(r'(_left|_right)?_merged_with_TIV$', '', clean_name)
    return clean_name, metric

for cfg in file_configs:
    if not os.path.exists(cfg['file']):
        print(f"跳过不存在的文件: {cfg['file']}")
        continue
        
    df = pd.read_csv(cfg['file'])
    for _, row in df.iterrows():
        sulcus, metric = parse_region(row['Region'])
        all_data.append({
            'Sulcus': sulcus,
            'Cognition': f"{cfg['domain']}\n({cfg['side']})",
            'r': row['Partial_cor'],
            'p_fdr': row['Partial_cor_p_fdr'],
            'Metric': metric,
            'Abs_r': abs(row['Partial_cor'])
        })

summary_df = pd.DataFrame(all_data)

# 聚合逻辑：对于同一个脑沟和认知维度，如果存在多个显著指标，保留 |r| 最大的那个进行显色
agg_df = summary_df.sort_values('Abs_r', ascending=False).drop_duplicates(['Sulcus', 'Cognition'])

# ================= 绘图准备 =================
pivot_r = agg_df.pivot(index='Sulcus', columns='Cognition', values='r').fillna(0)

def get_label(row):
    # 标注显著性星号
    sig = ""
    if row['p_fdr'] < 0.001: sig = "***"
    elif row['p_fdr'] < 0.01: sig = "**"
    elif row['p_fdr'] < 0.05: sig = "*"
    return f"{row['Metric']}{sig}"

agg_df['Annot'] = agg_df.apply(get_label, axis=1)
pivot_annot = agg_df.pivot(index='Sulcus', columns='Cognition', values='Annot').fillna("")

# 排序：按脑沟名称字母顺序
pivot_r = pivot_r.sort_index()
pivot_annot = pivot_annot.reindex(pivot_r.index)

# ================= 绘制热图 =================
# 根据脑沟数量动态计算高度 (每个脑沟占用 0.35 inch，基础高度 4 inch)
dynamic_height = max(10, len(pivot_r) * 0.35 + 4)
plt.figure(figsize=(10, dynamic_height))

# 使用 vlag 配色 (冷暖平衡，符合科研审美)
ax = sns.heatmap(pivot_r, 
                 annot=pivot_annot, 
                 fmt="", 
                 cmap='vlag', 
                 center=0,
                 linewidths=0.5, 
                 linecolor='lightgray',
                 cbar_kws={'label': 'Partial Correlation (r)', 'shrink': 0.5},
                 annot_kws={"size": 9, "family": "Arial"})

# 美化设置
plt.title('Summary of Structure-Behavior Partial Correlations', fontsize=16, fontweight='bold', pad=30)
plt.xlabel('Cognitive Domains (Hemisphere)', fontsize=12, fontweight='bold')
plt.ylabel('Sulcal Morphometry', fontsize=12, fontweight='bold')

# 调整刻度标签
plt.xticks(rotation=0)
plt.yticks(rotation=0)

# 添加详细图注
legend_text = "Metric Index: SA(adj)=Surface Area, maxD(adj)=Max Depth, meanD(adj)=Mean Depth, SL(adj)=Sulcal Length, SW=Sulcal Width, CT=Cortical Thickness.\nSignificance (FDR corrected): * p<0.05, ** p<0.01, *** p<0.001"
plt.figtext(0.5, 0.01, legend_text, ha="center", fontsize=10, 
            bbox={"facecolor":"whitesmoke", "alpha":0.8, "pad":8})

# 保存文件
plt.tight_layout(rect=[0, 0.05, 1, 0.95])
plt.savefig("Cognition_Summary_Heatmap.pdf", dpi=300, bbox_inches='tight')
plt.savefig("Cognition_Summary_Heatmap.svg", format='svg', bbox_inches='tight')

print(f"全部特征已处理。图中共包含 {len(pivot_r)} 个特征。")
print("已成功保存 PDF 和可编辑的 SVG 格式。")