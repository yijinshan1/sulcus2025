import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# 1. 数据读取 (请确保文件名正确)
df_main = pd.read_excel('demographic.xlsx')
ant = pd.read_excel('ANT.xlsx')
wm = pd.read_excel('WM.xlsx')

# 2. 预处理：确定状态
ant_ids = set(ant['id'])
wm_ids = set(wm['id'])

def get_status(row):
    has_ant = row['id'] in ant_ids
    has_wm = row['id'] in wm_ids
    if has_ant and has_wm: return 'Both'
    if has_ant: return 'ANT only'
    if has_wm: return 'WM only'
    return 'None'

df_main['status'] = df_main.apply(get_status, axis=1)
df_main['scan_count'] = df_main.groupby('subj_unique')['id'].transform('count')

# 映射配置
color_palette = {1: '#0072B2', 0: '#D55E00'}
marker_map = {
    'ANT only': 'o',  # 圆圈
    'WM only': '^',   # 三角
    'Both': 's',      # 方块
    'None': '.'       # 小点
}

# 3. 增强版绘图函数：支持设置目标比例 (target_aspect = 宽/高)
def plot_styled_distribution(data, title, ax, target_aspect=1.0):
    if data.empty: return
    
    # 确定横轴位置
    min_ages = data.groupby('subj_unique')['Age'].min()
    sorted_subj = min_ages.sort_values().index.tolist()
    x_pos_map = {subj: i for i, subj in enumerate(sorted_subj)}
    data = data.copy()
    data['x_pos'] = data['subj_unique'].map(x_pos_map)

    # 绘制纵向随访连线
    for subj, group in data.groupby('subj_unique'):
        if len(group) > 1:
            group = group.sort_values('Age')
            ax.plot(group['x_pos'], group['Age'], color='#999999', alpha=0.3, linewidth=2, zorder=1)

    # 按状态绘制点
    for status, marker in marker_map.items():
        subset = data[data['status'] == status]
        if not subset.empty:
            sns.scatterplot(
                data=subset, x='x_pos', y='Age', hue='gender',
                palette=color_palette, marker=marker, 
                s=100 if marker != '.' else 40,
                alpha=0.85, legend=False, ax=ax, zorder=2
            )

    # 核心修改：强制设置坐标轴的物理比例
    x_range = len(sorted_subj) if len(sorted_subj) > 0 else 1
    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min
    # target_aspect = Width/Height. Matplotlib aspect = Ly/Lx
    # 我们需要 Ly/Lx = x_range / (y_range * target_aspect)
    ax.set_aspect(x_range / (y_range * target_aspect))

    ax.set_title(title, fontsize=15, fontweight='bold')
    ax.set_xlabel('Participants (Sorted by Age)', fontsize=12)
    ax.set_ylabel('Age (years)', fontsize=12)
    
    if len(sorted_subj) > 15:
        ticks = range(0, len(sorted_subj), max(1, len(sorted_subj)//15))
        ax.set_xticks(ticks)
        ax.set_xticklabels([sorted_subj[i] for i in ticks], rotation=45, fontsize=8)
    else:
        ax.set_xticks(range(len(sorted_subj)))
        ax.set_xticklabels(sorted_subj, rotation=45, fontsize=8)

# 4. 图例设置
legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='Boy (gender=1)', markerfacecolor='#0072B2', markersize=10),
    Line2D([0], [0], marker='o', color='w', label='Girl (gender=0)', markerfacecolor='#D55E00', markersize=10),
    Line2D([0], [0], marker='o', color='gray', label='ANT only', markerfacecolor='gray', markersize=10, linestyle='None'),
    Line2D([0], [0], marker='^', color='gray', label='WM only', markerfacecolor='gray', markersize=10, linestyle='None'),
    Line2D([0], [0], marker='s', color='gray', label='Both (ANT & WM)', markerfacecolor='gray', markersize=10, linestyle='None')
]

# --- 绘制全样本大图 (3:1) ---
fig_main, ax_main = plt.subplots(figsize=(18, 6)) # 增加画布宽度
plot_styled_distribution(df_main, 'Full Sample Age Distribution (3:1 Ratio)', ax_main, target_aspect=3.0)
ax_main.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.01, 1), title="Legend")
plt.tight_layout()
plt.savefig('age_distribution_main_3to1.pdf', dpi=300, bbox_inches='tight')

# --- 绘制三个子图 (1:1 正方形) ---
fig_sub, axes = plt.subplots(1, 3, figsize=(18, 7))
titles = ['A. 1 Scan', 'B. 2 Scans', 'C. 3+ Scans']
filters = [df_main['scan_count'] == 1, df_main['scan_count'] == 2, df_main['scan_count'] >= 3]

for i in range(3):
    plot_styled_distribution(df_main[filters[i]], titles[i], axes[i], target_aspect=1.0)

axes[2].legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.05, 1), title="Legend")
plt.tight_layout()
plt.savefig('age_distribution_subpanels_square.pdf', dpi=300, bbox_inches='tight')

plt.show()