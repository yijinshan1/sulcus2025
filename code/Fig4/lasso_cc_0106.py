

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LassoCV, Lasso, RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from statsmodels.stats.multitest import multipletests

# ================= 配置区域 =================
# 请修改为您的实际路径
input_folder = r"E:\lsy_group\7.9reorganize\7.9reorganize\6.paper\CB_revise\submit\4.cognition_analysis\change_analysis\change_data\left\WM"
output_folder = r"E:\lsy_group\7.9reorganize\7.9reorganize\6.paper\CB_revise\submit\4.cognition_analysis\change_analysis\change_data\left\WM\result_new_style"
os.makedirs(output_folder, exist_ok=True)

# 特征列配置
# cols_to_use_x = list(range(7, 33))#左脑34右脑33 ATTENTION
# cols_to_use_y = list(range(3, 4))
cols_to_use_x = list(range(5, 32)) #左脑32右脑31  WORKING MEMORY
cols_to_use_y = list(range(3, 4))

# 字体设置
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False
sns.set_context("talk", font_scale=0.9) # 调整绘图风格
# ===========================================

all_results = []

print("开始分析：LASSO (筛选) vs Ridge (全模型)...")

for fname in os.listdir(input_folder):
    if not fname.lower().endswith('.csv'):
        continue
        
    in_fp = os.path.join(input_folder, fname)
    out_prefix = os.path.splitext(fname)[0]
    print(f"\n正在处理: {fname}")

    # 1. 读取数据
    try:
        X_raw = pd.read_csv(in_fp, usecols=cols_to_use_x)
        Y_raw = pd.read_csv(in_fp, usecols=cols_to_use_y)
    except Exception as e:
        print(f"读取错误: {e}")
        continue

    X = X_raw.fillna(0)
    Y = Y_raw.fillna(0)
    feature_names = np.array(X.columns.tolist())

    # 2. 数据划分与标准化 (严格防止数据泄漏)
    # random_state=23 保持您之前的结果一致性
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=23)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train) # 仅在训练集fit
    X_test_scaled = scaler.transform(X_test)       # 应用于测试集
    
    y_train_flat = y_train.values.ravel()
    y_test_flat = y_test.values.ravel()

    # ==========================================================================
    # 模型 A: LASSO (特征筛选 - 您的核心方法)
    # ==========================================================================
    # 使用 LassoCV 自动寻找最佳 alpha
    lasso_cv = LassoCV(eps=1e-4, n_alphas=100, cv=5, random_state=42, max_iter=10000)
    lasso_cv.fit(X_train_scaled, y_train_flat)
    
    best_alpha = lasso_cv.alpha_
    y_pred_lasso = lasso_cv.predict(X_test_scaled)
    lasso_rmse = mean_squared_error(y_test_flat, y_pred_lasso, squared=False)
    
    # 提取 LASSO 非零系数
    lasso_coefs = lasso_cv.coef_
    lasso_selected = feature_names[np.abs(lasso_coefs) > 1e-5]

    if np.std(y_pred_lasso) < 1e-9:
        r_lasso, p_lasso = 0.0, 1.0
    else:
        r_lasso, p_lasso = stats.spearmanr(y_test_flat, y_pred_lasso)

    # ==========================================================================
    # 模型 B: Ridge Regression (全模型 - 类似 Voorhies et al., 2021)
    # ==========================================================================
    # Ridge 保留所有特征，解决共线性，作为“Full Model”的基准
    ridge_cv = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0], cv=5)
    ridge_cv.fit(X_train_scaled, y_train_flat)
    
    y_pred_ridge = ridge_cv.predict(X_test_scaled)
    ridge_rmse = mean_squared_error(y_test_flat, y_pred_ridge, squared=False)
    ridge_coefs = ridge_cv.coef_
    
    if np.std(y_pred_ridge) < 1e-9:
        r_ridge, p_ridge = 0.0, 1.0
    else:
        r_ridge, p_ridge = stats.spearmanr(y_test_flat, y_pred_ridge)

    # ==========================================================================
    # 准备绘图数据：生成 LASSO Path (为了画热图)
    # ==========================================================================
    # 手动生成 alpha 路径以绘制平滑的热图
    alpha_range = np.logspace(np.log10(best_alpha) - 2.5, np.log10(best_alpha) + 1.5, 60)
    coefs_path = []
    for a in alpha_range:
        l = Lasso(alpha=a, max_iter=10000, random_state=42)
        l.fit(X_train_scaled, y_train_flat)
        coefs_path.append(l.coef_)
    coefs_path = np.array(coefs_path).T

    # ==========================================================================
    # ==========================================================================
    # 绘图 Dashboard (2x2) - 修正维度对齐及字母排序版
    # ==========================================================================
    sns.set_style("white") 
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    plt.subplots_adjust(hspace=0.4, wspace=0.3)
    
    # 提取统一的横坐标范围
    log_alpha_range = np.log10(alpha_range)
    x_min, x_max = log_alpha_range.min(), log_alpha_range.max()

    # --- [关键修改]：确保纵轴特征按字母顺序排序 ---
    sort_idx = np.argsort(feature_names) # 获取字母排序索引
    feature_names_sorted = feature_names[sort_idx]
    coefs_path_sorted = coefs_path[sort_idx, :]

    # --- 图1: LASSO 系数热图 (左上) ---
    # 使用排序后的数据
    X_grid, Y_grid = np.meshgrid(log_alpha_range, np.arange(len(feature_names_sorted)))
    
    im = axs[0,0].pcolormesh(X_grid, Y_grid, coefs_path_sorted, cmap='vlag', shading='nearest', 
                             edgecolors='lightgray', linewidth=0.5, 
                             vmin=-np.max(np.abs(coefs_path_sorted)), vmax=np.max(np.abs(coefs_path_sorted)))
    
    # 设置 Y 轴标签
    axs[0,0].set_yticks(np.arange(len(feature_names_sorted)))
    axs[0,0].set_yticklabels(feature_names_sorted, fontsize=10)
    
    # [重要]：反转 Y 轴，使字母 A 在最上方
    axs[0,0].invert_yaxis()
    
    # 标记最佳 Alpha
    best_log_alpha = np.log10(best_alpha)
    axs[0,0].axvline(best_log_alpha, color='red', linestyle='--', linewidth=1.5)
    
    # 因为反转了 Y 轴，0 现在在顶部，所以将文字放在 0 附近
    axs[0,0].text(best_log_alpha + 0.05, 0, 'Best alpha', color='red', 
                  rotation=90, va='top', ha='left', fontsize=10)
    
    axs[0,0].set_title('Heatmap of Lasso Coefficients', fontsize=14, fontweight='bold')
    axs[0,0].set_ylabel('Features (A-Z Order)', fontsize=12)
    axs[0,0].set_xlabel('log10(alpha)', fontsize=12)
    axs[0,0].set_xlim(x_min, x_max)

    # 颜色条
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(axs[0,0])
    cax = divider.append_axes("right", size="3%", pad=0.1)
    plt.colorbar(im, cax=cax, label='Coefficient Value')

    # --- 图2: CV RMSE 变化图 (右上) ---
    mean_mse_path = np.mean(lasso_cv.mse_path_, axis=1)
    mean_rmse_path = np.sqrt(mean_mse_path)
    log_alphas_cv = np.log10(lasso_cv.alphas_)
    
    axs[0,1].plot(log_alphas_cv, mean_rmse_path, linestyle='-', color='black', linewidth=2)
    axs[0,1].axvline(best_log_alpha, linestyle='--', color='red', linewidth=1.5)
    
    # 标注 Best alpha
    y_min_cv, y_max_cv = axs[0,1].get_ylim()
    axs[0,1].text(best_log_alpha + 0.05, y_min_cv + (y_max_cv - y_min_cv) * 0.1, 
                  'Best alpha', color='red', rotation=90, fontsize=10)
    
    axs[0,1].set_xlim(x_min, x_max) # 强制 X 轴对齐
    axs[0,1].set_title('Cross-Validation RMSE vs. log10(alpha)', fontsize=14, fontweight='bold')
    axs[0,1].set_xlabel('log10(alpha)', fontsize=12)
    axs[0,1].set_ylabel('CV RMSE', fontsize=12)
    axs[0,1].grid(True, linestyle='--', alpha=0.5)

    # 强制 top row 两个子图的长宽比一致
    for ax in [axs[0,0], axs[0,1]]:
        ax.set_box_aspect(0.6) 

    # --- 图3 & 图4: 预测对比图 ---
    # (这部分代码建议保留之前的灰色调科研风格，确保美观)
    for ax, y_pred, title, r_val, p_val in zip(
        [axs[1,0], axs[1,1]], 
        [y_pred_lasso, y_pred_ridge], 
        ['LASSO: Predictions vs. Actuals', 'Ridge: Predictions vs. Actuals'],
        [r_lasso, r_ridge],
        [p_lasso, p_ridge]
    ):
        sns.regplot(x=y_test_flat, y=y_pred, ax=ax, 
                    scatter_kws={'color': 'darkgrey', 's': 50, 'alpha': 0.6},
                    line_kws={'color': 'black', 'linewidth': 2})
        ax.text(0.05, 0.9, f'r = {r_val:.3f}\np = {p_val:.2e}', 
                transform=ax.transAxes, fontsize=11, bbox=dict(facecolor='white', alpha=0.5))
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_box_aspect(0.8)

    # 保存为 PDF
    plt.savefig(os.path.join(output_folder, f"{out_prefix}_Analysis_Dashboard.pdf"), bbox_inches='tight', format='pdf')
    plt.close()

    

    # ==========================================================================
    # 结果收集 (逻辑不变)
    # ==========================================================================
    lasso_coef_dict = dict(zip(lasso_selected, lasso_coefs[np.abs(lasso_coefs) > 1e-5]))
    all_results.append({
        'File': out_prefix,
        'Best_Alpha': best_alpha,
        'Lasso_RMSE': lasso_rmse,
        'FullModel_RMSE': ridge_rmse,
        'Lasso_r': r_lasso,
        'Lasso_p': p_lasso,
        'FullModel_r': r_ridge,
        'FullModel_p': p_ridge,
        'Lasso_Features_N': len(lasso_selected),
        'Lasso_Coefficients': str(lasso_coef_dict)
    })

# ==========================================================================
# 汇总与生成对比表
# ==========================================================================
summary_df = pd.DataFrame(all_results)
if not summary_df.empty:
    # 1. 额外生成专门的 RMSE 对比表
    rmse_table = summary_df[['File', 'Lasso_RMSE', 'FullModel_RMSE']].copy()
    rmse_table['Improvement'] = rmse_table['FullModel_RMSE'] - rmse_table['Lasso_RMSE']
    rmse_table.to_csv(os.path.join(output_folder, "Model_RMSE_Comparison_Table.csv"), index=False)

    # 2. 原有的 FDR 逻辑及完整汇总保存
    p_values_lasso = np.nan_to_num(summary_df['Lasso_p'].values, nan=1.0)
    reject_lasso, q_values_lasso, _, _ = multipletests(p_values_lasso, method='fdr_bh', alpha=0.05)
    summary_df['Lasso_FDR_q'] = q_values_lasso
    summary_df['Lasso_Significant'] = reject_lasso

    p_values_full = np.nan_to_num(summary_df['FullModel_p'].values, nan=1.0)
    reject_full, q_values_full, _, _ = multipletests(p_values_full, method='fdr_bh', alpha=0.05)
    summary_df['FullModel_FDR_q'] = q_values_full
    summary_df['FullModel_Significant'] = reject_full

summary_path = os.path.join(output_folder, "Final_Comparison_Results.csv")
summary_df.to_csv(summary_path, index=False)
print(f"PDF图表与 RMSE 对比表已保存至: {output_folder}")