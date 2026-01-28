# -*- coding: utf-8 -*-
"""
Created on Mon Jul 14 17:57:43 2025

@author: ASUS
"""

import os
import numpy as np
import pandas as pd
from scipy.io import loadmat, savemat
from scipy.stats import pearsonr
from statsmodels.stats.multitest import multipletests

# 1. 载入基因表达
gexpr = loadmat('GeneExpression.mat')
# 推测变量
GE = gexpr['GeneExpression']  # n_brainregion × (1+ngene)
X = GE[:,1:]                  # 全部基因表达
AreaID = GE[:,0].astype(int)  # 脑区ID

n_brainregion, n_gene = X.shape

# 2. 批量处理 features 文件夹
features_dir = './features_withexpression'
surro_dir = './surrogates'

features_files = [f for f in os.listdir(features_dir) if f.endswith('.txt')]

for ff in features_files:
    name = os.path.splitext(ff)[0]   # 如CT
    feat_path = os.path.join(features_dir, ff)
    surro_path = os.path.join(surro_dir, f'surrogates_{name}.mat')
    print(f'\n>>> 正在处理特征: {name}...')

    # 读取Y
    Y = np.loadtxt(feat_path)
    # 若需Y与AreaID对齐（但通常不需要；如有需求可特殊写）
    # Y_aligned = Y[AreaID - 1]  # 假设ID编号从1开始且Y数组按脑区编号升序排列
    # 但大多数场景直接提取Y即可

    # 读取surrogates
    matdict = loadmat(surro_path)
    surrogates = matdict['surrogates']   # 1000 × n_brainregion

    # 检查长度一致
    assert X.shape[0] == len(Y) == surrogates.shape[1]

    # --- 预申请结果 ---
    r_pearson = np.zeros(n_gene)
    p_pearson = np.zeros(n_gene)
    p_permutaion = np.zeros(n_gene)
    r_perm_all = np.zeros((n_gene, surrogates.shape[0]))  # 可选

    # --- 计算相关和置换 ---
    for g in range(n_gene):
        x = X[:,g]
        r, p = pearsonr(x, Y)
        r_pearson[g] = r
        p_pearson[g] = p
        # 置换检验
        r_perm = np.array([pearsonr(x, surrogates[j,:])[0] for j in range(surrogates.shape[0])])
        r_perm_all[g,:] = r_perm
        p_value = (np.sum(r_perm >= r) + 1) / (surrogates.shape[0] + 1)
        p_permutaion[g] = p_value

    # --- FDR校正 ---
    p_pearson_fdr = multipletests(p_pearson, method='fdr_bh')[1]
    p_permutaion_fdr = multipletests(p_permutaion, method='fdr_bh')[1]

    # --- 输出表 ---
    result_df = pd.DataFrame({
        'r_pearson': r_pearson,
        'p_pearson': p_pearson,
        'p_pearson_fdr': p_pearson_fdr,
        'p_permutaion': p_permutaion,
        'p_permutaion_fdr': p_permutaion_fdr
    })
    out_excel = f'corr_pearson_permutation_surrogate_{name}.xlsx'
    result_df.to_excel(out_excel, index=False)
    print(f"结果保存为: {out_excel}")

    # 可选，保存全部置换相关矩阵
    # savemat(f'perm_r_{name}.mat', {'r_perm': r_perm_all})

print("全部批量检验完成！")




firstrow = pd.read_csv('expression_BV.csv', nrows=1, header=None)
firstrow_T = firstrow.T
firstrow_T.to_csv('genelist.csv', index=False, header=False)