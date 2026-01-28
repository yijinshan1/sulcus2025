import numpy as np
from sklearn.neighbors import LocalOutlierFactor as LOF
import matplotlib.pyplot as plt
import pandas as pd
import os

folder_path = r"E:\lsy_group\7.9reorganize\7.9reorganize\4.cognition_analysis\mergeaslabel\right"
file_names = os.listdir(folder_path)
for file_name in file_names:

    df = pd.read_csv(os.path.join(folder_path, file_name))
    # last_col = df.columns[-1]

    # # 重新排列列顺序
    # new_order = [last_col] + list(df.columns[:-1])
    # df_reordered = df[new_order]

    # === 1. 只用第6,7,8,9,10,12列检测离群 ===
    feature_indices = [10, 11, 12, 13, 14,16]
    X_lof = df.iloc[:, feature_indices]
    X_no_na = X_lof.dropna()
    df_reordered_no_na = df.loc[X_no_na.index, :]
    X_for_lof = X_no_na

    clf = LOF(n_neighbors=10, algorithm='auto', contamination=0.05, n_jobs=-1, p=2)
    y_pred = clf.fit_predict(X_for_lof)
    X_scores = clf.negative_outlier_factor_

    # # 画图还是用新特征中前两列（如果需要）
    # X1 = X_for_lof[y_pred == 1]
    # X2 = X_for_lof[y_pred == -1]
    # X1 = np.array(X1, dtype=float)
    # X2 = np.array(X2, dtype=float)
    # plt.title('Local Outlier Factor (LOF)')
    # plt.scatter(X1[:, 0], X1[:, 1], color='b', s=1.5, label='Normal')
    # plt.scatter(X2[:, 0], X2[:, 1], color='r', s=1.5, label='Outliers')
    # radius = (X_scores.max() - X_scores) / (X_scores.max() - X_scores.min())
    # X_plot = np.array(X_for_lof, dtype=float)
    # plt.scatter(X_plot[:, 0], X_plot[:, 1], s=2000 * radius, edgecolors='g', linewidths=.4,
    #             facecolors='none', label='Scores')
    # plt.axis('tight')
    # legend = plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    # legend.legendHandles[0]._sizes = [10]
    # legend.legendHandles[1]._sizes = [10]
    # legend.legendHandles[2]._sizes = [20]
    # new_labels = ["Normal", "Outliers", "Scores"]
    # for text, label in zip(legend.texts, new_labels):
    #     text.set_text(label)
    # plt.show()

    # === 2. 用LOF结果筛整行（保留正常点） ===
    After_LOF = df_reordered_no_na[y_pred == 1]

    # === 3. zscore全部数据（除前两列，全都标准化） ===
    data_to_normalize = After_LOF.iloc[:, 7:]
    def z_score_normalize(data):
        mean = np.mean(data, axis=0)
        std_dev = np.std(data, axis=0)
        return (data - mean) / std_dev
    After_zscore = pd.DataFrame(z_score_normalize(data_to_normalize),
                                columns=data_to_normalize.columns,
                                index=data_to_normalize.index)
    columns_to_keep = ['SW', 'CT', 'SA_TIV_adjusted', 'maxD_TIV_adjusted', 'meanD_TIV_adjusted', 'SL_TIV_adjusted']

    # 从 After_zscore 中只选取这些列，并覆盖原变量（或者赋值给新变量）
    After_zscore = After_zscore[columns_to_keep]
    # 拼回前两列
    result = pd.concat([After_LOF.iloc[:, 0:7], After_zscore], axis=1)

    # === 4. 按原After_LOF的列顺序保存 ===
    save_path = r'E:\lsy_group\7.9reorganize\7.9reorganize\4.cognition_analysis\mergeaslabel_std\left\\' + file_name
    result.to_csv(save_path, index=False)