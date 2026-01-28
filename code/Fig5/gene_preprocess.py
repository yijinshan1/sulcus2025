import os
import pandas as pd
from scipy.spatial.distance import pdist, squareform
import numpy as np
import abagen

import nibabel as nib
from abagen import reporting,images
from brainsmash.mapgen.base import Base
from scipy import io


# print("当前目录：", os.getcwd())
# df = pd.read_csv('features.csv', sep=',')
# print("读取到的列名:", df.columns.tolist())
# feature_columns = df.columns[1:]

# for col in feature_columns:
#     out_path = os.path.join(os.getcwd(), f"{col}.txt")
#     df[col].to_csv(out_path, index=False, header=False)
#     print(f"{col}.txt 保存到 {out_path}")

# print("当前目录下所有文件：", os.listdir(os.getcwd()))


# ================= 配置区 =================
data_dir = r'C:/Users/15542/abagen-data/microarray'
atlas_info_path = r'E:/lsy_group/7.9reorganize/7.9reorganize/5.gene_analysis/atlas_generate/bvatlas_info.csv'
atlas_path = r'E:/lsy_group/7.9reorganize/7.9reorganize/5.gene_analysis/atlas_generate/merged_atlas_relabel.nii'
output_expression_path = r'E:/lsy_group/7.9reorganize/7.9reorganize/5.gene_analysis/expression_BV.csv'
output_excluded_path = r'E:/lsy_group/7.9reorganize/7.9reorganize/5.gene_analysis/excluded_samples_report.csv'
output_kept_path= r'E:/lsy_group/7.9reorganize/7.9reorganize/5.gene_analysis/included_samples_report.csv'
# ================= 1. 准备工作 =================
print("正在加载图谱...")
atlas = images.check_atlas(atlas_path)
atlas_img = nib.load(atlas_path)
atlas_data = atlas_img.get_fdata()
atlas_affine = atlas_img.affine

donors = ['9861', '10021', '12876', '14380', '15496', '15697']
files = abagen.fetch_microarray(donors=donors, data_dir=data_dir)

# ================= 2. 手动执行“探针-脑沟”匹配检查 =================
print("正在执行探针位置检查 (基于 MNI 坐标)...")

all_samples_list = []

for donor in donors:
    # 读取注释
    annot_path = files[donor]['annotation']
    df_annot = pd.read_csv(annot_path)
    
    # 坐标转换
    mni_coords = df_annot[['mni_x', 'mni_y', 'mni_z']].values
    inv_affine = np.linalg.inv(atlas_affine)
    coords_homo = np.hstack((mni_coords, np.ones((mni_coords.shape[0], 1))))
    voxel_coords = np.dot(inv_affine, coords_homo.T).T[:, :3]
    voxel_coords = np.round(voxel_coords).astype(int)
    
    labels_found = []
    max_x, max_y, max_z = atlas_data.shape
    
    for idx, (x, y, z) in enumerate(voxel_coords):
        if 0 <= x < max_x and 0 <= y < max_y and 0 <= z < max_z:
            val = atlas_data[x, y, z]
            if val > 0:
                labels_found.append(int(val))
            else:
                labels_found.append(0) 
        else:
            labels_found.append(0)
            
    df_annot['mapped_label'] = labels_found
    df_annot['donor_id'] = donor
    all_samples_list.append(df_annot)

full_annot_df = pd.concat(all_samples_list, ignore_index=True)
excluded_df = full_annot_df[full_annot_df['mapped_label'] == 0]
kept_df = full_annot_df[full_annot_df['mapped_label'] != 0]

print(f"\n===== 探针位置预检报告 =====")
print(f"总探针数: {len(full_annot_df)}")
print(f"直接落在脑沟内: {len(kept_df)}")
print(f"落在背景/脑回上: {len(excluded_df)}")

if not excluded_df.empty:
    cols_to_save = ['mni_x', 'mni_y', 'mni_z', 'structure_name', 'well_id', 'donor_id']
    excluded_df[cols_to_save].to_csv(output_excluded_path, index=False)
    print(f"[OK] 被排除的样本清单已保存: {output_excluded_path}")
# 筛选出 mask > 0 的样本


print(f"\n======== 结果分析 (总保留数: {len(kept_df)}) ========")

# A. 打印出现频率最高的 20 个解剖名称
print("\n[Top 20] 保留样本的解剖学名称统计:")
print(kept_df['structure_name'].value_counts().head(20))

# B. 检查含 'fundus' (沟底) 的比例
fundus_count = kept_df['structure_name'].str.contains('fundus', case=False).sum()
sulcus_count = kept_df['structure_name'].str.contains('sulcus', case=False).sum()
gyrus_count = kept_df['structure_name'].str.contains('gyrus', case=False).sum()

print(f"\n[关键词统计]")
print(f"包含 'Fundus' (沟底): {fundus_count} 个")
print(f"包含 'Sulcus' (沟):   {sulcus_count} 个")
print(f"包含 'Gyrus' (脑回):  {gyrus_count} 个 (如果很少，说明过滤成功)")

# C. 保存详细清单供你人工检查
cols_to_save = ['structure_name', 'mni_x', 'mni_y', 'mni_z', 'mapped_label', 'donor_id']
kept_df[cols_to_save].to_csv(output_kept_path, index=False)
print(f"\n[OK] 详细清单已保存至: {output_kept_path}")
print("你可以打开这个表格，看看是不是剩下的全都是 'Fundus' 或者核心沟区域。")
# ================= 3. 生成最终表达矩阵 =================
print("\n正在运行 Abagen 生成最终矩阵...")

# --- 修正点在这里 ---
# 去掉了 ibbf_threshold 参数，以兼容旧版本
expression = abagen.get_expression_data(atlas, atlas_info_path,
                                        tolerance=2, 
                                        donors=donors,
                                        data_dir=data_dir,
                                        probe_selection='diff_stability',
                                        lr_mirror=None)

expression.to_csv(output_expression_path, index=False, header=True)
print(f"[OK] 最终基因表达矩阵已保存: {output_expression_path}")

# files = abagen.fetch_microarray(donors=['9861','10021','12876','14380','15496','15697'], data_dir=r'C:/Users/15542/abagen-data/microarray') 
# atlas_info=r'E:/lsy_group/7.9reorganize/7.9reorganize/5.gene_analysis/atlas_generate/bvatlas_info.csv'
# atlas =  ('E:/lsy_group/7.9reorganize/7.9reorganize/5.gene_analysis/atlas_generate/merged_atlas_relabel.nii')
# atlas = images.check_atlas(atlas)

# expression=abagen.get_expression_data(atlas, atlas_info)
# outputpath=r'E:/lsy_group/7.9reorganize/7.9reorganize/5.gene_analysis/expression_BV__0116.csv'
# expression.to_csv(outputpath,index=False,header=True)
# generator = reporting.Report(atlas, atlas_info)
# report = generator.gen_report()
# print(report)


# txt_dir = 'C:/Users/ASUS/Desktop/gene/features_withexpression' 

# for fname in os.listdir(txt_dir):
#     if fname.lower().endswith('.txt'):
#         fpath = os.path.join(txt_dir, fname)
#         with open(fpath, 'r', encoding='utf-8') as f:
#             lines = f.readlines()

#         # 删除第18和第46行（索引分别为17和45）
#         to_delete = [17, 45]
#         new_lines = [line for idx, line in enumerate(lines) if idx not in to_delete]

#         with open(fpath, 'w', encoding='utf-8') as f:
#             f.writelines(new_lines)

#         print(f"{fname} 已处理")





# # 读取质心坐标
# df = pd.read_excel(r'E:\lsy_group\7.9reorganize\7.9reorganize\5.gene_analysis\1210\expression_BV_0.35.xlsx')
# coords = df[['x', 'y', 'z']].values

# # 计算欧氏距离矩阵
# D = pdist(coords)                      # (1) 得到压缩距离向量
# distance_matrix = squareform(D)        # (2) 转成全矩阵

# # 保存为不带任何标签的纯数字txt
# np.savetxt('distance_matrix_0.35.txt', distance_matrix, fmt='%.6f', delimiter='\t')



# # 1. 路径参数
# features_dir = r'E:\lsy_group\7.9reorganize\7.9reorganize\5.gene_analysis\1210\features_expression'   
# dist_mat_file = 'E:/lsy_group/7.9reorganize/7.9reorganize/5.gene_analysis/distance_matrix.txt'

# # 2. 列出全部txt文件
# feature_files = [f for f in os.listdir(features_dir) if f.endswith('.txt')]

# for fname in feature_files:
#     feature_path = os.path.join(features_dir, fname)
#     base = Base(x=feature_path, D=dist_mat_file)
#     surrogates = base(n=1000)

#     # 去掉后缀作为基名
#     base_name = os.path.splitext(fname)[0]
#     npyfile = f'surrogates_{base_name}.npy'
#     matfile = f'surrogates_{base_name}.mat'

#     # 保存npy和mat
#     np.save(npyfile, surrogates, allow_pickle=True, fix_imports=True)
#     io.savemat(matfile, {'surrogates': surrogates})

#     print(f"[完成] {fname} → {npyfile} , {matfile}")


