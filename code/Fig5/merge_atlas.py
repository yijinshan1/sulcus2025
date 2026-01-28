# -*- coding: utf-8 -*-
"""
Created on Mon Jul 14 15:48:41 2025

@author: ASUS
"""

import nibabel as nib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 加载数据
nii_path = r'E:/lsy_group/7.9reorganize/7.9reorganize/5.gene_analysis/atlas_generate/anat_landmark_atlas_hardlabel_thres0.5.nii'
img = nib.load(nii_path)
label_data = img.get_fdata().astype(int)

stats_df = pd.read_excel('anat_landmark_atlas_stats.xlsx')
merge_df = pd.read_excel('merge.xlsx')

label_col = 'Label'
r_col, g_col, b_col = 'r', 'g', 'b'
acronym_col = 'brainvisa_acronym'

# 1. 只保留RGB非全0的分区
rgb_mask = ~((stats_df[r_col]==0) & (stats_df[g_col]==0) & (stats_df[b_col]==0))
stats_valid = stats_df[rgb_mask].copy()

# 1.5 手动删除指定脑沟
remove_acronyms = {'F.I.P.r.int.2_right', 'S.Pa.t._left', 'S.F.orbitaire._right'}
stats_valid = stats_valid[~stats_valid[acronym_col].isin(remove_acronyms)]
acronyms_valid = set(stats_valid[acronym_col])
label2acronym = dict(zip(stats_valid[label_col], stats_valid[acronym_col]))
acronym2label = dict(zip(stats_valid[acronym_col], stats_valid[label_col]))

# [重点] 从nii空间把这3个acronym对应label全置0！
remove_labels = [acronym2label[a] for a in remove_acronyms if a in acronym2label]
mask_valid = np.isin(label_data, list(label2acronym.keys()))
label_data_filtered = label_data.copy()
label_data_filtered[~mask_valid] = 0
for l in remove_labels:
    label_data_filtered[label_data_filtered == l] = 0

# 3. 新建分区（顺序int编号，每行一个acronym新分区名，对应合并哪些老acronym）
acronym2newint = {}
merge_labels = []
new_label_img = np.zeros_like(label_data, dtype=np.int16)
new_stats = []
for idx, row in merge_df.iterrows():
    new_acronym = row['Label']
    parts = [x.strip() for x in str(row['merge']).split('+')]
    # 只取有效part（滤除因被删除的脑区acronym）
    valid_parts = [p for p in parts if p in acronyms_valid]
    if not valid_parts:
        continue
    new_idx = idx+1  # 新分区编码（顺序分配，1开始，暂时）
    acronym2newint[new_acronym] = new_idx
    # 将这些part对应nii标签所有位置设置为新分区号
    part_label_ids = [acronym2label[p] for p in valid_parts]
    mask = np.isin(label_data_filtered, part_label_ids)
    new_label_img[mask] = new_idx
    
    # 新rgb就用第一个脑区
    first_part = valid_parts[0]
    rgb_row = stats_valid[stats_valid[acronym_col]==first_part].iloc[0]
    r,g,b = rgb_row[r_col], rgb_row[g_col], rgb_row[b_col]
    vox_ind = np.argwhere(mask)
    count = int(mask.sum())
    if count == 0:
        x,y,z = np.nan, np.nan, np.nan
    else:
        x, y, z = vox_ind.mean(axis=0)
    new_stats.append({
        'Label': new_idx,
        'Name': new_acronym,
        'MergedFrom': '+'.join(valid_parts),
        'r': r, 'g': g, 'b': b,
        'VoxelCount': count,
        'x': x, 'y': y, 'z': z
    })
    merge_labels.append(new_acronym)

new_stats_df = pd.DataFrame(new_stats)

# ===【插入：连续新编号重排映射】===
unique_labels = sorted([int(l) for l in np.unique(new_label_img) if l != 0])
old2new_label = {old: newidx+1 for newidx, old in enumerate(unique_labels)}
# relabel nii
new_label_img_relabel = np.zeros_like(new_label_img)
for old, new in old2new_label.items():
    new_label_img_relabel[new_label_img == old] = new
# relabel 统计表
new_stats_df['Label_raw'] = new_stats_df['Label']
new_stats_df['Label'] = new_stats_df['Label'].map(old2new_label)

# 保存新nii和统计
new_nii = nib.Nifti1Image(new_label_img_relabel, affine=img.affine, header=img.header)
nib.save(new_nii, 'merged_atlas_relabel_0.5.nii.gz')
new_stats_df.to_excel('merged_atlas_stats_relabel_0.5.xlsx', index=False)
print("新nii已保存为 merged_atlas_relabel_0.8.nii.gz, 连续编号统计表保存为 merged_atlas_stats_relabel_0.5.xlsx")

# 可视化
color_dict = {row['Label']: [row['r'], row['g'], row['b']] for _, row in new_stats_df.iterrows()}
name_dict = {row['Label']: row['Name'] for _, row in new_stats_df.iterrows()}
labels_vis = np.unique(new_label_img_relabel)
rgb_img = np.zeros(new_label_img_relabel.shape + (3,), dtype=np.float32)
for label in labels_vis:
    if label == 0: continue
    rgb = color_dict.get(label, [1,0,0])
    mask = (new_label_img_relabel == label)
    for c in range(3): rgb_img[...,c][mask]=rgb[c]

mid_x, mid_y, mid_z = [d//2 for d in new_label_img_relabel.shape]
slices = [
    ('矢状面', rgb_img[mid_x, :, :, :], new_label_img_relabel[mid_x, :, :], 0),
    ('冠状面', rgb_img[:, mid_y, :, :], new_label_img_relabel[:, mid_y, :], 1),
    ('轴状面', rgb_img[:, :, mid_z, :], new_label_img_relabel[:, :, mid_z], 2)
]
fig, axes = plt.subplots(1, 3, figsize=(18, 7), dpi=150)
min_area = 20
for i, (title, img_slice, label_slice, typ) in enumerate(slices):
    ax = axes[i]
    ax.imshow(img_slice)
    ax.set_title(title, fontsize=14)
    ax.axis('off')
    for label in np.unique(label_slice):
        if label==0: continue
        mask = (label_slice==label)
        area = mask.sum()
        if area<min_area: continue
        coords = np.argwhere(mask)
        yx = coords.mean(axis=0)
        if typ==0:
            y,z= yx; pos=(z,y)
        elif typ==1:
            x,z= yx; pos=(z,x)
        else:
            x,y0= yx; pos=(y0,x)
        name = name_dict.get(label, f'L{label}')
        ax.text(*pos, name, fontsize=9, color='w', ha='center', va='center',
            bbox=dict(facecolor='black', alpha=0.45, lw=0, pad=1.2))
fig.tight_layout()
plt.savefig('merged_atlas_slices_relabel_0.5.pdf', bbox_inches='tight', dpi=200)
plt.show()