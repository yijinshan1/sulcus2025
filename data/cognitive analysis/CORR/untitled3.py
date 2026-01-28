# -*- coding: utf-8 -*-
"""
Created on Tue Jan 27 14:07:07 2026

@author: ASUS
"""

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# 特征及其对应的色系映射
feature_cmaps = {
    'CT': 'Blues',
    'maxD(adj)': 'Greens',
    'meanD(adj)': 'Reds',
    'SA(adj)': 'Purples',
    'SL(adj)': 'Greys',
    'SW': 'Oranges'
}

# 统一量程: 0 到 0.4 (代表偏相关系数的强度)
norm = mcolors.Normalize(vmin=0, vmax=0.4)

# 绘制 Colorbars
fig, axes = plt.subplots(6, 1, figsize=(6, 9))
plt.subplots_adjust(hspace=1.2)

for i, (feat, cmap_name) in enumerate(feature_cmaps.items()):
    sm = plt.cm.ScalarMappable(cmap=cmap_name, norm=norm)
    cb = plt.colorbar(sm, cax=axes[i], orientation='horizontal')
    axes[i].set_title(f'Feature: {feat}', fontsize=12, fontweight='bold')
    cb.set_label('Absolute Partial Correlation |r|', fontsize=10)

plt.savefig('ant_feature_colorbars.svg', dpi=300, bbox_inches='tight')
plt.show()