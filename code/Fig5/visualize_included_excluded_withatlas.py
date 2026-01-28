# -*- coding: utf-8 -*-
"""
Analysis Script: Corrected Aspect Ratio 3D Brain Visualization
Language: English
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from nilearn import datasets, surface
import nibabel as nib
from scipy.spatial import cKDTree

# ==========================================
# 1. Configuration & Paths
# ==========================================
atlas_nii_path = r"E:\lsy_group\7.9reorganize\7.9reorganize\5.gene_analysis\atlas_generate\merged_atlas_relabel.nii" 
included_csv = r"E:\lsy_group\7.9reorganize\7.9reorganize\5.gene_analysis\included_samples_report_0116.csv"
excluded_csv = r"E:\lsy_group\7.9reorganize\7.9reorganize\5.gene_analysis\excluded_samples_report_0116.csv"
atlas_excel = r"E:\lsy_group\7.9reorganize\7.9reorganize\5.gene_analysis\atlas_generate\merged_atlas_stats_relabel.xlsx"

# ==========================================
# 2. Logic: Atlas Color Lookup
# ==========================================
print("Initializing Atlas...")

def build_atlas_lookup_tree(nii_path, excel_path):
    df_color = pd.read_excel(excel_path)
    df_color.columns = [c.lower().strip() for c in df_color.columns]
    max_val = df_color[['r', 'g', 'b']].max().max()
    scale = 255.0 if max_val > 1.0 else 1.0
    color_dict = {}
    for _, row in df_color.iterrows():
        color_dict[int(row['label'])] = [row['r']/scale, row['g']/scale, row['b']/scale]
        
    img = nib.load(nii_path)
    data = img.get_fdata()
    affine = img.affine
    voxel_indices = np.argwhere(data > 0)
    labels = data[voxel_indices[:,0], voxel_indices[:,1], voxel_indices[:,2]]
    real_coords = nib.affines.apply_affine(affine, voxel_indices)
    
    return cKDTree(real_coords), labels, color_dict

atlas_tree, atlas_labels, color_map = build_atlas_lookup_tree(atlas_nii_path, atlas_excel)

def get_probe_colors(coords):
    if len(coords) == 0: return []
    _, indices = atlas_tree.query(coords, k=1)
    final_colors = []
    for idx in indices:
        target_label = int(atlas_labels[idx])
        final_colors.append(color_map.get(target_label, [0.5, 0.5, 0.5]))
    return np.array(final_colors)

# ==========================================
# 3. Data Loading
# ==========================================
print("Loading Coordinates...")

def get_coords(df):
    cols = [c for c in df.columns if 'mni' in c.lower() or c.lower() in ['x', 'y', 'z']]
    x_col = next((c for c in cols if 'x' in c.lower()), None)
    y_col = next((c for c in cols if 'y' in c.lower()), None)
    z_col = next((c for c in cols if 'z' in c.lower()), None)
    if x_col and y_col and z_col: return df[[x_col, y_col, z_col]].values
    return np.empty((0, 3))

try:
    df_inc = pd.read_csv(included_csv)
    df_exc = pd.read_csv(excluded_csv)
    coords_inc = get_coords(df_inc)
    coords_exc = get_coords(df_exc)
except Exception:
    coords_inc = np.empty((0, 3))
    coords_exc = np.empty((0, 3))

# ==========================================
# 4. Visualization Logic
# ==========================================
fsaverage = datasets.fetch_surf_fsaverage('fsaverage')

def project_points_to_surface(coords, mesh_pial, mesh_inflated):
    if len(coords) == 0: return coords
    pial_coords, _ = surface.load_surf_mesh(mesh_pial)
    inf_coords, _ = surface.load_surf_mesh(mesh_inflated)
    tree = cKDTree(pial_coords)
    _, indices = tree.query(coords)
    return inf_coords[indices]

def plot_transparent_hemisphere(hemi='left'):
    print(f"Plotting {hemi} hemisphere...")
    
    if hemi == 'left':
        surf_inf = fsaverage.infl_left
        surf_pial = fsaverage.pial_left
        mask_inc = coords_inc[:, 0] < 0
        mask_exc = coords_exc[:, 0] < 0
    else:
        surf_inf = fsaverage.infl_right
        surf_pial = fsaverage.pial_right
        mask_inc = coords_inc[:, 0] > 0
        mask_exc = coords_exc[:, 0] > 0
        
    pts_inc = coords_inc[mask_inc]
    pts_exc = coords_exc[mask_exc]
    
    colors_inc = get_probe_colors(pts_inc)
    proj_inc = project_points_to_surface(pts_inc, surf_pial, surf_inf)
    proj_exc = project_points_to_surface(pts_exc, surf_pial, surf_inf)

    # Load mesh data once to calculate limits
    mesh_coords, mesh_faces = surface.load_surf_mesh(surf_inf)
    
    # --- ASPECT RATIO CALCULATION START ---
    # Calculate the center and max range to force a cubic bounding box
    x_limits = [mesh_coords[:, 0].min(), mesh_coords[:, 0].max()]
    y_limits = [mesh_coords[:, 1].min(), mesh_coords[:, 1].max()]
    z_limits = [mesh_coords[:, 2].min(), mesh_coords[:, 2].max()]
    
    x_range = x_limits[1] - x_limits[0]
    y_range = y_limits[1] - y_limits[0]
    z_range = z_limits[1] - z_limits[0]
    
    max_range = max(x_range, y_range, z_range)
    
    mid_x = np.mean(x_limits)
    mid_y = np.mean(y_limits)
    mid_z = np.mean(z_limits)
    # --- ASPECT RATIO CALCULATION END ---

    fig = plt.figure(figsize=(18, 6))
    views = ['lateral', 'medial', 'dorsal']
    titles = ['Lateral', 'Medial', 'Dorsal']
    
    for i, view in enumerate(views):
        ax = fig.add_subplot(1, 3, i+1, projection='3d')
        
        # Plot Brain Mesh
        ax.plot_trisurf(
            mesh_coords[:, 0], 
            mesh_coords[:, 1], 
            mesh_coords[:, 2], 
            triangles=mesh_faces,
            color='whitesmoke',
            alpha=0.12,         
            edgecolor='none',
            shade=True
        )
        
        # Plot Probes
        if len(proj_exc) > 0:
            ax.scatter(
                proj_exc[:, 0], proj_exc[:, 1], proj_exc[:, 2],
                c='silver', s=15, alpha=0.3, 
                depthshade=False
            )
            
        if len(proj_inc) > 0:
            ax.scatter(
                proj_inc[:, 0], proj_inc[:, 1], proj_inc[:, 2],
                c=colors_inc, s=40, alpha=1.0,
                edgecolors='black', linewidth=0.5,
                depthshade=False
            )
            
        # --- APPLY ASPECT RATIO FIX ---
        # 1. Set the limits to be a perfect cube centered on the brain
        ax.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
        ax.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
        ax.set_zlim(mid_z - max_range/2, mid_z + max_range/2)
        
        # 2. Force matplotlib to respect these dimensions equally
        ax.set_box_aspect([1, 1, 1]) 
        # -----------------------------

        # Camera Angles
        if view == 'lateral':
            if hemi == 'left': ax.view_init(elev=0, azim=180)
            else: ax.view_init(elev=0, azim=0)
        elif view == 'medial':
            if hemi == 'left': ax.view_init(elev=0, azim=0)
            else: ax.view_init(elev=0, azim=180)
        elif view == 'dorsal':
            ax.view_init(elev=90, azim=90)
            
        ax.set_title(titles[i], fontsize=14)
        ax.set_axis_off()

    # Legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Excluded',
               markerfacecolor='silver', markersize=8, markeredgecolor='gray', alpha=0.5),
        Line2D([0], [0], marker='o', color='w', label='Included',
               markerfacecolor='black', markersize=10, markeredgecolor='black'), 
    ]
    fig.legend(handles=legend_elements, loc='lower left', bbox_to_anchor=(0.05, 0.1))
    fig.suptitle(f'{hemi.capitalize()} Hemisphere Probes (Corrected Aspect)', fontsize=16)
    
    return fig

# ==========================================
# 5. Execution
# ==========================================

fig_left = plot_transparent_hemisphere('left')
fig_left.savefig('brain_vis_fixed_left.png', dpi=300, bbox_inches='tight', transparent=True)
print("Saved left.")

fig_right = plot_transparent_hemisphere('right')
fig_right.savefig('brain_vis_fixed_right.png', dpi=300, bbox_inches='tight', transparent=True)
print("Saved right.")

plt.show()