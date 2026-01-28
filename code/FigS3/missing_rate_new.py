

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from pathlib import Path

# 配置参数
CSV_FOLDER = 'E:\\lsy_group\\7.9reorganize\\7.9reorganize\\6.paper\\CB_revise\\submit\\2.data_process\\sulcal_data_merge'# 替换为你的CSV文件夹路径
OUTPUT_DIR = "E:\\lsy_group\\7.9reorganize\\7.9reorganize\\6.paper\\CB_revise\\submit\\2.data_process\\"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 预期的列名配置（根据实际数据调整）
# EXPECTED_COLUMNS = {
#     "label_column": "Label",  # 标签列的名称
#     "feature_columns": ["SA", "maxD", "meanD", "SL", "SW"]  # 特征列名称
# }

EXPECTED_COLUMNS = {
    "label_column": "Label",  # 标签列的名称
    "feature_columns": ["SA"]  # 特征列名称
}

def load_and_validate_data(folder):
    """加载并验证CSV文件"""
    csv_files = list(Path(folder).glob("*.csv"))
    valid_dfs = []
    problematic_files = []
    
    for file_path in csv_files:
        try:
            # 读取CSV文件
            df = pd.read_csv(file_path)
            
            # 列名校验
            missing_columns = [
                col for col in [EXPECTED_COLUMNS["label_column"]] + EXPECTED_COLUMNS["feature_columns"] 
                if col not in df.columns
            ]
            
            if missing_columns:
                print(f"文件 {file_path.name} 缺少必要列: {missing_columns}")
                problematic_files.append(file_path.name)
                continue
                
            # 添加来源标记
            df["_SourceFile"] = file_path.name
            valid_dfs.append(df)
            
        except Exception as e:
            print(f"读取文件 {file_path.name} 时出错: {str(e)}")
            problematic_files.append(file_path.name)
    

    
    return pd.concat(valid_dfs, ignore_index=True) if valid_dfs else None

def analyze_missing_rates(df):
    """分析缺失率"""
    label_col = EXPECTED_COLUMNS["label_column"]
    features = EXPECTED_COLUMNS["feature_columns"]
    
    # 计算每个标签的出现次数
    total_counts = df.groupby(label_col).size().rename("total_samples")
    
    # 计算缺失数量
    missing_counts = df.groupby(label_col)[features].agg(lambda x: x.isna().sum())
    
    # 计算缺失率
    missing_rates = (missing_counts.T / total_counts).T * 100  # 百分比形式
    missing_rates = missing_rates.reset_index().melt(
        id_vars=label_col, 
        var_name="feature", 
        value_name="missing_rate"
    )
    
    return missing_rates, total_counts

def visualize_sa_missing_rate(missing_rates_df, output_dir):
    
    
    # 1. 筛选SA特征并过滤0缺失率
    sa_df = missing_rates_df[
        (missing_rates_df['feature'] == 'SA') 
    ].copy()


    # 2. 按缺失率降序排序
    sa_df_sorted = sa_df.sort_values(by='missing_rate', ascending=False)
    
    # 设置科研绘图风格
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 8)) # 调整尺寸以适应标签
    
    # 3. 创建条形图
    ax = sns.barplot(
        data=sa_df_sorted,
        x='missing_rate',
        y=EXPECTED_COLUMNS["label_column"],
        color='steelblue', # 使用单一颜色
        orient='h'
    )
    
    # *** 新增代码：在25%位置添加灰色虚线 ***
    ax.axvline(x=25, color='gray', linestyle='--', linewidth=1)
    
    # 设置标题和标签
    plt.title('Missing Rate of SA Feature by Label', fontsize=16, weight='bold')
    plt.xlabel('Missing Rate (%)', fontsize=12)
    plt.ylabel('Label', fontsize=12)
    
    # 优化刻度
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    # 在每个条形上显示数值
    for p in ax.patches:
        width = p.get_width()
        plt.text(width + 0.3, p.get_y() + p.get_height()/2.,
                 f'{width:.2f}%',
                 va='center', fontsize=9)
    
    # 调整布局并移除右和上边框
    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    
    # 4. 保存为PDF
    output_path = Path(output_dir) / "missing_rate.pdf"
    plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"SA特征缺失率图表已保存至: {output_path}")


def main():
    print("开始数据分析...")
    
    # 加载数据
    combined_data = load_and_validate_data(CSV_FOLDER)
    
    if combined_data is None:
        print("错误：没有有效数据可供分析")
        return
    
    # 执行分析
    missing_rates_df, total_counts_df = analyze_missing_rates(combined_data)
    
    # 保存原始分析结果
    missing_rates_df.to_csv(Path(OUTPUT_DIR)/"missing_rates_all_features.csv", index=False)
    
    # (新) 生成SA特征的科研图表
    visualize_sa_missing_rate(missing_rates_df, OUTPUT_DIR)
    
    print(f"分析完成！结果已保存至: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
