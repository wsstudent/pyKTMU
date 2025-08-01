import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re

# ==============================================================================
#                                  配置与加载
# ==============================================================================
# 定义输入和输出文件的路径
INPUT_CSV_PATH = "./data/evaluation_results.csv"
PROCESSED_CSV_PATH = "processed_results_summary_acc.csv" # 新的、处理后的结果文件名
PLOTS_DIR = "analysis_plots_acc" # 保存所有图表的文件夹名

# --- 第1步：加载数据 ---
try:
    df = pd.read_csv(INPUT_CSV_PATH)
    print("原始数据加载成功。")
except FileNotFoundError:
    print(f"错误：找不到 {INPUT_CSV_PATH} 文件。请确保路径正确。")
    exit()


# ==============================================================================
#                                  数据清洗与重塑
# ==============================================================================
# --- 第2步：数据清洗 - 从路径中提取关键信息 ---
def extract_model_name(path_string):
    if '_dkt_' in path_string: return 'dkt'
    if '_dkvmn_' in path_string: return 'dkvmn'
    if '_sakt_' in path_string: return 'sakt'
    return 'unknown'

def extract_alpha_value(path_string):
    match = re.search(r'_alpha(\d+\.?\d*)', path_string)
    return float(match.group(1)) if match else None

df['Model_Name'] = df['模型'].apply(extract_model_name)
df['alpha'] = df['模型'].apply(extract_alpha_value)
print("数据清洗完成，已提取'Model_Name'和'alpha'。")

# --- 第3步：数据重塑 (Pivot) ---
pivoted_df = df.pivot_table(
    index=['Model_Name', '数据集', '遗忘策略', '遗忘比例', 'alpha'],
    columns='测试集类型',
    values='acc'
).reset_index()

pivoted_df.rename(columns={
    'forget': 'Forget_Set_ACC', 'retain': 'Retain_Set_ACC',
    '数据集': 'Dataset', '遗忘策略': 'Strategy', '遗忘比例': 'Ratio'
}, inplace=True)
print("数据重塑完成。")


# --- ★ 新增：第4步 - 保存处理后的数据到新的CSV文件 ★ ---
pivoted_df.to_csv(PROCESSED_CSV_PATH, index=False, encoding='utf-8-sig')
print(f"\n处理后的规整数据已保存至: {PROCESSED_CSV_PATH}\n")


# ==============================================================================
#                                  循环分析与可视化
# ==============================================================================
# --- 第5步：循环分析并保存图表 ---
# 创建用于保存图表的文件夹
os.makedirs(PLOTS_DIR, exist_ok=True)
print(f"开始生成分析图表，将保存至 ./{PLOTS_DIR}/ 文件夹...")

group_keys = ['Model_Name', 'Dataset', 'Strategy', 'Ratio']

for name, group in pivoted_df.groupby(group_keys):
    print("-" * 60)
    scenario_name_str = f"Model_{name[0]}_Dataset_{name[1]}_Strategy_{name[2]}_Ratio_{name[3]}"
    print(f"正在处理场景: {scenario_name_str}")

    group = group.sort_values(by='alpha').reset_index(drop=True)
    
    # --- 可视化 ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(group['alpha'], group['Forget_Set_ACC'], marker='o', linestyle='--', label='Forget Set ACC (lower is better)')
    ax.plot(group['alpha'], group['Retain_Set_ACC'], marker='s', linestyle='-', label='Retain Set ACC (higher is better)')

    title_str = f"Model: {name[0]}, Dataset: {name[1]}\nStrategy: {name[2]}, Ratio: {name[3]}"
    ax.set_title(title_str, fontsize=14)
    ax.set_xlabel('Alpha (α) Value')
    ax.set_ylabel('ACC Score')
    ax.set_xscale('log')
    ax.legend()
    plt.tight_layout()
    
    # ★ 改动：将图表保存为图片文件，而不是尝试显示它 ★
    plot_filename = os.path.join(PLOTS_DIR, f"{scenario_name_str}.png")
    plt.savefig(plot_filename)
    plt.close(fig) # 关闭图表以释放内存
    print(f"✅ 图表已保存: {plot_filename}")

print("\n所有分析和可视化已完成！")