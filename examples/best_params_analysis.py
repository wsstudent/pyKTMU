import os
import itertools
import argparse
import csv
import subprocess
import re

# ==============================================================================
#                                  配置区
# ==============================================================================
# --- 实验参数 ---
MODELS = ["dkt", "dkvmn", "sakt", "dkt+"]
DATASETS = ["assist2009", "assist2017", "nips_task34"]
STRATEGIES = [ "low_performance", "high_performance", "low_engagement"]
RATIOS = [0.2, 0.4, 0.8]
ALPHAS = [1.0, 5.0, 10.0, 20.0, 50.0, 100.0]

# --- 自动重试参数 ---
BATCH_SIZES_TO_TRY = [None, 256, 128, 64, 32]
MEMORY_ERROR_KEYWORDS = [
    "cuda out of memory",  # PyTorch 标准 OOM
    "out of memory",       # 通用 OOM
    "nvml_success",        # 您遇到的 NVML/CUDACachingAllocator 错误
    "cudacachingallocator.cpp" # 同上，增加一个特征词
]

# --- 路径定义 ---
# 预训练模型所在的父目录
PRETRAINED_MODEL_PARENT_DIR = "saved_model/standard_training"
# 遗忘实验结果保存的父目录
PARENT_SAVE_DIR = "saved_model/unlearning_runs"
# 评估结果CSV文件的保存路径
RESULTS_CSV_PATH = "../data/evaluation_results.csv"

# --- 预训练模型检查点路径映射 ---
CKPT_MAP = {
    ("dkt", "assist2009"): "dkt_assist2009_seed42_fold0_412eb83f",
    ("dkt", "assist2017"): "dkt_assist2017_seed42_fold0_9decca36",
    ("dkt", "nips_task34"): "dkt_nips_task34_seed42_fold0_0a68a45b",
    ("dkt+", "assist2009"): "dkt+_assist2009_seed42_fold0_8692c728",
    ("dkt+", "assist2017"): "dkt+_assist2017_seed42_fold0_a54da986",
    ("dkt+", "nips_task34"): "dkt+_nips_task34_seed42_fold0_4b2cba7f",
    ("dkvmn", "assist2009"): "dkvmn_assist2009_seed42_fold0_38beccef",
    ("dkvmn", "assist2017"): "dkvmn_assist2017_seed42_fold0_ebee298a",
    ("dkvmn", "nips_task34"): "dkvmn_nips_task34_seed42_fold0_c50f8c31",
    ("sakt", "assist2009"): "sakt_assist2009_seed42_fold0_3a7ced70",
    ("sakt", "assist2017"): "sakt_assist2017_seed42_fold0_fbba0205",
    ("sakt", "nips_task34"): "sakt_nips_task34_seed42_fold0_5f025f8d",
}
# ==============================================================================


# ==============================================================================
#                                  辅助函数
# ==============================================================================
def run_command_with_retry(base_command, batch_sizes):
    """
    执行一个训练命令。仅当遇到内存相关错误时，才用更小的 batch_size 自动重试。
    对于其他任何情况（成功或非内存错误），则直接“放行”，并停止重试。

    :param base_command: str, 不包含 --batch_size 参数的基础命令字符串。
    :param batch_sizes: list, 一个包含要尝试的 batch_size 的列表，例如 [256, 128, 64]。
    :return: bool, 包装脚本的任务是否完成 (True) 或因内存耗尽而彻底失败 (False)。
    """
    for bs in batch_sizes:
        # 2. 构建当前要执行的完整命令
        command = base_command
        if bs is not None:
            command += f" --batch_size {bs}"

        current_bs_str = "默认值" if bs is None else str(bs)
        print(f"🚀 正在尝试使用 batch_size: {current_bs_str}")
        print(f"   命令: {command}")

        # 3. 执行命令
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True, encoding="utf-8", errors="ignore"
        )

        # 4. 分析错误输出，判断是否为内存错误
        stderr_lower = result.stderr.lower()
        is_memory_error = any(keyword in stderr_lower for keyword in MEMORY_ERROR_KEYWORDS)

        # 5. 核心判断逻辑
        #   仅当【确实发生了错误(returncode!=0)】且【是内存相关错误】时，才重试
        if result.returncode != 0 and is_memory_error:
            print(f"🟡 检测到内存相关错误 with batch_size: {current_bs_str}。准备重试...")
            # 让循环继续，尝试下一个更小的 batch_size
            continue
        else:
            # 对于任何其他情况 (成功 或 非内存错误)，我们都“放行”
            print(f"✅ 任务执行完毕或遇到非内存错误，按要求放行。")
            if result.returncode == 0:
                print(f"   状态: 执行成功 (返回码: 0)。")
            else:
                print(f"   状态: 执行时发生非内存错误 (返回码: {result.returncode})。")
            
            # 打印最终的输出，供用户自己判断
            print("------ Begin Stderr (如有) ------")
            print(result.stderr)
            print("------- End Stderr -------")
            return True  # 返回 True，表示“哨兵”任务完成，不再干预

    # 6. 如果所有 batch_size 都因内存错误而失败
    print(f"❌ 任务失败。已尝试所有指定的 batch_size，但均因内存不足而失败: {batch_sizes}")
    return False



def run_simple_command(command):
    """
    为评估过程设计的简单命令执行函数。
    """
    print(f"🚀 Executing: {command}")
    return_code = os.system(command)
    if return_code != 0:
        print(f"❌ Error: Command failed with exit code {return_code}. Halting script.")
        exit(1)


# ==============================================================================


# ==============================================================================
#                                  训练函数
# ==============================================================================
def run_unlearning_experiments():
    """执行训练任务，带断点续跑和自动batch_size调整功能"""
    print("===== 🚀 开始执行遗忘训练任务 (支持断点续跑和OOM重试) 🚀 =====")
    os.makedirs(PARENT_SAVE_DIR, exist_ok=True)

    combinations = list(itertools.product(MODELS, DATASETS, STRATEGIES, RATIOS, ALPHAS))

    for i, (model, dataset, strategy, ratio, alpha) in enumerate(combinations):
        print("-" * 80)
        print(
            f"🔄 检查训练任务: {i + 1}/{len(combinations)} -> M:{model}, D:{dataset}, S:{strategy}, R:{ratio}, A:{alpha}"
        )

        # --- 断点续跑逻辑 ---
        expected_prefix = (
            f"surgical_{model}_{dataset}_{strategy}_ratio{ratio}_alpha{alpha}"
        )
        try:
            all_dirs = [
                d
                for d in os.listdir(PARENT_SAVE_DIR)
                if os.path.isdir(os.path.join(PARENT_SAVE_DIR, d))
            ]
            matches = [d for d in all_dirs if d.startswith(expected_prefix)]
            if len(matches) > 0:
                print(f"✅ 跳过: 已找到输出文件夹 {matches[0]}。")
                continue
        except FileNotFoundError:
            pass

        # --- 执行命令逻辑 ---
        train_script = f"wandb_{model}_train.py"
        model_ckpt_key = (model, dataset)
        if model_ckpt_key not in CKPT_MAP:
            print(
                f"⚠️  警告: 未找到模型 {model} 在数据集 {dataset} 上的检查点路径，跳过..."
            )
            continue
        model_ckpt_folder = CKPT_MAP[model_ckpt_key]

        base_command = (
            f"python {train_script} --dataset_name {dataset} --unlearn_method surgical "
            f"--model_ckpt_path {PRETRAINED_MODEL_PARENT_DIR}/{model_ckpt_folder} "
            f"--alpha {alpha} --unlearn_strategy {strategy} --forget_ratio {ratio} "
            f"--save_dir {PARENT_SAVE_DIR} --use_wandb 0"
        )

        run_command_with_retry(base_command, BATCH_SIZES_TO_TRY)

    print("✅ 所有遗忘训练任务已完成！")


# ==============================================================================


# ==============================================================================
#                                  评估函数
# ==============================================================================
def run_evaluation():
    """执行评估任务，带断点续跑功能"""
    print("===== 📊 开始执行评估任务 (支持断点续跑) 📊 =====")

    # --- 断点续跑逻辑 ---
    completed_evals = set()
    try:
        # 确保CSV文件的父目录存在
        os.makedirs(os.path.dirname(RESULTS_CSV_PATH), exist_ok=True)
        with open(RESULTS_CSV_PATH, "r", newline="", encoding="utf-8-sig") as csvfile:
            reader = csv.reader(csvfile)
            header = next(reader)
            # 根据您CSV的列名来确定索引
            model_path_idx = header.index("模型")
            test_type_idx = header.index("测试集类型")

            for row in reader:
                if row:  # 避免空行
                    completed_evals.add((row[model_path_idx], row[test_type_idx]))
        print(
            f"已从 {RESULTS_CSV_PATH} 加载 {len(completed_evals)} 条已完成的评估记录。"
        )
    except (FileNotFoundError, StopIteration):
        print("未找到现有结果文件或文件为空，将从头开始评估。")
    except ValueError as e:
        print(f"CSV文件表头错误，请检查列名是否包含'模型'和'测试集类型'。错误: {e}")

    combinations = list(itertools.product(MODELS, DATASETS, STRATEGIES, RATIOS, ALPHAS))

    for i, (model, dataset, strategy, ratio, alpha) in enumerate(combinations):
        expected_prefix = (
            f"surgical_{model}_{dataset}_{strategy}_ratio{ratio}_alpha{alpha}"
        )
        try:
            all_dirs = [
                d
                for d in os.listdir(PARENT_SAVE_DIR)
                if os.path.isdir(os.path.join(PARENT_SAVE_DIR, d))
            ]
        except FileNotFoundError:
            print(f"❌ 错误: 训练输出目录 {PARENT_SAVE_DIR} 不存在。请先运行训练。")
            break

        matches = [d for d in all_dirs if d.startswith(expected_prefix)]
        if len(matches) != 1:
            continue

        eval_save_dir = os.path.join(PARENT_SAVE_DIR, matches[0])

        for test_file_type in ["forget", "retain"]:
            print("-" * 80)
            print(f"🔄 检查评估任务: {eval_save_dir} on {test_file_type} SET")

            # --- 断点续跑检查 ---
            if (eval_save_dir, test_file_type) in completed_evals:
                print(f"✅ 跳过: 在CSV中已找到该评估记录。")
                continue

            # --- 执行命令 ---
            command = (
                f"python wandb_predict.py "  # 假设您的评估脚本名为 wandb_predict.py
                f"--save_dir {eval_save_dir} --unlearn_strategy {strategy} "
                f"--forget_ratio {ratio} --unlearn_test_file {test_file_type} "
                f"--use_wandb 0"
            )

            run_simple_command(command)

    print(f"✅ 所有评估调用已完成！")


# ==============================================================================


# ==============================================================================
#                                  主程序入口
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="模型遗忘和评估自动化调度器 (最终版)")
    parser.add_argument(
        "action",
        choices=["train", "eval", "all"],
        help="选择要执行的操作: 'train', 'eval', 'all'",
    )
    args = parser.parse_args()

    if args.action == "train":
        run_unlearning_experiments()
    elif args.action == "eval":
        run_evaluation()
    elif args.action == "all":
        run_unlearning_experiments()
        run_evaluation()
# ==============================================================================
