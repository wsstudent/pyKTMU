import os
import itertools
import argparse

# --- 1. 实验配置 ---
MODELS = ["dkt", "dkvmn", "sakt"]
DATASETS = ["assist2009", "assist2017", "nips_task34"]
STRATEGIES = ["random", "low_performance", "high_performance"]
RATIOS = [0.2, 0.4, 0.8]
ALPHAS = [1.0, 5.0, 10.0, 20.0, 50.0, 100.0]

# --- 2. 预训练模型检查点路径映射 ---
CKPT_MAP = {
    ("dkt", "assist2009"): "dkt_assist2009_seed42_fold0_412eb83f",
    ("dkt", "assist2017"): "dkt_assist2017_seed42_fold0_9decca36",
    ("dkt", "nips_task34"): "dkt_nips_task34_seed42_fold0_0a68a45b",
    ("dkvmn", "assist2009"): "dkvmn_assist2009_seed42_fold0_38beccef",
    ("dkvmn", "assist2017"): "dkvmn_assist2017_seed42_fold0_ebee298a",
    ("dkvmn", "nips_task34"): "dkvmn_nips_task34_seed42_fold0_c50f8c31",
    ("sakt", "assist2009"): "sakt_assist2009_seed42_fold0_3a7ced70",
    ("sakt", "assist2017"): "sakt_assist2017_seed42_fold0_fbba0205d",
    ("sakt", "nips_task34"): "sakt_nips_task34_seed42_fold0_5f025f8d",
}

# --- 3. 定义统一的父目录 ---
# 所有遗忘模型都将保存在这个目录下，每个任务一个子文件夹
PARENT_SAVE_DIR = "saved_model/unlearning_runs"


def run_command(command):
    """一个辅助函数，用于打印并执行系统命令，并在出错时停止脚本"""
    print(f"🚀 Executing: {command}")
    return_code = os.system(command)
    if return_code != 0:
        print(f"❌ Error: Command failed with exit code {return_code}. Halting script.")
        exit(1)


def run_unlearning_experiments():
    """执行训练任务"""
    print("===== 🚀 开始执行遗忘训练任务 (支持断点续跑) 🚀 =====")
    os.makedirs(PARENT_SAVE_DIR, exist_ok=True)

    combinations = list(itertools.product(MODELS, DATASETS, STRATEGIES, RATIOS, ALPHAS))

    for i, (model, dataset, strategy, ratio, alpha) in enumerate(combinations):
        print("-" * 80)
        print(
            f"🔄 检查任务: {i + 1}/{len(combinations)} -> M:{model}, D:{dataset}, S:{strategy}, R:{ratio}, A:{alpha}"
        )

        # ★ 新增：断点续跑逻辑 (训练) ★
        # 1. 检查预期的输出文件夹是否已存在
        expected_prefix = (
            f"surgical_{model}_{dataset}_{strategy}_ratio{ratio}_alpha{alpha}"
        )
        try:
            # 列出父目录下的所有文件夹
            all_dirs = [
                d
                for d in os.listdir(PARENT_SAVE_DIR)
                if os.path.isdir(os.path.join(PARENT_SAVE_DIR, d))
            ]
            # 查找匹配前缀的文件夹
            matches = [d for d in all_dirs if d.startswith(expected_prefix)]
            if len(matches) > 0:
                print(f"✅ 跳过: 已找到输出文件夹 {matches[0]}。")
                continue  # 如果已存在，直接跳到下一个循环
        except FileNotFoundError:
            # 如果父目录不存在，说明是第一次运行，正常继续
            pass

        # 2. 如果文件夹不存在，则执行训练命令
        train_script = f"wandb_{model}_train.py"
        model_ckpt_key = (model, dataset)
        if model_ckpt_key not in CKPT_MAP:
            print(
                f"⚠️  警告: 未找到模型 {model} 在数据集 {dataset} 上的检查点路径，跳过..."
            )
            continue
        model_ckpt_folder = CKPT_MAP[model_ckpt_key]

        command = (
            f"python {train_script} --dataset_name {dataset} --unlearn_method surgical "
            f"--model_ckpt_path saved_model/standard_training/{model_ckpt_folder} "
            f"--alpha {alpha} --unlearn_strategy {strategy} --forget_ratio {ratio} "
            f"--save_dir {PARENT_SAVE_DIR} --use_wandb 0"
        )
        run_command(command)

    print("✅ 所有遗忘训练任务已完成！")


def run_evaluation():
    """对所有已训练的遗忘模型进行评估 (智能搜索版)"""
    print("===== 📊 开始执行评估任务 📊 =====")

    if not os.path.isdir(PARENT_SAVE_DIR):
        print(f"❌ 错误: 父目录 {PARENT_SAVE_DIR} 不存在, 请先运行训练。")
        return

    combinations = list(itertools.product(MODELS, DATASETS, STRATEGIES, RATIOS, ALPHAS))

    for i, (model, dataset, strategy, ratio, alpha) in enumerate(combinations):
        # 1. 构建预期的目录前缀，确保与训练脚本的命名规则一致
        expected_prefix = (
            f"surgical_{model}_{dataset}_{strategy}_ratio{ratio}_alpha{alpha}"
        )

        # 2. 在父目录中搜索所有文件夹
        try:
            all_dirs = [
                d
                for d in os.listdir(PARENT_SAVE_DIR)
                if os.path.isdir(os.path.join(PARENT_SAVE_DIR, d))
            ]
        except FileNotFoundError:
            print(f"❌ 错误: 无法访问目录 {PARENT_SAVE_DIR}。")
            break

        # 3. 找到匹配的文件夹
        matches = [d for d in all_dirs if d.startswith(expected_prefix)]

        if len(matches) != 1:
            # 如果没有找到或找到多个，打印警告并跳过
            if len(matches) > 1:
                print(
                    f"⚠️ 警告: 找到多个匹配 '{expected_prefix}' 的目录: {matches}。请检查命名规则。跳过此项评估。"
                )
            continue

        # 成功找到唯一的目录
        eval_save_dir = os.path.join(PARENT_SAVE_DIR, matches[0])

        print("-" * 80)
        print(f"🔄 进度: {i + 1}/{len(combinations)}")
        print(f"✅ 找到评估目录: {eval_save_dir}")
        print("-" * 80)

        # 4. 对 "forget" 和 "retain" 集合分别进行评估
        for test_file_type in ["forget", "retain"]:
            print(f"  - 正在评估 {test_file_type.upper()} SET...")
            command = (
                f"python wandb_predict.py "
                f"--save_dir {eval_save_dir} "
                f"--unlearn_strategy {strategy} "
                f"--forget_ratio {ratio} "
                f"--unlearn_test_file {test_file_type} "
                f"--use_wandb 0"
            )
            run_command(command)

    print("✅ 所有评估任务已完成！")


# --- 主程序入口 ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="运行模型遗忘和评估的自动化脚本 (最终版)"
    )
    parser.add_argument(
        "action",
        choices=["train", "eval", "all"],
        help="选择要执行的操作: 'train' - 仅运行遗忘训练, 'eval' - 仅运行评估, 'all' - 依次运行训练和评估",
    )
    args = parser.parse_args()

    if args.action == "train":
        run_unlearning_experiments()
    elif args.action == "eval":
        run_evaluation()
    elif args.action == "all":
        run_unlearning_experiments()
        run_evaluation()
