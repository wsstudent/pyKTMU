import os
import argparse
import json
import copy
import torch
import pandas as pd
import pdb  # 导入Python调试器

# 从pykt库导入模型评估和数据加载相关函数
from pykt.models import evaluate, evaluate_question, load_model
from pykt.datasets import init_test_datasets

# 设置运行设备，优先使用CUDA，否则使用CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
# 设置环境变量，确保CUDA操作的可复现性
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:2"

# 全局加载wandb配置文件
try:
    with open("../configs/wandb.json") as fin:
        WANDB_CONFIG = json.load(fin)
except FileNotFoundError:
    print("Warning: wandb.json not found. WandB integration will be disabled.")
    WANDB_CONFIG = None


def load_configs(params):
    """
    加载模型和数据配置。
    Args:
        params (dict): 包含命令行参数的字典。
    Returns:
        tuple: 包含 model_config, data_config, trained_params 的元组。
    """
    save_dir = params["save_dir"]

    # 1. 加载已保存的模型训练配置 (config.json)
    config_path = os.path.join(save_dir, "config.json")
    try:
        with open(config_path) as fin:
            config = json.load(fin)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading {config_path}: {e}")
        # 如果配置文件无法加载，则无法继续，程序退出
        exit(1)

    # 提取模型配置，并移除训练时使用但预测时不需要的参数
    model_config = copy.deepcopy(config["model_config"])
    for key in ["use_wandb", "learning_rate", "add_uuid", "l2"]:
        model_config.pop(key, None)  # 使用pop(key, None)避免KeyError

    # 提取训练时的参数，如模型名称、数据集名称等
    trained_params = config["params"]
    model_name = trained_params["model_name"]
    dataset_name = trained_params["dataset_name"]

    # 部分模型需要额外的配置（如序列长度）
    if model_name in ["saint", "sakt", "atdkt"]:
        model_config["seq_len"] = config["train_config"]["seq_len"]

    # 2. 加载数据集配置 (data_config.json)
    try:
        with open("../configs/data_config.json") as fin:
            data_config = json.load(fin)[dataset_name]
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading data_config.json: {e}")
        exit(1)

    data_config["dataset_name"] = dataset_name

    # 为特定模型加载额外的数据相关配置
    if model_name in ["dkt_forget", "bakt_time"]:
        for key in ["num_rgap", "num_sgap", "num_pcount"]:
            data_config[key] = config["data_config"][key]
    elif model_name == "lpkt":
        for key in ["num_at", "num_it"]:
            data_config[key] = config["data_config"][key]

    # 3. 如果指定了 unlearning 策略，则修改测试文件路径
    if params["unlearn_strategy"] != "none":
        strategy = params["unlearn_strategy"]
        test_file = params["unlearn_test_file"]
        ratio = params["forget_ratio"]
        # 构建新的测试文件名
        new_test_file = f"test_sequences_{test_file}_{strategy}_ratio{ratio}.csv"
        print(f"已加载对应的{params['unlearn_test_file']}数据集, '{new_test_file}'")
        data_config["test_file"] = new_test_file

    return model_config, data_config, trained_params


def initialize_dataloaders(data_config, model_name, batch_size, trained_params):
    """
    根据配置初始化测试数据加载器。
    Args:
        data_config (dict): 数据集配置。
        model_name (str): 模型名称。
        batch_size (int): 批处理大小。
        trained_params (dict): 训练参数。
    Returns:
        tuple: 包含各种测试数据加载器的元组。
    """
    if model_name == "dimkt":
        # DIMKT模型需要额外的难度等级参数
        diff_level = trained_params["difficult_levels"]
        return init_test_datasets(
            data_config, model_name, batch_size, diff_level=diff_level
        )
    else:
        # 其他模型使用标准初始化
        return init_test_datasets(data_config, model_name, batch_size)


def run_evaluation(model, loader, model_name, save_path, rel=None):
    """
    在给定的数据加载器上运行评估。
    Args:
        model: 已加载的模型。
        loader: 数据加载器。
        model_name (str): 模型名称。
        save_path (str): 预测结果保存路径。
        rel (any, optional): RKT模型需要的额外关系数据。 Defaults to None.
    Returns:
        tuple: (AUC, ACC) 分数。
    """
    if loader is None:
        return -1, -1  # 如果加载器不存在，返回默认值

    print(f"Evaluating on: {os.path.basename(save_path)}")
    if model_name == "rkt":
        return evaluate(model, loader, model_name, rel, save_path)
    else:
        return evaluate(model, loader, model_name, save_path)


def run_question_evaluation(model, loader, model_name, fusion_types, save_path):
    """
    在给定的数据加载器上运行基于问题的评估。
    Args:
        model: 已加载的模型。
        loader: 数据加载器。
        model_name (str): 模型名称。
        fusion_types (list): 融合类型列表。
        save_path (str): 预测结果保存路径。
    Returns:
        tuple: (AUCs, ACCs) 字典。
    """
    if loader is None:
        return {}, {}  # 如果加载器不存在，返回空字典

    print(f"Evaluating questions on: {os.path.basename(save_path)}")
    return evaluate_question(model, loader, model_name, fusion_types, save_path)


def save_results_to_csv(result_data, csv_path):
    """将单次评估结果增量保存到指定的CSV文件。

    Args:
        result_data (dict): 包含单行结果的字典。
        csv_path (str): 目标CSV文件的路径。
    """
    # 检查文件是否存在，以决定是否需要写入表头
    file_exists = os.path.exists(csv_path)

    # 将字典转换为pandas DataFrame
    df = pd.DataFrame([result_data])

    print(f"正在将结果保存至: {csv_path}...")
    try:
        if not file_exists:
            # 如果文件不存在，创建新文件并写入表头
            df.to_csv(
                csv_path, mode="w", header=True, index=False, encoding="utf-8-sig"
            )
            print("CSV文件已创建，并写入表头。")
        else:
            # 如果文件已存在，以追加模式写入，不包含表头
            df.to_csv(
                csv_path, mode="a", header=False, index=False, encoding="utf-8-sig"
            )
            print("结果已成功追加到现有CSV文件。")
    except Exception as e:
        print(f"保存到CSV时发生错误: {e}")


def main(params):
    """
    主执行函数。
    """
    # 1. 初始化 WandB (Weights & Biases)
    if params["use_wandb"] == 1 and WANDB_CONFIG:
        import wandb

        os.environ["WANDB_API_KEY"] = WANDB_CONFIG["api_key"]
        wandb.init(project="wandb_predict_optimized")

    # 2. 加载配置
    model_config, data_config, trained_params = load_configs(params)

    model_name = trained_params["model_name"]
    dataset_name = trained_params["dataset_name"]
    emb_type = trained_params["emb_type"]
    fold = trained_params["fold"]
    save_dir = params["save_dir"]

    print("\n" + "=" * 50)
    print(f"Starting Prediction for Model: {model_name}")
    print(f"Dataset: {dataset_name}, Embedding: {emb_type}, Fold: {fold}")
    print(f"Save Directory: {save_dir}")
    print("=" * 50 + "\n")
    print(f"Model Config: {model_config}")
    print(f"Data Config: {data_config}")

    # 调试点：检查加载的配置是否正确
    if params["debug"]:
        print("\n--- DEBUGGER: Post-Config-Load ---")
        print("Inspect `params`, `model_config`, `data_config`, `trained_params`")
        pdb.set_trace()

    # 3. 初始化数据加载器
    dataloaders = initialize_dataloaders(
        data_config, model_name, params["bz"], trained_params
    )
    (
        test_loader,
        test_window_loader,
        test_question_loader,
        test_question_window_loader,
    ) = dataloaders

    # 4. 加载模型
    model = load_model(model_name, model_config, data_config, emb_type, save_dir)

    # 5. 特殊处理：为 RKT 模型加载关系数据
    rel = None
    if model_name == "rkt":
        dpath = data_config["dpath"]
        # 计算除了当前折叠之外的其他所有折叠
        other_folds = set(data_config["folds"]) - {fold}
        folds_str = "_" + "_".join(map(str, sorted(list(other_folds))))

        # 根据数据集名称确定关系文件名
        rel_filename = (
            "phi_dict"
            if dpath.split("/")[-1] in ["algebra2005", "bridge2algebra2006"]
            else "phi_array"
        )
        rel_path = os.path.join(dpath, f"{rel_filename}{folds_str}.pkl")

        try:
            print(f"Loading RKT relation data from: {rel_path}")
            rel = pd.read_pickle(rel_path)
        except FileNotFoundError:
            print(
                f"Warning: RKT relation file not found at {rel_path}. `rel` will be None."
            )

    # 调试点：检查已加载的模型和数据加载器
    if params["debug"]:
        print("\n--- DEBUGGER: Pre-Evaluation ---")
        print("Inspect `model`, `test_loader`, and other dataloaders, `rel` for RKT")
        pdb.set_trace()

    # 6. 执行评估并收集结果
    dres = {}  # 用于存储所有评估结果的字典

    # 6.1 常规测试集评估
    save_test_path = os.path.join(save_dir, f"{model.emb_type}_test_predictions.txt")
    dres["testauc"], dres["testacc"] = run_evaluation(
        model, test_loader, model_name, save_test_path, rel
    )

    # 定义要保存的CSV文件名，可以根据需要修改
    results_csv_path = "../data/evaluation_results.csv"

    # 准备要保存到CSV的数据行
    # 注意：这里的键名将作为CSV文件的列名
    csv_row_data = {
        "模型": params.get("save_dir", "N/A"),
        "数据集": trained_params.get("dataset_name", "N/A"),
        "遗忘策略": params.get("unlearn_strategy", "none"),
        "遗忘比例": params.get("forget_ratio", 0.0),
        "测试集类型": params.get(
            "unlearn_test_file", "original"
        ),  # 例如 'forget', 'retain', 或 'original'
        "auc": dres["testauc"],
        "acc": dres["testacc"],
    }

    # 调用函数保存数据
    save_results_to_csv(csv_row_data, results_csv_path)

    return

    # 6.2 滑动窗口测试集评估
    save_test_window_path = os.path.join(
        save_dir, f"{model.emb_type}_test_window_predictions.txt"
    )
    dres["window_testauc"], dres["window_testacc"] = run_evaluation(
        model, test_window_loader, model_name, save_test_window_path, rel
    )

    print(
        f"\nStandard Evaluation Results:\n"
        f"  Test AUC: {dres['testauc']:.4f}, Test ACC: {dres['testacc']:.4f}\n"
        f"  Window AUC: {dres['window_testauc']:.4f}, Window ACC: {dres['window_testacc']:.4f}"
    )

    # 6.3 基于问题的评估
    fusion_types = params["fusion_type"].split(",")

    # 基于问题的常规测试集评估
    if test_question_loader:
        save_path = os.path.join(
            save_dir, f"{model.emb_type}_test_question_predictions.txt"
        )
        q_aucs, q_accs = run_question_evaluation(
            model, test_question_loader, model_name, fusion_types, save_path
        )
        for key, val in q_aucs.items():
            dres[f"oriauc{key}"] = val
        for key, val in q_accs.items():
            dres[f"oriacc{key}"] = val

    # 基于问题的滑动窗口测试集评估
    if test_question_window_loader:
        save_path = os.path.join(
            save_dir, f"{model.emb_type}_test_question_window_predictions.txt"
        )
        qw_aucs, qw_accs = run_question_evaluation(
            model, test_question_window_loader, model_name, fusion_types, save_path
        )
        for key, val in qw_aucs.items():
            dres[f"windowauc{key}"] = val
        for key, val in qw_accs.items():
            dres[f"windowacc{key}"] = val

    # 7. 打印并记录最终结果
    print("\n" + "=" * 50)
    print("Final Evaluation Results Summary:")
    print(json.dumps(dres, indent=4))
    print("=" * 50 + "\n")

    # 将原始训练参数合并到结果中，以便于追溯
    dres.update(trained_params)

    if params["use_wandb"] == 1 and WANDB_CONFIG:
        wandb.log(dres)
        print("Results successfully logged to WandB.")


if __name__ == "__main__":
    # 定义命令行参数解析器
    parser = argparse.ArgumentParser(description="Run model evaluation.")
    parser.add_argument("--bz", type=int, default=256, help="批处理大小 (Batch size)")
    parser.add_argument(
        "--save_dir", type=str, default="saved_model", help="已保存模型的目录路径"
    )
    parser.add_argument(
        "--fusion_type",
        type=str,
        default="early_fusion,late_fusion",
        help="问题融合策略，用逗号分隔",
    )
    parser.add_argument(
        "--use_wandb",
        type=int,
        default=1,
        choices=[0, 1],
        help="是否使用WandB记录结果 (1 for yes, 0 for no)",
    )

    # 非学习 (Unlearning) 相关参数
    parser.add_argument(
        "--unlearn_strategy",
        type=str,
        default="none",
        help="数据划分策略 (e.g., 'random', 'last', 'none')",
    )
    parser.add_argument(
        "--forget_ratio",
        type=float,
        default=0.2,
        help="遗忘数据比例 (e.g., 0.2 for 20%%)",
    )
    parser.add_argument(
        "--unlearn_test_file",
        type=str,
        default="forget",
        help="测试集类型 ('forget' or 'retain')",
    )

    # 调试参数
    parser.add_argument(
        "--debug", action="store_true", help="启用调试模式 (Enable pdb debugger)"
    )

    # 解析命令行参数
    args = parser.parse_args()

    # 将解析后的参数转换为字典
    params = vars(args)

    # 打印本次运行的参数配置
    print("Running with parameters:")
    print(json.dumps(params, indent=4))

    # 调用主函数
    main(params)
