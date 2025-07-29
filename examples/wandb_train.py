import os
import argparse
import json
import torch
import copy
import uuid

from torch.optim import SGD, Adam
from pykt.models import train_model, init_model
from pykt.utils import set_seed
from pykt.datasets import init_dataset4train
from pykt.utils.Unlearner import Unlearner

# ————————————————————————————————
# 共享参数定义 (提供给启动器脚本继承)
# ————————————————————————————————
unlearning_arg_parser = argparse.ArgumentParser(add_help=False)
unlearning_arg_parser.add_argument(
    "--unlearn_method",
    type=str,
    default=None,
    help="选择遗忘方法 ('retrain', 'fisher')。默认为None，即执行标准训练。",
)
unlearning_arg_parser.add_argument(
    "--unlearn_strategy", type=str, default="random", help="[retrain专用] 数据划分策略"
)
unlearning_arg_parser.add_argument(
    "--forget_ratio", type=float, default=0.2, help="[retrain专用] 遗忘数据比例"
)
unlearning_arg_parser.add_argument(
    "--model_ckpt_path",
    type=str,
    default=None,
    help="[fisher专用] 预训练模型检查点路径",
)
unlearning_arg_parser.add_argument(
    "--alpha", type=float, default=10.0, help="[fisher专用] 遗忘强度超参数"
)


# ————————————————————————————————
# 全局设置
# ————————————————————————————————
device = "cpu" if not torch.cuda.is_available() else "cuda"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:2"
torch.set_num_threads(4)


# ————————————————————————————————
# 辅助函数区域
# ————————————————————————————————
def save_config(train_config, model_config, data_config, params, save_dir):
    """将训练配置、模型配置、数据配置和超参数保存到指定目录的json文件中。"""
    d = {
        "train_config": train_config,
        "model_config": model_config,
        "data_config": data_config,
        "params": params,
    }
    save_path = os.path.join(save_dir, "config.json")
    with open(save_path, "w") as fout:
        json.dump(d, fout, indent=4)


def replace_dataset(params, data_config, dataset_type):
    """根据参数中的 dataset_name 替换数据配置中的数据集路径。"""
    retain_file_name = f"train_valid_sequences_{dataset_type}_{params['unlearn_strategy']}_ratio{params['forget_ratio']}.csv"
    retain_file_patr = os.path.join(
        data_config[params["dataset_name"]]["dpath"], retain_file_name
    )
    if not os.path.exists(retain_file_patr):
        raise FileNotFoundError(f"指定的保留集文件不存在: {retain_file_patr}")
    data_config[params["dataset_name"]]["train_valid_file"] = retain_file_name
    print(
        f"已将数据{data_config[params['dataset_name']]['train_valid_file']}的替换为: {retain_file_name}"
    )
    return data_config


# ————————————————————————————————
# 任务执行函数区域
# ————————————————————————————————


def run_standard_training(params, data_config, train_config):
    """任务一：标准模型训练 (已修正所有流程)"""
    print("✨ 执行任务：[标准训练]")

    model_name, dataset_name, fold, emb_type, save_dir = (
        params["model_name"],
        params["dataset_name"],
        params["fold"],
        params["emb_type"],
        params["save_dir"],
    )
    num_epochs = params["num_epochs"]
    batch_size = params["batch_size"]

    print("正在初始化训练和验证数据加载器...")
    train_loader, valid_loader, *_ = init_dataset4train(
        dataset_name, model_name, data_config, fold, batch_size
    )

    model_config = copy.deepcopy(params)
    non_model_keys = [
        "model_name",
        "dataset_name",
        "emb_type",
        "save_dir",
        "fold",
        "seed",
        "use_wandb",
        "add_uuid",
        "unlearn_method",
        "model_ckpt_path",
        "alpha",
        "unlearn_strategy",
        "forget_ratio",
        "num_epochs",
        "batch_size",
        "optimizer",
        "seq_len",
        "learning_rate",
    ]
    for key in non_model_keys:
        if key in model_config:
            del model_config[key]

    model = init_model(model_name, model_config, data_config[dataset_name], emb_type)

    optimizer_type = params.get("optimizer", "adam")
    learning_rate = params["learning_rate"]

    print(f"使用优化器: {optimizer_type}, 学习率: {learning_rate}")
    if optimizer_type == "hawkes_adam":
        weight_p, bias_p = [], []
        for name, p in filter(lambda x: x[1].requires_grad, model.named_parameters()):
            if "bias" in name:
                bias_p.append(p)
            else:
                weight_p.append(p)
        optdict = [{"params": weight_p}, {"params": bias_p, "weight_decay": 0}]
        opt = torch.optim.Adam(
            optdict, lr=learning_rate, weight_decay=params.get("l2", 0)
        )
    elif optimizer_type == "adam_wd_1e-5":
        opt = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    elif optimizer_type == "adam_wd_1e-6":
        opt = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-6)
    elif optimizer_type == "sgd":
        opt = SGD(model.parameters(), learning_rate, momentum=0.9)
    else:
        opt = Adam(model.parameters(), lr=learning_rate)

    params_str = (
        f"{model_name}_{dataset_name}_seed{params['seed']}_fold{params['fold']}"
    )
    if params.get("add_uuid", 0) == 1:
        params_str += f"_{str(uuid.uuid4())[:8]}"
    ckpt_path = os.path.join(save_dir, params_str)
    if not os.path.isdir(ckpt_path):
        os.makedirs(ckpt_path)
    print(f"模型和日志将保存至: {ckpt_path}")

    print(f"正在向 {ckpt_path} 保存配置文件 (config.json)...")
    save_config(
        train_config, model_config, data_config[dataset_name], params, ckpt_path
    )
    print("配置文件保存成功。")

    print("开始调用核心训练/评估函数 train_model...")
    testauc, testacc, window_testauc, window_testacc, validauc, validacc, best_epoch = (
        train_model(
            model,
            train_loader,
            valid_loader,
            num_epochs,
            opt,
            ckpt_path,
            save_model=True,
        )
    )

    print("【训练与评估完成】")
    print(f"最佳验证集 AUC: {validauc:.4f}, ACC: {validacc:.4f} (在第 {best_epoch} 轮)")
    print(f"对应测试集 AUC: {testauc:.4f}, ACC: {testacc:.4f}")

    if params.get("use_wandb", 0) == 1:
        import wandb

        wandb.log(
            {
                "testauc": testauc,
                "testacc": testacc,
                "window_testauc": window_testauc,
                "window_testacc": window_testacc,
                "validauc": validauc,
                "validacc": validacc,
                "best_epoch": best_epoch,
            }
        )


def run_unlearning_retrain(params, data_config, train_config):
    print("执行任务：[遗忘方法 - 从头重训练]")
    data_config = replace_dataset(params, data_config, "retain")
    run_standard_training(params, data_config, train_config)


def run_unlearning_task(params, data_config):
    """
    【最终完整版】通用的遗忘任务执行器 (处理 surgical, ascent, finetune)
    """
    unlearn_method = params["unlearn_method"]
    print(f"✨ 执行任务：[遗忘方法 - {unlearn_method}]")

    # --- 1. 加载预训练模型 ---
    model_name, dataset_name, fold, emb_type, save_dir = (
        params["model_name"],
        params["dataset_name"],
        params["fold"],
        params["emb_type"],
        params["save_dir"],
    )
    if not params.get("model_ckpt_path"):
        raise ValueError(
            f"使用 '{unlearn_method}' 方法时，必须提供 --model_ckpt_path 参数。"
        )

    print(f"加载预训练模型于: {params['model_ckpt_path']}")
    with open(os.path.join(params["model_ckpt_path"], "config.json")) as f:
        pretrain_config = json.load(f)
        model_config = pretrain_config["model_config"]
        original_train_config = pretrain_config["train_config"]

    original_model = init_model(
        model_name, model_config, data_config[dataset_name], emb_type
    )
    model_path = os.path.join(params["model_ckpt_path"], f"{emb_type}_model.ckpt")
    original_model.load_state_dict(torch.load(model_path, map_location=device))
    original_model.to(device)
    original_model.eval()  # 先设置为评估模式
    print(f"已加载预训练模型: {model_name} 于 {model_path}")

    # --- 2. 初始化 Unlearner 和数据加载器 ---
    unlearner = Unlearner(model=original_model, model_name=model_name)
    batch_size = params.get("batch_size", 64)

    retain_loader, forget_loader = None, None
    retain_valid_loader, forget_valid_loader = None, None
    if unlearn_method in ["surgical", "finetune"]:
        print("正在初始化保留集数据加载器...")
        retain_data_config = replace_dataset(params, data_config, "retain")
        retain_loader, retain_valid_loader, *_ = init_dataset4train(
            dataset_name, model_name, retain_data_config, fold, batch_size
        )
        print(
            f"保留集数据: {len(retain_loader.dataset)} 条, 其中验证集: {len(retain_valid_loader.dataset)} 条"
        )

    if unlearn_method in ["surgical", "ascent"]:
        print("正在初始化遗忘集数据加载器...")
        forget_data_config = replace_dataset(params, data_config, "forget")
        forget_loader, forget_valid_loader, *_ = init_dataset4train(
            dataset_name, model_name, forget_data_config, fold, batch_size
        )
        print(
            f"遗忘集数据: {len(forget_loader.dataset)} 条， 其中验证集: {len(forget_valid_loader.dataset)} 条"
        )

    # --- 3. 调用统一接口执行遗忘 ---
    unlearner.unlearn(
        method=unlearn_method,
        retain_loader=retain_loader,
        forget_loader=forget_loader,
        alpha=params.get("alpha"),
        device=device,
        # 传递 finetune 的特定参数
        finetune_epochs=params.get("finetune_epochs"),
        finetune_lr=params.get("finetune_lr"),
        finetune_layers=params.get("finetune_layers"),
    )

    unlearned_model = unlearner.model
    print("遗忘过程执行完毕！")

    # --- 4. 保存遗忘后的模型和配置 ---
    print("正在保存遗忘后的结果...")
    # 构建动态的、信息丰富的文件夹名称
    method_params_str = ""
    if unlearn_method in ["surgical", "ascent"]:
        method_params_str = f"alpha{params['alpha']}"
    elif unlearn_method == "finetune":
        method_params_str = f"ep{params['finetune_epochs']}_lr{params['finetune_lr']}"

    params_str = f"{unlearn_method}_{model_name}_{dataset_name}_{method_params_str}_seed{params['seed']}"
    if params.get("add_uuid", 0) == 1:
        params_str += f"_{str(uuid.uuid4())[:8]}"

    ckpt_path = os.path.join(save_dir, params_str)
    if not os.path.isdir(ckpt_path):
        os.makedirs(ckpt_path)

    print(f"遗忘后的模型和日志将保存至: {ckpt_path}")
    model_save_path = os.path.join(ckpt_path, f"{emb_type}_model.ckpt")
    torch.save(unlearned_model.state_dict(), model_save_path)
    print(f"遗忘后的模型权重已保存至: {model_save_path}")

    # 保存本次运行的完整配置
    final_params_for_save = copy.deepcopy(pretrain_config["params"])
    final_params_for_save.update(params)
    final_params_for_save["original_model_path"] = params["model_ckpt_path"]
    save_config(
        original_train_config,
        model_config,
        data_config[dataset_name],
        final_params_for_save,
        ckpt_path,
    )
    print("遗忘后模型的配置文件 (config.json) 已保存。")


# ————————————————————————————————
# 统一的主函数和分发逻辑
# ————————————————————————————————
def main(params):
    """
    程序主入口和任务分发器
    """
    # --- Wandb 初始化 ---
    if params.get("use_wandb", 1) == 1:
        import wandb

        # 过滤掉值为None的参数，避免wandb报错
        wandb_config = {k: v for k, v in params.items() if v is not None}
        wandb.init(config=wandb_config, project="pykt-unlearn-project")

    # --- 设置随机种子和加载配置 ---
    set_seed(params.get("seed", 42))

    # 加载通用配置和数据配置
    with open("../configs/kt_config.json") as f:
        config_from_json = json.load(f)
    with open("../configs/data_config.json") as fin:
        data_config = json.load(fin)

    # --- 合并与更新参数 ---
    # 优先级顺序: 命令行 > 模型专属 > unlearning配置 > 通用训练配置
    model_name = params["model_name"]

    # 1. 从通用训练配置开始
    final_params = config_from_json.get("train_config", {}).copy()

    # 2. 加载并合并 unlearning 的默认配置 (这是之前缺失的关键步骤)
    if params.get("unlearn_method") == "finetune":
        if params.get("finetune_epochs") is None:
            params["finetune_epochs"] = 3
            print("Info: 'finetune_epochs' 未提供, 已在代码中设置为默认值 3")
        if params.get("finetune_lr") is None:
            params["finetune_lr"] = 1e-4
            print("Info: 'finetune_lr' 未提供, 已在代码中设置为默认值 1e-4")
        if params.get("finetune_layers") is None:
            params["finetune_layers"] = ["out", "output"]
            print(
                "Info: 'finetune_layers' 未提供, 已在代码中设置为默认值 ['out', 'output']"
            )

    # 3. 合并模型专属配置
    model_specific_config = config_from_json.get(model_name, {})
    final_params.update(model_specific_config)

    # 4. 最后，用命令行传入的非None参数覆盖，这是最高优先级
    explicit_params = {k: v for k, v in params.items() if v is not None}
    final_params.update(explicit_params)

    # 5. 将最终合并好的参数赋给 params 变量，供后续所有流程统一使用
    params = final_params
    print(f"最终生效的参数配置: {json.dumps(params, indent=4)}")

    # 准备 train_config, 主要用于 retrain 任务
    train_config = config_from_json.get("train_config", {})
    train_config["batch_size"] = params.get("batch_size", train_config["batch_size"])
    train_config["num_epochs"] = params.get("num_epochs", train_config["num_epochs"])

    # --- 核心任务分发逻辑 ---
    unlearn_method = params.get("unlearn_method")

    if unlearn_method is None:
        # 情况1: 标准训练
        run_standard_training(params, data_config, train_config)

    elif unlearn_method == "retrain":
        # 情况2: 从头重训练
        run_unlearning_retrain(params, data_config, train_config)

    elif unlearn_method in ["surgical", "ascent", "finetune"]:
        # 情况3: 其他所有需要预训练模型的遗忘方法
        run_unlearning_task(params, data_config)

    else:
        # 情况4: 无效的方法名
        available = ", ".join(["retrain", "surgical", "ascent", "finetune"])
        raise ValueError(f"不支持的遗忘方法: '{unlearn_method}'。可用: {available}")
