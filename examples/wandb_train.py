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
unlearning_arg_parser.add_argument(
    "--batch_size", type=int, default=None, help="训练时的 batch size, 会覆盖配置文件中的值"
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
def prepare_model_config(params, data_config, model_name, dataset_name):
    """
    根据传入的参数，准备并返回一个干净的、仅包含模型构造所需参数的字典。
    这是从主训练流程中分离出来的关键函数。
    """
    # 1. 复制所有参数作为基础
    model_config = copy.deepcopy(params)

    # 2. 定义需要 seq_len 参数的模型白名单 (官方逻辑)
    models_that_need_seq_len = ["saint", "saint++", "sakt", "atdkt", "simplekt", "stablekt", "bakt_time", "folibikt"]

    # 3.定义一个需要从 model_config 中移除的参数黑名单
    # 这个列表包含所有与模型构造无关的训练参数、标识符等
    keys_to_remove = [
        "model_name", "dataset_name", "emb_type", "save_dir", "fold", "seed",
        "use_wandb", "add_uuid", "config_path", "unlearn_method", "model_ckpt_path",
        "alpha", "unlearn_strategy", "forget_ratio", "num_epochs", "batch_size",
        "optimizer", "learning_rate", "l2", "seq_len" # 即使不在白名单中，也确保从最终配置中移除
    ]
    for key in keys_to_remove:
        if key in model_config:
            del model_config[key]
    
    # 4. 按需添加 seq_len
    if model_name in models_that_need_seq_len:
        # 优先使用数据集配置中的maxlen
        if 'maxlen' in data_config[dataset_name]:
            seq_len_value = data_config[dataset_name]['maxlen']
        else:
            seq_len_value = params['seq_len']
        model_config['seq_len'] = seq_len_value

    # 5. 针对特定模型的额外清理 (官方逻辑)
    if model_name == "dimkt":
        if 'weight_decay' in model_config:
            # weight_decay是优化器参数，不应传给模型构造函数
            del model_config['weight_decay']

    return model_config


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
    retain_file_path = os.path.join(
        data_config[params["dataset_name"]]["dpath"], retain_file_name
    )
    if not os.path.exists(retain_file_path):
        raise FileNotFoundError(f"指定的保留集文件不存在: {retain_file_path}")
    # 创建一个深拷贝以避免修改原始的data_config
    temp_data_config = copy.deepcopy(data_config)
    temp_data_config[params["dataset_name"]]["train_valid_file"] = retain_file_name
    print(
        f"已将数据集临时替换为: {retain_file_name}"
    )
    return temp_data_config


# ————————————————————————————————
# 任务执行函数区域
# ————————————————————————————————

def run_training(params, data_config, train_config):
    # --- 1. 提取核心参数 ---
    model_name, dataset_name, fold, emb_type, save_dir = (
        params["model_name"],
        params["dataset_name"],
        params["fold"],
        params["emb_type"],
        params["save_dir"],
    )
    num_epochs = params["num_epochs"]
    batch_size = params["batch_size"]
    learning_rate = params["learning_rate"]
    optimizer_type = params.get("optimizer", "adam")

    # --- 2. 准备模型配置 (调用新函数) ---
    print("正在准备模型专属配置...")
    model_config = prepare_model_config(params, data_config, model_name, dataset_name)
    print(f"模型配置准备完毕。")
    print(f"为模型 {model_name} 准备的专属的模型配置: {model_config}")
    print(f"为模型 {model_name} 准备的专属的训练配置: {train_config}")

    # --- 3. 初始化数据加载器 (集成dimkt特殊逻辑) ---
    print("正在初始化训练和验证数据加载器...")
    if model_name == "dimkt":
        if "difficult_levels" not in params:
            raise ValueError("运行dimkt模型时, 必须在参数中提供 'difficult_levels'")
        diff_level = params["difficult_levels"]
        train_loader, valid_loader, *_ = init_dataset4train(
            dataset_name, model_name, data_config, fold, batch_size, diff_level=diff_level
        )
    else:
        train_loader, valid_loader, *_ = init_dataset4train(
            dataset_name, model_name, data_config, fold, batch_size
        )
    
    # --- 4.初始化模型和优化器 (集成所有特殊逻辑) ---
    print(f"正在初始化模型: {model_name}...") 
    model = init_model(model_name, model_config, data_config[dataset_name], emb_type)
    print("模型初始化成功。")

    print(f"使用优化器: {optimizer_type}, 学习率: {learning_rate}")
    if model_name == "hawkes":
        weight_p, bias_p = [], []
        for name, p in filter(lambda x: x[1].requires_grad, model.named_parameters()):
            if 'bias' in name:
                bias_p.append(p)
            else:
                weight_p.append(p)
        optdict = [{'params': weight_p}, {'params': bias_p, 'weight_decay': 0}]
        opt = torch.optim.Adam(optdict, lr=learning_rate, weight_decay=params.get('l2', 0))
    elif model_name == "iekt":
        opt = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-6)
    elif model_name == "dtransformer":
        opt = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    elif model_name == "dimkt":
        opt = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=params.get('weight_decay', 0))
    else: # 通用优化器逻辑
        if optimizer_type == "sgd":
            opt = SGD(model.parameters(), learning_rate, momentum=0.9)
        elif optimizer_type == "adam":
            opt = Adam(model.parameters(), lr=learning_rate)
        else:
            # 作为默认回退，避免未定义optimizer_type时报错
            print(f"警告: 未知的优化器类型 '{optimizer_type}', 将使用默认的Adam。")
            opt = Adam(model.parameters(), lr=learning_rate)
            
    # --- 6. 准备保存路径并保存配置 ---
    params_str = f"{model_name}_{dataset_name}_seed{params['seed']}_fold{params['fold']}"
    if params.get("add_uuid", 0) == 1:
        params_str += f"_{str(uuid.uuid4())[:8]}"
    
    ckpt_path = os.path.join(save_dir, params_str)
    if not os.path.isdir(ckpt_path):
        os.makedirs(ckpt_path)
    
    print(f"模型和日志将保存至: {ckpt_path}")
    print(f"正在向 {ckpt_path} 保存最终配置文件 (config.json)...")
    # 注意保存的是修正后的 model_config
    save_config(train_config, model_config, data_config[dataset_name], params, ckpt_path)
    print("配置文件保存成功。")

    # --- 7. 开始训练 (集成rkt特殊逻辑) ---
    print("开始调用核心训练/评估函数 train_model...")
    if model_name == "rkt":
        testauc, testacc, window_testauc, window_testacc, validauc, validacc, best_epoch = train_model(
            model, train_loader, valid_loader, num_epochs, opt, ckpt_path, None, None, True, data_config[dataset_name], fold
        )
    else:
        testauc, testacc, window_testauc, window_testacc, validauc, validacc, best_epoch = train_model(
            model, train_loader, valid_loader, num_epochs, opt, ckpt_path, None, None, True
        )

    print("\n【训练与评估完成】")
    print(f"最佳验证集 AUC: {validauc:.4f}, ACC: {validacc:.4f} (在第 {best_epoch} 轮)")
    print(f"对应测试集 AUC: {testauc:.4f}, ACC: {testacc:.4f}")

    if params.get("use_wandb", 0) == 1:
        import wandb
        wandb.log({
            "testauc": testauc, "testacc": testacc,
            "window_testauc": window_testauc, "window_testacc": window_testacc,
            "validauc": validauc, "validacc": validacc,
            "best_epoch": best_epoch,
        })

def run_standard_training(params, data_config, train_config):
    """任务一：标准训练"""
    print("="*40)
    print("✨ 执行任务：[标准训练]")
    run_training(params, data_config, train_config)

# (你的遗忘学习相关函数无需修改，因为它们依赖于已保存的正确配置)
def run_unlearning_retrain(params, data_config, train_config):
    """任务二：遗忘方法 - 从头重训练"""
    print("✨ 执行任务：[遗忘方法 - 从头重训练]")
    # 临时替换数据集为保留集
    temp_data_config = replace_dataset(params, data_config, "retain")
    run_training(params, temp_data_config, train_config)

def run_unlearning_task(params, data_config):
    """
    通用的遗忘任务执行器 (处理 surgical, ascent, finetune)
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
    程序主入口和任务分发器 (采用你设计的清晰的参数合并逻辑)
    """
    # --- Wandb 初始化 ---
    if params.get("use_wandb", 0) == 1:
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
    # 优先级顺序: 命令行 > 模型专属 > 通用训练配置
    model_name = params["model_name"]

    # 1. 从通用训练配置开始
    final_params = config_from_json.get("train_config", {}).copy()

    # 2. 合并模型专属配置
    model_specific_config = config_from_json.get(model_name, {})
    final_params.update(model_specific_config)

    # 3. 最后，用命令行传入的非None参数覆盖，这是最高优先级
    # 注意: argparse会将未提供的参数设为None
    explicit_params = {k: v for k, v in params.items() if v is not None}
    final_params.update(explicit_params)

    # 4. 将最终合并好的参数赋给 params 变量，供后续所有流程统一使用
    params = final_params

    # 准备 train_config, 主要用于 retrain 任务时的日志记录
    train_config = config_from_json.get("train_config", {})
    # 如果传入的batchsize大小为None，则根据模型名称调整 batch_size
    if params.get("batch_size") == 256:
        if model_name in ["dkvmn","deep_irt", "sakt", "saint","saint++", "akt", "robustkt", "folibikt", "atkt", "lpkt", "skvmn", "dimkt"]:
                train_config["batch_size"] = 64 ## because of OOM
        if model_name in ["simplekt","stablekt", "bakt_time", "sparsekt"]:
            train_config["batch_size"] = 64 ## because of OOM
        if model_name in ["gkt"]:
            train_config["batch_size"] = 16 
        if model_name in ["qdkt","qikt"] and dataset_name in ['algebra2005','bridge2algebra2006']:
            train_config["batch_size"] = 32 
        if model_name in ["dtransformer"]:
            train_config["batch_size"] = 32 ## because of OOM
    else:
        train_config["batch_size"] = params.get("batch_size")
    train_config["num_epochs"] = params.get("num_epochs")

    # --- 核心任务分发逻辑 ---
    unlearn_method = params.get("unlearn_method")

    if unlearn_method is None:
        run_standard_training(params, data_config, train_config)
    elif unlearn_method == "retrain":
        run_unlearning_retrain(params, data_config, train_config)
    elif unlearn_method in ["surgical", "ascent", "finetune", "fisher"]:
        run_unlearning_task(params, data_config)
    else:
        available = ", ".join(["retrain", "surgical", "ascent", "finetune", "fisher"])
        raise ValueError(f"不支持的遗忘方法: '{unlearn_method}'。可用: {available}")
