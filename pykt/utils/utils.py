import os
import torch
import numpy as np
import json
import copy

from torch.optim import SGD, Adam


def set_seed(seed):
    """Set the global random seed.

    Args:
        seed (int): random seed
    """
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except Exception as e:
        print("Set seed failed,details are ", e)
        pass

    np.random.seed(seed)
    import random as python_random

    python_random.seed(seed)
    # cuda env
    import os

    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"


import datetime


def get_now_time():
    """Return the time string, the format is %Y-%m-%d %H:%M:%S

    Returns:
        str: now time
    """
    now = datetime.datetime.now()
    dt_string = now.strftime("%Y-%m-%d %H:%M:%S")
    return dt_string


def debug_print(text, fuc_name=""):
    """Printing text with function name.

    Args:
        text (str): the text will print
        fuc_name (str, optional): _description_. Defaults to "".
    """
    print(f"{get_now_time()} - {fuc_name} - said: {text}")


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
    models_that_need_seq_len = [
        "saint",
        "saint++",
        "sakt",
        "atdkt",
        "simplekt",
        "stablekt",
        "bakt_time",
        "folibikt",
    ]

    # 3.定义一个需要从 model_config 中移除的参数黑名单
    # 这个列表包含所有与模型构造无关的训练参数、标识符等
    keys_to_remove = [
        "model_name",
        "dataset_name",
        "emb_type",
        "save_dir",
        "fold",
        "seed",
        "use_wandb",
        "add_uuid",
        "config_path",
        "unlearn_method",
        "model_ckpt_path",
        "alpha",
        "unlearn_strategy",
        "forget_ratio",
        "num_epochs",
        "batch_size",
        "optimizer",
        "learning_rate",
        "l2",
        "seq_len",  # 即使不在白名单中，也确保从最终配置中移除
    ]
    for key in keys_to_remove:
        if key in model_config:
            del model_config[key]

    # 4. 按需添加 seq_len
    if model_name in models_that_need_seq_len:
        # 优先使用数据集配置中的maxlen
        if "maxlen" in data_config[dataset_name]:
            seq_len_value = data_config[dataset_name]["maxlen"]
        else:
            seq_len_value = params["seq_len"]
        model_config["seq_len"] = seq_len_value

    # 5. 针对特定模型的额外清理 (官方逻辑)
    if model_name == "dimkt":
        if "weight_decay" in model_config:
            # weight_decay是优化器参数，不应传给模型构造函数
            del model_config["weight_decay"]

    return model_config


def prepare_model_optimizer(
    params, model_name, optimizer_type, parameters, learning_rate
):
    if model_name == "hawkes":
        opt = torch.optim.Adam(
            parameters, lr=learning_rate, weight_decay=params.get("l2", 0)
        )
    elif model_name == "iekt":
        opt = torch.optim.Adam(parameters, lr=learning_rate, weight_decay=1e-6)
    elif model_name == "dtransformer":
        opt = torch.optim.Adam(parameters, lr=learning_rate, weight_decay=1e-5)
    elif model_name == "dimkt":
        opt = torch.optim.Adam(
            parameters, lr=learning_rate, weight_decay=params.get("weight_decay", 0)
        )
    else:  # 通用优化器逻辑
        if optimizer_type == "sgd":
            opt = SGD(parameters, learning_rate, momentum=0.9)
        elif optimizer_type == "adam":
            opt = Adam(parameters, lr=learning_rate)
        else:
            # 作为默认回退，避免未定义optimizer_type时报错
            print(f"警告: 未知的优化器类型 '{optimizer_type}', 将使用默认的Adam。")
            opt = Adam(parameters, lr=learning_rate)
    return opt


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
    print(f"已将数据集临时替换为: {retain_file_name}")
    return temp_data_config
