import os
import argparse
import json

import torch

# 设置 PyTorch 使用的线程数，以优化性能
torch.set_num_threads(4)
from torch.optim import SGD, Adam
import copy

from pykt.models import train_model, evaluate, init_model
from pykt.utils import debug_print, set_seed
from pykt.datasets import init_dataset4train
import datetime

# 设置环境变量，用于CUDA调试
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# 判断使用CPU还是GPU
device = "cpu" if not torch.cuda.is_available() else "cuda"
# 设置CUDA环境配置，以保证结果可复现
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:2"


def save_config(train_config, model_config, data_config, params, save_dir):
    """
    将训练配置、模型配置、数据配置和超参数保存到指定目录的json文件中。

    Args:
        train_config (dict): 训练相关的配置。
        model_config (dict): 模型相关的配置。
        data_config (dict): 数据集相关的配置。
        params (dict): 命令行传入的超参数。
        save_dir (str): 保存配置文件的目录路径。
    """
    d = {
        "train_config": train_config,
        "model_config": model_config,
        "data_config": data_config,
        "params": params,
    }
    save_path = os.path.join(save_dir, "config.json")
    with open(save_path, "w") as fout:
        json.dump(d, fout, indent=4)  # 使用indent参数美化输出


def main(params):
    """
    主训练函数。
    """
    # 如果未指定，默认使用wandb进行实验跟踪,
    # 如果启用wandb，则初始化
    if "use_wandb" not in params:
        params["use_wandb"] = 1

    # 如果启用wandb，则初始化
    if params["use_wandb"] == 1:
        import wandb

        wandb.init()

    # 设置随机种子以保证实验可复现
    set_seed(params["seed"])

    # 从参数字典中提取常用变量
    model_name, dataset_name, fold, emb_type, save_dir = (
        params["model_name"],
        params["dataset_name"],
        params["fold"],
        params["emb_type"],
        params["save_dir"],
    )

    debug_print(text="开始加载配置文件。", fuc_name="main")

    # 加载通用的训练配置文件
    with open("../configs/kt_config.json") as f:
        config = json.load(f)
        train_config = config["train_config"]
        # 针对特定模型调整batch_size，通常是为了防止显存不足(OOM)
        if model_name in [
            "dkvmn",
            "deep_irt",
            "sakt",
            "saint",
            "saint++",
            "akt",
            "robustkt",
            "folibikt",
            "atkt",
            "lpkt",
            "skvmn",
            "dimkt",
        ]:
            train_config["batch_size"] = 64
        if model_name in ["simplekt", "stablekt", "bakt_time", "sparsekt"]:
            train_config["batch_size"] = 64
        if model_name in ["gkt"]:
            train_config["batch_size"] = 16
        if model_name in ["qdkt", "qikt"] and dataset_name in [
            "algebra2005",
            "bridge2algebra2006",
        ]:
            train_config["batch_size"] = 32
        if model_name in ["dtransformer"]:
            train_config["batch_size"] = 32

        # 复制命令行参数作为模型配置的基础
        model_config = copy.deepcopy(params)
        # 从模型配置中移除与模型结构无关的参数
        for key in [
            "model_name",
            "dataset_name",
            "emb_type",
            "save_dir",
            "fold",
            "seed",
        ]:
            del model_config[key]
        # 允许通过命令行参数覆盖配置文件中的batch_size和num_epochs
        if "batch_size" in params:
            train_config["batch_size"] = params["batch_size"]
        if "num_epochs" in params:
            train_config["num_epochs"] = params["num_epochs"]

    batch_size, num_epochs, optimizer = (
        train_config["batch_size"],
        train_config["num_epochs"],
        train_config["optimizer"],
    )

    # 加载数据集相关的配置文件
    with open("../configs/data_config.json") as fin:
        data_config = json.load(fin)
    # 优先使用数据配置文件中指定的maxlen
    if "maxlen" in data_config[dataset_name]:
        train_config["seq_len"] = data_config[dataset_name]["maxlen"]
    seq_len = train_config["seq_len"]

    print("开始初始化数据集")
    print(
        f"数据集: {dataset_name}, 模型: {model_name}, 数据配置: {data_config[dataset_name]}, 折数: {fold}, 批大小: {batch_size}"
    )

    debug_print(text="初始化数据集加载器", fuc_name="main")
    # 根据模型名称初始化数据集加载器
    if model_name not in ["dimkt"]:
        train_loader, valid_loader, *_ = init_dataset4train(
            dataset_name, model_name, data_config, fold, batch_size
        )
    else:
        # dimkt模型需要额外的难度等级参数
        diff_level = params["difficult_levels"]
        train_loader, valid_loader, *_ = init_dataset4train(
            dataset_name,
            model_name,
            data_config,
            fold,
            batch_size,
            diff_level=diff_level,
        )

    # 将参数字典拼接成字符串，用于创建唯一的模型保存路径
    params_str = "_".join(
        [str(v) for k, v in params.items() if not k in ["other_config"]]
    )

    print(f"所有参数: {params}, 用于路径的参数字符串: {params_str}")
    # 如果使用wandb，可以添加一个UUID来确保每次运行的路径都是唯一的
    if params["add_uuid"] == 1 and params["use_wandb"] == 1:
        import uuid

        params_str = params_str + f"_{str(uuid.uuid4())}"

    # 创建模型检查点(checkpoint)的保存路径
    ckpt_path = os.path.join(save_dir, params_str)
    if not os.path.isdir(ckpt_path):
        os.makedirs(ckpt_path)

    print(
        f"开始训练模型: {model_name}, embedding类型: {emb_type}, 保存目录: {ckpt_path}, 数据集: {dataset_name}"
    )
    print(f"模型配置: {model_config}")
    print(f"训练配置: {train_config}")

    # dimkt模型的特殊处理
    if model_name in ["dimkt"]:
        del model_config["weight_decay"]

    # 保存本次运行的所有配置
    save_config(
        train_config, model_config, data_config[dataset_name], params, ckpt_path
    )

    learning_rate = params["learning_rate"]
    # 从模型配置中移除不属于模型超参的项
    for remove_item in ["use_wandb", "learning_rate", "add_uuid", "l2"]:
        if remove_item in model_config:
            del model_config[remove_item]
    # 一些模型需要序列长度作为参数
    if model_name in [
        "saint",
        "saint++",
        "sakt",
        "atdkt",
        "simplekt",
        "stablekt",
        "bakt_time",
        "folibikt",
    ]:
        model_config["seq_len"] = seq_len

    debug_print(text="初始化模型", fuc_name="main")
    print(f"模型名称: {model_name}")
    model = init_model(model_name, model_config, data_config[dataset_name], emb_type)
    print(f"初始化的模型实例: {model}")

    # 根据模型和优化器类型配置优化器
    if model_name == "hawkes":
        # hawkes模型对权重和偏置使用不同的衰减策略
        weight_p, bias_p = [], []
        for name, p in filter(lambda x: x[1].requires_grad, model.named_parameters()):
            if "bias" in name:
                bias_p.append(p)
            else:
                weight_p.append(p)
        optdict = [{"params": weight_p}, {"params": bias_p, "weight_decay": 0}]
        opt = torch.optim.Adam(optdict, lr=learning_rate, weight_decay=params["l2"])
    elif model_name == "iekt":
        opt = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-6)
    elif model_name == "dtransformer":
        print(f"dtransformer模型使用 weight_decay = 1e-5")
        opt = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    elif model_name == "dimkt":
        opt = torch.optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=params["weight_decay"]
        )
    else:
        # 通用优化器配置
        if optimizer == "sgd":
            opt = SGD(model.parameters(), learning_rate, momentum=0.9)
        elif optimizer == "adam":
            opt = Adam(model.parameters(), learning_rate)

    # 初始化性能指标
    testauc, testacc = -1, -1
    window_testauc, window_testacc = -1, -1
    validauc, validacc = -1, -1
    best_epoch = -1
    save_model = True

    debug_print(text="开始模型训练", fuc_name="main")

    # 调用核心训练函数
    if model_name == "rkt":
        # rkt模型需要额外的参数
        (
            testauc,
            testacc,
            window_testauc,
            window_testacc,
            validauc,
            validacc,
            best_epoch,
        ) = train_model(
            model,
            train_loader,
            valid_loader,
            num_epochs,
            opt,
            ckpt_path,
            None,
            None,
            save_model,
            data_config[dataset_name],
            fold,
        )
    else:
        (
            testauc,
            testacc,
            window_testauc,
            window_testacc,
            validauc,
            validacc,
            best_epoch,
        ) = train_model(
            model,
            train_loader,
            valid_loader,
            num_epochs,
            opt,
            ckpt_path,
            None,
            None,
            save_model,
        )

    # 如果保存了模型，加载性能最好的模型
    if save_model:
        best_model = init_model(
            model_name, model_config, data_config[dataset_name], emb_type
        )
        net = torch.load(os.path.join(ckpt_path, emb_type + "_model.ckpt"))
        best_model.load_state_dict(net)

    # 打印最终的性能报告表头
    print(
        "折数\t模型\temb类型\t测试AUC\t测试ACC\t窗口测试AUC\t窗口测试ACC\t验证AUC\t验证ACC\t最佳epoch"
    )
    # 打印性能数据
    print(
        str(fold)
        + "\t"
        + model_name
        + "\t"
        + emb_type
        + "\t"
        + str(round(testauc, 4))
        + "\t"
        + str(round(testacc, 4))
        + "\t"
        + str(round(window_testauc, 4))
        + "\t"
        + str(round(window_testacc, 4))
        + "\t"
        + str(validauc)
        + "\t"
        + str(validacc)
        + "\t"
        + str(best_epoch)
    )
    model_save_path = os.path.join(ckpt_path, emb_type + "_model.ckpt")
    print(f"训练结束时间: {datetime.datetime.now()}")

    # 如果使用wandb，记录最终结果
    if params["use_wandb"] == 1:
        wandb.log(
            {
                "validauc": validauc,
                "validacc": validacc,
                "best_epoch": best_epoch,
                "model_save_path": model_save_path,
            }
        )
