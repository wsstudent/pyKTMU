import os
import argparse
import json
import torch
import copy
import datetime
import uuid

from torch.optim import SGD, Adam
from pykt.models import train_model, evaluate, init_model
from pykt.utils import debug_print, set_seed
from pykt.datasets import init_dataset4train
from pykt.utils.Fisher import Fisher

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
    "--forget_ratio", type=float, default=0.1, help="[retrain专用] 遗忘数据比例"
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


def get_dataloader(dataset_name, model_name, data_config, fold, batch_size, data_type):
    """根据数据类型（retain, forget, test）获取数据加载器"""
    temp_config = copy.deepcopy(data_config)
    if data_type in ["retain", "forget"]:
        file_key = f"{data_type}_file"
        if file_key not in temp_config[dataset_name]:
            raise ValueError(
                f"请在 data_config.json 的 '{dataset_name}' 中定义 '{file_key}'"
            )
        temp_config[dataset_name]["train_valid_file"] = temp_config[dataset_name][
            file_key
        ]
        loader, _ = init_dataset4train(
            dataset_name, model_name, temp_config, fold=0, batch_size=batch_size
        )
        return loader
    elif data_type == "test":
        _, _, loader = init_dataset4train(
            dataset_name, model_name, temp_config, -1, batch_size
        )
        return loader
    raise ValueError(f"不支持的数据类型: {data_type}")


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

    print(f"【训练与评估完成】")
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
    print("✨ 执行任务：[遗忘方法 - 从头重训练]")
    retrain_file_name = f"train_valid_sequences_retain_{params['unlearn_strategy']}_ratio{params['forget_ratio']}.csv"
    retrain_file_path = os.path.join(
        data_config[params["dataset_name"]]["dpath"], retrain_file_name
    )
    if not os.path.exists(retrain_file_path):
        raise FileNotFoundError(f"指定的重训练文件不存在: {retrain_file_path}")
    data_config[params["dataset_name"]]["train_valid_file"] = retrain_file_name
    print(f"已切换训练数据至: {retrain_file_name}")
    run_standard_training(params, data_config, train_config)


def run_unlearning_fisher(params, data_config, train_config):
    print("✨ 执行任务：[遗忘方法 - Fisher Information]")
    model_name, dataset_name, fold, emb_type, save_dir = (
        params["model_name"],
        params["dataset_name"],
        params["fold"],
        params["emb_type"],
        params["save_dir"],
    )
    if not params.get("model_ckpt_path"):
        raise ValueError("使用Fisher遗忘方法时，必须提供 --model_ckpt_path 参数。")
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
    original_model.eval()
    batch_size = params.get("batch_size", 64)
    retain_loader = get_dataloader(
        dataset_name, model_name, data_config, fold, batch_size, "retain"
    )
    forget_loader = get_dataloader(
        dataset_name, model_name, data_config, fold, batch_size, "forget"
    )
    test_loader = get_dataloader(
        dataset_name, model_name, data_config, fold, batch_size, "test"
    )
    pre_test_auc, pre_test_acc = evaluate(original_model, test_loader, model_name)
    print(f"【遗忘前】 => 测试集 AUC: {pre_test_auc:.4f}, ACC: {pre_test_acc:.4f}")
    fisher_unlearner = Fisher(model_to_wrap=original_model, model_name=model_name)
    fisher_unlearner.to(device)

    class TempDataHandler:
        def __init__(self, retain_loader, forget_loader):
            self.loaders = {"retain": retain_loader, "forget": forget_loader}

        def get_data_loader(self, split, shuffle=True):
            return self.loaders.get(split)

    fisher_unlearner.unlearn(
        data_handler=TempDataHandler(retain_loader, forget_loader),
        alpha=params.get("alpha", 10.0),
        device=device,
    )
    unlearned_model = fisher_unlearner.model
    post_test_auc, post_test_acc = evaluate(unlearned_model, test_loader, model_name)
    post_forget_auc, post_forget_acc = evaluate(
        unlearned_model, forget_loader, model_name
    )
    print(f"【遗忘后】 => 测试集 AUC: {post_test_auc:.4f}, ACC: {post_test_acc:.4f}")
    print(
        f"【遗忘后】 => 遗忘集 AUC: {post_forget_auc:.4f}, ACC: {post_forget_acc:.4f}"
    )
    params_str = f"fisher_{model_name}_{dataset_name}_alpha{params['alpha']}_seed{params['seed']}"
    if params.get("add_uuid", 0) == 1:
        params_str += f"_{str(uuid.uuid4())[:8]}"
    ckpt_path = os.path.join(save_dir, params_str)
    if not os.path.isdir(ckpt_path):
        os.makedirs(ckpt_path)
    print(f"遗忘后的模型和日志将保存至: {ckpt_path}")
    model_save_path = os.path.join(ckpt_path, f"{emb_type}_model.ckpt")
    torch.save(unlearned_model.state_dict(), model_save_path)
    print(f"遗忘后的模型权重已保存至: {model_save_path}")
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
    print(f"遗忘后模型的配置文件 (config.json) 已保存。")
    if params.get("use_wandb", 0) == 1:
        import wandb

        wandb.log(
            {
                "pre_test_auc": pre_test_auc,
                "post_test_auc": post_test_auc,
                "post_forget_auc": post_forget_auc,
                "model_save_path": ckpt_path,
            }
        )


# ————————————————————————————————
# 统一的主函数和分发逻辑
# ————————————————————————————————
def main(params):
    if params.get("use_wandb", 1) == 1:
        import wandb

        wandb_config = {k: v for k, v in params.items() if v is not None}
        wandb.init(config=wandb_config, project="pykt-unlearn-project")
    set_seed(params.get("seed", 42))
    with open("../configs/kt_config.json") as f:
        config_from_json = json.load(f)
    with open("../configs/data_config.json") as fin:
        data_config = json.load(fin)
    model_name = params["model_name"]
    final_params = config_from_json.get("train_config", {}).copy()
    model_specific_config = config_from_json.get(model_name, {})
    final_params.update(model_specific_config)
    explicit_params = {k: v for k, v in params.items() if v is not None}
    final_params.update(explicit_params)
    params = final_params
    print(f"最终生效的参数配置: {json.dumps(params, indent=4)}")
    train_config = config_from_json.get("train_config", {})
    train_config["batch_size"] = params.get("batch_size", train_config["batch_size"])
    train_config["num_epochs"] = params.get("num_epochs", train_config["num_epochs"])
    unlearn_method = params.get("unlearn_method")
    if unlearn_method is None:
        run_standard_training(params, data_config, train_config)
    else:
        UNLEARN_HANDLERS = {
            "retrain": run_unlearning_retrain,
            "fisher": run_unlearning_fisher,
        }
        handler = UNLEARN_HANDLERS.get(unlearn_method)
        if handler:
            handler(params=params, data_config=data_config, train_config=train_config)
        else:
            available = ", ".join(UNLEARN_HANDLERS.keys())
            raise ValueError(f"不支持的遗忘方法: '{unlearn_method}'。可用: {available}")
