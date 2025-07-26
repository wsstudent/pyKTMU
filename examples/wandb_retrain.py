import os
import argparse
import json

import torch

# Set number of threads for PyTorch
torch.set_num_threads(4)
from torch.optim import SGD, Adam
import copy

from pykt.models import train_model, evaluate, init_model
from pykt.utils import debug_print, set_seed
from pykt.datasets import init_dataset4train
import datetime

# Setup environment for CUDA debugging and reproducibility
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
device = "cpu" if not torch.cuda.is_available() else "cuda"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:2"


def save_config(train_config, model_config, data_config, params, save_dir):
    """
    Saves training, model, data configurations, and hyperparameters to a json file.
    """
    d = {
        "train_config": train_config,
        "model_config": model_config,
        "data_config": data_config,
        "params": params,
    }
    save_path = os.path.join(save_dir, "config.json")
    with open(save_path, "w") as fout:
        json.dump(d, fout, indent=4)


def main(params):
    """
    Main retraining function.
    """
    if "use_wandb" not in params:
        params["use_wandb"] = 1

    if params["use_wandb"] == 1:
        import wandb

        wandb.init()

    set_seed(params["seed"])

    model_name, dataset_name, fold, emb_type, save_dir = (
        params["model_name"],
        params["dataset_name"],
        params["fold"],
        params["emb_type"],
        params["save_dir"],
    )

    debug_print(text="Loading configuration files.", fuc_name="main")

    with open("../configs/kt_config.json") as f:
        config = json.load(f)
        train_config = config["train_config"]
        # Adjust batch_size for specific models to prevent OOM
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

        model_config = copy.deepcopy(params)
        for key in [
            "model_name",
            "dataset_name",
            "emb_type",
            "save_dir",
            "fold",
            "seed",
        ]:
            del model_config[key]

        if "batch_size" in params:
            train_config["batch_size"] = params["batch_size"]
        if "num_epochs" in params:
            train_config["num_epochs"] = params["num_epochs"]

    batch_size, num_epochs, optimizer = (
        train_config["batch_size"],
        train_config["num_epochs"],
        train_config["optimizer"],
    )

    with open("../configs/data_config.json") as fin:
        data_config = json.load(fin)

    # === Core logic for retraining ===
    if params.get("retrain_from_scratch"):
        print("Retraining mode enabled. Modifying data loading path...")
        strategy = params["unlearn_strategy"]
        ratio = params["forget_ratio"]

        # Construct the filename for the retained dataset
        retrain_file_name = f"train_valid_sequences_retain_{strategy}_ratio{ratio}.csv"

        # Check if the file exists
        retrain_file_path = os.path.join(
            data_config[dataset_name]["dpath"], retrain_file_name
        )
        if not os.path.exists(retrain_file_path):
            raise FileNotFoundError(
                f"The specified retrain file does not exist: {retrain_file_path}"
            )

        # Overwrite the default training file with the retain set file
        data_config[dataset_name]["train_valid_file"] = retrain_file_name
        print(f"Switched training data to: {retrain_file_name}")
    # =================================

    if "maxlen" in data_config[dataset_name]:
        train_config["seq_len"] = data_config[dataset_name]["maxlen"]
    seq_len = train_config["seq_len"]

    print("Initializing dataset")
    print(
        f"Dataset: {dataset_name}, Model: {model_name}, Data Config: {data_config[dataset_name]}, Fold: {fold}, Batch Size: {batch_size}"
    )

    debug_print(text="Initializing data loaders", fuc_name="main")
    if model_name not in ["dimkt"]:
        train_loader, valid_loader, *_ = init_dataset4train(
            dataset_name, model_name, data_config, fold, batch_size
        )
    else:
        diff_level = params["difficult_levels"]
        train_loader, valid_loader, *_ = init_dataset4train(
            dataset_name,
            model_name,
            data_config,
            fold,
            batch_size,
            diff_level=diff_level,
        )

    params_str = "_".join(
        [str(v) for k, v in params.items() if not k in ["other_config"]]
    )
    if params["add_uuid"] == 1 and params["use_wandb"] == 1:
        import uuid

        params_str = params_str + f"_{str(uuid.uuid4())}"

    ckpt_path = os.path.join(save_dir, params_str)
    if not os.path.isdir(ckpt_path):
        os.makedirs(ckpt_path)

    print(
        f"Start training model: {model_name}, emb_type: {emb_type}, save_dir: {ckpt_path}, dataset_name: {dataset_name}"
    )
    print(f"model_config: {model_config}")
    print(f"train_config: {train_config}")

    if model_name in ["dimkt"]:
        del model_config["weight_decay"]

    save_config(
        train_config, model_config, data_config[dataset_name], params, ckpt_path
    )

    learning_rate = params["learning_rate"]
    for remove_item in [
        "use_wandb",
        "learning_rate",
        "add_uuid",
        "l2",
        "retrain_from_scratch",
        "unlearn_strategy",
        "forget_ratio",
    ]:
        if remove_item in model_config:
            del model_config[remove_item]

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

    debug_print(text="Initializing model", fuc_name="main")
    print(f"model_name: {model_name}")
    model = init_model(model_name, model_config, data_config[dataset_name], emb_type)
    print(f"model: {model}")

    if model_name == "hawkes":
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
        print(f"dtransformer with weight_decay=1e-5")
        opt = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    elif model_name == "dimkt":
        opt = torch.optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=params["weight_decay"]
        )
    else:
        if optimizer == "sgd":
            opt = SGD(model.parameters(), learning_rate, momentum=0.9)
        elif optimizer == "adam":
            opt = Adam(model.parameters(), learning_rate)

    testauc, testacc = -1, -1
    window_testauc, window_testacc = -1, -1
    validauc, validacc = -1, -1
    best_epoch = -1
    save_model = True

    debug_print(text="Training model", fuc_name="main")
    if model_name == "rkt":
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

    if save_model:
        best_model = init_model(
            model_name, model_config, data_config[dataset_name], emb_type
        )
        net = torch.load(os.path.join(ckpt_path, emb_type + "_model.ckpt"))
        best_model.load_state_dict(net)

    print(
        "fold\tmodel\temb_type\ttest_auc\ttest_acc\twindow_test_auc\twindow_test_acc\tvalid_auc\tvalid_acc\tbest_epoch"
    )
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
    print(f"end time: {datetime.datetime.now()}")

    if params["use_wandb"] == 1:
        wandb.log(
            {
                "validauc": validauc,
                "validacc": validacc,
                "best_epoch": best_epoch,
                "model_save_path": model_save_path,
            }
        )
