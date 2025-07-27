import os, sys
import json

from torch.utils.data import DataLoader
import numpy as np
from .data_loader import KTDataset
from .dkt_forget_dataloader import DktForgetDataset
from .atdkt_dataloader import ATDKTDataset
from .lpkt_dataloader import LPKTDataset
from .lpkt_utils import generate_time2idx
from .que_data_loader import KTQueDataset
from pykt.config import que_type_models
from .dimkt_dataloader import DIMKTDataset
from .que_data_loader_promptkt import KTQueDataset_promptKT
from .pretrain_utils import get_pretrain_data


def init_test_datasets(
    data_config, model_name, batch_size, diff_level=None, args=None, re_mapping=False
):
    """
    初始化用于测试的数据集和数据加载器。

    Args:
        data_config (dict): 数据集配置。
        model_name (str): 模型名称。
        batch_size (int): 批处理大小。
        diff_level (any, optional): 难度等级，用于DIMKT等模型。
        args (any, optional): 其他命令行参数。
        re_mapping (bool, optional): 是否重新映射。

    Returns:
        tuple: (test_loader, test_window_loader, test_question_loader, test_question_window_loader)
    """
    # 获取数据集名称
    dataset_name = data_config["dataset_name"]
    # 打印模型和数据集名称以供调试
    print(f"模型名称是 {model_name}, 数据集名称是 {dataset_name}")
    test_question_loader, test_question_window_loader = None, None

    # 根据不同的模型名称，使用不同的Dataset类来加载数据
    if model_name in ["dkt_forget", "bakt_time"]:
        # 为 dkt_forget, bakt_time 模型加载测试集
        test_dataset = DktForgetDataset(
            os.path.join(data_config["dpath"], data_config["test_file"]),
            data_config["input_type"],
            {-1},
        )
        test_window_dataset = DktForgetDataset(
            os.path.join(data_config["dpath"], data_config["test_window_file"]),
            data_config["input_type"],
            {-1},
        )
        # 如果存在问题级别的测试文件，也一并加载
        if "test_question_file" in data_config:
            test_question_dataset = DktForgetDataset(
                os.path.join(data_config["dpath"], data_config["test_question_file"]),
                data_config["input_type"],
                {-1},
                True,
            )
            test_question_window_dataset = DktForgetDataset(
                os.path.join(
                    data_config["dpath"], data_config["test_question_window_file"]
                ),
                data_config["input_type"],
                {-1},
                True,
            )
    elif model_name in ["lpkt"]:
        # 为 lpkt 模型加载测试集，需要额外生成时间相关的索引
        print(f"模型是lpkt")
        at2idx, it2idx = generate_time2idx(data_config)
        test_dataset = LPKTDataset(
            os.path.join(data_config["dpath"], data_config["test_file_quelevel"]),
            at2idx,
            it2idx,
            data_config["input_type"],
            {-1},
        )
        test_window_dataset = LPKTDataset(
            os.path.join(
                data_config["dpath"], data_config["test_window_file_quelevel"]
            ),
            at2idx,
            it2idx,
            data_config["input_type"],
            {-1},
        )
        test_question_dataset = None
        test_question_window_dataset = None
    elif model_name in ["rkt"] and dataset_name in ["statics2011", "assist2015", "poj"]:
        # 为特定的 rkt 模型和数据集组合加载测试集
        test_dataset = KTDataset(
            os.path.join(data_config["dpath"], data_config["test_file"]),
            data_config["input_type"],
            {-1},
        )
        test_window_dataset = KTDataset(
            os.path.join(data_config["dpath"], data_config["test_window_file"]),
            data_config["input_type"],
            {-1},
        )
        if "test_question_file" in data_config:
            test_question_dataset = KTDataset(
                os.path.join(data_config["dpath"], data_config["test_question_file"]),
                data_config["input_type"],
                {-1},
                True,
            )
            test_question_window_dataset = KTDataset(
                os.path.join(
                    data_config["dpath"], data_config["test_question_window_file"]
                ),
                data_config["input_type"],
                {-1},
                True,
            )
    elif model_name in que_type_models:
        # 为需要问题信息的模型加载测试集
        if model_name not in ["promptkt", "unikt"]:
            test_dataset = KTQueDataset(
                os.path.join(data_config["dpath"], data_config["test_file_quelevel"]),
                input_type=data_config["input_type"],
                folds=[-1],
                concept_num=data_config["num_c"],
                max_concepts=data_config["max_concepts"],
            )
            test_window_dataset = KTQueDataset(
                os.path.join(
                    data_config["dpath"], data_config["test_window_file_quelevel"]
                ),
                input_type=data_config["input_type"],
                folds=[-1],
                concept_num=data_config["num_c"],
                max_concepts=data_config["max_concepts"],
            )
        else:
            # 为 promptkt, unikt 等特殊模型加载测试集
            dataset = data_config["dpath"].split("/")[-1]
            if dataset == "":
                dataset = data_config["dpath"].split("/")[-2]
            if dataset in [
                "assist2009",
                "algebra2005",
                "bridge2algebra2006",
                "nips_task34",
                "ednet",
                "peiyou",
                "ednet5w",
                "ednet_all",
            ]:
                test_dataset = KTQueDataset_promptKT(
                    os.path.join(
                        data_config["dpath"],
                        data_config["test_file_quelevel"],
                    ),
                    input_type=data_config["input_type"],
                    folds=[-1],
                    concept_num=data_config["num_c"],
                    max_concepts=data_config["max_concepts"],
                    dataset_name=args.dataset_name,
                )
                test_path = os.path.join(
                    data_config["dpath"],
                    data_config["test_window_file_quelevel"],
                )
                if not os.path.exists(test_path):
                    print("文件不存在")
                    sys.exit(1)
                test_window_dataset = KTQueDataset_promptKT(
                    test_path,
                    input_type=data_config["input_type"],
                    folds=[-1],
                    concept_num=data_config["num_c"],
                    max_concepts=data_config["max_concepts"],
                    dataset_name=args.dataset_name,
                )
        test_question_dataset = None
        test_question_window_dataset = None
    elif model_name in ["atdkt"]:
        # 为 atdkt 模型加载测试集
        test_dataset = ATDKTDataset(
            os.path.join(data_config["dpath"], data_config["test_file"]),
            data_config["input_type"],
            {-1},
        )
        test_window_dataset = ATDKTDataset(
            os.path.join(data_config["dpath"], data_config["test_window_file"]),
            data_config["input_type"],
            {-1},
        )
        if "test_question_file" in data_config:
            test_question_dataset = ATDKTDataset(
                os.path.join(data_config["dpath"], data_config["test_question_file"]),
                data_config["input_type"],
                {-1},
                True,
            )
            test_question_window_dataset = ATDKTDataset(
                os.path.join(
                    data_config["dpath"], data_config["test_question_window_file"]
                ),
                data_config["input_type"],
                {-1},
                True,
            )
    elif model_name in ["dimkt"]:
        # 为 dimkt 模型加载测试集，需要额外的难度等级参数
        test_dataset = DIMKTDataset(
            data_config["dpath"],
            os.path.join(data_config["dpath"], data_config["test_file"]),
            data_config["input_type"],
            {-1},
            diff_level=diff_level,
        )
        test_window_dataset = DIMKTDataset(
            data_config["dpath"],
            os.path.join(data_config["dpath"], data_config["test_window_file"]),
            data_config["input_type"],
            {-1},
            diff_level=diff_level,
        )
        if "test_question_file" in data_config:
            test_question_dataset = DIMKTDataset(
                data_config["dpath"],
                os.path.join(data_config["dpath"], data_config["test_question_file"]),
                data_config["input_type"],
                {-1},
                True,
                diff_level=diff_level,
            )
            test_question_window_dataset = DIMKTDataset(
                data_config["dpath"],
                os.path.join(
                    data_config["dpath"], data_config["test_question_window_file"]
                ),
                data_config["input_type"],
                {-1},
                True,
                diff_level=diff_level,
            )
    else:
        # 默认情况下，使用通用的 KTDataset 加载测试集
        test_dataset = KTDataset(
            os.path.join(data_config["dpath"], data_config["test_file"]),
            data_config["input_type"],
            {-1},
        )
        test_window_dataset = KTDataset(
            os.path.join(data_config["dpath"], data_config["test_window_file"]),
            data_config["input_type"],
            {-1},
        )
        if "test_question_file" in data_config:
            test_question_dataset = KTDataset(
                os.path.join(data_config["dpath"], data_config["test_question_file"]),
                data_config["input_type"],
                {-1},
                True,
            )
            test_question_window_dataset = KTDataset(
                os.path.join(
                    data_config["dpath"], data_config["test_question_window_file"]
                ),
                data_config["input_type"],
                {-1},
                True,
            )

    # 将创建的 Dataset 封装成 DataLoader
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    test_window_loader = DataLoader(
        test_window_dataset, batch_size=batch_size, shuffle=False
    )
    if "test_question_file" in data_config:
        print(f"数据配置中包含test_question_file！")
        test_question_loader, test_question_window_loader = None, None
        if not test_question_dataset is None:
            test_question_loader = DataLoader(
                test_question_dataset, batch_size=batch_size, shuffle=False
            )
        if not test_question_window_dataset is None:
            test_question_window_loader = DataLoader(
                test_question_window_dataset, batch_size=batch_size, shuffle=False
            )

    return (
        test_loader,
        test_window_loader,
        test_question_loader,
        test_question_window_loader,
    )


def update_gap(max_rgap, max_sgap, max_pcount, cur):
    """辅助函数，用于更新 dkt_forget 模型所需的时间间隔和练习次数的最大值。"""
    max_rgap = cur.max_rgap if cur.max_rgap > max_rgap else max_rgap
    max_sgap = cur.max_sgap if cur.max_sgap > max_sgap else max_sgap
    max_pcount = cur.max_pcount if cur.max_pcount > max_pcount else max_pcount
    return max_rgap, max_sgap, max_pcount


def init_dataset4train(
    dataset_name,
    model_name,
    data_config,
    i,
    batch_size,
    diff_level=None,
    args=None,
    not_select_dataset=None,
    re_mapping=False,
):
    """
    初始化用于训练和验证的数据集和数据加载器。

    Args:
        dataset_name (str): 数据集名称。
        model_name (str): 模型名称。
        data_config (dict): 数据集配置。
        i (int): 当前的折数（fold），用于划分验证集。
        batch_size (int): 批处理大小。
        diff_level (any, optional): 难度等级，用于DIMKT等模型。
        args (any, optional): 其他命令行参数。
        not_select_dataset (any, optional): 不选择的数据集，用于promptkt。
        re_mapping (bool, optional): 是否重新映射。

    Returns:
        tuple: (train_loader, valid_loader)
    """
    print(f"数据集名称:{dataset_name}")
    print(f"数据配置:{data_config}")
    data_config = data_config[dataset_name]
    all_folds = set(data_config["folds"])

    # 根据模型名称选择不同的Dataset类
    if model_name in ["dkt_forget", "bakt_time"]:
        max_rgap, max_sgap, max_pcount = 0, 0, 0
        # 验证集使用第 i 折
        curvalid = DktForgetDataset(
            os.path.join(data_config["dpath"], data_config["train_valid_file"]),
            data_config["input_type"],
            {i},
        )
        # 训练集使用剩下的所有折
        curtrain = DktForgetDataset(
            os.path.join(data_config["dpath"], data_config["train_valid_file"]),
            data_config["input_type"],
            all_folds - {i},
        )
        # 更新时间间隔等最大值
        max_rgap, max_sgap, max_pcount = update_gap(
            max_rgap, max_sgap, max_pcount, curtrain
        )
        max_rgap, max_sgap, max_pcount = update_gap(
            max_rgap, max_sgap, max_pcount, curvalid
        )
    elif model_name == "lpkt":
        at2idx, it2idx = generate_time2idx(data_config)
        # # 以下是用于调试的代码，可以将生成的映射保存为json文件
        # json_str = json.dumps(at2idx)
        # with open('at2idx.json', 'w') as json_file:
        #     json_file.write(json_str)
        # json_str_2 = json.dumps(it2idx)
        # with open('it2idx.json', 'w') as json_file2:
        #     json_file2.write(json_str_2)
        curvalid = LPKTDataset(
            os.path.join(
                data_config["dpath"], data_config["train_valid_file_quelevel"]
            ),
            at2idx,
            it2idx,
            data_config["input_type"],
            {i},
        )
        curtrain = LPKTDataset(
            os.path.join(
                data_config["dpath"], data_config["train_valid_file_quelevel"]
            ),
            at2idx,
            it2idx,
            data_config["input_type"],
            all_folds - {i},
        )
    elif model_name in ["rkt"] and dataset_name in ["statics2011", "assist2015", "poj"]:
        curvalid = KTDataset(
            os.path.join(data_config["dpath"], data_config["train_valid_file"]),
            data_config["input_type"],
            {i},
        )
        curtrain = KTDataset(
            os.path.join(data_config["dpath"], data_config["train_valid_file"]),
            data_config["input_type"],
            all_folds - {i},
        )
    elif model_name in que_type_models:
        if model_name in ["promptkt"]:
            dataset_name = args.dataset_name
            train_ratio = args.dataset_name
            if args.train_mode == "pretrain":
                dpath = os.path.join(
                    data_config["dpath"],
                    f"train_valid_sequences_quelevel_pretrain_nomapping.csv",
                )
            else:
                dpath = os.path.join(
                    data_config["dpath"],
                    f"train_valid_sequences_quelevel.csv",
                )
            print(f"训练数据路径:{dpath}")
            if not os.path.exists(dpath) and args.train_mode == "pretrain":
                print(f"正在加载预训练数据")
                get_pretrain_data(data_config)
            curvalid = KTQueDataset_promptKT(
                dpath,
                input_type=data_config["input_type"],
                folds={i},
                concept_num=data_config["num_c"],
                max_concepts=data_config["max_concepts"],
                not_select_dataset=not_select_dataset,
                train_ratio=train_ratio,
                dataset_name=dataset_name,
            )
            curtrain = KTQueDataset_promptKT(
                dpath,
                input_type=data_config["input_type"],
                folds=all_folds - {i},
                concept_num=data_config["num_c"],
                max_concepts=data_config["max_concepts"],
                not_select_dataset=not_select_dataset,
                train_ratio=train_ratio,
                dataset_name=dataset_name,
            )
        else:
            curvalid = KTQueDataset(
                os.path.join(
                    data_config["dpath"], data_config["train_valid_file_quelevel"]
                ),
                input_type=data_config["input_type"],
                folds={i},
                concept_num=data_config["num_c"],
                max_concepts=data_config["max_concepts"],
            )
            curtrain = KTQueDataset(
                os.path.join(
                    data_config["dpath"], data_config["train_valid_file_quelevel"]
                ),
                input_type=data_config["input_type"],
                folds=all_folds - {i},
                concept_num=data_config["num_c"],
                max_concepts=data_config["max_concepts"],
            )
    elif model_name in ["atdkt"]:
        curvalid = ATDKTDataset(
            os.path.join(data_config["dpath"], data_config["train_valid_file"]),
            data_config["input_type"],
            {i},
        )
        curtrain = ATDKTDataset(
            os.path.join(data_config["dpath"], data_config["train_valid_file"]),
            data_config["input_type"],
            all_folds - {i},
        )
    elif model_name == "dimkt":
        curvalid = DIMKTDataset(
            data_config["dpath"],
            os.path.join(data_config["dpath"], data_config["train_valid_file"]),
            data_config["input_type"],
            {i},
            diff_level=diff_level,
        )
        curtrain = DIMKTDataset(
            data_config["dpath"],
            os.path.join(data_config["dpath"], data_config["train_valid_file"]),
            data_config["input_type"],
            all_folds - {i},
            diff_level=diff_level,
        )
    else:
        # 默认使用通用的 KTDataset
        curvalid = KTDataset(
            os.path.join(data_config["dpath"], data_config["train_valid_file"]),
            data_config["input_type"],
            {i},
        )
        curtrain = KTDataset(
            os.path.join(data_config["dpath"], data_config["train_valid_file"]),
            data_config["input_type"],
            all_folds - {i},
        )

    # 将 Dataset 封装成 DataLoader
    train_loader = DataLoader(curtrain, batch_size=batch_size)
    valid_loader = DataLoader(curvalid, batch_size=batch_size)

    # 尝试更新 dkt_forget 模型所需的全局配置
    try:
        if model_name in ["dkt_forget", "bakt_time"]:
            test_dataset = DktForgetDataset(
                os.path.join(data_config["dpath"], data_config["test_file"]),
                data_config["input_type"],
                {-1},
            )
            max_rgap, max_sgap, max_pcount = update_gap(
                max_rgap, max_sgap, max_pcount, test_dataset
            )
    except:
        pass

    # 将计算出的最大值存入配置字典，供模型初始化时使用
    if model_name in ["dkt_forget", "bakt_time"]:
        data_config["num_rgap"] = max_rgap + 1
        data_config["num_sgap"] = max_sgap + 1
        data_config["num_pcount"] = max_pcount + 1
    if model_name == "lpkt":
        print(f"答题时间间隔类别数:{len(at2idx)}")
        print(f"题目用时类别数:{len(it2idx)}")
        data_config["num_at"] = len(at2idx) + 1
        data_config["num_it"] = len(it2idx) + 1

    return train_loader, valid_loader

