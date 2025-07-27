import os
import argparse
import json
import copy
from pykt.config import que_type_models
import torch

# 设置PyTorch使用的线程数，以优化在CPU上的并行性能
torch.set_num_threads(2)

# 从pykt库中导入核心函数
from pykt.models import (
    evaluate_splitpred_question,
    load_model,
    lpkt_evaluate_multi_ahead,
)


def main(params):
    # 根据命令行参数决定是否初始化wandb进行实验跟踪
    if params["use_wandb"] == 1:
        import wandb

        wandb.init()

    # 从参数字典中获取核心变量
    save_dir, use_pred, ratio = (
        params["save_dir"],
        params["use_pred"],
        params["train_ratio"],
    )

    # --- 1. 加载已保存的配置 ---
    # 这个脚本是用来评估一个已经训练好的模型，所以需要从模型的保存目录加载其配置
    print(f"正在从目录 '{save_dir}' 加载配置文件...")
    with open(os.path.join(save_dir, "config.json")) as fin:
        config = json.load(fin)
        # 深拷贝模型配置，以防意外修改
        model_config = copy.deepcopy(config["model_config"])

        # 定义一个列表，包含所有不属于模型超参数的“流程控制参数”
        non_model_keys = [
            "use_wandb",
            "learning_rate",
            "add_uuid",
            "l2",
            "retrain_from_scratch",
            "unlearn_strategy",
            "forget_ratio",
            "unlearn_method",
            "model_ckpt_path",
            "alpha",
            "num_epochs",
            "batch_size",
            "optimizer",
            "seq_len",
        ]
        # 遍历这个列表，从 model_config 中安全地移除它们
        print("正在清理模型配置参数...")
        for key in non_model_keys:
            if key in model_config:
                del model_config[key]
                print(f"  - 已移除参数: {key}")

        # 从配置中恢复训练时使用的参数
        trained_params = config["params"]
        model_name, dataset_name, emb_type = (
            trained_params["model_name"],
            trained_params["dataset_name"],
            trained_params["emb_type"],
        )
        seq_len = config["train_config"]["seq_len"]

        # 部分模型在初始化时需要显式地传入序列长度
        if model_name in ["saint", "sakt", "atdkt"]:
            model_config["seq_len"] = seq_len

        # 获取数据配置
        data_config = config["data_config"]

    # --- 2. 打印配置信息用于调试 ---
    print(f"--- 开始评估 ---")
    print(f"模型名称: {model_name}, Embedding类型: {emb_type}, 数据集: {dataset_name}")
    print(f"模型保存目录: {save_dir}")
    print(f"模型配置: {model_config}")
    print(f"数据配置: {data_config}")

    # 将整数参数转换为布尔值
    use_pred = True if use_pred == 1 else False
    atkt_pad = True if params["atkt_pad"] == 1 else False

    # --- 3. 加载模型 ---
    # 使用保存的配置和权重来加载模型实例
    print("正在加载模型权重...")
    model = load_model(model_name, model_config, data_config, emb_type, save_dir)
    print("模型加载成功！")

    # --- 4. 执行分割预测评估 ---
    print(
        f"开始执行分割预测, 使用历史答案(use_pred): {use_pred}, 观察比例(ratio): {ratio}..."
    )

    # 构造保存预测结果的文件路径
    save_test_path = os.path.join(
        save_dir, f"{model.emb_type}_test_ratio{ratio}_{use_pred}_predictions.txt"
    )

    # 构造待评估数据文件的完整路径
    testf = os.path.join(data_config["dpath"], params["test_filename"])
    print(f"待评估文件路径: {testf}")

    # 根据模型名称，调用不同的评估函数
    if model_name in que_type_models and model_name != "lpkt":
        print("调用 evaluate_multi_ahead 评估函数...")
        dfinal = model.evaluate_multi_ahead(
            data_config, batch_size=16, ob_portions=ratio, accumulative=use_pred
        )
    elif model_name == "lpkt":
        print("调用 lpkt_evaluate_multi_ahead 评估函数...")
        dfinal = lpkt_evaluate_multi_ahead(
            model, data_config, batch_size=64, ob_portions=ratio, accumulative=use_pred
        )
    else:
        print("调用 evaluate_splitpred_question 评估函数...")
        dfinal = evaluate_splitpred_question(
            model,
            data_config,
            testf,
            model_name,
            save_test_path,
            use_pred,
            ratio,
            atkt_pad,
        )

    # --- 5. 输出和记录结果 ---
    print("--- 评估完成，输出结果 ---")
    for key in dfinal:
        print(f"{key}: {dfinal[key]}")

    # 将原始的训练参数也合并到结果中，方便追溯
    dfinal.update(config["params"])

    # 如果启用，将最终结果记录到wandb
    if params["use_wandb"] == 1:
        print("正在将结果记录到 wandb...")
        wandb.log(dfinal)
    print("评估脚本执行完毕。")


# 当这个文件作为主程序运行时，会执行以下代码
if __name__ == "__main__":
    # 创建一个命令行参数解析器
    parser = argparse.ArgumentParser()
    # 定义所有可接受的命令行参数
    parser.add_argument(
        "--save_dir", type=str, default="saved_model", help="已训练模型的保存目录"
    )
    parser.add_argument(
        "--test_filename",
        type=str,
        default="test.csv",
        help="要评估的数据文件名，例如 test.csv 或 forget_set.csv",
    )
    parser.add_argument(
        "--use_pred",
        type=int,
        default=0,
        help="是否使用模型对历史信息的预测来代替真实答案 (0: False, 1: True)",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.9,
        help="在分割预测中，用作'观察历史'的序列比例",
    )
    parser.add_argument(
        "--atkt_pad",
        type=int,
        default=0,
        help="针对ATKT模型的特殊padding设置 (0: False, 1: True)",
    )
    parser.add_argument(
        "--use_wandb",
        type=int,
        default=1,
        help="是否使用wandb记录实验 (0: False, 1: True)",
    )

    # 解析命令行传入的参数
    args = parser.parse_args()

    # 打印所有传入的参数，方便调试
    print("传入的命令行参数: ", args)

    # 将解析后的参数对象转换为字典格式
    params = vars(args)

    # 调用主函数，开始执行
    main(params)
