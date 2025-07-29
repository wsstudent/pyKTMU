# pyKT-Unlearn

**集成机器遗忘功能的增强版知识追踪工具库**

本项目是 [pyKT](https://github.com/pykt-team/pykt-toolkit) 的二次开发版本。在继承原项目所有功能的基础上，本项目进行了**重构**与**功能扩展**，核心特性包括：

## 🔧 核心特性

- 🤖 **机器遗忘 (Machine Unlearning)**  
  集成多种前沿的机器遗忘算法，包括：
  - `Retrain`（重训练）
  - `Finetune`（微调）
  - `Surgical`（精准手术）
  - `Gradient Ascent`（梯度上升）  

- 📊 **端到端流程**  
  提供从数据预处理（生成遗忘集与保留集）、模型训练/遗忘到最终评估的**全套脚本**。


- ⚙️ **模块化重构**  
  对 `wandb_train.py` 和 `wandb_predict.py` 脚本进行**完全重构**，结构清晰、逻辑更稳、易扩展。

- 💡 **灵活的参数配置**  
  实现统一参数管理，支持通过命令行控制训练与遗忘的各项超参数。
---

## 🛠️ 安装

本项目使用 [`uv`](https://github.com/astral-sh/uv) 进行包管理。

### 创建虚拟环境（推荐）

```bash
# 创建名为 .venv 的虚拟环境
uv venv

# 激活环境
# Windows
.\.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
````

### 安装依赖

项目依赖已在 `pyproject.toml` 中定义，执行以下命令进行安装：

```bash
uv pip install -e .
```


---

## 🚀 快速开始
> ### 请先到exapmles文件夹下再执行以下指令

``` bash
cd examples
```

### 1. 数据预处理
使用 `examples\data_preprocess.py`处理原始数据，一键生成用于机器遗忘实验的保留集和遗忘集。

```bash
python data_preprocess.py \
    --dataset_name assist2009 \
    --gen_forget_data \
    --forget_ratio 0.2 \
```

#### 参数说明：

* `--dataset_name`: 处理的数据集名称（如 `assist2009`）
* `--gen_forget_data`: 是否生成遗忘数据集，有这个参数表示生成遗忘集和保留集
* `--forget_ratio`: 遗忘比例（如 `0.2` 表示划出 20% 的数据）

生成的文件包括：

* `train_valid_sequences_retain_{strategy}_ratio{ratio}.csv`
* `train_valid_sequences_forget_{strategy}_ratio{ratio}.csv`
* `test_sequences_retrain_{strategy}_ratio{ratio}.csv`
* `test_sequences_forget_{strategy}_ratio{ratio}.csv`
其中遗忘策略有5种：`random`, `sequential`, `random_sequential`, `sequential_random`, `sequential_sequential`。

---

### 2. 模型训练与机器遗忘

`examples/wandb_train.py` 是中央控制器，支持标准训练与遗忘任务。

#### A. 标准训练

```bash
python wandb_train.py \
    --model_name dkt \
    --dataset_name assist2009 \
    --save_dir saved_model \
    --seed 42 \
    --fold 0 \
    --use_wandb 1
```

#### B. 机器遗忘

通过 `--unlearn_method` 选择遗忘策略：
下面的示例展示了dkt如何使用不同的遗忘方法进行模型训练与遗忘。
当训练其他模型时，使用相应的训练文件：`wandb_xxx_train.py`

```bash
##### 示例 1：Retrain（重训练）

```bash
python wandb_dkt_train.py \
    --dataset_name assist2009 \
    --unlearn_method retrain \
    --unlearn_strategy random \
    --forget_ratio 0.2 \
    --save_dir saved_model/unlearning \
    --use_wandb 1
```

##### 示例 2：Surgical / Ascent / Finetune

```bash
python wandb_dkt_train.py \
    --dataset_name assist2009 \
    --unlearn_method surgical \
    --model_ckpt_path saved_model/dkt_assist2009_seed42_fold0 \
    --alpha 10.0 \
    --unlearn_strategy random \
    --forget_ratio 0.2 \
    --save_dir saved_model/unlearning \
    --use_wandb 1
```

#### 关键参数说明：

* `--unlearn_method`: 遗忘方法（可选：`retrain`, `finetune`, `surgical`, `ascent`）
* `--model_ckpt_path`: 预训练模型路径（finetune/surgical/ascent 等遗忘方法必须带上此参数）
* `--alpha`: 遗忘强度（用于 surgical/ascent）
* `--finetune_epochs`, `--finetune_lr`: 微调轮数与学习率
* `--unlearn_strategy`, `--forget_ratio`: 选择对应的数据

---

### 3. 模型评估

使用 `examples/wandb_predict.py` 在不同测试集上评估模型性能。

```bash
python wandb_predict.py \
    --save_dir saved_model/unlearning/surgical_dkt_assist2009... \
    --unlearn_strategy random \
    --forget_ratio 0.2 \
    --unlearn_test_file forget \
    --use_wandb 1
```

#### 参数说明：

* `--save_dir`: 模型存储路径
* `--unlearn_strategy`: 数据划分策略（如 `random`）
* `--forget_ratio`: 遗忘比例
* `--unlearn_test_file`: `forget` 表示遗忘集，`retain` 表示保留集，留空则评估原始测试集

---

## 📚 引用

本项目基于 `pyKT`，如使用本工具库，请引用原始论文：

```bibtex
@inproceedings{liupykt2022,
  title={pyKT: A Python Library to Benchmark Deep Learning based Knowledge Tracing Models},
  author={Liu, Zitao and Liu, Qiongqiong and Chen, Jiahao and Huang, Shuyan and Tang, Jiliang and Luo, Weiqi},
  booktitle={Thirty-sixth Conference on Neural Information Processing Systems Datasets and Benchmarks Track},
  year={2022}
}
```
