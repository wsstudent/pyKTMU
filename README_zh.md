# pyKT-Unlearn

**集成机器遗忘功能的增强版知识追踪工具库**

本项目是 [pyKT](https://github.com/pykt-team/pykt-toolkit) 的二次开发版本。在继承原项目所有功能的基础上，本项目进行了**重构**与**功能扩展**，核心特性包括：

## 🔧 核心特性

- 🤖 **机器遗忘 (Machine Unlearning)**  
本工具库集成了论文  
**“NeuS: A Neural Suppression-based Unlearning Mechanism for Privacy-preserving Knowledge Tracing”**  
中提出与评估的四种代表性遗忘方法，适用于教育数据删除任务场景。

| `unlearn_method` 参数 | 对应方法名       | 方法简介 |
|----------------------|------------------|------------------------|
| `retrain`            | 重训练 Retraining | 使用保留集 $D_r$ 从头重新训练模型，被视为遗忘完整性的黄金标准。 |
| `finetune`           | 微调 Fine-tuning | 冻结部分层，仅在 $D_r$ 上微调非冻结参数，效率较高但可能遗忘不彻底。 |
| `surgical`           | NeuS（本研究方法） | 基于 Fisher 信息与保留集/遗忘集敏感度，精细抑制与遗忘集高度相关的参数。 |
| `gradient_ascent`    | Naïve Fisher     | 仅基于遗忘集 Fisher 分数进行梯度上升，作为对比基线分析遗忘敏感性。 |


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

支持的策略：
  - low_performance: 选择低表现用户进行遗忘
  - high_performance: 选择高表现用户进行遗忘  
  - low_engagement: 选择低参与度用户进行遗忘
  - unstable_performance: 选择表现不稳定的用户进行遗忘

---

### 2. 模型训练与机器遗忘

`examples/wandb_xxx_train.py` 是中央控制器，支持标准训练与遗忘任务。
下面的示例展示了dkt如何使用不同的遗忘方法进行模型训练与遗忘。
当训练其他模型时，使用相应的训练文件：`wandb_xxx_train.py`


#### A. 标准训练

```bash
python wandb_dkt_train.py \
    --dataset_name assist2009 \
    --save_dir saved_model \
    --fold 0 \
    --use_wandb 0
```

#### B. 机器遗忘

通过 `--unlearn_method` 选择遗忘策略：
```bash
##### 示例 1：Retrain（重训练）

```bash
python wandb_dkt_train.py \
    --dataset_name assist2009 \
    --unlearn_method retrain \
    --unlearn_strategy low_performance \
    --forget_ratio 0.2 \
    --save_dir saved_model/unlearning \
    --use_wandb 0
```

##### 示例 2：Surgical / Ascent / Finetune

```bash
python wandb_dkt_train.py \
    --dataset_name assist2009 \
    --unlearn_method surgical \
    --model_ckpt_path saved_model/dkt_assist2009_seed42_fold0 \
    --alpha 10.0 \
    --unlearn_strategy low_performance \
    --forget_ratio 0.2 \
    --save_dir saved_model/unlearning \
    --use_wandb 0
```

#### 关键参数说明：

* `--unlearn_method`: 遗忘方法（可选： `finetune`, `surgical`, `ascent`）
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
    --unlearn_strategy low_performance \
    --forget_ratio 0.2 \
    --unlearn_test_file forget \
    --use_wandb 0
```

#### 参数说明：

* `--save_dir`: 模型存储路径
* `--unlearn_strategy`: 数据划分策略（如 `low_performance`）
* `--forget_ratio`: 遗忘比例
* `--unlearn_test_file`: `forget` 表示遗忘集，`retain` 表示保留集

---
#### 隐私攻击评估
为评估遗忘方法的隐私保护能力，本项目支持两种攻击方式：

成员推理攻击（Membership Inference）
判断某学生是否出现在训练集中。
实现路径：pykt/utils/attacks/membership_inference.py

模型反演攻击（Model Inversion）
通过模型输出反推原始输入内容，可能泄露学生隐私。
实现路径：pykt/utils/attacks/model_inversion.py

✅ 这两种攻击可用于对比不同模型在遗忘前后的隐私泄露程度。

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
