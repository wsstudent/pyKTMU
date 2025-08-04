# pyKT-Unlearn

**An enhanced knowledge tracing toolkit with built-in machine unlearning**

This project is a secondary development of [pyKT](https://github.com/pykt-team/pykt-toolkit). On top of all features from the original project, this repository includes **refactoring** and **feature extensions**. Key highlights:

## 🔧 Key Features

- 🤖 **Machine Unlearning**  
Supports four representative unlearning approaches aligned with educational data deletion tasks,  
as proposed and evaluated in our paper:  
**NeuS: A Neural Suppression-based Unlearning Mechanism for Privacy-preserving Knowledge Tracing**.

| `unlearn_method` | Corresponding Method | Description |
|--------------------|----------------------|-------------|
| `retrain`          | Retraining           | Rebuild the model from scratch using only the retain set $D_r$; serves as the gold standard for unlearning completeness. |
| `finetune`         | Fine-tuning          | Freeze selected layers and fine-tune remaining parameters on $D_r$; efficient but may not fully erase sensitive information. |
| `surgical`         | NeuS (ours)          | A fine-grained method that suppresses parameters sensitive to the forget set, guided by Fisher Information and retain-aware sensitivity factors. |
| `gradient_ascent`  | Naïve Fisher         | Applies gradient ascent using Fisher scores computed on the forget set alone; a baseline for analyzing forgetting sensitivity. |

- 📊 **End-to-End Workflow**  
  Provides a **full set of scripts** covering data preprocessing (generating retain/forget splits), model training/unlearning, and final evaluation.

- ⚙️ **Modular Refactor**  
  The `wandb_train.py` and `wandb_predict.py` scripts are **fully refactored** for clearer structure, more robust logic, and easier extension.

- 💡 **Flexible Hyperparameter Configuration**  
  Unified argument management via CLI to control training and unlearning hyperparameters.

---

## 🛠️ Installation

This project uses [`uv`](https://github.com/astral-sh/uv) for package management.

### Create a virtual environment (recommended)

```bash
# Create a virtual environment named .venv
uv venv

# Activate the environment
# Windows
.\.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
````

### Install dependencies

All dependencies are defined in `pyproject.toml`. Install with:

```bash
uv pip install -e .
```
---

## 🚀 Quick Start

> **Please run the following commands from the `examples` directory.**

```bash
cd examples
```

### 1) Data Preprocessing

Use `examples/data_preprocess.py` to process raw data and generate the retain and forget splits for unlearning experiments.

```bash
python data_preprocess.py \
    --dataset_name assist2009 \
    --gen_forget_data \
    --forget_ratio 0.2 \
```

**Arguments:**

- `--dataset_name`: Dataset name (e.g., `assist2009`)
- `--gen_forget_data`: Whether to generate the forget/retain splits (presence of this flag enables split generation)
- `--forget_ratio`: Proportion to forget (e.g., `0.2` for 20%)

**Generated files include:**

- `train_valid_sequences_retain_{strategy}_ratio{ratio}.csv`
- `train_valid_sequences_forget_{strategy}_ratio{ratio}.csv`
- `test_sequences_retrain_{strategy}_ratio{ratio}.csv`
- `test_sequences_forget_{strategy}_ratio{ratio}.csv`

**Supported strategies:**
  
- `low_performance`: forget users with low performance  
- `high_performance`: forget users with high performance  
- `low_engagement`: forget users with low engagement  
- `unstable_performance`: forget users with unstable performance
- `random`: randomly select users to forget

---

### 2) Model Training & Machine Unlearning

`examples/wandb_xxx_train.py` serves as the central controller for both standard training and unlearning tasks.  
The examples below demonstrate how **DKT** uses various unlearning methods. For other models, switch to the corresponding training file: `wandb_xxx_train.py`.

#### A. Standard Training

```bash
python wandb_dkt_train.py \
    --dataset_name assist2009 \
    --save_dir saved_model \
    --fold 0 \
    --use_wandb 0
```

#### B. Machine Unlearning

Choose an unlearning strategy with `--unlearn_method`:

##### Example 1: `retrain`

```bash
python wandb_dkt_train.py \
    --dataset_name assist2009 \
    --unlearn_method retrain \
    --unlearn_strategy low_performance\
    --forget_ratio 0.2 \
    --save_dir saved_model/unlearning \
    --use_wandb 0
```

##### Example 2: `surgical` / `ascent` / `finetune`

```bash
python wandb_dkt_train.py \
    --dataset_name assist2009 \
    --unlearn_method surgical \
    --model_ckpt_path saved_model/dkt_assist2009_seed42_fold0 \
    --alpha 10.0 \
    --unlearn_strategy low_performance \
    --forget_ratio 0.2 \
    --save_dir saved_model/unlearning \
    --use_wandb 1
```

**Key arguments:**

- `--unlearn_method`: one of `{finetune, surgical, ascent}`
- `--model_ckpt_path`: path to a pretrained checkpoint (required for `finetune`/`surgical`/`ascent`)
- `--alpha`: unlearning strength (used by `surgical` / `ascent`)
- `--finetune_epochs`, `--finetune_lr`: epochs and learning rate for fine-tuning
- `--unlearn_strategy`, `--forget_ratio`: specify the split strategy and ratio

---

### 3) Evaluation

Use `examples/wandb_predict.py` to evaluate model performance on different test splits.

```bash
python wandb_predict.py \
    --save_dir saved_model/unlearning/surgical_dkt_assist2009... \
    --unlearn_strategy low_performance \
    --forget_ratio 0.2 \
    --unlearn_test_file forget \
    --use_wandb 1
```
**Arguments:**

- `--save_dir`: model directory
- `--unlearn_strategy`: data split strategy (e.g., `low_performance`)
- `--forget_ratio`: forget ratio
- `--unlearn_test_file`: `forget` for the forget set, `retain` for the retain set

---
> ✅ After each evaluation run, the results will be automatically saved to `../data/evaluation_results.csv`.  
> If the file already exists, new results will be **appended** without overwriting previous entries.
### 4) Privacy Risk Evaluation
To assess the privacy risks of different unlearning strategies, this toolkit supports two common attack methods:
Membership Inference Attack
Determines whether a specific student's data was used during model training, implemented in pykt/utils/attacks/membership_inference.py.
Model Inversion Attack
Attempts to reconstruct input interactions from the model’s outputs, revealing possible private information. Implementation: pykt/utils/attacks/model_inversion.py.

✅ These attacks help evaluate the privacy leakage of different models before and after unlearning.

## 📚 Citation

This project is based on `pyKT`. If you use this toolkit, please cite the original paper:

```bibtex
@inproceedings{liupykt2022,
  title={pyKT: A Python Library to Benchmark Deep Learning based Knowledge Tracing Models},
  author={Liu, Zitao and Liu, Qiongqiong and Chen, Jiahao and Huang, Shuyan and Tang, Jiliang and Luo, Weiqi},
  booktitle={Thirty-sixth Conference on Neural Information Processing Systems Datasets and Benchmarks Track},
  year={2022}
}
```
