# 文件路径: pykt/utils/Unlearner.py

import torch
import torch.nn as nn
import torch.optim as optim
import time
from pykt.models.train_model import model_forward
from pykt.utils.utils import prepare_model_optimizer


class Unlearner:
    """
    一个集成了多种遗忘策略的、通用的遗忘处理器。
    所有方法都基于通用的 `model_forward` 函数，以实现对多种模型的广泛兼容。
    """

    def __init__(self, model, model_name, params, optimizer_type="adam"):
        """
        初始化 Unlearner。
        Args:
            model: 需要进行遗忘操作的已训练好的模型。
            model_name (str): 模型的名称，用于 `model_forward` 函数选择正确的处理逻辑。
        """
        if not hasattr(model, "model_name"):
            model.model_name = model_name
        self.model = model
        self.fisher_dict = {}
        self.params = params
        self.optimizer_type = optimizer_type

    def unlearn(
        self,
        method,
        forget_loader=None,
        retain_loader=None,
        alpha=1.0,
        device="cuda",
        **kwargs,
    ):
        """
        执行机器学习遗忘的统一入口。
        """
        print(f"Unlearner: 执行遗忘策略 '{method}'")

        if method == "surgical":
            if retain_loader is None or forget_loader is None:
                raise ValueError(
                    "使用 'surgical' 方法时，必须同时提供 `retain_loader` 和 `forget_loader`。"
                )
            self._execute_surgical(retain_loader, forget_loader, alpha, device)

        elif method == "ascent":
            if forget_loader is None:
                raise ValueError("使用 'ascent' 方法时，必须提供 `forget_loader`。")
            self._execute_ascent(forget_loader, alpha, device)

        elif method == "finetune":
            if retain_loader is None:
                raise ValueError("使用 'finetune' 方法时，必须提供 `retain_loader`。")
            self._execute_finetune(retain_loader, device, **kwargs)

        else:
            raise ValueError(
                f"未知的遗忘方法: '{method}'。可用选项为 'surgical', 'ascent', 'finetune'。"
            )

    # --- 私有辅助方法 ---
    def _execute_surgical(self, retain_loader, forget_loader, alpha, device):
        print("步骤 1/3: 正在保留集上计算费雪信息...")
        start_time = time.time()  # Record the start time
        self._compute_fisher(retain_loader, device)
        fisher_retain = self.fisher_dict.copy()
        print("完成。")
        print("步骤 2/3: 正在遗忘集上计算费雪信息...")
        self._compute_fisher(forget_loader, device)
        fisher_forget = self.fisher_dict.copy()
        print("完成。")
        print("步骤 3/3: 正在执行梯度擦除...")
        self._gradient_erasure(
            forget_loader, fisher_retain, fisher_forget, alpha, device
        )
        print("精准手术式遗忘完成。")
        end_time = time.time()  # Record the end time
        # Calculate and print the runtime
        print(f"执行时间: {end_time - start_time:.2f} 秒")

    def _execute_ascent(self, forget_loader, alpha, device):
        print("阶段 1/2: 正在遗忘集上计算费雪信息...")
        start_time = time.time()
        self._compute_fisher(forget_loader, device)
        print("完成。")
        print("阶段 2/2: 正在执行梯度上升遗忘...")
        self._perform_ascent(forget_loader, alpha, device)
        print("梯度上升式遗忘完成。")
        end_time = time.time()  # Record the end time
        # Calculate and print the runtime
        print(f"执行时间: {end_time - start_time:.2f} 秒")
    def _execute_finetune(self, retain_loader, device, **kwargs):
        # 从 kwargs 获取参数，如果值为 None，则使用默认值
        epochs = kwargs.get("finetune_epochs")
        lr = kwargs.get("finetune_lr")
        layers_to_finetune = kwargs.get("finetune_layers")
        if layers_to_finetune is None:
            layers_to_finetune = ["out", "output"]
        print(f"开始部分微调，共 {epochs} 轮，学习率为 {lr}。")
        print(f"将解冻包含以下关键字的层: {layers_to_finetune}")
        self._set_trainable_layers(layers_to_finetune)
        self._perform_finetune_loop(retain_loader, epochs, lr, device)
        print("微调完成，正在解冻所有模型参数...")
        self._unfreeze_all_layers()
        print("模型已恢复正常可训练状态。")

    def _compute_fisher(self, loader, device):
        self.model.train()
        self.fisher_dict.clear()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.fisher_dict[name] = torch.zeros_like(param.data)
        for batch in loader:
            self.model.zero_grad()
            loss = model_forward(self.model, batch)
            loss.backward()
            for name, param in self.model.named_parameters():
                if param.grad is not None and name in self.fisher_dict:
                    self.fisher_dict[name] += param.grad.pow(2)
        self.model.zero_grad()

    def _gradient_erasure(
        self, forget_loader, fisher_retain, fisher_forget, alpha, device
    ):
        self.model.train()
        for batch in forget_loader:
            loss = model_forward(self.model, batch)
            loss.backward()
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if param.grad is not None and name in fisher_retain:
                    f_retain = fisher_retain[name].clamp(min=1e-8)
                    f_forget = fisher_forget.get(name, torch.zeros_like(f_retain))
                    # importance = f_forget / (f_forget + f_retain + 1e-8)
                    importance = f_forget / (f_forget + f_retain + 1e-8)
                    param.data += alpha * importance * param.grad
        self.model.zero_grad()

    def _perform_ascent(self, forget_loader, alpha, device):
        self.model.train()
        for batch in forget_loader:
            loss = model_forward(self.model, batch)
            loss.backward()
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if param.grad is not None and name in self.fisher_dict:
                    fisher = self.fisher_dict[name].clamp(min=1e-8)
                    param.data += alpha * param.grad / fisher
        self.model.zero_grad()

    def _set_trainable_layers(self, keywords: list):
        for name, param in self.model.named_parameters():
            if any(key in name for key in keywords):
                param.requires_grad = True
            else:
                param.requires_grad = False

    def _unfreeze_all_layers(self):
        for param in self.model.parameters():
            param.requires_grad = True

    def _perform_finetune_loop(self, loader, epochs, lr, device):
        trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer = prepare_model_optimizer(
            params=self.params,
            model_name=self.model.model_name,
            optimizer_type=self.optimizer_type,
            parameters=trainable_params,  # 关键：只为可训练的参数创建优化器！
            learning_rate=lr,  # 使用微调专用的学习率
        )
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch in loader:
                optimizer.zero_grad()
                loss = model_forward(self.model, batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(loader)
            print(f"微调 Epoch {epoch + 1}/{epochs}, 平均损失: {avg_loss:.4f}")
