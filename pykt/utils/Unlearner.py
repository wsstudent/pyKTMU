# File path: pykt/utils/Unlearner.py

import torch
import torch.nn as nn
import torch.optim as optim
import time
from pykt.models.train_model import model_forward
from pykt.utils.utils import prepare_model_optimizer


class Unlearner:
    """
    A general unlearning handler that integrates multiple unlearning strategies.
    All methods rely on the generic `model_forward` function to ensure broad compatibility across models.
    """

    def __init__(self, model, model_name, params, optimizer_type="adam"):
        """
        Initialize Unlearner.
        Args:
            model: A trained model that will undergo unlearning.
            model_name (str): The model name used by `model_forward` to select the correct handling logic.
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
        Unified entry point for machine unlearning execution.
        """
        print(f"Unlearner: executing unlearning strategy '{method}'")

        if method == "surgical":
            if retain_loader is None or forget_loader is None:
                raise ValueError(
                    "For the 'surgical' method, both `retain_loader` and `forget_loader` must be provided."
                )
            self._execute_surgical(retain_loader, forget_loader, alpha, device)

        elif method == "ascent":
            if forget_loader is None:
                raise ValueError("For the 'ascent' method, `forget_loader` must be provided.")
            self._execute_ascent(forget_loader, alpha, device)

        elif method == "finetune":
            if retain_loader is None:
                raise ValueError("For the 'finetune' method, `retain_loader` must be provided.")
            self._execute_finetune(retain_loader, device, **kwargs)

        else:
            raise ValueError(
                f"Unknown unlearning method: '{method}'. Available options: 'surgical', 'ascent', 'finetune'."
            )

    # --- Private helper methods ---
    def _execute_surgical(self, retain_loader, forget_loader, alpha, device):
        print("Step 1/3: computing Fisher information on the retain set...")
        start_time = time.time()  # Record the start time
        self._compute_fisher(retain_loader, device)
        fisher_retain = self.fisher_dict.copy()
        print("Done.")
        print("Step 2/3: computing Fisher information on the forget set...")
        self._compute_fisher(forget_loader, device)
        fisher_forget = self.fisher_dict.copy()
        print("Done.")
        print("Step 3/3: performing gradient erasure...")
        self._gradient_erasure(
            forget_loader, fisher_retain, fisher_forget, alpha, device
        )
        print("Surgical unlearning completed.")
        end_time = time.time()  # Record the end time
        # Calculate and print the runtime
        print(f"Elapsed time: {end_time - start_time:.2f} seconds")

    def _execute_ascent(self, forget_loader, alpha, device):
        print("Phase 1/2: computing Fisher information on the forget set...")
        start_time = time.time()
        self._compute_fisher(forget_loader, device)
        print("Done.")
        print("Phase 2/2: performing gradient-ascent unlearning...")
        self._perform_ascent(forget_loader, alpha, device)
        print("Gradient-ascent unlearning completed.")
        end_time = time.time()  # Record the end time
        # Calculate and print the runtime
        print(f"Elapsed time: {end_time - start_time:.2f} seconds")

    def _execute_finetune(self, retain_loader, device, **kwargs):
        # Get parameters from kwargs; if None, use defaults
        epochs = kwargs.get("finetune_epochs")
        lr = kwargs.get("finetune_lr")
        layers_to_finetune = kwargs.get("finetune_layers")
        if layers_to_finetune is None:
            layers_to_finetune = ["out", "output"]
        print(f"Start partial fine-tuning: {epochs} epochs, learning rate {lr}.")
        print(f"Unfreezing layers containing keywords: {layers_to_finetune}")
        self._set_trainable_layers(layers_to_finetune)
        self._perform_finetune_loop(retain_loader, epochs, lr, device)
        print("Fine-tuning finished; unfreezing all model parameters...")
        self._unfreeze_all_layers()
        print("Model restored to normal trainable state.")

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
            parameters=trainable_params,  # Key: create optimizer only for trainable parameters!
            learning_rate=lr,  # Use the fine-tuning-specific learning rate
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
            print(f"Fine-tune Epoch {epoch + 1}/{epochs}, average loss: {avg_loss:.4f}")
