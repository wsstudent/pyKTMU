# coding=utf-8
"""
    @Author: shimKang
    @file: model_inversion.py.py
    @date: 2025/7/21 4:02 PM
    @blogs: https://blog.csdn.net/ksm180038
"""

import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

class ModelInversionAttack:
    def __init__(self, model, num_questions, seq_length, device='cuda',lr=0.01):
        """
        Model inversion attack class.
        Args:
            model (nn.Module): Target knowledge tracing model
            num_questions (int): Number of questions
            seq_length (int): Maximum sequence length
            device (str): Compute device
        """
        self.model = model
        self.num_q = num_questions
        self.seq_len = seq_length
        self.device = device
        self.model.eval()
        self.default_lr = lr
        with torch.no_grad():
            self.embedding_matrix = self.model.interaction_emb.weight.data.clone()

    def _initialize_embeddings(self):
        """Initialize an optimizable embedding sequence."""
        indices = torch.randint(0, len(self.embedding_matrix), (self.seq_len,))
        return self.embedding_matrix[indices].clone().detach().requires_grad_(True)

    def _decode_embeddings(self, optimized_emb):
        """
        Decode optimized embeddings into question and response sequences.
        Args:
            optimized_emb (Tensor): Optimized embedding sequence (seq_len, emb_size)
        Returns:
            tuple: (question sequence, response sequence)
        """
        # Compute similarity against all embeddings
        similarities = torch.cdist(optimized_emb, self.embedding_matrix)

        # Get the nearest interaction IDs
        interaction_ids = torch.argmin(similarities, dim=1)

        # Decode to questions and responses
        questions = interaction_ids % self.num_q
        responses = (interaction_ids // self.num_q).float()  # Convert to 0/1 responses

        return questions.cpu().numpy(), responses.cpu().numpy()

    def reconstruct(self, target_output, num_steps=1000,  verbose=True):
        # Dynamically determine the sequence length
        batch_size, seq_len, _ = target_output.shape
        self.seq_len = seq_len
        self.model.train()
        optimized_emb = self._initialize_embeddings().to(self.device)
        optimized_emb.requires_grad_(True)
        optimizer = torch.optim.Adam([optimized_emb], lr=self.default_lr)
        best_loss = float('inf')
        best_q, best_r = None, None
        for step in range(num_steps):
            optimizer.zero_grad()
            with torch.enable_grad():
                h, _ = self.model.lstm_layer(optimized_emb.unsqueeze(0))
                pred_output = torch.sigmoid(self.model.out_layer(h))

                loss = F.mse_loss(pred_output, target_output)

            # Gradient check
            assert pred_output.requires_grad, "Predicted output does not carry gradients!"
            loss.backward()
            # Gradient existence check
            if optimized_emb.grad is None:
                raise RuntimeError("Gradients did not propagate to the embedding tensor!")
            optimizer.step()
            # Decode the current best result
            with torch.no_grad():
                current_q, current_r = self._decode_embeddings(optimized_emb)
                if loss.item() < best_loss:
                    best_loss = loss.item()
                    best_q = current_q
                    best_r = current_r

            if verbose and step % 10 == 0:
                print(f"Step {step:04d} | Loss: {loss.item():.4f} | Current Acc: {self._calc_accuracy(current_q, current_r):.2f}")

        return best_q, best_r

    def _calc_accuracy(self, pred_q, pred_r, true_q=None, true_r=None):
        """
        Compute reconstruction accuracy (if ground truth is available).
        Args:
            pred_q (np.ndarray): Predicted question sequence
            pred_r (np.ndarray): Predicted response sequence
            true_q (np.ndarray): Ground truth question sequence
            true_r (np.ndarray): Ground truth response sequence
        Returns:
            float: Accuracy
        """
        if true_q is None or true_r is None:
            return 0.0

        min_len = min(len(pred_q), len(true_q))
        acc_q = np.mean(pred_q[:min_len] == true_q[:min_len])
        acc_r = np.mean(np.round(pred_r[:min_len]) == true_r[:min_len])
        return (acc_q + acc_r) / 2

    def evaluate_leakage(self, data_loader, num_samples=10):
        """
        Evaluate model privacy leakage risk.
        Args:
            data_loader (DataLoader): Test data loader
            num_samples (int): Number of samples to evaluate
        Returns:
            dict: Evaluation metrics
        """
        self.model.eval()
        total_acc = 0.0
        results = []

        with torch.no_grad():
            for idx, (q, r, _, _, _) in enumerate(data_loader):
                if idx >= num_samples:
                    break
                q = q.long().to(self.device)
                r = r.long().to(self.device)
                target_out = self.model(q, r)

                # Perform inversion attack
                recon_q, recon_r = self.reconstruct(target_out, verbose=False)
                self.model.eval()
                # Convert to numpy and remove padding
                true_q = q[0].cpu().numpy()
                true_r = r[0].cpu().numpy()
                valid_mask = true_q != 0

                # Compute accuracy
                acc_q = np.mean(recon_q[valid_mask] == true_q[valid_mask])
                acc_r = np.mean(np.round(recon_r[valid_mask]) == true_r[valid_mask])
                total_acc += (acc_q + acc_r) / 2

                results.app
