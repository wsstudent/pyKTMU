# coding=utf-8
"""
    @Author：shimKang
    @file： model_inversion.py.py
    @date：2025/7/21 下午4:02
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
        模型反演攻击类
        Args:
            model (nn.Module): 目标知识追踪模型
            num_questions (int): 题目数量
            seq_length (int): 序列最大长度
            device (str): 计算设备
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
        """初始化可优化的嵌入序列"""
        indices = torch.randint(0, len(self.embedding_matrix), (self.seq_len,))
        return self.embedding_matrix[indices].clone().detach().requires_grad_(True)

    def _decode_embeddings(self, optimized_emb):
        """
        将优化后的嵌入向量解码为题目和响应序列
        Args:
            optimized_emb (Tensor): 优化后的嵌入序列 (seq_len, emb_size)
        Returns:
            tuple: (问题序列, 响应序列)
        """
        # 计算与所有嵌入的相似度
        similarities = torch.cdist(optimized_emb, self.embedding_matrix)

        # 获取最接近的交互ID
        interaction_ids = torch.argmin(similarities, dim=1)

        # 解码为题目和响应
        questions = interaction_ids % self.num_q
        responses = (interaction_ids // self.num_q).float()  # 转为0/1响应

        return questions.cpu().numpy(), responses.cpu().numpy()

    def reconstruct(self, target_output, num_steps=1000,  verbose=True):
        # 动态获取序列长度
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

            # 梯度检查
            assert pred_output.requires_grad, "预测输出未携带梯度！"
            loss.backward()
            # 梯度存在性检查
            if optimized_emb.grad is None:
                raise RuntimeError("梯度未传播到嵌入张量！")
            optimizer.step()
            # 解码当前最优结果
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
        计算重建准确率（如有真实标签）
        Args:
            pred_q (np.ndarray): 预测问题序列
            pred_r (np.ndarray): 预测响应序列
            true_q (np.ndarray): 真实问题序列
            true_r (np.ndarray): 真实响应序列
        Returns:
            float: 准确率
        """
        if true_q is None or true_r is None:
            return 0.0


        min_len = min(len(pred_q), len(true_q))
        acc_q = np.mean(pred_q[:min_len] == true_q[:min_len])
        acc_r = np.mean(np.round(pred_r[:min_len]) == true_r[:min_len])
        return (acc_q + acc_r) / 2

    def evaluate_leakage(self, data_loader, num_samples=10):
        """
        评估模型隐私泄露风险
        Args:
            data_loader (DataLoader): 测试数据加载器
            num_samples (int): 评估样本数量
        Returns:
            dict: 各指标的评估结果
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

                # 执行反演攻击
                recon_q, recon_r = self.reconstruct(target_out, verbose=False)
                self.model.eval()
                # 转换为numpy并去除填充
                true_q = q[0].cpu().numpy()
                true_r = r[0].cpu().numpy()
                valid_mask = true_q != 0

                # 计算准确率
                acc_q = np.mean(recon_q[valid_mask] == true_q[valid_mask])
                acc_r = np.mean(np.round(recon_r[valid_mask]) == true_r[valid_mask])
                total_acc += (acc_q + acc_r) / 2

                results.append({
                    'true_q': true_q[valid_mask],
                    'true_r': true_r[valid_mask],
                    'pred_q': recon_q[valid_mask],
                    'pred_r': recon_r[valid_mask],
                    'acc_q': acc_q,
                    'acc_r': acc_r
                })

        return {
            'average_accuracy': total_acc / num_samples,
            'details': results
        }
