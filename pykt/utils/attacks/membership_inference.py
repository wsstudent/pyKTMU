# coding=utf-8
"""

    @Author：shimKang
    @file： membership_inference.py.py
    @date：2025/7/21 下午4:03
    @blogs: https://blog.csdn.net/ksm180038
"""
import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader
import numpy as np
from torch.utils.data import Subset
from sklearn import metrics
class MembershipInferenceAttack:
    def __init__(self, model, device='cuda'):
        """改进版成员推演攻击类"""
        self.model = model.to(device)
        self.device = device
        self.model.eval()
        self.temperature = 1.5  # 新增温度缩放参数


    def _get_combined_features(self, data_loader):

        """获取联合特征（损失值+校准后置信度）"""
        losses = []
        confidences = []

        with torch.no_grad():
            for batch in data_loader:
                q, r, qshft, rshft, mask = batch
                q, r, qshft, rshft = q.to(self.device), r.to(self.device), qshft.to(self.device), rshft.to(self.device)
                mask = mask.to(self.device).bool()

                # 前向传播
                pred = self.model(q, r)
                y_pred = torch.gather(pred, 2, qshft.unsqueeze(2)).squeeze(2)

                # 应用温度缩放校准置信度
                calibrated_pred = torch.sigmoid(torch.log(y_pred / (1 - y_pred)) / self.temperature)

                # 计算特征
                loss = F.binary_cross_entropy(calibrated_pred, rshft.float().to(self.device), reduction='none')
                confidence = torch.abs(calibrated_pred - 0.5)  # 校准后置信度

                # 按样本处理
                batch_size = q.size(0)
                for i in range(batch_size):
                    sample_mask = mask[i]
                    if sample_mask.sum() == 0:
                        continue

                    valid_loss = loss[i][sample_mask].mean().item()
                    valid_conf = confidence[i][sample_mask].mean().item()

                    losses.append(valid_loss)
                    confidences.append(valid_conf)

        return np.array(losses), np.array(confidences)

    def evaluate_attack(self, member_loader, non_member_loader, n_runs=5):
        """稳定性遗忘评估（自动平衡数据集）"""
        auc_list = []
        acc_list = []

        for _ in range(n_runs):
            # 动态采样平衡数据集
            balanced_member = self._resample_loader(member_loader)
            balanced_non_member = self._resample_loader(non_member_loader)

            # 获取特征
            member_loss, member_conf = self._get_combined_features(balanced_member)
            non_member_loss, non_member_conf = self._get_combined_features(balanced_non_member)

            # 生成平衡评分
            scores = self._generate_scores(member_loss, member_conf, non_member_loss, non_member_conf)

            # 计算指标
            labels = np.concatenate([np.ones_like(member_loss), np.zeros_like(non_member_loss)])
            auc = metrics.roc_auc_score(labels, scores)
            acc = metrics.accuracy_score(labels, (scores >= 0.5).astype(int))

            auc_list.append(auc)
            acc_list.append(acc)

        return {
            'auc': np.mean(auc_list),
            'accuracy': np.mean(acc_list),
            'auc_std': np.std(auc_list),
            'acc_std': np.std(acc_list)
        }

    def _resample_loader(self, loader):
        """数据重采样（平衡序列长度分布）"""
        indices = np.random.choice(
            range(len(loader.dataset)),
            size=len(loader.dataset),
            replace=True
        )
        return DataLoader(
            Subset(loader.dataset, indices),
            batch_size=loader.batch_size,
            collate_fn=loader.collate_fn
        )

    def _generate_scores(self, member_loss, member_conf, non_member_loss, non_member_conf):
        """生成稳定性评分"""
        # 标准化处理
        all_loss = np.concatenate([member_loss, non_member_loss])
        all_conf = np.concatenate([member_conf, non_member_conf])

        loss_mean, loss_std = np.mean(all_loss), np.std(all_loss)
        conf_mean, conf_std = np.mean(all_conf), np.std(all_conf)

        # 计算标准化得分
        member_scores = (member_conf - conf_mean)/conf_std - (member_loss - loss_mean)/loss_std
        non_member_scores = (non_member_conf - conf_mean)/conf_std - (non_member_loss - loss_mean)/loss_std

        return np.concatenate([member_scores, non_member_scores])
