# coding=utf-8
"""
    @Author: shimKang
    @file: membership_inference.py.py
    @date: 2025/7/21 4:03 PM
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
        """Improved Membership Inference Attack class"""
        self.model = model.to(device)
        self.device = device
        self.model.eval()
        self.temperature = 1.5  # Added temperature scaling parameter

    def _get_combined_features(self, data_loader):
        """Obtain combined features (loss + calibrated confidence)"""
        losses = []
        confidences = []

        with torch.no_grad():
            for batch in data_loader:
                q, r, qshft, rshft, mask = batch
                q, r, qshft, rshft = q.to(self.device), r.to(self.device), qshft.to(self.device), rshft.to(self.device)
                mask = mask.to(self.device).bool()

                # Forward pass
                pred = self.model(q, r)
                y_pred = torch.gather(pred, 2, qshft.unsqueeze(2)).squeeze(2)

                # Apply temperature scaling to calibrate confidence
                calibrated_pred = torch.sigmoid(torch.log(y_pred / (1 - y_pred)) / self.temperature)

                # Compute features
                loss = F.binary_cross_entropy(calibrated_pred, rshft.float().to(self.device), reduction='none')
                confidence = torch.abs(calibrated_pred - 0.5)  # Calibrated confidence

                # Process per sample
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
        """Stability-focused forgetting evaluation (auto-balanced datasets)"""
        auc_list = []
        acc_list = []

        for _ in range(n_runs):
            # Dynamically sample to balance datasets
            balanced_member = self._resample_loader(member_loader)
            balanced_non_member = self._resample_loader(non_member_loader)

            # Extract features
            member_loss, member_conf = self._get_combined_features(balanced_member)
            non_member_loss, non_member_conf = self._get_combined_features(balanced_non_member)

            # Generate balanced scores
            scores = self._generate_scores(member_loss, member_conf, non_member_loss, non_member_conf)

            # Compute metrics
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
        """Data resampling (balance sequence-length distribution)"""
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
        """Generate stability scores"""
        # Standardization
        all_loss = np.concatenate([member_loss, non_member_loss])
        all_conf = np.concatenate([member_conf, non_member_conf])

        loss_mean, loss_std = np.mean(all_loss), np.std(all_loss)
        conf_mean, conf_std = np.mean(all_conf), np.std(all_conf)

        # Compute standardized scores
        member_scores = (member_conf - conf_mean)/conf_std - (member_loss - loss_mean)/loss_std
        non_member_scores = (non_member_conf - conf_mean)/conf_std - (non_member_loss - loss_mean)/loss_std

        return np.concatenate([member_scores, non_member_scores])
