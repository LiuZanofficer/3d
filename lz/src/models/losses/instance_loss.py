from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class InstanceDiscriminativeLoss(nn.Module):
    def __init__(
        self,
        delta_var: float = 0.5,
        delta_dist: float = 1.5,
        alpha: float = 1.0,
        beta: float = 1.0,
        gamma: float = 0.001,
        ignore_label: int = -100,
    ) -> None:
        super().__init__()
        self.delta_var = delta_var
        self.delta_dist = delta_dist
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.ignore_label = ignore_label

    def forward(self, embeddings: torch.Tensor, instance_labels: torch.Tensor) -> torch.Tensor:
        if embeddings.numel() == 0 or instance_labels is None:
            return embeddings.sum() * 0.0

        instance_labels = instance_labels.view(-1)
        embeddings = embeddings.view(instance_labels.shape[0], -1)

        valid_mask = instance_labels != self.ignore_label
        embeddings = embeddings[valid_mask]
        instance_labels = instance_labels[valid_mask]

        unique_ids = torch.unique(instance_labels)
        if unique_ids.numel() == 0:
            return embeddings.sum() * 0.0

        means = []
        for inst_id in unique_ids:
            mask = instance_labels == inst_id
            if mask.sum() == 0:
                continue
            mean = embeddings[mask].mean(dim=0)
            means.append(mean)
        if len(means) == 0:
            return embeddings.sum() * 0.0
        means = torch.stack(means, dim=0)

        # variance loss
        var_loss = 0.0
        for i, inst_id in enumerate(unique_ids):
            mask = instance_labels == inst_id
            if mask.sum() == 0:
                continue
            dist = torch.norm(embeddings[mask] - means[i], dim=1)
            var = torch.clamp(dist - self.delta_var, min=0.0) ** 2
            var_loss += var.mean()
        var_loss = var_loss / max(len(unique_ids), 1)

        # distance loss
        if means.shape[0] > 1:
            diff = means.unsqueeze(0) - means.unsqueeze(1)
            dist = torch.norm(diff, dim=-1)
            eye = torch.eye(dist.shape[0], device=dist.device).bool()
            dist = dist[~eye].view(dist.shape[0], -1)
            dist_loss = torch.clamp(2 * self.delta_dist - dist, min=0.0) ** 2
            dist_loss = dist_loss.mean()
        else:
            dist_loss = torch.tensor(0.0, device=embeddings.device)

        # regularization
        reg_loss = means.norm(dim=1).mean()

        loss = self.alpha * var_loss + self.beta * dist_loss + self.gamma * reg_loss
        return loss
