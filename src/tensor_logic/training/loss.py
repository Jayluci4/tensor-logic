"""
Loss Functions for Tensor Logic Training

Mathematical formulations:

1. Margin Ranking Loss:
   L = Σ max(0, γ - s(h,r,t) + s(h',r,t'))

   Where:
   - (h,r,t) is positive triple
   - (h',r,t') is negative triple (corrupted)
   - γ is margin hyperparameter

2. Cross-Entropy Loss (1-vs-All):
   L = -log(σ(s(h,r,t))) - Σ log(1 - σ(s(h,r,t')))

   Where t' ranges over all entities except t.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class MarginRankingLoss(nn.Module):
    """
    Margin ranking loss for knowledge graph completion.

    L = Σ max(0, γ - s_pos + s_neg)

    Properties:
        - Pushes positive scores above negative by margin γ
        - Standard loss for TransE, RESCAL, etc.
    """

    def __init__(self, margin: float = 1.0, reduction: str = "mean"):
        super().__init__()
        self.margin = margin
        self.reduction = reduction

    def forward(
        self,
        pos_scores: torch.Tensor,
        neg_scores: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute margin ranking loss.

        Args:
            pos_scores: [batch] scores for positive triples
            neg_scores: [batch] or [batch, num_neg] scores for negative triples

        Returns:
            loss: Scalar loss value
        """
        if neg_scores.dim() == 2:
            # Multiple negatives per positive
            # pos_scores: [batch] -> [batch, 1]
            pos_scores = pos_scores.unsqueeze(1)
            # Loss for each negative
            losses = F.relu(self.margin - pos_scores + neg_scores)
            # Mean over negatives, then over batch
            if self.reduction == "mean":
                return losses.mean()
            elif self.reduction == "sum":
                return losses.sum()
            else:
                return losses
        else:
            # One negative per positive
            losses = F.relu(self.margin - pos_scores + neg_scores)
            if self.reduction == "mean":
                return losses.mean()
            elif self.reduction == "sum":
                return losses.sum()
            else:
                return losses


class CrossEntropyLoss(nn.Module):
    """
    Cross-entropy loss for knowledge graph completion.

    L = -log(exp(s_pos) / Σ exp(s_all))

    Also known as 1-vs-All or softmax loss.

    Properties:
        - Considers all entities as potential negatives
        - More discriminative than margin loss
        - Higher computational cost
    """

    def __init__(self, reduction: str = "mean", label_smoothing: float = 0.0):
        super().__init__()
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def forward(
        self,
        scores: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute cross-entropy loss.

        Args:
            scores: [batch, num_entities] scores for all entities
            targets: [batch] indices of correct entities

        Returns:
            loss: Scalar loss value
        """
        return F.cross_entropy(
            scores,
            targets,
            reduction=self.reduction,
            label_smoothing=self.label_smoothing,
        )


class RuleRegularizationLoss(nn.Module):
    """
    Regularization loss encouraging rule compliance.

    For rule: R3(x,z) <- R1(x,y), R2(y,z)
    Loss: ||R3 - R1 @ R2||_F^2

    This encourages learned relation matrices to respect logical rules.
    """

    def __init__(self, weight: float = 0.1):
        super().__init__()
        self.weight = weight

    def forward(
        self,
        head_matrix: torch.Tensor,
        composed_matrix: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute rule regularization loss.

        Args:
            head_matrix: [d, d] actual head relation matrix
            composed_matrix: [d, d] composed body relation matrices

        Returns:
            loss: Frobenius norm of difference
        """
        diff = head_matrix - composed_matrix
        return self.weight * torch.norm(diff, p="fro") ** 2
