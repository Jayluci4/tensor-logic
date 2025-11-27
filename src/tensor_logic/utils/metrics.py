"""
Evaluation Metrics for Knowledge Graph Completion

Standard metrics:
    - MRR (Mean Reciprocal Rank): Average of 1/rank
    - Hits@K: Fraction of correct answers in top K
    - MR (Mean Rank): Average rank of correct answer
"""

import torch
from typing import Dict, List


def compute_ranks(
    scores: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """
    Compute ranks of target entities.

    rank = number of entities with higher score + 1

    Args:
        scores: [batch, num_entities] scores for all entities
        targets: [batch] indices of correct entities

    Returns:
        ranks: [batch] rank of each target (1-indexed)
    """
    # Get target scores
    target_scores = scores.gather(1, targets.unsqueeze(1)).squeeze(1)

    # Count entities with higher scores
    # rank = (number with score > target) + 1
    ranks = (scores > target_scores.unsqueeze(1)).sum(dim=1) + 1

    return ranks


def compute_mrr(ranks: torch.Tensor) -> float:
    """
    Compute Mean Reciprocal Rank.

    MRR = mean(1 / rank)

    Args:
        ranks: [N] ranks (1-indexed)

    Returns:
        MRR score (higher is better)
    """
    return (1.0 / ranks.float()).mean().item()


def compute_hits_at_k(ranks: torch.Tensor, k: int) -> float:
    """
    Compute Hits@K.

    Hits@K = fraction of ranks <= K

    Args:
        ranks: [N] ranks (1-indexed)
        k: Threshold

    Returns:
        Hits@K score (higher is better)
    """
    return (ranks <= k).float().mean().item()


def compute_mean_rank(ranks: torch.Tensor) -> float:
    """
    Compute Mean Rank.

    MR = mean(rank)

    Args:
        ranks: [N] ranks (1-indexed)

    Returns:
        Mean rank (lower is better)
    """
    return ranks.float().mean().item()


def compute_metrics(ranks: torch.Tensor) -> Dict[str, float]:
    """
    Compute all standard metrics.

    Args:
        ranks: [N] ranks (1-indexed)

    Returns:
        Dict with MRR, MR, Hits@1, Hits@3, Hits@10
    """
    return {
        "mrr": compute_mrr(ranks),
        "mr": compute_mean_rank(ranks),
        "hits@1": compute_hits_at_k(ranks, 1),
        "hits@3": compute_hits_at_k(ranks, 3),
        "hits@10": compute_hits_at_k(ranks, 10),
    }


def format_metrics(metrics: Dict[str, float]) -> str:
    """Format metrics for display."""
    return (
        f"MRR: {metrics['mrr']:.4f} | "
        f"MR: {metrics['mr']:.1f} | "
        f"H@1: {metrics['hits@1']:.4f} | "
        f"H@3: {metrics['hits@3']:.4f} | "
        f"H@10: {metrics['hits@10']:.4f}"
    )
