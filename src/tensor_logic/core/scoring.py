"""
Scoring Functions for Tensor Logic

Mathematical basis for scoring a triple (h, r, t):

1. Bilinear (RESCAL-like):
   score(h, r, t) = h^T @ R @ t = Σ_ij h_i R_ij t_j

2. Distance (TransE-like):
   score(h, r, t) = -||h + r - t||

3. Tensor Logic interpretation:
   The bilinear score is equivalent to:
   D = EmbR[i,j] * Emb[h]_i * Emb[t]_j

   Where:
   - Emb[h] is entity h's embedding
   - EmbR is relation R's embedding (matrix)
   - D is the query result (scalar)
"""

import torch
import torch.nn as nn
from typing import Optional


class BilinearScoring(nn.Module):
    """
    Bilinear scoring function.

    score(h, r, t) = h^T @ R @ t

    This is the native Tensor Logic scoring:
    - Equivalent to einsum('i,ij,j->', h, R, t)
    - Supports rule composition via R1 @ R2
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        head: torch.Tensor,
        relation: torch.Tensor,
        tail: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute bilinear scores.

        Args:
            head: [batch, d] head entity embeddings
            relation: [batch, d, d] relation matrices
            tail: [batch, d] tail entity embeddings

        Returns:
            scores: [batch] scalar scores
        """
        # h^T @ R @ t = einsum('bi,bij,bj->b', h, R, t)
        scores = torch.einsum('bi,bij,bj->b', head, relation, tail)
        return scores

    def score_all_tails(
        self,
        head: torch.Tensor,
        relation: torch.Tensor,
        all_tails: torch.Tensor,
    ) -> torch.Tensor:
        """
        Score all possible tails for given (head, relation) pairs.

        Args:
            head: [batch, d]
            relation: [batch, d, d]
            all_tails: [num_entities, d]

        Returns:
            scores: [batch, num_entities]
        """
        # h^T @ R -> [batch, d]
        hR = torch.einsum('bi,bij->bj', head, relation)
        # (h^T @ R) @ t^T -> [batch, num_entities]
        scores = torch.mm(hR, all_tails.t())
        return scores

    def score_all_heads(
        self,
        all_heads: torch.Tensor,
        relation: torch.Tensor,
        tail: torch.Tensor,
    ) -> torch.Tensor:
        """
        Score all possible heads for given (relation, tail) pairs.

        Args:
            all_heads: [num_entities, d]
            relation: [batch, d, d]
            tail: [batch, d]

        Returns:
            scores: [batch, num_entities]
        """
        # R @ t -> [batch, d]
        Rt = torch.einsum('bij,bj->bi', relation, tail)
        # h @ (R @ t)^T -> [batch, num_entities]
        scores = torch.mm(Rt, all_heads.t())
        return scores


class DistanceScoring(nn.Module):
    """
    Distance-based scoring (TransE-style).

    score(h, r, t) = -||h + r - t||_p

    Where r is a translation vector, not a matrix.
    Included for comparison; not native to Tensor Logic.
    """

    def __init__(self, p: int = 2):
        super().__init__()
        self.p = p

    def forward(
        self,
        head: torch.Tensor,
        relation: torch.Tensor,
        tail: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute distance scores.

        Args:
            head: [batch, d]
            relation: [batch, d] (vector, not matrix)
            tail: [batch, d]

        Returns:
            scores: [batch] (negative distance)
        """
        diff = head + relation - tail
        dist = torch.norm(diff, p=self.p, dim=-1)
        return -dist


class DiagonalScoring(nn.Module):
    """
    Diagonal bilinear scoring (DistMult-style).

    score(h, r, t) = Σ_i h_i * r_i * t_i

    This is bilinear with R = diag(r).
    More efficient: O(d) instead of O(d²).
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        head: torch.Tensor,
        relation: torch.Tensor,
        tail: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute diagonal bilinear scores.

        Args:
            head: [batch, d]
            relation: [batch, d] (diagonal elements)
            tail: [batch, d]

        Returns:
            scores: [batch]
        """
        return (head * relation * tail).sum(dim=-1)

    def score_all_tails(
        self,
        head: torch.Tensor,
        relation: torch.Tensor,
        all_tails: torch.Tensor,
    ) -> torch.Tensor:
        """
        Score all possible tails.

        Args:
            head: [batch, d]
            relation: [batch, d]
            all_tails: [num_entities, d]

        Returns:
            scores: [batch, num_entities]
        """
        # h * r -> [batch, d]
        hr = head * relation
        # (h * r) @ t^T -> [batch, num_entities]
        return torch.mm(hr, all_tails.t())
