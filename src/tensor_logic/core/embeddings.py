"""
Entity and Relation Embeddings for Tensor Logic

Mathematical basis:
    Entity embedding:   e ∈ R^d, ||e|| = 1 (unit normalized)
    Relation embedding: R ∈ R^(d×d) (bilinear matrix)

For triple (h, r, t):
    score = h^T @ R @ t

This is equivalent to:
    score = einsum('i,ij,j->', h, R, t)

Which implements the Tensor Logic query:
    D[h,t] = EmbR[i,j] * Emb[h]_i * Emb[t]_j
"""

import torch
import torch.nn as nn
import math
from typing import Optional


class EntityEmbedding(nn.Module):
    """
    Entity embedding table.

    Each entity is represented as a d-dimensional vector.
    Vectors are L2 normalized to unit sphere.

    Mathematical property:
        For random unit vectors e1, e2:
        E[e1 · e2] = 0
        Var[e1 · e2] = 1/d

        This ensures orthogonality in expectation,
        enabling clean separation of facts.
    """

    def __init__(
        self,
        num_entities: int,
        embedding_dim: int,
        normalize: bool = True,
        init_scale: float = 1.0,
    ):
        super().__init__()
        self.num_entities = num_entities
        self.embedding_dim = embedding_dim
        self.normalize = normalize

        # Initialize embeddings
        self.weight = nn.Parameter(torch.empty(num_entities, embedding_dim))
        self._init_weights(init_scale)

    def _init_weights(self, scale: float):
        """
        Xavier uniform initialization, then normalize.

        Xavier bound: sqrt(6 / (fan_in + fan_out))
        For embeddings: sqrt(6 / (1 + d)) ≈ sqrt(6/d)
        """
        bound = scale * math.sqrt(6.0 / self.embedding_dim)
        nn.init.uniform_(self.weight, -bound, bound)

        if self.normalize:
            with torch.no_grad():
                self.weight.data = nn.functional.normalize(self.weight.data, p=2, dim=1)

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Look up embeddings for given indices.

        Args:
            indices: [batch_size] or [batch_size, seq_len]

        Returns:
            embeddings: [batch_size, d] or [batch_size, seq_len, d]
        """
        embeddings = nn.functional.embedding(indices, self.weight)

        if self.normalize:
            embeddings = nn.functional.normalize(embeddings, p=2, dim=-1)

        return embeddings

    def all_embeddings(self) -> torch.Tensor:
        """Return all entity embeddings (normalized if applicable)."""
        if self.normalize:
            return nn.functional.normalize(self.weight, p=2, dim=1)
        return self.weight


class RelationEmbedding(nn.Module):
    """
    Relation embedding table.

    Each relation is represented as a d×d matrix (bilinear form).
    This enables the tensor logic operation:
        R(h, t) = h^T @ R @ t

    Mathematical properties:
        - Full bilinear: R ∈ R^(d×d) has d² parameters
        - Diagonal: R = diag(r), r ∈ R^d has d parameters
        - We use full bilinear for expressiveness

    Rule composition:
        R_composed = R1 @ R2
        This implements: Composed(x,z) <- R1(x,y), R2(y,z)
    """

    def __init__(
        self,
        num_relations: int,
        embedding_dim: int,
        diagonal: bool = False,
        init_scale: float = 1.0,
    ):
        super().__init__()
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        self.diagonal = diagonal

        if diagonal:
            # Diagonal relation: d parameters per relation
            self.weight = nn.Parameter(torch.empty(num_relations, embedding_dim))
        else:
            # Full bilinear: d² parameters per relation
            self.weight = nn.Parameter(
                torch.empty(num_relations, embedding_dim, embedding_dim)
            )

        self._init_weights(init_scale)

    def _init_weights(self, scale: float):
        """
        Initialize relation matrices.

        For bilinear: Xavier uniform
        For diagonal: Uniform around 1.0 (identity-like)
        """
        if self.diagonal:
            nn.init.uniform_(self.weight, 1.0 - scale * 0.5, 1.0 + scale * 0.5)
        else:
            bound = scale * math.sqrt(6.0 / (2 * self.embedding_dim))
            nn.init.uniform_(self.weight, -bound, bound)

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Look up relation embeddings.

        Args:
            indices: [batch_size]

        Returns:
            If diagonal: [batch_size, d]
            If full: [batch_size, d, d]
        """
        return nn.functional.embedding(indices, self.weight.view(self.num_relations, -1)).view(
            indices.shape[0], *self.weight.shape[1:]
        )

    def get_matrix(self, index: int) -> torch.Tensor:
        """
        Get relation matrix for a single relation.

        Args:
            index: Relation index

        Returns:
            R: [d, d] matrix (expands diagonal if needed)
        """
        if self.diagonal:
            return torch.diag(self.weight[index])
        return self.weight[index]

    def compose(self, idx1: int, idx2: int) -> torch.Tensor:
        """
        Compose two relations via matrix multiplication.

        R_composed = R1 @ R2

        This implements the rule:
            Composed(x,z) <- R1(x,y), R2(y,z)

        Args:
            idx1: First relation index
            idx2: Second relation index

        Returns:
            R_composed: [d, d] composed relation matrix
        """
        R1 = self.get_matrix(idx1)
        R2 = self.get_matrix(idx2)
        return torch.mm(R1, R2)
