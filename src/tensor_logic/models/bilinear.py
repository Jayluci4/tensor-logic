"""
Bilinear Tensor Logic Model

The native Tensor Logic model using bilinear scoring:
    score(h, r, t) = h^T @ R @ t

This is equivalent to the Tensor Logic query:
    D = EmbR[i,j] * Emb[h]_i * Emb[t]_j = einsum('ij,i,j->', R, h, t)

Properties:
    - Supports rule composition: R3 = R1 @ R2
    - Supports inverse relations: R_inv = R^T
    - Enables spectral analysis of learned relations
"""

import torch
import torch.nn as nn
from typing import Optional

from .base import BaseModel
from ..core.embeddings import EntityEmbedding, RelationEmbedding
from ..core.scoring import BilinearScoring


class BilinearModel(BaseModel):
    """
    Bilinear Tensor Logic model.

    Mathematical formulation:
        Entity embeddings: E ∈ R^(n×d)
        Relation embeddings: R ∈ R^(m×d×d)
        Score: s(h,r,t) = e_h^T @ R_r @ e_t

    Training:
        Margin ranking loss with negative sampling:
        L = Σ max(0, γ - s(h,r,t) + s(h',r,t'))

    This model enables:
        - Learning entity and relation representations
        - Rule composition via matrix multiplication
        - Spectral analysis of learned representations
    """

    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        embedding_dim: int,
        normalize_entities: bool = True,
        init_scale: float = 1.0,
        device: torch.device = None,
    ):
        super().__init__(num_entities, num_relations, embedding_dim, device)

        # Entity embeddings
        self.entity_embeddings = EntityEmbedding(
            num_entities=num_entities,
            embedding_dim=embedding_dim,
            normalize=normalize_entities,
            init_scale=init_scale,
        )

        # Relation embeddings (bilinear matrices)
        self.relation_embeddings = RelationEmbedding(
            num_relations=num_relations,
            embedding_dim=embedding_dim,
            diagonal=False,
            init_scale=init_scale,
        )

        # Scoring function
        self.scoring = BilinearScoring()

        # Move to device
        self.to(self.device)

    def score(
        self,
        head_idx: torch.Tensor,
        relation_idx: torch.Tensor,
        tail_idx: torch.Tensor,
    ) -> torch.Tensor:
        """
        Score a batch of triples.

        s(h,r,t) = h^T @ R @ t

        Args:
            head_idx: [batch] head entity indices
            relation_idx: [batch] relation indices
            tail_idx: [batch] tail entity indices

        Returns:
            scores: [batch]
        """
        head = self.entity_embeddings(head_idx)
        relation = self.relation_embeddings(relation_idx)
        tail = self.entity_embeddings(tail_idx)

        return self.scoring(head, relation, tail)

    def score_all_tails(
        self,
        head_idx: torch.Tensor,
        relation_idx: torch.Tensor,
    ) -> torch.Tensor:
        """
        Score all possible tails for given (head, relation) pairs.

        Args:
            head_idx: [batch]
            relation_idx: [batch]

        Returns:
            scores: [batch, num_entities]
        """
        head = self.entity_embeddings(head_idx)
        relation = self.relation_embeddings(relation_idx)
        all_tails = self.entity_embeddings.all_embeddings()

        return self.scoring.score_all_tails(head, relation, all_tails)

    def score_all_heads(
        self,
        relation_idx: torch.Tensor,
        tail_idx: torch.Tensor,
    ) -> torch.Tensor:
        """
        Score all possible heads for given (relation, tail) pairs.

        Args:
            relation_idx: [batch]
            tail_idx: [batch]

        Returns:
            scores: [batch, num_entities]
        """
        relation = self.relation_embeddings(relation_idx)
        tail = self.entity_embeddings(tail_idx)
        all_heads = self.entity_embeddings.all_embeddings()

        return self.scoring.score_all_heads(all_heads, relation, tail)

    def get_relation_matrix(self, relation_idx: int) -> torch.Tensor:
        """
        Get the d×d matrix for a relation.

        Args:
            relation_idx: Relation index

        Returns:
            R: [d, d]
        """
        idx = torch.tensor([relation_idx], device=self.device)
        return self.relation_embeddings(idx).squeeze(0)

    def get_entity_embeddings(self) -> torch.Tensor:
        """
        Get all entity embeddings.

        Returns:
            E: [num_entities, d]
        """
        return self.entity_embeddings.all_embeddings()

    def get_all_relation_matrices(self) -> torch.Tensor:
        """
        Get all relation matrices.

        Returns:
            R: [num_relations, d, d]
        """
        return self.relation_embeddings.weight

    def embed_facts(
        self,
        facts: torch.Tensor,
        relation_idx: int,
    ) -> torch.Tensor:
        """
        Embed a set of facts as a relation matrix.

        This implements the Tensor Logic fact embedding:
        R[i,j] = Σ_{(h,t) ∈ facts} e_h[i] * e_t[j]

        Args:
            facts: [num_facts, 2] tensor of (head_idx, tail_idx)
            relation_idx: Which relation these facts belong to

        Returns:
            R: [d, d] embedded relation matrix
        """
        result = torch.zeros(
            self.embedding_dim, self.embedding_dim, device=self.device
        )

        for i in range(facts.shape[0]):
            h_idx, t_idx = facts[i]
            h = self.entity_embeddings(h_idx.unsqueeze(0)).squeeze(0)
            t = self.entity_embeddings(t_idx.unsqueeze(0)).squeeze(0)
            result += torch.outer(h, t)

        return result

    def forward(
        self,
        head_idx: torch.Tensor,
        relation_idx: torch.Tensor,
        tail_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass = score."""
        return self.score(head_idx, relation_idx, tail_idx)
