"""
Base Model for Tensor Logic

Abstract base class defining the interface for all Tensor Logic models.
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Optional


class BaseModel(nn.Module, ABC):
    """
    Abstract base class for Tensor Logic models.

    All models must implement:
        - score(): Score a batch of triples
        - score_all_tails(): Score all tails for (h, r)
        - score_all_heads(): Score all heads for (r, t)
        - get_relation_matrix(): Get the d×d matrix for a relation
    """

    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        embedding_dim: int,
        device: torch.device = None,
    ):
        super().__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @abstractmethod
    def score(
        self,
        head_idx: torch.Tensor,
        relation_idx: torch.Tensor,
        tail_idx: torch.Tensor,
    ) -> torch.Tensor:
        """
        Score a batch of triples.

        Args:
            head_idx: [batch] head entity indices
            relation_idx: [batch] relation indices
            tail_idx: [batch] tail entity indices

        Returns:
            scores: [batch] triple scores
        """
        pass

    @abstractmethod
    def score_all_tails(
        self,
        head_idx: torch.Tensor,
        relation_idx: torch.Tensor,
    ) -> torch.Tensor:
        """
        Score all possible tails for given (head, relation) pairs.

        Args:
            head_idx: [batch] head entity indices
            relation_idx: [batch] relation indices

        Returns:
            scores: [batch, num_entities]
        """
        pass

    @abstractmethod
    def score_all_heads(
        self,
        relation_idx: torch.Tensor,
        tail_idx: torch.Tensor,
    ) -> torch.Tensor:
        """
        Score all possible heads for given (relation, tail) pairs.

        Args:
            relation_idx: [batch] relation indices
            tail_idx: [batch] tail entity indices

        Returns:
            scores: [batch, num_entities]
        """
        pass

    @abstractmethod
    def get_relation_matrix(self, relation_idx: int) -> torch.Tensor:
        """
        Get the d×d matrix representation of a relation.

        This is used for:
            - Rule composition (R1 @ R2)
            - Rule compliance checking
            - Spectral analysis

        Args:
            relation_idx: Relation index

        Returns:
            R: [d, d] relation matrix
        """
        pass

    def get_entity_embeddings(self) -> torch.Tensor:
        """
        Get all entity embeddings.

        Returns:
            E: [num_entities, d] entity embedding matrix
        """
        raise NotImplementedError

    def compose_relations(self, rel1_idx: int, rel2_idx: int) -> torch.Tensor:
        """
        Compose two relations via matrix multiplication.

        R_composed = R1 @ R2

        Implements: Composed(x,z) <- R1(x,y), R2(y,z)

        Args:
            rel1_idx: First relation index
            rel2_idx: Second relation index

        Returns:
            R_composed: [d, d]
        """
        R1 = self.get_relation_matrix(rel1_idx)
        R2 = self.get_relation_matrix(rel2_idx)
        return torch.mm(R1, R2)

    def check_rule_compliance(
        self,
        head_rel: int,
        body_rels: Tuple[int, ...],
    ) -> float:
        """
        Check how well the model respects a composition rule.

        Rule: head_rel(x,z) <- body_rels[0](x,y), body_rels[1](y,z), ...

        Compliance = ||R_head - R_body1 @ R_body2 @ ...|| / ||R_head||

        Args:
            head_rel: Head relation index
            body_rels: Tuple of body relation indices

        Returns:
            Compliance error (0 = complete compliance)
        """
        # Get head relation matrix
        R_head = self.get_relation_matrix(head_rel)

        # Compose body relations
        R_composed = self.get_relation_matrix(body_rels[0])
        for rel_idx in body_rels[1:]:
            R_composed = torch.mm(R_composed, self.get_relation_matrix(rel_idx))

        # Compute relative error
        error = torch.norm(R_head - R_composed) / (torch.norm(R_head) + 1e-8)
        return error.item()
