"""
Negative Sampling for Tensor Logic Training

Negative sampling strategies:
1. Uniform: Sample entities uniformly at random
2. Bernoulli: Sample head or tail based on relation statistics
3. Self-adversarial: Weight negatives by their scores
"""

import torch
from typing import Set, Tuple, Optional


class NegativeSampler:
    """
    Negative sampler for knowledge graph completion.

    Given positive triple (h, r, t), generates negative triples
    by corrupting either head or tail:
        - (h', r, t) with h' ≠ h
        - (h, r, t') with t' ≠ t
    """

    def __init__(
        self,
        num_entities: int,
        num_negatives: int = 1,
        filter_positives: bool = True,
        corrupt_head_prob: float = 0.5,
        device: torch.device = None,
    ):
        """
        Initialize negative sampler.

        Args:
            num_entities: Total number of entities
            num_negatives: Number of negatives per positive
            filter_positives: Whether to filter out true positives
            corrupt_head_prob: Probability of corrupting head (vs tail)
            device: Torch device
        """
        self.num_entities = num_entities
        self.num_negatives = num_negatives
        self.filter_positives = filter_positives
        self.corrupt_head_prob = corrupt_head_prob
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Set of known true positives: (h, r, t)
        self.positives: Set[Tuple[int, int, int]] = set()

    def add_positives(self, triples: torch.Tensor):
        """
        Add known positive triples (for filtering).

        Args:
            triples: [N, 3] tensor of (head, relation, tail)
        """
        for i in range(triples.shape[0]):
            h, r, t = triples[i].tolist()
            self.positives.add((h, r, t))

    def sample(
        self,
        head: torch.Tensor,
        relation: torch.Tensor,
        tail: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate negative samples for a batch of positive triples.

        Args:
            head: [batch] head indices
            relation: [batch] relation indices
            tail: [batch] tail indices

        Returns:
            neg_head: [batch, num_negatives] negative head indices
            neg_relation: [batch, num_negatives] relation indices (same)
            neg_tail: [batch, num_negatives] negative tail indices
        """
        batch_size = head.shape[0]

        # Decide whether to corrupt head or tail for each sample
        corrupt_head = torch.rand(batch_size, device=self.device) < self.corrupt_head_prob

        # Sample random entities
        random_entities = torch.randint(
            0, self.num_entities,
            (batch_size, self.num_negatives),
            device=self.device,
        )

        # Expand originals
        head_exp = head.unsqueeze(1).expand(-1, self.num_negatives)
        tail_exp = tail.unsqueeze(1).expand(-1, self.num_negatives)
        relation_exp = relation.unsqueeze(1).expand(-1, self.num_negatives)

        # Create negatives
        corrupt_head_exp = corrupt_head.unsqueeze(1).expand(-1, self.num_negatives)

        neg_head = torch.where(corrupt_head_exp, random_entities, head_exp)
        neg_tail = torch.where(corrupt_head_exp, tail_exp, random_entities)

        if self.filter_positives and len(self.positives) > 0:
            # Replace any accidental true positives
            for i in range(batch_size):
                for j in range(self.num_negatives):
                    h, r, t = neg_head[i, j].item(), relation[i].item(), neg_tail[i, j].item()
                    attempts = 0
                    while (h, r, t) in self.positives and attempts < 10:
                        if corrupt_head[i]:
                            h = torch.randint(0, self.num_entities, (1,)).item()
                            neg_head[i, j] = h
                        else:
                            t = torch.randint(0, self.num_entities, (1,)).item()
                            neg_tail[i, j] = t
                        attempts += 1

        return neg_head, relation_exp, neg_tail

    def sample_uniform(
        self,
        batch_size: int,
        num_samples: int,
    ) -> torch.Tensor:
        """
        Sample uniform random entities.

        Args:
            batch_size: Batch size
            num_samples: Number of samples per batch item

        Returns:
            samples: [batch_size, num_samples]
        """
        return torch.randint(
            0, self.num_entities,
            (batch_size, num_samples),
            device=self.device,
        )
