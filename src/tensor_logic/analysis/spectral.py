"""
Spectral Analysis for Tensor Logic Embeddings

Analyze learned embeddings using SVD to understand:
    1. Effective dimensionality (how many dimensions are used)
    2. Energy distribution (where information is concentrated)
    3. Principal vs off-principal subspaces
    4. Connection to rule-encoding vs fact-encoding

Mathematical basis:
    SVD: M = U @ S @ V^T

    - U: Left singular vectors (entity structure)
    - S: Singular values (importance weights)
    - V: Right singular vectors (relation structure)

    Energy in top-k dimensions: sum(S[:k]^2) / sum(S^2)
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class SpectralStats:
    """Statistics from spectral analysis."""

    singular_values: torch.Tensor
    energy_distribution: torch.Tensor
    effective_rank: float
    principal_ratio: float  # Energy in top 10%
    condition_number: float


class SpectralAnalyzer:
    """
    Analyze learned embeddings via spectral decomposition.

    Key analyses:
        1. SVD of entity embedding matrix
        2. SVD of each relation matrix
        3. Energy concentration (principal vs off-principal)
        4. Correlation between relations
    """

    def __init__(self, device: torch.device = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def analyze_matrix(self, matrix: torch.Tensor) -> SpectralStats:
        """
        Perform spectral analysis on a matrix.

        Args:
            matrix: [m, n] matrix to analyze

        Returns:
            SpectralStats with SVD-based metrics
        """
        # Move to CPU for SVD stability
        M = matrix.detach().cpu().float()

        # SVD
        U, S, Vh = torch.linalg.svd(M, full_matrices=False)

        # Energy distribution
        energy = S ** 2
        total_energy = energy.sum()
        energy_dist = energy / (total_energy + 1e-10)
        cumulative_energy = torch.cumsum(energy_dist, dim=0)

        # Effective rank (number of dimensions for 99% energy)
        effective_rank = (cumulative_energy < 0.99).sum().item() + 1

        # Principal ratio (energy in top 10% of dimensions)
        top_k = max(1, int(0.1 * len(S)))
        principal_ratio = energy[:top_k].sum().item() / (total_energy.item() + 1e-10)

        # Condition number
        condition_number = (S[0] / (S[-1] + 1e-10)).item()

        return SpectralStats(
            singular_values=S,
            energy_distribution=energy_dist,
            effective_rank=effective_rank,
            principal_ratio=principal_ratio,
            condition_number=condition_number,
        )

    def analyze_entity_embeddings(self, embeddings: torch.Tensor) -> SpectralStats:
        """
        Analyze entity embedding matrix.

        Args:
            embeddings: [num_entities, d] entity embeddings

        Returns:
            SpectralStats for entity embedding space
        """
        return self.analyze_matrix(embeddings)

    def analyze_relation_embeddings(
        self,
        relation_matrices: torch.Tensor,
    ) -> Dict[int, SpectralStats]:
        """
        Analyze each relation matrix.

        Args:
            relation_matrices: [num_relations, d, d] relation matrices

        Returns:
            Dict mapping relation index to SpectralStats
        """
        results = {}
        num_relations = relation_matrices.shape[0]

        for i in range(num_relations):
            R = relation_matrices[i]
            results[i] = self.analyze_matrix(R)

        return results

    def compute_relation_similarity(
        self,
        relation_matrices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute similarity between relation matrices.

        Uses Frobenius inner product normalized by norms.

        Args:
            relation_matrices: [num_relations, d, d]

        Returns:
            similarity: [num_relations, num_relations] similarity matrix
        """
        num_relations = relation_matrices.shape[0]
        R_flat = relation_matrices.view(num_relations, -1)

        # Normalize
        R_norm = R_flat / (R_flat.norm(dim=1, keepdim=True) + 1e-10)

        # Cosine similarity
        similarity = torch.mm(R_norm, R_norm.t())

        return similarity

    def decompose_relation(
        self,
        relation_matrix: torch.Tensor,
        principal_ratio: float = 0.1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decompose relation into principal and off-principal components.

        R = R_principal + R_off_principal

        Where R_principal contains top (principal_ratio * d) singular values.

        Args:
            relation_matrix: [d, d] relation matrix
            principal_ratio: Fraction of dimensions for principal

        Returns:
            R_principal, R_off_principal
        """
        R = relation_matrix.detach().cpu().float()
        U, S, Vh = torch.linalg.svd(R, full_matrices=False)

        k = max(1, int(principal_ratio * len(S)))

        # Principal component
        U_p = U[:, :k]
        S_p = S[:k]
        Vh_p = Vh[:k, :]
        R_principal = U_p @ torch.diag(S_p) @ Vh_p

        # Off-principal component
        R_off_principal = R - R_principal

        return R_principal.to(self.device), R_off_principal.to(self.device)

    def analyze_rule_structure(
        self,
        R_head: torch.Tensor,
        R_body_composed: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Analyze how well rule structure is captured.

        For rule: Head <- Body1 @ Body2 @ ...
        Compare: R_head vs R_body_composed

        Args:
            R_head: [d, d] head relation matrix
            R_body_composed: [d, d] composed body relation matrices

        Returns:
            Dict with analysis metrics
        """
        # Get spectral stats for both
        head_stats = self.analyze_matrix(R_head)
        composed_stats = self.analyze_matrix(R_body_composed)

        # Compute difference
        diff = R_head - R_body_composed
        diff_stats = self.analyze_matrix(diff)

        # Decompose difference into principal/off-principal of head
        R_head_principal, R_head_off = self.decompose_relation(R_head, 0.1)
        R_composed_principal, R_composed_off = self.decompose_relation(R_body_composed, 0.1)

        # Compute errors in each subspace
        principal_error = torch.norm(R_head_principal - R_composed_principal).item()
        off_principal_error = torch.norm(R_head_off - R_composed_off).item()

        return {
            "total_error": torch.norm(diff).item(),
            "relative_error": torch.norm(diff).item() / (torch.norm(R_head).item() + 1e-10),
            "principal_error": principal_error,
            "off_principal_error": off_principal_error,
            "head_effective_rank": head_stats.effective_rank,
            "composed_effective_rank": composed_stats.effective_rank,
            "head_principal_ratio": head_stats.principal_ratio,
            "composed_principal_ratio": composed_stats.principal_ratio,
        }

    def summary(
        self,
        entity_embeddings: torch.Tensor,
        relation_matrices: torch.Tensor,
        relation_names: Optional[List[str]] = None,
    ) -> str:
        """
        Generate summary of spectral analysis.

        Args:
            entity_embeddings: [num_entities, d]
            relation_matrices: [num_relations, d, d]
            relation_names: Optional names for relations

        Returns:
            Formatted summary string
        """
        lines = ["=" * 60, " Spectral Analysis Summary", "=" * 60, ""]

        # Entity embeddings
        entity_stats = self.analyze_entity_embeddings(entity_embeddings)
        lines.append("Entity Embeddings:")
        lines.append(f"  Shape: {entity_embeddings.shape}")
        lines.append(f"  Effective rank: {entity_stats.effective_rank:.1f}")
        lines.append(f"  Energy in top 10%: {entity_stats.principal_ratio:.2%}")
        lines.append(f"  Condition number: {entity_stats.condition_number:.2f}")
        lines.append("")

        # Relation embeddings
        relation_stats = self.analyze_relation_embeddings(relation_matrices)
        lines.append("Relation Embeddings:")

        for i, stats in relation_stats.items():
            name = relation_names[i] if relation_names else f"R{i}"
            lines.append(f"  {name}:")
            lines.append(f"    Effective rank: {stats.effective_rank:.1f}")
            lines.append(f"    Energy in top 10%: {stats.principal_ratio:.2%}")

        lines.append("")
        lines.append("=" * 60)

        return "\n".join(lines)
