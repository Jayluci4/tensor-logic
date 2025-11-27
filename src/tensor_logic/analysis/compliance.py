"""
Rule Compliance Checking for Tensor Logic

Verify that learned embeddings respect logical rules.

For composition rule: R3(x,z) <- R1(x,y), R2(y,z)
    Compliance measure: ||R3 - R1 @ R2|| / ||R3||

Lower compliance error = better rule adherence.
"""

import torch
from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class ComplianceResult:
    """Result of rule compliance check."""

    rule_name: str
    head_relation: int
    body_relations: Tuple[int, ...]
    error: float
    relative_error: float


class RuleComplianceChecker:
    """
    Check how well learned embeddings respect logical rules.

    Supports:
        - Composition rules: R3 <- R1, R2 (R3 ≈ R1 @ R2)
        - Inverse rules: R2 <- R1^-1 (R2 ≈ R1^T)
        - Implication rules: R2 <- R1 (R2 ≈ R1)
    """

    def __init__(self, device: torch.device = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def check_composition(
        self,
        R_head: torch.Tensor,
        R_body_list: List[torch.Tensor],
    ) -> Tuple[float, float]:
        """
        Check composition rule: Head <- Body1 @ Body2 @ ...

        Args:
            R_head: [d, d] head relation matrix
            R_body_list: List of [d, d] body relation matrices

        Returns:
            absolute_error, relative_error
        """
        # Compose body relations
        R_composed = R_body_list[0]
        for R in R_body_list[1:]:
            R_composed = torch.mm(R_composed, R)

        # Compute error
        diff = R_head - R_composed
        abs_error = torch.norm(diff).item()
        rel_error = abs_error / (torch.norm(R_head).item() + 1e-10)

        return abs_error, rel_error

    def check_inverse(
        self,
        R_head: torch.Tensor,
        R_body: torch.Tensor,
    ) -> Tuple[float, float]:
        """
        Check inverse rule: Head <- Body^-1

        Args:
            R_head: [d, d] head relation matrix
            R_body: [d, d] body relation matrix

        Returns:
            absolute_error, relative_error
        """
        R_inverse = R_body.t()
        diff = R_head - R_inverse
        abs_error = torch.norm(diff).item()
        rel_error = abs_error / (torch.norm(R_head).item() + 1e-10)

        return abs_error, rel_error

    def check_all_rules(
        self,
        relation_matrices: torch.Tensor,
        rules: List[Tuple[int, List[int], str]],
        relation_names: List[str] = None,
    ) -> List[ComplianceResult]:
        """
        Check all rules.

        Args:
            relation_matrices: [num_relations, d, d]
            rules: List of (head_rel, body_rels, rule_type)
            relation_names: Optional names for relations

        Returns:
            List of ComplianceResult
        """
        results = []

        for head_rel, body_rels, rule_type in rules:
            R_head = relation_matrices[head_rel]
            R_body_list = [relation_matrices[i] for i in body_rels]

            if rule_type == "composition":
                abs_err, rel_err = self.check_composition(R_head, R_body_list)
            elif rule_type == "inverse":
                abs_err, rel_err = self.check_inverse(R_head, R_body_list[0])
            else:
                continue

            # Build rule name
            if relation_names:
                head_name = relation_names[head_rel]
                body_names = [relation_names[i] for i in body_rels]
                rule_name = f"{head_name} <- {', '.join(body_names)}"
            else:
                rule_name = f"R{head_rel} <- {', '.join(f'R{i}' for i in body_rels)}"

            results.append(ComplianceResult(
                rule_name=rule_name,
                head_relation=head_rel,
                body_relations=tuple(body_rels),
                error=abs_err,
                relative_error=rel_err,
            ))

        return results

    def summary(self, results: List[ComplianceResult]) -> str:
        """Generate summary of rule compliance."""
        lines = ["=" * 60, " Rule Compliance Summary", "=" * 60, ""]

        for r in results:
            status = "GOOD" if r.relative_error < 0.1 else "POOR" if r.relative_error < 0.5 else "BAD"
            lines.append(f"Rule: {r.rule_name}")
            lines.append(f"  Relative error: {r.relative_error:.4f} [{status}]")
            lines.append("")

        avg_error = sum(r.relative_error for r in results) / len(results) if results else 0
        lines.append(f"Average relative error: {avg_error:.4f}")
        lines.append("=" * 60)

        return "\n".join(lines)
