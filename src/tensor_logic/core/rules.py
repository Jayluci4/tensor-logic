"""
Rule Engine for Tensor Logic

Mathematical basis:
    Datalog rule: Head(x,z) <- Body1(x,y), Body2(y,z)
    Tensor form:  H = σ(B1 @ B2)

Where:
    - @ is matrix multiplication (join on y)
    - σ is threshold function

Rule types:
    1. Composition: R3(x,z) <- R1(x,y), R2(y,z)
       Tensor: R3 = R1 @ R2

    2. Inverse: R2(y,x) <- R1(x,y)
       Tensor: R2 = R1^T

    3. Implication: R2(x,y) <- R1(x,y)
       Tensor: R2 += R1

Forward chaining:
    Repeatedly apply rules until fixpoint.
    In tensor form: iterate matrix operations until convergence.
"""

import torch
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class Rule:
    """
    A logical rule in tensor form.

    head: Name of head relation
    body: List of body relation names
    rule_type: 'composition', 'inverse', 'implication'
    """

    head: str
    body: List[str]
    rule_type: str

    def __str__(self):
        body_str = ", ".join(self.body)
        return f"{self.head}(x,z) <- {body_str}"


class RuleEngine:
    """
    Engine for applying logical rules as tensor operations.

    Supports:
        - Rule composition (matrix multiply)
        - Rule inverse (transpose)
        - Forward chaining to fixpoint
        - Rule compliance checking
    """

    def __init__(self, device: torch.device = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.rules: List[Rule] = []
        self.relation_matrices: Dict[str, torch.Tensor] = {}

    def add_rule(self, head: str, body: List[str], rule_type: str = "composition"):
        """
        Add a rule to the engine.

        Args:
            head: Head relation name
            body: List of body relation names
            rule_type: Type of rule ('composition', 'inverse', 'implication')
        """
        self.rules.append(Rule(head=head, body=body, rule_type=rule_type))

    def set_relation_matrix(self, name: str, matrix: torch.Tensor):
        """Set the tensor representation of a relation."""
        self.relation_matrices[name] = matrix.to(self.device)

    def apply_rule(self, rule: Rule, threshold: float = 0.0) -> Optional[torch.Tensor]:
        """
        Apply a single rule to derive the head relation.

        Args:
            rule: The rule to apply
            threshold: Threshold for converting to Boolean (0 = soft)

        Returns:
            Derived relation matrix, or None if body relations missing
        """
        # Check all body relations exist
        for rel in rule.body:
            if rel not in self.relation_matrices:
                return None

        if rule.rule_type == "composition":
            # R_head = R_body1 @ R_body2 @ ... @ R_bodyN
            result = self.relation_matrices[rule.body[0]]
            for rel in rule.body[1:]:
                result = torch.mm(result, self.relation_matrices[rel])

        elif rule.rule_type == "inverse":
            # R_head = R_body^T
            result = self.relation_matrices[rule.body[0]].t()

        elif rule.rule_type == "implication":
            # R_head = R_body (direct copy/transfer)
            result = self.relation_matrices[rule.body[0]].clone()

        else:
            raise ValueError(f"Unknown rule type: {rule.rule_type}")

        # Apply threshold
        if threshold > 0:
            result = (result > threshold).float()

        return result

    def forward_chain(
        self,
        max_iterations: int = 10,
        threshold: float = 0.0,
        convergence_tol: float = 1e-6,
    ) -> int:
        """
        Forward chaining: apply all rules until fixpoint.

        Args:
            max_iterations: Maximum iterations
            threshold: Threshold for Boolean conversion
            convergence_tol: Tolerance for convergence check

        Returns:
            Number of iterations until convergence
        """
        for iteration in range(max_iterations):
            max_change = 0.0

            for rule in self.rules:
                # Compute new relation matrix from rule
                new_matrix = self.apply_rule(rule, threshold)
                if new_matrix is None:
                    continue

                # Check if head relation exists
                if rule.head in self.relation_matrices:
                    old_matrix = self.relation_matrices[rule.head]
                    # Combine: take max (OR in Boolean case)
                    combined = torch.maximum(old_matrix, new_matrix)
                    change = (combined - old_matrix).abs().max().item()
                    max_change = max(max_change, change)
                    self.relation_matrices[rule.head] = combined
                else:
                    self.relation_matrices[rule.head] = new_matrix
                    max_change = float("inf")

            if max_change < convergence_tol:
                return iteration + 1

        return max_iterations

    def check_rule_compliance(
        self,
        learned_relations: Dict[str, torch.Tensor],
        rules: List[Rule] = None,
    ) -> Dict[str, float]:
        """
        Check how well learned relation embeddings comply with rules.

        For rule R3 <- R1, R2:
            Compliance = ||R3 - R1 @ R2|| / ||R3||

        Lower is better (0 = complete compliance).

        Args:
            learned_relations: Dict of relation name -> matrix
            rules: Rules to check (default: all rules)

        Returns:
            Dict of rule -> compliance score
        """
        rules = rules or self.rules
        compliance = {}

        for rule in rules:
            if rule.rule_type != "composition":
                continue

            # Check all relations exist
            missing = [r for r in [rule.head] + rule.body if r not in learned_relations]
            if missing:
                continue

            # Compute expected head from body
            expected = learned_relations[rule.body[0]]
            for rel in rule.body[1:]:
                expected = torch.mm(expected, learned_relations[rel])

            # Compute actual head
            actual = learned_relations[rule.head]

            # Compute relative error
            error = torch.norm(actual - expected) / (torch.norm(actual) + 1e-8)
            compliance[str(rule)] = error.item()

        return compliance

    def query(
        self,
        relation: str,
        head_emb: torch.Tensor,
        tail_emb: torch.Tensor,
        temperature: float = 0.0,
    ) -> float:
        """
        Query a relation using tensor logic.

        score = h^T @ R @ t

        Args:
            relation: Relation name
            head_emb: Head entity embedding [d]
            tail_emb: Tail entity embedding [d]
            temperature: Temperature for soft reasoning

        Returns:
            Query score (0 or 1 at T=0, continuous at T>0)
        """
        if relation not in self.relation_matrices:
            return 0.0

        R = self.relation_matrices[relation]
        score = torch.einsum("i,ij,j->", head_emb, R, tail_emb)

        if temperature == 0.0:
            # Hard threshold
            return 1.0 if score.item() > 0.5 else 0.0
        else:
            # Soft sigmoid
            return torch.sigmoid(score / temperature).item()
