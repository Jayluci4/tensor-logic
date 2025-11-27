"""
Trainer for Tensor Logic Models

Training loop with:
    - Negative sampling
    - Margin ranking loss
    - Gradient clipping
    - Learning rate scheduling
    - Validation and checkpointing
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Optional, Callable, List
import time
from pathlib import Path

from .loss import MarginRankingLoss, RuleRegularizationLoss
from .sampler import NegativeSampler


class Trainer:
    """
    Trainer for Tensor Logic models.

    Implements standard knowledge graph embedding training:
        1. Sample positive triples
        2. Generate negative samples
        3. Compute margin ranking loss
        4. Update parameters via SGD/Adam
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        lr: float = 0.001,
        margin: float = 1.0,
        num_negatives: int = 10,
        weight_decay: float = 0.0,
        grad_clip: float = 1.0,
        device: torch.device = None,
        rules: Optional[List] = None,
        rule_weight: float = 0.1,
    ):
        """
        Initialize trainer.

        Args:
            model: Tensor Logic model to train
            train_loader: DataLoader for training triples
            val_loader: Optional DataLoader for validation
            lr: Learning rate
            margin: Margin for ranking loss
            num_negatives: Number of negative samples per positive
            weight_decay: L2 regularization weight
            grad_clip: Gradient clipping threshold
            device: Torch device
            rules: List of (head_rel, body_rels, rule_type) for regularization
            rule_weight: Weight for rule compliance loss
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Move model to device
        self.model.to(self.device)

        # Optimizer
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

        # Loss function
        self.loss_fn = MarginRankingLoss(margin=margin)

        # Rule regularization
        self.rules = rules or []
        self.rule_loss_fn = RuleRegularizationLoss(weight=rule_weight)

        # Negative sampler
        self.sampler = NegativeSampler(
            num_entities=model.num_entities,
            num_negatives=num_negatives,
            device=self.device,
        )

        # Training config
        self.grad_clip = grad_clip

        # Metrics history
        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "rule_loss": [],
            "val_loss": [],
            "val_mrr": [],
            "val_hits1": [],
            "val_hits10": [],
        }

    def compute_rule_loss(self) -> torch.Tensor:
        """
        Compute rule compliance regularization loss.

        Returns:
            Rule loss (0 if no rules)
        """
        if not self.rules:
            return torch.tensor(0.0, device=self.device)

        rule_loss = torch.tensor(0.0, device=self.device)
        relation_matrices = self.model.get_all_relation_matrices()

        for head_rel, body_rels, rule_type in self.rules:
            R_head = relation_matrices[head_rel]

            if rule_type == "composition":
                # R_head should equal R_body1 @ R_body2 @ ...
                R_composed = relation_matrices[body_rels[0]]
                for i in body_rels[1:]:
                    R_composed = torch.mm(R_composed, relation_matrices[i])
                rule_loss = rule_loss + self.rule_loss_fn(R_head, R_composed)

            elif rule_type == "inverse":
                # R_head should equal R_body^T
                R_body = relation_matrices[body_rels[0]]
                rule_loss = rule_loss + self.rule_loss_fn(R_head, R_body.t())

        return rule_loss

    def train_epoch(self) -> tuple:
        """
        Train for one epoch.

        Returns:
            (average_loss, average_rule_loss)
        """
        self.model.train()
        total_loss = 0.0
        total_rule_loss = 0.0
        num_batches = 0

        for batch in self.train_loader:
            # Unpack batch
            head, relation, tail = batch
            head = head.to(self.device)
            relation = relation.to(self.device)
            tail = tail.to(self.device)

            # Generate negative samples
            neg_head, neg_relation, neg_tail = self.sampler.sample(head, relation, tail)

            # Compute positive scores
            pos_scores = self.model.score(head, relation, tail)

            # Compute negative scores
            batch_size = head.shape[0]
            num_neg = neg_head.shape[1]

            neg_head_flat = neg_head.reshape(-1)
            neg_relation_flat = neg_relation.reshape(-1)
            neg_tail_flat = neg_tail.reshape(-1)

            neg_scores_flat = self.model.score(neg_head_flat, neg_relation_flat, neg_tail_flat)
            neg_scores = neg_scores_flat.reshape(batch_size, num_neg)

            # Compute ranking loss
            ranking_loss = self.loss_fn(pos_scores, neg_scores)

            # Compute rule compliance loss
            rule_loss = self.compute_rule_loss()

            # Total loss
            loss = ranking_loss + rule_loss

            # Backprop
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            if self.grad_clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            self.optimizer.step()

            total_loss += ranking_loss.item()
            total_rule_loss += rule_loss.item()
            num_batches += 1

        return total_loss / num_batches, total_rule_loss / num_batches

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """
        Validate the model.

        Returns:
            Dict with metrics: loss, mrr, hits@1, hits@10
        """
        if self.val_loader is None:
            return {}

        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        # For ranking metrics
        ranks = []

        for batch in self.val_loader:
            head, relation, tail = batch
            head = head.to(self.device)
            relation = relation.to(self.device)
            tail = tail.to(self.device)

            # Score all tails
            all_scores = self.model.score_all_tails(head, relation)

            # Get target scores
            target_scores = all_scores.gather(1, tail.unsqueeze(1)).squeeze(1)

            # Compute ranks
            # rank = number of entities scoring higher + 1
            ranks_batch = (all_scores > target_scores.unsqueeze(1)).sum(dim=1) + 1
            ranks.extend(ranks_batch.tolist())

            # Compute loss (using random negatives)
            neg_head, neg_relation, neg_tail = self.sampler.sample(head, relation, tail)
            neg_scores_flat = self.model.score(
                neg_head.reshape(-1), neg_relation.reshape(-1), neg_tail.reshape(-1)
            )
            neg_scores = neg_scores_flat.reshape(head.shape[0], -1)
            pos_scores = self.model.score(head, relation, tail)
            loss = self.loss_fn(pos_scores, neg_scores)
            total_loss += loss.item()
            num_batches += 1

        # Compute metrics
        ranks = torch.tensor(ranks, dtype=torch.float)
        mrr = (1.0 / ranks).mean().item()
        hits1 = (ranks <= 1).float().mean().item()
        hits10 = (ranks <= 10).float().mean().item()

        return {
            "loss": total_loss / num_batches,
            "mrr": mrr,
            "hits1": hits1,
            "hits10": hits10,
        }

    def train(
        self,
        num_epochs: int,
        validate_every: int = 1,
        checkpoint_dir: Optional[str] = None,
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        """
        Train the model for multiple epochs.

        Args:
            num_epochs: Number of training epochs
            validate_every: Validate every N epochs
            checkpoint_dir: Directory to save checkpoints
            verbose: Print progress

        Returns:
            Training history dict
        """
        if checkpoint_dir:
            Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

        best_mrr = 0.0

        for epoch in range(1, num_epochs + 1):
            start_time = time.time()

            # Train
            train_loss, rule_loss = self.train_epoch()
            self.history["train_loss"].append(train_loss)
            self.history["rule_loss"].append(rule_loss)

            # Validate
            if epoch % validate_every == 0 and self.val_loader is not None:
                val_metrics = self.validate()
                self.history["val_loss"].append(val_metrics["loss"])
                self.history["val_mrr"].append(val_metrics["mrr"])
                self.history["val_hits1"].append(val_metrics["hits1"])
                self.history["val_hits10"].append(val_metrics["hits10"])

                # Save best model
                if checkpoint_dir and val_metrics["mrr"] > best_mrr:
                    best_mrr = val_metrics["mrr"]
                    self.save_checkpoint(f"{checkpoint_dir}/best.pt")

                if verbose:
                    elapsed = time.time() - start_time
                    rule_str = f"Rule: {rule_loss:.4f} | " if self.rules else ""
                    print(
                        f"Epoch {epoch:3d} | "
                        f"Loss: {train_loss:.4f} | "
                        f"{rule_str}"
                        f"Val MRR: {val_metrics['mrr']:.4f} | "
                        f"H@1: {val_metrics['hits1']:.4f} | "
                        f"H@10: {val_metrics['hits10']:.4f} | "
                        f"Time: {elapsed:.1f}s"
                    )
            elif verbose:
                elapsed = time.time() - start_time
                rule_str = f"Rule: {rule_loss:.4f} | " if self.rules else ""
                print(f"Epoch {epoch:3d} | Loss: {train_loss:.4f} | {rule_str}Time: {elapsed:.1f}s")

        # Save final model
        if checkpoint_dir:
            self.save_checkpoint(f"{checkpoint_dir}/final.pt")

        return self.history

    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "history": self.history,
        }, path)

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.history = checkpoint["history"]
