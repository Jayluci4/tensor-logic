#!/usr/bin/env python3
"""
Compositional Generalization Experiment

Key hypothesis: If R_grandparent = R_parent @ R_parent holds algebraically,
then R_great_grandparent = R_parent @ R_parent @ R_parent should work
WITHOUT ever seeing great-grandparent facts during training.

This tests whether Tensor Logic enables reasoning about unseen relation types
through pure composition - something current LLMs cannot do.

Experiment design:
    Training: parent facts, grandparent facts
    Rules: R_gp = R_p @ R_p (enforced via regularization)
    Test: great_grandparent facts (NEVER seen during training)
    Inference: R_ggp = R_p @ R_p @ R_p (composed at test time)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

from tensor_logic.models import BilinearModel
from tensor_logic.training import Trainer
from tensor_logic.data import create_dataloaders
from torch.utils.data import TensorDataset


class CompositionalFamilyTree:
    """
    Family tree dataset with multiple composition depths.

    Relations:
        0: parent (base)
        1: grandparent = parent @ parent (depth 2)
        2: great_grandparent = parent @ parent @ parent (depth 3)
        3: great_great_grandparent = parent^4 (depth 4)
    """

    PARENT = 0
    GRANDPARENT = 1
    GREAT_GRANDPARENT = 2
    GREAT_GREAT_GRANDPARENT = 3

    RELATION_NAMES = ["parent", "grandparent", "great_grandparent", "great_great_grandparent"]

    def __init__(self, num_families: int = 20, chain_length: int = 6, seed: int = 42):
        """
        Args:
            num_families: Number of family chains to generate
            chain_length: Length of each chain (person_0 -> person_1 -> ... -> person_n)
            seed: Random seed
        """
        np.random.seed(seed)

        self.num_families = num_families
        self.chain_length = chain_length

        # Generate entities
        self.entities = []
        self.entity_to_idx = {}

        # Generate family chains
        self.parent_facts = []  # (child, parent)
        self.grandparent_facts = []  # (grandchild, grandparent)
        self.great_grandparent_facts = []  # (great_grandchild, great_grandparent)
        self.great_great_grandparent_facts = []  # depth 4

        entity_idx = 0

        for fam in range(num_families):
            # Create a chain: person_0 is oldest ancestor
            chain = []
            for i in range(chain_length):
                name = f"family{fam}_gen{i}"
                self.entities.append(name)
                self.entity_to_idx[name] = entity_idx
                chain.append(entity_idx)
                entity_idx += 1

            # Generate facts at each depth
            # parent: gen_i -> gen_{i-1} means gen_i's parent is gen_{i-1}
            for i in range(1, chain_length):
                self.parent_facts.append((chain[i], chain[i-1]))

            # grandparent: gen_i -> gen_{i-2}
            for i in range(2, chain_length):
                self.grandparent_facts.append((chain[i], chain[i-2]))

            # great_grandparent: gen_i -> gen_{i-3}
            for i in range(3, chain_length):
                self.great_grandparent_facts.append((chain[i], chain[i-3]))

            # great_great_grandparent: gen_i -> gen_{i-4}
            for i in range(4, chain_length):
                self.great_great_grandparent_facts.append((chain[i], chain[i-4]))

        self.num_entities = len(self.entities)
        self.num_relations = 4

    def get_training_triples(self):
        """Get training data: ONLY parent and grandparent facts."""
        triples = []

        for h, t in self.parent_facts:
            triples.append((h, self.PARENT, t))

        for h, t in self.grandparent_facts:
            triples.append((h, self.GRANDPARENT, t))

        return triples

    def get_test_triples(self, depth: int):
        """Get test data at specific composition depth."""
        if depth == 3:
            facts = self.great_grandparent_facts
            rel = self.GREAT_GRANDPARENT
        elif depth == 4:
            facts = self.great_great_grandparent_facts
            rel = self.GREAT_GREAT_GRANDPARENT
        else:
            raise ValueError(f"Unsupported depth: {depth}")

        return [(h, rel, t) for h, t in facts]

    def get_rules(self):
        """Get composition rules for training regularization."""
        return [
            # grandparent = parent @ parent
            (self.GRANDPARENT, [self.PARENT, self.PARENT], "composition"),
        ]

    def __repr__(self):
        return (
            f"CompositionalFamilyTree(\n"
            f"  entities={self.num_entities},\n"
            f"  parent_facts={len(self.parent_facts)},\n"
            f"  grandparent_facts={len(self.grandparent_facts)},\n"
            f"  great_grandparent_facts={len(self.great_grandparent_facts)},\n"
            f"  great_great_grandparent_facts={len(self.great_great_grandparent_facts)}\n"
            f")"
        )


def evaluate_with_composition(
    model: BilinearModel,
    test_triples: list,
    composition_depth: int,
    device: torch.device,
) -> dict:
    """
    Evaluate using composed relation matrix.

    For depth 3: R_test = R_parent @ R_parent @ R_parent
    For depth 4: R_test = R_parent @ R_parent @ R_parent @ R_parent
    """
    model.eval()

    # Get parent relation matrix
    R_parent = model.get_relation_matrix(0)  # Parent is relation 0

    # Compose to target depth
    R_composed = R_parent.clone()
    for _ in range(composition_depth - 1):
        R_composed = torch.mm(R_composed, R_parent)

    # Evaluate each test triple
    ranks = []

    with torch.no_grad():
        for head_idx, rel_idx, tail_idx in test_triples:
            # Get head embedding
            head_emb = model.get_entity_embeddings()[head_idx]  # [d]

            # Score all tails using composed relation
            all_tail_emb = model.get_entity_embeddings()  # [num_entities, d]

            # score = h^T @ R_composed @ t for all t
            # h^T @ R_composed = [1, d] @ [d, d] = [1, d]
            transformed = torch.mv(R_composed.t(), head_emb)  # [d]

            # scores = transformed @ all_tails^T = [d] @ [d, num_entities] = [num_entities]
            scores = torch.mv(all_tail_emb, transformed)

            # Get rank of correct tail
            target_score = scores[tail_idx]
            rank = (scores > target_score).sum().item() + 1
            ranks.append(rank)

    ranks = torch.tensor(ranks, dtype=torch.float)

    return {
        "mrr": (1.0 / ranks).mean().item(),
        "hits1": (ranks <= 1).float().mean().item(),
        "hits3": (ranks <= 3).float().mean().item(),
        "hits10": (ranks <= 10).float().mean().item(),
        "mean_rank": ranks.mean().item(),
    }


def evaluate_random_baseline(
    num_entities: int,
    num_test: int,
) -> dict:
    """Random baseline for comparison."""
    # Random ranks uniformly distributed
    expected_mr = (num_entities + 1) / 2
    expected_mrr = sum(1/r for r in range(1, num_entities+1)) / num_entities
    expected_hits1 = 1 / num_entities
    expected_hits10 = min(10, num_entities) / num_entities

    return {
        "mrr": expected_mrr,
        "hits1": expected_hits1,
        "hits3": 3 / num_entities,
        "hits10": expected_hits10,
        "mean_rank": expected_mr,
    }


def main():
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print(" Compositional Generalization Experiment")
    print("=" * 70)
    print(f"Device: {device}")
    print()

    # Create dataset
    print("Creating dataset...")
    dataset = CompositionalFamilyTree(
        num_families=30,
        chain_length=6,
        seed=42,
    )
    print(dataset)
    print()

    # Training data: ONLY parent and grandparent
    train_triples = dataset.get_training_triples()
    print(f"Training triples: {len(train_triples)}")
    print(f"  - Contains: parent, grandparent facts")
    print(f"  - Does NOT contain: great_grandparent, great_great_grandparent")
    print()

    # Test data: great_grandparent (NEVER seen during training)
    test_triples_depth3 = dataset.get_test_triples(depth=3)
    test_triples_depth4 = dataset.get_test_triples(depth=4)
    print(f"Test triples (depth 3, great_grandparent): {len(test_triples_depth3)}")
    print(f"Test triples (depth 4, great_great_grandparent): {len(test_triples_depth4)}")
    print()

    # Create data loaders
    train_tensor = torch.tensor(train_triples)
    train_ds = TensorDataset(train_tensor[:, 0], train_tensor[:, 1], train_tensor[:, 2])
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=64, shuffle=True)

    # Create model
    print("Creating model...")
    embedding_dim = 64
    model = BilinearModel(
        num_entities=dataset.num_entities,
        num_relations=dataset.num_relations,
        embedding_dim=embedding_dim,
        device=device,
    )
    print(f"Entity embeddings: {model.entity_embeddings.weight.shape}")
    print(f"Relation embeddings: {model.relation_embeddings.weight.shape}")
    print()

    # Train WITH rule regularization
    print("Training with rule regularization...")
    print("Rule: R_grandparent = R_parent @ R_parent")
    print("-" * 70)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=None,
        lr=0.01,
        margin=1.0,
        num_negatives=10,
        device=device,
        rules=dataset.get_rules(),
        rule_weight=1.0,  # Strong rule enforcement
    )
    trainer.sampler.add_positives(train_tensor)

    # Train
    num_epochs = 100
    for epoch in range(1, num_epochs + 1):
        train_loss, rule_loss = trainer.train_epoch()
        if epoch % 20 == 0:
            print(f"Epoch {epoch:3d} | Loss: {train_loss:.4f} | Rule: {rule_loss:.6f}")

    print("-" * 70)
    print()

    # Evaluate rule compliance
    print("Checking rule compliance...")
    R_parent = model.get_relation_matrix(0)
    R_grandparent = model.get_relation_matrix(1)
    R_composed = torch.mm(R_parent, R_parent)

    error = torch.norm(R_grandparent - R_composed) / torch.norm(R_grandparent)
    print(f"||R_gp - R_p @ R_p|| / ||R_gp|| = {error.item():.6f}")
    print()

    # Key experiment: Evaluate on UNSEEN composition depths
    print("=" * 70)
    print(" ZERO-SHOT COMPOSITIONAL GENERALIZATION RESULTS")
    print("=" * 70)
    print()
    print("Testing on relations NEVER seen during training:")
    print("  - great_grandparent: R = R_parent @ R_parent @ R_parent")
    print("  - great_great_grandparent: R = R_parent^4")
    print()

    # Random baseline
    random_baseline = evaluate_random_baseline(dataset.num_entities, len(test_triples_depth3))
    print("Random baseline:")
    print(f"  MRR: {random_baseline['mrr']:.4f} | H@1: {random_baseline['hits1']:.4f} | H@10: {random_baseline['hits10']:.4f}")
    print()

    # Depth 3: great_grandparent
    results_depth3 = evaluate_with_composition(
        model, test_triples_depth3, composition_depth=3, device=device
    )
    print("Depth 3 (great_grandparent) - ZERO-SHOT:")
    print(f"  MRR: {results_depth3['mrr']:.4f} | H@1: {results_depth3['hits1']:.4f} | H@10: {results_depth3['hits10']:.4f} | MR: {results_depth3['mean_rank']:.1f}")

    improvement_mrr_d3 = results_depth3['mrr'] / random_baseline['mrr']
    print(f"  Improvement over random: {improvement_mrr_d3:.1f}x MRR")
    print()

    # Depth 4: great_great_grandparent
    results_depth4 = evaluate_with_composition(
        model, test_triples_depth4, composition_depth=4, device=device
    )
    print("Depth 4 (great_great_grandparent) - ZERO-SHOT:")
    print(f"  MRR: {results_depth4['mrr']:.4f} | H@1: {results_depth4['hits1']:.4f} | H@10: {results_depth4['hits10']:.4f} | MR: {results_depth4['mean_rank']:.1f}")

    improvement_mrr_d4 = results_depth4['mrr'] / random_baseline['mrr']
    print(f"  Improvement over random: {improvement_mrr_d4:.1f}x MRR")
    print()

    # Summary
    print("=" * 70)
    print(" SUMMARY")
    print("=" * 70)
    print()
    print("Trained on: parent (depth 1), grandparent (depth 2)")
    print("Tested on: great_grandparent (depth 3), great_great_grandparent (depth 4)")
    print()
    print("| Depth | Relation               | MRR    | H@1    | H@10   | vs Random |")
    print("|-------|------------------------|--------|--------|--------|-----------|")
    print(f"|   3   | great_grandparent      | {results_depth3['mrr']:.4f} | {results_depth3['hits1']:.4f} | {results_depth3['hits10']:.4f} | {improvement_mrr_d3:.1f}x      |")
    print(f"|   4   | great_great_grandparent| {results_depth4['mrr']:.4f} | {results_depth4['hits1']:.4f} | {results_depth4['hits10']:.4f} | {improvement_mrr_d4:.1f}x      |")
    print()

    if results_depth3['hits1'] > 0.5 and results_depth4['hits1'] > 0.3:
        print("RESULT: Compositional generalization DEMONSTRATED")
        print("The model correctly predicts relations it has NEVER seen,")
        print("purely through algebraic composition of learned matrices.")
    elif results_depth3['mrr'] > 0.3:
        print("RESULT: Partial compositional generalization")
        print("The model shows some ability to compose relations.")
    else:
        print("RESULT: Limited compositional generalization")
        print("Further investigation needed.")

    print()
    print("=" * 70)

    # Save visualization
    output_dir = Path("results_compositional")
    output_dir.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    depths = [2, 3, 4]
    metrics = ['MRR', 'Hits@1', 'Hits@10']

    # For depth 2, use training performance (we'll compute it)
    # Actually, let's just show depth 3 and 4 comparison
    x = np.arange(len(metrics))
    width = 0.35

    depth3_vals = [results_depth3['mrr'], results_depth3['hits1'], results_depth3['hits10']]
    depth4_vals = [results_depth4['mrr'], results_depth4['hits1'], results_depth4['hits10']]
    random_vals = [random_baseline['mrr'], random_baseline['hits1'], random_baseline['hits10']]

    bars1 = ax.bar(x - width, depth3_vals, width, label='Depth 3 (great_grandparent)', color='steelblue')
    bars2 = ax.bar(x, depth4_vals, width, label='Depth 4 (great_great_grandparent)', color='darkorange')
    bars3 = ax.bar(x + width, random_vals, width, label='Random Baseline', color='gray', alpha=0.5)

    ax.set_ylabel('Score')
    ax.set_title('Zero-Shot Compositional Generalization\n(Relations NEVER seen during training)')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / "compositional_generalization.png", dpi=150)
    plt.close()

    print(f"Visualization saved to: {output_dir}/compositional_generalization.png")


if __name__ == "__main__":
    main()
