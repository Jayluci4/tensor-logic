#!/usr/bin/env python3
"""
Train Tensor Logic Model on Family Tree Dataset

This script:
    1. Generates synthetic family tree data
    2. Trains a bilinear Tensor Logic model
    3. Evaluates on held-out data
    4. Checks rule compliance
    5. Performs spectral analysis
    6. Visualizes results
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import argparse
import json
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

from tensor_logic.models import BilinearModel
from tensor_logic.training import Trainer
from tensor_logic.data import FamilyTreeDataset, create_dataloaders
from tensor_logic.analysis import SpectralAnalyzer, RuleComplianceChecker
from tensor_logic.utils.metrics import format_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Train Tensor Logic model")
    parser.add_argument("--embedding_dim", type=int, default=64)
    parser.add_argument("--num_families", type=int, default=20)
    parser.add_argument("--family_depth", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--num_negatives", type=int, default=10)
    parser.add_argument("--rule_weight", type=float, default=0.0,
                        help="Weight for rule compliance regularization (0 = disabled)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def plot_training_history(history, output_dir):
    """Plot training curves."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Loss
    ax = axes[0, 0]
    ax.plot(history["train_loss"], label="Train")
    if history["val_loss"]:
        ax.plot(history["val_loss"], label="Val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # MRR
    ax = axes[0, 1]
    if history["val_mrr"]:
        ax.plot(history["val_mrr"])
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MRR")
        ax.set_title("Validation MRR")
        ax.grid(True, alpha=0.3)

    # Hits@1
    ax = axes[1, 0]
    if history["val_hits1"]:
        ax.plot(history["val_hits1"])
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Hits@1")
        ax.set_title("Validation Hits@1")
        ax.grid(True, alpha=0.3)

    # Hits@10
    ax = axes[1, 1]
    if history["val_hits10"]:
        ax.plot(history["val_hits10"])
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Hits@10")
        ax.set_title("Validation Hits@10")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/training_curves.png", dpi=150)
    plt.close()


def plot_spectral_analysis(entity_stats, relation_stats, relation_names, output_dir):
    """Plot spectral analysis results."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Entity embedding singular values
    ax = axes[0, 0]
    sv = entity_stats.singular_values.numpy()
    ax.semilogy(sv)
    ax.set_xlabel("Singular Value Index")
    ax.set_ylabel("Singular Value (log scale)")
    ax.set_title("Entity Embedding Spectrum")
    ax.grid(True, alpha=0.3)

    # Entity embedding cumulative energy
    ax = axes[0, 1]
    cumulative = np.cumsum(entity_stats.energy_distribution.numpy())
    ax.plot(cumulative)
    ax.axhline(y=0.9, color='r', linestyle='--', label='90% energy')
    ax.axhline(y=0.99, color='g', linestyle='--', label='99% energy')
    ax.set_xlabel("Number of Dimensions")
    ax.set_ylabel("Cumulative Energy")
    ax.set_title("Entity Embedding Energy Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Relation effective ranks
    ax = axes[1, 0]
    ranks = [relation_stats[i].effective_rank for i in range(len(relation_stats))]
    ax.bar(relation_names, ranks)
    ax.set_xlabel("Relation")
    ax.set_ylabel("Effective Rank")
    ax.set_title("Relation Effective Ranks")
    ax.tick_params(axis='x', rotation=45)

    # Relation principal ratios
    ax = axes[1, 1]
    ratios = [relation_stats[i].principal_ratio for i in range(len(relation_stats))]
    ax.bar(relation_names, ratios)
    ax.set_xlabel("Relation")
    ax.set_ylabel("Energy in Top 10%")
    ax.set_title("Relation Energy Concentration")
    ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/spectral_analysis.png", dpi=150)
    plt.close()


def plot_rule_compliance(compliance_results, output_dir):
    """Plot rule compliance results."""
    fig, ax = plt.subplots(figsize=(10, 6))

    rules = [r.rule_name for r in compliance_results]
    errors = [r.relative_error for r in compliance_results]

    colors = ['green' if e < 0.1 else 'orange' if e < 0.5 else 'red' for e in errors]
    ax.barh(rules, errors, color=colors)
    ax.axvline(x=0.1, color='g', linestyle='--', label='Good (<0.1)')
    ax.axvline(x=0.5, color='orange', linestyle='--', label='Acceptable (<0.5)')
    ax.set_xlabel("Relative Error")
    ax.set_title("Rule Compliance (lower is better)")
    ax.legend()

    plt.tight_layout()
    plt.savefig(f"{output_dir}/rule_compliance.png", dpi=150)
    plt.close()


def main():
    args = parse_args()

    # Setup
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(" Tensor Logic Training")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Output: {output_dir}")
    print()

    # Create dataset
    print("Generating family tree dataset...")
    dataset = FamilyTreeDataset(
        num_families=args.num_families,
        family_depth=args.family_depth,
        seed=args.seed,
    )
    print(dataset)
    print()

    # Create train/val/test splits
    train_ds, val_ds, test_ds = dataset.create_datasets(
        train_ratio=0.8,
        val_ratio=0.1,
        seed=args.seed,
    )

    print(f"Train: {len(train_ds)} triples")
    print(f"Val: {len(val_ds)} triples")
    print(f"Test: {len(test_ds)} triples")
    print()

    # Create data loaders
    train_loader, val_loader, test_loader = create_dataloaders(
        train_ds, val_ds, test_ds,
        batch_size=args.batch_size,
    )

    # Create model
    print("Creating model...")
    model = BilinearModel(
        num_entities=dataset.num_entities,
        num_relations=dataset.num_relations,
        embedding_dim=args.embedding_dim,
        device=device,
    )
    print(f"Entity embeddings: {model.entity_embeddings.weight.shape}")
    print(f"Relation embeddings: {model.relation_embeddings.weight.shape}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()

    # Get rules for regularization
    rules = dataset.get_rules() if args.rule_weight > 0 else []

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=args.lr,
        margin=args.margin,
        num_negatives=args.num_negatives,
        device=device,
        rules=rules,
        rule_weight=args.rule_weight,
    )

    if args.rule_weight > 0:
        print(f"Rule regularization enabled (weight={args.rule_weight})")
        print(f"Rules: {len(rules)}")

    # Add positive triples to sampler for filtering
    all_triples = torch.tensor(dataset.get_triples())
    trainer.sampler.add_positives(all_triples)

    # Train
    print("Training...")
    print("-" * 60)
    history = trainer.train(
        num_epochs=args.epochs,
        validate_every=5,
        checkpoint_dir=str(output_dir),
        verbose=True,
    )
    print("-" * 60)
    print()

    # Plot training curves
    print("Plotting training curves...")
    plot_training_history(history, output_dir)

    # Final evaluation on test set
    print("Evaluating on test set...")
    trainer.val_loader = test_loader
    test_metrics = trainer.validate()
    print(f"Test: {format_metrics({'mrr': test_metrics['mrr'], 'mr': 0, 'hits@1': test_metrics['hits1'], 'hits@3': 0, 'hits@10': test_metrics['hits10']})}")
    print()

    # Spectral analysis
    print("Performing spectral analysis...")
    analyzer = SpectralAnalyzer(device=device)

    entity_emb = model.get_entity_embeddings()
    relation_matrices = model.get_all_relation_matrices()

    entity_stats = analyzer.analyze_entity_embeddings(entity_emb)
    relation_stats = analyzer.analyze_relation_embeddings(relation_matrices)

    print(analyzer.summary(entity_emb, relation_matrices, dataset.RELATION_NAMES))

    plot_spectral_analysis(
        entity_stats, relation_stats,
        dataset.RELATION_NAMES, output_dir
    )

    # Rule compliance
    print("Checking rule compliance...")
    checker = RuleComplianceChecker(device=device)
    rules = dataset.get_rules()
    compliance_results = checker.check_all_rules(
        relation_matrices, rules, dataset.RELATION_NAMES
    )
    print(checker.summary(compliance_results))

    plot_rule_compliance(compliance_results, output_dir)

    # Save results
    results = {
        "config": vars(args),
        "dataset": {
            "num_entities": dataset.num_entities,
            "num_relations": dataset.num_relations,
            "num_train": len(train_ds),
            "num_val": len(val_ds),
            "num_test": len(test_ds),
        },
        "test_metrics": test_metrics,
        "spectral": {
            "entity_effective_rank": entity_stats.effective_rank,
            "entity_principal_ratio": entity_stats.principal_ratio,
            "relation_effective_ranks": {
                dataset.RELATION_NAMES[i]: relation_stats[i].effective_rank
                for i in range(len(relation_stats))
            },
        },
        "rule_compliance": {
            r.rule_name: r.relative_error for r in compliance_results
        },
    }

    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print()
    print("=" * 60)
    print(" Training Complete")
    print("=" * 60)
    print(f"Results saved to: {output_dir}")
    print(f"  - training_curves.png")
    print(f"  - spectral_analysis.png")
    print(f"  - rule_compliance.png")
    print(f"  - results.json")
    print(f"  - best.pt (best model)")
    print(f"  - final.pt (final model)")


if __name__ == "__main__":
    main()
