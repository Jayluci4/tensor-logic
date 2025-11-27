"""
Dataset Classes for Tensor Logic

Includes:
    - TripleDataset: Generic dataset for (head, relation, tail) triples
    - FamilyTreeDataset: Synthetic family tree with logical rules
"""

import torch
from torch.utils.data import Dataset
from typing import List, Tuple, Dict, Set, Optional
import random


class TripleDataset(Dataset):
    """
    Dataset of (head, relation, tail) triples.

    Each triple represents a fact: relation(head, tail)
    """

    def __init__(
        self,
        triples: List[Tuple[int, int, int]],
        num_entities: int,
        num_relations: int,
    ):
        """
        Initialize dataset.

        Args:
            triples: List of (head_idx, relation_idx, tail_idx) tuples
            num_entities: Total number of entities
            num_relations: Total number of relations
        """
        self.triples = triples
        self.num_entities = num_entities
        self.num_relations = num_relations

        # Convert to tensor
        self.data = torch.tensor(triples, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.triples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a triple.

        Returns:
            head, relation, tail: Each a scalar tensor
        """
        triple = self.data[idx]
        return triple[0], triple[1], triple[2]


class FamilyTreeDataset:
    """
    Synthetic family tree dataset with logical structure.

    Relations:
        0: parent(x, y)     - x is parent of y
        1: child(x, y)      - x is child of y (inverse of parent)
        2: grandparent(x, y) - x is grandparent of y
        3: sibling(x, y)    - x and y are siblings
        4: ancestor(x, y)   - x is ancestor of y (transitive closure)

    Rules:
        - grandparent(x, z) <- parent(x, y), parent(y, z)
        - child(x, y) <- parent(y, x)
        - sibling(x, y) <- parent(z, x), parent(z, y), x != y
        - ancestor(x, y) <- parent(x, y)
        - ancestor(x, z) <- ancestor(x, y), parent(y, z)

    This dataset tests whether the model can:
        1. Learn entity embeddings
        2. Learn relation embeddings
        3. Infer derived relations (grandparent, ancestor)
        4. Respect logical rules
    """

    PARENT = 0
    CHILD = 1
    GRANDPARENT = 2
    SIBLING = 3
    ANCESTOR = 4

    RELATION_NAMES = ["parent", "child", "grandparent", "sibling", "ancestor"]

    def __init__(
        self,
        num_families: int = 10,
        family_depth: int = 4,
        children_per_parent: int = 2,
        seed: int = 42,
    ):
        """
        Generate synthetic family tree.

        Args:
            num_families: Number of independent family trees
            family_depth: Generations per family
            children_per_parent: Children per parent
            seed: Random seed
        """
        random.seed(seed)
        torch.manual_seed(seed)

        self.num_families = num_families
        self.family_depth = family_depth
        self.children_per_parent = children_per_parent

        # Generate families
        self.entities: List[str] = []
        self.entity2idx: Dict[str, int] = {}

        # Facts by relation
        self.facts: Dict[int, Set[Tuple[int, int]]] = {
            self.PARENT: set(),
            self.CHILD: set(),
            self.GRANDPARENT: set(),
            self.SIBLING: set(),
            self.ANCESTOR: set(),
        }

        self._generate_families()
        self._derive_relations()

        self.num_entities = len(self.entities)
        self.num_relations = len(self.RELATION_NAMES)

    def _add_entity(self, name: str) -> int:
        """Add entity and return its index."""
        if name not in self.entity2idx:
            idx = len(self.entities)
            self.entities.append(name)
            self.entity2idx[name] = idx
        return self.entity2idx[name]

    def _add_fact(self, relation: int, head: int, tail: int):
        """Add a fact."""
        self.facts[relation].add((head, tail))

    def _generate_families(self):
        """Generate family tree structure."""
        for family_id in range(self.num_families):
            # Create root (generation 0)
            generation = [[self._add_entity(f"F{family_id}_G0_P0")]]

            # Create subsequent generations
            for gen in range(1, self.family_depth):
                next_gen = []
                for parent_idx, parent in enumerate(generation[-1]):
                    for child_num in range(self.children_per_parent):
                        child_name = f"F{family_id}_G{gen}_P{len(next_gen)}"
                        child = self._add_entity(child_name)
                        next_gen.append(child)

                        # Add parent fact
                        self._add_fact(self.PARENT, parent, child)

                generation.append(next_gen)

    def _derive_relations(self):
        """Derive relations from parent facts using logical rules."""
        # Child is inverse of parent
        for h, t in self.facts[self.PARENT]:
            self._add_fact(self.CHILD, t, h)

        # Grandparent from parent composition
        parent_dict: Dict[int, List[int]] = {}
        for h, t in self.facts[self.PARENT]:
            if h not in parent_dict:
                parent_dict[h] = []
            parent_dict[h].append(t)

        for x, y_list in parent_dict.items():
            for y in y_list:
                if y in parent_dict:
                    for z in parent_dict[y]:
                        self._add_fact(self.GRANDPARENT, x, z)

        # Sibling: same parent, different person
        children_of: Dict[int, List[int]] = {}
        for h, t in self.facts[self.PARENT]:
            if h not in children_of:
                children_of[h] = []
            children_of[h].append(t)

        for parent, children in children_of.items():
            for i, c1 in enumerate(children):
                for c2 in children[i + 1:]:
                    self._add_fact(self.SIBLING, c1, c2)
                    self._add_fact(self.SIBLING, c2, c1)

        # Ancestor: transitive closure of parent
        # Start with parent
        for h, t in self.facts[self.PARENT]:
            self._add_fact(self.ANCESTOR, h, t)

        # Iterate until no new facts
        changed = True
        while changed:
            changed = False
            new_ancestor = set()
            for x, y in self.facts[self.ANCESTOR]:
                if y in parent_dict:
                    for z in parent_dict[y]:
                        if (x, z) not in self.facts[self.ANCESTOR]:
                            new_ancestor.add((x, z))
                            changed = True
            self.facts[self.ANCESTOR].update(new_ancestor)

    def get_triples(self, relations: Optional[List[int]] = None) -> List[Tuple[int, int, int]]:
        """
        Get all triples for specified relations.

        Args:
            relations: List of relation indices (default: all)

        Returns:
            List of (head, relation, tail) tuples
        """
        if relations is None:
            relations = list(self.facts.keys())

        triples = []
        for rel in relations:
            for h, t in self.facts[rel]:
                triples.append((h, rel, t))

        return triples

    def split(
        self,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42,
    ) -> Tuple[List[Tuple], List[Tuple], List[Tuple]]:
        """
        Split triples into train/val/test.

        Args:
            train_ratio: Fraction for training
            val_ratio: Fraction for validation
            test_ratio: Fraction for testing
            seed: Random seed

        Returns:
            train_triples, val_triples, test_triples
        """
        random.seed(seed)

        # Get all triples
        all_triples = self.get_triples()
        random.shuffle(all_triples)

        n = len(all_triples)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        train = all_triples[:n_train]
        val = all_triples[n_train:n_train + n_val]
        test = all_triples[n_train + n_val:]

        return train, val, test

    def create_datasets(
        self,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        seed: int = 42,
    ) -> Tuple[TripleDataset, TripleDataset, TripleDataset]:
        """
        Create train/val/test TripleDatasets.

        Returns:
            train_dataset, val_dataset, test_dataset
        """
        train, val, test = self.split(train_ratio, val_ratio, 1 - train_ratio - val_ratio, seed)

        return (
            TripleDataset(train, self.num_entities, self.num_relations),
            TripleDataset(val, self.num_entities, self.num_relations),
            TripleDataset(test, self.num_entities, self.num_relations),
        )

    def get_rules(self) -> List[Tuple[int, List[int], str]]:
        """
        Get logical rules in this dataset.

        Returns:
            List of (head_rel, body_rels, rule_type)
        """
        return [
            (self.GRANDPARENT, [self.PARENT, self.PARENT], "composition"),
            (self.CHILD, [self.PARENT], "inverse"),
            # Ancestor is more complex (transitive closure)
        ]

    def __repr__(self) -> str:
        return (
            f"FamilyTreeDataset(\n"
            f"  num_entities={self.num_entities},\n"
            f"  num_relations={self.num_relations},\n"
            f"  facts={{{', '.join(f'{self.RELATION_NAMES[k]}: {len(v)}' for k, v in self.facts.items())}}}\n"
            f")"
        )
