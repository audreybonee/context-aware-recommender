import logging
from typing import Dict, List, Optional, Tuple

import networkx as nx
from SpreadPy.Models.models import BaseSpreading

from fathom.config import (
    SAN_DECAY,
    SAN_INITIAL_ENERGY,
    SAN_ITERATIONS,
    SAN_MIN_ACTIVATION,
    SAN_RETENTION,
    SAN_SUPPRESS,
)

logger = logging.getLogger(__name__)


class SpreadingActivationEngine:
    """Runs spreading activation on the book knowledge graph."""

    def __init__(
        self,
        graph: nx.Graph,
        retention: float = SAN_RETENTION,
        decay: float = SAN_DECAY,
        suppress: float = SAN_SUPPRESS,
        iterations: int = SAN_ITERATIONS,
    ):
        self.graph = graph
        self.retention = retention
        self.decay = decay
        self.suppress = suppress
        self.iterations = iterations

    def activate_from_books(
        self,
        seed_isbns: List[str],
        initial_energy: float = SAN_INITIAL_ENERGY,
    ) -> Dict[str, float]:
        """Seed activation at book nodes and spread through the graph.

        Args:
            seed_isbns: ISBN-13s of seed books to activate.
            initial_energy: Starting energy for each seed node.

        Returns:
            Dict mapping node_id → final activation score, sorted descending.
        """
        seed_node_ids = {f"book:{isbn}" for isbn in seed_isbns}
        valid_seeds = seed_node_ids & set(self.graph.nodes())

        if not valid_seeds:
            logger.warning("No valid seed nodes found in graph.")
            return {}

        logger.info(
            "Running SAN with %d seeds, %d iterations...",
            len(valid_seeds),
            self.iterations,
        )

        # Initialize the spreading model
        model = BaseSpreading(
            self.graph,
            retention=self.retention,
            decay=self.decay,
            suppress=self.suppress,
            weighted=False,
        )

        # Set initial activation: seed nodes get energy, others get 0
        initial_status = {
            node: initial_energy if node in valid_seeds else 0.0
            for node in self.graph.nodes()
        }
        model.status = initial_status

        # Run spreading iterations
        results = model.iteration_bunch(self.iterations)

        # Extract final activation values
        final_status = results[-1]["status"]

        return dict(
            sorted(final_status.items(), key=lambda x: x[1], reverse=True)
        )

    def get_activated_books(
        self,
        activation_results: Dict[str, float],
        exclude_seeds: Optional[List[str]] = None,
        top_k: int = 20,
        min_activation: float = SAN_MIN_ACTIVATION,
    ) -> List[Tuple[str, float]]:
        """Filter activation results to book nodes only.

        Args:
            activation_results: Full activation dict from activate_from_books.
            exclude_seeds: ISBNs to exclude (the seed books themselves).
            top_k: Maximum number of results.
            min_activation: Minimum activation threshold.

        Returns:
            List of (isbn13, activation_score) tuples, sorted by score.
        """
        exclude_set = {f"book:{isbn}" for isbn in (exclude_seeds or [])}

        book_activations = [
            (node_id.replace("book:", ""), score)
            for node_id, score in activation_results.items()
            if node_id.startswith("book:")
            and node_id not in exclude_set
            and score >= min_activation
        ]

        return sorted(book_activations, key=lambda x: x[1], reverse=True)[:top_k]

    def explain_path(
        self,
        source_isbn: str,
        target_isbn: str,
    ) -> List[str]:
        """Find the shortest path between two books through the KG.

        Used for explainability: shows which concepts connect two books.

        Returns:
            List of node IDs forming the path, or empty list if no path.
        """
        source = f"book:{source_isbn}"
        target = f"book:{target_isbn}"

        if source not in self.graph or target not in self.graph:
            return []

        try:
            return nx.shortest_path(self.graph, source, target)
        except nx.NetworkXNoPath:
            return []

    def explain_activation(
        self,
        seed_isbns: List[str],
        target_isbn: str,
    ) -> Optional[dict]:
        """Explain why a target book was activated by finding the best
        connecting path to any seed book.

        Returns:
            Dict with 'seed_isbn', 'path', and 'via_concepts', or None.
        """
        best_path: Optional[List[str]] = None
        best_seed: Optional[str] = None

        for seed_isbn in seed_isbns:
            path = self.explain_path(seed_isbn, target_isbn)
            if path and (best_path is None or len(path) < len(best_path)):
                best_path = path
                best_seed = seed_isbn

        if not best_path or not best_seed:
            return None

        # Extract the intermediate concept/author/location nodes
        via_concepts = [
            node for node in best_path if not node.startswith("book:")
        ]

        return {
            "seed_isbn": best_seed,
            "path": best_path,
            "via_concepts": via_concepts,
        }
