"""Knowledge Graph construction and persistence using NetworkX.

Builds a heterogeneous graph where books, concepts, authors, and
locations are nodes connected by typed edges. The graph is serialized
as GraphML for fast loading at runtime.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Set

import networkx as nx
import pandas as pd

from fathom.config import KNOWLEDGE_GRAPH_PATH
from fathom.ontology import CONCEPT_TO_CLUSTER
from fathom.schemas import RelationType, Triple

logger = logging.getLogger(__name__)

# Relation → node type prefix mapping
_RELATION_PREFIX: Dict[RelationType, str] = {
    RelationType.HAS_THEME: "concept",
    RelationType.HAS_MOOD: "concept",
    RelationType.HAS_TROPE: "concept",
    RelationType.HAS_FORM: "concept",
    RelationType.SET_IN: "location",
    RelationType.WRITTEN_BY: "author",
    RelationType.HAS_CATEGORY: "category",
}


class BookKnowledgeGraph:
    """Heterogeneous knowledge graph for the Fathom recommender."""

    def __init__(self):
        self.graph: nx.Graph = nx.Graph()

    def build_from_triples(
        self,
        triples: List[Triple],
        books_df: pd.DataFrame,
    ) -> "BookKnowledgeGraph":
        """Build the complete graph from triples and book metadata.

        Args:
            triples: All extracted and deduplicated triples.
            books_df: DataFrame with isbn13, title, authors, average_rating, etc.

        Returns:
            self, for chaining.
        """
        # Add book nodes
        for _, row in books_df.iterrows():
            isbn = str(row["isbn13"])
            self._add_book_node(
                isbn,
                {
                    "title": str(row.get("title", "")),
                    "authors": str(row.get("authors", "")),
                    "average_rating": float(row.get("average_rating", 0)),
                    "simple_categories": str(row.get("simple_categories", "")),
                },
            )

        # Add edges (and target nodes) from triples
        for triple in triples:
            self._add_edge(triple)

        logger.info(
            "Graph built: %d nodes, %d edges.",
            self.graph.number_of_nodes(),
            self.graph.number_of_edges(),
        )
        return self

    def _add_book_node(self, isbn13: str, metadata: dict) -> None:
        node_id = f"book:{isbn13}"
        self.graph.add_node(node_id, node_type="book", **metadata)

    def _add_edge(self, triple: Triple) -> None:
        source = f"book:{triple.subject_isbn}"
        prefix = _RELATION_PREFIX.get(triple.relation, "concept")
        target = f"{prefix}:{triple.object_concept}"

        # Ensure target node exists with appropriate type
        if target not in self.graph:
            node_attrs: Dict[str, str] = {"node_type": prefix}
            if prefix == "concept":
                cluster = CONCEPT_TO_CLUSTER.get(triple.object_concept, "unknown")
                node_attrs["cluster"] = (
                    cluster.value if hasattr(cluster, "value") else str(cluster)
                )
            self.graph.add_node(target, **node_attrs)

        self.graph.add_edge(source, target, relation=triple.relation.value)

    def save(self, path: str | Path = KNOWLEDGE_GRAPH_PATH) -> None:
        """Serialize graph to GraphML."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        nx.write_graphml(self.graph, str(path))
        logger.info("Graph saved to %s", path)

    @classmethod
    def load(cls, path: str | Path = KNOWLEDGE_GRAPH_PATH) -> "BookKnowledgeGraph":
        """Load graph from GraphML file."""
        instance = cls()
        instance.graph = nx.read_graphml(str(path))
        logger.info(
            "Graph loaded: %d nodes, %d edges.",
            instance.graph.number_of_nodes(),
            instance.graph.number_of_edges(),
        )
        return instance

    def get_book_neighbors(
        self,
        isbn13: str,
        relation: Optional[RelationType] = None,
    ) -> List[str]:
        """Get all concept/author/location nodes connected to a book."""
        node_id = f"book:{isbn13}"
        if node_id not in self.graph:
            return []
        neighbors = []
        for neighbor in self.graph.neighbors(node_id):
            if relation is not None:
                edge_data = self.graph.edges[node_id, neighbor]
                if edge_data.get("relation") != relation.value:
                    continue
            neighbors.append(neighbor)
        return neighbors

    def get_concept_books(self, concept: str) -> List[str]:
        """Get all book isbn13s connected to a given concept node."""
        node_id = f"concept:{concept}"
        if node_id not in self.graph:
            return []
        return [
            n.replace("book:", "")
            for n in self.graph.neighbors(node_id)
            if n.startswith("book:")
        ]

    def get_all_books(self) -> Set[str]:
        """Return set of all book isbn13s in the graph."""
        return {
            n.replace("book:", "")
            for n in self.graph.nodes()
            if n.startswith("book:")
        }

    def get_stats(self) -> dict:
        """Return graph statistics: node/edge counts by type, density."""
        node_types: Dict[str, int] = {}
        for _, data in self.graph.nodes(data=True):
            nt = data.get("node_type", "unknown")
            node_types[nt] = node_types.get(nt, 0) + 1

        edge_types: Dict[str, int] = {}
        for _, _, data in self.graph.edges(data=True):
            rt = data.get("relation", "unknown")
            edge_types[rt] = edge_types.get(rt, 0) + 1

        return {
            "total_nodes": self.graph.number_of_nodes(),
            "total_edges": self.graph.number_of_edges(),
            "node_types": node_types,
            "edge_types": edge_types,
            "density": nx.density(self.graph),
            "connected_components": nx.number_connected_components(self.graph),
        }
