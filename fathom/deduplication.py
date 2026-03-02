"""Tier 3: Vector-Assisted Deduplication.

Embeds all unique concept strings and merges near-duplicates
(cosine similarity > threshold) into canonical forms. This bridges
the neural and symbolic layers during data prep.
"""

import logging
from collections import Counter
from typing import Dict, List

import numpy as np
from langchain_openai import OpenAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity

from fathom.config import DEDUP_SIMILARITY_THRESHOLD
from fathom.schemas import Triple

logger = logging.getLogger(__name__)


class UnionFind:
    """Disjoint-set data structure for clustering similar concepts."""

    def __init__(self, elements: List[str]):
        self.parent: Dict[str, str] = {e: e for e in elements}
        self.rank: Dict[str, int] = {e: 0 for e in elements}

    def find(self, x: str) -> str:
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: str, y: str) -> None:
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1


class ConceptDeduplicator:
    """Merges near-duplicate concept strings using vector similarity."""

    def __init__(
        self,
        threshold: float = DEDUP_SIMILARITY_THRESHOLD,
        embeddings: OpenAIEmbeddings | None = None,
    ):
        self.threshold = threshold
        self.embeddings = embeddings or OpenAIEmbeddings()

    def deduplicate_concepts(
        self,
        concepts: List[str],
        concept_counts: Counter | None = None,
    ) -> Dict[str, str]:
        """Cluster near-duplicate concepts and return a canonical mapping.

        Args:
            concepts: All unique concept strings to consider.
            concept_counts: Optional frequency counts to prefer
                           the most common variant as canonical.

        Returns:
            Dict mapping each concept to its canonical form.
            Concepts with no near-duplicates map to themselves.
        """
        if len(concepts) <= 1:
            return {c: c for c in concepts}

        logger.info("Embedding %d unique concepts for deduplication...", len(concepts))
        vectors = self.embeddings.embed_documents(concepts)
        sim_matrix = cosine_similarity(np.array(vectors))

        uf = UnionFind(concepts)
        for i in range(len(concepts)):
            for j in range(i + 1, len(concepts)):
                if sim_matrix[i][j] >= self.threshold:
                    uf.union(concepts[i], concepts[j])
                    logger.debug(
                        "Merging '%s' and '%s' (similarity=%.3f)",
                        concepts[i],
                        concepts[j],
                        sim_matrix[i][j],
                    )

        # Group concepts by their root
        clusters: Dict[str, List[str]] = {}
        for concept in concepts:
            root = uf.find(concept)
            clusters.setdefault(root, []).append(concept)

        # Pick canonical name: most frequent, then shortest
        counts = concept_counts or Counter()
        mapping: Dict[str, str] = {}
        merged_count = 0
        for members in clusters.values():
            canonical = max(members, key=lambda c: (counts.get(c, 0), -len(c)))
            for member in members:
                mapping[member] = canonical
            if len(members) > 1:
                merged_count += len(members) - 1
                logger.info("Merged cluster: %s → '%s'", members, canonical)

        logger.info(
            "Deduplication complete: %d concepts merged into %d canonical forms.",
            len(concepts),
            len(concepts) - merged_count,
        )
        return mapping

    @staticmethod
    def apply_to_triples(
        triples: List[Triple], mapping: Dict[str, str]
    ) -> List[Triple]:
        """Replace object_concept in each triple using the dedup mapping."""
        updated: List[Triple] = []
        for triple in triples:
            canonical = mapping.get(triple.object_concept, triple.object_concept)
            updated.append(
                Triple(
                    subject_isbn=triple.subject_isbn,
                    relation=triple.relation,
                    object_concept=canonical,
                )
            )
        return updated
