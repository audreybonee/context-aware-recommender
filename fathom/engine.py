"""Cognitive Engine: orchestrates Knowledge Graph + SAN + vector search.

This is the single integration point that the Gradio dashboard calls.
It runs the existing vector search, then uses the top results as seeds
for spreading activation to discover structurally connected books.
"""

import logging
from typing import List, Optional, Tuple

import pandas as pd

from fathom.config import (
    SAN_MIN_ACTIVATION,
    SAN_RESULT_TOP_K,
    SAN_SEED_COUNT,
    VECTOR_FINAL_TOP_K,
    VECTOR_INITIAL_TOP_K,
)
from fathom.graph import BookKnowledgeGraph
from fathom.spreading import SpreadingActivationEngine

logger = logging.getLogger(__name__)


class CognitiveEngine:
    """Orchestrates vector search + Knowledge Graph spreading activation.

    The recommendation flow:
    1. Vector search (Chroma) → top 50 → filter → 16 "Semantic Matches"
    2. Top 5 vector results seed the SAN
    3. SAN spreads through KG → 16 "Discovered via Knowledge Graph"
    4. Explanation paths built for each SAN result
    """

    def __init__(
        self,
        books_df: pd.DataFrame,
        db_books,  # Chroma vector store instance
        kg: BookKnowledgeGraph,
    ):
        self.books_df = books_df
        self.db_books = db_books
        self.kg = kg
        self.san = SpreadingActivationEngine(kg.graph)

    def recommend(
        self,
        query: str,
        category: str = "All",
        tone: str = "All",
        initial_top_k: int = VECTOR_INITIAL_TOP_K,
        final_top_k: int = VECTOR_FINAL_TOP_K,
        san_seed_count: int = SAN_SEED_COUNT,
        san_top_k: int = SAN_RESULT_TOP_K,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, List[dict]]:
        """Full recommendation pipeline.

        Returns:
            (vector_results_df, san_results_df, explanations)
        """
        # Step 1: Vector search (reuses existing logic)
        vector_results = self._vector_search(
            query, category, tone, initial_top_k, final_top_k
        )

        # Step 2: Use top vector results as SAN seeds
        seed_isbns = [
            str(isbn) for isbn in vector_results["isbn13"].head(san_seed_count).tolist()
        ]

        if not seed_isbns:
            logger.warning("No vector results to seed SAN.")
            return vector_results, pd.DataFrame(), []

        # Step 3: Run spreading activation
        san_results, explanations = self._san_search(
            seed_isbns, san_top_k, category
        )

        return vector_results, san_results, explanations

    def _vector_search(
        self,
        query: str,
        category: str,
        tone: str,
        initial_top_k: int,
        final_top_k: int,
    ) -> pd.DataFrame:
        """Run semantic vector search (mirrors existing retrieve_semantic_recommendations)."""
        recs = self.db_books.similarity_search(query, k=initial_top_k)
        books_list = [
            int(rec.page_content.strip('"').split()[0]) for rec in recs
        ]
        book_recs = self.books_df[
            self.books_df["isbn13"].isin(books_list)
        ].head(initial_top_k)

        if category != "All":
            book_recs = book_recs[
                book_recs["simple_categories"] == category
            ].head(final_top_k)
        else:
            book_recs = book_recs.head(final_top_k)

        # Sort by emotional tone
        tone_map = {
            "Happy": "joy",
            "Surprising": "surprise",
            "Angry": "anger",
            "Suspenseful": "fear",
            "Sad": "sadness",
        }
        if tone in tone_map:
            book_recs = book_recs.sort_values(
                by=tone_map[tone], ascending=False
            )

        return book_recs

    def _san_search(
        self,
        seed_isbns: List[str],
        top_k: int,
        category: str = "All",
    ) -> Tuple[pd.DataFrame, List[dict]]:
        """Run SAN from seed books, return df of discovered books + explanations."""
        activation_results = self.san.activate_from_books(seed_isbns)

        activated_books = self.san.get_activated_books(
            activation_results,
            exclude_seeds=seed_isbns,
            top_k=top_k * 3,  # get extra to allow for category filtering
            min_activation=SAN_MIN_ACTIVATION,
        )

        if not activated_books:
            return pd.DataFrame(), []

        # Look up book data
        activated_isbns = [isbn for isbn, _ in activated_books]
        activation_scores = {isbn: score for isbn, score in activated_books}

        san_df = self.books_df[
            self.books_df["isbn13"].astype(str).isin(activated_isbns)
        ].copy()

        # Apply category filter
        if category != "All" and not san_df.empty:
            san_df = san_df[san_df["simple_categories"] == category]

        # Add activation score and sort by it
        san_df["activation_score"] = san_df["isbn13"].astype(str).map(activation_scores)
        san_df = san_df.sort_values("activation_score", ascending=False).head(top_k)

        # Build explanations
        explanations = []
        for _, row in san_df.iterrows():
            isbn = str(row["isbn13"])
            explanation = self.san.explain_activation(seed_isbns, isbn)
            if explanation:
                explanations.append(
                    {
                        "isbn": isbn,
                        "title": row.get("title", "Unknown"),
                        "seed_isbn": explanation["seed_isbn"],
                        "via_concepts": explanation["via_concepts"],
                        "path": explanation["path"],
                    }
                )

        return san_df, explanations
