"""Cognitive Engine: orchestrates Knowledge Graph + SAN + vector search.

This is the single integration point that the Gradio dashboard calls.
Supports two modes:
  1. Two-stage funnel: separate vector + SAN galleries (legacy)
  2. Hybrid scoring: α-blended vector + SAN into a single ranked list
"""

import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from fathom.config import (
    HYBRID_ALPHA_DEFAULT,
    HYBRID_POOL_TOP_K,
    SAN_MIN_ACTIVATION,
    SAN_RESULT_TOP_K,
    SAN_SEED_COUNT,
    VECTOR_FINAL_TOP_K,
    VECTOR_INITIAL_TOP_K,
)
from fathom.graph import BookKnowledgeGraph
from fathom.spreading import SpreadingActivationEngine

logger = logging.getLogger(__name__)


def _min_max_normalize(scores: Dict[str, float]) -> Dict[str, float]:
    """Normalize a dict of scores to [0, 1] via min-max scaling."""
    if not scores:
        return {}
    vals = np.array(list(scores.values()))
    lo, hi = vals.min(), vals.max()
    if hi - lo < 1e-9:
        return {k: 1.0 for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


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

    def hybrid_recommend(
        self,
        query: str,
        alpha: float = HYBRID_ALPHA_DEFAULT,
        category: str = "All",
        tone: str = "All",
        san_seed_count: int = SAN_SEED_COUNT,
        final_top_k: int = VECTOR_FINAL_TOP_K,
    ) -> Tuple[pd.DataFrame, List[dict]]:
        """Blended recommendation: α * vector_score + (1 - α) * san_score.

        Args:
            query: Natural language book description.
            alpha: Blend weight. 1.0 = pure vector (standard),
                   0.0 = pure SAN (wildcard / surprise me).
            category: Category filter or "All".
            tone: Emotional tone filter or "All".
            san_seed_count: Number of top vector results to seed the SAN.
            final_top_k: Number of results to return.

        Returns:
            (blended_df with 'vector_score', 'san_score', 'hybrid_score' columns,
             explanations list for SAN-discovered books)
        """
        # 1. Vector search with similarity scores (larger pool for blending)
        vector_scores = self._vector_search_with_scores(
            query, pool_size=HYBRID_POOL_TOP_K
        )

        if not vector_scores:
            logger.warning("No vector results for hybrid scoring.")
            return pd.DataFrame(), []

        # 2. Seed the SAN from top vector hits
        sorted_by_vector = sorted(
            vector_scores.items(), key=lambda x: x[1], reverse=True
        )
        seed_isbns = [isbn for isbn, _ in sorted_by_vector[:san_seed_count]]

        # 3. Run SAN and collect peak activations
        activation_results = self.san.activate_from_books(seed_isbns)
        activated_books = self.san.get_activated_books(
            activation_results,
            exclude_seeds=seed_isbns,
            top_k=HYBRID_POOL_TOP_K,
            min_activation=SAN_MIN_ACTIVATION,
        )
        san_scores: Dict[str, float] = {
            isbn: peak for isbn, peak, _ in activated_books
        }

        # 4. Normalize both score sets to [0, 1]
        norm_vector = _min_max_normalize(vector_scores)
        norm_san = _min_max_normalize(san_scores)

        # 5. Union of all candidate ISBNs
        all_isbns = set(norm_vector.keys()) | set(norm_san.keys())

        # 6. Blend scores
        blended: Dict[str, Tuple[float, float, float]] = {}
        for isbn in all_isbns:
            v = norm_vector.get(isbn, 0.0)
            s = norm_san.get(isbn, 0.0)
            hybrid = alpha * v + (1.0 - alpha) * s
            blended[isbn] = (v, s, hybrid)

        # 7. Build result DataFrame
        blended_isbns = sorted(
            blended.keys(), key=lambda x: blended[x][2], reverse=True
        )
        result_df = self.books_df[
            self.books_df["isbn13"].astype(str).isin(blended_isbns)
        ].copy()

        result_df["vector_score"] = (
            result_df["isbn13"].astype(str).map(lambda x: blended.get(x, (0, 0, 0))[0])
        )
        result_df["san_score"] = (
            result_df["isbn13"].astype(str).map(lambda x: blended.get(x, (0, 0, 0))[1])
        )
        result_df["hybrid_score"] = (
            result_df["isbn13"].astype(str).map(lambda x: blended.get(x, (0, 0, 0))[2])
        )

        # Apply category filter
        if category != "All" and not result_df.empty:
            result_df = result_df[result_df["simple_categories"] == category]

        # Sort by emotional tone as tiebreaker within similar hybrid scores
        tone_map = {
            "Happy": "joy",
            "Surprising": "surprise",
            "Angry": "anger",
            "Suspenseful": "fear",
            "Sad": "sadness",
        }
        sort_cols = ["hybrid_score"]
        ascending = [False]
        if tone in tone_map:
            sort_cols.append(tone_map[tone])
            ascending.append(False)
        result_df = result_df.sort_values(
            by=sort_cols, ascending=ascending
        ).head(final_top_k)

        # 8. Build explanations for books that came via the SAN
        explanations = []
        for _, row in result_df.iterrows():
            isbn = str(row["isbn13"])
            if isbn in san_scores:
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

        logger.info(
            "Hybrid recommend: alpha=%.2f, %d vector candidates, %d SAN candidates, "
            "%d blended results returned.",
            alpha,
            len(norm_vector),
            len(norm_san),
            len(result_df),
        )

        return result_df, explanations

    def _vector_search_with_scores(
        self,
        query: str,
        pool_size: int = HYBRID_POOL_TOP_K,
    ) -> Dict[str, float]:
        """Run vector search and return isbn → relevance score mapping.

        Uses Chroma's similarity_search_with_relevance_scores so scores
        are already cosine-similarity based (higher = more similar).
        """
        results = self.db_books.similarity_search_with_relevance_scores(
            query, k=pool_size
        )
        scores: Dict[str, float] = {}
        for doc, score in results:
            try:
                isbn = str(int(doc.page_content.strip('"').split()[0]))
                scores[isbn] = score
            except (ValueError, IndexError):
                continue
        return scores

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
        activated_isbns = [isbn for isbn, _, _ in activated_books]
        activation_scores = {isbn: score for isbn, score, _ in activated_books}

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
