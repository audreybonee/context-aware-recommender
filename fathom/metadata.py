"""Tier 4: Metadata Baseline — structured triples from existing CSV data.

Extracts WRITTEN_BY and HAS_CATEGORY triples from the books DataFrame
without LLM calls. Author, genre, and category data already exist in
the dataset and are more reliable than LLM extraction for factual metadata.
"""

import logging
from typing import List

import pandas as pd

from fathom.schemas import RelationType, Triple

logger = logging.getLogger(__name__)


class MetadataExtractor:
    """Extracts structured triples from existing book metadata columns."""

    @staticmethod
    def extract_author_triples(books_df: pd.DataFrame) -> List[Triple]:
        """Parse the authors column and produce WRITTEN_BY triples.

        The authors field may contain multiple authors separated by
        semicolons (e.g., "Author A;Author B").
        """
        triples: List[Triple] = []
        for _, row in books_df.iterrows():
            isbn = str(row["isbn13"])
            authors_raw = row.get("authors", "")
            if pd.isna(authors_raw) or not str(authors_raw).strip():
                continue
            for author in str(authors_raw).split(";"):
                author_clean = author.strip().lower().replace(" ", "-")
                if author_clean:
                    triples.append(
                        Triple(
                            subject_isbn=isbn,
                            relation=RelationType.WRITTEN_BY,
                            object_concept=author_clean,
                        )
                    )
        logger.info("Extracted %d WRITTEN_BY triples.", len(triples))
        return triples

    @staticmethod
    def extract_category_triples(books_df: pd.DataFrame) -> List[Triple]:
        """Produce HAS_CATEGORY triples from the simple_categories column."""
        triples: List[Triple] = []
        for _, row in books_df.iterrows():
            isbn = str(row["isbn13"])
            category = row.get("simple_categories", "")
            if pd.isna(category) or not str(category).strip():
                continue
            cat_clean = str(category).strip().lower().replace(" ", "-")
            if cat_clean:
                triples.append(
                    Triple(
                        subject_isbn=isbn,
                        relation=RelationType.HAS_CATEGORY,
                        object_concept=cat_clean,
                    )
                )
        logger.info("Extracted %d HAS_CATEGORY triples.", len(triples))
        return triples

    @staticmethod
    def extract_all(books_df: pd.DataFrame) -> List[Triple]:
        """Extract all metadata-based triples."""
        triples = MetadataExtractor.extract_author_triples(books_df)
        triples.extend(MetadataExtractor.extract_category_triples(books_df))
        return triples
