import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from tqdm import tqdm

from fathom.config import (
    EXTRACTION_BATCH_SIZE,
    EXTRACTION_CACHE_PATH,
    EXTRACTION_MODEL,
    EXTRACTION_TEMPERATURE,
)
from fathom.ontology import (
    FORM_LIST_PROMPT,
    MOOD_LIST_PROMPT,
    THEME_LIST_PROMPT,
    TROPE_LIST_PROMPT,
)
from fathom.schemas import BookExtraction

logger = logging.getLogger(__name__)

EXTRACTION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a literary analyst. Given a book's title and description, \
extract structured metadata. You MUST only select from the provided lists.

THEMES: Select 1-5 themes from ONLY this list:
{theme_list}

MOODS: Select 0-2 literary moods from ONLY this list:
{mood_list}

TROPES: Select 0-2 narrative tropes from ONLY this list:
{trope_list}

FORMS: Select 0-2 literary forms from ONLY this list:
{form_list}

SETTINGS: Name 0-2 geographic or temporal settings as short strings \
(e.g., "victorian-england", "1960s-new-york"). These are free-form but \
should be specific and hyphenated.

If nothing fits a category, return an empty list for that field.
Do NOT invent concepts outside the provided lists for themes, moods, tropes, or forms.""",
        ),
        (
            "human",
            "Title: {title}\nDescription: {description}",
        ),
    ]
)


class BookTripleExtractor:
    """Extracts structured triples from book descriptions using an LLM."""

    def __init__(
        self,
        model_name: str = EXTRACTION_MODEL,
        temperature: float = EXTRACTION_TEMPERATURE,
        cache_path: Path = EXTRACTION_CACHE_PATH,
    ):
        self.llm = ChatOpenAI(model=model_name, temperature=temperature)
        self.structured_llm = self.llm.with_structured_output(BookExtraction)
        self.cache_path = Path(cache_path)
        self.cache: Dict[str, dict] = self._load_cache()

    def _load_cache(self) -> Dict[str, dict]:
        if self.cache_path.exists():
            with open(self.cache_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _save_cache(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_path, "w", encoding="utf-8") as f:
            json.dump(self.cache, f, indent=2)

    def extract(
        self,
        isbn13: str,
        title: str,
        description: str,
        max_retries: int = 2,
    ) -> Optional[BookExtraction]:
        """Extract triples for a single book. Returns cached result if available."""
        if isbn13 in self.cache:
            return BookExtraction(**self.cache[isbn13])

        if not description or not description.strip():
            logger.warning("Empty description for ISBN %s (%s), skipping.", isbn13, title)
            return None

        for attempt in range(max_retries + 1):
            try:
                chain = EXTRACTION_PROMPT | self.structured_llm
                result: BookExtraction = chain.invoke(
                    {
                        "theme_list": THEME_LIST_PROMPT,
                        "mood_list": MOOD_LIST_PROMPT,
                        "trope_list": TROPE_LIST_PROMPT,
                        "form_list": FORM_LIST_PROMPT,
                        "title": title,
                        "description": description[:2000],  # truncate long descriptions
                    }
                )
                # Ensure the isbn13 matches the input
                result.isbn13 = isbn13
                self.cache[isbn13] = result.model_dump()
                return result
            except Exception as e:
                if attempt < max_retries:
                    logger.warning(
                        "Extraction attempt %d failed for ISBN %s: %s. Retrying...",
                        attempt + 1,
                        isbn13,
                        e,
                    )
                else:
                    logger.error(
                        "Extraction failed for ISBN %s (%s) after %d attempts: %s",
                        isbn13,
                        title,
                        max_retries + 1,
                        e,
                    )
                    return None

    def extract_batch(
        self,
        books_df: pd.DataFrame,
        batch_size: int = EXTRACTION_BATCH_SIZE,
        save_every: int = 50,
    ) -> List[BookExtraction]:
        """Process all books with progress bar and periodic cache saves.

        Args:
            books_df: DataFrame with columns isbn13, title, description.
            batch_size: Not used for API batching (sequential calls), but
                        controls the cache save frequency.
            save_every: Save cache to disk every N books.

        Returns:
            List of successfully extracted BookExtraction objects.
        """
        results: List[BookExtraction] = []
        total = len(books_df)

        for i, (_, row) in enumerate(
            tqdm(books_df.iterrows(), total=total, desc="Extracting triples")
        ):
            isbn = str(row["isbn13"])
            title = str(row.get("title", ""))
            description = str(row.get("description", ""))

            extraction = self.extract(isbn, title, description)
            if extraction:
                results.append(extraction)

            if (i + 1) % save_every == 0:
                self._save_cache()
                logger.info("Cache saved after %d/%d books.", i + 1, total)

        self._save_cache()
        logger.info(
            "Extraction complete: %d/%d books extracted successfully.",
            len(results),
            total,
        )
        return results
