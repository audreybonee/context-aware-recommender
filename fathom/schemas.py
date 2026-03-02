from enum import Enum
from typing import List

from pydantic import BaseModel, field_validator

from fathom.ontology import ALL_CONCEPTS, MOOD_CONCEPTS, THEME_CONCEPTS, TROPE_CONCEPTS


class RelationType(str, Enum):
    HAS_THEME = "HAS_THEME"
    SET_IN = "SET_IN"
    WRITTEN_BY = "WRITTEN_BY"
    HAS_MOOD = "HAS_MOOD"
    HAS_TROPE = "HAS_TROPE"
    HAS_FORM = "HAS_FORM"
    HAS_CATEGORY = "HAS_CATEGORY"


class Triple(BaseModel):
    """A single (subject, relation, object) triple for the knowledge graph."""

    subject_isbn: str
    relation: RelationType
    object_concept: str

    @field_validator("object_concept")
    @classmethod
    def normalize_concept(cls, v: str) -> str:
        return v.strip().lower().replace(" ", "-")


class BookExtraction(BaseModel):
    """Structured extraction result from one book's description.

    The LLM fills these fields; Pydantic validates them against
    the seed ontology before they enter the pipeline.
    """

    isbn13: str
    themes: List[str] = []
    settings: List[str] = []  # free-form locations (deduplicated later)
    moods: List[str] = []
    tropes: List[str] = []
    forms: List[str] = []

    @field_validator("themes")
    @classmethod
    def validate_themes(cls, v: List[str]) -> List[str]:
        normalized = [t.strip().lower().replace(" ", "-") for t in v]
        invalid = [t for t in normalized if t not in THEME_CONCEPTS]
        if invalid:
            raise ValueError(
                f"Themes not in seed ontology: {invalid}. "
                f"Must select from the provided theme list."
            )
        return normalized[:5]  # cap at 5

    @field_validator("moods")
    @classmethod
    def validate_moods(cls, v: List[str]) -> List[str]:
        normalized = [m.strip().lower().replace(" ", "-") for m in v]
        invalid = [m for m in normalized if m not in MOOD_CONCEPTS]
        if invalid:
            raise ValueError(
                f"Moods not in seed ontology: {invalid}. "
                f"Must select from the provided mood list."
            )
        return normalized[:2]  # cap at 2

    @field_validator("tropes")
    @classmethod
    def validate_tropes(cls, v: List[str]) -> List[str]:
        normalized = [t.strip().lower().replace(" ", "-") for t in v]
        invalid = [t for t in normalized if t not in TROPE_CONCEPTS]
        if invalid:
            raise ValueError(
                f"Tropes not in seed ontology: {invalid}. "
                f"Must select from the provided trope list."
            )
        return normalized[:2]  # cap at 2

    @field_validator("settings")
    @classmethod
    def normalize_settings(cls, v: List[str]) -> List[str]:
        return [s.strip().lower().replace(" ", "-") for s in v][:2]

    @field_validator("forms")
    @classmethod
    def validate_forms(cls, v: List[str]) -> List[str]:
        from fathom.ontology import FORM_CONCEPTS

        normalized = [f.strip().lower().replace(" ", "-") for f in v]
        invalid = [f for f in normalized if f not in FORM_CONCEPTS]
        if invalid:
            raise ValueError(
                f"Forms not in seed ontology: {invalid}. "
                f"Must select from the provided form list."
            )
        return normalized[:2]

    def to_triples(self) -> List[Triple]:
        """Convert this extraction into a flat list of Triple objects."""
        triples: List[Triple] = []
        for theme in self.themes:
            triples.append(
                Triple(
                    subject_isbn=self.isbn13,
                    relation=RelationType.HAS_THEME,
                    object_concept=theme,
                )
            )
        for setting in self.settings:
            triples.append(
                Triple(
                    subject_isbn=self.isbn13,
                    relation=RelationType.SET_IN,
                    object_concept=setting,
                )
            )
        for mood in self.moods:
            triples.append(
                Triple(
                    subject_isbn=self.isbn13,
                    relation=RelationType.HAS_MOOD,
                    object_concept=mood,
                )
            )
        for trope in self.tropes:
            triples.append(
                Triple(
                    subject_isbn=self.isbn13,
                    relation=RelationType.HAS_TROPE,
                    object_concept=trope,
                )
            )
        for form in self.forms:
            triples.append(
                Triple(
                    subject_isbn=self.isbn13,
                    relation=RelationType.HAS_FORM,
                    object_concept=form,
                )
            )
        return triples
