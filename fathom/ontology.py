"""Tier 1: Top-Down Seed Ontology for constrained LLM extraction.

A curated vocabulary of ~90 concepts organized into semantic clusters.
The LLM is constrained to select ONLY from this list, which caps graph
size and prevents entity explosion / concept drift.
"""

from enum import Enum
from typing import Dict, FrozenSet, List


class ConceptCluster(str, Enum):
    PSYCHOLOGICAL = "psychological_states"
    INTERPERSONAL = "interpersonal_themes"
    SOCIETAL = "societal_themes"
    PHILOSOPHICAL = "philosophical_themes"
    NARRATIVE = "narrative_tropes"
    SETTINGS = "settings_atmosphere"
    MOODS = "literary_moods"
    SUBJECTS = "subject_domains"
    FORM = "audience_form"


SEED_ONTOLOGY: Dict[ConceptCluster, List[str]] = {
    ConceptCluster.PSYCHOLOGICAL: [
        "identity",
        "grief",
        "trauma",
        "obsession",
        "loneliness",
        "guilt",
        "madness",
        "desire",
        "jealousy",
        "shame",
        "resilience",
        "self-discovery",
    ],
    ConceptCluster.INTERPERSONAL: [
        "forbidden-love",
        "betrayal",
        "friendship",
        "family-dynamics",
        "mentor-student",
        "rivalry",
        "forgiveness",
        "sacrifice",
        "loyalty",
        "revenge",
    ],
    ConceptCluster.SOCIETAL: [
        "class-struggle",
        "racism",
        "colonialism",
        "war",
        "revolution",
        "justice",
        "corruption",
        "censorship",
        "immigration",
        "gender-roles",
        "religious-conflict",
        "surveillance",
    ],
    ConceptCluster.PHILOSOPHICAL: [
        "free-will",
        "mortality",
        "good-vs-evil",
        "existentialism",
        "faith",
        "truth-vs-illusion",
        "nature-vs-nurture",
        "power",
        "redemption",
        "absurdism",
    ],
    ConceptCluster.NARRATIVE: [
        "coming-of-age",
        "quest",
        "unreliable-narrator",
        "time-travel",
        "dystopia",
        "utopia",
        "chosen-one",
        "forbidden-knowledge",
        "double-life",
        "survival",
        "escape",
        "transformation",
    ],
    ConceptCluster.SETTINGS: [
        "small-town",
        "urban-decay",
        "wilderness",
        "gothic",
        "maritime",
        "wartime",
        "post-apocalyptic",
        "academic",
        "domestic",
        "underworld",
    ],
    ConceptCluster.MOODS: [
        "satirical",
        "elegiac",
        "whimsical",
        "noir",
        "pastoral",
        "claustrophobic",
        "dreamlike",
        "epic",
    ],
    ConceptCluster.SUBJECTS: [
        "science",
        "art",
        "music",
        "politics",
        "crime",
        "spirituality",
        "technology",
        "nature",
        "history",
        "mythology",
    ],
    ConceptCluster.FORM: [
        "autobiographical",
        "epistolary",
        "allegorical",
        "metafiction",
        "bildungsroman",
        "magical-realism",
    ],
}

# Flat set of all valid concepts (frozen for safety)
ALL_CONCEPTS: FrozenSet[str] = frozenset(
    concept for concepts in SEED_ONTOLOGY.values() for concept in concepts
)

# Cluster-specific subsets for prompt building
THEME_CONCEPTS: FrozenSet[str] = frozenset(
    concept
    for cluster in [
        ConceptCluster.PSYCHOLOGICAL,
        ConceptCluster.INTERPERSONAL,
        ConceptCluster.SOCIETAL,
        ConceptCluster.PHILOSOPHICAL,
        ConceptCluster.SUBJECTS,
    ]
    for concept in SEED_ONTOLOGY[cluster]
)

MOOD_CONCEPTS: FrozenSet[str] = frozenset(SEED_ONTOLOGY[ConceptCluster.MOODS])

TROPE_CONCEPTS: FrozenSet[str] = frozenset(SEED_ONTOLOGY[ConceptCluster.NARRATIVE])

SETTING_CONCEPTS: FrozenSet[str] = frozenset(SEED_ONTOLOGY[ConceptCluster.SETTINGS])

FORM_CONCEPTS: FrozenSet[str] = frozenset(SEED_ONTOLOGY[ConceptCluster.FORM])

# Lookup: concept -> cluster
CONCEPT_TO_CLUSTER: Dict[str, ConceptCluster] = {
    concept: cluster
    for cluster, concepts in SEED_ONTOLOGY.items()
    for concept in concepts
}

# Comma-separated string for LLM prompts
CONCEPT_LIST_PROMPT: str = ", ".join(sorted(ALL_CONCEPTS))
THEME_LIST_PROMPT: str = ", ".join(sorted(THEME_CONCEPTS))
MOOD_LIST_PROMPT: str = ", ".join(sorted(MOOD_CONCEPTS))
TROPE_LIST_PROMPT: str = ", ".join(sorted(TROPE_CONCEPTS))
FORM_LIST_PROMPT: str = ", ".join(sorted(FORM_CONCEPTS))
