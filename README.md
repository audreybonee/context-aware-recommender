# Fathom: A Neuro-Symbolic Book Recommendation Engine

> *Discover what you didn't know you wanted.*

A hybrid book recommendation system that combines vector-based semantic search with a neuro-symbolic knowledge graph and spreading activation to surface structurally connected, serendipitous recommendations. Developed as part of a graduate research experiment in the Master of Science in Engineering in Artificial Intelligence program at the University of Pennsylvania.

---

## Motivation

Traditional recommender systems rely on collaborative filtering or content-based similarity, which tend to reinforce narrow reading patterns. Fathom explores whether augmenting semantic vector search with a **knowledge graph** and **spreading activation network (SAN)** can produce recommendations that are thematically coherent yet structurally surprising -- connecting books through shared concepts, moods, tropes, and narrative forms rather than surface-level similarity alone.

### Stage 1: Semantic Vector Search
The user's natural-language query is embedded using OpenAI embeddings and matched against pre-embedded book descriptions stored in ChromaDB. Results are filtered by category and sorted by emotional tone (joy, surprise, anger, fear, sadness) derived from sentiment analysis.

### Stage 2: Knowledge Graph Discovery
The top 5 vector search results seed a **Spreading Activation Network** that propagates energy through a heterogeneous knowledge graph. Books that receive sufficient activation through shared concepts -- but were *not* in the original vector results -- are surfaced as structurally connected discoveries.

The system then traces the shortest path between seed and discovered books through the graph, extracting the intermediate concepts to generate human-readable explanations (e.g., *"Connected through themes of grief and redemption"*).

## Knowledge Graph

The knowledge graph is a heterogeneous NetworkX graph with five node types and seven relation types:

| Node Type | Example |
|-----------|---------|
| `book` | `book:9780143127550` |
| `concept` | `concept:grief`, `concept:unreliable-narrator` |
| `author` | `author:Toni Morrison` |
| `location` | `location:new-york` |
| `category` | `category:Fiction` |

| Relation | Connects |
|----------|----------|
| `HAS_THEME` | book → concept |
| `HAS_MOOD` | book → concept |
| `HAS_TROPE` | book → concept |
| `HAS_FORM` | book → concept |
| `WRITTEN_BY` | book → author |
| `SET_IN` | book → location |
| `HAS_CATEGORY` | book → category |

### Seed Ontology

To prevent entity explosion and concept drift, LLM extraction is constrained to a curated **seed ontology** of ~90 concepts organized into 8 semantic clusters:

- **Psychological** -- identity, grief, trauma, obsession, resilience, self-discovery, ...
- **Interpersonal** -- forbidden-love, betrayal, friendship, family-dynamics, forgiveness, ...
- **Societal** -- class-struggle, racism, colonialism, war, justice, immigration, ...
- **Philosophical** -- free-will, mortality, existentialism, faith, redemption, absurdism, ...
- **Narrative Tropes** -- coming-of-age, unreliable-narrator, dystopia, time-travel, quest, ...
- **Settings/Atmosphere** -- small-town, gothic, post-apocalyptic, academic, wilderness, ...
- **Literary Moods** -- satirical, elegiac, whimsical, noir, dreamlike, epic, ...
- **Audience/Form** -- epistolary, allegorical, metafiction, magical-realism, bildungsroman, ...

Book descriptions are processed by GPT-4o-mini (temperature=0.0) which extracts structured triples validated by Pydantic against this ontology before insertion into the graph.

## Spreading Activation

The SAN is implemented via [SpreadPy](https://github.com/cog-isa/spread-py) with the following hyperparameters:

| Parameter | Value | Description |
|-----------|-------|-------------|
| Retention | 0.6 | Fraction of energy a node retains per iteration |
| Decay | 0.05 | Energy lost per iteration |
| Suppress | 0.01 | Minimum activation threshold to keep propagating |
| Iterations | 10 | Number of spreading iterations |
| Initial Energy | 100.0 | Energy injected into each seed node |
| Min Activation | 0.1 | Minimum activation for a book to appear in results |

## Project Structure

```
context-aware-recommender/
├── gradio-dashboard.py          # Main entry point (Gradio web UI)
├── fathom/                      # Core recommendation engine
│   ├── config.py                # Paths and hyperparameters
│   ├── engine.py                # CognitiveEngine (orchestration layer)
│   ├── graph.py                 # BookKnowledgeGraph (NetworkX)
│   ├── spreading.py             # SpreadingActivationEngine (SpreadPy)
│   ├── extraction.py            # LLM-based triple extraction
│   ├── deduplication.py         # Concept deduplication
│   ├── ontology.py              # Seed ontology (~90 concepts)
│   ├── schemas.py               # Pydantic models (Triple, BookExtraction)
│   ├── dashboard.py             # HTML formatting for Gradio UI
│   └── metadata.py              # Metadata utilities
├── notebooks/
│   ├── build-knowledge-graph.ipynb  # KG construction pipeline
│   ├── vector-search.ipynb
│   └── text-classification.ipynb
├── ida/                         # Initial data analysis
│   ├── data-exploration.ipynb
│   └── sentiment-analysis.ipynb
├── data/                        # Data files (not tracked in git)
├── tests/
├── requirements.txt
└── README.md
```

# Academic Context

This project was developed as part of a graduate research experiment in the **MSE in Artificial Intelligence**. It investigates the use of neuro-symbolic methods -- specifically the combination of neural embeddings with symbolic knowledge graphs and spreading activation -- for context-aware recommendation.