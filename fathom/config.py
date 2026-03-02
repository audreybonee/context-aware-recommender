"""Fathom configuration: paths, hyperparameters, and constants."""

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
KNOWLEDGE_GRAPH_PATH = DATA_DIR / "knowledge_graph.graphml"
EXTRACTION_CACHE_PATH = DATA_DIR / "extraction_cache.json"
BOOKS_CSV_PATH = PROJECT_ROOT / "books_with_emotions.csv"

# ── LLM Extraction ────────────────────────────────────────────────────
EXTRACTION_MODEL = "gpt-4o-mini"
EXTRACTION_TEMPERATURE = 0.0
EXTRACTION_BATCH_SIZE = 20

# ── Deduplication ──────────────────────────────────────────────────────
DEDUP_SIMILARITY_THRESHOLD = 0.92

# ── Spreading Activation (SpreadPy) ───────────────────────────────────
SAN_RETENTION = 0.6
SAN_DECAY = 0.05
SAN_SUPPRESS = 0.01
SAN_ITERATIONS = 10
SAN_INITIAL_ENERGY = 100.0

# ── Cognitive Engine ───────────────────────────────────────────────────
VECTOR_INITIAL_TOP_K = 50
VECTOR_FINAL_TOP_K = 16
SAN_SEED_COUNT = 5
SAN_RESULT_TOP_K = 16
SAN_MIN_ACTIVATION = 0.1
