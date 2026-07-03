"""Runtime configuration, loaded from environment (``.env`` supported).

All filesystem paths are anchored to the repository root so the app behaves
identically regardless of the current working directory.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent

# LLM / embedding models
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.1-8b-instant")
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", "Omartificial-Intelligence-Space/Arabic-Triplet-Matryoshka-V2"
)
FALLBACK_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Retrieval
SEARCH_ALPHA = float(os.getenv("SEARCH_ALPHA", "0.7"))
TOP_K = int(os.getenv("TOP_K", "5"))

# Preload the embedding model at startup so the first query is fast. Disable in
# tests / lightweight environments that never run semantic search.
WARMUP_ON_STARTUP = os.getenv("WARMUP_ON_STARTUP", "true").lower() in ("1", "true", "yes")

# Paths (anchored to repo root, CWD-independent)
CHROMA_PATH = str(BASE_DIR / "chroma_db")
DATA_PATH = BASE_DIR / "data"
STATIC_DIR = BASE_DIR / "static"
BM25_INDEX_PATH = BASE_DIR / "bm25_index.pkl"

# Observability (optional). Tracing is a no-op unless explicitly enabled and
# the Langfuse credentials are present.
TRACING_ENABLED = os.getenv("TRACING_ENABLED", "false").lower() in ("1", "true", "yes")
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

# Per-1M-token pricing for the answer model, used to estimate cost in the
# benchmark. Override to match your provider's rates.
LLM_INPUT_COST_PER_1M = float(os.getenv("LLM_INPUT_COST_PER_1M", "0.05"))
LLM_OUTPUT_COST_PER_1M = float(os.getenv("LLM_OUTPUT_COST_PER_1M", "0.08"))
