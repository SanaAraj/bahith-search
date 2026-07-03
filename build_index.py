"""Build both search indices from the documents in ``data/``.

This is the single entry point the quickstart and the Docker image use so a
fresh checkout becomes queryable in one command:

    python build_index.py            # seed if empty, then build both indices
    python build_index.py --offline  # use the bundled fallback corpus only

The vector store (ChromaDB) is rebuilt from scratch each run; the BM25 index is
pickled to ``bm25_index.pkl``.
"""

from __future__ import annotations

import argparse
import logging

import bm25
import embeddings
from config import DATA_PATH
from ingest import process_documents

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("build_index")


def ensure_seeded(offline: bool) -> None:
    existing = list(DATA_PATH.glob("*.txt")) if DATA_PATH.exists() else []
    if existing:
        logger.info("Found %d documents in %s", len(existing), DATA_PATH)
        return

    import seed_data

    if offline:
        logger.info("Seeding from bundled offline corpus")
        seed_data.use_fallback_articles()
    else:
        seed_data.main()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Bahith search indices")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Seed from the bundled fallback corpus instead of Wikipedia",
    )
    args = parser.parse_args()

    ensure_seeded(args.offline)

    docs = process_documents()
    if not docs:
        raise SystemExit(f"No documents to index in {DATA_PATH}. Seeding failed.")

    logger.info("Building BM25 index (%d chunks)", len(docs))
    bm25.build_index(docs)
    bm25.save_index()

    logger.info("Building vector store")
    embeddings.clear_collection()
    embeddings.add_documents(docs)
    count = embeddings.get_document_count()
    logger.info("Done. %d chunks indexed in BM25 and vector store", count)


if __name__ == "__main__":
    main()
