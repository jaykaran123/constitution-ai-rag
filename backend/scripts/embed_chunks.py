"""Generate Gemini embeddings for chunks that lack one for the configured model/dimension.

Usage (from backend/):
    python -m scripts.embed_chunks                 # embed everything pending
    python -m scripts.embed_chunks --limit 200     # embed a batch
    python -m scripts.embed_chunks --reuse-from path/to/other.db
        # copy embeddings for identical chunk text (same text_hash, model, dimension) from another
        # database first, so an existing index does not have to be re-embedded.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import get_settings  # noqa: E402
from app.database import get_session_factory, init_db  # noqa: E402
from app.knowledge.ingestion import embed_pending_chunks, index_report  # noqa: E402
from app.llm.provider_factory import get_embedding_provider  # noqa: E402
from app.models import DocumentChunk  # noqa: E402


def reuse(db, other: Path) -> int:
    settings = get_settings()
    src = sqlite3.connect(f"file:{other}?mode=ro", uri=True)
    rows = src.execute("SELECT text_hash, embedding FROM document_chunks WHERE embedding IS NOT NULL AND "
                       "embedding_model = ? AND embedding_dim = ?",
                       (settings.gemini_embedding_model, settings.gemini_embedding_dimensions)).fetchall()
    by_hash = {h: e for h, e in rows if h}
    copied = 0
    for chunk in db.query(DocumentChunk).filter(DocumentChunk.embedding.is_(None)).all():
        if chunk.text_hash in by_hash:
            chunk.embedding = by_hash[chunk.text_hash]
            chunk.embedding_model = settings.gemini_embedding_model
            chunk.embedding_dim = settings.gemini_embedding_dimensions
            chunk.has_embedding = True
            copied += 1
    db.commit()
    return copied


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--reuse-from", type=Path, default=None)
    parser.add_argument("--reuse-only", action="store_true", help="copy existing vectors, make no API calls")
    args = parser.parse_args()
    init_db()
    db = get_session_factory()()
    try:
        if args.reuse_from:
            print(json.dumps({"reused": reuse(db, args.reuse_from)}))
        if args.reuse_only:
            print(json.dumps(index_report(db), indent=1))
            return
        provider = get_embedding_provider()
        if provider is None:
            raise SystemExit("Embeddings are disabled or GEMINI_API_KEY is not set.")
        print(json.dumps(await embed_pending_chunks(db, provider, limit=args.limit)))
        print(json.dumps(index_report(db), indent=1))
    finally:
        db.close()


if __name__ == "__main__":
    asyncio.run(main())
