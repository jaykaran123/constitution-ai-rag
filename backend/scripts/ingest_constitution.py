"""Constitutional ingestion utility: verify, parse, chunk, index and build the graph.

    python -m scripts.ingest_constitution                 # idempotent (no-op when unchanged)
    python -m scripts.ingest_constitution --force         # re-parse even if unchanged
    python -m scripts.ingest_constitution --pdf path.pdf  # use another official copy
    python -m scripts.ingest_constitution --embed         # also generate embeddings
"""
import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.database import get_session_factory, init_db  # noqa: E402
from app.graph.builder import build_graph  # noqa: E402
from app.graph.store import persist_graph  # noqa: E402
from app.knowledge.ingestion import embed_pending_chunks, index_report, ingest_constitution  # noqa: E402
from app.llm.provider_factory import get_embedding_provider  # noqa: E402

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", type=Path)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--embed", action="store_true")
    args = parser.parse_args()
    init_db()
    db = get_session_factory()()
    print(json.dumps(ingest_constitution(db, path=args.pdf, force=args.force), indent=1))
    print(json.dumps(persist_graph(db, build_graph(db)), indent=1))
    if args.embed:
        provider = get_embedding_provider()
        if provider is None:
            print("embeddings skipped: GEMINI_API_KEY not set or ENABLE_EMBEDDINGS=false")
        else:
            print(json.dumps(asyncio.run(embed_pending_chunks(db, provider))))
    print(json.dumps(index_report(db), indent=1))
