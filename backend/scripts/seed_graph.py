"""Seed / rebuild the knowledge graph from the ingested Constitution and the curated legal
knowledge file (app/knowledge/data/legal_knowledge.json), verify statutes, and optionally
project it into Neo4j.

    python -m scripts.seed_graph            # relational graph (system of record)
    python -m scripts.seed_graph --neo4j    # ... and sync to Neo4j (NEO4J_* settings)
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import get_settings  # noqa: E402
from app.database import get_session_factory, init_db  # noqa: E402
from app.graph.builder import build_graph  # noqa: E402
from app.graph.store import persist_graph  # noqa: E402
from app.knowledge.statutes import verify_statutes  # noqa: E402

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--neo4j", action="store_true")
    args = parser.parse_args()
    init_db()
    db = get_session_factory()()
    print(json.dumps({"statute_sections": {k: v["status"] for k, v in verify_statutes().items()}}, indent=1))
    build = build_graph(db)
    if build.rejected:
        print(json.dumps({"rejected_relationships": build.rejected[:20]}, indent=1))
    print(json.dumps(persist_graph(db, build), indent=1))
    if args.neo4j:
        get_settings().enable_neo4j = True
        from app.graph.neo4j_store import neo4j_available, sync_to_neo4j
        if not neo4j_available(max_age=0):
            raise SystemExit("Neo4j is not reachable with the configured NEO4J_URI/NEO4J_PASSWORD.")
        print(json.dumps(sync_to_neo4j(db), indent=1))
