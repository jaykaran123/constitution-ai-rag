"""Project the relational knowledge graph into Neo4j (idempotent MERGE; stale items removed).

    NEO4J_URI=bolt://localhost:7687 NEO4J_PASSWORD=... python -m scripts.sync_neo4j
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import get_settings  # noqa: E402
from app.database import get_session_factory, init_db  # noqa: E402
from app.graph.neo4j_store import neo4j_available, neo4j_counts, sync_to_neo4j  # noqa: E402

if __name__ == "__main__":
    settings = get_settings()
    settings.enable_neo4j = True
    if not settings.neo4j_password:
        raise SystemExit("Set NEO4J_PASSWORD (and NEO4J_URI) first.")
    if not neo4j_available(max_age=0):
        raise SystemExit(f"Neo4j is not reachable at {settings.neo4j_uri}.")
    init_db()
    print(json.dumps(sync_to_neo4j(get_session_factory()()), indent=1))
    print(json.dumps(neo4j_counts(), indent=1))
