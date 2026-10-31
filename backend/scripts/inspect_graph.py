"""Graph inspection utility: totals, nodes/relationships by type, orphans, Articles without
chunks, chunks without documents, missing required relationships, provenance gaps.

    python -m scripts.inspect_graph            # relational graph
    python -m scripts.inspect_graph --neo4j    # also Neo4j counts
Exit code 1 when the graph is not healthy.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import get_settings  # noqa: E402
from app.database import get_session_factory, init_db  # noqa: E402
from app.graph.inspect import inspect_graph  # noqa: E402

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--neo4j", action="store_true")
    args = parser.parse_args()
    init_db()
    report = inspect_graph(get_session_factory()())
    if args.neo4j:
        get_settings().enable_neo4j = True
        from app.graph.neo4j_store import neo4j_available, neo4j_counts
        report["neo4j"] = neo4j_counts() if neo4j_available(max_age=0) else {"status": "unreachable"}
    print(json.dumps(report, indent=1))
    raise SystemExit(0 if report["healthy"] else 1)
