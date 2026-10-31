"""Apply pending database migrations (a SQLite backup is taken automatically first).

    python -m scripts.migrate            # apply
    python -m scripts.migrate --status   # list pending migrations only
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.database import get_engine  # noqa: E402
from app.migrations import pending_migrations, run_migrations  # noqa: E402

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()
    engine = get_engine()
    print(json.dumps({"pending": pending_migrations(engine)} if args.status else {"applied": run_migrations(engine)}))
