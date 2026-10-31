"""Restore a SQLite backup created before a migration. Stop the application first.

    python -m scripts.rollback_db --list
    python -m scripts.rollback_db --backup data/backups/app-20261030T120000Z-pre-v2.db
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import get_settings  # noqa: E402
from app.migrations import restore_backup  # noqa: E402

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backup", type=Path)
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()
    settings = get_settings()
    if args.list or not args.backup:
        for path in sorted(settings.backup_dir.glob("*.db")):
            print(path)
    else:
        url = settings.resolved_database_url
        if not url.startswith("sqlite:///"):
            raise SystemExit("Rollback helper supports SQLite only; restore PostgreSQL with pg_restore.")
        target = Path(url.replace("sqlite:///", ""))
        restore_backup(args.backup, target)
        print(f"restored {args.backup} -> {target}")
