"""Versioned, non-destructive schema migrations.

* New tables are created; existing tables only gain columns (``ALTER TABLE ADD COLUMN``).
* Before any pending migration runs on a file-based SQLite database, the database file is
  copied to ``data/backups/app-<timestamp>-pre-v<N>.db``. ``scripts/rollback_db.py``
  restores such a backup. For PostgreSQL take a ``pg_dump`` first (see DEPLOYMENT_GUIDE.md).
* Derived data (Constitution chunks and graph) is rebuilt by ingestion, never by migrations.
  User data (conversations, messages, uploads) is never deleted.
"""
from __future__ import annotations

import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import inspect, text
from sqlalchemy.engine import Engine

from app.config import get_settings
from app.database import Base

logger = logging.getLogger("app.migrations")

V2_COLUMNS: dict[str, list[tuple[str, str]]] = {
    "legal_documents": [("parser_version", "VARCHAR(20)"), ("amendment_status", "VARCHAR(255)"),
                        ("last_verified_at", "TIMESTAMP"), ("metadata_json", "JSON")],
    "document_chunks": [("chunk_key", "VARCHAR(160)"), ("provision_id", "VARCHAR(120)"),
                        ("clause_number", "VARCHAR(10)"), ("subclause", "VARCHAR(5)"), ("chunk_type", "VARCHAR(30)"),
                        ("document_version", "VARCHAR(100)"), ("text_hash", "VARCHAR(64)"), ("page_end", "INTEGER"),
                        ("printed_page", "VARCHAR(10)"), ("status", "VARCHAR(20)"), ("embedding", "BLOB"),
                        ("embedding_dim", "INTEGER")],
    "knowledge_entities": [("key", "VARCHAR(160)"), ("labels", "JSON"), ("provenance", "JSON"),
                           ("build_id", "VARCHAR(40)")],
    "knowledge_relations": [("key", "VARCHAR(400)"), ("provenance_json", "JSON"),
                            ("extraction_method", "VARCHAR(30)"), ("confidence", "FLOAT"),
                            ("build_id", "VARCHAR(40)")],
}

V2_INDEXES = [
    "CREATE UNIQUE INDEX IF NOT EXISTS uq_chunk_key ON document_chunks (chunk_key)",
    "CREATE INDEX IF NOT EXISTS idx_chunk_provision ON document_chunks (provision_id)",
    "CREATE UNIQUE INDEX IF NOT EXISTS uq_entity_key ON knowledge_entities (key)",
    "CREATE UNIQUE INDEX IF NOT EXISTS uq_relation_key ON knowledge_relations (key)",
]


def _v1_baseline(engine: Engine) -> None:
    Base.metadata.create_all(bind=engine)


def _v2_structured_knowledge(engine: Engine) -> None:
    import app.models  # noqa: F401 - register tables
    Base.metadata.create_all(bind=engine)  # creates constitution_provisions, query_traces, schema_migrations
    inspector = inspect(engine)
    with engine.begin() as conn:
        for table, columns in V2_COLUMNS.items():
            existing = {c["name"] for c in inspector.get_columns(table)}
            for name, ddl_type in columns:
                if name not in existing:
                    if engine.dialect.name == "postgresql":
                        ddl_type = {"BLOB": "BYTEA", "JSON": "JSONB"}.get(ddl_type, ddl_type)
                    conn.execute(text(f'ALTER TABLE {table} ADD COLUMN "{name}" {ddl_type}'))
        for statement in V2_INDEXES:
            conn.execute(text(statement))


def _v3_user_accounts(engine: Engine) -> None:
    import app.models  # noqa: F401 - register the account table
    Base.metadata.create_all(bind=engine, tables=[Base.metadata.tables["user_accounts"],
                                                   Base.metadata.tables["account_sessions"]])


MIGRATIONS = [
    (1, "baseline schema", _v1_baseline),
    (2, "structured constitution provisions, chunk metadata, embeddings, graph identity and traces",
     _v2_structured_knowledge),
    (3, "email and password user accounts", _v3_user_accounts),
]


def _applied_versions(engine: Engine) -> set[int]:
    if "schema_migrations" not in inspect(engine).get_table_names():
        return set()
    with engine.connect() as conn:
        return {row[0] for row in conn.execute(text("SELECT version FROM schema_migrations"))}


def _backup_sqlite(engine: Engine, version: int) -> str | None:
    url = engine.url
    if url.get_backend_name() != "sqlite" or not url.database or url.database == ":memory:":
        return None
    source = Path(url.database)
    if not source.exists() or source.stat().st_size == 0:
        return None
    backup_dir = get_settings().backup_dir
    backup_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    target = backup_dir / f"{source.stem}-{stamp}-pre-v{version}.db"
    # Use SQLite's online backup so WAL contents are included.
    import sqlite3
    with sqlite3.connect(str(source)) as src, sqlite3.connect(str(target)) as dst:
        src.backup(dst)
    return str(target)


def pending_migrations(engine: Engine) -> list[int]:
    applied = _applied_versions(engine)
    return [version for version, _, _ in MIGRATIONS if version not in applied]


def run_migrations(engine: Engine) -> list[int]:
    import app.models  # noqa: F401
    pending = pending_migrations(engine)
    if not pending:
        return []
    had_tables = bool(inspect(engine).get_table_names())
    backup = _backup_sqlite(engine, pending[0]) if had_tables else None
    if backup:
        logger.info("database backup created before migration", extra={"event": "db.backup", "data": {"path": backup}})
    for version, name, func in MIGRATIONS:
        if version not in pending:
            continue
        func(engine)
        Base.metadata.tables["schema_migrations"].create(bind=engine, checkfirst=True)
        with engine.begin() as conn:
            conn.execute(text("INSERT INTO schema_migrations (version, name, applied_at, backup_path) "
                              "VALUES (:v, :n, :t, :b)"),
                         {"v": version, "n": name, "t": datetime.now(timezone.utc), "b": backup})
        logger.info("migration applied", extra={"event": "db.migrated", "data": {"version": version, "name": name}})
    return pending


def restore_backup(backup_path: str | Path, database_path: str | Path) -> None:
    """Rollback helper: replace the database file with a backup (app must be stopped)."""
    backup_path, database_path = Path(backup_path), Path(database_path)
    if not backup_path.exists():
        raise FileNotFoundError(backup_path)
    for suffix in ("-wal", "-shm"):
        sidecar = Path(str(database_path) + suffix)
        if sidecar.exists():
            sidecar.unlink()
    shutil.copy2(backup_path, database_path)
