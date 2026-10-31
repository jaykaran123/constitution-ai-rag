"""
Indian Constitution AI - Database Configuration
"""
import os

from sqlalchemy import create_engine, event
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from app.config import get_settings


class Base(DeclarativeBase):
    pass


def _build_engine(url: str):
    settings = get_settings()
    if url.startswith("sqlite"):
        if ":memory:" not in url:
            db_path = url.replace("sqlite:///", "")
            os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        engine = create_engine(url, connect_args={"check_same_thread": False}, echo=False)

        @event.listens_for(engine, "connect")
        def set_sqlite_pragma(dbapi_connection, connection_record):
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()
    else:
        engine = create_engine(url, pool_size=10, max_overflow=20, pool_pre_ping=True, echo=False)
    del settings
    return engine


_engine = None
_SessionLocal = None


def get_engine():
    """Return the process-wide engine (previously a new engine was created per call)."""
    global _engine, _SessionLocal
    if _engine is None:
        _engine = _build_engine(get_settings().resolved_database_url)
        _SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=_engine)
    return _engine


def get_session_factory():
    get_engine()
    return _SessionLocal


def get_db():
    """Dependency for FastAPI routes."""
    db = get_session_factory()()
    try:
        yield db
    finally:
        db.close()


def reset_engine():
    global _engine, _SessionLocal
    if _engine is not None:
        _engine.dispose()
    _engine = None
    _SessionLocal = None


def init_db():
    """Create tables and apply pending migrations."""
    from app.migrations import run_migrations
    engine = get_engine()
    run_migrations(engine)
    return engine
