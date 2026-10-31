"""Structured logging, request IDs and per-question debug traces.

Logs are JSON lines by default. Secrets are never logged: values for keys that look like
credentials are redacted, and user questions are stored only as a hash and length unless
``STORE_QUERY_TEXT=true`` is set for local debugging.
"""
from __future__ import annotations

import contextvars
import hashlib
import json
import logging
import re
import sys
import time
import uuid
from typing import Any

request_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar("request_id", default=None)

_SECRET_KEYS = re.compile(r"(api[_-]?key|password|secret|token|authorization|credential)", re.I)
_SECRET_VALUES = re.compile(r"AIza[0-9A-Za-z_\-]{20,}")


def new_request_id() -> str:
    return uuid.uuid4().hex[:16]


def redact(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: ("[redacted]" if _SECRET_KEYS.search(str(k)) else redact(v)) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact(v) for v in value]
    if isinstance(value, str):
        return _SECRET_VALUES.sub("[redacted]", value)
    return value


def fingerprint(text: str) -> dict:
    return {"sha256_12": hashlib.sha256(text.encode("utf-8")).hexdigest()[:12], "length": len(text)}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {"ts": round(record.created, 3), "level": record.levelname, "logger": record.name,
                   "msg": _SECRET_VALUES.sub("[redacted]", record.getMessage())}
        rid = getattr(record, "request_id", None) or request_id_var.get()
        if rid:
            payload["request_id"] = rid
        event = getattr(record, "event", None)
        if event:
            payload["event"] = event
        data = getattr(record, "data", None)
        if data:
            payload["data"] = redact(data)
        if record.exc_info:
            payload["exc_type"] = record.exc_info[0].__name__ if record.exc_info[0] else None
        return json.dumps(payload, default=str, ensure_ascii=False)


def configure_logging(level: str = "INFO", fmt: str = "json") -> None:
    root = logging.getLogger()
    if getattr(root, "_constitution_ai_configured", False):
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter() if fmt == "json" else logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s %(message)s"))
    root.handlers = [handler]
    root.setLevel(level.upper())
    # Never let HTTP client libraries log request URLs/headers (they can contain API keys).
    for noisy in ("httpx", "httpcore", "google_genai", "urllib3", "neo4j"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    root._constitution_ai_configured = True  # type: ignore[attr-defined]


def log_event(logger: logging.Logger, event: str, level: int = logging.INFO, **data: Any) -> None:
    logger.log(level, event, extra={"event": event, "data": data})


class Trace:
    """Collects what happened at each pipeline stage for one question.

    The trace answers "where did a wrong answer come from?": query understanding,
    embedding, vector/keyword/graph retrieval, source extraction, generation or
    citation validation. It is returned only to administrators.
    """

    def __init__(self, request_id: str | None = None, question: str | None = None, store_text: bool = False):
        self.request_id = request_id or request_id_var.get() or new_request_id()
        self.started = time.perf_counter()
        self.stages: dict[str, dict] = {}
        self.timings_ms: dict[str, int] = {}
        self.question = fingerprint(question or "")
        if store_text and question:
            self.question["text"] = question

    def stage(self, name: str, **data: Any) -> None:
        self.stages.setdefault(name, {}).update(redact(data))

    def timer(self, name: str):
        trace = self

        class _Timer:
            def __enter__(self_inner):
                self_inner.t = time.perf_counter()
                return self_inner

            def __exit__(self_inner, *exc):
                trace.timings_ms[name] = trace.timings_ms.get(name, 0) + int((time.perf_counter() - self_inner.t) * 1000)
                return False
        return _Timer()

    def as_dict(self) -> dict:
        return {"request_id": self.request_id, "question": self.question, "stages": self.stages,
                "timings_ms": {**self.timings_ms, "total": int((time.perf_counter() - self.started) * 1000)}}
