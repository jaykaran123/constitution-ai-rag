"""Curated knowledge loading and statutory text verification.

Statutory excerpts are marked ``verified`` only when the official India Code PDF for the
statute is present in ``data/statutes`` (as named by ``local_file``) and contains the
excerpt after normalisation. Otherwise they stay ``unverified`` and answers must say so.
"""
from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path

from app.config import get_settings
from app.knowledge.text_utils import contains_quote

KNOWLEDGE_PATH = Path(__file__).resolve().parent / "data" / "legal_knowledge.json"


@lru_cache(maxsize=1)
def load_knowledge() -> dict:
    return json.loads(KNOWLEDGE_PATH.read_text(encoding="utf-8"))


def knowledge_checksum() -> str:
    return hashlib.sha256(KNOWLEDGE_PATH.read_bytes()).hexdigest()[:16]


@lru_cache(maxsize=8)
def _pdf_text(path_str: str, mtime: float) -> str:
    import fitz
    with fitz.open(path_str) as pdf:
        return "\n".join(page.get_text("text") for page in pdf)


def statute_pdf_text(local_file: str) -> tuple[str | None, str | None]:
    """Return (text, sha256) of a downloaded official statute PDF, or (None, None)."""
    path = get_settings().statutes_dir / local_file
    if not path.exists():
        return None, None
    raw = path.read_bytes()
    if not raw.startswith(b"%PDF"):
        return None, None
    return _pdf_text(str(path), path.stat().st_mtime), hashlib.sha256(raw).hexdigest()


def verify_statutes() -> dict[str, dict]:
    """Return verification results keyed by section id."""
    results: dict[str, dict] = {}
    for statute in load_knowledge()["statutes"]:
        text, checksum = statute_pdf_text(statute.get("local_file") or "")
        for section in statute["sections"]:
            if text is None:
                status, reason = "unverified", "official PDF not available locally"
            elif contains_quote(text, section["text"]):
                status, reason = "verified", "excerpt found verbatim in the official PDF"
            else:
                status, reason = "unverified", "excerpt not found in the official PDF"
            results[section["id"]] = {"status": status, "reason": reason, "pdf_sha256": checksum,
                                      "statute": statute["id"]}
    return results
