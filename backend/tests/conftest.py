"""Shared fixtures.

A temporary SQLite database is built once per test session from the *real* Constitution
PDF (structured extraction, chunks and knowledge graph), so retrieval and reasoning tests
exercise the real data rather than toy fixtures. Gemini is replaced by a scriptable fake
unless a test is marked ``live``.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

_TMP = Path(tempfile.mkdtemp(prefix="constitution-ai-tests-"))
os.environ.update({
    "APP_ENV": "test", "DATABASE_URL": f"sqlite:///{(_TMP / 'test.db').as_posix()}",
    "DATA_DIR": str(Path(__file__).resolve().parents[1] / "data"), "UPLOAD_DIR": str(_TMP / "uploads"),
    "GEMINI_API_KEY": "", "ENABLE_EMBEDDINGS": "false", "ENABLE_NEO4J": "false", "ADMIN_API_TOKEN": "test-admin",
    "QUERY_CLASSIFIER": "rules", "LOG_LEVEL": "WARNING", "APP_SECRET_KEY": "test-secret-key-for-signing",
    "RATE_LIMIT_REQUESTS": "1000", "GEMINI_FALLBACK_MODELS": "fallback-model", "GEMINI_MODEL": "primary-model",
})

import asyncio  # noqa: E402
import re  # noqa: E402
from typing import AsyncIterator  # noqa: E402

import pytest  # noqa: E402

from app.config import get_settings, reset_settings  # noqa: E402

reset_settings()
settings = get_settings()
settings.gemini_api_key = None   # never use a real key from backend/.env in tests

from app.database import get_session_factory, init_db  # noqa: E402
from app.llm.base import GenerationRequest, LLMProvider, LLMResponse, StreamChunk, TokenUsage  # noqa: E402
from app.llm.exceptions import LLMError  # noqa: E402

CONSTITUTION_PDF = Path(__file__).resolve().parents[1] / "data" / "constitution" / "constitution.pdf"


@pytest.fixture(scope="session")
def knowledge_db():
    """Database with the real Constitution ingested and the graph built (once per session)."""
    if not CONSTITUTION_PDF.exists():
        pytest.skip("Constitution PDF not installed")
    init_db()
    from app.main import refresh_knowledge
    db = get_session_factory()()
    refresh_knowledge(db)
    yield db
    db.close()


@pytest.fixture
def db(knowledge_db):
    return knowledge_db


class FakeLLM(LLMProvider):
    """Scriptable stand-in for GeminiProvider.

    ``answers`` is a list of callables or strings used in order for ``final_answer`` and
    ``citation_repair`` requests; an ``LLMError`` instance in the list is raised instead.
    """

    name = "fake"

    def __init__(self, answers=None, classification=None):
        super().__init__()
        self.answers = list(answers or [])
        self.classification = classification
        self.requests: list[GenerationRequest] = []

    @property
    def model(self) -> str:
        return "fake-model"

    def _next(self, request: GenerationRequest) -> str:
        self.requests.append(request)
        if request.purpose == "scenario_classification":
            if isinstance(self.classification, LLMError):
                raise self.classification
            return self.classification or "{}"
        if not self.answers:
            return default_answer(request.prompt)
        item = self.answers.pop(0)
        if isinstance(item, LLMError):
            raise item
        return item(request.prompt) if callable(item) else item

    async def generate(self, request: GenerationRequest) -> LLMResponse:
        text = self._next(request)
        response = LLMResponse(text=text, model=self.model, latency_ms=5,
                               usage=TokenUsage(prompt_tokens=100, output_tokens=50, total_tokens=150))
        if request.response_schema:
            import json
            response.parsed = json.loads(text)
        self.usage.record_success(response)
        return response

    async def stream(self, request: GenerationRequest) -> AsyncIterator[StreamChunk]:
        text = self._next(request)
        for i in range(0, len(text), 40):
            yield StreamChunk(text=text[i:i + 40])
        yield StreamChunk(done=True, response=LLMResponse(text=text, model=self.model, latency_ms=5))

    async def health_check(self, timeout=None) -> dict:
        return {"provider": "fake", "status": "connected", "model": self.model, "latency_ms": 1}


def evidence_ids(prompt: str) -> dict[str, str]:
    """Map evidence ID -> label from the EVIDENCE block in a prompt."""
    return {m.group(1): m.group(2) for m in re.finditer(r"^\[([CSU]\d+)\] [^—\n]*— ([^|(\n]+)", prompt, re.M)}


def default_answer(prompt: str) -> str:
    """A well-formed, fully cited answer built only from the evidence in the prompt."""
    ids = evidence_ids(prompt)
    lines = ["### Short answer", "Based on the retrieved constitutional text, here is the position."]
    lines.append("### Relevant constitutional provisions")
    for eid, label in list(ids.items())[:4]:
        lines.append(f"- {label.strip()} is relevant here [{eid}].")
    return "\n".join(lines)


@pytest.fixture
def fake_llm():
    from app.llm.provider_factory import set_llm_provider
    llm = FakeLLM()
    set_llm_provider(llm)
    yield llm
    set_llm_provider(None)


def run(coro):
    return asyncio.run(coro)
