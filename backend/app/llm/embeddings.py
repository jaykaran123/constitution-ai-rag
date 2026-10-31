"""Gemini embeddings.

``gemini-embedding-001`` returns one vector per input text and supports ``task_type`` and
``output_dimensionality``. ``gemini-embedding-2`` is multimodal and, when given a list of
texts in one call, returns a single aggregated embedding - so batching it like a text
model silently loses vectors. This provider verifies that the number and size of returned
vectors match the request and raises ``EmbeddingDimensionMismatch`` otherwise.
"""
from __future__ import annotations

import asyncio
import logging
import math
import random
from typing import Any

from app.llm.base import EmbeddingProvider
from app.llm.config import LLMConfig
from app.llm.exceptions import (EmbeddingDimensionMismatch, EmbeddingError, LLMConfigurationError, LLMError,
                                LLMQuotaExhaustedError)
from app.llm.gemini_provider import map_exception
from app.observability import log_event

logger = logging.getLogger("app.llm.embeddings")
_sleep = asyncio.sleep


def l2_normalise(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vector)) or 1.0
    return [v / norm for v in vector]


class GeminiEmbeddingProvider(EmbeddingProvider):
    name = "gemini"

    def __init__(self, config: LLMConfig, client: Any | None = None, batch_size: int = 64,
                 texts_per_minute: int = 0):
        self.config = config
        self._client = client
        self.batch_size = batch_size
        self.texts_per_minute = texts_per_minute
        self._window: list[tuple[float, int]] = []
        self._exhausted_until = 0.0
        self._per_text = config.embedding_model.startswith("gemini-embedding-2")

    async def _pace(self, count: int) -> None:
        """Client-side pacing for per-minute quotas (EMBEDDING_TEXTS_PER_MINUTE; 0 disables)."""
        if not self.texts_per_minute:
            return
        import time
        while True:
            now = time.monotonic()
            self._window = [(t, n) for t, n in self._window if now - t < 60]
            used = sum(n for _, n in self._window)
            if used + count <= self.texts_per_minute or not self._window:
                self._window.append((now, count))
                return
            await asyncio.sleep(max(0.5, 60 - (now - self._window[0][0])))

    @property
    def model(self) -> str:
        return self.config.embedding_model

    @property
    def dimensions(self) -> int:
        return self.config.embedding_dimensions

    def _get_client(self):
        if self._client is None:
            if not self.config.api_key:
                raise LLMConfigurationError("GEMINI_API_KEY is not configured")
            from google import genai
            from google.genai import types
            self._client = genai.Client(api_key=self.config.api_key, http_options=types.HttpOptions(
                timeout=int(self.config.timeout * 1000), retry_options=types.HttpRetryOptions(attempts=1)))
        return self._client

    async def _embed_batch(self, texts: list[str], task_type: str, patient: bool = True) -> list[list[float]]:
        from google.genai import types
        client = self._get_client()
        config_kwargs: dict[str, Any] = {"output_dimensionality": self.dimensions}
        if not self._per_text:
            config_kwargs["task_type"] = task_type
        last: LLMError | None = None
        import time
        if time.time() < self._exhausted_until:
            raise LLMQuotaExhaustedError("Embedding quota exhausted (cooling down)", model=self.model)
        if patient:
            await self._pace(len(texts))
        for attempt in range(self.config.max_retries + 5 if patient else 1):
            try:
                response = await asyncio.wait_for(client.aio.models.embed_content(
                    model=self.model, contents=texts, config=types.EmbedContentConfig(**config_kwargs)),
                    timeout=self.config.timeout)
                vectors = [list(e.values or []) for e in (response.embeddings or [])]
                if len(vectors) != len(texts):
                    raise EmbeddingDimensionMismatch(
                        f"Requested {len(texts)} embeddings from {self.model}, received {len(vectors)}")
                bad = {len(v) for v in vectors if len(v) != self.dimensions}
                if bad:
                    raise EmbeddingDimensionMismatch(
                        f"Expected {self.dimensions}-dimensional vectors from {self.model}, received {sorted(bad)}")
                # Truncated (non-3072) Gemini embeddings must be re-normalised for cosine similarity.
                return [l2_normalise(v) for v in vectors]
            except EmbeddingError:
                raise
            except Exception as exc:  # noqa: BLE001
                error = map_exception(exc, self.model)
                last = error
                log_event(logger, "embedding.error", logging.WARNING, model=self.model, attempt=attempt + 1,
                          error=error.kind, status=error.status_code)
                if error.kind == "quota_exhausted":
                    self._exhausted_until = time.time() + min(error.retry_after or 3600, 24 * 3600)
                if not error.retryable:
                    raise error from exc
                if error.kind == "rate_limited":
                    # Embedding quotas are per minute; wait for the window to reset.
                    delay = error.retry_after if error.retry_after and error.retry_after < 90 else 62.0
                else:
                    delay = min(20.0, 1.5 * 2 ** attempt)
                await _sleep(delay + random.uniform(0, 1.0))
        assert last is not None
        raise last

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        if self._per_text:
            for text in texts:
                out.extend(await self._embed_batch([text], "RETRIEVAL_DOCUMENT"))
            return out
        for start in range(0, len(texts), self.batch_size):
            out.extend(await self._embed_batch(texts[start:start + self.batch_size], "RETRIEVAL_DOCUMENT"))
        return out

    async def embed_query(self, text: str) -> list[float]:
        # Query time: fail fast so a busy quota only disables vector search for this question.
        return (await self._embed_batch([text], "RETRIEVAL_QUERY", patient=False))[0]
