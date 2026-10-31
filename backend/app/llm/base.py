"""Provider-neutral interfaces for text generation and embeddings."""
from __future__ import annotations

import abc
import threading
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator


@dataclass
class TokenUsage:
    prompt_tokens: int = 0
    output_tokens: int = 0
    thinking_tokens: int = 0
    total_tokens: int = 0

    def as_dict(self) -> dict:
        return {"prompt_tokens": self.prompt_tokens, "output_tokens": self.output_tokens,
                "thinking_tokens": self.thinking_tokens, "total_tokens": self.total_tokens}


@dataclass
class LLMResponse:
    text: str
    model: str
    latency_ms: int
    usage: TokenUsage = field(default_factory=TokenUsage)
    finish_reason: str | None = None
    attempts: int = 1
    parsed: Any = None            # structured output when a JSON schema was requested

    def as_log(self) -> dict:
        return {"model": self.model, "latency_ms": self.latency_ms, "attempts": self.attempts,
                "finish_reason": self.finish_reason, "usage": self.usage.as_dict()}


@dataclass
class StreamChunk:
    text: str = ""
    done: bool = False
    response: LLMResponse | None = None   # populated on the final chunk


@dataclass
class GenerationRequest:
    prompt: str
    system_instruction: str | None = None
    temperature: float | None = None
    max_output_tokens: int | None = None
    response_schema: dict | None = None    # JSON schema -> structured JSON output
    purpose: str = "generation"            # used in logs and metrics
    model: str | None = None               # override the configured model


class UsageTracker:
    """In-process counters for observability (requests, failures, tokens, latency)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.reset()

    def reset(self) -> None:
        with getattr(self, "_lock", threading.Lock()):
            self.requests = 0
            self.successes = 0
            self.failures: dict[str, int] = {}
            self.tokens = TokenUsage()
            self.total_latency_ms = 0
            self.started_at = time.time()

    def record_success(self, response: LLMResponse) -> None:
        with self._lock:
            self.requests += 1
            self.successes += 1
            self.total_latency_ms += response.latency_ms
            self.tokens.prompt_tokens += response.usage.prompt_tokens
            self.tokens.output_tokens += response.usage.output_tokens
            self.tokens.thinking_tokens += response.usage.thinking_tokens
            self.tokens.total_tokens += response.usage.total_tokens

    def record_failure(self, kind: str) -> None:
        with self._lock:
            self.requests += 1
            self.failures[kind] = self.failures.get(kind, 0) + 1

    def snapshot(self) -> dict:
        with self._lock:
            return {"requests": self.requests, "successes": self.successes,
                    "success_rate": round(self.successes / self.requests, 3) if self.requests else None,
                    "failures": dict(self.failures), "tokens": self.tokens.as_dict(),
                    "avg_latency_ms": round(self.total_latency_ms / self.successes) if self.successes else None}


class LLMProvider(abc.ABC):
    name: str = "base"

    def __init__(self) -> None:
        self.usage = UsageTracker()

    @property
    @abc.abstractmethod
    def model(self) -> str: ...

    @abc.abstractmethod
    async def generate(self, request: GenerationRequest) -> LLMResponse:
        """Return a complete response or raise an ``LLMError``. Never returns empty text."""

    @abc.abstractmethod
    def stream(self, request: GenerationRequest) -> AsyncIterator[StreamChunk]:
        """Yield text deltas, then a final chunk with ``done=True`` and the full response."""

    @abc.abstractmethod
    async def health_check(self, timeout: float | None = None) -> dict:
        """Verify authentication, model availability and a minimal generation."""


class EmbeddingProvider(abc.ABC):
    name: str = "base"

    @property
    @abc.abstractmethod
    def model(self) -> str: ...

    @property
    @abc.abstractmethod
    def dimensions(self) -> int: ...

    @abc.abstractmethod
    async def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    @abc.abstractmethod
    async def embed_query(self, text: str) -> list[float]: ...
