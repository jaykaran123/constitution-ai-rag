"""LLM configuration derived from application settings (single source of truth)."""
from __future__ import annotations

from dataclasses import dataclass, field

from app.config import Settings, get_settings


@dataclass(frozen=True)
class LLMConfig:
    provider: str
    api_key: str | None = field(repr=False)
    model: str
    fallback_models: tuple[str, ...]
    classifier_model: str | None
    embedding_model: str
    embedding_dimensions: int
    timeout: float
    max_retries: int
    temperature: float
    max_output_tokens: int
    thinking_level: str | None

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> "LLMConfig":
        s = settings or get_settings()
        return cls(
            provider=s.llm_provider, api_key=s.gemini_api_key, model=s.gemini_model,
            fallback_models=tuple(m for m in s.fallback_model_list if m != s.gemini_model),
            classifier_model=s.gemini_classifier_model, embedding_model=s.gemini_embedding_model,
            embedding_dimensions=s.gemini_embedding_dimensions, timeout=s.gemini_timeout,
            max_retries=max(0, s.gemini_max_retries), temperature=s.gemini_temperature,
            max_output_tokens=s.gemini_max_output_tokens,
            thinking_level=(s.gemini_thinking_level or "").strip() or None,
        )

    def public_dict(self) -> dict:
        """Safe to expose to administrators: never contains the API key."""
        return {"provider": self.provider, "model": self.model, "fallback_models": list(self.fallback_models),
                "embedding_model": self.embedding_model, "embedding_dimensions": self.embedding_dimensions,
                "timeout_s": self.timeout, "max_retries": self.max_retries,
                "api_key_configured": bool(self.api_key)}
