"""Single place where LLM and embedding providers are created.

The rest of the application asks the factory for a provider; nothing else imports the
Gemini SDK. Tests can install fakes with ``set_llm_provider`` / ``set_embedding_provider``.
"""
from __future__ import annotations

from app.config import get_settings
from app.llm.base import EmbeddingProvider, LLMProvider
from app.llm.config import LLMConfig
from app.llm.exceptions import LLMConfigurationError

_llm: LLMProvider | None = None
_embedder: EmbeddingProvider | None = None
_embedder_disabled = False


class ProviderFactory:
    @staticmethod
    def create_llm(config: LLMConfig) -> LLMProvider:
        if config.provider == "gemini":
            from app.llm.gemini_provider import GeminiProvider
            return GeminiProvider(config)
        raise LLMConfigurationError(f"Unsupported LLM_PROVIDER {config.provider!r}")

    @staticmethod
    def create_embeddings(config: LLMConfig) -> EmbeddingProvider:
        if config.provider == "gemini":
            from app.llm.embeddings import GeminiEmbeddingProvider
            s = get_settings()
            return GeminiEmbeddingProvider(config, batch_size=s.embedding_batch_size,
                                           texts_per_minute=s.embedding_texts_per_minute)
        raise LLMConfigurationError(f"Unsupported embedding provider {config.provider!r}")


def get_llm_provider() -> LLMProvider | None:
    """Return the configured provider, or ``None`` when no API key is configured."""
    global _llm
    if _llm is None:
        config = LLMConfig.from_settings()
        if not config.api_key:
            return None
        _llm = ProviderFactory.create_llm(config)
    return _llm


def get_embedding_provider() -> EmbeddingProvider | None:
    global _embedder
    if _embedder_disabled:
        return None
    settings = get_settings()
    if _embedder is None:
        if not settings.enable_embeddings:
            return None
        config = LLMConfig.from_settings(settings)
        if not config.api_key:
            return None
        _embedder = ProviderFactory.create_embeddings(config)
    return _embedder


def set_llm_provider(provider: LLMProvider | None) -> None:
    global _llm
    _llm = provider


def set_embedding_provider(provider: EmbeddingProvider | None, disabled: bool = False) -> None:
    global _embedder, _embedder_disabled
    _embedder = provider
    _embedder_disabled = disabled


def reset_providers() -> None:
    global _llm, _embedder, _embedder_disabled
    _llm = None
    _embedder = None
    _embedder_disabled = False
