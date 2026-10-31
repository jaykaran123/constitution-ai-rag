"""Typed LLM errors.

Every provider failure is mapped to one of these classes so callers never mistake an
API error for an empty successful answer, and so retry decisions are made in one place.
"""
from __future__ import annotations


class LLMError(Exception):
    """Base class. ``retryable`` tells the retry loop whether another attempt can help."""

    retryable: bool = False
    kind: str = "llm_error"
    user_message: str = "The language model service returned an error."

    def __init__(self, message: str = "", *, status_code: int | None = None, model: str | None = None,
                 retry_after: float | None = None):
        super().__init__(message or self.user_message)
        self.status_code = status_code
        self.model = model
        self.retry_after = retry_after

    def to_dict(self) -> dict:
        return {"type": self.kind, "status_code": self.status_code, "model": self.model,
                "retryable": self.retryable, "message": self.user_message}


class LLMConfigurationError(LLMError):
    kind = "configuration"
    user_message = "The language model is not configured. Set GEMINI_API_KEY and GEMINI_MODEL on the server."


class LLMAuthenticationError(LLMError):
    kind = "authentication"
    user_message = "The Gemini API rejected the API key. Check GEMINI_API_KEY on the server."


class LLMPermissionError(LLMError):
    kind = "permission"
    user_message = "The Gemini API key is not permitted to use this model or API."


class LLMModelNotFoundError(LLMError):
    kind = "model_not_found"
    user_message = "The configured Gemini model is not available. Check GEMINI_MODEL."


class LLMInvalidRequestError(LLMError):
    kind = "invalid_request"
    user_message = "The request to the language model was invalid."


class LLMRateLimitError(LLMError):
    retryable = True
    kind = "rate_limited"
    user_message = "The Gemini API rate limit was reached. Please try again shortly."


class LLMQuotaExhaustedError(LLMError):
    kind = "quota_exhausted"
    user_message = "The Gemini API quota for this key is exhausted. Check the quota in Google AI Studio."


class LLMTimeoutError(LLMError):
    retryable = True
    kind = "timeout"
    user_message = "The language model did not respond in time."


class LLMNetworkError(LLMError):
    retryable = True
    kind = "network"
    user_message = "The server could not reach the Gemini API. Check outbound network access."


class LLMServerError(LLMError):
    retryable = True
    kind = "server_error"
    user_message = "The Gemini service returned a temporary error."


class LLMEmptyResponseError(LLMError):
    retryable = True
    kind = "empty_response"
    user_message = "The language model returned no text."


class LLMBlockedResponseError(LLMError):
    kind = "blocked"
    user_message = "The language model declined to answer this request."


class LLMResponseParseError(LLMError):
    retryable = True
    kind = "parse_error"
    user_message = "The language model returned output that could not be parsed."


class EmbeddingError(LLMError):
    kind = "embedding_error"
    user_message = "Embedding generation failed."


class EmbeddingDimensionMismatch(EmbeddingError):
    kind = "embedding_dimension_mismatch"
    user_message = "The embedding model returned vectors of an unexpected size."
