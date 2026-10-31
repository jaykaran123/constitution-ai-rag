"""Google Gemini provider built on the official ``google-genai`` SDK.

Design notes
* One ``genai.Client`` per provider instance; the SDK's own retry is disabled so retry,
  backoff and model fallback are decided here, consistently, for every call.
* Each attempt has a hard timeout (``GEMINI_TIMEOUT``). A timed-out model is not retried;
  the next fallback model is tried instead, because a slow model rarely recovers quickly.
* Authentication, permission and invalid-request errors are never retried.
* Empty or blocked responses raise errors; they are never returned as an empty answer.
"""
from __future__ import annotations

import asyncio
import json
import logging
import random
import re
import time
from typing import Any, AsyncIterator

from app.llm.base import GenerationRequest, LLMProvider, LLMResponse, StreamChunk, TokenUsage
from app.llm.config import LLMConfig
from app.llm.exceptions import (
    LLMAuthenticationError, LLMBlockedResponseError, LLMConfigurationError, LLMEmptyResponseError, LLMError,
    LLMInvalidRequestError, LLMModelNotFoundError, LLMNetworkError, LLMPermissionError, LLMQuotaExhaustedError,
    LLMRateLimitError, LLMResponseParseError, LLMServerError, LLMTimeoutError,
)
from app.observability import log_event

logger = logging.getLogger("app.llm.gemini")
_sleep = asyncio.sleep          # backoff sleep (patched in tests)

_BLOCK_REASONS = {"SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII", "RECITATION", "IMAGE_SAFETY"}


def map_exception(exc: BaseException, model: str | None = None) -> LLMError:
    """Translate SDK/HTTP exceptions into typed ``LLMError`` instances."""
    if isinstance(exc, LLMError):
        return exc
    if isinstance(exc, asyncio.TimeoutError):
        return LLMTimeoutError("Gemini request timed out", model=model)
    try:
        import httpx
        if isinstance(exc, httpx.TimeoutException):
            return LLMTimeoutError("Gemini HTTP timeout", model=model)
        if isinstance(exc, httpx.TransportError):
            return LLMNetworkError(f"Network error: {type(exc).__name__}", model=model)
    except ImportError:  # pragma: no cover
        pass
    try:
        from google.genai import errors as genai_errors
    except ImportError:  # pragma: no cover
        genai_errors = None
    if genai_errors is not None and isinstance(exc, genai_errors.APIError):
        code = getattr(exc, "code", None)
        status = str(getattr(exc, "status", "") or "")
        message = str(getattr(exc, "message", "") or exc)
        lowered = message.lower()
        if code == 400 and ("api key" in lowered or "api_key" in lowered):
            return LLMAuthenticationError("Invalid API key", status_code=code, model=model)
        if code == 401:
            return LLMAuthenticationError("Unauthenticated", status_code=code, model=model)
        if code == 403:
            if "api key" in lowered or "api_key" in lowered:
                return LLMAuthenticationError("API key rejected", status_code=code, model=model)
            return LLMPermissionError("Permission denied", status_code=code, model=model)
        if code == 404:
            return LLMModelNotFoundError(f"Model not found: {model}", status_code=code, model=model)
        if code == 408:
            return LLMTimeoutError("Request timeout", status_code=code, model=model)
        if code == 429 or status == "RESOURCE_EXHAUSTED":
            retry_after = _retry_delay(message + " " + str(exc))
            # Daily quotas ("... per day limit exceeded ... retry in 12h54m44s") and models with no quota
            # ("limit: 0") will not recover within a request; per-minute limits will.
            if re.search(r"per ?day|PerDay|limit: 0", message, re.I) or (retry_after or 0) > 600:
                return LLMQuotaExhaustedError("Quota exhausted", status_code=code, model=model, retry_after=retry_after)
            return LLMRateLimitError("Rate limited", status_code=code, model=model, retry_after=retry_after)
        if code is not None and 500 <= int(code) < 600:
            return LLMServerError(f"Gemini server error {code}", status_code=code, model=model)
        return LLMInvalidRequestError(f"Gemini rejected the request ({code})", status_code=code, model=model)
    return LLMError(f"Unexpected Gemini client error: {type(exc).__name__}", model=model)


def _retry_delay(text: str) -> float | None:
    """Parse "retry in 7s", "retry in 12h54m44.2s" or a RetryInfo "retryDelay: '46484s'"."""
    match = re.search(r"retry in (?:(\d+)h)?(?:(\d+)m)?([\d.]+)s", text, re.I)
    if match:
        hours, minutes, seconds = match.groups()
        return int(hours or 0) * 3600 + int(minutes or 0) * 60 + float(seconds)
    match = re.search(r"retryDelay['\"]?:\s*['\"]?([\d.]+)s", text)
    return float(match.group(1)) if match else None


def _usage(response: Any) -> TokenUsage:
    meta = getattr(response, "usage_metadata", None)
    if not meta:
        return TokenUsage()
    prompt = getattr(meta, "prompt_token_count", 0) or 0
    output = getattr(meta, "candidates_token_count", 0) or 0
    thoughts = getattr(meta, "thoughts_token_count", 0) or 0
    total = getattr(meta, "total_token_count", 0) or (prompt + output + thoughts)
    return TokenUsage(prompt_tokens=prompt, output_tokens=output, thinking_tokens=thoughts, total_tokens=total)


def _finish_reason(response: Any) -> str | None:
    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        return None
    reason = getattr(candidates[0], "finish_reason", None)
    return getattr(reason, "name", None) or (str(reason) if reason is not None else None)


def _text(response: Any) -> str:
    try:
        return response.text or ""
    except Exception:  # SDK raises when the candidate has no text parts
        return ""


def _check_blocked(response: Any, model: str) -> None:
    feedback = getattr(response, "prompt_feedback", None)
    block = getattr(feedback, "block_reason", None) if feedback else None
    if block:
        raise LLMBlockedResponseError(f"Prompt blocked: {getattr(block, 'name', block)}", model=model)
    reason = _finish_reason(response)
    if reason in _BLOCK_REASONS and not _text(response):
        raise LLMBlockedResponseError(f"Response blocked: {reason}", model=model)


class GeminiProvider(LLMProvider):
    name = "gemini"

    def __init__(self, config: LLMConfig, client: Any | None = None):
        super().__init__()
        self.config = config
        self._client = client
        self._no_thinking_models: set[str] = set()
        self._cooldown: dict[str, float] = {}

    # ------------------------------------------------------------------ client
    @property
    def model(self) -> str:
        return self.config.model

    def _get_client(self):
        if self._client is None:
            if not self.config.api_key:
                raise LLMConfigurationError("GEMINI_API_KEY is not configured")
            from google import genai
            from google.genai import types
            self._client = genai.Client(
                api_key=self.config.api_key,
                http_options=types.HttpOptions(
                    timeout=int(self.config.timeout * 1000),
                    retry_options=types.HttpRetryOptions(attempts=1),
                ),
            )
        return self._client

    def _models_to_try(self, request: GenerationRequest) -> list[str]:
        primary = request.model or self.config.model
        models = list(dict.fromkeys([primary, *self.config.fallback_models]))
        # Skip models whose (per-model) quota is known to be exhausted, unless every model is.
        now = time.time()
        available = [m for m in models if self._cooldown.get(m, 0) <= now]
        return available or models

    def _mark_exhausted(self, model: str, error: LLMError) -> None:
        self._cooldown[model] = time.time() + min(error.retry_after or 3600, 24 * 3600)
        log_event(logger, "llm.model.cooldown", logging.WARNING, model=model,
                  seconds=int(self._cooldown[model] - time.time()))

    def _build_config(self, request: GenerationRequest, model: str):
        from google.genai import types
        kwargs: dict[str, Any] = {
            "temperature": self.config.temperature if request.temperature is None else request.temperature,
            "max_output_tokens": request.max_output_tokens or self.config.max_output_tokens,
        }
        if request.system_instruction:
            kwargs["system_instruction"] = request.system_instruction
        if request.response_schema:
            kwargs["response_mime_type"] = "application/json"
            kwargs["response_json_schema"] = request.response_schema
        if (self.config.thinking_level and re.match(r"gemini-(3|4)", model)
                and model not in self._no_thinking_models):
            kwargs["thinking_config"] = types.ThinkingConfig(thinking_level=self.config.thinking_level)
        return types.GenerateContentConfig(**kwargs)

    def _backoff(self, attempt: int, error: LLMError) -> float:
        if error.retry_after and error.retry_after <= 20:
            return error.retry_after
        return min(8.0, 0.75 * (2 ** attempt)) + random.uniform(0, 0.4)

    # ------------------------------------------------------------------ generate
    async def _generate_once(self, request: GenerationRequest, model: str, timeout: float) -> LLMResponse:
        client = self._get_client()
        started = time.perf_counter()
        try:
            response = await asyncio.wait_for(
                client.aio.models.generate_content(model=model, contents=request.prompt,
                                                   config=self._build_config(request, model)),
                timeout=timeout)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - mapped below
            error = map_exception(exc, model)
            if isinstance(error, LLMInvalidRequestError) and "thinking" in str(exc).lower():
                self._no_thinking_models.add(model)
            raise error from exc
        _check_blocked(response, model)
        text = _text(response)
        if not text.strip():
            raise LLMEmptyResponseError("Empty response", model=model)
        result = LLMResponse(text=text, model=model, latency_ms=int((time.perf_counter() - started) * 1000),
                             usage=_usage(response), finish_reason=_finish_reason(response))
        if request.response_schema:
            try:
                result.parsed = json.loads(text)
            except json.JSONDecodeError as exc:
                raise LLMResponseParseError("Structured output was not valid JSON", model=model) from exc
        return result

    async def generate(self, request: GenerationRequest) -> LLMResponse:
        attempts = 0
        last_error: LLMError | None = None
        for model in self._models_to_try(request):
            for attempt in range(self.config.max_retries + 1):
                attempts += 1
                try:
                    result = await self._generate_once(request, model, self.config.timeout)
                    result.attempts = attempts
                    self.usage.record_success(result)
                    log_event(logger, "llm.generate.ok", purpose=request.purpose, **result.as_log())
                    return result
                except LLMError as error:
                    last_error = error
                    self.usage.record_failure(error.kind)
                    log_event(logger, "llm.generate.error", logging.WARNING, purpose=request.purpose,
                              model=model, attempt=attempt + 1, error=error.kind, status=error.status_code)
                    if isinstance(error, LLMInvalidRequestError) and model in self._no_thinking_models and attempt == 0:
                        continue  # retry once without thinking config
                    if isinstance(error, (LLMAuthenticationError, LLMPermissionError, LLMConfigurationError,
                                          LLMBlockedResponseError)):
                        raise
                    if isinstance(error, LLMQuotaExhaustedError):
                        self._mark_exhausted(model, error)   # quotas are per model: try the next one
                        break
                    if isinstance(error, (LLMModelNotFoundError, LLMTimeoutError, LLMInvalidRequestError)):
                        break  # next model
                    if not error.retryable or attempt >= self.config.max_retries:
                        break
                    await _sleep(self._backoff(attempt, error))
        assert last_error is not None
        raise last_error

    # ------------------------------------------------------------------ stream
    async def stream(self, request: GenerationRequest) -> AsyncIterator[StreamChunk]:
        last_error: LLMError | None = None
        attempts = 0
        for model in self._models_to_try(request):
            for attempt in range(self.config.max_retries + 1):
                attempts += 1
                yielded = False
                started = time.perf_counter()
                parts: list[str] = []
                usage = TokenUsage()
                finish = None
                try:
                    client = self._get_client()
                    iterator = await asyncio.wait_for(
                        client.aio.models.generate_content_stream(
                            model=model, contents=request.prompt, config=self._build_config(request, model)),
                        timeout=self.config.timeout)
                    aiter = iterator.__aiter__()
                    while True:
                        try:
                            chunk = await asyncio.wait_for(aiter.__anext__(), timeout=self.config.timeout)
                        except StopAsyncIteration:
                            break
                        _check_blocked(chunk, model)
                        delta = _text(chunk)
                        usage = _usage(chunk) if getattr(chunk, "usage_metadata", None) else usage
                        finish = _finish_reason(chunk) or finish
                        if delta:
                            parts.append(delta)
                            yielded = True
                            yield StreamChunk(text=delta)
                    full = "".join(parts)
                    if not full.strip():
                        raise LLMEmptyResponseError("Empty streamed response", model=model)
                    result = LLMResponse(text=full, model=model, usage=usage, finish_reason=finish,
                                         latency_ms=int((time.perf_counter() - started) * 1000), attempts=attempts)
                    self.usage.record_success(result)
                    log_event(logger, "llm.stream.ok", purpose=request.purpose, **result.as_log())
                    yield StreamChunk(done=True, response=result)
                    return
                except asyncio.CancelledError:
                    log_event(logger, "llm.stream.cancelled", model=model)
                    raise
                except Exception as exc:  # noqa: BLE001
                    error = map_exception(exc, model)
                    last_error = error
                    self.usage.record_failure(error.kind)
                    log_event(logger, "llm.stream.error", logging.WARNING, model=model, attempt=attempt + 1,
                              error=error.kind, status=error.status_code, after_partial_output=yielded)
                    if yielded:
                        raise error from exc  # cannot transparently retry after text was sent
                    if isinstance(error, (LLMAuthenticationError, LLMPermissionError, LLMConfigurationError,
                                          LLMBlockedResponseError)):
                        raise error from exc
                    if isinstance(error, LLMQuotaExhaustedError):
                        self._mark_exhausted(model, error)
                        break
                    if isinstance(error, (LLMModelNotFoundError, LLMTimeoutError, LLMInvalidRequestError)):
                        break
                    if not error.retryable or attempt >= self.config.max_retries:
                        break
                    await _sleep(self._backoff(attempt, error))
        assert last_error is not None
        raise last_error

    # ------------------------------------------------------------------ health
    async def health_check(self, timeout: float | None = None, model: str | None = None) -> dict:
        model = model or self.config.model
        timeout = timeout or min(self.config.timeout, 30.0)
        result: dict[str, Any] = {
            "provider": self.name, "status": "unknown", "model": model, "latency_ms": None,
            "checks": {"api_key_configured": bool(self.config.api_key), "authenticated": False,
                       "model_available": False, "generation_ok": False, "response_parsed": False},
        }
        if not self.config.api_key:
            result["status"] = "not_configured"
            result["error"] = LLMConfigurationError().to_dict()
            return result
        try:
            client = self._get_client()
            info = await asyncio.wait_for(client.aio.models.get(model=model), timeout=timeout)
            result["checks"]["authenticated"] = True
            result["checks"]["model_available"] = True
            result["model_info"] = {"display_name": getattr(info, "display_name", None),
                                    "input_token_limit": getattr(info, "input_token_limit", None),
                                    "output_token_limit": getattr(info, "output_token_limit", None)}
        except Exception as exc:  # noqa: BLE001
            error = map_exception(exc, model)
            result["status"] = {"authentication": "auth_failed", "permission": "auth_failed",
                                "model_not_found": "model_unavailable", "timeout": "timeout",
                                "network": "unreachable", "rate_limited": "rate_limited",
                                "quota_exhausted": "quota_exhausted"}.get(error.kind, "error")
            if error.kind in {"model_not_found", "rate_limited", "quota_exhausted"}:   # key was accepted
                result["checks"]["authenticated"] = True
            result["error"] = error.to_dict()
            return result
        try:
            response = await self._generate_once(
                GenerationRequest(prompt="Reply with the single word OK.", max_output_tokens=256,
                                  temperature=0.0, purpose="health_check"), model, timeout)
            result["latency_ms"] = response.latency_ms
            result["checks"]["generation_ok"] = True
            result["checks"]["response_parsed"] = "ok" in response.text.lower()
            result["usage"] = response.usage.as_dict()
            result["status"] = "connected" if result["checks"]["response_parsed"] else "degraded"
        except LLMError as error:
            result["status"] = {"timeout": "timeout", "rate_limited": "rate_limited",
                                "quota_exhausted": "quota_exhausted", "network": "unreachable"}.get(error.kind, "degraded")
            result["error"] = error.to_dict()
        return result
