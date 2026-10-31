"""Gemini provider: error mapping, retries, fallback, streaming, health checks, embeddings.

The SDK client is replaced with a scripted fake so every failure mode is exercised
deterministically. ``test_live_*`` run against the real API only when RUN_LIVE_GEMINI=1.
"""
from __future__ import annotations

import asyncio
import os
from types import SimpleNamespace

import pytest
from google.genai import errors as genai_errors

from app.llm import exceptions as exc
from app.llm.base import GenerationRequest
from app.llm.config import LLMConfig
from app.llm.embeddings import GeminiEmbeddingProvider
from app.llm.gemini_provider import GeminiProvider, map_exception
from tests.conftest import run


def config(**overrides) -> LLMConfig:
    base = dict(provider="gemini", api_key="test-key-not-real", model="primary-model", fallback_models=("fallback-model",),
                classifier_model=None, embedding_model="gemini-embedding-001", embedding_dimensions=4, timeout=0.3,
                max_retries=2, temperature=0.2, max_output_tokens=256, thinking_level=None)
    base.update(overrides)
    return LLMConfig(**base)


def api_error(code: int, message: str = "error", status: str = "") -> Exception:
    body = {"error": {"code": code, "message": message, "status": status}}
    cls = genai_errors.ServerError if code >= 500 else genai_errors.ClientError
    return cls(code, body)


def response(text="OK", finish="STOP", block=None):
    usage = SimpleNamespace(prompt_token_count=10, candidates_token_count=5, thoughts_token_count=2, total_token_count=17)
    return SimpleNamespace(text=text, usage_metadata=usage,
                           candidates=[SimpleNamespace(finish_reason=SimpleNamespace(name=finish))],
                           prompt_feedback=SimpleNamespace(block_reason=block) if block else None)


class FakeModels:
    def __init__(self, script):
        self.script = list(script)       # items: response, Exception, ("sleep", seconds)
        self.calls: list[str] = []

    async def generate_content(self, *, model, contents, config):
        self.calls.append(model)
        item = self.script.pop(0)
        if isinstance(item, tuple) and item[0] == "sleep":
            await asyncio.sleep(item[1])
            return response()
        if isinstance(item, Exception):
            raise item
        return item

    async def generate_content_stream(self, *, model, contents, config):
        self.calls.append(model)
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item

        async def gen():
            for part in item:
                if isinstance(part, Exception):
                    raise part
                yield response(part)
        return gen()

    async def get(self, *, model):
        self.calls.append(f"get:{model}")
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return SimpleNamespace(display_name=model, input_token_limit=1000, output_token_limit=100)

    async def embed_content(self, *, model, contents, config):
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return SimpleNamespace(embeddings=[SimpleNamespace(values=v) for v in item])


def provider(script, **cfg):
    models = FakeModels(script)
    return GeminiProvider(config(**cfg), client=SimpleNamespace(aio=SimpleNamespace(models=models))), models


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    async def fast(seconds, *a, **k):
        await asyncio.sleep(0)
    monkeypatch.setattr("app.llm.gemini_provider._sleep", fast)
    monkeypatch.setattr("app.llm.embeddings._sleep", fast)


# ----------------------------------------------------------------------------- error mapping
@pytest.mark.parametrize("error,kind,retryable", [
    (api_error(400, "API key not valid. Please pass a valid API key."), "authentication", False),
    (api_error(403, "Permission denied on resource"), "permission", False),
    (api_error(404, "models/x is not found"), "model_not_found", False),
    (api_error(429, "Resource exhausted. Please retry in 7s.", "RESOURCE_EXHAUSTED"), "rate_limited", True),
    (api_error(429, "Quota exceeded for metric generate_requests_per_model_per_day, limit: 0", "RESOURCE_EXHAUSTED"),
     "quota_exhausted", False),
    (api_error(503, "The model is overloaded"), "server_error", True),
    (api_error(400, "Invalid argument"), "invalid_request", False),
    (asyncio.TimeoutError(), "timeout", True),
])
def test_error_mapping(error, kind, retryable):
    mapped = map_exception(error, "m")
    assert mapped.kind == kind and mapped.retryable is retryable


def test_rate_limit_retry_after_is_parsed():
    assert map_exception(api_error(429, "Please retry in 7s.", "RESOURCE_EXHAUSTED")).retry_after == 7.0


def test_network_errors_are_retryable():
    import httpx
    assert map_exception(httpx.ConnectError("boom")).kind == "network"
    assert map_exception(httpx.ReadTimeout("slow")).kind == "timeout"


# ----------------------------------------------------------------------------- generate
def test_transient_server_error_is_retried_then_succeeds():
    p, models = provider([api_error(503), response("Grounded answer")])
    result = run(p.generate(GenerationRequest(prompt="q")))
    assert result.text == "Grounded answer" and result.attempts == 2
    assert models.calls == ["primary-model", "primary-model"]
    assert result.usage.total_tokens == 17
    assert p.usage.snapshot()["failures"] == {"server_error": 1}


def test_authentication_failure_is_not_retried():
    p, models = provider([api_error(400, "API key not valid")])
    with pytest.raises(exc.LLMAuthenticationError):
        run(p.generate(GenerationRequest(prompt="q")))
    assert models.calls == ["primary-model"]


def test_quota_exhaustion_is_not_retried_but_falls_back_to_another_model():
    quota = api_error(429, "Quota exceeded: requests per day. Please retry in 12h54m44.2s.", "RESOURCE_EXHAUSTED")
    p, models = provider([quota, response("from fallback")])
    assert run(p.generate(GenerationRequest(prompt="q"))).model == "fallback-model"
    assert models.calls == ["primary-model", "fallback-model"]          # primary not retried
    # the exhausted model is skipped on the next request until its quota resets
    models.script = [response("again")]
    assert run(p.generate(GenerationRequest(prompt="q"))).model == "fallback-model"
    assert models.calls[-1] == "fallback-model" and models.calls.count("primary-model") == 1


def test_quota_exhausted_on_every_model_raises():
    quota = api_error(429, "per day limit: 0", "RESOURCE_EXHAUSTED")
    p, models = provider([quota, quota])
    with pytest.raises(exc.LLMQuotaExhaustedError):
        run(p.generate(GenerationRequest(prompt="q")))
    assert models.calls == ["primary-model", "fallback-model"]


def test_rate_limit_backs_off_and_retries():
    p, models = provider([api_error(429, "retry in 1s", "RESOURCE_EXHAUSTED"), response("ok")])
    assert run(p.generate(GenerationRequest(prompt="q"))).text == "ok"
    assert len(models.calls) == 2


def test_timeout_moves_to_fallback_model():
    p, models = provider([("sleep", 1.0), response("from fallback")], timeout=0.05)
    result = run(p.generate(GenerationRequest(prompt="q")))
    assert result.model == "fallback-model" and result.text == "from fallback"
    assert models.calls == ["primary-model", "fallback-model"]


def test_unknown_model_falls_back():
    p, models = provider([api_error(404, "not found"), response("ok")])
    assert run(p.generate(GenerationRequest(prompt="q"))).model == "fallback-model"


def test_empty_response_is_an_error_not_an_empty_answer():
    p, _ = provider([response("")] * 6)
    with pytest.raises(exc.LLMEmptyResponseError):
        run(p.generate(GenerationRequest(prompt="q")))


def test_blocked_response_raises():
    p, _ = provider([response("", finish="SAFETY")])
    with pytest.raises(exc.LLMBlockedResponseError):
        run(p.generate(GenerationRequest(prompt="q")))


def test_exhausted_retries_raise_the_last_error():
    p, models = provider([api_error(503)] * 6)
    with pytest.raises(exc.LLMServerError):
        run(p.generate(GenerationRequest(prompt="q")))
    assert models.calls.count("primary-model") == 3 and models.calls.count("fallback-model") == 3


def test_structured_output_is_parsed_and_validated():
    p, _ = provider([response('{"issues": ["arrest_safeguards"]}')])
    result = run(p.generate(GenerationRequest(prompt="q", response_schema={"type": "object"})))
    assert result.parsed == {"issues": ["arrest_safeguards"]}
    p, _ = provider([response("not json")] * 6)
    with pytest.raises(exc.LLMResponseParseError):
        run(p.generate(GenerationRequest(prompt="q", response_schema={"type": "object"})))


def test_cancellation_propagates():
    p, _ = provider([("sleep", 5.0)], timeout=10)

    async def scenario():
        task = asyncio.create_task(p.generate(GenerationRequest(prompt="q")))
        await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    run(scenario())


def test_missing_api_key_is_a_configuration_error():
    p = GeminiProvider(config(api_key=None))
    with pytest.raises(exc.LLMConfigurationError):
        run(p.generate(GenerationRequest(prompt="q")))


# ----------------------------------------------------------------------------- streaming
def test_streaming_yields_deltas_then_final_response():
    p, _ = provider([["Article ", "21 ", "says…"]])

    async def collect():
        return [c async for c in p.stream(GenerationRequest(prompt="q"))]
    chunks = run(collect())
    assert [c.text for c in chunks if not c.done] == ["Article ", "21 ", "says…"]
    assert chunks[-1].done and chunks[-1].response.text == "Article 21 says…"


def test_stream_failure_before_output_falls_back():
    p, models = provider([api_error(404), ["fallback text"]])

    async def collect():
        return [c async for c in p.stream(GenerationRequest(prompt="q"))]
    assert run(collect())[-1].response.model == "fallback-model"


def test_stream_failure_after_partial_output_raises():
    p, _ = provider([["partial ", api_error(503)]])

    async def collect():
        return [c async for c in p.stream(GenerationRequest(prompt="q"))]
    with pytest.raises(exc.LLMServerError):
        run(collect())


# ----------------------------------------------------------------------------- health check
def test_health_check_connected_without_secrets():
    p, _ = provider([SimpleNamespace(display_name="m"), response("OK")])   # models.get, then generate
    result = run(p.health_check())
    assert result["status"] == "connected" and result["latency_ms"] is not None
    assert all(result["checks"].values())
    assert "test-key-not-real" not in str(result)


@pytest.mark.parametrize("error,status", [
    (api_error(400, "API key not valid"), "auth_failed"),
    (api_error(404, "not found"), "model_unavailable"),
    (api_error(429, "retry in 5s", "RESOURCE_EXHAUSTED"), "rate_limited"),
])
def test_health_check_reports_failures(error, status):
    p, _ = provider([error])
    assert run(p.health_check())["status"] == status


def test_health_check_not_configured():
    assert run(GeminiProvider(config(api_key=None)).health_check())["status"] == "not_configured"


def test_public_config_never_contains_key():
    assert "test-key-not-real" not in str(config().public_dict())
    assert "test-key-not-real" not in repr(config())


def test_model_configuration_is_environment_driven(monkeypatch):
    from app.config import Settings
    monkeypatch.setenv("GEMINI_MODEL", "gemini-env-model")
    monkeypatch.setenv("GEMINI_TIMEOUT", "12")
    monkeypatch.setenv("GEMINI_MAX_RETRIES", "4")
    monkeypatch.setenv("LLM_PROVIDER", "google")
    s = Settings()
    assert (s.gemini_model, s.gemini_timeout, s.gemini_max_retries, s.llm_provider) == ("gemini-env-model", 12.0, 4, "gemini")


# ----------------------------------------------------------------------------- embeddings
def embedder(script, **cfg):
    models = FakeModels(script)
    return GeminiEmbeddingProvider(config(**cfg), client=SimpleNamespace(aio=SimpleNamespace(models=models)), batch_size=2)


def test_embeddings_are_batched_and_normalised():
    e = embedder([[[3, 4, 0, 0], [0, 0, 1, 0]], [[1, 0, 0, 0]]])
    vectors = run(e.embed_documents(["a", "b", "c"]))
    assert len(vectors) == 3 and abs(vectors[0][0] - 0.6) < 1e-9


def test_embedding_count_mismatch_is_detected():
    e = embedder([[[1, 0, 0, 0]]])          # two texts in, one vector out (the gemini-embedding-2 pitfall)
    with pytest.raises(exc.EmbeddingDimensionMismatch):
        run(e.embed_documents(["a", "b"]))


def test_embedding_dimension_mismatch_is_detected():
    e = embedder([[[1, 0, 0]]])
    with pytest.raises(exc.EmbeddingDimensionMismatch):
        run(e.embed_query("a"))


# ----------------------------------------------------------------------------- live (opt-in)
live = pytest.mark.skipif(os.environ.get("RUN_LIVE_GEMINI") != "1", reason="set RUN_LIVE_GEMINI=1 to call the real API")


@live
def test_live_gemini_health_and_generation():
    from dotenv import dotenv_values
    from app.config import BACKEND_DIR
    values = dotenv_values(BACKEND_DIR / ".env")
    key = values.get("GEMINI_API_KEY") or values.get("LLM_API_KEY")
    model = values.get("GEMINI_MODEL") or "gemini-3.6-flash"
    fallback = (values.get("GEMINI_FALLBACK_MODELS") or "gemini-3.5-flash-lite").split(",")[0].strip()
    p = GeminiProvider(config(api_key=key, model=model, fallback_models=(fallback,), timeout=45))
    e = GeminiEmbeddingProvider(config(api_key=key, embedding_dimensions=768, timeout=45))

    async def scenario():   # one event loop: the SDK's async client is bound to the loop that created it
        health = {m: await p.health_check(model=m) for m in (model, fallback)}
        result = await p.generate(GenerationRequest(prompt="Reply with the single word OK.", max_output_tokens=64))
        try:
            vector = await e.embed_query("Article 21")
        except exc.LLMQuotaExhaustedError:
            vector = None     # free-tier daily embedding quota; reported, not a connectivity failure
        return health, result, vector

    health, result, vector = run(scenario())
    # Free-tier keys have small per-model daily quotas; connectivity is verified if any configured model answers.
    assert any(h["status"] == "connected" for h in health.values()), health
    assert all(h["checks"]["authenticated"] for h in health.values()), health
    assert key not in str(health)
    assert "ok" in result.text.lower() and result.usage.total_tokens > 0
    assert vector is None or len(vector) == 768
