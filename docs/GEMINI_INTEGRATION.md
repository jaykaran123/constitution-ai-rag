# Gemini integration

## SDK and structure

Official Google GenAI SDK `google-genai==1.75.0` (`from google import genai`). The deprecated `google-generativeai` package is not used. Only `app/llm/` imports the SDK.

```
backend/app/llm/
  __init__.py          public API
  base.py              LLMProvider / EmbeddingProvider interfaces, GenerationRequest, LLMResponse, StreamChunk, UsageTracker
  config.py            LLMConfig built from settings (repr hides the key; public_dict never contains it)
  exceptions.py        typed errors with retryable flags and user-facing messages
  gemini_provider.py   GeminiProvider: generate, stream, health_check, error mapping, retries, fallback
  embeddings.py        GeminiEmbeddingProvider: batching, pacing, count/dimension validation
  provider_factory.py  ProviderFactory + get_llm_provider()/get_embedding_provider() (test overrides)
  prompts.py           the seven task prompts
```

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `LLM_PROVIDER` | `gemini` | `google` is accepted as an alias |
| `GEMINI_API_KEY` | – | server only; legacy `LLM_API_KEY` still read |
| `GEMINI_MODEL` | `gemini-3.6-flash` | primary model |
| `GEMINI_FALLBACK_MODELS` | `gemini-3.5-flash-lite` | comma-separated |
| `GEMINI_EMBEDDING_MODEL` | `gemini-embedding-001` | |
| `GEMINI_EMBEDDING_DIMENSIONS` | `768` | 768 / 1536 / 3072 |
| `GEMINI_TIMEOUT` | `45` | seconds per attempt |
| `GEMINI_MAX_RETRIES` | `2` | retries per model for retryable errors |
| `GEMINI_TEMPERATURE`, `GEMINI_MAX_OUTPUT_TOKENS`, `GEMINI_THINKING_LEVEL` | 0.2, 3072, low | |
| `EMBEDDING_TEXTS_PER_MINUTE` | 90 | client-side pacing for free-tier quotas |

**Model identifiers were verified on 30 October 2026** against https://ai.google.dev/gemini-api/docs/models and the key's own `models.list()` result. Both list `gemini-3.8-flash`, `gemini-3.6-flash`, `gemini-3.5-flash-lite` and `gemini-embedding-001`.

| Model | Health check result with this key |
|---|---|
| `gemini-3.8-flash` (previous default) | 332 s for one "OK" reply, then timeouts at 25 s |
| `gemini-3.6-flash` (new default) | connected, ~3 s |
| `gemini-3.5-flash-lite` (fallback) | connected, ~2 s |

`gemini-embedding-2` was tested and rejected for batching. It returned one aggregated vector for a two-text request, so the embedding provider checks that the number and size of vectors match the request.

## Request handling

* **Async** calls via `client.aio.models.*`; the SDK's own retry is disabled (`HttpRetryOptions(attempts=1)`) so policy lives in one place.
* **Timeouts** per attempt (`asyncio.wait_for`), per streamed chunk, and in the HTTP client.
* **Retry policy**

| Error | Mapped to | Action |
|---|---|---|
| 400 "API key not valid", 401, 403 with key message | `LLMAuthenticationError` | fail immediately |
| 403 | `LLMPermissionError` | fail immediately |
| 404 | `LLMModelNotFoundError` | next fallback model |
| 408, timeout | `LLMTimeoutError` | next fallback model (a slow model rarely recovers) |
| 429 with a short retry delay | `LLMRateLimitError` | back off (uses the server's delay when ≤ 20 s) and retry |
| 429 per-day / `limit: 0` / delay > 10 min | `LLMQuotaExhaustedError` | fail immediately |
| 5xx | `LLMServerError` | exponential backoff with jitter, then fallback |
| network | `LLMNetworkError` | backoff and retry |
| empty text | `LLMEmptyResponseError` | retry, never returned as an answer |
| blocked prompt/response | `LLMBlockedResponseError` | fail |
| invalid JSON for a schema request | `LLMResponseParseError` | retry |

* **Streaming** with `generate_content_stream`. Before the first token, failures are retried and fall back. After partial output they raise, and the pipeline returns the labelled evidence-only answer instead.
* **Cancellation**: `asyncio.CancelledError` is never swallowed. When the browser disconnects, the SSE generator stops and the Gemini stream is closed.
* **Structured output** with `response_mime_type="application/json"` and `response_json_schema`, used for scenario classification and graph extraction.
* **Token usage** (`prompt`, `output`, `thinking`, `total`) is recorded per response, in logs and traces, and aggregated in `provider.usage.snapshot()`, which `/api/health/llm` exposes.
* **No silent failures.** When Gemini is unavailable, the chat response has `status: degraded` and an `error` object with type and message. The content is explicitly labelled as assembled from verified sources only.

## Health check

`GET /api/health/llm` (admin) verifies:

* the key is configured;
* authentication and model availability (`models.get`);
* a minimal generation succeeds and its response parses.

If the primary model fails, each fallback is probed. Example (live, 30 October 2026):

```json
{"provider": "gemini", "status": "connected", "model": "gemini-3.6-flash", "latency_ms": 3148,
 "checks": {"api_key_configured": true, "authenticated": true, "model_available": true,
            "generation_ok": true, "response_parsed": true},
 "configuration": {"model": "gemini-3.6-flash", "fallback_models": ["gemini-3.5-flash-lite"],
                   "api_key_configured": true}}
```

Statuses: `connected`, `degraded` (fallback in use), `not_configured`, `auth_failed`, `model_unavailable`, `rate_limited`, `quota_exhausted`, `timeout`, `unreachable`. The response never contains the key. `python -m scripts.check_gemini` runs the same checks plus streaming and an embedding from the command line.

## Prompts (`app/llm/prompts.py`)

1. `CONSTITUTIONAL_SYSTEM`: the standing principles (the 15 rules in the brief, condensed).
2. `ARTICLE_IDENTIFICATION_*`: choose provisions only from a candidate list (available for re-ranking experiments; selection is deterministic by default).
3. `SCENARIO_CLASSIFICATION_*`: closed-list issue classification with a JSON schema.
4. `GRAPH_EXTRACTION_*`: propose relationships from uploads, each with a verbatim supporting sentence (stored as unverified suggestions).
5. `CONTEXT_SUMMARY_*`: summarise long conversations for follow-ups.
6. `CITATION_VERIFICATION_*`: repair a draft using the validator's error list.
7. `final_answer_prompt` + `ANSWER_FORMAT`: the user-facing answer structure.

Evidence and documents are wrapped in `<<< >>>` delimiters and described as untrusted data. The prompt-injection evaluation case checks that instructions in the question are not followed.

## Free-tier quota observations (30 October 2026)

* Embeddings: 1 000 requests/day per model on the free tier, counted per text. The 1 688 chunks therefore need two days on the free tier. 914 are embedded. `python -m scripts.embed_chunks` resumes and reuses existing vectors.
* Query-time embedding fails fast without waiting for the quota. A busy or exhausted quota only removes the vector signal for that question; exact, graph and BM25 retrieval continue.
