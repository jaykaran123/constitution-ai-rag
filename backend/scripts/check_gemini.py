"""Verify Gemini connectivity: API key, model availability, generation, streaming, embeddings.
Prints no secrets.

    python -m scripts.check_gemini
"""
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.llm.base import GenerationRequest  # noqa: E402
from app.llm.config import LLMConfig  # noqa: E402
from app.llm.provider_factory import get_embedding_provider, get_llm_provider  # noqa: E402


async def main() -> None:
    config = LLMConfig.from_settings()
    print(json.dumps({"configuration": config.public_dict()}, indent=1))
    provider = get_llm_provider()
    if provider is None:
        raise SystemExit("GEMINI_API_KEY is not set.")
    for model in [config.model, *config.fallback_models]:
        print(json.dumps({"health": await provider.health_check(model=model)}, indent=1))
    parts: list[str] = []
    request = GenerationRequest(prompt="Name the Article of the Indian Constitution that protects life and personal "
                                       "liberty. One short sentence.", purpose="connectivity_check")
    async for chunk in provider.stream(request):
        if chunk.done:
            print(json.dumps({"stream": {"model": chunk.response.model, "latency_ms": chunk.response.latency_ms,
                                         "usage": chunk.response.usage.as_dict(), "text": "".join(parts)}}, indent=1))
        else:
            parts.append(chunk.text)
    embedder = get_embedding_provider()
    if embedder:
        vector = await embedder.embed_query("Article 21 personal liberty")
        print(json.dumps({"embedding": {"model": embedder.model, "dimensions": len(vector)}}))
    print(json.dumps({"usage": provider.usage.snapshot()}, indent=1))


if __name__ == "__main__":
    asyncio.run(main())
