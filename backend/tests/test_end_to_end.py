"""End-to-end API tests through FastAPI (fake Gemini; real Constitution data)."""
from __future__ import annotations

import json
import re

import fitz
import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.llm.exceptions import LLMServerError
from app.llm.provider_factory import set_llm_provider
from app.security import rate_limiter
from tests.conftest import FakeLLM, default_answer

ADMIN = {"X-Admin-Token": "test-admin"}


@pytest.fixture(scope="module")
def client(knowledge_db):
    from app.main import app
    with TestClient(app) as c:
        yield c


@pytest.fixture
def session(client):
    token = client.post("/api/session").json()["session_token"]
    return {"X-Session-Token": token}


@pytest.fixture
def llm():
    fake = FakeLLM()
    set_llm_provider(fake)
    yield fake
    set_llm_provider(None)


def chat(client, session, message, **extra):
    r = client.post("/api/chat", json={"message": message, **extra}, headers=session)
    assert r.status_code == 200, r.text
    return r.json()


def sse_events(text: str) -> list[tuple[str, dict]]:
    out = []
    for block in text.strip().split("\n\n"):
        event = re.search(r"^event: (.*)$", block, re.M).group(1)
        data = json.loads(re.search(r"^data: (.*)$", block, re.M).group(1))
        out.append((event, data))
    return out


# ----------------------------------------------------------------------------- health / frontend
def test_health_endpoints(client):
    assert client.get("/health/live").json() == {"status": "ok"}
    ready = client.get("/health/ready").json()
    assert ready["ready"] and ready["database"] and ready["default_knowledge"] and ready["knowledge_graph"]
    assert ready["details"]["index"]["articles"] >= 495


def test_frontend_is_served_and_uses_streaming_api(client):
    assert "Indian Constitution AI" in client.get("/").text
    js = client.get("/static/app.js").text
    assert "/api/chat/stream" in js and "X-Session-Token" in js


def test_source_status_reports_verified_official_edition(client):
    status = client.get("/api/sources/status").json()
    assert status["is_available"] and status["is_verified"]
    assert "1st May 2024" in status["details"]["version"]
    assert status["details"]["checksum_sha256"]


# ----------------------------------------------------------------------------- sessions / auth
def test_chat_requires_a_signed_session(client):
    assert client.post("/api/chat", json={"message": "Explain Article 21"}).status_code == 401
    forged = {"X-Session-Token": "attacker-session.0000000000"}
    assert client.post("/api/chat", json={"message": "Explain Article 21"}, headers=forged).status_code == 401


def test_admin_endpoints_require_token(client):
    assert client.get("/api/health/llm").status_code == 403
    assert client.get("/api/admin/graph/inspect").status_code == 403
    r = client.get("/api/health/llm", headers=ADMIN)
    assert r.status_code == 200 and r.json()["status"] in {"not_configured", "connected"}


def test_secrets_are_never_returned(client, session, llm):
    settings = get_settings()
    secret = "AIzaSyFAKE-KEY-FOR-LEAK-TEST-0123456789"
    previous, settings.gemini_api_key = settings.gemini_api_key, secret
    try:
        bodies = [client.get("/health/ready").text, client.get("/api/sources/status").text,
                  client.get("/api/health/llm", headers=ADMIN).text,
                  client.post("/api/chat", json={"message": "Explain Article 21"}, headers=session).text]
        assert all(secret not in body for body in bodies)
    finally:
        settings.gemini_api_key = previous


# ----------------------------------------------------------------------------- chat
def test_validated_answer_with_citations(client, session, llm):
    data = chat(client, session, "Police arrested me without telling me the grounds of arrest and refused access to a lawyer.")
    assert data["status"] == "answered"
    assert data["validation"]["status"] == "passed"
    cited = [c for c in data["citations"] if c["cited_in_answer"]]
    assert cited and all(c["verification_status"] == "verified" for c in cited if c["kind"] == "constitution")
    assert {"Article 22(1)", "Article 21"} <= {c["source_location"] for c in data["citations"]}
    assert data["assessment"]["actor_type"] == "state"
    assert data["confidence"]["level"] in {"high", "medium", "low"} and "components" in data["confidence"]
    assert data["model_used"] == "fake-model" and data["knowledge_graph_used"]
    assert "graph_sql" in data["retrieval_strategy"] and "bm25" in data["retrieval_strategy"]
    prompt = llm.requests[-1].prompt
    assert "APPLICABILITY ASSESSMENT" in prompt and "EVIDENCE" in prompt
    history = client.get(f"/api/conversations/{data['conversation_id']}", headers=session).json()
    assert [m["role"] for m in history["messages"]] == ["user", "assistant"]
    assert history["messages"][1]["status"] == "answered"


def test_streaming_answer_emits_tokens_then_validated_final(client, session, llm):
    r = client.post("/api/chat/stream", json={"message": "Explain Article 21A"}, headers=session)
    events = sse_events(r.text)
    names = [e for e, _ in events]
    assert names[0] == "stage" and "meta" in names and "token" in names and names[-1] == "final"
    assert names.index("token") < names.index("final")
    final = events[-1][1]
    assert final["status"] == "answered" and final["citations"]


def test_gemini_failure_returns_grounded_evidence_only_answer(client, session, llm):
    llm.answers = [LLMServerError("down", status_code=503)]
    data = chat(client, session, "Someone stole my mobile phone on a crowded bus. Has my Article 21 right been violated?")
    assert data["status"] == "degraded"
    assert data["error"]["type"] == "server_error"
    assert "assembled only from verified source text" in data["content"]
    assert "Article 21" in data["content"] and "[C1]" in data["content"]
    assert "does not directly apply to a private person" in data["content"]


def test_no_api_key_returns_evidence_only_answer(client, session):
    set_llm_provider(None)
    data = chat(client, session, "Explain Article 14")
    assert data["status"] == "degraded" and data["error"]["type"] == "llm_not_configured"
    assert data["model_used"] is None


def test_hallucinated_draft_is_repaired(client, session, llm):
    bad = "Your rights were violated under Article 999 [C1]."
    llm.answers = [bad, default_answer]
    data = chat(client, session, "Explain Article 19(1)(a)", debug=True)
    assert data["status"] == "answered"
    assert "Article 999" not in data["content"]
    assert "debug" not in data                      # debug requires admin token
    assert [r.purpose for r in llm.requests][-2:] == ["final_answer", "citation_repair"]


def test_unrepairable_draft_is_not_shown_as_a_legal_answer(client, session, llm):
    bad = "Article 21 has been violated and the thief must pay damages under Section 999 [C7]."
    llm.answers = [bad, bad]
    data = client.post("/api/chat", json={"message": "Someone stole my mobile phone. Has my Article 21 right been "
                                                      "violated?", "debug": True},
                       headers={**session, **ADMIN}).json()
    assert data["status"] == "unverified"
    assert data["unverified_draft"] == bad
    assert "did not pass citation checks" in data["content"]
    assert data["validation"]["status"] == "failed"
    stages = data["debug"]["stages"]
    for stage in ["query_understanding", "retrieval", "source_extraction", "applicability", "generation",
                  "citation_validation", "final"]:
        assert stage in stages
    assert stages["citation_validation"]["repaired"] is True
    trace = client.get(f"/api/admin/traces/{data['request_id']}", headers=ADMIN).json()
    assert trace["stages"]["final"]["status"] == "unverified"
    assert "text" not in trace["question"]           # question stored as a hash by default


def test_out_of_scope_question_abstains_without_calling_gemini(client, session, llm):
    data = chat(client, session, "Who won the cricket match yesterday?")
    assert data["status"] == "abstained" and not llm.requests


def test_follow_up_question_uses_conversation_context(client, session, llm):
    first = chat(client, session, "Police arrested me without telling me why.")
    chat(client, session, "What about the 24 hour rule?", conversation_id=first["conversation_id"])
    prompt = llm.requests[-1].prompt
    assert "EARLIER CONVERSATION" in prompt and "arrested me" in prompt
    assert "Article 22(2)" in prompt


def test_conversations_are_private_to_the_session(client, session, llm):
    data = chat(client, session, "Explain Article 17")
    other = {"X-Session-Token": client.post("/api/session").json()["session_token"]}
    assert client.get(f"/api/conversations/{data['conversation_id']}", headers=other).status_code == 404
    assert all(c["id"] != data["conversation_id"] for c in client.get("/api/conversations", headers=other).json())


def test_upload_is_session_scoped(client, session, tmp_path):
    pdf = fitz.open()
    pdf.new_page().insert_text((72, 72), "Hostel rule 12: the warden must give written notice before a room search.")
    data = pdf.tobytes()
    pdf.close()
    r = client.post("/api/documents/upload", files={"file": ("rules.pdf", data, "application/pdf")}, headers=session)
    assert r.status_code == 200 and r.json()["ingestion_status"] == "completed"
    assert r.json()["verification_status"] == "unverified"
    assert any(d["title"] == "rules.pdf" for d in client.get("/api/documents", headers=session).json())
    other = {"X-Session-Token": client.post("/api/session").json()["session_token"]}
    assert client.get("/api/documents", headers=other).json() == []
    bad = client.post("/api/documents/upload", files={"file": ("x.pdf", b"not a pdf", "application/pdf")}, headers=session)
    assert bad.status_code == 415


def test_rate_limit(client, session, llm):
    settings = get_settings()
    previous, settings.rate_limit_requests = settings.rate_limit_requests, 1
    rate_limiter.reset()
    try:
        chat(client, session, "Explain Article 14")
        assert client.post("/api/chat", json={"message": "Explain Article 15"}, headers=session).status_code == 429
    finally:
        settings.rate_limit_requests = previous
        rate_limiter.reset()


# ----------------------------------------------------------------------------- graph / provisions
def test_graph_endpoint_returns_bounded_subgraph_and_rejects_injection(client):
    graph = client.get("/api/knowledge-graph/related", params={"provisions": "COI:art:22:cl:1"}).json()
    assert graph["nodes"] and graph["edges"]
    assert any(e["relation"] == "LIMITED_BY" for e in graph["edges"])
    injected = client.get("/api/knowledge-graph/related", params={"provisions": "MATCH (n) DETACH DELETE n"}).json()
    assert injected["nodes"] == []


def test_provision_endpoint(client):
    data = client.get("/api/provisions/COI:art:19:cl:2").json()
    assert data["label"] == "Article 19(2)" and data["amendments"][0]["amendment_act"].startswith("Constitution (First")


def test_admin_graph_inspection(client):
    report = client.get("/api/admin/graph/inspect", headers=ADMIN).json()
    assert report["healthy"] and report["total_nodes"] > 3000
