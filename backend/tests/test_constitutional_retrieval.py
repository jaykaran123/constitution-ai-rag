"""Hybrid retrieval: exact lookup, BM25, vectors, graph routing, fusion and expansion."""
from __future__ import annotations

import hashlib
import math

import numpy as np
import pytest

from app.llm.base import EmbeddingProvider
from app.models import DocumentChunk, LegalDocument
from app.retrieval.hybrid import HybridRetriever, ProvisionStore
from app.retrieval.indexes import BM25Index, VectorIndex, clear_index_caches
from app.retrieval.query_understanding import analyse, parse_refs
from tests.conftest import run


class HashEmbedder(EmbeddingProvider):
    """Deterministic bag-of-words embedder (no network) for testing the vector path."""

    def __init__(self, dims=64, model="test-embed"):
        self._dims, self._model, self.calls = dims, model, 0

    @property
    def model(self):
        return self._model

    @property
    def dimensions(self):
        return self._dims

    def _vec(self, text):
        v = [0.0] * self._dims
        for word in text.lower().split():
            v[int(hashlib.md5(word.strip(".,;:()").encode()).hexdigest(), 16) % self._dims] += 1
        n = math.sqrt(sum(x * x for x in v)) or 1
        return [x / n for x in v]

    async def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    async def embed_query(self, text):
        self.calls += 1
        return self._vec(text)


def retrieve(db, question, embedder=None, session_id=None):
    by_id, _ = ProvisionStore.load(db)
    analysis = run(analyse(question, set(by_id), mode="rules"))
    return analysis, run(HybridRetriever(db, embedder).retrieve(analysis, question, session_id))


def labels(result):
    return [e.label for e in result.constitution]


@pytest.mark.parametrize("text,expected", [
    ("Explain Article 19(1)(a)", ["COI:art:19:cl:1:sub:a"]),
    ("art. 21a and Article 22 (2)", ["COI:art:21A", "COI:art:22:cl:2"]),
    ("Articles 14, 15 and 16", ["COI:art:14", "COI:art:15", "COI:art:16"]),
    ("Article 51A(a)", ["COI:art:51A:sub:a"]),
])
def test_reference_parsing(text, expected):
    assert [r.provision_id for r in parse_refs(text)] == expected


def test_exact_clause_lookup_pulls_its_restriction(db):
    _, result = retrieve(db, "What does Article 19(1)(a) say?")
    assert labels(result)[0] == "Article 19(1)(a)"
    restriction = next(e for e in result.constitution if e.label == "Article 19(2)")
    assert "restriction" in restriction.roles
    # sub-clause evidence is shown with the lead-in of its clause, so it reads as complete law
    assert result.constitution[0].text.startswith("(1) All citizens shall have the right")


def test_exact_article_lookup_returns_the_full_article(db):
    _, result = retrieve(db, "Explain Article 21")
    article = result.constitution[0]
    assert article.label == "Article 21" and "explicit" in article.roles
    assert "procedure established by law" in article.text
    assert article.page == 42 and article.verification == "verified"


def test_keyword_retrieval_without_routing(db):
    _, result = retrieve(db, "abolition of untouchability")
    assert "Article 17" in labels(result)


def test_schedule_entries_do_not_masquerade_as_articles(db):
    _, result = retrieve(db, "Explain Article 19")
    assert all("Drugs and poisons" not in e.text for e in result.constitution)
    _, result = retrieve(db, "Which list of the Seventh Schedule contains police?")
    assert any(e.provision_id.startswith("COI:sch:7:list:II:entry:2") for e in result.constitution)


def test_graph_routing_supplies_statutes_and_remedies(db):
    analysis, result = retrieve(db, "My bike was stolen and the police refused to register my FIR.")
    assert "police_refusal_complaint" in analysis.issues
    assert any("173(4)" in s.label for s in result.statutes)
    assert all(s.verification in {"verified", "unverified"} for s in result.statutes)
    assert result.diagnostics["graph"]["backend"] == "sql"


def test_omitted_provisions_are_excluded_unless_asked(db):
    _, result = retrieve(db, "compulsory acquisition of property by the State")
    assert "Article 31" not in labels(result)      # omitted 1978; matches by title but must not be cited as law
    _, result = retrieve(db, "Was Article 31 omitted by an amendment?")
    assert "Article 31" in labels(result)


def test_vector_search_participates_in_fusion(db):
    embedder = HashEmbedder()
    chunks = db.query(DocumentChunk).filter(DocumentChunk.article_number.in_(["17", "23", "24", "21A"])).all()
    vectors = run(embedder.embed_documents([c.content for c in chunks]))
    for chunk, vector in zip(chunks, vectors):
        chunk.embedding = np.asarray(vector, dtype=np.float32).tobytes()
        chunk.embedding_model, chunk.embedding_dim = embedder.model, embedder.dimensions
    db.commit()
    clear_index_caches()
    try:
        _, result = retrieve(db, "children below fourteen employed in a factory or mine", embedder)
        assert "vector" in result.diagnostics["strategy"]
        assert embedder.calls == 1
        child = next(e for e in result.constitution if e.label == "Article 24")
        assert "vector" in child.scores and "bm25" in child.scores
        # a different model/dimension must never be mixed into the index
        other = HashEmbedder(dims=32, model="other-model")
        index = VectorIndex.for_documents(db, [chunks[0].document_id], other.model, other.dimensions)
        assert index.size == 0
    finally:
        for chunk in chunks:
            chunk.embedding = chunk.embedding_model = chunk.embedding_dim = None
        db.commit()
        clear_index_caches()


def test_uploads_are_session_scoped_and_not_authoritative(db):
    doc = LegalDocument(title="hostel-rules.pdf", document_type="user_upload", session_id="session-a",
                        ingestion_status="completed", verification_status="unverified")
    db.add(doc)
    db.flush()
    db.add(DocumentChunk(document_id=doc.id, content="Hostel regulation: written notice is required before any "
                         "room search by the warden.", chunk_index=0, chunk_key=f"UPLOAD:{doc.id}:0",
                         chunk_type="upload"))
    db.commit()
    try:
        _, mine = retrieve(db, "room search by the warden without notice", session_id="session-a")
        _, theirs = retrieve(db, "room search by the warden without notice", session_id="session-b")
        assert mine.uploads and mine.uploads[0].verification == "user_supplied_unverified"
        assert not theirs.uploads
    finally:
        db.delete(doc)
        db.commit()
        clear_index_caches()


def test_bm25_index_is_cached_until_chunks_change(db):
    doc = db.query(LegalDocument).filter_by(is_default_source=True).one()
    assert BM25Index.for_documents(db, [doc.id]) is BM25Index.for_documents(db, [doc.id])
