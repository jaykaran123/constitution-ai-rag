"""Knowledge graph integrity, provenance, idempotency and (optionally) Neo4j projection."""
from __future__ import annotations

import os

import pytest
from sqlalchemy.exc import IntegrityError

from app.graph.builder import GraphBuild, GraphValidationError, build_graph
from app.graph.inspect import inspect_graph
from app.graph.queries import SqlGraphQueries
from app.graph.store import GraphCache, persist_graph
from app.models import KnowledgeEntity, KnowledgeRelation


def test_graph_is_healthy_and_connected(db):
    report = inspect_graph(db)
    assert report["healthy"], report
    assert report["orphan_nodes"]["count"] == 0
    assert report["articles_without_chunks_count"] == 0
    assert report["chunks_without_source_documents"] == 0
    assert report["invalid_relationships"]["count"] == 0
    assert report["relationships_missing_provenance"]["count"] == 0
    assert not report["missing_required_relationships"]
    for label in ["Constitution", "Preamble", "Part", "Chapter", "Article", "Clause", "SubClause", "Schedule",
                  "Amendment", "FundamentalRight", "ConstitutionalPrinciple", "LegalConcept", "Statute",
                  "StatutorySection", "Offence", "LegalRemedy", "GovernmentAuthority", "Court", "LegalRestriction",
                  "Exception", "Document", "DocumentVersion", "Chunk", "Source", "Citation", "LegalScenario",
                  "LegalIssue", "UserRight"]:
        assert report["nodes_by_type"].get(label), f"no {label} nodes"


def test_structural_relationships_carry_exact_provenance(db):
    rel = (db.query(KnowledgeRelation).filter_by(key="COI:art:22|HAS_CLAUSE|COI:art:22:cl:1").one())
    prov = rel.provenance_json
    assert prov["source_document_id"] and prov["document_version"]
    assert prov["provision_id"] == "COI:art:22:cl:1" and prov["page"] == 42
    assert prov["extraction_method"] == "structural" and prov["verification_status"] == "verified"


def test_text_pattern_restriction_is_traceable_to_the_sentence(db):
    rel = db.query(KnowledgeRelation).filter_by(
        key="COI:art:19:cl:1:sub:a|SUBJECT_TO|LegalRestriction:COI:art:19:cl:2").one()
    assert rel.extraction_method == "text_pattern"
    assert "Nothing in sub-clause (a) of clause (1)" in rel.provenance_json["supporting_text"]
    exception = db.query(KnowledgeRelation).filter_by(
        key="COI:art:22:cl:1|LIMITED_BY|Exception:COI:art:22:cl:3").one()
    assert exception.verification_status == "verified"


def test_curated_quotes_are_verified_against_source(db):
    cache = GraphCache.get(db)
    annotated = [n for n in cache.nodes.values() if n["props"].get("addressee")]
    assert len(annotated) >= 20
    assert all(n["props"]["addressee_verification"] == "verified" for n in annotated), \
        [n["key"] for n in annotated if n["props"]["addressee_verification"] != "verified"]
    rights = [n for n in cache.nodes.values() if n["label"] == "UserRight"]
    assert rights and all(r["status"] == "verified" for r in rights)


def test_curated_routing_is_never_marked_as_verified_law(db):
    for rel in db.query(KnowledgeRelation).filter_by(relation_type="MAY_ENGAGE").all():
        assert rel.verification_status == "curated"
        assert rel.provenance_json["review_status"] == "needs_legal_review"


def test_statutes_are_unverified_without_official_pdf(db):
    from app.config import get_settings
    has_pdfs = any(get_settings().statutes_dir.glob("*.pdf")) if get_settings().statutes_dir.exists() else False
    sections = db.query(KnowledgeEntity).filter_by(entity_type="StatutorySection").all()
    assert sections
    if not has_pdfs:
        assert all(s.verification_status == "unverified" for s in sections)


def test_rebuild_is_idempotent(db):
    before = (db.query(KnowledgeEntity).count(), db.query(KnowledgeRelation).count())
    build = build_graph(db)
    summary = persist_graph(db, build)
    after = (db.query(KnowledgeEntity).count(), db.query(KnowledgeRelation).count())
    assert before == after
    assert summary["nodes_created"] == 0 and summary["relationships_created"] == 0
    assert build_graph(db).build_id == build.build_id


def test_invalid_relationships_are_rejected():
    g = GraphBuild()
    g.node("A", "Article", "Article 1")
    g.node("O", "Offence", "Theft")
    assert g.rel("A", "HAS_CLAUSE", "O", provenance={"source_document_id": "x"}) is None
    assert g.rejected and "not allowed" in g.rejected[0]["reason"]
    g.node("C", "Clause", "Article 1(1)")
    with pytest.raises(GraphValidationError):
        g.rel("A", "HAS_CLAUSE", "C", provenance={}, method="structural")


def test_duplicate_node_keys_are_prevented_by_constraint(db):
    db.add(KnowledgeEntity(key="COI:art:21", entity_type="Article", name="duplicate"))
    with pytest.raises(IntegrityError):
        db.commit()
    db.rollback()


def test_bounded_issue_traversal(db):
    queries = SqlGraphQueries(db)
    ctx = queries.issue_context(["arrest_safeguards"])
    engaged = {e["provision_id"] for e in ctx.engaged}
    assert {"COI:art:22:cl:1", "COI:art:22:cl:2", "COI:art:21"} <= engaged
    assert any(r["id"] == "writ_high_court" for r in ctx.remedies)
    ctx = queries.provision_context(["COI:art:22:cl:1"], ctx)
    assert any(e["provision_id"] == "COI:art:22:cl:3" for e in ctx.expansions)
    assert any(u["name"].startswith("Right to be informed") for u in ctx.user_rights)
    sub = queries.subgraph_for(["COI:art:21"])
    assert 0 < len(sub["nodes"]) <= 40


@pytest.mark.skipif(not os.environ.get("NEO4J_TEST_PASSWORD"), reason="set NEO4J_TEST_URI/NEO4J_TEST_PASSWORD")
def test_neo4j_projection_matches_relational_graph(db):
    from app.config import get_settings
    from app.graph import neo4j_store
    s = get_settings()
    s.neo4j_uri = os.environ.get("NEO4J_TEST_URI", "bolt://localhost:7687")
    s.neo4j_password, s.enable_neo4j = os.environ["NEO4J_TEST_PASSWORD"], True
    neo4j_store.close_driver()
    try:
        assert neo4j_store.neo4j_available(max_age=0)
        first = neo4j_store.sync_to_neo4j(db)
        second = neo4j_store.sync_to_neo4j(db)          # idempotent
        counts = neo4j_store.neo4j_counts()
        assert counts["total_nodes"] == db.query(KnowledgeEntity).count()
        assert counts["total_relationships"] == db.query(KnowledgeRelation).count()
        assert counts["orphans"] == 0
        assert second["removed_nodes"] == 0 and first["nodes"] == second["nodes"]
        q = neo4j_store.Neo4jGraphQueries()
        ctx = q.issue_context(["arrest_safeguards"])
        assert "COI:art:22:cl:1" in {e["provision_id"] for e in ctx.engaged}
        ctx = q.provision_context(["COI:art:19:cl:1:sub:a"], ctx)
        assert "COI:art:19:cl:2" in {e["provision_id"] for e in ctx.expansions}
    finally:
        s.enable_neo4j = False
        neo4j_store.close_driver()
