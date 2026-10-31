"""Graph inspection and integrity checks (relational store; Neo4j counts when enabled)."""
from __future__ import annotations

from collections import Counter

from sqlalchemy.orm import Session

from app.graph.schema import REQUIRED_RELATIONSHIPS, RELATIONSHIP_RULES, VERIFICATION_STATUSES
from app.models import ConstitutionProvision, DocumentChunk, KnowledgeEntity, KnowledgeRelation


def inspect_graph(db: Session, sample: int = 10) -> dict:
    entities = db.query(KnowledgeEntity).filter(KnowledgeEntity.key.is_not(None)).all()
    relations = db.query(KnowledgeRelation).filter(KnowledgeRelation.key.is_not(None)).all()
    by_id = {e.id: e for e in entities}
    degree: Counter = Counter()
    incoming: dict[str, Counter] = {}
    outgoing: dict[str, Counter] = {}
    invalid_rels = []
    missing_provenance = []
    duplicate_keys = [k for k, c in Counter(e.key for e in entities).items() if c > 1]
    for r in relations:
        degree[r.source_entity_id] += 1
        degree[r.target_entity_id] += 1
        outgoing.setdefault(r.source_entity_id, Counter())[r.relation_type] += 1
        incoming.setdefault(r.target_entity_id, Counter())[r.relation_type] += 1
        src, dst = by_id.get(r.source_entity_id), by_id.get(r.target_entity_id)
        if not src or not dst or (src.entity_type, dst.entity_type) not in RELATIONSHIP_RULES.get(r.relation_type, set()):
            invalid_rels.append(r.key)
        prov = r.provenance_json or {}
        if r.extraction_method in {"structural", "text_pattern", "footnote"} and not prov.get("source_document_id"):
            missing_provenance.append(r.key)
        if r.verification_status not in VERIFICATION_STATUSES:
            invalid_rels.append(r.key)
    orphans = [e.key for e in entities if degree[e.id] == 0]
    missing_required: dict[str, list[str]] = {}
    for e in entities:
        for direction, rtype in REQUIRED_RELATIONSHIPS.get(e.entity_type, []):
            counts = (incoming if direction == "in" else outgoing).get(e.id, Counter())
            if counts[rtype] == 0:
                missing_required.setdefault(f"{e.entity_type} without {direction} {rtype}", []).append(e.key)
    # Articles without chunks / chunks without documents (relational check, independent of the graph)
    article_ids = {p.id for p in db.query(ConstitutionProvision.id).filter_by(kind="article")}
    chunked = {c.provision_id for c in db.query(DocumentChunk.provision_id).filter(DocumentChunk.provision_id.is_not(None))}
    chunk_docs = db.query(DocumentChunk).filter(DocumentChunk.document_id.is_(None)).count()
    statuses = Counter(r.verification_status for r in relations)
    methods = Counter(r.extraction_method for r in relations)
    report = {
        "total_nodes": len(entities),
        "total_relationships": len(relations),
        "nodes_by_type": dict(Counter(e.entity_type for e in entities).most_common()),
        "relationships_by_type": dict(Counter(r.relation_type for r in relations).most_common()),
        "relationships_by_verification": dict(statuses),
        "relationships_by_extraction_method": dict(methods),
        "orphan_nodes": {"count": len(orphans), "sample": orphans[:sample]},
        "articles_without_chunks": sorted(article_ids - chunked)[:sample],
        "articles_without_chunks_count": len(article_ids - chunked),
        "chunks_without_source_documents": chunk_docs,
        "missing_required_relationships": {k: {"count": len(v), "sample": v[:sample]}
                                           for k, v in missing_required.items()},
        "invalid_relationships": {"count": len(invalid_rels), "sample": invalid_rels[:sample]},
        "relationships_missing_provenance": {"count": len(missing_provenance), "sample": missing_provenance[:sample]},
        "duplicate_node_keys": duplicate_keys[:sample],
        "build_ids": sorted({e.build_id for e in entities if e.build_id}),
    }
    report["healthy"] = (not invalid_rels and not missing_provenance and not duplicate_keys
                         and report["articles_without_chunks_count"] == 0 and len(report["build_ids"]) <= 1)
    return report
