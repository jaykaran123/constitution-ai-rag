"""Graph persistence.

The relational tables ``knowledge_entities`` / ``knowledge_relations`` are the system of
record (always available, versioned with the database, migrated and backed up). When
``ENABLE_NEO4J=true`` the same graph is projected into Neo4j with unique constraints and
queried there with predefined, parameterised Cypher.

Writes are idempotent: nodes and relationships are keyed (``key`` columns have unique
indexes) and upserted; managed items not present in the current build are removed.
"""
from __future__ import annotations

import logging
from collections import defaultdict

from sqlalchemy.orm import Session

from app.graph.builder import GraphBuild
from app.models import KnowledgeEntity, KnowledgeRelation
from app.observability import log_event

logger = logging.getLogger("app.graph.store")


def persist_graph(db: Session, build: GraphBuild) -> dict:
    build_id = build.build_id
    existing_nodes = {e.key: e for e in db.query(KnowledgeEntity).all() if e.key}
    # Rows written by the previous unkeyed implementation are derived data; replace them.
    legacy = db.query(KnowledgeEntity).filter(KnowledgeEntity.key.is_(None)).all()
    if legacy:
        legacy_ids = [e.id for e in legacy]
        db.query(KnowledgeRelation).filter(
            KnowledgeRelation.source_entity_id.in_(legacy_ids) | KnowledgeRelation.target_entity_id.in_(legacy_ids)
        ).delete(synchronize_session=False)
        db.query(KnowledgeEntity).filter(KnowledgeEntity.id.in_(legacy_ids)).delete(synchronize_session=False)
    created = updated = 0
    id_by_key: dict[str, str] = {}
    for key, node in build.nodes.items():
        row = existing_nodes.get(key)
        if row is None:
            row = KnowledgeEntity(key=key)
            db.add(row)
            created += 1
        else:
            updated += 1
        row.entity_type = node.label
        row.name = node.name[:500]
        row.description = (node.props.get("title") or node.props.get("description") or None)
        row.properties = node.props
        row.labels = node.extra_labels or None
        row.provenance = node.provenance
        row.verification_status = node.status
        row.source_document_id = (node.provenance or {}).get("source_document_id")
        row.build_id = build_id
    db.flush()
    for row in db.query(KnowledgeEntity.id, KnowledgeEntity.key).all():
        id_by_key[row.key] = row.id

    existing_rels = {r.key: r for r in db.query(KnowledgeRelation).all() if r.key}
    rel_created = 0
    for key, rel in build.rels.items():
        row = existing_rels.get(key)
        if row is None:
            row = KnowledgeRelation(key=key, source_entity_id=id_by_key[rel.src], target_entity_id=id_by_key[rel.dst],
                                    relation_type=rel.type)
            db.add(row)
            rel_created += 1
        row.properties = rel.props or None
        row.provenance_json = rel.provenance
        row.provenance = _provenance_text(rel.provenance)
        row.extraction_method = rel.method
        row.verification_status = rel.status
        row.confidence = rel.confidence
        row.build_id = build_id
    db.flush()
    stale_rels = db.query(KnowledgeRelation).filter(KnowledgeRelation.build_id != build_id).delete(
        synchronize_session=False)
    stale_nodes = db.query(KnowledgeEntity).filter(KnowledgeEntity.build_id != build_id).all()
    stale_ids = [n.id for n in stale_nodes]
    if stale_ids:
        db.query(KnowledgeRelation).filter(
            KnowledgeRelation.source_entity_id.in_(stale_ids) | KnowledgeRelation.target_entity_id.in_(stale_ids)
        ).delete(synchronize_session=False)
        db.query(KnowledgeEntity).filter(KnowledgeEntity.id.in_(stale_ids)).delete(synchronize_session=False)
    db.commit()
    GraphCache.invalidate()
    summary = {"build_id": build_id, "nodes": len(build.nodes), "relationships": len(build.rels),
               "nodes_created": created, "nodes_updated": updated, "relationships_created": rel_created,
               "stale_nodes_removed": len(stale_ids), "stale_relationships_removed": stale_rels,
               "rejected_relationships": len(build.rejected)}
    log_event(logger, "graph.persisted", **summary)
    return summary


def _provenance_text(prov: dict | None) -> str | None:
    if not prov:
        return None
    bits = []
    if prov.get("provision_id"):
        bits.append(prov["provision_id"])
    if prov.get("page"):
        bits.append(f"PDF page {prov['page']}")
    bits.append(prov.get("extraction_method") or "")
    return ", ".join(b for b in bits if b)[:500]


class GraphCache:
    """In-memory adjacency index over the relational graph for fast bounded traversal."""

    _instance: "GraphCache | None" = None

    def __init__(self, db: Session):
        self.nodes: dict[str, dict] = {}
        self.out: dict[str, list[dict]] = defaultdict(list)
        self.inc: dict[str, list[dict]] = defaultdict(list)
        id_to_key: dict[str, str] = {}
        for e in db.query(KnowledgeEntity).filter(KnowledgeEntity.key.is_not(None)).all():
            id_to_key[e.id] = e.key
            self.nodes[e.key] = {"key": e.key, "label": e.entity_type, "name": e.name, "props": e.properties or {},
                                 "labels": e.labels or [], "status": e.verification_status,
                                 "provenance": e.provenance}
        for r in db.query(KnowledgeRelation).filter(KnowledgeRelation.key.is_not(None)).all():
            src, dst = id_to_key.get(r.source_entity_id), id_to_key.get(r.target_entity_id)
            if not src or not dst:
                continue
            edge = {"src": src, "dst": dst, "type": r.relation_type, "props": r.properties or {},
                    "provenance": r.provenance_json or {}, "status": r.verification_status,
                    "method": r.extraction_method}
            self.out[src].append(edge)
            self.inc[dst].append(edge)
        self.signature = (len(self.nodes), sum(len(v) for v in self.out.values()))

    @classmethod
    def get(cls, db: Session) -> "GraphCache":
        if cls._instance is None:
            cls._instance = GraphCache(db)
        return cls._instance

    @classmethod
    def invalidate(cls) -> None:
        cls._instance = None
