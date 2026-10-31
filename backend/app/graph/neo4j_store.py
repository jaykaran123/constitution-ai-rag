"""Neo4j projection and parameterised queries.

* ``ensure_schema`` creates a uniqueness constraint on ``key`` for every label plus the
  generic ``:KGNode`` label, and full-text/lookup indexes used by retrieval.
* ``sync_to_neo4j`` MERGEs every node and relationship from the relational graph in
  batches (idempotent) and then deletes managed items whose ``build_id`` is stale.
* ``Neo4jGraphQueries`` implements the same operations as ``SqlGraphQueries`` with fixed
  Cypher statements and bounded patterns. No dynamic Cypher is built from user input:
  labels and relationship types come from the closed schema, values are parameters.
"""
from __future__ import annotations

import logging
import time


from sqlalchemy.orm import Session

from app.config import get_settings
from app.graph.queries import GraphContext
from app.graph.schema import ALL_LABELS, RELATIONSHIP_RULES
from app.models import KnowledgeEntity, KnowledgeRelation
from app.observability import log_event

logger = logging.getLogger("app.graph.neo4j")
_driver = None
_available_cache: tuple[float, bool] | None = None


def get_driver():
    global _driver
    if _driver is None:
        from neo4j import GraphDatabase
        s = get_settings()
        _driver = GraphDatabase.driver(s.neo4j_uri, auth=(s.neo4j_user, s.neo4j_password),
                                       connection_timeout=5, max_connection_lifetime=600)
    return _driver


def close_driver() -> None:
    global _driver
    if _driver is not None:
        _driver.close()
        _driver = None


def neo4j_available(max_age: float = 30.0) -> bool:
    global _available_cache
    now = time.time()
    if _available_cache and now - _available_cache[0] < max_age:
        return _available_cache[1]
    try:
        get_driver().verify_connectivity()
        ok = True
    except Exception as exc:  # noqa: BLE001
        log_event(logger, "neo4j.unavailable", logging.WARNING, error=type(exc).__name__)
        ok = False
    _available_cache = (now, ok)
    return ok


def _run(query: str, **params):
    with get_driver().session(database=get_settings().neo4j_database) as session:
        return list(session.run(query, **params))


def ensure_schema() -> None:
    _run("CREATE CONSTRAINT kgnode_key IF NOT EXISTS FOR (n:KGNode) REQUIRE n.key IS UNIQUE")
    for label in ALL_LABELS:
        _run(f"CREATE CONSTRAINT {label.lower()}_key IF NOT EXISTS FOR (n:{label}) REQUIRE n.key IS UNIQUE")
    _run("CREATE INDEX article_number IF NOT EXISTS FOR (n:Article) ON (n.article_number)")
    _run("CREATE INDEX kg_status IF NOT EXISTS FOR (n:KGNode) ON (n.status)")
    _run("CREATE FULLTEXT INDEX provision_text IF NOT EXISTS FOR (n:Article|Clause|SubClause) ON EACH [n.text, n.name]")


def _flatten(props: dict) -> dict:
    """Neo4j properties must be primitives or arrays of primitives."""
    import json
    out = {}
    for key, value in (props or {}).items():
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            out[key] = value
        elif isinstance(value, list) and all(isinstance(v, (str, int, float, bool)) for v in value):
            out[key] = value
        else:
            out[key] = json.dumps(value, ensure_ascii=False, default=str)
    return out


def sync_to_neo4j(db: Session, batch: int = 500) -> dict:
    ensure_schema()
    started = time.time()
    entities = db.query(KnowledgeEntity).filter(KnowledgeEntity.key.is_not(None)).all()
    if not entities:
        return {"nodes": 0, "relationships": 0}
    build_id = entities[0].build_id
    id_to_key = {e.id: e.key for e in entities}
    by_label: dict[str, list[dict]] = {}
    for e in entities:
        if e.entity_type not in ALL_LABELS:
            continue
        props = _flatten(e.properties or {})
        props.update(_flatten({f"prov_{k}": v for k, v in (e.provenance or {}).items()}))
        props.update({"key": e.key, "name": e.name, "status": e.verification_status, "build_id": e.build_id})
        by_label.setdefault(e.entity_type, []).append({"key": e.key, "props": props,
                                                         "extra": [l for l in (e.labels or []) if l in ALL_LABELS]})
    node_count = 0
    for label, rows in by_label.items():
        for start in range(0, len(rows), batch):
            chunk = rows[start:start + batch]
            # label is from the closed schema list, never from user input
            _run(f"UNWIND $rows AS row MERGE (n:KGNode:{label} {{key: row.key}}) SET n = row.props", rows=chunk)
            extra_rows = [r for r in chunk if r["extra"]]
            for extra_label in {l for r in extra_rows for l in r["extra"]}:
                _run(f"UNWIND $keys AS k MATCH (n:KGNode {{key: k}}) SET n:{extra_label}",
                     keys=[r["key"] for r in extra_rows if extra_label in r["extra"]])
            node_count += len(chunk)
    rels_by_type: dict[str, list[dict]] = {}
    for r in db.query(KnowledgeRelation).filter(KnowledgeRelation.key.is_not(None)).all():
        if r.relation_type not in RELATIONSHIP_RULES:
            continue
        props = _flatten(r.properties or {})
        props.update(_flatten({f"prov_{k}": v for k, v in (r.provenance_json or {}).items()}))
        props.update({"key": r.key, "status": r.verification_status, "method": r.extraction_method or "",
                      "build_id": r.build_id})
        rels_by_type.setdefault(r.relation_type, []).append(
            {"src": id_to_key.get(r.source_entity_id), "dst": id_to_key.get(r.target_entity_id), "props": props})
    rel_count = 0
    for rtype, rows in rels_by_type.items():
        for start in range(0, len(rows), batch):
            chunk = rows[start:start + batch]
            _run(f"UNWIND $rows AS row MATCH (a:KGNode {{key: row.src}}) MATCH (b:KGNode {{key: row.dst}}) "
                 f"MERGE (a)-[r:{rtype} {{key: row.props.key}}]->(b) SET r = row.props", rows=chunk)
            rel_count += len(chunk)
    removed_rels = _run("MATCH ()-[r]->() WHERE r.build_id IS NOT NULL AND r.build_id <> $b DELETE r RETURN count(r) AS c",
                        b=build_id)[0]["c"]
    removed_nodes = _run("MATCH (n:KGNode) WHERE n.build_id <> $b DETACH DELETE n RETURN count(n) AS c",
                         b=build_id)[0]["c"]
    result = {"nodes": node_count, "relationships": rel_count, "removed_nodes": removed_nodes,
              "removed_relationships": removed_rels, "build_id": build_id,
              "seconds": round(time.time() - started, 1)}
    log_event(logger, "neo4j.synced", **result)
    return result


def neo4j_counts() -> dict:
    nodes = {r["label"]: r["c"] for r in _run(
        "MATCH (n:KGNode) UNWIND [l IN labels(n) WHERE l <> 'KGNode'] AS label RETURN label, count(*) AS c")}
    rels = {r["t"]: r["c"] for r in _run("MATCH ()-[r]->() RETURN type(r) AS t, count(*) AS c")}
    orphans = _run("MATCH (n:KGNode) WHERE NOT (n)--() RETURN count(n) AS c")[0]["c"]
    return {"nodes_by_label": nodes, "relationships_by_type": rels, "total_nodes": sum(
        _run("MATCH (n:KGNode) RETURN count(n) AS c")[0].values()), "total_relationships": sum(rels.values()),
        "orphans": orphans}


# ------------------------------------------------------------------ predefined queries
Q_ISSUE_ENGAGES = """
MATCH (i:LegalIssue {key: $issue})-[r:MAY_ENGAGE]->(p)
RETURN p.key AS provision_id, r.role AS role, r.condition AS condition, r.rationale AS rationale,
       r.status AS status, r.method AS method, p.name AS name, labels(p) AS labels
LIMIT 20
"""
Q_ISSUE_OFFENCE_SECTIONS = """
MATCH (i:LegalIssue {key: $issue})-[:MAY_INVOLVE]->(o:Offence)<-[:DEFINES]-(s:StatutorySection)<-[:CONTAINS]-(st:Statute)
RETURN s.key AS key, s.name AS label, s.text AS text, s.status AS status, s.verification_reason AS reason,
       st.name AS statute, st.source_url AS url, st.in_force_from AS in_force_from, o.name AS offence
LIMIT 10
"""
Q_ISSUE_REMEDIES = """
MATCH (rem:LegalRemedy)-[:AVAILABLE_FOR]->(i:LegalIssue {key: $issue})
OPTIONAL MATCH (h)-[:HANDLES]->(rem)
OPTIONAL MATCH (rem)-[:PROVIDED_BY]->(b)
OPTIONAL MATCH (b)<-[:CONTAINS]-(st:Statute)
RETURN rem.key AS key, rem.name AS name, collect(DISTINCT h.name) AS authority,
       collect(DISTINCT {key: b.key, label: b.name, text: b.text, status: b.status, statute: st.name,
                         url: st.source_url, in_force_from: st.in_force_from, reason: b.verification_reason}) AS basis
LIMIT 10
"""
Q_PROVISION_EXPANSION = """
MATCH (p:KGNode {key: $pid})
OPTIONAL MATCH path = (p)<-[:HAS_CLAUSE|HAS_SUBCLAUSE*0..2]-(anc)
WITH p, collect(DISTINCT anc) AS chain
UNWIND chain AS a
MATCH (a)-[rel:SUBJECT_TO|LIMITED_BY]->(x)-[:STATED_IN]->(c)
WHERE (a = p OR type(rel) = 'LIMITED_BY') AND c.key <> $pid
RETURN DISTINCT c.key AS provision_id, type(rel) AS rel, a.key AS because_of, rel.prov_supporting_text AS supporting_text
LIMIT 6
"""
Q_PROVISION_RIGHTS = """
MATCH (u:UserRight)-[g:GUARANTEED_BY]->(p:KGNode {key: $pid})
RETURN u.key AS key, u.name AS name, g.status AS status LIMIT 6
"""
Q_PROVISION_CONCEPTS = """
MATCH (p:KGNode {key: $pid})-[:DEFINES]->(c:LegalConcept) RETURN c.key AS key, c.name AS name LIMIT 4
"""
Q_SUBGRAPH = """
MATCH (p:KGNode {key: $pid})
OPTIONAL MATCH (p)-[r1:PROTECTS|SUBJECT_TO|LIMITED_BY|DEFINES|HAS_CLAUSE|REFERENCES]->(o)
WITH p, collect({src: p.key, dst: o.key, type: type(r1), status: r1.status, method: r1.method,
                 support: r1.prov_supporting_text, dname: o.name, dlabel: [l IN labels(o) WHERE l <> 'KGNode'][0],
                 dstatus: o.status})[..10] AS outs
OPTIONAL MATCH (i)-[r2:CONTAINS|HAS_CLAUSE|HAS_SUBCLAUSE|GUARANTEED_BY]->(p)
RETURN p.key AS key, p.name AS name, [l IN labels(p) WHERE l <> 'KGNode'][0] AS label, p.status AS status, outs,
       collect({src: i.key, dst: p.key, type: type(r2), status: r2.status, method: r2.method,
                support: r2.prov_supporting_text, sname: i.name,
                slabel: [l IN labels(i) WHERE l <> 'KGNode'][0], sstatus: i.status})[..6] AS ins
"""


class Neo4jGraphQueries:
    backend = "neo4j"

    def issue_context(self, issue_ids: list[str], ctx: GraphContext | None = None) -> GraphContext:
        ctx = ctx or GraphContext(backend=self.backend)
        seen_sections: set[str] = set()
        for issue_id in issue_ids[:6]:
            ikey = f"LegalIssue:{issue_id}"
            for row in _run(Q_ISSUE_ENGAGES, issue=ikey):
                ctx.engaged.append({"provision_id": row["provision_id"], "issue": issue_id, "role": row["role"],
                                    "condition": row["condition"], "rationale": row["rationale"]})
                ctx.edges.append({"source": ikey, "target": row["provision_id"], "relation": "MAY_ENGAGE",
                                  "status": row["status"], "method": row["method"], "supporting_text": None})
            for row in _run(Q_ISSUE_OFFENCE_SECTIONS, issue=ikey):
                if row["key"] in seen_sections:
                    continue
                seen_sections.add(row["key"])
                ctx.statutes.append({"id": row["key"].split(":", 1)[1], "key": row["key"], "role": "offence",
                                     "issue": issue_id, "statute": row["statute"], "label": row["label"],
                                     "text": row["text"], "status": row["status"],
                                     "verification_reason": row["reason"],
                                     "statute_props": {"source_url": row["url"], "in_force_from": row["in_force_from"]}})
            for row in _run(Q_ISSUE_REMEDIES, issue=ikey):
                basis = [b for b in row["basis"] if b.get("key")]
                for b in basis:
                    if b["key"].startswith("StatutorySection:") and b["key"] not in seen_sections:
                        seen_sections.add(b["key"])
                        ctx.statutes.append({"id": b["key"].split(":", 1)[1], "key": b["key"], "role": "remedy",
                                             "issue": issue_id, "statute": b["statute"], "label": b["label"],
                                             "text": b["text"], "status": b["status"],
                                             "verification_reason": b["reason"],
                                             "statute_props": {"source_url": b["url"],
                                                               "in_force_from": b["in_force_from"]}})
                if not any(r["id"] == row["key"].split(":", 1)[1] for r in ctx.remedies):
                    ctx.remedies.append({"id": row["key"].split(":", 1)[1], "name": row["name"],
                                         "authority": row["authority"], "basis": [b["key"] for b in basis],
                                         "issue": issue_id})
        return ctx

    def provision_context(self, provision_ids: list[str], ctx: GraphContext | None = None) -> GraphContext:
        ctx = ctx or GraphContext(backend=self.backend)
        seen = {e["provision_id"] for e in ctx.expansions}
        for pid in provision_ids[:20]:
            for row in _run(Q_PROVISION_EXPANSION, pid=pid):
                if row["provision_id"] in seen:
                    continue
                seen.add(row["provision_id"])
                ctx.expansions.append({"provision_id": row["provision_id"],
                                       "role": "restriction" if row["rel"] == "SUBJECT_TO" else "exception",
                                       "because_of": row["because_of"], "supporting_text": row["supporting_text"]})
            for row in _run(Q_PROVISION_RIGHTS, pid=pid):
                ctx.user_rights.append({"key": row["key"], "name": row["name"], "provision_id": pid,
                                        "status": row["status"]})
            for row in _run(Q_PROVISION_CONCEPTS, pid=pid):
                ctx.concepts.append({"key": row["key"], "name": row["name"], "defined_by": pid})
        return ctx

    def subgraph_for(self, provision_ids: list[str], limit: int = 40) -> dict:
        nodes: dict[str, dict] = {}
        edges: list[dict] = []
        for pid in provision_ids[:4]:
            for row in _run(Q_SUBGRAPH, pid=pid):
                nodes[row["key"]] = {"key": row["key"], "label": row["label"], "name": row["name"], "status": row["status"]}
                for e in row["outs"]:
                    if e["dst"]:
                        nodes.setdefault(e["dst"], {"key": e["dst"], "label": e["dlabel"], "name": e["dname"],
                                                    "status": e["dstatus"]})
                        edges.append({"source": e["src"], "target": e["dst"], "relation": e["type"],
                                      "status": e["status"], "method": e["method"], "supporting_text": e["support"]})
                for e in row["ins"]:
                    if e["src"]:
                        nodes.setdefault(e["src"], {"key": e["src"], "label": e["slabel"], "name": e["sname"],
                                                    "status": e["sstatus"]})
                        edges.append({"source": e["src"], "target": e["dst"], "relation": e["type"],
                                      "status": e["status"], "method": e["method"], "supporting_text": e["support"]})
        keep = list(nodes)[:limit]
        return {"nodes": [nodes[k] for k in keep],
                "edges": [e for e in edges if e["source"] in keep and e["target"] in keep]}



