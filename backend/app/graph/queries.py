"""Bounded, predefined graph retrieval.

Two interchangeable back ends implement the same operations:

* ``SqlGraphQueries`` - traverses the in-memory adjacency index of the relational graph.
* ``Neo4jGraphQueries`` - runs fixed, parameterised Cypher (``app.graph.neo4j_store``).

No free-form or model-generated Cypher is ever executed. Every traversal has a fixed
depth and result limit.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from sqlalchemy.orm import Session

from app.graph.store import GraphCache

MAX_REFERENCES = 6


@dataclass
class GraphContext:
    engaged: list[dict] = field(default_factory=list)       # provisions engaged by issues (with role/condition)
    expansions: list[dict] = field(default_factory=list)    # restrictions/exceptions/clauses pulled in
    statutes: list[dict] = field(default_factory=list)      # statutory sections (offences / remedies)
    remedies: list[dict] = field(default_factory=list)
    concepts: list[dict] = field(default_factory=list)
    user_rights: list[dict] = field(default_factory=list)
    nodes: dict[str, dict] = field(default_factory=dict)    # for visualisation / debug
    edges: list[dict] = field(default_factory=list)
    backend: str = "sql"

    def add_edge(self, edge: dict, cache_nodes: dict) -> None:
        for key in (edge["src"], edge["dst"]):
            if key in cache_nodes and key not in self.nodes:
                n = cache_nodes[key]
                self.nodes[key] = {"key": key, "label": n["label"], "name": n["name"], "status": n["status"]}
        self.edges.append({"source": edge["src"], "target": edge["dst"], "relation": edge["type"],
                           "status": edge["status"], "method": edge.get("method"),
                           "supporting_text": (edge.get("provenance") or {}).get("supporting_text")})

    def summary(self) -> dict:
        return {"backend": self.backend, "engaged": [e["provision_id"] for e in self.engaged],
                "expansions": [e["provision_id"] for e in self.expansions],
                "statutes": [s["id"] for s in self.statutes], "remedies": [r["id"] for r in self.remedies],
                "nodes": len(self.nodes), "edges": len(self.edges)}


class SqlGraphQueries:
    backend = "sql"

    def __init__(self, db: Session):
        self.cache = GraphCache.get(db)

    # -------------------------------------------------------------- primitives
    def _out(self, key: str, *types: str) -> list[dict]:
        return [e for e in self.cache.out.get(key, []) if not types or e["type"] in types]

    def _in(self, key: str, *types: str) -> list[dict]:
        return [e for e in self.cache.inc.get(key, []) if not types or e["type"] in types]

    def node(self, key: str) -> dict | None:
        return self.cache.nodes.get(key)

    # -------------------------------------------------------------- operations
    def issue_context(self, issue_ids: list[str], ctx: GraphContext | None = None) -> GraphContext:
        ctx = ctx or GraphContext(backend=self.backend)
        seen_sections: set[str] = set()
        seen_remedies: set[str] = set()
        for issue_id in issue_ids[:6]:
            ikey = f"LegalIssue:{issue_id}"
            if ikey not in self.cache.nodes:
                continue
            for edge in self._out(ikey, "MAY_ENGAGE"):
                ctx.engaged.append({"provision_id": edge["dst"], "issue": issue_id, **edge["props"]})
                ctx.add_edge(edge, self.cache.nodes)
            for edge in self._out(ikey, "MAY_INVOLVE"):
                ctx.add_edge(edge, self.cache.nodes)
                for definer in self._in(edge["dst"], "DEFINES"):
                    if definer["src"].startswith("StatutorySection:") and definer["src"] not in seen_sections:
                        seen_sections.add(definer["src"])
                        ctx.statutes.append(self._section(definer["src"], role="offence", issue=issue_id))
                        ctx.add_edge(definer, self.cache.nodes)
            for edge in self._in(ikey, "AVAILABLE_FOR"):
                rkey = edge["src"]
                ctx.add_edge(edge, self.cache.nodes)
                if rkey in seen_remedies:
                    continue
                seen_remedies.add(rkey)
                remedy = self.cache.nodes[rkey]
                handlers = [self.cache.nodes[h["src"]]["name"] for h in self._in(rkey, "HANDLES")]
                basis = []
                for pb in self._out(rkey, "PROVIDED_BY"):
                    basis.append(pb["dst"])
                    if pb["dst"].startswith("StatutorySection:") and pb["dst"] not in seen_sections:
                        seen_sections.add(pb["dst"])
                        ctx.statutes.append(self._section(pb["dst"], role="remedy", issue=issue_id))
                ctx.remedies.append({"id": rkey.split(":", 1)[1], "name": remedy["name"], "authority": handlers,
                                     "basis": basis, "issue": issue_id})
        return ctx

    def _section(self, key: str, role: str, issue: str) -> dict:
        n = self.cache.nodes[key]
        statute_edge = next(iter(self._in(key, "CONTAINS")), None)
        statute = self.cache.nodes.get(statute_edge["src"]) if statute_edge else None
        return {"id": key.split(":", 1)[1], "key": key, "role": role, "issue": issue,
                "statute": (statute or {}).get("name"), "statute_props": (statute or {}).get("props", {}),
                "label": n["name"], "text": n["props"].get("text"), "status": n["status"],
                "verification_reason": n["props"].get("verification_reason")}

    def provision_context(self, provision_ids: list[str], ctx: GraphContext | None = None) -> GraphContext:
        """Restrictions, exceptions, parent article, defined concepts, rights and references."""
        ctx = ctx or GraphContext(backend=self.backend)
        seen = {e["provision_id"] for e in ctx.expansions}
        for pid in provision_ids[:20]:
            if pid not in self.cache.nodes:
                continue
            chain = [pid]
            parent = self._in(pid, "HAS_CLAUSE", "HAS_SUBCLAUSE")
            while parent and len(chain) < 3:
                chain.append(parent[0]["src"])
                parent = self._in(parent[0]["src"], "HAS_CLAUSE", "HAS_SUBCLAUSE")
            for key in chain:
                pulled = 0
                for edge in self._out(key, "SUBJECT_TO", "LIMITED_BY"):
                    # Article-level restrictions cover every sub-clause (e.g. 19(2)-(6)); when a specific
                    # sub-clause was asked about, only its own restriction is relevant.
                    if key != pid and edge["type"] == "SUBJECT_TO":
                        continue
                    if pulled >= 4:
                        break
                    pulled += 1
                    ctx.add_edge(edge, self.cache.nodes)
                    for stated in self._out(edge["dst"], "STATED_IN"):
                        target = stated["dst"]
                        # Only pull a restriction/exception that is specific to the provision asked about,
                        # or that limits the whole Article.
                        if target not in seen and target != pid:
                            seen.add(target)
                            ctx.expansions.append({"provision_id": target, "role":
                                                   "restriction" if edge["type"] == "SUBJECT_TO" else "exception",
                                                   "because_of": key, "supporting_text":
                                                   edge["provenance"].get("supporting_text")})
                            ctx.add_edge(stated, self.cache.nodes)
                for edge in self._out(key, "DEFINES"):
                    if edge["dst"].startswith("LegalConcept:"):
                        ctx.concepts.append({"key": edge["dst"], "name": self.cache.nodes[edge["dst"]]["name"],
                                             "defined_by": key})
                        ctx.add_edge(edge, self.cache.nodes)
            article = chain[-1]
            for edge in self._out(article, "PROTECTS"):
                ctx.add_edge(edge, self.cache.nodes)
            for edge in self._out(article, "REFERENCES")[:MAX_REFERENCES]:
                ctx.add_edge(edge, self.cache.nodes)
            for edge in self._in(pid, "GUARANTEED_BY"):
                ctx.user_rights.append({"key": edge["src"], "name": self.cache.nodes[edge["src"]]["name"],
                                        "provision_id": pid, "status": edge["status"]})
                ctx.add_edge(edge, self.cache.nodes)
            for edge in self._in(pid, "AMENDED")[:5]:
                ctx.add_edge(edge, self.cache.nodes)
        return ctx

    def subgraph_for(self, provision_ids: list[str], limit: int = 40) -> dict:
        """Small visualisation subgraph around the cited provisions."""
        ctx = GraphContext(backend=self.backend)
        for pid in provision_ids[:4]:
            if pid not in self.cache.nodes:
                continue
            ctx.nodes[pid] = {"key": pid, **{k: self.cache.nodes[pid][k] for k in ("label", "name", "status")}}
            for edge in self._in(pid, "CONTAINS", "HAS_CLAUSE", "HAS_SUBCLAUSE", "GUARANTEED_BY", "MAY_ENGAGE")[:6]:
                ctx.add_edge(edge, self.cache.nodes)
            for edge in self._out(pid, "PROTECTS", "SUBJECT_TO", "LIMITED_BY", "DEFINES", "HAS_CLAUSE",
                                  "REFERENCES")[:10]:
                ctx.add_edge(edge, self.cache.nodes)
        nodes = list(ctx.nodes.values())[:limit]
        keep = {n["key"] for n in nodes}
        edges = [e for e in ctx.edges if e["source"] in keep and e["target"] in keep]
        return {"nodes": nodes, "edges": edges}


def get_graph_queries(db: Session):
    """Neo4j when enabled and reachable, otherwise the relational graph."""
    from app.config import get_settings
    settings = get_settings()
    if settings.is_neo4j_configured:
        from app.graph.neo4j_store import Neo4jGraphQueries, neo4j_available
        if neo4j_available():
            return Neo4jGraphQueries()
    return SqlGraphQueries(db)
