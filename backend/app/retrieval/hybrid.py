"""Hybrid retrieval: exact lookup + knowledge graph + BM25 + vectors, fused by weighted RRF.

Methods and their default weights (configurable in settings):

    exact Article/clause lookup   RETRIEVAL_WEIGHT_EXACT   3.0   (user named the provision)
    knowledge-graph routing       RETRIEVAL_WEIGHT_GRAPH   1.6   (issue -> provisions, restrictions)
    BM25 keyword search           RETRIEVAL_WEIGHT_BM25    1.0
    semantic vector search        RETRIEVAL_WEIGHT_VECTOR  1.0

Results are fused at the *provision* level (several chunks of one clause count once),
deduplicated along the hierarchy (a clause already inside a selected Article is not
repeated), then expanded with restrictions/exceptions that the graph links to the selected
provisions (e.g. Article 19(1)(a) pulls in Article 19(2)). A rule-based reranker demotes
Schedules/Appendices unless the question is about them and drops omitted provisions unless
the question is about amendments or names them.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from sqlalchemy.orm import Session

from app.config import get_settings
from app.graph.queries import GraphContext, get_graph_queries
from app.knowledge.constitution_parser import clean_text
from app.models import ConstitutionProvision, DocumentChunk, LegalDocument
from app.retrieval.indexes import BM25Index, VectorIndex
from app.retrieval.query_understanding import QueryAnalysis

logger = logging.getLogger("app.retrieval.hybrid")

ROLE_PRIORITY = {"explicit": 0, "primary": 1, "user_premise": 1, "restriction": 2, "exception": 2,
                 "conditional": 3, "context": 4, "distinguish": 4, "remedy": 5, "historical": 6, "retrieved": 7}


@dataclass
class Evidence:
    provision_id: str
    kind: str                                 # constitution | statute | upload
    label: str
    text: str
    title: str | None = None
    page: int | None = None
    printed_page: str | None = None
    part: str | None = None
    status: str = "in_force"
    amendments: list[dict] = field(default_factory=list)
    source_title: str = "Constitution of India"
    source_url: str | None = None
    version: str | None = None
    verification: str = "verified"
    roles: list[str] = field(default_factory=list)
    scores: dict[str, float] = field(default_factory=dict)
    chunk_ids: list[str] = field(default_factory=list)
    rationale: list[str] = field(default_factory=list)
    conditions: list[str] = field(default_factory=list)
    addressee: str | None = None
    applicability_note: str | None = None
    scope_terms: list[str] | None = None
    evidence_id: str = ""

    @property
    def best_role(self) -> str:
        return min(self.roles, key=lambda r: ROLE_PRIORITY.get(r, 9)) if self.roles else "retrieved"

    def public(self) -> dict:
        return {"evidence_id": self.evidence_id, "provision_id": self.provision_id, "kind": self.kind,
                "label": self.label, "title": self.title, "text": self.text, "page": self.page,
                "printed_page": self.printed_page, "status": self.status, "source_title": self.source_title,
                "source_url": self.source_url, "version": self.version, "verification": self.verification,
                "roles": self.roles, "amendments": [
                    {k: a.get(k) for k in ("amendment_act", "action", "effective_from")} for a in self.amendments[:4]]}


@dataclass
class RetrievalResult:
    constitution: list[Evidence]
    statutes: list[Evidence]
    uploads: list[Evidence]
    remedies: list[dict]
    graph: GraphContext
    diagnostics: dict

    @property
    def all_evidence(self) -> list[Evidence]:
        return self.constitution + self.statutes + self.uploads


class ProvisionStore:
    _cache: tuple[str, dict[str, ConstitutionProvision], dict[str, list[str]]] | None = None

    @classmethod
    def load(cls, db: Session) -> tuple[dict[str, ConstitutionProvision], dict[str, list[str]]]:
        doc = db.query(LegalDocument).filter_by(is_default_source=True).first()
        signature = f"{doc.id if doc else ''}:{doc.content_hash if doc else ''}:{doc.parser_version if doc else ''}"
        if cls._cache is None or cls._cache[0] != signature:
            rows = db.query(ConstitutionProvision).all()
            for row in rows:
                db.expunge(row)
            by_id = {r.id: r for r in rows}
            children: dict[str, list[str]] = {}
            for r in sorted(rows, key=lambda r: (r.ordinal, r.id)):
                if r.parent_id:
                    children.setdefault(r.parent_id, []).append(r.id)
            cls._cache = (signature, by_id, children)
        return cls._cache[1], cls._cache[2]

    @classmethod
    def invalidate(cls) -> None:
        cls._cache = None


def _display_text(p: ConstitutionProvision, by_id: dict, limit: int = 3200) -> str:
    text = clean_text(p.text)
    if p.kind == "subclause" and p.parent_id in by_id:
        parent = by_id[p.parent_id]
        parent_text = clean_text(parent.text)
        lead = re.split(r"\n\s*\[?\(a\)", parent_text, maxsplit=1)[0] if parent.kind == "clause" else \
            re.split(r"\n\s*\[?\(a\)", parent_text, maxsplit=1)[0]
        if parent.kind == "article":
            lead = lead.split("\n", 1)[-1] if "\n" in lead else lead
        # Sub-clause (a) follows its lead-in directly in the source; later ones are separated by an ellipsis.
        text = f"{lead.strip()} {text}" if p.subclause == "a" else f"{lead.strip()} … {text}"
    text = re.sub(r"[ \t]*\n[ \t]*", " ", text)
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0] + " …"


def _evidence_from_provision(p: ConstitutionProvision, by_id: dict, doc: LegalDocument, graph_node: dict | None) -> Evidence:
    article = by_id.get(f"COI:art:{p.article_number}") if p.article_number else None
    props = (graph_node or {}).get("props", {})
    return Evidence(
        provision_id=p.id, kind="constitution", label=p.label, text=_display_text(p, by_id),
        title=(article.title if article else p.title), page=p.page_start, printed_page=p.printed_page,
        part=p.part_number, status=p.status or "in_force", amendments=p.amendments or [],
        source_title=doc.title, source_url=doc.source_url, version=doc.version,
        verification="verified" if doc.verification_status == "verified" else "unverified",
        addressee=props.get("addressee"), applicability_note=props.get("applicability_note"),
        scope_terms=props.get("scope_terms"),
    )


def _rrf(rank: int, weight: float, k: int) -> float:
    return weight / (k + rank)


class HybridRetriever:
    def __init__(self, db: Session, embedder=None):
        self.db = db
        self.embedder = embedder
        self.settings = get_settings()

    async def retrieve(self, analysis: QueryAnalysis, query_text: str, session_id: str | None,
                       top_k: int | None = None) -> RetrievalResult:
        s = self.settings
        top_k = top_k or s.retrieval_top_k
        by_id, children = ProvisionStore.load(self.db)
        doc = self.db.query(LegalDocument).filter_by(is_default_source=True).first()
        diagnostics: dict = {"strategy": [], "exact": [], "bm25": [], "vector": [], "graph": {}, "fused": []}
        if not doc or not by_id:
            return RetrievalResult([], [], [], [], GraphContext(), {**diagnostics, "error": "no_default_source"})
        queries = get_graph_queries(self.db)
        graph_nodes = getattr(getattr(queries, "cache", None), "nodes", {})
        candidates: dict[str, dict] = {}

        def add(pid: str, method: str, rank: int, role: str | None = None, rationale: str | None = None,
                condition: str | None = None, score: float | None = None) -> None:
            if pid not in by_id:
                return
            entry = candidates.setdefault(pid, {"fused": 0.0, "roles": set(), "scores": {}, "rationale": [],
                                                "conditions": [], "chunks": set()})
            weight = {"exact": s.retrieval_weight_exact, "graph": s.retrieval_weight_graph,
                      "bm25": s.retrieval_weight_bm25, "vector": s.retrieval_weight_vector,
                      "expansion": s.retrieval_weight_graph}[method]
            if method not in entry["scores"]:
                entry["fused"] += _rrf(rank, weight, s.retrieval_rrf_k)
                entry["scores"][method] = round(score, 4) if score is not None else rank
            if role:
                entry["roles"].add(role)
            if rationale and rationale not in entry["rationale"]:
                entry["rationale"].append(rationale)
            if condition and condition not in entry["conditions"]:
                entry["conditions"].append(condition)

        # A. exact lookup -------------------------------------------------------------
        for rank, ref in enumerate(analysis.explicit_refs, start=1):
            add(ref.provision_id, "exact", rank, role="explicit")
            diagnostics["exact"].append(ref.provision_id)
        if analysis.explicit_refs:
            diagnostics["strategy"].append("exact_lookup")

        # B. knowledge graph routing ------------------------------------------------------
        graph = GraphContext(backend=getattr(queries, "backend", "sql"))
        if analysis.issues:
            graph = queries.issue_context(analysis.issues, graph)
            diagnostics["strategy"].append(f"graph_{graph.backend}")
            for rank, engaged in enumerate(graph.engaged, start=1):
                add(engaged["provision_id"], "graph", rank, role=engaged.get("role") or "primary",
                    rationale=engaged.get("rationale"), condition=engaged.get("condition"))
            # Incidents involving a private or unclear actor turn on who is "the State": Article 12 must be
            # citable whenever the answer explains that distinction.
            if analysis.is_incident and analysis.actor_type in {"private", "mixed", "unknown"}:
                add("COI:art:12", "graph", len(graph.engaged) + 1, role="context",
                    rationale="Defines 'the State' for Part III (State vs private actor)")
            # The constitutional basis of each remedy (e.g. Article 226(1) for a writ) must be in the evidence,
            # otherwise the answer cannot cite it.
            basis_added = 0
            for remedy in sorted(graph.remedies, key=lambda r: r["id"]):      # deterministic across backends
                for basis in sorted(remedy.get("basis", [])):
                    if basis.startswith("COI:") and basis_added < 4 and basis not in candidates:
                        basis_added += 1
                        add(basis, "graph", len(graph.engaged) + basis_added, role="remedy",
                            rationale=f"Legal basis of the remedy: {remedy['name']}")

        # C. BM25 ----------------------------------------------------------------------
        doc_ids = [doc.id]
        upload_ids: list[str] = []
        if session_id:
            upload_ids = [d.id for d in self.db.query(LegalDocument).filter(
                LegalDocument.session_id == session_id, LegalDocument.ingestion_status == "completed").all()]
        bm25 = BM25Index.for_documents(self.db, doc_ids)
        bm25_hits = bm25.search(query_text, top_k=40)
        diagnostics["strategy"].append("bm25")
        rank = 0
        seen_bm25: set[str] = set()
        for meta, score in bm25_hits:
            if not meta.provision_id or meta.provision_id in seen_bm25:
                continue
            seen_bm25.add(meta.provision_id)
            rank += 1
            add(meta.provision_id, "bm25", rank, role="retrieved", score=score)
            candidates.get(meta.provision_id, {}).get("chunks", set()).add(meta.chunk_id)
            diagnostics["bm25"].append({"provision_id": meta.provision_id, "score": round(score, 3)})

        # D. vector ------------------------------------------------------------------
        query_vector = None
        if self.embedder is not None:
            try:
                vindex = VectorIndex.for_documents(self.db, doc_ids + upload_ids, self.embedder.model,
                                                   self.embedder.dimensions)
                if vindex.size:
                    query_vector = await self.embedder.embed_query(query_text)
                    rank = 0
                    seen_vec: set[str] = set()
                    for meta, sim in vindex.search(query_vector, top_k=40):
                        if meta.document_id != doc.id or not meta.provision_id or meta.provision_id in seen_vec:
                            continue
                        seen_vec.add(meta.provision_id)
                        rank += 1
                        add(meta.provision_id, "vector", rank, role="retrieved", score=sim)
                        candidates.get(meta.provision_id, {}).get("chunks", set()).add(meta.chunk_id)
                        diagnostics["vector"].append({"provision_id": meta.provision_id, "similarity": round(sim, 4)})
                    diagnostics["strategy"].append("vector")
                    diagnostics["vector_index_size"] = vindex.size
                else:
                    diagnostics["vector_status"] = "no embeddings for configured model"
            except Exception as exc:  # noqa: BLE001 - vector search is best effort
                diagnostics["vector_status"] = f"error: {getattr(exc, 'kind', type(exc).__name__)}"
        else:
            diagnostics["vector_status"] = "embedding provider not configured"

        # E. rerank / select -----------------------------------------------------------
        about_schedules = bool(re.search(r"\b(schedule|list [i]{1,3}|entry|union list|state list|concurrent list)\b",
                                         query_text, re.I))
        about_amendments = bool(re.search(r"\b(amend\w*|omitted|repeal\w*|substituted|inserted|history|original)\b",
                                          query_text, re.I))
        ranked: list[tuple[str, dict]] = []
        for pid, entry in candidates.items():
            p = by_id[pid]
            if p.kind in {"schedule_entry", "schedule", "appendix"} and not about_schedules and "explicit" not in entry["roles"]:
                entry["fused"] *= 0.35
            if p.status == "omitted" and not about_amendments and "explicit" not in entry["roles"] \
                    and "historical" not in entry["roles"]:
                continue
            if p.kind in {"part", "chapter"}:
                continue
            ranked.append((pid, entry))
        ranked.sort(key=lambda item: item[1]["fused"], reverse=True)
        diagnostics["fused"] = [{"provision_id": pid, "score": round(e["fused"], 4), "roles": sorted(e["roles"])}
                                for pid, e in ranked[:20]]

        guaranteed = [pid for pid, e in ranked if e["roles"] & {"explicit", "primary", "user_premise", "restriction",
                                                                 "exception", "conditional", "context", "distinguish",
                                                                 "historical", "remedy"}]
        # Retrieval-only items need a minimum fused score: agreement of two methods, or a top-3 rank in one.
        # When exact lookup or graph routing already found the relevant provisions, a search-only item must
        # be confirmed by two methods (keyword *and* semantic); a lone keyword hit is too often noise.
        min_score = _rrf(3, min(s.retrieval_weight_bm25, s.retrieval_weight_vector), s.retrieval_rrf_k)
        retrieved = [pid for pid, e in ranked if pid not in guaranteed and e["fused"] >= min_score
                     and (not guaranteed or len({"bm25", "vector"} & e["scores"].keys()) == 2)]
        if not guaranteed and not analysis.explicit_refs:
            # With no explicit Article or issue-graph route, lexical rank alone
            # is too weak: broad words such as "property" and "complaint" were
            # surfacing unrelated provisions. Require independent BM25 and
            # vector agreement before using retrieval-only evidence.
            retrieved = [pid for pid, e in ranked if e["fused"] >= min_score
                         and {"bm25", "vector"} <= e["scores"].keys()]
        selected = guaranteed[:top_k] + retrieved[: max(0, min(4, top_k - len(guaranteed)))]

        # Hierarchy dedup: drop a sub-clause when its clause is selected (the clause text contains it),
        # and drop a long Article overview when one of its clauses is selected (unless asked for by name).
        selected_set = set(selected)
        final: list[str] = []
        for pid in selected:
            p = by_id[pid]
            entry = candidates[pid]
            if p.kind == "subclause" and p.parent_id in selected_set and "explicit" not in entry["roles"]:
                continue
            if p.kind == "article" and not entry["roles"] & {"explicit", "primary", "user_premise"} \
                    and len(p.text) > 1800 and any(c in selected_set for c in children.get(pid, [])):
                continue
            final.append(pid)

        # F. context expansion (restrictions / exceptions) --------------------------------
        graph = queries.provision_context(final, graph)
        for exp in graph.expansions:
            pid = exp["provision_id"]
            if pid not in candidates:
                add(pid, "expansion", 1, role=exp["role"], rationale=f"{exp['role'].title()} linked to "
                    f"{by_id[exp['because_of']].label if exp['because_of'] in by_id else exp['because_of']} in the text")
            else:
                candidates[pid]["roles"].add(exp["role"])
            if pid not in final and pid in by_id:
                final.append(pid)
        diagnostics["graph"] = graph.summary()

        # Exceptions that only empower the State ("Nothing in this article shall prevent the State ...")
        # are irrelevant when the conduct is by a private actor; keep restrictions and other exceptions.
        if analysis.actor_type == "private":
            dropped = []
            for pid in list(final):
                roles = candidates[pid]["roles"] - {"retrieved"}
                if roles and roles <= {"exception"} and re.search(r"prevent\s+the\s+State", by_id[pid].text):
                    final.remove(pid)
                    dropped.append(pid)
            if dropped:
                diagnostics["dropped_state_only_exceptions"] = dropped

        chunk_map: dict[str, set[str]] = {}
        for chunk_id, provision_id in self.db.query(DocumentChunk.id, DocumentChunk.provision_id).filter(
                DocumentChunk.document_id == doc.id, DocumentChunk.provision_id.in_(final)).all():
            chunk_map.setdefault(provision_id, set()).add(chunk_id)
        constitution: list[Evidence] = []
        for pid in final[: top_k + 4]:
            entry = candidates[pid]
            entry["chunks"] |= chunk_map.get(pid, set())
            ev = _evidence_from_provision(by_id[pid], by_id, doc, graph_nodes.get(pid))
            ev.roles = sorted(entry["roles"], key=lambda r: ROLE_PRIORITY.get(r, 9))
            ev.scores = {**entry["scores"], "fused": round(entry["fused"], 4)}
            ev.rationale = entry["rationale"]
            ev.conditions = entry["conditions"]
            ev.chunk_ids = sorted(entry["chunks"])
            if not ev.addressee and pid not in graph_nodes:
                ev.addressee = None
            # Sub-clause/clause inherits the Article's applicability annotation when it has none.
            if not ev.addressee:
                for ancestor in (by_id[pid].parent_id, f"COI:art:{by_id[pid].article_number}"):
                    props = graph_nodes.get(ancestor or "", {}).get("props", {}) if graph_nodes else {}
                    if props.get("addressee"):
                        ev.addressee, ev.applicability_note = props["addressee"], props.get("applicability_note")
                        break
            constitution.append(ev)
        constitution.sort(key=lambda e: (ROLE_PRIORITY.get(e.best_role, 9), -e.scores.get("fused", 0)))

        statutes = [Evidence(
            provision_id=st["key"], kind="statute", label=st["label"], text=st.get("text") or "",
            title=st.get("statute"), source_title=st.get("statute") or "Statute",
            source_url=(st.get("statute_props") or {}).get("source_url"),
            version=f"In force from {(st.get('statute_props') or {}).get('in_force_from')}",
            verification=st.get("status") or "unverified", roles=[st["role"]],
            rationale=[st.get("verification_reason") or ""]) for st in graph.statutes]

        uploads: list[Evidence] = []
        if upload_ids:
            index = BM25Index.for_documents(self.db, upload_ids)
            for meta, score in index.search(query_text, top_k=3):
                chunk = self.db.get(DocumentChunk, meta.chunk_id)
                udoc = self.db.get(LegalDocument, meta.document_id)
                uploads.append(Evidence(
                    provision_id=f"UPLOAD:{meta.chunk_id}", kind="upload",
                    label=f"{udoc.title}, page {chunk.page_number}" if chunk.page_number else udoc.title,
                    text=chunk.content[:1500], page=chunk.page_number, source_title=udoc.title,
                    verification="user_supplied_unverified", roles=["retrieved"], scores={"bm25": round(score, 3)},
                    chunk_ids=[meta.chunk_id]))
            diagnostics["strategy"].append("uploads_bm25")
        diagnostics["selected"] = [e.provision_id for e in constitution]
        diagnostics["chunk_ids"] = sorted({c for e in constitution for c in e.chunk_ids})
        return RetrievalResult(constitution=constitution, statutes=statutes, uploads=uploads,
                               remedies=graph.remedies, graph=graph, diagnostics=diagnostics)
