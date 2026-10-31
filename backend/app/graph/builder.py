"""Builds the legal knowledge graph from verified sources.

Three kinds of relationships are produced, each with provenance:

* ``structural`` - hierarchy read from the document layout (Part CONTAINS Article,
  Article HAS_CLAUSE Clause, Amendment AMENDED Clause from footnotes, Chunk SUPPORTS Article).
* ``text_pattern`` - relationships stated in the provision text itself and recorded with
  the exact supporting sentence (Article 19(2) restricts 19(1)(a); 22(3) excepts 22(1)-(2);
  explicit "article N" references).
* ``curated`` - editorial routing knowledge from ``legal_knowledge.json`` (LegalIssue
  MAY_ENGAGE Article ...). These are marked ``curated``/``needs_legal_review``; any quote
  they rely on is checked verbatim against the source and the edge is downgraded to
  ``unverified`` if the quote is missing.

Nothing produced by an LLM is written here.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

from sqlalchemy.orm import Session

from app.graph.schema import GRAPH_SCHEMA_VERSION, RELATIONSHIP_RULES, provision_label_to_node
from app.knowledge.constitution_parser import clean_text
from app.knowledge.statutes import knowledge_checksum, load_knowledge, verify_statutes
from app.knowledge.text_utils import contains_quote
from app.models import ConstitutionProvision, DocumentChunk, LegalDocument


@dataclass
class GNode:
    key: str
    label: str
    name: str
    props: dict = field(default_factory=dict)
    extra_labels: list[str] = field(default_factory=list)
    provenance: dict | None = None
    status: str = "verified"


@dataclass
class GRel:
    src: str
    type: str
    dst: str
    props: dict = field(default_factory=dict)
    provenance: dict = field(default_factory=dict)
    method: str = "structural"
    status: str = "verified"
    confidence: float | None = None

    @property
    def key(self) -> str:
        return f"{self.src}|{self.type}|{self.dst}"


class GraphValidationError(ValueError):
    pass


class GraphBuild:
    def __init__(self) -> None:
        self.nodes: dict[str, GNode] = {}
        self.rels: dict[str, GRel] = {}
        self.rejected: list[dict] = []

    def node(self, key: str, label: str, name: str, **kwargs) -> GNode:
        existing = self.nodes.get(key)
        if existing:
            existing.props.update(kwargs.get("props") or {})
            return existing
        node = GNode(key=key, label=label, name=name, **kwargs)
        self.nodes[key] = node
        return node

    def rel(self, src: str, rtype: str, dst: str, **kwargs) -> GRel | None:
        if src not in self.nodes or dst not in self.nodes:
            self.rejected.append({"rel": f"{src}|{rtype}|{dst}", "reason": "missing endpoint"})
            return None
        pair = (self.nodes[src].label, self.nodes[dst].label)
        if pair not in RELATIONSHIP_RULES.get(rtype, set()):
            self.rejected.append({"rel": f"{src}|{rtype}|{dst}", "reason": f"{pair} not allowed for {rtype}"})
            return None
        rel = GRel(src=src, type=rtype, dst=dst, **kwargs)
        if rel.method in {"structural", "text_pattern"} and not rel.provenance.get("source_document_id"):
            raise GraphValidationError(f"Relationship {rel.key} lacks source provenance")
        self.rels.setdefault(rel.key, rel)
        return rel

    @property
    def build_id(self) -> str:
        digest = hashlib.sha256()
        # Include curated routing content as well as graph topology. Relationship
        # properties (conditions, roles, rationales) affect query behavior even
        # when the node and edge keys stay the same.
        digest.update(knowledge_checksum().encode())
        for key in sorted(self.nodes):
            digest.update(key.encode())
        for key in sorted(self.rels):
            digest.update(key.encode())
        return digest.hexdigest()[:16]


def _prov(doc: LegalDocument, p: ConstitutionProvision | None = None, *, supporting_text: str | None = None,
          method: str = "structural", chunk_id: str | None = None, status: str = "verified",
          page: int | None = None) -> dict:
    return {
        "source_document_id": doc.id, "document_version": doc.version, "source_chunk_id": chunk_id,
        "provision_id": p.id if p else None, "article_number": p.article_number if p else None,
        "clause_number": p.clause_number if p else None,
        "page": page if page is not None else (p.page_start if p else None),
        "printed_page": p.printed_page if p else None,
        "supporting_text": supporting_text, "extraction_method": method, "verification_status": status,
    }


_SUB_LIST = re.compile(r"\(([a-z])\)")
RESTRICTION_PATTERN = re.compile(
    r"Nothing in \[?(sub-clauses?\s+(?P<subs>\([a-z]\)(?:\s*(?:,|and)\s*\([a-z]\))*)\]?\s+of\s+"
    r"(?:the said clause|clause\s*\((?P<clause>\d+)\)))", re.I)
EXCEPTION_PATTERN = re.compile(
    r"Nothing in (?P<what>this article|clauses?\s+(?P<clauses>\(\d+[A-Z]?\)(?:\s*(?:,|and|or)\s*\(\d+[A-Z]?\))*))"
    r"(?:\s+or in [^.;]{0,140}?)?\s*shall", re.I)
REFERENCE_PATTERN = re.compile(r"\barticles?\s+(\d{1,3}[A-Z]{0,3})(?:\s*(?:,|and|or|to)\s*(\d{1,3}[A-Z]{0,3}))*", re.I)
NUMBER_PATTERN = re.compile(r"\d{1,3}[A-Z]{0,3}")


def _sentence_containing(text: str, start: int) -> str:
    clean = text
    left = max(clean.rfind(".", 0, start), clean.rfind(";", 0, start), clean.rfind("\n(", 0, start))
    right_candidates = [i for i in (clean.find(".", start), clean.find(";", start)) if i != -1]
    right = min(right_candidates) if right_candidates else len(clean)
    return re.sub(r"\s+", " ", clean_text(clean[left + 1:right + 1])).strip()


def build_graph(db: Session) -> GraphBuild:
    doc = db.query(LegalDocument).filter_by(is_default_source=True).first()
    g = GraphBuild()
    if not doc:
        return g
    status = "verified" if doc.verification_status == "verified" else "unverified"
    provisions = db.query(ConstitutionProvision).filter_by(document_id=doc.id).order_by(
        ConstitutionProvision.ordinal).all()
    by_id = {p.id: p for p in provisions}

    # ---------------------------------------------------------------- provenance nodes
    source_key = "Source:legislative.gov.in:constitution"
    g.node(source_key, "Source", "Legislative Department, Government of India",
           props={"url": doc.source_url, "authority": doc.source_authority})
    doc_key = f"Document:{doc.id}"
    g.node(doc_key, "Document", doc.title, props={"document_id": doc.id, "document_type": doc.document_type,
                                                  "checksum_sha256": doc.content_hash})
    version_key = f"DocumentVersion:{doc.id}:{(doc.content_hash or '')[:12]}"
    g.node(version_key, "DocumentVersion", doc.version or "unknown version",
           props={"version": doc.version, "publication_date": doc.publication_date,
                  "amendment_status": doc.amendment_status, "checksum_sha256": doc.content_hash,
                  "verification_status": doc.verification_status, "parser_version": doc.parser_version,
                  "last_verified_at": doc.last_verified_at.isoformat() if doc.last_verified_at else None},
           status=status)
    constitution_key = "Constitution:COI"
    g.node(constitution_key, "Constitution", "Constitution of India", status=status)
    base = _prov(doc)
    g.rel(doc_key, "HAS_VERSION", version_key, provenance=base, status=status)
    g.rel(version_key, "TEXT_OF", constitution_key, provenance=base, status=status)
    g.rel(version_key, "PUBLISHED_BY", source_key, provenance=base, status=status)

    # ---------------------------------------------------------------- provision nodes
    for p in provisions:
        label = provision_label_to_node(p.kind)
        if not label:
            continue
        extra = []
        if p.kind == "article" and p.part_number == "IV":
            extra.append("DirectivePrinciple")
        if p.kind == "subclause" and p.article_number == "51A":
            extra.append("FundamentalDuty")
        g.node(p.id, label, p.label if p.kind != "article" else f"{p.label}: {p.title}", extra_labels=extra,
               props={"provision_id": p.id, "article_number": p.article_number, "clause_number": p.clause_number,
                      "subclause": p.subclause, "part_number": p.part_number, "title": p.title,
                      "status": p.status, "page": p.page_start, "printed_page": p.printed_page,
                      "text": clean_text(p.text)[:4000]},
               provenance=_prov(doc, p, method="structural", status=status), status=status)

    for p in provisions:
        label = provision_label_to_node(p.kind)
        if not label:
            continue
        prov = _prov(doc, p, method="structural", status=status)
        if p.kind in {"preamble", "part", "schedule"}:
            g.rel(constitution_key, "CONTAINS", p.id, provenance=prov, status=status)
        elif p.kind == "chapter":
            g.rel(f"COI:part:{p.part_number}", "CONTAINS", p.id, provenance=prov, status=status)
        elif p.kind == "article":
            chapter_id = f"COI:part:{p.part_number}:ch:{p.chapter}" if p.chapter else None
            parent = chapter_id if chapter_id in by_id else f"COI:part:{p.part_number}"
            g.rel(parent, "CONTAINS", p.id, provenance=prov, status=status)
        elif p.kind == "clause":
            g.rel(p.parent_id, "HAS_CLAUSE", p.id, provenance=prov, status=status)
        elif p.kind == "subclause":
            g.rel(p.parent_id, "HAS_SUBCLAUSE", p.id, provenance=prov, status=status)

    # ---------------------------------------------------------------- amendments (footnotes)
    for p in provisions:
        if not provision_label_to_node(p.kind):
            continue
        for note in p.amendments or []:
            act = note.get("amendment_act")
            if not act:
                continue
            akey = f"Amendment:{act}"
            year = re.search(r"(\d{4})$", act)
            g.node(akey, "Amendment", act, props={"year": int(year.group(1)) if year else None})
            g.rel(akey, "AMENDED", p.id,
                  props={"action": note.get("action"), "effective_from": note.get("effective_from"),
                         "footnote": note.get("note")},
                  provenance=_prov(doc, p, supporting_text=note.get("note"), method="footnote", status=status,
                                   page=note.get("page")),
                  method="footnote", status=status)

    # ---------------------------------------------------------------- Fundamental Right groups
    for p in provisions:
        if p.kind == "article" and p.part_number == "III" and p.group_heading and p.status != "omitted" \
                and p.group_heading not in {"General", "Saving of Certain Laws"}:
            rkey = f"FundamentalRight:{p.group_heading}"
            g.node(rkey, "FundamentalRight", p.group_heading, status=status)
            g.rel(p.id, "PROTECTS", rkey, provenance=_prov(doc, p, supporting_text=p.group_heading,
                                                           method="structural", status=status), status=status)

    # ---------------------------------------------------------------- Preamble principles
    preamble = by_id.get("COI:preamble")
    if preamble:
        for word in ("JUSTICE", "LIBERTY", "EQUALITY", "FRATERNITY"):
            match = re.search(rf"{word}[^\n;]*", preamble.text)
            if match:
                ckey = f"ConstitutionalPrinciple:{word.title()}"
                g.node(ckey, "ConstitutionalPrinciple", word.title(), props={"text": match.group(0).strip()},
                       status=status)
                g.rel(preamble.id, "DECLARES", ckey, provenance=_prov(doc, preamble, supporting_text=match.group(0).strip(),
                                                                       method="text_pattern", status=status),
                      method="text_pattern", status=status)

    # ---------------------------------------------------------------- restrictions & exceptions in text
    for p in provisions:
        if p.kind != "clause" or p.status == "omitted":
            continue
        text = clean_text(p.text)
        for match in RESTRICTION_PATTERN.finditer(text):
            clause_no = match.group("clause") or "1"
            subs = _SUB_LIST.findall(match.group("subs"))
            sentence = _sentence_containing(text, match.start())
            is_restriction = "reasonable restriction" in text.lower()
            label = "LegalRestriction" if is_restriction else "Exception"
            nkey = f"{label}:{p.id}"
            g.node(nkey, label, f"{'Restriction' if is_restriction else 'Exception'} in {p.label}",
                   props={"provision_id": p.id, "text": text[:2000]}, status=status)
            prov = _prov(doc, p, supporting_text=sentence, method="text_pattern", status=status)
            g.rel(nkey, "STATED_IN", p.id, provenance=prov, method="text_pattern", status=status)
            for sub in subs:
                target = f"COI:art:{p.article_number}:cl:{clause_no}:sub:{sub}"
                if target in g.nodes:
                    g.rel(target, "SUBJECT_TO" if is_restriction else "LIMITED_BY", nkey, provenance=prov,
                          method="text_pattern", status=status)
            article_id = f"COI:art:{p.article_number}"
            if article_id in g.nodes:
                g.rel(article_id, "SUBJECT_TO" if is_restriction else "LIMITED_BY", nkey, provenance=prov,
                      method="text_pattern", status=status)
        for match in EXCEPTION_PATTERN.finditer(text):
            if RESTRICTION_PATTERN.match(text, match.start()):
                continue
            sentence = _sentence_containing(text, match.start())
            nkey = f"Exception:{p.id}"
            g.node(nkey, "Exception", f"Exception in {p.label}", props={"provision_id": p.id, "text": text[:2000]},
                   status=status)
            prov = _prov(doc, p, supporting_text=sentence, method="text_pattern", status=status)
            g.rel(nkey, "STATED_IN", p.id, provenance=prov, method="text_pattern", status=status)
            article_id = f"COI:art:{p.article_number}"
            if match.group("what").lower() == "this article":
                g.rel(article_id, "LIMITED_BY", nkey, provenance=prov, method="text_pattern", status=status)
            else:
                for number in re.findall(r"\((\d+[A-Z]?)\)", match.group("clauses")):
                    target = f"COI:art:{p.article_number}:cl:{number}"
                    if target in g.nodes and target != p.id:
                        g.rel(target, "LIMITED_BY", nkey, provenance=prov, method="text_pattern", status=status)

    # ---------------------------------------------------------------- explicit article references
    for p in provisions:
        if p.kind != "article" or p.status == "omitted":
            continue
        text = clean_text(p.text)
        seen: set[str] = set()
        for match in REFERENCE_PATTERN.finditer(text):
            for number in NUMBER_PATTERN.findall(match.group(0)):
                target = f"COI:art:{number.upper()}"
                if target == p.id or target not in g.nodes or target in seen:
                    continue
                seen.add(target)
                g.rel(p.id, "REFERENCES", target, props={"order": len(seen)},
                      provenance=_prov(doc, p, supporting_text=_sentence_containing(text, match.start()),
                                       method="text_pattern", status=status),
                      method="text_pattern", status=status)

    # ---------------------------------------------------------------- chunks
    for chunk in db.query(DocumentChunk).filter_by(document_id=doc.id).all():
        target = chunk.provision_id
        if target and target not in g.nodes and by_id.get(target) and by_id[target].parent_id in g.nodes:
            target = by_id[target].parent_id      # schedule entries support their Schedule
        if not target or target not in g.nodes:
            continue
        ckey = f"Chunk:{chunk.chunk_key}"
        g.node(ckey, "Chunk", chunk.chunk_key, props={"chunk_id": chunk.id, "chunk_type": chunk.chunk_type,
                                                       "page": chunk.page_number, "document_version": doc.version},
               status=status)
        p = by_id.get(chunk.provision_id)
        prov = _prov(doc, p, method="structural", chunk_id=chunk.id, status=status)
        g.rel(doc_key, "CONTAINS", ckey, provenance=prov, status=status)
        g.rel(ckey, "SUPPORTS", target, provenance=prov, status=status)
        # A clause chunk also supports its Article, so every Article is traceable to a chunk.
        if chunk.chunk_type == "clause" and p and p.parent_id in g.nodes:
            g.rel(ckey, "SUPPORTS", p.parent_id, provenance=prov, status=status)

    _add_curated(g, db, doc, by_id, status)
    return g


def _quote_status(by_id: dict, provision_id: str, quote: str | None, doc_status: str) -> str:
    if not quote:
        return "curated"
    p = by_id.get(provision_id)
    if p and contains_quote(p.text, quote):
        return "verified" if doc_status == "verified" else "unverified"
    return "unverified"


def _add_curated(g: GraphBuild, db: Session, doc: LegalDocument, by_id: dict, doc_status: str) -> None:
    kb = load_knowledge()
    kb_version = kb["_meta"]["version"]
    statute_checks = verify_statutes()

    def curated_prov(provision_id: str | None, quote: str | None, quote_status: str) -> dict:
        p = by_id.get(provision_id) if provision_id else None
        prov = _prov(doc, p, supporting_text=quote, method="curated", status=quote_status)
        prov.update({"knowledge_version": kb_version, "review_status": "needs_legal_review",
                     "knowledge_checksum": knowledge_checksum()})
        return prov

    # Provision annotations become node properties (addressee, scope) with their own verification.
    for ann in kb["provision_annotations"]:
        node = g.nodes.get(ann["provision_id"])
        if not node:
            continue
        qs = _quote_status(by_id, ann["provision_id"], ann.get("supporting_text"), doc_status)
        node.props.update({"addressee": ann["addressee"], "applicability_note": ann.get("note"),
                           "addressee_verification": qs, "scope_limited": ann.get("scope_limited", False),
                           "scope_terms": ann.get("scope_terms")})

    # Legal concepts
    for concept in kb["legal_concepts"]:
        ckey = f"LegalConcept:{concept['id']}"
        g.node(ckey, "LegalConcept", concept["name"], props={"description": concept["description"]},
               status="curated")
        if concept.get("defined_by") and concept["defined_by"] in g.nodes:
            ann = next((a for a in kb["provision_annotations"] if a["provision_id"] == concept["defined_by"]), {})
            qs = _quote_status(by_id, concept["defined_by"], ann.get("supporting_text"), doc_status)
            g.rel(concept["defined_by"], "DEFINES", ckey, provenance=curated_prov(concept["defined_by"],
                  ann.get("supporting_text"), qs), method="curated", status=qs)
        for related in concept.get("related_provisions") or []:
            if related in g.nodes:
                g.rel(ckey, "RELATED_TO", related, provenance=curated_prov(related, None, "curated"),
                      method="curated", status="curated")

    # Statutes, sections, citations, offences
    for statute in kb["statutes"]:
        skey = f"Statute:{statute['id']}"
        g.node(skey, "Statute", statute["title"], props={"act_number": statute.get("act_number"),
                                                         "in_force_from": statute.get("in_force_from"),
                                                         "source_url": statute["source_url"]}, status="curated")
        src = f"Source:{statute['source_url']}"
        g.node(src, "Source", statute.get("authority") or "India Code", props={"url": statute["source_url"]},
               status="curated")
        g.rel(skey, "PUBLISHED_BY", src, provenance=curated_prov(None, None, "curated"), method="curated",
              status="curated")
        for section in statute["sections"]:
            check = statute_checks.get(section["id"], {"status": "unverified"})
            sec_key = f"StatutorySection:{section['id']}"
            label = f"{statute['id']} s.{section['section']}{section.get('subsection') or ''}"
            g.node(sec_key, "StatutorySection", f"{label} {section['heading']}",
                   props={"statute": statute["id"], "section": section["section"],
                          "subsection": section.get("subsection"), "heading": section["heading"],
                          "text": section["text"], "verification_reason": check.get("reason")},
                   status=check["status"])
            prov = curated_prov(None, section["text"], check["status"])
            prov.update({"statute": statute["id"], "statute_pdf_sha256": check.get("pdf_sha256")})
            g.rel(skey, "CONTAINS", sec_key, provenance=prov, method="curated", status=check["status"])
            cit = f"Citation:{section['id']}"
            g.node(cit, "Citation", f"{statute['title']}, section {section['section']}{section.get('subsection') or ''}",
                   props={"url": statute["source_url"]}, status=check["status"])
            g.rel(sec_key, "CITED_AS", cit, provenance=prov, method="curated", status=check["status"])
            g.rel(cit, "REFERENCES", src, provenance=prov, method="curated", status=check["status"])

    for offence in kb["offences"]:
        okey = f"Offence:{offence['id']}"
        g.node(okey, "Offence", offence["name"], status="curated")
        definer = offence["defined_by"]
        if definer.startswith("COI:"):
            if definer in g.nodes:
                g.rel(definer, "DEFINES", okey, provenance=curated_prov(definer, None, "curated"), method="curated",
                      status="curated")
        else:
            sec_key = f"StatutorySection:{definer}"
            if sec_key in g.nodes:
                g.rel(sec_key, "DEFINES", okey, provenance=curated_prov(None, None, g.nodes[sec_key].status),
                      method="curated", status=g.nodes[sec_key].status)

    # Authorities, courts and remedies
    for authority in kb["authorities"]:
        label = "Court" if authority["type"] == "Court" else "GovernmentAuthority"
        akey = f"{label}:{authority['id']}"
        g.node(akey, label, authority["name"], props={"is_private_body": authority.get("is_private_body", False)},
               status="curated")
        if authority.get("basis") in g.nodes:
            g.rel(akey, "ESTABLISHED_BY", authority["basis"], provenance=curated_prov(authority["basis"], None, "curated"),
                  method="curated", status="curated")
    authority_keys = {a["id"]: f"{'Court' if a['type'] == 'Court' else 'GovernmentAuthority'}:{a['id']}"
                      for a in kb["authorities"]}
    for remedy in kb["remedies"]:
        rkey = f"LegalRemedy:{remedy['id']}"
        g.node(rkey, "LegalRemedy", remedy["name"], props={"basis": remedy["basis"]}, status="curated")
        if remedy.get("authority") in authority_keys:
            g.rel(authority_keys[remedy["authority"]], "HANDLES", rkey, provenance=curated_prov(None, None, "curated"),
                  method="curated", status="curated")
        for basis in remedy["basis"]:
            target = basis if basis.startswith("COI:") else f"StatutorySection:{basis}"
            if target in g.nodes:
                g.rel(rkey, "PROVIDED_BY", target, provenance=curated_prov(basis if basis.startswith("COI:") else None,
                      None, g.nodes[target].status), method="curated", status=g.nodes[target].status)

    # Issues and scenarios (routing knowledge)
    for issue in kb["legal_issues"]:
        ikey = f"LegalIssue:{issue['id']}"
        g.node(ikey, "LegalIssue", issue["name"],
               props={"category": issue["category"], "ordinary_law_primary": issue["ordinary_law_primary"],
                      "missing_facts": issue.get("missing_facts", [])}, status="curated")
        for engage in issue["engages"]:
            if engage["provision_id"] in g.nodes:
                g.rel(ikey, "MAY_ENGAGE", engage["provision_id"],
                      props={"role": engage["role"], "condition": engage.get("condition"),
                             "rationale": engage.get("rationale")},
                      provenance=curated_prov(engage["provision_id"], None, "curated"), method="curated",
                      status="curated")
        for offence in issue.get("offences", []):
            if f"Offence:{offence}" in g.nodes:
                g.rel(ikey, "MAY_INVOLVE", f"Offence:{offence}", provenance=curated_prov(None, None, "curated"),
                      method="curated", status="curated")
        for remedy in issue.get("remedies", []):
            if f"LegalRemedy:{remedy}" in g.nodes:
                g.rel(f"LegalRemedy:{remedy}", "AVAILABLE_FOR", ikey, provenance=curated_prov(None, None, "curated"),
                      method="curated", status="curated")
    for scenario in kb["scenarios"]:
        skey = f"LegalScenario:{scenario['id']}"
        g.node(skey, "LegalScenario", scenario["name"], props={"description": scenario["description"]},
               status="curated")
        for issue in scenario["issues"]:
            g.rel(skey, "INVOLVES", f"LegalIssue:{issue}", provenance=curated_prov(None, None, "curated"),
                  method="curated", status="curated")

    for right in kb["user_rights"]:
        ukey = f"UserRight:{right['id']}"
        qs = _quote_status(by_id, right["provision_id"], right["supporting_text"], doc_status)
        g.node(ukey, "UserRight", right["name"], status=qs)
        if right["provision_id"] in g.nodes:
            g.rel(ukey, "GUARANTEED_BY", right["provision_id"],
                  provenance=curated_prov(right["provision_id"], right["supporting_text"], qs),
                  method="curated", status=qs)
    for case in kb.get("case_law", []):
        ckey = f"CaseLaw:{case['id']}"
        g.node(ckey, "CaseLaw", case["name"], props=case, status="unverified")


GRAPH_VERSION = GRAPH_SCHEMA_VERSION
