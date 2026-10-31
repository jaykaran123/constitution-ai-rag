"""Idempotent ingestion of the Constitution and of user uploads.

Pipeline (Constitution):
    source PDF -> edition/authenticity check -> checksum -> structured extraction
    -> provisions table -> structure-aware chunks -> embeddings (separate async step)
    -> knowledge graph build -> index verification.

Re-running ingestion on an unchanged PDF with the same parser version is a no-op.
When the PDF or the parser changes, provisions and chunks for that document are replaced
in one transaction; embeddings are carried over for chunks whose text hash is unchanged.
"""
from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy.orm import Session

from app.config import get_settings
from app.knowledge.constitution_parser import (PARSER_VERSION, ParsedConstitution, Provision, clean_text,
                                               parse_constitution)
from app.models import ConstitutionProvision, DocumentChunk, LegalDocument
from app.observability import log_event

logger = logging.getLogger("app.ingestion")

OFFICIAL_CONSTITUTION_PDF = ("https://www.legislative.gov.in/static/uploads/2025/08/"
                             "cb1b190ea633a1746368ed1fac35fb30.pdf")
CONSTITUTION_AUTHORITY = "Legislative Department, Ministry of Law and Justice, Government of India"
EDITION_MARKERS = ("the constitution of india", "as on 1st may, 2024", "government of india",
                   "legislative department", "one hundred and sixth amendment) act, 2023", "1st may, 2024")
EDITION_VERSION = "As on 1st May 2024 (up to the Constitution (One Hundred and Sixth Amendment) Act, 2023)"
ARTICLE_CHUNK_LIMIT = 2400


def verify_constitution_edition(path: Path) -> dict:
    """Check identifying cover/preface text of the official 2024 English edition."""
    result = {"verified": False, "markers_found": [], "markers_missing": []}
    try:
        import fitz
        with fitz.open(path) as pdf:
            text = "\n".join(pdf[i].get_text("text") for i in range(min(8, len(pdf))))
        normalized = re.sub(r"\s+", " ", text).lower()
        for marker in EDITION_MARKERS:
            (result["markers_found"] if marker in normalized else result["markers_missing"]).append(marker)
        result["verified"] = not result["markers_missing"]
    except Exception as exc:  # noqa: BLE001
        result["error"] = type(exc).__name__
    return result


@dataclass
class ChunkDraft:
    key: str
    provision: Provision
    content: str          # exact source text of the unit
    chunk_type: str


def enriched_text(chunk: DocumentChunk) -> str:
    """Text used for BM25 and embeddings: label and headings plus the clean source text."""
    label_bits = []
    if chunk.article_number:
        label = f"Article {chunk.article_number}"
        if chunk.clause_number:
            label += f"({chunk.clause_number})"
        label_bits.append(label)
    if chunk.article_title:
        label_bits.append(chunk.article_title)
    if chunk.chapter:
        label_bits.append(chunk.chapter)
    if chunk.part:
        label_bits.append(f"Part {chunk.part}")
    if chunk.schedule:
        label_bits.append(f"{chunk.schedule} Schedule")
    return " | ".join(label_bits) + "\n" + clean_text(chunk.content)


def build_chunks(parsed: ParsedConstitution, doc_key: str) -> list[ChunkDraft]:
    """Structure-aware chunking.

    * Article with no numbered clauses: one chunk with the full Article.
    * Article with clauses: one overview chunk (heading + full text when short) and one
      chunk per clause (each clause chunk keeps its sub-clauses, provisos, Explanations).
    * Omitted Articles: one chunk carrying the omission note.
    * Preamble, Schedule entries/segments and Appendix segments: one chunk each.
    """
    drafts: list[ChunkDraft] = []
    children: dict[str, list[Provision]] = {}
    for p in parsed.provisions:
        if p.parent_id:
            children.setdefault(p.parent_id, []).append(p)
    for p in parsed.provisions:
        if p.kind == "preamble":
            drafts.append(ChunkDraft(f"{doc_key}:{p.provision_id}", p, p.text, "preamble"))
        elif p.kind == "article":
            clauses = [c for c in children.get(p.provision_id, []) if c.kind == "clause"]
            if p.status == "omitted" or not clauses or len(p.text) <= ARTICLE_CHUNK_LIMIT:
                drafts.append(ChunkDraft(f"{doc_key}:{p.provision_id}", p, p.text, "article"))
            else:
                heading = p.text.split("\n", 1)[0]
                drafts.append(ChunkDraft(f"{doc_key}:{p.provision_id}", p, heading, "article_heading"))
            if p.status != "omitted" and clauses and len(clauses) > 1:
                for clause in clauses:
                    drafts.append(ChunkDraft(f"{doc_key}:{clause.provision_id}", clause, clause.text, "clause"))
        elif p.kind in {"schedule_entry", "appendix"}:
            drafts.append(ChunkDraft(f"{doc_key}:{p.provision_id}", p, p.text, p.kind))
    return drafts


def _provision_row(p: Provision, doc: LegalDocument) -> ConstitutionProvision:
    return ConstitutionProvision(
        id=p.provision_id, document_id=doc.id, document_version=doc.version, kind=p.kind, parent_id=p.parent_id,
        label=p.label, title=p.title, text=p.text, part_number=p.part_number, part_title=p.part_title,
        chapter=p.chapter, chapter_title=p.chapter_title, group_heading=p.group_heading,
        article_number=p.article_number, clause_number=p.clause_number, subclause=p.subclause,
        schedule=p.schedule, page_start=p.page_start, page_end=p.page_end, printed_page=p.printed_page,
        status=p.status, amendments=p.amendments or None, ordinal=p.ordinal,
    )


def _article_titles(parsed: ParsedConstitution) -> dict[str, str]:
    return {p.article_number: p.title for p in parsed.provisions if p.kind == "article" and p.article_number}


def ingest_constitution(db: Session, path: Path | None = None, force: bool = False) -> dict:
    """Register and (re)ingest the default Constitution source. Safe to call repeatedly."""
    settings = get_settings()
    path = path or settings.constitution_dir / "constitution.pdf"
    doc = db.query(LegalDocument).filter_by(is_default_source=True).first()
    if not path.exists():
        if doc is None:
            doc = LegalDocument(title="Constitution of India", document_type="constitution", is_default_source=True,
                                source_url=OFFICIAL_CONSTITUTION_PDF, source_authority=CONSTITUTION_AUTHORITY,
                                verification_status="unavailable", ingestion_status="unavailable")
            db.add(doc)
            db.commit()
        return {"status": "unavailable", "reason": f"{path} not found"}

    raw = path.read_bytes()
    checksum = hashlib.sha256(raw).hexdigest()
    edition = verify_constitution_edition(path)
    if doc is None:
        doc = LegalDocument(title="Constitution of India", document_type="constitution", is_default_source=True)
        db.add(doc)
        db.flush()
    doc.source_url = OFFICIAL_CONSTITUTION_PDF
    doc.source_authority = CONSTITUTION_AUTHORITY
    doc.file_path = str(path)
    doc.file_type = "pdf"
    doc.file_size = len(raw)
    doc.version = EDITION_VERSION if edition["verified"] else "Administrator supplied edition (unverified)"
    doc.publication_date = "2024-05-01" if edition["verified"] else None
    doc.amendment_status = ("Incorporates amendments up to the Constitution (One Hundred and Sixth Amendment) "
                            "Act, 2023; later amendments are not included." if edition["verified"] else "unknown")
    doc.verification_status = "verified" if edition["verified"] else "pending"
    doc.last_verified_at = datetime.now(timezone.utc)
    doc.metadata_json = {"edition_check": edition, "checksum_sha256": checksum, "local_path": str(path)}

    existing_provisions = db.query(ConstitutionProvision).filter_by(document_id=doc.id).count()
    up_to_date = (not force and doc.content_hash == checksum and doc.parser_version == PARSER_VERSION
                  and existing_provisions > 0 and doc.ingestion_status == "completed")
    if up_to_date:
        db.commit()
        return {"status": "unchanged", "document_id": doc.id, "provisions": existing_provisions,
                "chunks": doc.total_chunks}

    doc.ingestion_status = "processing"
    db.flush()
    parsed = parse_constitution(path)
    titles = _article_titles(parsed)
    doc_key = "COI"

    # Keep embeddings for unchanged text so re-ingestion does not re-embed everything.
    previous = {c.text_hash: (c.embedding, c.embedding_model, c.embedding_dim)
                for c in db.query(DocumentChunk).filter(DocumentChunk.document_id == doc.id,
                                                        DocumentChunk.embedding.is_not(None)).all()
                if c.text_hash}
    db.query(DocumentChunk).filter_by(document_id=doc.id).delete(synchronize_session=False)
    db.query(ConstitutionProvision).filter_by(document_id=doc.id).delete(synchronize_session=False)
    db.flush()

    db.bulk_save_objects([_provision_row(p, doc) for p in parsed.provisions])
    drafts = build_chunks(parsed, doc_key)
    rows = []
    reused = 0
    for index, draft in enumerate(drafts):
        p = draft.provision
        chunk = DocumentChunk(
            document_id=doc.id, content=draft.content, chunk_index=index, chunk_key=draft.key,
            provision_id=p.provision_id, chunk_type=draft.chunk_type, part=p.part_number,
            chapter=p.chapter_title, article_number=p.article_number,
            article_title=titles.get(p.article_number) if p.article_number else p.title,
            clause_number=p.clause_number, subclause=p.subclause, schedule=p.schedule,
            page_number=p.page_start, page_end=p.page_end, printed_page=p.printed_page, status=p.status,
            extraction_method="pdf_text", document_version=doc.version,
        )
        chunk.text_hash = hashlib.sha256(enriched_text(chunk).encode("utf-8")).hexdigest()
        if chunk.text_hash in previous:
            chunk.embedding, chunk.embedding_model, chunk.embedding_dim = previous[chunk.text_hash]
            chunk.has_embedding = True
            reused += 1
        rows.append(chunk)
    db.add_all(rows)
    doc.total_chunks = len(rows)
    doc.content_hash = checksum
    doc.parser_version = PARSER_VERSION
    doc.ingestion_status = "completed" if rows else "failed"
    db.commit()
    summary = {"status": "ingested", "document_id": doc.id, "provisions": len(parsed.provisions),
               "articles": len(parsed.articles()), "chunks": len(rows), "embeddings_reused": reused,
               "edition_verified": edition["verified"], "checksum": checksum}
    log_event(logger, "ingestion.constitution", **summary)
    return summary


def index_report(db: Session) -> dict:
    """Index verification: counts and coverage checks used by readiness and tests."""
    doc = db.query(LegalDocument).filter_by(is_default_source=True).first()
    if not doc:
        return {"available": False}
    provisions = db.query(ConstitutionProvision).filter_by(document_id=doc.id)
    articles = provisions.filter_by(kind="article").count()
    chunks = db.query(DocumentChunk).filter_by(document_id=doc.id)
    embedded = chunks.filter(DocumentChunk.embedding.is_not(None)).count()
    settings = get_settings()
    matching = chunks.filter(DocumentChunk.embedding_model == settings.gemini_embedding_model,
                             DocumentChunk.embedding_dim == settings.gemini_embedding_dimensions).count()
    article_ids = {r[0] for r in db.query(ConstitutionProvision.id).filter_by(document_id=doc.id, kind="article")}
    chunked = {r[0] for r in db.query(DocumentChunk.provision_id).filter_by(document_id=doc.id)}
    return {"available": True, "document_id": doc.id, "verified": doc.verification_status == "verified",
            "version": doc.version, "articles": articles, "provisions": provisions.count(),
            "chunks": chunks.count(), "embedded_chunks": embedded, "embeddings_matching_config": matching,
            "embedding_dimension_mismatches": embedded - matching,
            "articles_without_chunks": sorted(article_ids - chunked)[:20],
            "parser_version": doc.parser_version}


# --------------------------------------------------------------------------------------
# Embeddings
# --------------------------------------------------------------------------------------

async def embed_pending_chunks(db: Session, provider, limit: int | None = None,
                               document_ids: list[str] | None = None) -> dict:
    """Embed chunks that have no embedding for the configured model/dimension."""
    import numpy as np
    query = db.query(DocumentChunk).filter(
        (DocumentChunk.embedding.is_(None)) | (DocumentChunk.embedding_model != provider.model) |
        (DocumentChunk.embedding_dim != provider.dimensions))
    if document_ids:
        query = query.filter(DocumentChunk.document_id.in_(document_ids))
    pending = query.order_by(DocumentChunk.chunk_index).limit(limit).all() if limit else \
        query.order_by(DocumentChunk.chunk_index).all()
    done = 0
    batch = getattr(provider, "batch_size", 64)
    for start in range(0, len(pending), batch):
        group = pending[start:start + batch]
        vectors = await provider.embed_documents([enriched_text(c)[:6000] for c in group])
        for chunk, vector in zip(group, vectors):
            chunk.embedding = np.asarray(vector, dtype=np.float32).tobytes()
            chunk.embedding_model = provider.model
            chunk.embedding_dim = provider.dimensions
            chunk.has_embedding = True
        db.commit()
        done += len(group)
    result = {"embedded": done, "model": provider.model, "dimensions": provider.dimensions}
    log_event(logger, "ingestion.embeddings", **result)
    return result


# --------------------------------------------------------------------------------------
# User uploads
# --------------------------------------------------------------------------------------

def chunk_plain_text(text: str, limit: int = 1400) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks, current = [], ""
    for paragraph in paragraphs:
        while len(paragraph) > limit:
            cut = paragraph.rfind(". ", 0, limit)
            cut = cut + 1 if cut > limit // 2 else limit
            if current:
                chunks.append(current)
                current = ""
            chunks.append(paragraph[:cut].strip())
            paragraph = paragraph[cut:].strip()
        if current and len(current) + len(paragraph) + 2 > limit:
            chunks.append(current)
            current = paragraph
        else:
            current = f"{current}\n\n{paragraph}".strip()
    if current:
        chunks.append(current)
    return chunks


def ingest_upload(db: Session, doc: LegalDocument, path: Path) -> int:
    """Index a user PDF page by page. Uploads are never treated as authoritative."""
    import fitz
    db.query(DocumentChunk).filter_by(document_id=doc.id).delete(synchronize_session=False)
    count = 0
    with fitz.open(path) as pdf:
        for page_no, page in enumerate(pdf, start=1):
            text = page.get_text("text").strip()
            if not text:
                continue
            for part in chunk_plain_text(text):
                key = f"UPLOAD:{doc.id}:{page_no}:{count}"
                chunk = DocumentChunk(document_id=doc.id, content=part, chunk_index=count, page_number=page_no,
                                      extraction_method="pdf_text", chunk_type="upload", chunk_key=key,
                                      text_hash=hashlib.sha256(part.encode("utf-8")).hexdigest(), status="unverified")
                db.add(chunk)
                count += 1
    doc.total_chunks = count
    doc.ingestion_status = "completed" if count else "failed"
    db.commit()
    return count


def ingest_ocr_text(db: Session, doc: LegalDocument, text: str) -> int:
    db.query(DocumentChunk).filter_by(document_id=doc.id).delete(synchronize_session=False)
    count = 0
    for part in chunk_plain_text(text):
        db.add(DocumentChunk(document_id=doc.id, content=part, chunk_index=count, extraction_method="ocr",
                             chunk_type="upload", chunk_key=f"UPLOAD:{doc.id}:ocr:{count}", status="unverified",
                             text_hash=hashlib.sha256(part.encode("utf-8")).hexdigest()))
        count += 1
    doc.total_chunks = count
    doc.ingestion_status = "completed" if count else "failed"
    db.commit()
    return count
