"""FastAPI application for the single page Constitution assistant."""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import uuid
from datetime import timedelta
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import BackgroundTasks, Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.auth import account_owner_id, hash_password, normalize_email, user_id_from_owner, verify_password
from app.config import PROJECT_DIR, get_settings
from app.database import get_db, get_session_factory, init_db
from app.graph.inspect import inspect_graph
from app.graph.queries import get_graph_queries
from app.knowledge.ingestion import index_report, ingest_constitution, ingest_ocr_text, ingest_upload
from app.llm.config import LLMConfig
from app.llm.provider_factory import get_embedding_provider, get_llm_provider
from app.models import (AccountSession, ConstitutionProvision, Conversation, KnowledgeEntity, LegalDocument,
                        QueryTrace, UserAccount, utcnow)
from app.observability import Trace, configure_logging, log_event, new_request_id, request_id_var
from app.reasoning.pipeline import ChatPipeline
from app.schemas import ChatRequest, LoginRequest, RegisterRequest, SearchRequest
from app.security import (is_admin, issue_account_session, issue_session, rate_limiter, require_admin,
                          session_from_request, verify_session, ACCOUNT_SESSION_TTL)

settings = get_settings()
configure_logging(settings.log_level, settings.log_format)
logger = logging.getLogger("app.main")
WEB = PROJECT_DIR / "web"
_background: set[asyncio.Task] = set()


def _spawn(coro) -> None:
    task = asyncio.create_task(coro)
    _background.add(task)
    task.add_done_callback(_background.discard)


def refresh_knowledge(db: Session, force: bool = False) -> dict:
    """Idempotent: ingest the Constitution, then (re)build and persist the graph if it changed."""
    from app.graph.builder import build_graph
    from app.graph.store import persist_graph
    from app.retrieval.hybrid import ProvisionStore
    from app.retrieval.indexes import clear_index_caches
    summary: dict[str, Any] = {"ingestion": ingest_constitution(db, force=force)}
    if settings.enable_knowledge_graph:
        build = build_graph(db)
        current = {row[0] for row in db.query(KnowledgeEntity.build_id).distinct()}
        if force or current != {build.build_id}:
            summary["graph"] = persist_graph(db, build)
        else:
            summary["graph"] = {"status": "unchanged", "build_id": build.build_id}
    ProvisionStore.invalidate()
    clear_index_caches()
    return summary


async def _post_startup() -> None:
    """Slow, optional work: Neo4j projection and embeddings. Never blocks the API."""
    db = get_session_factory()()
    try:
        if settings.is_neo4j_configured:
            from app.graph import neo4j_store
            if await asyncio.to_thread(neo4j_store.neo4j_available):
                result = await asyncio.to_thread(neo4j_store.sync_to_neo4j, db)
                log_event(logger, "startup.neo4j_sync", **result)
        embedder = get_embedding_provider()
        if embedder is not None:
            from app.knowledge.ingestion import embed_pending_chunks
            from app.retrieval.indexes import clear_index_caches
            try:
                result = await embed_pending_chunks(db, embedder)
                clear_index_caches()
                log_event(logger, "startup.embeddings", **result)
            except Exception as exc:  # noqa: BLE001
                log_event(logger, "startup.embeddings_failed", logging.WARNING,
                          error=getattr(exc, "kind", type(exc).__name__))
    finally:
        db.close()


@asynccontextmanager
async def lifespan(_: FastAPI):
    for problem in settings.validation_problems():
        log_event(logger, "config.problem", logging.WARNING, problem=problem)
    init_db()
    if settings.enable_default_knowledge:
        db = get_session_factory()()
        try:
            log_event(logger, "startup.knowledge", **refresh_knowledge(db))
        finally:
            db.close()
    if settings.app_env != "test":
        _spawn(_post_startup())
    yield
    for task in list(_background):
        task.cancel()


app = FastAPI(title=settings.app_name, version="2.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origin_list, allow_credentials=False,
                   allow_methods=["GET", "POST", "DELETE"], allow_headers=["Content-Type", "X-Session-Token",
                                                                             "X-Admin-Token"])
app.mount("/static", StaticFiles(directory=WEB), name="static")


@app.middleware("http")
async def request_context(request: Request, call_next):
    rid = request.headers.get("X-Request-ID") or new_request_id()
    rid = "".join(ch for ch in rid if ch.isalnum())[:32] or new_request_id()
    token = request_id_var.set(rid)
    try:
        response = await call_next(request)
    finally:
        request_id_var.reset(token)
    response.headers["X-Request-ID"] = rid
    response.headers["X-Content-Type-Options"] = "nosniff"
    return response


@app.exception_handler(Exception)
async def unhandled(request: Request, exc: Exception):
    logger.exception("unhandled error", extra={"event": "http.unhandled", "data": {"path": request.url.path}})
    return JSONResponse(status_code=500, content={"detail": "Internal server error.",
                                                  "request_id": request_id_var.get()})


# ------------------------------------------------------------------------ pages & health
@app.get("/")
def home() -> FileResponse:
    return FileResponse(WEB / "index.html", headers={"Cache-Control": "no-store, max-age=0"})


@app.get("/health/live")
def live() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/health/ready")
def ready(db: Session = Depends(get_db)) -> dict[str, Any]:
    database = True
    try:
        db.execute(text("SELECT 1"))
    except Exception:  # noqa: BLE001
        database = False
    report = index_report(db) if database else {"available": False}
    graph_nodes = db.query(KnowledgeEntity).filter(KnowledgeEntity.key.is_not(None)).count() if database else 0
    neo4j = "disabled"
    if settings.is_neo4j_configured:
        from app.graph.neo4j_store import neo4j_available
        neo4j = "connected" if neo4j_available() else "unreachable"
    ready_flag = database and report.get("available", False) and report.get("articles", 0) > 0
    return {"ready": ready_flag, "database": database,
            "default_knowledge": bool(report.get("articles")), "llm_configured": settings.is_llm_configured,
            "embedding_configured": get_embedding_provider() is not None,
            "knowledge_graph": graph_nodes > 0, "neo4j": neo4j,
            "details": {"index": report, "knowledge_graph_nodes": graph_nodes,
                        "configuration_problems": settings.validation_problems()}}


@app.get("/api/health/llm", dependencies=[Depends(require_admin)])
async def llm_health() -> dict[str, Any]:
    provider = get_llm_provider()
    config = LLMConfig.from_settings()
    if provider is None:
        return {"provider": config.provider, "status": "not_configured", "model": config.model, "latency_ms": None,
                "configuration": config.public_dict()}
    result = await provider.health_check()
    if result["status"] != "connected":
        result["fallbacks"] = []
        for model in config.fallback_models:
            fb = await provider.health_check(model=model)
            result["fallbacks"].append({"model": model, "status": fb["status"], "latency_ms": fb["latency_ms"]})
            if fb["status"] == "connected":
                result["status"] = "degraded"
                result["active_model"] = model
                break
    result["configuration"] = config.public_dict()
    result["usage_since_start"] = provider.usage.snapshot()
    return result


# ------------------------------------------------------------------------ session
@app.post("/api/session")
def create_session(x_session_token: str | None = Header(default=None), legacy_session_id: str | None = None,
                   db: Session = Depends(get_db)) -> dict:
    if x_session_token:
        try:
            from app.security import verify_session
            owner_id = verify_session(x_session_token)
            if owner_id.startswith("user_"):
                session = db.get(AccountSession, x_session_token.partition(".")[0])
                if not session or session.user_id.replace("-", "") != owner_id[5:] or session.revoked_at:
                    raise HTTPException(401, "Your sign-in is no longer active. Please sign in again.")
            return {"session_token": x_session_token}
        except HTTPException:
            # Never silently turn an expired or corrupted account token into a new anonymous identity.
            if x_session_token.partition(".")[0].startswith("u_"):
                raise
    return {"session_token": issue_session(legacy_session_id)}


@app.post("/api/auth/register")
def register(body: RegisterRequest, db: Session = Depends(get_db),
             x_session_token: str | None = Header(default=None)) -> dict[str, Any]:
    try:
        email = normalize_email(body.email)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    rate_limiter.check(f"register:{email}")
    if db.query(UserAccount).filter_by(email=email).first():
        raise HTTPException(409, "An account with that email already exists. Sign in instead.")
    user = UserAccount(email=email, password_hash=hash_password(body.password))
    db.add(user)
    try:
        db.flush()
        owner_id = account_owner_id(user.id)
        old_owner = None
        if x_session_token:
            try:
                old_owner = verify_session(x_session_token)
            except HTTPException:
                pass
        # Keep this browser's anonymous history when it creates its first account.
        if old_owner and not old_owner.startswith("user_"):
            db.query(Conversation).filter_by(session_id=old_owner).update(
                {Conversation.session_id: owner_id}, synchronize_session=False)
            db.query(LegalDocument).filter_by(session_id=old_owner).update(
                {LegalDocument.session_id: owner_id}, synchronize_session=False)
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(409, "An account with that email already exists. Sign in instead.") from exc
    token = issue_account_session(user.id)
    db.add(AccountSession(id=token.partition(".")[0], user_id=user.id,
                          expires_at=utcnow() + timedelta(seconds=ACCOUNT_SESSION_TTL)))
    db.commit()
    return {"session_token": token,
            "user": {"id": user.id, "email": user.email}}


@app.post("/api/auth/login")
def login(body: LoginRequest, request: Request, db: Session = Depends(get_db)) -> dict[str, Any]:
    client_ip = request.client.host if request.client else "unknown"
    rate_limiter.check(f"login:{client_ip}")
    try:
        email = normalize_email(body.email)
    except ValueError:
        email = body.email.strip().casefold()[:254]
    user = db.query(UserAccount).filter_by(email=email).first()
    if not user or not verify_password(body.password, user.password_hash):
        raise HTTPException(401, "Email or password is incorrect.")
    token = issue_account_session(user.id)
    db.add(AccountSession(id=token.partition(".")[0], user_id=user.id,
                          expires_at=utcnow() + timedelta(seconds=ACCOUNT_SESSION_TTL)))
    db.commit()
    return {"session_token": token,
            "user": {"id": user.id, "email": user.email}}


@app.get("/api/auth/me")
def account_me(db: Session = Depends(get_db), session_id: str = Depends(session_from_request)) -> dict[str, Any]:
    user_id = user_id_from_owner(session_id)
    user = db.get(UserAccount, user_id) if user_id else None
    if not user:
        return {"authenticated": False, "user": None}
    return {"authenticated": True, "user": {"id": user.id, "email": user.email}}


@app.post("/api/auth/logout")
def logout(db: Session = Depends(get_db), session_id: str = Depends(session_from_request),
           x_session_token: str | None = Header(default=None)) -> dict[str, bool]:
    if session_id.startswith("user_") and x_session_token:
        account_session = db.get(AccountSession, x_session_token.partition(".")[0])
        if account_session and not account_session.revoked_at:
            account_session.revoked_at = utcnow()
            db.commit()
    return {"logged_out": True}


# ------------------------------------------------------------------------ sources
@app.get("/api/sources/status")
def source_status(db: Session = Depends(get_db)) -> dict[str, Any]:
    doc = db.query(LegalDocument).filter_by(is_default_source=True).first()
    report = index_report(db)
    available = bool(report.get("articles"))
    return {"name": "Constitution of India", "source_type": "default", "is_available": available,
            "is_verified": bool(doc and doc.verification_status == "verified"), "is_indexed": available,
            "total_chunks": report.get("chunks", 0),
            "details": {"status": doc.ingestion_status if doc else "missing", "version": doc.version if doc else None,
                        "publication_date": doc.publication_date if doc else None,
                        "amendment_status": doc.amendment_status if doc else None,
                        "source_url": doc.source_url if doc else None,
                        "source_authority": doc.source_authority if doc else None,
                        "checksum_sha256": doc.content_hash if doc else None,
                        "last_verified_at": doc.last_verified_at.isoformat() if doc and doc.last_verified_at else None,
                        "articles": report.get("articles", 0), "provisions": report.get("provisions", 0),
                        "embedded_chunks": report.get("embedded_chunks", 0),
                        "setup": ("Default source installed and indexed. This edition is dated 1 May 2024; later "
                                  "amendments are not included." if available else
                                  "Place the official PDF at backend/data/constitution/constitution.pdf and restart.")}}


@app.get("/api/provisions/{provision_id}")
def provision(provision_id: str, db: Session = Depends(get_db)) -> dict[str, Any]:
    from app.knowledge.constitution_parser import clean_text
    p = db.get(ConstitutionProvision, provision_id)
    if not p:
        raise HTTPException(404, "Provision not found.")
    kids = db.query(ConstitutionProvision).filter_by(parent_id=p.id).order_by(ConstitutionProvision.id).all()
    return {"provision_id": p.id, "label": p.label, "title": p.title, "kind": p.kind, "status": p.status,
            "text": clean_text(p.text), "source_text": p.text, "part": p.part_number, "part_title": p.part_title,
            "chapter": p.chapter_title, "page": p.page_start, "printed_page": p.printed_page,
            "amendments": p.amendments or [], "children": [{"provision_id": k.id, "label": k.label, "status": k.status}
                                                           for k in kids]}


# ------------------------------------------------------------------------ chat
def _pipeline(db: Session) -> ChatPipeline:
    return ChatPipeline(db, llm=get_llm_provider(), embedder=get_embedding_provider())


@app.post("/api/chat")
async def chat(body: ChatRequest, request: Request, db: Session = Depends(get_db),
               session_id: str = Depends(session_from_request),
               x_admin_token: str | None = Header(default=None)) -> dict[str, Any]:
    rate_limiter.check(session_id)
    trace = Trace(question=body.message, store_text=settings.store_query_text)
    result = await _pipeline(db).run(body.message.strip(), session_id, body.conversation_id, trace)
    if body.debug and is_admin(request, x_admin_token):
        result["debug"] = trace.as_dict()
    return result


@app.post("/api/chat/stream")
async def chat_stream(body: ChatRequest, request: Request, session_id: str = Depends(session_from_request),
                      x_admin_token: str | None = Header(default=None)) -> StreamingResponse:
    rate_limiter.check(session_id)
    admin = body.debug and is_admin(request, x_admin_token)
    trace = Trace(question=body.message, store_text=settings.store_query_text)

    async def events():
        db = get_session_factory()()
        try:
            pipeline = _pipeline(db)
            async for event, data in pipeline.stream(body.message.strip(), session_id, body.conversation_id, trace):
                if await request.is_disconnected():
                    log_event(logger, "chat.stream.client_disconnected", request_id=trace.request_id)
                    return
                if event == "final" and admin:
                    data = {**data, "debug": trace.as_dict()}
                yield f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False, default=str)}\n\n"
        except Exception as exc:  # noqa: BLE001
            logger.exception("stream failed", extra={"event": "chat.stream.error"})
            payload = {"detail": "The answer could not be completed.", "error": type(exc).__name__,
                       "request_id": trace.request_id}
            yield f"event: error\ndata: {json.dumps(payload)}\n\n"
        finally:
            db.close()

    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/api/conversations")
def list_conversations(db: Session = Depends(get_db), session_id: str = Depends(session_from_request)) -> list[dict]:
    rows = db.query(Conversation).filter_by(session_id=session_id).order_by(Conversation.updated_at.desc()).limit(50).all()
    return [{"id": row.id, "title": row.title, "updated_at": row.updated_at.isoformat(),
             "message_count": len(row.messages)} for row in rows]


@app.get("/api/conversations/{conversation_id}")
def get_conversation(conversation_id: str, db: Session = Depends(get_db),
                     session_id: str = Depends(session_from_request)) -> dict[str, Any]:
    row = db.query(Conversation).filter_by(id=conversation_id, session_id=session_id).first()
    if not row:
        raise HTTPException(404, "Conversation not found.")
    messages = []
    for m in row.messages:
        meta = m.metadata_json or {}
        messages.append({"id": m.id, "role": m.role, "content": m.content, "citations": meta.get("citations", []),
                         "status": meta.get("status"), "validation": meta.get("validation"),
                         "assessment": meta.get("assessment"), "confidence": meta.get("confidence"),
                         "unverified_draft": meta.get("unverified_draft"), "created_at": m.created_at.isoformat()})
    return {"id": row.id, "title": row.title, "messages": messages}


@app.delete("/api/conversations/{conversation_id}")
def delete_conversation(conversation_id: str, db: Session = Depends(get_db),
                        session_id: str = Depends(session_from_request)) -> dict[str, bool]:
    row = db.query(Conversation).filter_by(id=conversation_id, session_id=session_id).first()
    if not row:
        raise HTTPException(404, "Conversation not found.")
    db.delete(row)
    db.commit()
    return {"deleted": True}


@app.post("/api/search")
async def search(body: SearchRequest, db: Session = Depends(get_db),
                 session_id: str = Depends(session_from_request)) -> dict[str, Any]:
    from app.retrieval.hybrid import HybridRetriever, ProvisionStore
    from app.retrieval.query_understanding import analyse
    if body.search_type not in {"lexical", "hybrid", "semantic"}:
        raise HTTPException(400, "search_type must be lexical, hybrid, or semantic.")
    by_id, _ = ProvisionStore.load(db)
    analysis = await analyse(body.query, set(by_id), mode="rules")
    embedder = get_embedding_provider() if body.search_type != "lexical" else None
    result = await HybridRetriever(db, embedder).retrieve(analysis, body.query, session_id, top_k=body.top_k)
    return {"results": [{"chunk_id": (e.chunk_ids or [None])[0], "provision_id": e.provision_id,
                         "document_title": e.source_title, "content": e.text, "article_number":
                         (by_id[e.provision_id].article_number if e.provision_id in by_id else None),
                         "article_title": e.title, "label": e.label, "part": e.part, "score": e.scores.get("fused", 0),
                         "scores": e.scores, "roles": e.roles, "search_method": "hybrid"}
                        for e in result.constitution + result.uploads],
            "total": len(result.constitution) + len(result.uploads), "search_type": "hybrid",
            "strategy": result.diagnostics.get("strategy")}


# ------------------------------------------------------------------------ documents
async def _embed_document(document_id: str) -> None:
    embedder = get_embedding_provider()
    if embedder is None:
        return
    from app.knowledge.ingestion import embed_pending_chunks
    db = get_session_factory()()
    try:
        await embed_pending_chunks(db, embedder, document_ids=[document_id])
    except Exception as exc:  # noqa: BLE001
        log_event(logger, "upload.embedding_failed", logging.WARNING, error=getattr(exc, "kind", type(exc).__name__))
    finally:
        db.close()


@app.post("/api/documents/upload")
async def upload(background: BackgroundTasks, file: UploadFile = File(...), session_id_form: str | None = Form(
        default=None, alias="session_id"), db: Session = Depends(get_db),
        session_id: str = Depends(session_from_request)) -> dict[str, Any]:
    rate_limiter.check(f"upload:{session_id}")
    if not settings.enable_user_documents:
        raise HTTPException(403, "Document uploads are disabled.")
    ext = Path(file.filename or "").suffix.lower()
    if ext.lstrip(".") not in settings.allowed_extension_list:
        raise HTTPException(415, "Supported uploads: PDF and PNG/JPG/TIFF/BMP images.")
    raw = await file.read(settings.max_upload_size_mb * 1024 * 1024 + 1)
    if len(raw) > settings.max_upload_size_mb * 1024 * 1024:
        raise HTTPException(413, "File exceeds the configured upload limit.")
    if ext == ".pdf" and not raw.startswith(b"%PDF-"):
        raise HTTPException(415, "The uploaded file is not a valid PDF.")
    dest = settings.upload_path / session_id
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / (uuid.uuid4().hex + ext)
    path.write_bytes(raw)
    doc = LegalDocument(title=Path(file.filename or "Upload").name[:200], document_type="user_upload",
                        verification_status="unverified", ingestion_status="pending", session_id=session_id,
                        file_path=str(path), file_type=ext.lstrip("."), file_size=len(raw),
                        content_hash=hashlib.sha256(raw).hexdigest())
    db.add(doc)
    db.commit()
    db.refresh(doc)
    if ext == ".pdf":
        try:
            ingest_upload(db, doc, path)
        except Exception as exc:  # noqa: BLE001
            doc.ingestion_status = "failed"
            db.commit()
            raise HTTPException(422, "The PDF could not be read.") from exc
    else:
        try:
            import io
            import pytesseract
            from PIL import Image
            pytesseract.pytesseract.tesseract_cmd = settings.tesseract_cmd
            text_value = pytesseract.image_to_string(Image.open(io.BytesIO(raw)), lang="eng")
        except Exception as exc:  # noqa: BLE001
            doc.ingestion_status = "failed"
            db.commit()
            raise HTTPException(503, "Image OCR is unavailable. Install Tesseract OCR and configure TESSERACT_CMD.") from exc
        ingest_ocr_text(db, doc, text_value)
    if doc.ingestion_status == "completed":
        background.add_task(_embed_document, doc.id)
    return {"id": doc.id, "title": doc.title, "document_type": doc.document_type,
            "ingestion_status": doc.ingestion_status, "total_chunks": doc.total_chunks,
            "verification_status": doc.verification_status}


@app.get("/api/documents")
def documents(db: Session = Depends(get_db), session_id: str = Depends(session_from_request)) -> list[dict]:
    return [{"id": d.id, "title": d.title, "ingestion_status": d.ingestion_status, "total_chunks": d.total_chunks,
             "verification_status": d.verification_status}
            for d in db.query(LegalDocument).filter(LegalDocument.session_id == session_id).all()]


# ------------------------------------------------------------------------ knowledge graph
@app.get("/api/knowledge-graph/related")
def graph_related(provisions: str | None = None, query: str | None = None, db: Session = Depends(get_db)) -> dict:
    """Small, validated subgraph around cited provisions (bounded; no free-form queries)."""
    import re
    ids: list[str] = []
    if provisions:
        ids = [p for p in provisions.split(",") if re.fullmatch(r"COI:[A-Za-z0-9:]{3,60}", p)][:4]
    elif query:
        if len(query) > 5000:
            raise HTTPException(422, "Graph query is too long.")
        ids = [f"COI:art:{m.upper()}" for m in re.findall(r"\bArticle\s+(\d{1,3}[A-Z]{0,3})\b", query, re.I)][:3]
    if not ids:
        return {"query": query or provisions, "nodes": [], "edges": []}
    sub = get_graph_queries(db).subgraph_for(ids)
    nodes = [{"id": n["key"], "label": n["name"], "entity_type": n["label"], "status": n["status"]}
             for n in sub["nodes"] if n["status"] in {"verified", "curated"}]
    keep = {n["id"] for n in nodes}
    edges = [{"source": e["source"], "target": e["target"], "relation": e["relation"], "status": e["status"],
              "provenance": e.get("supporting_text") or (e.get("method") or "")}
             for e in sub["edges"] if e["source"] in keep and e["target"] in keep]
    return {"query": query or provisions, "nodes": nodes, "edges": edges}


# ------------------------------------------------------------------------ admin
@app.get("/api/admin/graph/inspect", dependencies=[Depends(require_admin)])
def admin_graph_inspect(db: Session = Depends(get_db)) -> dict:
    report = inspect_graph(db)
    if settings.is_neo4j_configured:
        from app.graph.neo4j_store import neo4j_available, neo4j_counts
        report["neo4j"] = neo4j_counts() if neo4j_available() else {"status": "unreachable"}
    return report


@app.get("/api/admin/traces", dependencies=[Depends(require_admin)])
def admin_traces(limit: int = 20, db: Session = Depends(get_db)) -> list[dict]:
    rows = db.query(QueryTrace).order_by(QueryTrace.created_at.desc()).limit(min(limit, 100)).all()
    return [{"request_id": r.request_id, "status": r.status, "created_at": r.created_at.isoformat(),
             "conversation_id": r.conversation_id} for r in rows]


@app.get("/api/admin/traces/{request_id}", dependencies=[Depends(require_admin)])
def admin_trace(request_id: str, db: Session = Depends(get_db)) -> dict:
    row = db.get(QueryTrace, request_id)
    if not row:
        raise HTTPException(404, "Trace not found.")
    return row.trace


@app.post("/api/admin/reindex", dependencies=[Depends(require_admin)])
async def admin_reindex(force: bool = False, db: Session = Depends(get_db)) -> dict:
    summary = refresh_knowledge(db, force=force)
    _spawn(_post_startup())
    summary["background"] = "neo4j sync and embeddings scheduled"
    return summary


@app.post("/api/admin/documents/{document_id}/extract-graph", dependencies=[Depends(require_admin)])
async def admin_extract_graph(document_id: str, db: Session = Depends(get_db)) -> dict:
    """Model-assisted relationship extraction for an uploaded document.

    Candidates are stored on the document as *unverified* suggestions for human review; only
    those whose supporting text is found verbatim in the document are kept. They are never
    written into the authoritative graph."""
    from app.knowledge.text_utils import contains_quote
    from app.llm.base import GenerationRequest
    from app.llm.prompts import GRAPH_EXTRACTION_SCHEMA, GRAPH_EXTRACTION_SYSTEM, graph_extraction_prompt
    from app.models import DocumentChunk
    doc = db.get(LegalDocument, document_id)
    provider = get_llm_provider()
    if not doc or provider is None:
        raise HTTPException(404 if not doc else 503, "Document not found." if not doc else "Gemini not configured.")
    body = "\n\n".join(c.content for c in db.query(DocumentChunk).filter_by(document_id=doc.id)
                       .order_by(DocumentChunk.chunk_index).limit(12))
    response = await provider.generate(GenerationRequest(prompt=graph_extraction_prompt(doc.title, body),
                                                         system_instruction=GRAPH_EXTRACTION_SYSTEM,
                                                         response_schema=GRAPH_EXTRACTION_SCHEMA, temperature=0.0,
                                                         purpose="graph_extraction"))
    candidates = (response.parsed or {}).get("relationships", [])
    kept = [{**c, "verification_status": "unverified", "extraction_method": "model", "model": response.model}
            for c in candidates if c.get("supporting_text") and contains_quote(body, c["supporting_text"])]
    doc.metadata_json = {**(doc.metadata_json or {}), "candidate_relationships": kept}
    db.commit()
    return {"document_id": doc.id, "proposed": len(candidates), "kept_with_verbatim_support": len(kept),
            "relationships": kept}
