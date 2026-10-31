"""
Indian Constitution AI - Database Models

New columns are added by ``app.migrations`` (non-destructive ``ALTER TABLE ... ADD COLUMN``),
so existing conversations and uploads survive upgrades.
"""
import datetime
import uuid

from sqlalchemy import (JSON, Boolean, Column, DateTime, Float, ForeignKey, Index, Integer, LargeBinary, String,
                        Text)
from sqlalchemy.orm import relationship

from app.database import Base


def generate_uuid():
    return str(uuid.uuid4())


def utcnow():
    return datetime.datetime.now(datetime.timezone.utc)


# =============================================================================
# Conversations & Messages
# =============================================================================

class UserAccount(Base):
    __tablename__ = "user_accounts"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    email = Column(String(254), nullable=False, unique=True, index=True)
    password_hash = Column(String(256), nullable=False)
    created_at = Column(DateTime, default=utcnow, nullable=False)


class AccountSession(Base):
    __tablename__ = "account_sessions"

    id = Column(String(100), primary_key=True)  # signed session subject, never the bearer token
    user_id = Column(String(36), ForeignKey("user_accounts.id", ondelete="CASCADE"), nullable=False, index=True)
    expires_at = Column(DateTime, nullable=False)
    revoked_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=utcnow, nullable=False)

class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    session_id = Column(String(64), nullable=False, index=True)
    title = Column(String(255), default="New Conversation")
    created_at = Column(DateTime, default=utcnow)
    updated_at = Column(DateTime, default=utcnow, onupdate=utcnow)

    messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan",
                            order_by="Message.created_at")


class Message(Base):
    __tablename__ = "messages"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    conversation_id = Column(String(36), ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False)
    role = Column(String(20), nullable=False)  # user, assistant, system
    content = Column(Text, nullable=False)
    metadata_json = Column(JSON, nullable=True)  # citations, validation, analysis summary
    created_at = Column(DateTime, default=utcnow)

    conversation = relationship("Conversation", back_populates="messages")
    citations = relationship("Citation", back_populates="message", cascade="all, delete-orphan")


# =============================================================================
# Legal Documents, Provisions & Chunks
# =============================================================================

class LegalDocument(Base):
    __tablename__ = "legal_documents"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    title = Column(String(500), nullable=False)
    document_type = Column(String(50), nullable=False)  # constitution, statute, judgment, user_upload
    source_url = Column(String(1000), nullable=True)
    source_authority = Column(String(255), nullable=True)
    version = Column(String(100), nullable=True)
    publication_date = Column(String(50), nullable=True)
    effective_date = Column(String(50), nullable=True)
    verification_status = Column(String(20), default="unverified")  # verified, unverified, pending
    content_hash = Column(String(64), nullable=True)
    ingestion_status = Column(String(20), default="pending")  # pending, processing, completed, failed
    is_default_source = Column(Boolean, default=False)
    session_id = Column(String(64), nullable=True)  # null for system documents
    file_path = Column(String(1000), nullable=True)
    file_type = Column(String(20), nullable=True)
    file_size = Column(Integer, nullable=True)
    total_chunks = Column(Integer, default=0)
    # Added in schema v2
    parser_version = Column(String(20), nullable=True)
    amendment_status = Column(String(255), nullable=True)
    last_verified_at = Column(DateTime, nullable=True)
    metadata_json = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=utcnow)
    updated_at = Column(DateTime, default=utcnow, onupdate=utcnow)

    chunks = relationship("DocumentChunk", back_populates="document", cascade="all, delete-orphan")


class ConstitutionProvision(Base):
    """One structural unit of the Constitution (Article, clause, sub-clause, Part, Schedule...)."""

    __tablename__ = "constitution_provisions"

    id = Column(String(120), primary_key=True)  # e.g. COI:art:19:cl:1:sub:a
    document_id = Column(String(36), ForeignKey("legal_documents.id", ondelete="CASCADE"), nullable=False)
    document_version = Column(String(100), nullable=True)
    kind = Column(String(30), nullable=False)
    parent_id = Column(String(120), nullable=True)
    label = Column(String(200), nullable=False)          # "Article 19(1)(a)"
    title = Column(String(500), nullable=True)
    text = Column(Text, nullable=False)                  # exact source text incl. amendment markers
    part_number = Column(String(10), nullable=True)
    part_title = Column(String(255), nullable=True)
    chapter = Column(String(10), nullable=True)
    chapter_title = Column(String(255), nullable=True)
    group_heading = Column(String(255), nullable=True)
    article_number = Column(String(10), nullable=True)
    clause_number = Column(String(10), nullable=True)
    subclause = Column(String(5), nullable=True)
    schedule = Column(String(30), nullable=True)
    page_start = Column(Integer, nullable=True)
    page_end = Column(Integer, nullable=True)
    printed_page = Column(String(10), nullable=True)
    status = Column(String(20), default="in_force")      # in_force | amended | omitted
    amendments = Column(JSON, nullable=True)
    ordinal = Column(Integer, default=0)

    __table_args__ = (
        Index("idx_prov_article", "article_number"),
        Index("idx_prov_kind", "kind"),
        Index("idx_prov_parent", "parent_id"),
        Index("idx_prov_document", "document_id"),
    )


class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    document_id = Column(String(36), ForeignKey("legal_documents.id", ondelete="CASCADE"), nullable=False)
    content = Column(Text, nullable=False)
    chunk_index = Column(Integer, nullable=False)

    # Legal metadata
    part = Column(String(100), nullable=True)
    chapter = Column(String(100), nullable=True)
    article_number = Column(String(20), nullable=True)
    article_title = Column(String(500), nullable=True)
    schedule = Column(String(100), nullable=True)
    section = Column(String(50), nullable=True)
    page_number = Column(Integer, nullable=True)

    # Extraction metadata
    extraction_method = Column(String(50), default="text")  # text, ocr, manual
    embedding_model = Column(String(100), nullable=True)
    has_embedding = Column(Boolean, default=False)

    # Added in schema v2
    chunk_key = Column(String(160), nullable=True)            # deterministic identity (idempotent ingestion)
    provision_id = Column(String(120), nullable=True)
    clause_number = Column(String(10), nullable=True)
    subclause = Column(String(5), nullable=True)
    chunk_type = Column(String(30), nullable=True)            # article | clause | schedule_entry | upload ...
    document_version = Column(String(100), nullable=True)
    text_hash = Column(String(64), nullable=True)
    page_end = Column(Integer, nullable=True)
    printed_page = Column(String(10), nullable=True)
    status = Column(String(20), nullable=True)
    embedding = Column(LargeBinary, nullable=True)            # float32 vector bytes
    embedding_dim = Column(Integer, nullable=True)

    created_at = Column(DateTime, default=utcnow)

    document = relationship("LegalDocument", back_populates="chunks")

    __table_args__ = (
        Index("idx_chunk_article", "article_number"),
        Index("idx_chunk_document", "document_id"),
        Index("idx_chunk_content_fts", "content"),
    )


# =============================================================================
# Citations
# =============================================================================

class Citation(Base):
    __tablename__ = "citations"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    message_id = Column(String(36), ForeignKey("messages.id", ondelete="CASCADE"), nullable=False)
    document_id = Column(String(36), ForeignKey("legal_documents.id", ondelete="SET NULL"), nullable=True)
    chunk_id = Column(String(36), nullable=True)
    source_title = Column(String(500), nullable=True)
    source_location = Column(String(255), nullable=True)  # Article 21, Section 3, Page 5
    excerpt = Column(Text, nullable=True)
    relevance_reason = Column(Text, nullable=True)
    confidence = Column(Float, default=0.0)
    created_at = Column(DateTime, default=utcnow)

    message = relationship("Message", back_populates="citations")


# =============================================================================
# Knowledge Graph (relational store; projected into Neo4j when enabled)
# =============================================================================

class KnowledgeEntity(Base):
    __tablename__ = "knowledge_entities"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    entity_type = Column(String(50), nullable=False)  # Neo4j label, e.g. Article, Clause, LegalIssue
    name = Column(String(500), nullable=False)
    description = Column(Text, nullable=True)
    properties = Column(JSON, nullable=True)
    source_document_id = Column(String(36), nullable=True)
    verification_status = Column(String(20), default="verified")
    # Added in schema v2
    key = Column(String(160), nullable=True)                 # unique node identity, e.g. Article:19
    labels = Column(JSON, nullable=True)                     # extra labels, e.g. ["FundamentalDuty"]
    provenance = Column(JSON, nullable=True)
    build_id = Column(String(40), nullable=True)
    created_at = Column(DateTime, default=utcnow)

    __table_args__ = (
        Index("idx_entity_type", "entity_type"),
        Index("idx_entity_name", "name"),
    )


class KnowledgeRelation(Base):
    __tablename__ = "knowledge_relations"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    source_entity_id = Column(String(36), ForeignKey("knowledge_entities.id", ondelete="CASCADE"), nullable=False)
    target_entity_id = Column(String(36), ForeignKey("knowledge_entities.id", ondelete="CASCADE"), nullable=False)
    relation_type = Column(String(50), nullable=False)
    properties = Column(JSON, nullable=True)
    provenance = Column(String(500), nullable=True)
    verification_status = Column(String(20), default="verified")
    # Added in schema v2
    key = Column(String(400), nullable=True)                 # src_key|TYPE|dst_key
    provenance_json = Column(JSON, nullable=True)
    extraction_method = Column(String(30), nullable=True)    # structural | text_pattern | curated | model
    confidence = Column(Float, nullable=True)
    build_id = Column(String(40), nullable=True)
    created_at = Column(DateTime, default=utcnow)

    source_entity = relationship("KnowledgeEntity", foreign_keys=[source_entity_id])
    target_entity = relationship("KnowledgeEntity", foreign_keys=[target_entity_id])

    __table_args__ = (
        Index("idx_relation_type", "relation_type"),
        Index("idx_relation_source", "source_entity_id"),
        Index("idx_relation_target", "target_entity_id"),
    )


# =============================================================================
# Ingestion Jobs, Traces & Migrations
# =============================================================================

class IngestionJob(Base):
    __tablename__ = "ingestion_jobs"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    document_id = Column(String(36), ForeignKey("legal_documents.id", ondelete="CASCADE"), nullable=True)
    source = Column(String(255), nullable=False)
    status = Column(String(20), default="pending")  # pending, processing, completed, failed
    progress = Column(Float, default=0.0)
    total_chunks = Column(Integer, default=0)
    processed_chunks = Column(Integer, default=0)
    error_message = Column(Text, nullable=True)
    retry_count = Column(Integer, default=0)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=utcnow)


class QueryTrace(Base):
    """Per-question debug trace (admin only). Stores a question hash, not the text, by default."""

    __tablename__ = "query_traces"

    request_id = Column(String(32), primary_key=True)
    conversation_id = Column(String(36), nullable=True)
    status = Column(String(30), nullable=True)
    trace = Column(JSON, nullable=False)
    created_at = Column(DateTime, default=utcnow, index=True)


class SchemaMigration(Base):
    __tablename__ = "schema_migrations"

    version = Column(Integer, primary_key=True)
    name = Column(String(200), nullable=False)
    applied_at = Column(DateTime, default=utcnow)
    backup_path = Column(String(1000), nullable=True)
