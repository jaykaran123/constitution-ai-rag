"""
Indian Constitution AI - Pydantic Schemas
"""
from datetime import datetime
from typing import Optional, List, Any
from pydantic import BaseModel, Field


class RegisterRequest(BaseModel):
    email: str = Field(..., min_length=3, max_length=254)
    password: str = Field(..., min_length=8, max_length=128)


class LoginRequest(BaseModel):
    email: str = Field(..., min_length=3, max_length=254)
    password: str = Field(..., min_length=1, max_length=128)


# =============================================================================
# Chat
# =============================================================================

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)
    conversation_id: Optional[str] = Field(default=None, max_length=36)
    # Deprecated: the session is taken from the signed X-Session-Token header.
    session_id: Optional[str] = Field(default=None)
    include_graph: bool = Field(default=True)
    debug: bool = Field(default=False)  # honoured only for administrators


class CitationResponse(BaseModel):
    id: str
    source_title: Optional[str] = None
    source_location: Optional[str] = None
    excerpt: Optional[str] = None
    relevance_reason: Optional[str] = None
    confidence: float = 0.0
    document_id: Optional[str] = None


class ChatResponse(BaseModel):
    message_id: str
    conversation_id: str
    content: str
    citations: List[CitationResponse] = []
    sources_used: List[str] = []
    knowledge_graph_used: bool = False
    model_used: Optional[str] = None
    retrieval_method: str = "none"


# =============================================================================
# Conversations
# =============================================================================

class ConversationSummary(BaseModel):
    id: str
    title: str
    created_at: datetime
    updated_at: datetime
    message_count: int = 0


class MessageResponse(BaseModel):
    id: str
    role: str
    content: str
    citations: List[CitationResponse] = []
    created_at: datetime


class ConversationDetail(BaseModel):
    id: str
    title: str
    messages: List[MessageResponse] = []
    created_at: datetime
    updated_at: datetime


# =============================================================================
# Documents
# =============================================================================

class DocumentUploadResponse(BaseModel):
    id: str
    title: str
    document_type: str
    file_type: Optional[str] = None
    ingestion_status: str
    total_chunks: int = 0
    created_at: datetime


class DocumentListResponse(BaseModel):
    id: str
    title: str
    document_type: str
    verification_status: str
    ingestion_status: str
    is_default_source: bool
    total_chunks: int
    created_at: datetime


# =============================================================================
# Knowledge
# =============================================================================

class KnowledgeStatusResponse(BaseModel):
    default_source_available: bool
    default_source_name: str
    default_source_verified: bool
    total_chunks: int
    embeddings_available: bool
    knowledge_graph_available: bool
    knowledge_graph_entities: int = 0
    knowledge_graph_relations: int = 0
    vector_index_available: bool
    lexical_fallback_available: bool


class KnowledgeInitResponse(BaseModel):
    success: bool
    message: str
    chunks_created: int = 0
    entities_created: int = 0
    relations_created: int = 0
    errors: List[str] = []


# =============================================================================
# Search
# =============================================================================

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=5000)
    top_k: int = Field(default=10, ge=1, le=50)
    search_type: str = Field(default="hybrid")  # semantic, lexical, hybrid
    filters: Optional[dict] = None


class SearchResultItem(BaseModel):
    chunk_id: str
    document_title: str
    content: str
    article_number: Optional[str] = None
    article_title: Optional[str] = None
    part: Optional[str] = None
    score: float = 0.0
    search_method: str = "lexical"


class SearchResponse(BaseModel):
    results: List[SearchResultItem] = []
    total: int = 0
    search_type: str = "hybrid"


# =============================================================================
# Knowledge Graph
# =============================================================================

class GraphNode(BaseModel):
    id: str
    label: str
    entity_type: str
    properties: Optional[dict] = None


class GraphEdge(BaseModel):
    source: str
    target: str
    relation: str
    properties: Optional[dict] = None


class GraphResponse(BaseModel):
    nodes: List[GraphNode] = []
    edges: List[GraphEdge] = []
    query: str = ""


# =============================================================================
# Health
# =============================================================================

class HealthResponse(BaseModel):
    status: str
    version: str = "1.0.0"
    services: dict = {}


class ReadinessResponse(BaseModel):
    ready: bool
    database: bool
    default_knowledge: bool
    llm_configured: bool
    embedding_configured: bool
    knowledge_graph: bool
    details: dict = {}


# =============================================================================
# Source Status
# =============================================================================

class SourceStatusResponse(BaseModel):
    name: str
    source_type: str
    is_available: bool
    is_verified: bool
    is_indexed: bool
    total_chunks: int = 0
    last_updated: Optional[datetime] = None
    details: dict = {}
