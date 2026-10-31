"""Keyword (BM25) and vector indexes over document chunks.

Both indexes are built lazily from the database and cached in memory; they are rebuilt
when the set of chunks or embeddings changes (signature check). The vector index only
includes embeddings produced by the configured model *and* dimension, so a model change
can never mix incompatible vectors.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass

from sqlalchemy import func
from sqlalchemy.orm import Session

from app.knowledge.ingestion import enriched_text
from app.knowledge.text_utils import tokenize
from app.models import DocumentChunk

_lock = threading.Lock()


@dataclass
class ChunkMeta:
    chunk_id: str
    document_id: str
    provision_id: str | None
    chunk_type: str | None


class BM25Index:
    _cache: dict[str, "BM25Index"] = {}

    def __init__(self, chunks: list[DocumentChunk]):
        from rank_bm25 import BM25Okapi, BM25Plus
        self.meta = [ChunkMeta(c.id, c.document_id, c.provision_id, c.chunk_type) for c in chunks]
        corpus = [tokenize(enriched_text(c)) or ["_"] for c in chunks]
        # Okapi IDF turns negative when a term occurs in most documents, which makes every match in a
        # tiny corpus (a single uploaded page) score below zero; BM25+ keeps IDF positive.
        bm25_class = BM25Okapi if len(corpus) >= 50 else BM25Plus
        self.bm25 = bm25_class(corpus, k1=1.4, b=0.6) if corpus else None

    @classmethod
    def for_documents(cls, db: Session, document_ids: list[str]) -> "BM25Index":
        signature = "|".join(sorted(document_ids)) + ":" + str(
            db.query(func.count(DocumentChunk.id), func.max(DocumentChunk.chunk_key))
            .filter(DocumentChunk.document_id.in_(document_ids)).one())
        with _lock:
            index = cls._cache.get(signature)
            if index is None:
                chunks = db.query(DocumentChunk).filter(DocumentChunk.document_id.in_(document_ids)).all()
                index = BM25Index(chunks)
                if len(cls._cache) > 16:
                    cls._cache.clear()
                cls._cache[signature] = index
            return index

    def search(self, query: str, top_k: int = 30) -> list[tuple[ChunkMeta, float]]:
        if not self.bm25:
            return []
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = self.bm25.get_scores(tokens)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [(self.meta[i], float(scores[i])) for i in ranked if scores[i] > 0]


class VectorIndex:
    _cache: dict[str, "VectorIndex"] = {}

    def __init__(self, rows: list[tuple]):
        import numpy as np
        self.meta = [ChunkMeta(r[0], r[1], r[2], r[3]) for r in rows]
        self.matrix = np.vstack([np.frombuffer(r[4], dtype=np.float32) for r in rows]) if rows else None

    @classmethod
    def for_documents(cls, db: Session, document_ids: list[str], model: str, dim: int) -> "VectorIndex":
        base = db.query(DocumentChunk).filter(DocumentChunk.document_id.in_(document_ids),
                                              DocumentChunk.embedding.is_not(None),
                                              DocumentChunk.embedding_model == model,
                                              DocumentChunk.embedding_dim == dim)
        signature = f"{'|'.join(sorted(document_ids))}:{model}:{dim}:{base.count()}"
        with _lock:
            index = cls._cache.get(signature)
            if index is None:
                rows = base.with_entities(DocumentChunk.id, DocumentChunk.document_id, DocumentChunk.provision_id,
                                          DocumentChunk.chunk_type, DocumentChunk.embedding).all()
                index = VectorIndex(rows)
                if len(cls._cache) > 16:
                    cls._cache.clear()
                cls._cache[signature] = index
            return index

    @property
    def size(self) -> int:
        return 0 if self.matrix is None else self.matrix.shape[0]

    def search(self, vector: list[float], top_k: int = 30) -> list[tuple[ChunkMeta, float]]:
        import numpy as np
        if self.matrix is None:
            return []
        q = np.asarray(vector, dtype=np.float32)
        q = q / (np.linalg.norm(q) or 1.0)
        sims = self.matrix @ q
        top = np.argsort(-sims)[:top_k]
        return [(self.meta[i], float(sims[i])) for i in top]


def clear_index_caches() -> None:
    BM25Index._cache.clear()
    VectorIndex._cache.clear()
