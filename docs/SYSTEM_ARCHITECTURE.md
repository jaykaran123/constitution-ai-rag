# System architecture

## Components

```
Browser (web/index.html, app.js, graph.js)
   │  POST /api/session → signed session token
   │  POST /api/chat/stream (SSE)  ·  /api/chat  ·  /api/conversations  ·  /api/documents/upload
   ▼
FastAPI (backend/app/main.py)
   ├── security.py          signed sessions, admin token, rate limiting
   ├── reasoning/pipeline.py  orchestrates one question (see RAG_PIPELINE.md)
   │     ├── retrieval/query_understanding.py   rules (+ validated Gemini classification)
   │     ├── retrieval/hybrid.py                exact + graph + BM25 + vector → fused evidence
   │     ├── reasoning/applicability.py         State vs private actor, conditions, ordinary law
   │     ├── llm/ (GeminiProvider)              answer generation (streamed), repair
   │     └── reasoning/citations.py             deterministic citation validation
   ├── knowledge/           PDF parser, ingestion, curated legal knowledge, statute verification
   ├── graph/               graph builder, relational store, Neo4j projection, bounded queries, inspection
   ├── llm/                 provider interfaces, Gemini provider, embeddings, prompts, typed errors
   ├── migrations.py        versioned schema migrations with automatic backup
   └── observability.py     JSON logs, request IDs, per-question traces
   │
   ├── SQLite (default) or PostgreSQL via SQLAlchemy
   │     conversations, messages, legal_documents, constitution_provisions, document_chunks
   │     (+ float32 embeddings), knowledge_entities, knowledge_relations, query_traces, schema_migrations
   ├── Neo4j (optional)      projection of the knowledge graph, queried with fixed Cypher
   └── Google Gemini API     generation (gemini-3.6-flash, fallback gemini-3.5-flash-lite), embeddings
```

## Design decisions

* **The existing stack was kept.** The project is FastAPI + SQLite + a static single-page UI. Rewriting it in Next.js or moving to PostgreSQL would not fix any root cause in TECHNICAL_AUDIT.md, so the UI was upgraded in place and the database migrated in place.
* **Relational graph as system of record, Neo4j as projection.** Graph build output is written to keyed relational tables (migrated and backed up with everything else) and then MERGEd into Neo4j when `ENABLE_NEO4J=true`. Retrieval uses Neo4j when it is reachable and the relational graph otherwise; both implement the same predefined operations, so the app works without Neo4j.
* **Embeddings stored with the chunks.** Vectors are stored as float32 blobs in `document_chunks` together with the model name and dimension. The vector index loads only vectors for the configured model and dimension. This works on SQLite and PostgreSQL without pgvector; at ~1 700 chunks a NumPy dot product is a few milliseconds. pgvector is a drop-in later optimisation.
* **Deterministic legal guard-rails around a generative model.** Provision selection, applicability, and citation checks are deterministic and testable. Gemini writes the explanation within those constraints and cannot add a provision that was not retrieved.
* **Fail closed.** If Gemini is unavailable the user gets a clearly labelled answer built only from verified text and the rule-based assessment (`degraded`). If the generated answer fails citation checks twice, the user gets the same evidence-only answer and the draft is hidden behind a warning (`unverified`).

## Data model (main tables)

| Table | Purpose |
|---|---|
| `legal_documents` | Source documents with URL, authority, version, publication date, amendment status, checksum, verification status, parser version, last verification time. |
| `constitution_provisions` | One row per Preamble/Part/Chapter/Article/clause/sub-clause/Schedule entry with exact source text, page and printed page, status (`in_force`, `amended`, `omitted`) and amendment footnotes. |
| `document_chunks` | Retrieval units with `chunk_key` (unique), `provision_id`, Article/clause numbers, document version, text hash and embedding. |
| `knowledge_entities` / `knowledge_relations` | Keyed graph nodes and edges with labels, properties, provenance JSON, verification status, extraction method and build ID. |
| `conversations` / `messages` | Chat history; assistant messages keep citations, validation, assessment and confidence in `metadata_json`. |
| `query_traces` | Admin debug trace per question (question stored as hash unless `STORE_QUERY_TEXT=true`). |
| `schema_migrations` | Applied migrations and backup file paths. |

## API

| Method | Path | Access | Purpose |
|---|---|---|---|
| POST | `/api/session` | public | Issue a signed session token. |
| POST | `/api/chat` | session | Answer a question (JSON). `debug: true` returns the trace for admins. |
| POST | `/api/chat/stream` | session | Same, as Server-Sent Events: `stage`, `meta`, `token`, `final`, `error`. |
| GET/DELETE | `/api/conversations[/{id}]` | session | History, scoped to the session. |
| POST/GET | `/api/documents[/upload]` | session | Upload PDF/image (OCR); uploads are never authoritative. |
| POST | `/api/search` | session | Hybrid search results with per-method scores. |
| GET | `/api/sources/status` | public | Edition, checksum, verification and index status. |
| GET | `/api/provisions/{id}` | public | Structured provision with children and amendments. |
| GET | `/api/knowledge-graph/related?provisions=` | public | Small validated subgraph for the UI. |
| GET | `/health/live`, `/health/ready` | public | Liveness and readiness (database, index, graph, Neo4j, configuration problems). |
| GET | `/api/health/llm` | admin | Gemini connectivity check. |
| GET | `/api/admin/graph/inspect`, `/api/admin/traces[/{id}]` | admin | Graph integrity and debug traces. |
| POST | `/api/admin/reindex`, `/api/admin/documents/{id}/extract-graph` | admin | Rebuild knowledge; propose (unverified) relationships from an upload. |

Admin endpoints require the `X-Admin-Token` header (`ADMIN_API_TOKEN`). When no token is configured they work only from localhost in development.
