# Indian Constitution AI

A single-page legal-information assistant for the Constitution of India. You describe an everyday situation; it identifies the relevant Articles, clauses and restrictions from the official text, checks whether they can apply to the people involved (State or private actor), separates constitutional rights from ordinary criminal and civil law, lists practical remedies, and cites every legal statement. Each citation is checked against the source before the answer is shown.

* Backend: FastAPI, SQLAlchemy (SQLite by default), Google Gemini via the official `google-genai` SDK.
* Retrieval: exact provision lookup, knowledge-graph routing, BM25 and Gemini embeddings, fused by weighted reciprocal rank fusion.
* Knowledge graph: relational system of record, optionally projected into Neo4j.
* Frontend: the static single-page UI in `../web`, served by the backend, with streaming answers.

General legal information, not legal advice.

## Quick start (PowerShell)

```powershell
cd backend
python3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
Copy-Item ..\.env.example .env      # set GEMINI_API_KEY, APP_SECRET_KEY, ADMIN_API_TOKEN
uvicorn app.main:app --port 8000
```

Open http://localhost:8000. The official Constitution PDF (Legislative Department, as on 1 May 2024) ships in `data/constitution/` and is ingested automatically; no upload is needed.

Docker (backend + Neo4j): see `../docs/DEPLOYMENT_GUIDE.md`.

## Useful commands

| Command | Purpose |
|---|---|
| `python -m pytest` | test suite (live Gemini test with `RUN_LIVE_GEMINI=1`) |
| `python -m scripts.check_gemini` | verify Gemini key, models, streaming, embeddings |
| `python -m scripts.ingest_constitution [--force] [--embed]` | parse, index and graph the Constitution |
| `python -m scripts.embed_chunks` | generate missing embeddings (resumes after quota limits) |
| `python -m scripts.seed_graph [--neo4j]` | rebuild the knowledge graph, optionally sync Neo4j |
| `python -m scripts.inspect_graph [--neo4j]` | graph integrity report |
| `python -m scripts.fetch_statutes` | download official statute PDFs for verification |
| `python -m scripts.migrate [--status]` / `python -m scripts.rollback_db` | migrations and rollback |
| `python -m scripts.run_eval [--no-llm]` | evaluation suite -> `../docs/EVALUATION_REPORT.md` |

## Documentation

`../docs/`: TECHNICAL_AUDIT, SYSTEM_ARCHITECTURE, KNOWLEDGE_GRAPH_SCHEMA, RAG_PIPELINE, GEMINI_INTEGRATION, TESTING_REPORT, DEPLOYMENT_GUIDE, EVALUATION_REPORT.

## Known limitations

* Statutory excerpts (BNS, BNSS, Bonded Labour Act) are labelled unverified until the official India Code PDFs are placed in `data/statutes` (`scripts.fetch_statutes`).
* No case law is bundled; answers say when a question depends on judicial interpretation.
* The scenario routing in `app/knowledge/data/legal_knowledge.json` needs review by a qualified lawyer.
* The bundled edition is dated 1 May 2024; later amendments are not included.
