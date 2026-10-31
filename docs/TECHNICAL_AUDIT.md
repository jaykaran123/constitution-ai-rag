# Technical audit

Audit date: 30 October 2026. Scope: the repository as found (`backend/app/*.py`, `web/*`, `backend/tests`, configuration, the SQLite database `backend/data/app.db`, and server logs), followed by the fixes made in this upgrade.

## 1. What the project actually was

The prompt describes a Next.js + PostgreSQL/pgvector + Neo4j + Docker stack. The repository was smaller:

| Area | Found | Notes |
|---|---|---|
| Frontend | Static `index.html` + `app.js` (11 lines, minified) served by FastAPI | Not Next.js. Kept and upgraded, not replaced. |
| Backend | One 923-line `app/main.py` (FastAPI) | Ingestion, retrieval, prompting, Gemini calls and routes in one file. |
| Database | SQLite (`data/app.db`), SQLAlchemy models, no migrations | 42 conversations, 82 messages preserved. |
| Vector search | None | `chromadb`, `sentence-transformers` declared but never imported. |
| Neo4j | None | Driver declared, `ENABLE_NEO4J=false`, never used. |
| LLM | `google-genai` called inline in the chat route; OpenAI branch | Model `gemini-3.8-flash`, fallbacks hard-coded. |
| Docker | None | |
| Tests | 7 tests, 1 failing | |

## 2. Problems found and their root causes

### 2.1 Ingestion produced a corrupted Article index (main source of wrong answers)

* **Fixed 1 800-character slices per page.** Text was cut at arbitrary points, so a clause and its restriction (for example Article 19(1)(a) and 19(2)) could land in different, unlinked chunks.
* **Any numbered line was treated as an Article heading, including in the Schedules.** 1 017 chunks from Schedule and Appendix pages carried an Article number. Examples from the database: Seventh Schedule List I entry 22 "Railways" stored as *Article 22*; Ninth Schedule item "The Maharashtra Land Revenue …" stored as *Article 232*; seven different chunks titled differently but all tagged *Article 19*. Retrieval then cited Schedule entries as Articles.
* **Real Articles were missing.** Headings whose number stood on its own line or behind an editorial bracket were not recognised: **Article 368 (amendment procedure) was absent**, Article 174 was titled just "Sessions", Articles 232/259/272 were mis-tagged.
* **No clause or sub-clause structure, no amendment information.** Footnotes (Subs./Ins./Omitted by … Amendment Act) were ignored, so omitted provisions (Article 31, 19(1)(f)) could be presented as current law.
* **Part detection used the first "PART" string on a page**; 1 117 chunks had no Part.

### 2.2 Retrieval was keyword matching with hard-coded special cases

`retrieve()` used a hand-written lexical score plus scenario-specific branches:

* any question containing *criminal, complaint, FIR, prosecution* returned **no results at all** (unless it was the one hard-coded police case);
* "arrest/detain" questions were restricted to Articles 21/22 only; "police refused + theft" to 14/226/300A only;
* an Article number in the question discarded every other provision, so restrictions and definitions (Article 12) were never retrieved;
* the stop-word list removed legally meaningful words (*complaint, file, section, clause*).

Root cause: no structured index, no semantic search and no graph, so behaviour was patched case by case.

### 2.3 A hard-coded answer bypassed the model and the sources

For "police refused to register a theft complaint" the backend returned a fixed Markdown answer with statutory quotations embedded in code. Those quotations were never checked against an official source, and the answer was returned for any query matching four regular expressions.

### 2.4 Gemini integration failures

* **Model availability**: `gemini-3.8-flash` is a valid ID on the official model list, but in our tests it took **332 s** to answer "OK" once and timed out at 25 s on every later check. `uvicorn.err.log` shows repeated 503s. The code applied only the SDK timeout, so requests hung, and the fallback list was hard-coded.
* **Errors returned as answers**: on any provider error the API returned HTTP 200 with excerpts and "(Error type: RuntimeError)". In the database, **15 of 41 assistant replies** were such "could not reach the model" fallbacks.
* **System prompt concatenated into user content**, no structured output, no token tracking, no streaming, no cancellation.
* `LLM_API_KEY`/`LLM_MODEL` naming; no health check; the existing test asserted a message string the code no longer produced (the failing test).

### 2.5 Hallucination sources

1. Wrong evidence (Schedule entries labelled as Articles; missing Articles) given to the model as "verified".
2. No check that the model's answer cited only supplied evidence; any Article number from model memory passed through.
3. "Confidence" was `score/(score+4)` of a lexical score, shown to users as if meaningful.
4. No State/private actor analysis outside prompt wording, so "Article 21 was violated by the thief" style answers were possible.

### 2.6 Knowledge graph

455 entities and 614 relations in SQLite: only `Part CONTAINS Article` and `Article REFERENCES Article`, partly built from the corrupted index. No clauses, rights, restrictions, statutes or remedies. The graph was never used in retrieval (`knowledge_graph_used` was always `false`). Provenance was a free-text string.

### 2.7 Security and integration

* **Sessions were whatever string the browser sent**: anyone who knew or guessed another session ID could read that person's conversations and uploads.
* No admin authentication; configured rate limits were not enforced.
* `.env`, database path and data directory were resolved relative to the working directory, so running from another folder used a different database.
* Frontend showed mojibake ("âŒ„", "Â·", "â€¦") because `app.js` was saved with mis-decoded UTF-8, and rendered Markdown as raw text.
* `requirements.txt` pulled PyTorch via unused `sentence-transformers`; `pytesseract` was declared but not installed, so image OCR always failed with 503.

### 2.8 Checklist from the brief

| Item | Finding |
|---|---|
| Broken / incomplete components | Vector search, Neo4j, embeddings, streaming, auth: declared but absent. |
| Incorrect database relationships | Chunks linked to wrong Articles (Schedules); no provision hierarchy. |
| Missing graph relationships | No clause, restriction, exception, right, statute, remedy or provenance relations. |
| Embedding dimension mismatches | None in use (no embeddings). Found and guarded a new one: `gemini-embedding-2` returns one vector for a batch of texts. |
| Incorrect retrieval configuration | Hard-coded filters described in 2.2. |
| Gemini authentication failures | Key valid (verified). Failures were timeouts/503 on the configured model. |
| Incorrect model names | `gemini-3.8-flash` valid but not usable in practice; replaced by configurable `gemini-3.6-flash` + fallback. |
| Hallucination sources | Section 2.5. |
| Missing constitutional Articles | 368, 232, 259, 272 (and others with split headings) missing or mislabelled. |
| Incorrect Article-to-clause mappings | No clause mapping at all. |
| Missing source references | Page numbers only; no printed page, edition, amendment or exact-text flag. |
| Frontend/backend integration | Unsigned sessions, no streaming, raw Markdown, mojibake. |

## 3. Fixes implemented

| Problem | Fix | Where |
|---|---|---|
| Corrupted index | Structure-aware parser: Preamble, Parts, Chapters, group headings, Articles, clauses, sub-clauses, Schedules (Seventh Schedule split into Lists/entries), Appendices; footnotes parsed into amendment history; omitted provisions flagged. 499 Article headings recovered (all of 1–395; Article 238 lies in the wholly omitted Part VII and is not printed). | `app/knowledge/constitution_parser.py` |
| Fixed-size chunks | One chunk per Article, plus one per clause for multi-clause Articles; Schedule entries are separate and never carry an Article number. Exact source text preserved. | `app/knowledge/ingestion.py` |
| Keyword retrieval / hard-coded branches | Hybrid retrieval: exact provision lookup, knowledge-graph routing, BM25, Gemini embeddings; weighted reciprocal rank fusion; hierarchy-aware deduplication; restriction/exception expansion. All special cases removed. | `app/retrieval/` |
| No applicability reasoning | Deterministic State/private-actor assessment from verified provision annotations, conditions and stated facts; passed to Gemini as binding. | `app/reasoning/applicability.py` |
| Unchecked answers | Citation validator (evidence IDs, Article/section mentions, verbatim quotes, violation claims vs. assessment, uncited legal sentences), one repair round, then an evidence-only answer if still failing. | `app/reasoning/citations.py`, `pipeline.py` |
| Hard-coded answer | Removed. Statutory text now lives in a curated data file and is marked unverified unless the official PDF confirms it. | `app/knowledge/data/legal_knowledge.json`, `statutes.py` |
| Gemini integration | Provider abstraction; typed errors; per-attempt timeouts; exponential backoff; model fallback; streaming; structured JSON; usage tracking; health check; no retries on auth/quota. | `app/llm/` |
| Errors shown as answers | Failures produce `status: degraded` with an explicitly labelled evidence-only answer; failed validation produces `status: unverified`. The UI labels both. | `pipeline.py`, `web/app.js` |
| Graph | Rebuilt: 28 node types, 25 relationship types, schema-validated, keyed, idempotent, provenance on every edge, projected to Neo4j with constraints. | `app/graph/` |
| Sessions | HMAC-signed server-issued session tokens; admin token for diagnostics; rate limiting. | `app/security.py` |
| Paths | Settings resolve paths from the backend directory. | `app/config.py` |
| Schema changes | Versioned non-destructive migrations with automatic SQLite backup and rollback script. | `app/migrations.py`, `scripts/` |
| Frontend | Same layout; SSE streaming; safe Markdown; status/applicability/confidence chips; clickable [C1] citations; source previews with page, edition, amendment, verification. Encoding fixed. | `web/` |
| Dependencies | Unused heavy packages removed; versions pinned to the tested environment. | `requirements*.txt` |

Preserved: the single-page UI design, conversation history and the existing database rows, upload support (PDF and OCR), session-scoped uploads, the official PDF and its edition check.

## 4. Remaining limitations (also in TESTING_REPORT.md)

* Statutory excerpts (BNS, BNSS, Bonded Labour Act) are **unverified**: India Code returned HTTP 504 throughout the session, so the official PDFs could not be downloaded. The system labels them as unverified in every answer. Run `python -m scripts.fetch_statutes` when India Code is reachable.
* Embeddings: the free-tier daily quota (1 000 embedding requests) was exhausted after 914 of 1 688 chunks. Vector search uses the embedded chunks; the remainder can be embedded with `python -m scripts.embed_chunks` after the quota resets.
* Case law is not bundled. Interpretations that depend on judgments (for example horizontal application of Articles 19/21) are flagged as requiring case law rather than stated as settled.
* Scenario-to-provision routing (`legal_knowledge.json`) was prepared with AI assistance and is marked `needs_legal_review`.
* Docker: the Neo4j service was verified, but the backend image could not be built in this session. The Docker build network was too slow: Debian and PyPI downloads ran at 20–40 KB/s and timed out. See TESTING_REPORT.md section 4.
