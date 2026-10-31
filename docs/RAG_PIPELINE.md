# RAG pipeline

## 1. Ingestion (`app/knowledge/`)

```
official PDF ──► edition check ──► SHA-256 ──► structured extraction ──► provisions table
                 (cover/preface                (parser 2.0.x)              (exact text, pages,
                  markers)                                                  status, amendments)
            ──► structure-aware chunks ──► metadata ──► embeddings (async, paced) ──► graph build ──► index report
```

* **Source.** `backend/data/constitution/constitution.pdf`, the Legislative Department edition "as on 1st May, 2024" (up to the Constitution (One Hundred and Sixth Amendment) Act, 2023). Official URL: https://www.legislative.gov.in/static/uploads/2025/08/cb1b190ea633a1746368ed1fac35fb30.pdf. SHA-256 `819ba7ad…26ed`. Stored metadata: title, URL, authority, version, publication date, amendment status, checksum, last verification time, edition-marker check result.
* **Edition check.** The cover and preface must contain the markers of the 2024 edition (title, date, Legislative Department, One Hundred and Sixth Amendment). Otherwise the document is stored as `pending` and answers say so.
* **OCR.** The official PDF has a text layer, so OCR is not needed for it. Image uploads go through Tesseract (`TESSERACT_CMD`).
* **Structured extraction.** Running headers and printed page numbers are removed and footnotes are separated per page. Parts, Chapters, sub-headings and Article headings are detected; a heading is accepted only if its number continues the Article sequence, which rejects numbered lists. Clauses `(1)`, sub-clauses `(a)` and roman items are split with sequence checks (Article 51A has sub-clauses directly under the Article). Footnote markers such as `2[` are linked to the page's footnote ("Ins. by the Constitution (Ninety-seventh Amendment) Act, 2011 … w.e.f. 15-2-2012"), giving each provision an amendment history. `[… ] Omitted by …` headings and `* * *` clauses are marked `omitted`.
* **Chunking.**
  * Article without clauses: one chunk with the full Article.
  * Article with several clauses: an Article chunk (full text when ≤ 2 400 characters, otherwise its heading) plus one chunk per clause; each clause chunk keeps its sub-clauses, provisos and Explanations.
  * Omitted Article: one chunk carrying the omission note.
  * Preamble; Seventh Schedule entries (List/entry); other Schedules and Appendices in ≤ 1 600-character segments. None of these carry Article numbers.
  * Each chunk records `chunk_key`, `document_id`, `provision_id`, `article_number`, `clause_number`, `part`, `page_number`/`page_end`/`printed_page`, `document_version`, `status`, `text_hash`, `embedding_model`, `embedding_dim`.
* **Idempotency.** Unchanged checksum + parser version ⇒ no-op. Otherwise provisions and chunks for that document are replaced in one transaction and embeddings are reused for identical `text_hash` (914 were reused after the last parser change). Graph builds are keyed and upserted.
* **Index verification.** `index_report()` returns Article/provision/chunk counts, embedding coverage, dimension mismatches and Articles without chunks; `/health/ready` exposes it.

## 2. Query understanding (`app/retrieval/query_understanding.py`)

Rule-based analysis always runs and produces:

* **explicit references** (`Article 19(1)(a)`, `art. 21a`, `Articles 14, 15 and 16`), validated against the provisions table; unknown ones (for example `Article 500`) are reported;
* **legal issues** from the curated issue list (theft, arrest safeguards, discrimination by private/State actors, forced labour, expression, assembly, religion, education, property, voting, Emergency, amendment …), with keyword false positives suppressed (for example "religion" in a landlord question does not trigger *freedom of religion*);
* **categories** (constitutional lookup, everyday incident, police and arrest, discrimination, employment and forced labour, education, freedom of expression, criminal offence, constitutional interpretation, legal remedies, multi-issue, out of scope);
* **actors**: State actors (police, government, municipal …, but not "government *policy*"), private actors (landlord, employer, private college …), institutions of unknown status;
* **jurisdiction** (Indian State or city if named), whether the user asks about a **violation** or a **remedy**, whether this is an **incident** or a conceptual question, and **missing facts**.

With `QUERY_CLASSIFIER=hybrid`, Gemini also classifies the scenario with a JSON schema. Its issues are accepted only from the closed list, it can fill an unknown actor type and add missing facts, and it can never add an Article.

## 3. Hybrid retrieval (`app/retrieval/hybrid.py`)

| Method | Source | Signal |
|---|---|---|
| A. Exact lookup | provisions table | Article/clause/sub-clause the user named (weight 3.0) |
| B. Graph routing | `LegalIssue -MAY_ENGAGE-> provision` | role (primary, user premise, conditional, exception, context, distinguish, remedy) and condition (weight 1.6) |
| C. BM25 | chunk text + headings (BM25Okapi; BM25+ for tiny upload corpora) | keyword relevance (weight 1.0) |
| D. Vector | Gemini `gemini-embedding-001`, 768-d, cosine | semantic relevance (weight 1.0) |
| E. Hierarchy | provisions table | parent/child dedup, sub-clause shown with its clause lead-in |
| F. Reranking | rules | Schedules/Appendices ×0.35 unless asked; omitted provisions dropped unless asked; State-only exceptions dropped when the actor is private |

Fusion is weighted reciprocal rank fusion at the **provision** level: `score = Σ wₘ / (k + rankₘ)`, k = 60. Items routed by exact lookup or the graph are always kept (up to `top_k`). Search-only items need agreement of two methods or a top-3 rank in one, and at most 4 are added. After selection, `provision_context` adds restrictions and exceptions linked in the text, for example 19(1)(a) → 19(2) and 22(1) → 22(3). Statutory sections and remedies come from the graph. Uploaded documents are searched separately, only for the uploader's session, and labelled non-authoritative. All weights are environment settings.

## 4. Applicability assessment (`app/reasoning/applicability.py`)

For each retrieved provision a verdict is computed from its verified annotation (addressee: `state`, `state_and_private`, `arresting_authority`, `definition`, `remedy`, `exception`), its graph condition and the facts:

| Situation | Verdict |
|---|---|
| addressed to the State, actor private | `not_directly_applicable_to_private_actor` |
| addressed to the State, actor State | `may_be_engaged_if_facts_confirmed` |
| addressed to the State, actor unknown | `depends_on_whether_actor_is_the_state` |
| not limited to the State (Articles 17, 23, 24) | `may_be_engaged_regardless_of_actor` |
| Article 15(2) and the facts are not about a listed place | `outside_the_listed_scope_on_stated_facts` |
| condition not met (child, Emergency, public employment …) | `only_if_condition_met` |
| exception / restriction / remedy / definition | descriptive verdicts |
| omitted provision | `omitted_provision` |

Overall: primary domain (`ordinary_law`, `constitutional`, `mixed`, `general_information`), violation status (`unlikely_on_stated_facts`, `possible_but_cannot_be_determined_without_more_facts`, `depends_on_whether_the_actor_is_the_state`, `not_a_violation_question`), premise check for Articles the user named, ordinary-law offences, missing facts, a clarifying question and evidence sufficiency.

## 5. Generation and validation

Gemini receives the constitutional system prompt, the binding assessment (JSON) and an evidence block with IDs (`[C1]` Constitution, `[S1]` statute, `[U1]` upload) that states each item's status, page, edition, verification and why it was retrieved. The answer follows a fixed section structure.

`validate_answer()` then checks the answer:

* every tag exists;
* every Article mentioned was retrieved (or is named inside retrieved text);
* every section mentioned is in the statutory evidence;
* every quotation of 25+ characters appears verbatim in the evidence;
* no sentence makes an unqualified violation claim;
* no sentence suggests a violation of a provision the assessment found not directly applicable;
* legal sentences carry citations.

On failure, one repair call (citation-verification prompt) is made and the result is re-validated. If it still fails, the response is the evidence-only answer with `status: unverified` and the draft is hidden behind a warning.

## 6. Confidence

Confidence is computed, never self-reported by the model:
`0.35 × retrieval quality + 0.25 × share of cited evidence that is verified + 0.25 × validation result + 0.15 × fact completeness`. Levels are high ≥ 0.8, medium ≥ 0.6, and low otherwise. The components are returned with every answer.

## 7. Observability

Each request has an ID (`X-Request-ID`). Each question logs one JSON `chat.answer` event with classification, issues, retrieval strategy, selected provisions, chunk IDs, graph summary, model, latency, token usage, validation status, confidence and stage timings. The full trace is stored in `query_traces` and can be read with `GET /api/admin/traces/{request_id}` or `debug: true` plus the admin token. It covers query understanding, embedding status, BM25/vector/graph candidates and scores, fusion, source extraction, applicability, generation and the validation of both drafts. API keys are redacted by the log formatter. Questions are stored as hashes unless `STORE_QUERY_TEXT=true`.
