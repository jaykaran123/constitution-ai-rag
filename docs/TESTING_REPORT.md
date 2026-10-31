# Testing report

Date: 31 October 2026. Environment: Windows 11, Python 3.12.14, google-genai 1.75.0, SQLite, Neo4j 5.26 (Docker).

## 1. Automated test suite

`python -m pytest` (from `backend/`) runs against a temporary database built from the **real** Constitution PDF. Gemini is a scriptable fake, so every failure mode is deterministic.

| File | Tests | What it covers |
|---|---|---|
| `test_article_accuracy.py` | 27 | Every Article number 1–395 present (238 lies in the omitted Part VII); titles and Parts of key Articles; exact text and page of Article 21; clause/sub-clause trees of Articles 19, 22 and 51A; omitted provisions (2A, 31, 19(1)(f), 259, Part VII); amendment footnotes (42nd, 1st, 44th, 86th); Seventh Schedule entries not labelled as Articles (the old "19. Drugs and poisons" bug). |
| `test_constitutional_retrieval.py` | 13 | Reference parsing; exact clause lookup pulling its restriction (19(1)(a) → 19(2)); full-Article lookup with page and verification; keyword-only retrieval; Schedule handling; graph routing to statutes and remedies; omitted provisions excluded unless asked; vector search fused with BM25 and isolated by model/dimension; session-scoped, non-authoritative uploads; index caching. |
| `test_graph_integrity.py` | 11 (+1 Neo4j) | Healthy graph (no orphans, invalid triples, provenance gaps or Articles without chunks; every required label present); exact provenance on structural and text-pattern edges; curated quotes verified against the source; curated routing never marked as verified law; unverified statutes without official PDFs; idempotent rebuild; rejection of invalid relationships; unique-key constraint; bounded traversal. **Neo4j**: sync idempotent, counts equal to the relational graph, zero orphans, Cypher queries return the same engagements and expansions. |
| `test_citations.py` | 22 | Unknown tags; Articles or sections not in evidence; fabricated and altered quotations; curly quotes/spacing; `[t]he`-style and amendment-bracket quotations; ellipses; definitive and contradicting violation claims vs. allowed hedged and negated statements; uncited legal sentences; non-existent Articles cited by the user. |
| `test_constitutional_reasoning.py` | 24 | The five primary scenarios; wrong Article number; outdated provision; same conduct by State vs private actor; horizontal Article 17 against a misleading premise; multi-Article and amendment questions; out-of-scope; lookups; hybrid Gemini classification merge rules (cannot replace specific rule issues, closed list only, fills gaps, falls back on failure). |
| `test_gemini_connection.py` | 37 (+1 live) | Error mapping (auth, permission, model, rate limit with retry delay, daily quota, server, invalid request, timeout, network); retry then success; no retry on auth or quota; per-model quota fallback and cool-down; timeout → fallback; empty or blocked responses never returned as answers; structured JSON; cancellation; streaming (deltas, fallback before output, error after partial output); health check statuses without secrets; environment-driven model configuration; embedding batching, normalisation, count and dimension mismatch. |
| `test_end_to_end.py` | 20 | Health endpoints; frontend served; source status; signed sessions (forged tokens rejected); admin-only endpoints; **no secret in any response**; validated answer with citations, assessment and computed confidence; SSE event order; Gemini failure → labelled evidence-only answer; no key → evidence-only answer; hallucinated draft repaired; unrepairable draft not shown as a legal answer (draft hidden, full debug trace); out-of-scope question abstains without calling Gemini; follow-up context; conversation privacy; upload scoping and PDF validation; rate limiting; bounded graph endpoint rejects Cypher-like input; provision and admin inspection endpoints. |

**Result:** 148 passed and 2 opt-in tests skipped by default (live Gemini, Neo4j), in about 60 s.

Opt-in tests run separately:

```text
NEO4J_TEST_URI=bolt://127.0.0.1:7688 NEO4J_TEST_PASSWORD=… pytest tests/test_graph_integrity.py -k neo4j   -> 1 passed (36 s)
RUN_LIVE_GEMINI=1 pytest tests/test_gemini_connection.py -k live                                           -> see section 3
```

## 2. Live evaluation (real Gemini, deterministic scoring)

`python -m scripts.run_eval` ran the 21-case dataset (`backend/eval/dataset.json`) through the full pipeline. The cases are the 5 primary scenarios, 13 adversarial cases and 3 lookups. Each case has expected provisions, expected applicability verdicts, required and forbidden answer content, and authoritative source references with PDF page numbers. Scoring is deterministic; no model grades itself. The full per-case report is `docs/EVALUATION_REPORT.md`, and the raw results are in `backend/eval/results/latest.json`.

Final run (31 October 2026). The configured model is `gemini-3.6-flash`. **All 20 generated answers came from the fallback `gemini-3.5-flash-lite`**, because the primary model's free-tier quota of 20 requests/day was already used. Query-time vector search was unavailable for the same reason on the embedding quota, so retrieval used exact lookup, graph routing and BM25.

| Metric | Value |
|---|---|
| Primary scenarios passing every check | **5 / 5** |
| Article retrieval recall | 1.00 |
| Article retrieval precision | 0.61 |
| Clause-level accuracy | 1.00 |
| Citation accuracy (answers passing validation) | 1.00 |
| Quotations verified verbatim | 1.00 |
| Legal sentences carrying a citation | 0.985 |
| Constitutional applicability accuracy | 1.00 |
| Hallucination rate, first draft / final answer | 0.00 / 0.00 |
| Abstention quality | 1.00 |
| Graph retrieval relevance | 0.69 |
| Required answer content present | 1.00 |
| Forbidden claims absent | 0.917 (single hit is a dataset-regex false positive in A12; the answer correctly says the claim "is not supported") |
| Gemini call success rate | 0.953 (41/43; 1 quota, 1 rate limit, both recovered by fallback) |
| Latency mean / p50 / p95 | 9.8 s / 7.0 s / 30.0 s |
| Final statuses | 18 answered, 2 answered with warnings, 1 abstained (out of scope) |

**Precision is the weakest metric.** It is low mainly on lookup questions where the rules matched no issue and Gemini's gap-filling classification added a broad issue, for example "overview of Fundamental Rights" for "What does Article 19(1)(a) say?". The extra provisions are relevant context and are labelled as such, but they are not in the strict expected set. A proposed fix is to skip LLM gap-filling when the user names a provision and does not describe an incident.

### How the numbers changed during this work

| Run | Change before the run | Final status counts | Applicability accuracy | Final-answer hallucination |
|---|---|---|---|---|
| 1 | first live run | 11 degraded (primary model hit its 20/day free quota mid-run; quota errors were not falling back) | 0.87 | 0.00 |
| 2 | per-model quota fallback | 16 answered, 3 with warnings, 2 abstained | 0.87 | 0.00 |
| 3 | issue-merge fix, remedy evidence, precision rule, child-labour triggers | 17 answered, 3 unverified | 1.00 | 0.00 |
| 4–5 | validator false positives fixed (bracket quotes, "rather than a violation", Article 12 context, footnote sections) | see table above | 1.00 | 0.00 |

Each `unverified` result in runs 3–4 was traced to the deterministic validator, not to invented law: quote normalisation and claim detection were too strict. Each fix has a regression test in `test_citations.py`. In every run the final hallucination rate was 0. The validator blocked any draft that cited unretrieved provisions or misquoted the text, and the user saw a labelled evidence-only answer instead.

## 3. Gemini connectivity verification

* `models.list()` with the configured key: 60 models, including `gemini-3.8-flash`, `gemini-3.6-flash`, `gemini-3.5-flash-lite` and `gemini-embedding-001`.
* Health checks (`/api/health/llm` and `scripts/check_gemini.py`):
  * `gemini-3.6-flash` connected in about 3.1 s;
  * `gemini-3.5-flash-lite` connected in about 2.3 s;
  * `gemini-3.8-flash` timed out at 25 s, after one 332 s response.
* A streamed answer through the running server (`POST /api/chat/stream` on port 8001) produced 28 token events and a final event: `status: answered`, `validation: passed`, `confidence: high (0.834)`, all six citations verified.
* Quotas observed on this free-tier key:
  * 20 generation requests/day for `gemini-3.6-flash` and `gemini-3.5-flash` (exhausted during the evaluation; the provider switched to `gemini-3.5-flash-lite` automatically);
  * 1 000 embedding requests/day (914 chunks embedded).

## 4. Docker

**Neo4j container: verified.** `neo4j:5.26-community` started and passed its `cypher-shell` health probe within about 10 s. The full graph was projected into it (3 971 nodes, 7 415 relationships, 0 orphans), the Neo4j integration test passed, and retrieval with Neo4j as the graph back end gave the same provisions, statutes and remedies as the relational back end for all five primary scenarios.

**Backend image: not verified.** Three build attempts failed on network throughput from the Docker build environment, not on the Dockerfile:

1. The Debian mirror delivered about 20 KB/s; the Tesseract install had not finished after 24 minutes when the Docker engine stopped responding (HTTP 500).
2. With `INSTALL_OCR=false`, PyPI downloads ran at about 40 KB/s and timed out (`ReadTimeoutError: files.pythonhosted.org`).
3. With a 120 s timeout, 10 retries and a pip cache mount, pip's index lookup failed outright (`Could not find a version … pydantic-core==2.50.1 (from versions: none)`).

`docker compose config` validates. The same requirements install and run on the host, where the full test suite passes. To finish verification on a normal connection:

```powershell
$env:NEO4J_PASSWORD = "<password>"; $env:APP_HOST_PORT = "8000"
docker compose up -d --build
docker compose ps        # both services "healthy"
curl http://localhost:8000/health/ready
```

`APP_HOST_PORT` exists because port 8000 on this machine is held by the previously running development server.

## 5. Acceptance criteria

| Criterion | Status | Evidence |
|---|---|---|
| Gemini API connectivity verified | Met | Live health check (`connected`, 2–3 s), live streamed answer, live pytest passed |
| Model configuration environment-driven | Met | `GEMINI_*` variables; `test_model_configuration_is_environment_driven` |
| Gemini failures handled correctly | Met | 25 provider tests; live quota exhaustion fell back to `gemini-3.5-flash-lite` automatically |
| Constitution available by default | Met | Ingested at startup from the bundled official PDF; edition and checksum verified |
| Articles structured correctly | Met | 499 Article headings; all numbers 1–395 present except 238, which sits in the omitted Part VII and is not printed; 27 accuracy tests |
| Clauses and sub-clauses preserved | Met | 869 clauses, 646 sub-clauses, provisos and Explanations kept in their clause |
| Amendments handled | Met | Footnote amendment history on every provision; omitted provisions flagged and excluded unless asked |
| Neo4j contains a meaningful, connected graph | Met | 3 971 nodes, 7 415 relationships, 0 orphans in Neo4j 5.26; matches the relational graph |
| Graph relationships have provenance | Met | Every edge has `provenance_json`; inspection reports 0 gaps |
| Hybrid retrieval works | Met, partly limited | Exact, graph, BM25 and vector methods with weighted RRF; vectors cover 914/1 688 chunks (quota) |
| Article-level retrieval works | Met | Recall 1.00, clause accuracy 1.00 |
| Graph-based retrieval works | Met | Same results from SQL and Neo4j back ends for all five scenarios |
| Applicability reasoning works | Met | Applicability accuracy 1.00 |
| State vs private actor handled | Met | Scenarios 1, 3, 5, A6, A9 and the State vs private school test |
| Ordinary offences not treated as constitutional violations | Met | Scenario 1 classed as `ordinary_law` / `unlikely_on_stated_facts`; validator blocks contrary claims |
| Restrictions and exceptions retrieved | Met | 19(1)(a) → 19(2), 22(1) → 22(3); State-only exceptions dropped for private actors |
| Answers contain verified citations | Met | Citation accuracy 1.00, quotations verified 1.00 |
| Unsupported answers abstain or ask for clarification | Met | Abstention quality 1.00; clarifying question when facts are missing; fail-closed `unverified` / `degraded` statuses |
| All five primary scenarios pass | Met | 5/5 in the final live run |
| Frontend and backend communicate | Met (HTTP-level) | SSE stream, sessions, page and scripts verified over HTTP; not visually tested in a browser |
| Docker services start successfully | Partly met: Neo4j verified; backend image build blocked by network (see section 4) | See section 4 |
| Health endpoints return expected results | Met | `/health/live`, `/health/ready`, `/api/health/llm` (admin) |
| No API secrets exposed | Met | End-to-end leak test; repository scan found the key only in `backend/.env` |
| Automated evaluation report generated | Met | `docs/EVALUATION_REPORT.md`, `backend/eval/results/latest.json` |

## 6. Known limitations

* **Statutory text is unverified.** India Code returned HTTP 504 for every download attempt, and the Ministry of Home Affairs page does not link the PDFs. BNS/BNSS/Bonded Labour Act excerpts are therefore marked "not verified against the official text in this installation" in every answer. Run `python -m scripts.fetch_statutes` and then `python -m scripts.seed_graph` when India Code is reachable.
* **Partial vector coverage.** 914 of 1 688 chunks are embedded (free-tier daily quota). Query-time embedding was also unavailable during the evaluation for the same reason, so the evaluation measured exact + graph + BM25 retrieval. The vector path is covered by unit tests. Run `python -m scripts.embed_chunks` after the quota resets.
* **Answer quality was measured on `gemini-3.5-flash-lite`**, because the primary model's free daily quota was used up. With a paid key the primary model `gemini-3.6-flash` is used.
* **No case law is bundled.** Questions that turn on judicial interpretation are answered from the text, with that limitation stated.
* **Curated routing needs legal review.** `legal_knowledge.json` (issues, annotations, remedies) was prepared with AI assistance. Its constitutional quotes are machine-verified, but the legal mapping is marked `needs_legal_review`. Reference answers in the evaluation dataset carry the same flag.
* **Browser rendering was not tested visually.** The UI was exercised through HTTP: the page, scripts, SSE stream and Markdown renderer were checked with Node. No automated browser was available.
