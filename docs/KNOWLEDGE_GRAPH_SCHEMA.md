# Knowledge graph schema

The graph is built by `app/graph/builder.py`, stored in `knowledge_entities` / `knowledge_relations`, and projected to Neo4j by `app/graph/neo4j_store.py`. Every node has a unique `key`; every relationship has a unique key `source|TYPE|target`. Allowed `(source label, relationship, target label)` triples are declared in `app/graph/schema.py`; anything else is rejected at build time.

## Node types (counts from the current build)

| Group | Label | Key example | Count | Notes |
|---|---|---|---|---|
| Constitutional | Constitution | `Constitution:COI` | 1 | |
| | Preamble | `COI:preamble` | 1 | extra label for principles via `DECLARES` |
| | Part | `COI:part:III` | 26 | Part VII stored as omitted |
| | Chapter | `COI:part:VI:ch:V` | 23 | only where the Part has Chapters |
| | Article | `COI:art:21A` | 499 | `status` in_force/amended/omitted; Part IV Articles also labelled `DirectivePrinciple` |
| | Clause | `COI:art:22:cl:1` | 869 | |
| | SubClause | `COI:art:19:cl:1:sub:a` | 646 | Article 51A duties also labelled `FundamentalDuty` |
| | Schedule | `COI:sch:7` | 12 | |
| | Amendment | `Amendment:Constitution (Forty-fourth Amendment) Act, 1978` | 82 | from page footnotes |
| Legal | FundamentalRight | `FundamentalRight:Right to Equality` | 6 | from Part III sub-headings in the text |
| | ConstitutionalPrinciple | `ConstitutionalPrinciple:Justice` | 4 | from the Preamble text |
| | LegalConcept | `LegalConcept:the_state` | 4 | |
| | LegalRestriction | `LegalRestriction:COI:art:19:cl:2` | 5 | Article 19(2)–(6) |
| | Exception | `Exception:COI:art:22:cl:3` | ~35 | "Nothing in this article / clauses (1) and (2) …" |
| | Statute, StatutorySection, Citation | `StatutorySection:BNSS:173:4` | 3 / 10 / 10 | verified only against the official PDF |
| | Offence | `Offence:theft` | 5 | |
| | LegalRemedy | `LegalRemedy:writ_high_court` | 9 | |
| | GovernmentAuthority, Court | `Court:high_court` | 5 / 3 | |
| | CaseLaw | – | 0 | schema supported; none bundled (no verified judgment text) |
| Provenance | Document, DocumentVersion, Chunk, Source | `Chunk:COI:COI:art:21` | 1 / 1 / ~1 670 / 4 | |
| Scenario | LegalScenario, LegalIssue, UserRight | `LegalIssue:arrest_safeguards` | 6 / 21 / 11 | |
| | Incident | – | 0 | the incident is extracted per question and kept in the trace, not persisted (privacy) |

Attributes such as dates, page numbers, titles and amendment actions are properties, not nodes.

## Relationship types

| Relationship | Pattern | Extraction | Verification |
|---|---|---|---|
| CONTAINS | Constitution→Preamble/Part/Schedule, Part→Chapter/Article, Chapter→Article, Statute→StatutorySection, Document→Chunk | structural | verified |
| HAS_CLAUSE / HAS_SUBCLAUSE | Article→Clause, Clause→SubClause, Article→SubClause | structural | verified |
| HAS_VERSION, TEXT_OF, PUBLISHED_BY | Document→DocumentVersion→Constitution, →Source | structural | verified |
| AMENDED | Amendment→Article/Clause/SubClause/Part/Chapter/Preamble/Schedule (`action`, `effective_from`, `footnote`) | footnote | verified |
| PROTECTS | Article→FundamentalRight (Part III sub-heading) | structural | verified |
| DECLARES | Preamble→ConstitutionalPrinciple | text_pattern | verified |
| SUBJECT_TO | SubClause/Article→LegalRestriction ("Nothing in sub-clause (a) of clause (1) … reasonable restrictions") | text_pattern | verified |
| LIMITED_BY | Article/Clause→Exception ("Nothing in this article / in clauses (1) and (2) shall …") | text_pattern | verified |
| STATED_IN | LegalRestriction/Exception→Clause | text_pattern | verified |
| REFERENCES | Article→Article (explicit "article N" in the text), Citation→Source | text_pattern | verified |
| DEFINES | Article/Clause→LegalConcept, StatutorySection/Clause→Offence | curated (quote checked) | verified if the quote is found, else unverified |
| RELATED_TO | LegalConcept→Article/Clause | curated | curated |
| GUARANTEED_BY | UserRight→Clause/Article/SubClause | curated (quote checked) | verified if quote found |
| INVOLVES | LegalScenario→LegalIssue | curated | curated |
| MAY_ENGAGE | LegalIssue→Article/Clause/SubClause (`role`, `condition`, `rationale`) | curated | curated (routing only) |
| MAY_INVOLVE | LegalIssue→Offence | curated | curated |
| AVAILABLE_FOR | LegalRemedy→LegalIssue | curated | curated |
| HANDLES | GovernmentAuthority/Court→LegalRemedy | curated | curated |
| PROVIDED_BY | LegalRemedy→StatutorySection/Clause | curated | inherits the target's status |
| CITED_AS | StatutorySection→Citation | curated | inherits the section's status |
| ESTABLISHED_BY | Court→Clause (Articles 32, 226) | curated | curated |
| INTERPRETED_BY / APPLIES_TO | Article→CaseLaw→LegalConcept | – | reserved; no case law bundled |

`RELATED_TO` between Articles in the brief is represented by `REFERENCES`, which is created only where one Article's text names another, so every such edge has a supporting sentence.

## Provenance

Each relationship stores `provenance_json` with:
`source_document_id`, `document_version`, `source_chunk_id`, `provision_id`, `article_number`, `clause_number`, `page`, `printed_page`, `supporting_text`, `extraction_method`, `verification_status`, and for curated items `knowledge_version`, `knowledge_checksum`, `review_status: needs_legal_review`.

Verification rules:

* Structural, footnote and text-pattern relationships come from the verified official PDF and are `verified`.
* Curated relationships that quote the Constitution are `verified` only if the quote is found verbatim (after whitespace/quote normalisation) in the provision text; otherwise `unverified` and not used as evidence.
* Statutory sections are `verified` only if the official India Code PDF in `data/statutes` contains the excerpt.
* Model-proposed relationships (`/api/admin/documents/{id}/extract-graph`) are stored on the document as `unverified` suggestions and only if their supporting sentence is found in the document. They are never written into the graph.

## Integrity

* Unique keys (unique indexes in SQL; uniqueness constraints per label and on `:KGNode(key)` in Neo4j).
* Schema validation of every relationship triple; structural edges without source provenance raise an error.
* Idempotent builds: the build ID is a hash of all keys; re-running updates rows in place and removes only stale managed rows.
* `python -m scripts.inspect_graph` reports totals, nodes/relationships by type, orphans, Articles without chunks, chunks without documents, missing required relationships, invalid triples, provenance gaps and duplicate keys; it exits non-zero when unhealthy.

## Retrieval queries

Predefined operations (`app/graph/queries.py` for SQL, `Q_*` Cypher constants in `neo4j_store.py`):

* `issue_context(issue_ids)`: provisions an issue may engage, with role/condition/rationale; offences and their defining sections; remedies, authorities and their legal basis (max 6 issues).
* `provision_context(provision_ids)`: the restrictions and exceptions limiting the provision or its parent Article (depth ≤ 2, ≤ 4 per provision), user rights, defined concepts.
* `subgraph_for(provision_ids)`: ≤ 40 nodes for the UI.

All Cypher is static; labels and relationship types come from the schema, values are parameters. No free-form or model-generated Cypher is executed.

## Example (Neo4j Browser)

```cypher
MATCH (i:LegalIssue {key:'LegalIssue:arrest_safeguards'})-[r:MAY_ENGAGE]->(p)
OPTIONAL MATCH (p)-[:LIMITED_BY]->(e:Exception)-[:STATED_IN]->(c)
RETURN i.name, r.role, p.name, e.name, c.name
```
