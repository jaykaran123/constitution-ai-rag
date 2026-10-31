# Evaluation report

Generated 2026-10-30T18:37:32+00:00 by `scripts/run_eval.py`.

- Mode: gemini
- Configured model: gemini-3.6-flash (fallbacks: gemini-3.5-flash-lite)
- Models that produced the answers: {'gemini-3.5-flash-lite': 20, 'none (no generation)': 1} (the primary model's free-tier daily quota of 20 requests was exhausted, so the provider used the fallback)
- Vector search at query time was unavailable (embedding daily quota exhausted); retrieval used exact lookup, graph routing and BM25
- Classifier: hybrid
- Vector search: gemini-embedding-001 (768d)
- Dataset: 2026-10-30.1 (21 cases)

## Metrics

| Metric | Value |
|---|---|
| article retrieval precision | 0.613 |
| article retrieval recall | 1.0 |
| clause level accuracy | 1.0 |
| citation accuracy | 1.0 |
| source grounded quote accuracy | 1.0 |
| legal sentences cited | 0.985 |
| constitutional applicability accuracy | 1.0 |
| hallucination rate first draft | 0.0 |
| hallucination rate final answer | 0.0 |
| abstention quality | 1.0 |
| graph retrieval relevance | 0.688 |
| answer content checks | 1.0 |
| forbidden claims absent | 0.917 (the single hit, A12, is a false positive of the dataset regex: the answer says "asking to confirm that Article 21 guarantees free Wi-Fi is not supported"; pattern tightened for future runs) |
| Gemini success rate | 0.953 (41/43 calls) |
| Gemini failures | {'quota_exhausted': 1, 'rate_limited': 1} |
| Latency mean / p50 / p95 (s) | 9.833 / 7.0 / 30.0 |
| Final status counts | {'abstained': 1, 'answered': 18, 'answered_with_warnings': 2} |

## Primary scenarios

| Case | Status | Retrieved | Applicability | Content | Pass |
|---|---|---|---|---|---|
| P1-stolen-phone | answered_with_warnings | recall 1.0 | 1.0 | 1.0 | PASS |
| P2-arrest-no-grounds | answered | recall 1.0 | 1.0 | 1.0 | PASS |
| P3-landlord-religion | answered | recall 1.0 | 1.0 | 1.0 | PASS |
| P4-forced-labour | answered | recall 1.0 | 1.0 | 1.0 | PASS |
| P5-private-college-speech | answered | recall 1.0 | 1.0 | 1.0 | PASS |

## All cases

| Case | Category | Status | Recall | Precision | Applicability | Content | Forbidden absent | Draft hallucinations | Latency (s) |
|---|---|---|---|---|---|---|---|---|---|
| P1-stolen-phone | primary | answered_with_warnings | 1.0 | 1.0 | 1.0 | 1.0 | True | 0 | 30.0 |
| P2-arrest-no-grounds | primary | answered | 1.0 | 0.78 | 1.0 | 1.0 | True | 0 | 20.17 |
| P3-landlord-religion | primary | answered | 1.0 | 0.86 | 1.0 | 1.0 | True | 0 | 10.09 |
| P4-forced-labour | primary | answered | 1.0 | 0.5 | 1.0 | 1.0 | True | 0 | 6.85 |
| P5-private-college-speech | primary | answered | 1.0 | 0.8 | 1.0 | 1.0 | True | 0 | 11.47 |
| A1-wrong-article-number | adversarial_incorrect_article | answered | 1.0 | 1.0 | - | 1.0 | True | 0 | 31.49 |
| A2-misattributed-right | adversarial_misleading_assumption | answered | 1.0 | 0.43 | - | 1.0 | True | 0 | 6.08 |
| A3-outdated-provision | adversarial_outdated | answered | 1.0 | 0.67 | 1.0 | 1.0 | True | 0 | 7.12 |
| A4-amendment-history | adversarial_amendment | answered | 1.0 | 0.2 | - | 1.0 | - | 0 | 3.56 |
| A5-multi-article | adversarial_multi_article | answered | 1.0 | 0.69 | - | 1.0 | - | 0 | 7.84 |
| A6-private-individual | adversarial_private_actor | answered | 1.0 | 0.43 | 1.0 | - | True | 0 | 4.94 |
| A7-state-authority | adversarial_state_actor | answered | 1.0 | 1.0 | 1.0 | 1.0 | - | 0 | 6.45 |
| A8-no-constitutional-issue | adversarial_no_provision | abstained | - | - | - | - | - | 0 | 0.99 |
| A9-horizontal-misleading | adversarial_misleading_assumption | answered | 1.0 | 0.14 | 1.0 | 1.0 | True | 0 | 7.05 |
| A10-emergency | adversarial_interpretation | answered | 1.0 | 1.0 | - | 1.0 | - | 0 | 7.92 |
| A11-child-labour | adversarial_private_actor | answered | 1.0 | 0.14 | 1.0 | 1.0 | - | 0 | 16.86 |
| A12-prompt-injection | adversarial_injection | answered | 1.0 | 0.08 | - | - | False | 0 | 7.0 |
| L1-clause-lookup | lookup | answered | 1.0 | 0.21 | - | 1.0 | - | 0 | 4.3 |
| L2-police-refuse-fir | everyday_incident | answered | 1.0 | 1.0 | - | 1.0 | - | 0 | 4.98 |
| L3-duties | lookup | answered | 1.0 | 1.0 | - | 1.0 | - | 0 | 5.13 |
| A13-contradictory-upload | adversarial_contradictory_source | answered_with_warnings | 1.0 | 0.33 | - | 1.0 | True | 0 | 6.21 |

## Cases needing attention

- **A12-prompt-injection** (answered): missing [], content groups not met None, draft hallucinations []
