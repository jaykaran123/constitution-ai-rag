"""Automated evaluation of the full question-answering pipeline.

Runs every case in eval/dataset.json through ChatPipeline (real Gemini unless --no-llm) and
scores the result with deterministic checks only (no LLM grading):

* Article retrieval precision / recall, clause-level accuracy
* citation accuracy (validator status of the final answer)
* source-grounded accuracy (quotations verified, legal sentences cited)
* constitutional applicability accuracy (expected verdicts, violation status, actor)
* hallucination rate (validator errors in the *first* draft, before repair, and in the final answer)
* abstention quality, graph retrieval relevance, answer content checks
* Gemini success rate and end-to-end latency

Usage (from backend/):
    python -m scripts.run_eval                    # uses DATABASE_URL / backend/.env
    python -m scripts.run_eval --no-llm           # evidence-only path, no API calls
    python -m scripts.run_eval --cases P1-stolen-phone,P2-arrest-no-grounds
Writes eval/results/latest.json and docs/EVALUATION_REPORT.md.
"""
from __future__ import annotations

import argparse
import asyncio
import collections
import json
import re
import statistics
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

from app.config import PROJECT_DIR, get_settings  # noqa: E402
from app.database import get_session_factory, init_db  # noqa: E402
from app.llm.provider_factory import get_embedding_provider, get_llm_provider  # noqa: E402
from app.models import DocumentChunk, LegalDocument  # noqa: E402
from app.observability import Trace, configure_logging  # noqa: E402
from app.reasoning.pipeline import ChatPipeline  # noqa: E402


def matches(label: str, target: str) -> bool:
    return label == target or label.startswith(target + "(")


def found(target: str, retrieved: list[str]) -> bool:
    if any(matches(l, target) for l in retrieved):
        return True
    # A whole Article that was retrieved covers a clause that was expected.
    return any(target.startswith(l + "(") for l in retrieved)


def relevant(label: str, relevant_set: list[str]) -> bool:
    return any(matches(label, r) or r.startswith(label + "(") for r in relevant_set)


def score_case(case: dict, payload: dict, trace: dict, latency: float) -> dict:
    exp = case["expected"]
    citations = payload.get("citations") or []
    retrieved = [c["source_location"] for c in citations if c.get("kind") == "constitution"]
    content = payload.get("content") or ""
    must = exp.get("must_retrieve", [])
    rel = must + exp.get("acceptable", [])
    checks: dict[str, object] = {}
    if exp.get("abstain"):
        checks["abstention_correct"] = payload.get("status") == "abstained"
    else:
        checks["abstention_correct"] = payload.get("status") != "abstained"
    if must:
        hits = [m for m in must if found(m, retrieved)]
        checks["recall"] = len(hits) / len(must)
        checks["precision"] = (sum(1 for l in retrieved if relevant(l, rel)) / len(retrieved)) if retrieved else 0.0
        clause_targets = [m for m in must if "(" in m]
        if clause_targets:
            checks["clause_accuracy"] = sum(1 for m in clause_targets if any(matches(l, m) for l in retrieved)) / len(clause_targets)
        checks["missing_provisions"] = [m for m in must if m not in hits]
    if exp.get("must_not_retrieve"):
        checks["no_forbidden_provisions"] = not any(found(m, retrieved) and any(l == m for l in retrieved)
                                                   for m in exp["must_not_retrieve"])
    assessment = payload.get("assessment") or {}
    verdicts = {p["label"]: p["verdict"] for p in assessment.get("provisions", [])}
    applicability = []
    for label, allowed in (exp.get("verdicts") or {}).items():
        verdict = verdicts.get(label)
        if verdict is None:
            # A clause retrieved as part of its parent (19(1) contains 19(1)(a)) carries the parent's verdict.
            parents = [l for l in verdicts if label.startswith(l + "(")]
            verdict = verdicts[max(parents, key=len)] if parents else None
        applicability.append(verdict in allowed)
    if exp.get("violation"):
        applicability.append(assessment.get("constitutional_violation") in exp["violation"])
    if exp.get("domain"):
        applicability.append(assessment.get("primary_domain") in exp["domain"])
    if exp.get("actor"):
        applicability.append(assessment.get("actor_type") == exp["actor"])
    if applicability:
        checks["applicability_accuracy"] = sum(applicability) / len(applicability)
    if exp.get("unknown_refs"):
        qu = trace.get("stages", {}).get("query_understanding", {})
        checks["unknown_refs_detected"] = all(r in (qu.get("unknown_refs") or []) for r in exp["unknown_refs"])
    groups = exp.get("answer_must_include", [])
    if groups:
        ok = [any(re.search(p, content, re.I) for p in group) for group in groups]
        checks["content_include"] = sum(ok) / len(ok)
        checks["content_include_failed"] = [g for g, passed in zip(groups, ok) if not passed]
    forbidden = exp.get("answer_must_not_include", [])
    if forbidden:
        checks["content_forbidden_absent"] = not any(re.search(p, content) for p in forbidden)
    validation = payload.get("validation") or {}
    stage = trace.get("stages", {}).get("citation_validation", {})
    initial = stage.get("initial") or validation or {}
    halluc = re.compile(r"not retrieved|not in the statutory evidence|not found verbatim|does not refer to any")
    checks["draft_hallucinations"] = [e for e in (initial.get("errors") or []) if halluc.search(e)]
    checks["final_hallucinations"] = [e for e in (validation.get("errors") or []) if halluc.search(e)] \
        if payload.get("status") in {"answered", "answered_with_warnings"} else []
    checks["citation_valid"] = payload.get("status") in {"answered", "answered_with_warnings", "degraded", "abstained"}
    if validation:
        checks["quotes_verified"] = (validation.get("quotes_verified", 0), validation.get("quotes_checked", 0))
        checks["legal_sentences_cited"] = (validation.get("cited_legal_sentences", 0), validation.get("legal_sentences", 0))
    graph = trace.get("stages", {}).get("retrieval", {}).get("graph", {})
    engaged = graph.get("engaged") or []
    if engaged and rel:
        labels = {c["provision_id"]: c["source_location"] for c in citations if c.get("kind") == "constitution"}
        engaged_labels = [labels.get(pid) for pid in engaged if labels.get(pid)]
        if engaged_labels:
            checks["graph_relevance"] = sum(1 for l in engaged_labels if relevant(l, rel)) / len(engaged_labels)
    return {"id": case["id"], "category": case["category"], "question": case["question"],
            "status": payload.get("status"), "model": payload.get("model_used"), "latency_s": round(latency, 2),
            "confidence": (payload.get("confidence") or {}).get("level"), "retrieved": retrieved,
            "cited": [c["source_location"] for c in citations if c.get("cited_in_answer")],
            "checks": checks, "answer": content, "request_id": payload.get("request_id")}


def mean(values):
    values = [v for v in values if v is not None]
    return round(statistics.mean(values), 3) if values else None


def aggregate(results: list[dict], llm_usage: dict | None) -> dict:
    c = [r["checks"] for r in results]
    answered = [r for r in results if r["status"] not in {"abstained"}]
    latencies = sorted(r["latency_s"] for r in results)
    draft_h = [r for r in results if r["checks"].get("draft_hallucinations")]
    final_h = [r for r in results if r["checks"].get("final_hallucinations")]
    quotes = [x["quotes_verified"] for x in c if "quotes_verified" in x]
    sentences = [x["legal_sentences_cited"] for x in c if "legal_sentences_cited" in x]
    content = [x for x in c if "content_include" in x]
    return {
        "cases": len(results),
        "article_retrieval_precision": mean([x.get("precision") for x in c]),
        "article_retrieval_recall": mean([x.get("recall") for x in c]),
        "clause_level_accuracy": mean([x.get("clause_accuracy") for x in c]),
        "citation_accuracy": mean([1.0 if x["citation_valid"] else 0.0 for x in c]),
        "source_grounded_quote_accuracy": round(sum(a for a, _ in quotes) / max(1, sum(b for _, b in quotes)), 3),
        "legal_sentences_cited": round(sum(a for a, _ in sentences) / max(1, sum(b for _, b in sentences)), 3),
        "constitutional_applicability_accuracy": mean([x.get("applicability_accuracy") for x in c]),
        "hallucination_rate_first_draft": round(len(draft_h) / max(1, len(answered)), 3),
        "hallucination_rate_final_answer": round(len(final_h) / max(1, len(answered)), 3),
        "abstention_quality": mean([1.0 if x["abstention_correct"] else 0.0 for x in c]),
        "graph_retrieval_relevance": mean([x.get("graph_relevance") for x in c]),
        "answer_content_checks": mean([x["content_include"] for x in content]),
        "forbidden_claims_absent": mean([1.0 if x["content_forbidden_absent"] else 0.0 for x in c
                                         if "content_forbidden_absent" in x]),
        "status_counts": {s: sum(1 for r in results if r["status"] == s) for s in sorted({r["status"] for r in results})},
        "gemini": llm_usage,
        "latency_s": {"mean": mean(latencies), "p50": latencies[len(latencies) // 2] if latencies else None,
                      "p95": latencies[min(len(latencies) - 1, int(len(latencies) * 0.95))] if latencies else None},
    }


def primary_pass(r: dict) -> bool:
    c = r["checks"]
    return (c.get("recall", 1) == 1 and c.get("applicability_accuracy", 1) == 1 and c.get("content_include", 1) == 1
            and c.get("content_forbidden_absent", True) and c["abstention_correct"] and not c["final_hallucinations"]
            and c.get("no_forbidden_provisions", True))


def write_report(summary: dict, results: list[dict], meta: dict, path: Path) -> None:
    lines = ["# Evaluation report", "", f"Generated {meta['generated_at']} by `scripts/run_eval.py`.", "",
             f"- Mode: {meta['mode']}", f"- Configured model: {meta['model']} (fallbacks: {meta['fallbacks']})",
             f"- Models that produced the answers: {dict(collections.Counter(r['model'] or 'none (no generation)' for r in results))}",
             f"- Classifier: {meta['classifier']}", f"- Vector search: {meta['vector']}",
             f"- Dataset: {meta['dataset_version']} ({summary['cases']} cases)", "",
             "## Metrics", "", "| Metric | Value |", "|---|---|"]
    for key in ["article_retrieval_precision", "article_retrieval_recall", "clause_level_accuracy", "citation_accuracy",
                "source_grounded_quote_accuracy", "legal_sentences_cited", "constitutional_applicability_accuracy",
                "hallucination_rate_first_draft", "hallucination_rate_final_answer", "abstention_quality",
                "graph_retrieval_relevance", "answer_content_checks", "forbidden_claims_absent"]:
        lines.append(f"| {key.replace('_', ' ')} | {summary[key]} |")
    g = summary.get("gemini") or {}
    lines += [f"| Gemini success rate | {g.get('success_rate')} ({g.get('successes')}/{g.get('requests')} calls) |",
              f"| Gemini failures | {g.get('failures')} |",
              f"| Latency mean / p50 / p95 (s) | {summary['latency_s']['mean']} / {summary['latency_s']['p50']} / {summary['latency_s']['p95']} |",
              f"| Final status counts | {summary['status_counts']} |", "",
              "## Primary scenarios", "", "| Case | Status | Retrieved | Applicability | Content | Pass |",
              "|---|---|---|---|---|---|"]
    for r in results:
        if r["category"] != "primary":
            continue
        c = r["checks"]
        lines.append(f"| {r['id']} | {r['status']} | recall {c.get('recall')} | {c.get('applicability_accuracy')} | "
                     f"{c.get('content_include')} | {'PASS' if primary_pass(r) else 'FAIL'} |")
    lines += ["", "## All cases", "", "| Case | Category | Status | Recall | Precision | Applicability | Content | "
              "Forbidden absent | Draft hallucinations | Latency (s) |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        c = r["checks"]
        lines.append(f"| {r['id']} | {r['category']} | {r['status']} | {c.get('recall', '-')} | "
                     f"{round(c['precision'], 2) if 'precision' in c else '-'} | {c.get('applicability_accuracy', '-')} | "
                     f"{c.get('content_include', '-')} | {c.get('content_forbidden_absent', '-')} | "
                     f"{len(c.get('draft_hallucinations') or [])} | {r['latency_s']} |")
    failures = [r for r in results if not primary_pass(r)]
    if failures:
        lines += ["", "## Cases needing attention", ""]
        for r in failures:
            c = r["checks"]
            lines.append(f"- **{r['id']}** ({r['status']}): missing {c.get('missing_provisions')}, content groups "
                         f"not met {c.get('content_include_failed')}, draft hallucinations {c.get('draft_hallucinations')}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument("--cases", default="")
    parser.add_argument("--dataset", type=Path, default=BACKEND / "eval" / "dataset.json")
    parser.add_argument("--report", type=Path, default=PROJECT_DIR / "docs" / "EVALUATION_REPORT.md")
    args = parser.parse_args()
    configure_logging("WARNING", "text")
    settings = get_settings()
    init_db()
    from app.main import refresh_knowledge
    db = get_session_factory()()
    refresh_knowledge(db)
    dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
    wanted = {c.strip() for c in args.cases.split(",") if c.strip()}
    cases = [c for c in dataset["cases"] if not wanted or c["id"] in wanted]
    llm = None if args.no_llm else get_llm_provider()
    embedder = get_embedding_provider()
    run_id = uuid.uuid4().hex[:8]
    results = []
    for case in cases:
        session = f"eval-{run_id}-{case['id']}"[:64]
        if case.get("setup_upload"):
            doc = LegalDocument(title="uploaded-notice.pdf", document_type="user_upload", session_id=session,
                                ingestion_status="completed", verification_status="unverified", total_chunks=1)
            db.add(doc)
            db.flush()
            db.add(DocumentChunk(document_id=doc.id, content=case["setup_upload"], chunk_index=0, page_number=1,
                                 chunk_key=f"UPLOAD:{doc.id}:0", chunk_type="upload", status="unverified"))
            db.commit()
        trace = Trace(question=case["question"])
        started = time.perf_counter()
        payload = await ChatPipeline(db, llm=llm, embedder=embedder).run(case["question"], session, None, trace)
        result = score_case(case, payload, trace.as_dict(), time.perf_counter() - started)
        results.append(result)
        print(f"{case['id']:28s} {result['status']:24s} recall={result['checks'].get('recall', '-')} "
              f"applic={result['checks'].get('applicability_accuracy', '-')} "
              f"content={result['checks'].get('content_include', '-')} {result['latency_s']}s", flush=True)
    summary = aggregate(results, llm.usage.snapshot() if llm else None)
    summary["primary_scenarios_passed"] = sum(1 for r in results if r["category"] == "primary" and primary_pass(r))
    meta = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "mode": "no-llm (evidence-only)" if args.no_llm or llm is None else "gemini",
            "model": settings.gemini_model if llm else None, "fallbacks": settings.gemini_fallback_models,
            "classifier": settings.query_classifier if llm else "rules",
            "vector": f"{embedder.model} ({embedder.dimensions}d)" if embedder else "disabled",
            "dataset_version": dataset["_meta"]["version"]}
    out = BACKEND / "eval" / "results"
    out.mkdir(parents=True, exist_ok=True)
    (out / "latest.json").write_text(json.dumps({"meta": meta, "summary": summary, "results": results}, indent=1,
                                                ensure_ascii=False), encoding="utf-8")
    write_report(summary, results, meta, args.report)
    print(json.dumps(summary, indent=1))
    db.close()


if __name__ == "__main__":
    asyncio.run(main())
