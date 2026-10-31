"""Question-answering pipeline.

    user question
      -> follow-up context            (conversation history; summarised when long)
      -> intent classification        (rules, optionally validated Gemini classification)
      -> hybrid retrieval             (exact + graph + BM25 + vector, restrictions/exceptions expanded)
      -> applicability assessment     (State vs private actor, conditions, ordinary law, missing facts)
      -> answer generation            (Gemini, streamed or not, with evidence IDs)
      -> citation validation          (deterministic; one repair attempt)
      -> final answer                 (validated draft, or an evidence-only answer if validation fails
                                       or Gemini is unavailable - never an unsupported answer)

Every stage is recorded in a debug trace that administrators can retrieve by request ID.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, AsyncIterator

from sqlalchemy.orm import Session

from app.config import get_settings
from app.llm.base import GenerationRequest, LLMResponse
from app.llm.exceptions import LLMError
from app.llm.prompts import (CITATION_VERIFICATION_SYSTEM, CONSTITUTIONAL_SYSTEM, CONTEXT_SUMMARY_SYSTEM,
                             citation_verification_prompt, context_summary_prompt, final_answer_prompt,
                             render_assessment_block)
from app.models import Conversation, Message, QueryTrace
from app.observability import Trace, log_event
from app.reasoning.answer import (assessment_payload, assign_evidence_ids, citation_payload, evidence_block,
                                  evidence_only_answer)
from app.reasoning.applicability import Assessment, assess
from app.reasoning.citations import ValidationResult, validate_answer
from app.retrieval.hybrid import HybridRetriever, ProvisionStore, RetrievalResult, _rrf
from app.retrieval.query_understanding import QueryAnalysis, analyse

logger = logging.getLogger("app.pipeline")

SCOPE_MESSAGE = ("I can help with questions about the Constitution of India and how it relates to everyday legal "
                 "situations in India. Your question does not seem to involve a legal or constitutional issue. "
                 "Could you describe the legal situation or the provision you want to understand?")
FOLLOW_UP = re.compile(r"\b(that|this|it|they|them|those|there|same|former|latter|what about|how about|and if|"
                       r"what if|why|explain more|tell me more|elaborate)\b", re.I)


@dataclass
class Prepared:
    question: str
    conversation: Conversation
    analysis: QueryAnalysis
    result: RetrievalResult | None
    assessment: Assessment | None
    prompt: str | None
    evidence_text: str | None
    assessment_text: str | None
    trace: Trace
    context: str | None = None
    early_answer: str | None = None
    early_status: str | None = None
    extras: dict = field(default_factory=dict)


class ChatPipeline:
    def __init__(self, db: Session, llm=None, embedder=None):
        self.db = db
        self.llm = llm
        self.embedder = embedder
        self.settings = get_settings()

    # ------------------------------------------------------------------ preparation
    def _conversation(self, session_id: str, conversation_id: str | None, question: str) -> Conversation:
        conv = None
        if conversation_id:
            conv = self.db.query(Conversation).filter_by(id=conversation_id, session_id=session_id).first()
        if conv is None:
            conv = Conversation(session_id=session_id, title=question[:80])
            self.db.add(conv)
            self.db.commit()
            self.db.refresh(conv)
        return conv

    async def _context(self, conv: Conversation, question: str, trace: Trace) -> str | None:
        history = [m for m in conv.messages if m.role in {"user", "assistant"}]
        if not history:
            return None
        previous_users = [m.content for m in history if m.role == "user"]
        has_ref = bool(re.search(r"\b(?:article|art\.?)\s*\d", question, re.I))
        short = len(re.findall(r"\w+", question)) <= 8
        if has_ref or not previous_users or not (short or FOLLOW_UP.search(question)):
            return None
        if len(history) > 6 and self.llm is not None:
            try:
                response = await self.llm.generate(GenerationRequest(
                    prompt=context_summary_prompt([(m.role, m.content) for m in history[-8:]]),
                    system_instruction=CONTEXT_SUMMARY_SYSTEM, max_output_tokens=300, temperature=0.0,
                    purpose="context_summary"))
                trace.stage("context", method="llm_summary", turns=len(history))
                return response.text.strip()[:1200]
            except LLMError as exc:
                trace.stage("context", summary_error=exc.kind)
        trace.stage("context", method="previous_questions", turns=len(history))
        return " ".join(previous_users[-2:])[-1200:]

    async def prepare(self, question: str, session_id: str, conversation_id: str | None,
                      trace: Trace) -> Prepared:
        conv = self._conversation(session_id, conversation_id, question)
        context = await self._context(conv, question, trace)
        by_id, _ = ProvisionStore.load(self.db)
        with trace.timer("query_understanding"):
            analysis = await analyse(question, set(by_id), llm=self.llm, mode=self.settings.query_classifier,
                                     context=context)
        trace.stage("query_understanding", **analysis.as_dict())
        prepared = Prepared(question=question, conversation=conv, analysis=analysis, result=None, assessment=None,
                            prompt=None, evidence_text=None, assessment_text=None, trace=trace, context=context)
        if analysis.out_of_scope:
            prepared.early_answer, prepared.early_status = SCOPE_MESSAGE, "abstained"
            return prepared
        if not by_id:
            prepared.early_answer = ("The Constitution of India source text is not installed, so I cannot answer "
                                     "from verified text. An administrator needs to install the official PDF.")
            prepared.early_status = "abstained"
            return prepared
        query_text = f"{question}\n{context}" if context else question
        with trace.timer("retrieval"):
            result = await HybridRetriever(self.db, self.embedder).retrieve(analysis, query_text, session_id)
        assign_evidence_ids(result)
        trace.stage("retrieval", **result.diagnostics)
        trace.stage("source_extraction", evidence=[{"id": e.evidence_id, "provision": e.provision_id,
                                                    "verification": e.verification, "roles": e.roles,
                                                    "page": e.page} for e in result.all_evidence])
        assessment = assess(analysis, result)
        trace.stage("applicability", **assessment.as_dict())
        prepared.result, prepared.assessment = result, assessment
        if not result.constitution and not result.statutes:
            prepared.early_status = "abstained"
            prepared.early_answer = (
                "I could not find constitutional text that clearly relates to your question, so I will not guess "
                "an Article or a legal conclusion. " + (
                    f"It would help to know {analysis.missing_facts[0]}. " if analysis.missing_facts else "") +
                "Could you describe what happened, who was involved (for example the police, a government office, "
                "an employer or a private person), and what you want to know?")
            return prepared
        prepared.evidence_text = evidence_block(result)
        prepared.assessment_text = render_assessment_block(assessment_payload(assessment))
        prepared.prompt = final_answer_prompt(question, context, prepared.evidence_text, prepared.assessment_text)
        return prepared

    # ------------------------------------------------------------------ validation / repair
    async def _validate_and_repair(self, prepared: Prepared, draft: str) -> tuple[str, ValidationResult, dict]:
        evidence = prepared.result.all_evidence
        unknown = prepared.analysis.unknown_refs
        validation = validate_answer(draft, evidence, prepared.assessment, unknown)
        info: dict[str, Any] = {"initial": validation.as_dict(), "repaired": False}
        if validation.status != "failed" or self.llm is None:
            return draft, validation, info
        try:
            repaired = await self.llm.generate(GenerationRequest(
                prompt=citation_verification_prompt(draft, validation.errors, prepared.evidence_text,
                                                    prepared.assessment_text),
                system_instruction=CITATION_VERIFICATION_SYSTEM + "\n\n" + CONSTITUTIONAL_SYSTEM,
                temperature=0.0, purpose="citation_repair"))
            info["repair_usage"] = repaired.usage.as_dict()
            second = validate_answer(repaired.text, evidence, prepared.assessment, unknown)
            info["repaired"] = True
            info["after_repair"] = second.as_dict()
            if second.status != "failed":
                return repaired.text, second, info
            return repaired.text, second, info
        except LLMError as exc:
            info["repair_error"] = exc.kind
            return draft, validation, info

    # ------------------------------------------------------------------ confidence
    def _confidence(self, prepared: Prepared, validation: ValidationResult | None, status: str) -> dict:
        result, analysis = prepared.result, prepared.analysis
        if status in {"abstained", "error"} or result is None:
            return {"level": "low", "score": 0.0, "components": {}, "basis": "no verified answer"}
        s = self.settings
        if analysis.explicit_refs:
            found = sum(1 for r in analysis.explicit_refs
                        if any(e.provision_id == r.provision_id for e in result.constitution))
            retrieval = found / len(analysis.explicit_refs)
        elif result.graph.engaged:
            retrieval = 0.9
        else:
            top = max((e.scores.get("fused", 0) for e in result.constitution), default=0)
            retrieval = min(1.0, top / (2 * _rrf(1, s.retrieval_weight_bm25, s.retrieval_rrf_k)))
        cited = set(validation.cited_ids) if validation else set()
        relevant = [e for e in result.all_evidence if not cited or e.evidence_id in cited]
        verification = (sum(1 for e in relevant if e.verification == "verified") / len(relevant)) if relevant else 0.0
        validation_score = {"passed": 1.0, "passed_with_warnings": 0.8, "failed": 0.2}.get(
            validation.status if validation else "", 0.5)
        if status == "degraded":
            validation_score = 0.7       # evidence-only answer: grounded but not reasoned by the model
        facts = max(0.5, 1 - 0.08 * len(prepared.assessment.missing_facts if prepared.assessment else []))
        score = round(0.35 * retrieval + 0.25 * verification + 0.25 * validation_score + 0.15 * facts, 3)
        level = "high" if score >= 0.8 else "medium" if score >= 0.6 else "low"
        return {"level": level, "score": score, "basis": "retrieval quality, source verification, citation "
                "validation and missing facts (not model self-assessment)",
                "components": {"retrieval": round(retrieval, 3), "source_verification": round(verification, 3),
                               "citation_validation": validation_score, "fact_completeness": round(facts, 3)}}

    # ------------------------------------------------------------------ finalise
    def _finalise(self, prepared: Prepared, content: str, status: str, *, model: str | None = None,
                  llm_response: LLMResponse | None = None, validation: ValidationResult | None = None,
                  validation_info: dict | None = None, error: dict | None = None,
                  unverified_draft: str | None = None) -> dict:
        trace = prepared.trace
        result = prepared.result
        cited = set(validation.cited_ids) if validation else set()
        if status == "degraded" and result is not None:
            cited = set(re.findall(r"\[([CSU]\d+)\]", content))
        citations = citation_payload(result, cited) if result else []
        confidence = self._confidence(prepared, validation, status)
        assessment = prepared.assessment.as_dict() if prepared.assessment else None
        public_assessment = None
        if assessment:
            public_assessment = {k: assessment[k] for k in ("actor_type", "primary_domain", "constitutional_violation",
                                                             "missing_facts", "clarifying_question",
                                                             "evidence_sufficiency")}
            public_assessment["provisions"] = [{k: p[k] for k in ("evidence_id", "label", "verdict", "reason")}
                                               for p in assessment["provisions"]]
        metadata = {"citations": citations, "status": status, "validation": validation.as_dict() if validation else None,
                    "assessment": public_assessment, "confidence": confidence, "model": model,
                    "request_id": trace.request_id, "error": error, "unverified_draft": unverified_draft}
        conv = prepared.conversation
        user_msg = Message(conversation_id=conv.id, role="user", content=prepared.question)
        ai_msg = Message(conversation_id=conv.id, role="assistant", content=content, metadata_json=metadata)
        conv.updated_at = datetime.now(timezone.utc)
        self.db.add_all([user_msg, ai_msg])
        self.db.commit()
        self.db.refresh(ai_msg)
        trace.stage("generation", **{"model": model, **(llm_response.as_log() if llm_response else {}),
                                     "error": error})
        trace.stage("citation_validation", status=validation.status if validation else None,
                    **(validation_info or {}))
        trace.stage("final", status=status, confidence=confidence)
        self.db.merge(QueryTrace(request_id=trace.request_id, conversation_id=conv.id, status=status,
                                 trace=trace.as_dict()))
        self.db.commit()
        log_event(logger, "chat.answer", request_id=trace.request_id, status=status,
                  classification=prepared.analysis.categories, issues=prepared.analysis.issues,
                  retrieval_strategy=(result.diagnostics.get("strategy") if result else None),
                  articles=[e.provision_id for e in result.constitution] if result else [],
                  chunks=(result.diagnostics.get("chunk_ids", [])[:20] if result else []),
                  graph=(result.graph.summary() if result else None), model=model,
                  llm=llm_response.as_log() if llm_response else None,
                  validation=validation.status if validation else None, confidence=confidence["level"],
                  timings=trace.timings_ms)
        payload = {
            "message_id": ai_msg.id, "conversation_id": conv.id, "request_id": trace.request_id,
            "content": content, "status": status, "citations": citations, "validation": metadata["validation"],
            "assessment": public_assessment, "confidence": confidence, "model_used": model,
            "retrieval_method": "hybrid" if result else "none",
            "retrieval_strategy": result.diagnostics.get("strategy") if result else [],
            "knowledge_graph_used": bool(result and (result.graph.engaged or result.graph.expansions)),
            "sources_used": list(dict.fromkeys(c["source_title"] for c in citations if c.get("cited_in_answer"))),
            "error": error, "unverified_draft": unverified_draft,
        }
        return payload

    def _degraded(self, prepared: Prepared, error: LLMError | None, reason: str) -> dict:
        content = evidence_only_answer(prepared.result, prepared.assessment, prepared.analysis.issues, reason)
        return self._finalise(prepared, content, "degraded", error=error.to_dict() if error else
                              {"type": "llm_not_configured", "message": reason})

    def _after_validation(self, prepared: Prepared, text: str, validation: ValidationResult, info: dict,
                          response: LLMResponse | None) -> dict:
        if validation.status == "failed":
            content = evidence_only_answer(
                prepared.result, prepared.assessment, prepared.analysis.issues,
                "The AI-generated draft did not pass citation checks, so this answer is assembled only from the "
                "verified source text and rule-based analysis.")
            return self._finalise(prepared, content, "unverified", model=response.model if response else None,
                                  llm_response=response, validation=validation, validation_info=info,
                                  unverified_draft=text)
        status = "answered" if validation.status == "passed" else "answered_with_warnings"
        return self._finalise(prepared, text, status, model=response.model if response else None,
                              llm_response=response, validation=validation, validation_info=info)

    # ------------------------------------------------------------------ non-streaming
    async def run(self, question: str, session_id: str, conversation_id: str | None, trace: Trace) -> dict:
        prepared = await self.prepare(question, session_id, conversation_id, trace)
        if prepared.early_answer:
            return self._finalise(prepared, prepared.early_answer, prepared.early_status)
        if self.llm is None:
            return self._degraded(prepared, None, "AI generation is not configured on this server (GEMINI_API_KEY "
                                  "is missing), so this answer is assembled only from verified source text.")
        try:
            with trace.timer("generation"):
                response = await self.llm.generate(GenerationRequest(
                    prompt=prepared.prompt, system_instruction=CONSTITUTIONAL_SYSTEM, purpose="final_answer"))
        except LLMError as exc:
            return self._degraded(prepared, exc, f"The AI service is unavailable ({exc.user_message}) so this answer "
                                  "is assembled only from verified source text and rule-based analysis.")
        with trace.timer("validation"):
            text, validation, info = await self._validate_and_repair(prepared, response.text)
        return self._after_validation(prepared, text, validation, info, response)

    # ------------------------------------------------------------------ streaming
    async def stream(self, question: str, session_id: str, conversation_id: str | None,
                     trace: Trace) -> AsyncIterator[tuple[str, dict]]:
        yield "stage", {"stage": "understanding", "request_id": trace.request_id}
        prepared = await self.prepare(question, session_id, conversation_id, trace)
        yield "meta", {"conversation_id": prepared.conversation.id, "request_id": trace.request_id}
        if prepared.early_answer:
            yield "final", self._finalise(prepared, prepared.early_answer, prepared.early_status)
            return
        yield "stage", {"stage": "retrieved", "evidence": [e.evidence_id + ": " + e.label
                                                           for e in prepared.result.all_evidence][:16]}
        if self.llm is None:
            yield "final", self._degraded(prepared, None, "AI generation is not configured on this server, so this "
                                          "answer is assembled only from verified source text.")
            return
        yield "stage", {"stage": "generating"}
        parts: list[str] = []
        response: LLMResponse | None = None
        try:
            with trace.timer("generation"):
                async for chunk in self.llm.stream(GenerationRequest(
                        prompt=prepared.prompt, system_instruction=CONSTITUTIONAL_SYSTEM, purpose="final_answer")):
                    if chunk.done:
                        response = chunk.response
                    elif chunk.text:
                        parts.append(chunk.text)
                        yield "token", {"text": chunk.text}
        except LLMError as exc:
            yield "final", self._degraded(prepared, exc, f"The AI service failed while answering ({exc.user_message}) "
                                          "so this answer is assembled only from verified source text.")
            return
        yield "stage", {"stage": "validating"}
        draft = response.text if response else "".join(parts)
        with trace.timer("validation"):
            text, validation, info = await self._validate_and_repair(prepared, draft)
        yield "final", self._after_validation(prepared, text, validation, info, response)
