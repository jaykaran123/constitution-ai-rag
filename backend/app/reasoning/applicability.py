"""Constitutional applicability assessment.

Answers, before any text is generated:

1. Is the issue directly governed by the Constitution, or primarily ordinary law?
2. Which retrieved provisions could be engaged, and on what condition?
3. Is the alleged conduct attributable to the State (Article 12) or to a private actor?
4. Does the provision bind private actors (e.g. Articles 15(2), 17, 23, 24) or only the State?
5. Which restrictions/exceptions apply, and which facts are missing?

The result is deterministic, computed from the provision annotations (whose quotes are
verified against the source text), the graph routing conditions and the stated facts.
It is passed to Gemini as a binding constraint and used by the citation validator.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.knowledge.statutes import load_knowledge
from app.retrieval.hybrid import Evidence, RetrievalResult
from app.retrieval.query_understanding import QueryAnalysis

NOT_DIRECT = "not_directly_applicable_to_private_actor"
MAY_ENGAGE = "may_be_engaged_if_facts_confirmed"
DEPENDS_ON_ACTOR = "depends_on_whether_actor_is_the_state"
APPLIES_ANY_ACTOR = "may_be_engaged_regardless_of_actor"
OUTSIDE_SCOPE = "outside_the_listed_scope_on_stated_facts"
CONDITIONAL = "only_if_condition_met"
EXPLANATORY = "explanatory"
EXCEPTION = "limits_or_exceptions"
REMEDY = "remedy_provision"
DEFINITION = "definition_or_context"
OMITTED = "omitted_provision"

CONDITION_PATTERNS = {
    "child": r"\b(child|children|minor|boy|girl|kid|aged? (?:[0-9]|1[0-3])\b|\b(?:[0-9]|1[0-3]) years? old)\b",
    "child_6_14": r"\b(child|children|son|daughter|kid|school)\b",
    "untouchability": r"\b(untouchab\w*|caste|dalit|scheduled caste)\b",
    "emergency": r"\bemergenc\w*\b",
    "public_employment": r"\b(government job|public (?:sector|employment|service)|recruitment|civil service|government employ\w*)\b",
    "state_funded_institution": r"\b(government|aided|state[- ]funded|public)\b",
    "minority_institution": r"\bminority\b",
    "arrest": r"\b(arrest\w*|detain\w*|custody|detention)\b",
    "life_or_personal_liberty": r"\b(personal liberty|life-threatening|threatened to kill|arrest\w*|detain\w*|custody|detention)\b",
    "accused": r"\b(accused|interrogat\w*|confess\w*|statement)\b",
    "public_function": r"\b(government|aided|statutory|public)\b",
}


@dataclass
class ProvisionVerdict:
    evidence_id: str
    label: str
    role: str
    addressee: str | None
    verdict: str
    reason: str

    def as_dict(self) -> dict:
        return self.__dict__.copy()


@dataclass
class Assessment:
    actor_type: str
    primary_domain: str
    constitutional_violation: str
    provisions: list[ProvisionVerdict] = field(default_factory=list)
    premise_check: list[dict] = field(default_factory=list)
    ordinary_law: list[str] = field(default_factory=list)
    missing_facts: list[str] = field(default_factory=list)
    clarifying_question: str | None = None
    evidence_sufficiency: str = "sufficient"
    notes: list[str] = field(default_factory=list)

    def verdict_for_article(self, article_number: str) -> list[str]:
        return [p.verdict for p in self.provisions
                if re.match(rf"Article {re.escape(article_number)}(\(|$)", p.label)]

    def as_dict(self) -> dict:
        return {"actor_type": self.actor_type, "primary_domain": self.primary_domain,
                "constitutional_violation": self.constitutional_violation,
                "provisions": [p.as_dict() for p in self.provisions], "premise_check": self.premise_check,
                "ordinary_law_matters": self.ordinary_law, "missing_facts": self.missing_facts,
                "clarifying_question": self.clarifying_question,
                "evidence_sufficiency": self.evidence_sufficiency, "notes": self.notes}


def _verdict(ev: Evidence, analysis: QueryAnalysis) -> tuple[str, str]:
    role = ev.best_role
    question = analysis.question
    if ev.status == "omitted":
        return OMITTED, "This provision has been omitted from the Constitution; it is historical, not current law."
    if role == "exception" or ev.addressee == "exception":
        return EXCEPTION, "Exception or limitation stated in the text."
    if role == "restriction":
        return EXCEPTION, "Permits reasonable restrictions by law on the related freedom."
    if ev.addressee == "remedy" or role == "remedy":
        return REMEDY, "Court remedy for enforcing rights; not itself a right that was breached."
    if ev.addressee in {"definition", "directive"} or role == "context":
        return DEFINITION, ev.applicability_note or "Definition or context for the analysis."
    # Conditions attached by the routing graph (e.g. only for children, only during an Emergency)
    for condition in ev.conditions:
        pattern = CONDITION_PATTERNS.get(condition or "")
        if pattern and not re.search(pattern, question, re.I):
            return CONDITIONAL, f"Relevant only if this condition applies: {condition.replace('_', ' ')}."
    actor = analysis.actor_type
    if ev.addressee == "state_and_private":
        if ev.scope_terms:
            if any(term in question.lower() for term in ev.scope_terms):
                return APPLIES_ANY_ACTOR, "Not limited to the State, and the facts concern a listed matter."
            return OUTSIDE_SCOPE, ("Not limited to the State, but it covers only the matters listed in its text; "
                                   "the stated facts do not concern any of them.")
        return APPLIES_ANY_ACTOR, "The text is not limited to the State, so private conduct can engage it."
    if ev.addressee in {"state", "arresting_authority"}:
        if actor == "private":
            return NOT_DIRECT, ("The text restrains the State (see Article 12); the conduct described is by a "
                                "private person or body.")
        if actor == "state":
            return MAY_ENGAGE, "The conduct is attributed to a State authority; whether it breaches the provision depends on facts and law."
        if actor == "mixed":
            return MAY_ENGAGE, "Relevant only to the conduct of the State authority involved, not the private actor."
        if not analysis.is_incident:
            return EXPLANATORY, "General explanation; no specific incident was described."
        return DEPENDS_ON_ACTOR, "Applies only if the actor is 'the State' under Article 12, which the facts do not establish."
    return EXPLANATORY, "Retrieved for explanation."


def assess(analysis: QueryAnalysis, result: RetrievalResult) -> Assessment:
    kb = load_knowledge()
    issues = {i["id"]: i for i in kb["legal_issues"]}
    matched = [issues[i] for i in analysis.issues if i in issues]
    ordinary_primary = any(i["ordinary_law_primary"] for i in matched)
    offences = {o["id"]: o["name"] for o in kb["offences"]}
    ordinary_law = sorted({offences[o] for i in matched for o in i.get("offences", []) if o in offences
                           and not o.endswith("_offence")})

    verdicts: list[ProvisionVerdict] = []
    for ev in result.constitution:
        verdict, reason = _verdict(ev, analysis)
        verdicts.append(ProvisionVerdict(evidence_id=ev.evidence_id, label=ev.label, role=ev.best_role,
                                         addressee=ev.addressee, verdict=verdict, reason=reason))

    # An Article named as a whole takes its verdict from its retrieved clauses (e.g. "Is Article 15 violated?"
    # is answered by 15(1) binding only the State and 15(2) covering only listed matters).
    for v in verdicts:
        if v.verdict != EXPLANATORY or not re.fullmatch(r"Article \d{1,3}[A-Z]{0,3}", v.label):
            continue
        parts = [c for c in verdicts if c.label.startswith(v.label + "(") and c.verdict in
                 {MAY_ENGAGE, APPLIES_ANY_ACTOR, NOT_DIRECT, DEPENDS_ON_ACTOR, OUTSIDE_SCOPE, CONDITIONAL}]
        if not parts:
            continue
        if any(c.verdict in {MAY_ENGAGE, APPLIES_ANY_ACTOR} for c in parts):
            v.verdict, v.reason = MAY_ENGAGE, "At least one of its clauses may be engaged on these facts (see clause verdicts)."
        elif all(c.verdict in {NOT_DIRECT, OUTSIDE_SCOPE, CONDITIONAL} for c in parts):
            v.verdict = NOT_DIRECT
            v.reason = "None of its retrieved clauses applies directly on these facts: " + "; ".join(
                f"{c.label} {c.reason[0].lower()}{c.reason[1:]}" for c in parts)[:600]
        else:
            v.verdict, v.reason = DEPENDS_ON_ACTOR, "Depends on whether the actor is the State (see clause verdicts)."

    substantive = [v for v in verdicts if v.verdict in {MAY_ENGAGE, APPLIES_ANY_ACTOR, NOT_DIRECT, DEPENDS_ON_ACTOR,
                                                        OUTSIDE_SCOPE, CONDITIONAL}]
    if not analysis.is_incident:
        violation = "not_a_violation_question"
        domain = "general_information"
    elif substantive and all(v.verdict in {NOT_DIRECT, OUTSIDE_SCOPE, CONDITIONAL} for v in substantive):
        violation = "unlikely_on_stated_facts"
        domain = "ordinary_law" if ordinary_primary or analysis.actor_type == "private" else "constitutional"
    elif any(v.verdict in {MAY_ENGAGE, APPLIES_ANY_ACTOR} for v in substantive):
        violation = "possible_but_cannot_be_determined_without_more_facts"
        domain = "mixed" if ordinary_primary or ordinary_law else "constitutional"
    elif any(v.verdict == DEPENDS_ON_ACTOR for v in substantive):
        violation = "depends_on_whether_the_actor_is_the_state"
        domain = "mixed" if ordinary_primary else "constitutional"
    else:
        violation = "cannot_be_determined"
        domain = "ordinary_law" if ordinary_primary else "general_information"
    if ordinary_primary and analysis.actor_type == "private":
        domain = "ordinary_law"

    premise = []
    if analysis.asks_violation:
        for ref in analysis.explicit_refs:
            related = [v for v in verdicts if v.label == ref.label or v.label.startswith(f"Article {ref.article}(")
                       or v.label == f"Article {ref.article}"]
            if related:
                premise.append({"provision": ref.label, "verdicts": sorted({v.verdict for v in related}),
                                "assessment": related[0].reason})
            else:
                premise.append({"provision": ref.label, "verdicts": ["not_retrieved"],
                                "assessment": "This provision was not found in the retrieved text."})
    notes = []
    if analysis.unknown_refs:
        notes.append(f"These references were not found in the Constitution text as cited: {', '.join(analysis.unknown_refs)}. "
                     "Point this out to the user.")
    if any(v.verdict == NOT_DIRECT for v in verdicts):
        notes.append("Explain that most Fundamental Rights restrain 'the State' as defined in Article 12, and that "
                     "whether a body counts as the State depends on facts. Do not say a private actor violated them.")
    if ordinary_law:
        notes.append(f"The facts may involve ordinary criminal law ({', '.join(ordinary_law)}); describe these as "
                     "separate from constitutional rights.")
    if any(e.verification != "verified" for e in result.statutes):
        notes.append("Statutory text marked unverified must be described as not verified against the official text.")
    if any(v.verdict == OMITTED for v in verdicts):
        notes.append("Explain that omitted provisions are no longer part of the current text and give the amendment.")
    limits = [f"{v.label} [{v.evidence_id}]" for v in verdicts if v.verdict == EXCEPTION]
    if limits:
        notes.append("Explain these restrictions or exceptions, citing them: " + ", ".join(limits) + ".")
    if result.uploads:
        notes.append("Evidence marked [U…] comes from the user's own upload and is not authoritative. If it conflicts "
                     "with the Constitution evidence, say explicitly that the uploaded document is not authoritative "
                     "and state what the official text shows.")
    if result.remedies:
        notes.append("Describe the practical remedies listed under REMEDY OPTIONS and cite their basis IDs.")
    if not result.constitution:
        sufficiency = "insufficient"
    elif analysis.missing_facts or any(e.verification != "verified" for e in result.statutes):
        sufficiency = "partial"
    else:
        sufficiency = "sufficient"
    question = None
    if analysis.missing_facts and analysis.issues:
        question = f"Could you tell me {analysis.missing_facts[0]}?"
    return Assessment(actor_type=analysis.actor_type, primary_domain=domain, constitutional_violation=violation,
                      provisions=verdicts, premise_check=premise, ordinary_law=ordinary_law,
                      missing_facts=analysis.missing_facts[:6], clarifying_question=question,
                      evidence_sufficiency=sufficiency, notes=notes)
