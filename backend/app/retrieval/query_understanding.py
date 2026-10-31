"""Query understanding: classify the question before any retrieval happens.

Rule-based analysis always runs (deterministic, testable). In ``hybrid`` mode Gemini also
classifies the scenario, but its output is validated against the closed list of known
legal issues and categories; it can never introduce an Article number.
"""
from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass, field

from app.config import get_settings
from app.knowledge.statutes import load_knowledge

logger = logging.getLogger("app.retrieval.query")

CATEGORIES = [
    "constitutional_article_lookup", "fundamental_rights", "everyday_incident", "police_and_arrest",
    "discrimination", "employment_forced_labour", "education", "freedom_of_expression", "criminal_offence",
    "constitutional_interpretation", "legal_remedies", "multi_issue", "out_of_scope",
]

# "government" as the *subject* of criticism ("criticised a government policy") is not an actor.
STATE_ACTORS = r"police|cops?|constable|inspector|officer|" \
               r"(?:government|govt)(?!\s+(?:polic\w*|schemes?|decisions?|laws?|budget|programmes?|programs?))|" \
               r"municipal\w*|corporation|collector|" \
               r"magistrate|tehsildar|ministry|department|panchayat|public (?:university|college|school|hospital|sector)|" \
               r"government[- ](?:school|college|hospital|office|university|employer|job)|state[- ]run|army|railways?|" \
               r"court|judge|official|authority|the state"
PRIVATE_ACTORS = r"landlord|landlady|owner|employer|boss|manager|company|firm|private (?:college|school|university|" \
                 r"hospital|company|employer|institution|bank)|neighbou?r|someone|somebody|stranger|thief|thieves|" \
                 r"shopkeeper|shop|restaurant|hotel|relative|husband|wife|friend|colleague|contractor|society|builder|" \
                 r"a man|a woman|a person|people|gang|landowner"
INSTITUTIONS = r"college|university|school|hospital|institute|institution"
INDIAN_STATES = ["Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh", "Goa", "Gujarat", "Haryana",
                 "Himachal Pradesh", "Jharkhand", "Karnataka", "Kerala", "Madhya Pradesh", "Maharashtra", "Manipur",
                 "Meghalaya", "Mizoram", "Nagaland", "Odisha", "Punjab", "Rajasthan", "Sikkim", "Tamil Nadu",
                 "Telangana", "Tripura", "Uttar Pradesh", "Uttarakhand", "West Bengal", "Delhi", "Jammu and Kashmir",
                 "Ladakh", "Puducherry", "Chandigarh", "Mumbai", "Bengaluru", "Bangalore", "Chennai", "Kolkata",
                 "Hyderabad", "Pune"]
LEGAL_TERMS = r"\b(law|legal|right|rights|constitution\w*|court|police|article|act|section|illegal|complain\w*|case|" \
              r"employ\w*|factory|labou?r|wages?|landlord|tenant|evict\w*|stole\w*|stolen|theft|beat\w*|threat\w*|" \
              r"lawyer|advocate|fir|arrest|government|discriminat\w*|wage|salary|fundamental|duty|duties|writ|" \
              r"amend\w*|schedule|parliament|president|governor|citizen\w*|vote|election|freedom|liberty|equality)\b"
VIOLATION_Q = r"\b(violat\w*|infring\w*|breach\w*|unconstitutional|against my (?:fundamental )?rights?|my rights? (?:been|was|were)|is (?:it|this) (?:legal|illegal|allowed))\b"
REMEDY_Q = r"\b(what (?:can|should) i do|remed\w*|where (?:can|do|should) i (?:complain|go|file)|how (?:can|do) i (?:file|complain|report)|next steps?|whom to approach|help me)\b"

# Issue suppression: when the left issue matched, the right ones are keyword false positives.
SUPPRESSES = {
    "discrimination_private_actor": {"religious_freedom", "discrimination_by_state", "education_access"},
    "discrimination_by_state": {"religious_freedom"},
    "expression_private_institution": {"expression_state_restriction", "education_access", "assembly_protest"},
    "forced_labour": {"public_employment_equality"},
    "private_property_crime": {"property_deprivation_state"},
}


@dataclass
class ProvisionRef:
    article: str
    clause: str | None = None
    subclause: str | None = None
    raw: str = ""

    @property
    def provision_id(self) -> str:
        pid = f"COI:art:{self.article}"
        if self.clause:
            pid += f":cl:{self.clause}"
        if self.subclause:
            pid += f":sub:{self.subclause}"
        elif not self.clause and self.subclause:
            pid += f":sub:{self.subclause}"
        return pid

    @property
    def label(self) -> str:
        return f"Article {self.article}" + (f"({self.clause})" if self.clause else "") + \
            (f"({self.subclause})" if self.subclause else "")


@dataclass
class QueryAnalysis:
    question: str
    categories: list[str] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)
    issue_sources: dict[str, str] = field(default_factory=dict)
    explicit_refs: list[ProvisionRef] = field(default_factory=list)
    unknown_refs: list[str] = field(default_factory=list)
    actor_type: str = "unknown"          # state | private | mixed | unknown | not_applicable
    state_actors: list[str] = field(default_factory=list)
    private_actors: list[str] = field(default_factory=list)
    institution_status_unknown: list[str] = field(default_factory=list)
    jurisdiction: str | None = None
    asks_violation: bool = False
    asks_remedy: bool = False
    incident: str | None = None
    missing_facts: list[str] = field(default_factory=list)
    out_of_scope: bool = False
    is_incident: bool = False            # describes something that happened to the user (vs. a conceptual question)
    classifier: str = "rules"
    llm_error: str | None = None

    def as_dict(self) -> dict:
        data = asdict(self)
        data["explicit_refs"] = [r.label for r in self.explicit_refs]
        data.pop("question", None)
        return data


REF_PATTERN = re.compile(
    r"\b(?:articles?|arts?\.?|art)\s*((?:\d{1,3}[A-Za-z]{0,3}(?:\s*\(\s*\w{1,3}\s*\))*)"
    r"(?:\s*(?:,|and|&|or|/)\s*\d{1,3}[A-Za-z]{0,3}(?:\s*\(\s*\w{1,3}\s*\))*)*)", re.I)
SINGLE_REF = re.compile(r"(\d{1,3}[A-Za-z]{0,3})((?:\s*\(\s*\w{1,3}\s*\))*)")


def parse_refs(text: str) -> list[ProvisionRef]:
    refs: list[ProvisionRef] = []
    for match in REF_PATTERN.finditer(text):
        for single in SINGLE_REF.finditer(match.group(1)):
            number = single.group(1)
            # "21a" -> "21A"; but avoid swallowing words like "21st"
            if re.fullmatch(r"\d+(st|nd|rd|th)", number, re.I):
                number = re.match(r"\d+", number).group(0)
            article = number.upper()
            parts = re.findall(r"\(\s*(\w{1,3})\s*\)", single.group(2))
            clause = sub = None
            for part in parts:
                if re.fullmatch(r"\d+[A-Za-z]?", part) and clause is None and sub is None:
                    clause = part.upper()
                elif re.fullmatch(r"[a-z]{1,2}", part) and sub is None:
                    sub = part
            refs.append(ProvisionRef(article=article, clause=clause, subclause=sub, raw=single.group(0).strip()))
    unique: dict[str, ProvisionRef] = {}
    for ref in refs:
        unique.setdefault(ref.provision_id, ref)
    return list(unique.values())


def _find_all(pattern: str, text: str) -> list[str]:
    return sorted({m.group(0).lower() for m in re.finditer(rf"\b(?:{pattern})\b", text, re.I)})


def analyse_rules(question: str, known_provisions: set[str] | None = None) -> QueryAnalysis:
    kb = load_knowledge()
    analysis = QueryAnalysis(question=question)
    text = question.strip()
    analysis.explicit_refs = parse_refs(text)
    if known_provisions is not None:
        valid = []
        for ref in analysis.explicit_refs:
            if ref.provision_id in known_provisions:
                valid.append(ref)
            elif f"COI:art:{ref.article}" in known_provisions:
                analysis.unknown_refs.append(ref.label)
                valid.append(ProvisionRef(article=ref.article, raw=ref.raw))   # fall back to the Article
            else:
                analysis.unknown_refs.append(ref.label)
        analysis.explicit_refs = valid

    matched: list[str] = []
    for issue in kb["legal_issues"]:
        if any(re.search(trigger, text, re.I) for trigger in issue["triggers"]):
            matched.append(issue["id"])
    suppressed = set().union(*(SUPPRESSES.get(i, set()) for i in matched)) if matched else set()
    analysis.issues = [i for i in matched if i not in suppressed]
    analysis.issue_sources = {i: "rules" for i in analysis.issues}

    analysis.state_actors = _find_all(STATE_ACTORS, text)
    analysis.private_actors = _find_all(PRIVATE_ACTORS, text)
    for inst in _find_all(INSTITUTIONS, text):
        if not re.search(rf"\b(private|government|govt|public|state|central|aided)\s+{inst}", text, re.I):
            analysis.institution_status_unknown.append(inst)
    if analysis.state_actors and analysis.private_actors:
        analysis.actor_type = "mixed"
    elif analysis.state_actors:
        analysis.actor_type = "state"
    elif analysis.private_actors:
        analysis.actor_type = "private"
    elif analysis.institution_status_unknown:
        analysis.actor_type = "unknown"
    elif not analysis.issues:
        analysis.actor_type = "not_applicable"

    analysis.jurisdiction = next((s for s in INDIAN_STATES if re.search(rf"\b{re.escape(s)}\b", text, re.I)), None)
    analysis.asks_violation = bool(re.search(VIOLATION_Q, text, re.I))
    analysis.asks_remedy = bool(re.search(REMEDY_Q, text, re.I))

    issue_by_id = {i["id"]: i for i in kb["legal_issues"]}
    categories = []
    for issue_id in analysis.issues:
        categories.append(issue_by_id[issue_id]["category"])
    if analysis.explicit_refs:
        categories.append("constitutional_article_lookup")
    if re.search(r"\b(i|my|me|we|our)\b", text, re.I) and analysis.issues and \
            any(not issue_by_id[i]["category"].startswith("constitutional") for i in analysis.issues):
        categories.append("everyday_incident")
    if analysis.asks_remedy:
        categories.append("legal_remedies")
    if len({issue_by_id[i]["category"] for i in analysis.issues}) > 1:
        categories.append("multi_issue")
    analysis.categories = list(dict.fromkeys(categories))
    analysis.is_incident = analysis.asks_violation or (
        bool(analysis.issues) and bool(re.search(r"\b(i|my|me|we|our|us|mine)\b", text, re.I))
        and bool(analysis.state_actors or analysis.private_actors or analysis.institution_status_unknown
                 or re.search(r"\b(was|were|got|has|have|had|did|is|are)\b", text, re.I)))
    for issue_id in analysis.issues:
        for fact in issue_by_id[issue_id].get("missing_facts", []):
            if fact not in analysis.missing_facts:
                analysis.missing_facts.append(fact)
    if "police_refusal_complaint" in analysis.issues and re.search(
            r"\b(police|officer|station)\b.{0,120}\b(refus\w*|declin\w*|would not|wouldn't|did not|didn't)\b"
            r".{0,100}\b(register|record|lodge|file|complaint|fir)\b", text, re.I):
        analysis.missing_facts = [fact for fact in analysis.missing_facts
                                  if not fact.lower().startswith("whether the police refused")]
    if analysis.institution_status_unknown:
        analysis.missing_facts.insert(0, f"whether the {analysis.institution_status_unknown[0]} is government-run, "
                                         f"government-aided or private")
    analysis.out_of_scope = not analysis.issues and not analysis.explicit_refs and not re.search(LEGAL_TERMS, text, re.I)
    if analysis.out_of_scope:
        analysis.categories = ["out_of_scope"]
    return analysis


CLASSIFIER_SCHEMA = {
    "type": "object",
    "properties": {
        "issues": {"type": "array", "items": {"type": "string"}},
        "categories": {"type": "array", "items": {"type": "string"}},
        "actor_type": {"type": "string", "enum": ["state", "private", "mixed", "unknown", "not_applicable"]},
        "incident_summary": {"type": "string"},
        "jurisdiction": {"type": "string"},
        "asks_violation": {"type": "boolean"},
        "missing_facts": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["issues", "categories", "actor_type", "incident_summary", "asks_violation", "missing_facts"],
}


async def analyse(question: str, known_provisions: set[str] | None, llm=None, mode: str = "rules",
                  context: str | None = None) -> QueryAnalysis:
    analysis = analyse_rules(question if not context else f"{context}\n{question}", known_provisions)
    analysis.question = question
    if context:
        # Explicit references come from the current question when it has any.
        current_refs = analyse_rules(question, known_provisions).explicit_refs
        if current_refs:
            analysis.explicit_refs = current_refs
    # In hybrid mode Gemini also looks at questions the keyword rules consider out of scope
    # (e.g. "employs a 12 year old boy in his factory" has no legal keyword but is a legal issue).
    if mode != "hybrid" or llm is None:
        return analysis
    from app.llm.base import GenerationRequest
    from app.llm.prompts import SCENARIO_CLASSIFICATION_SYSTEM, scenario_classification_prompt
    kb = load_knowledge()
    allowed_issues = {i["id"]: i["name"] for i in kb["legal_issues"]}
    try:
        response = await llm.generate(GenerationRequest(
            prompt=scenario_classification_prompt(question, context, allowed_issues, CATEGORIES),
            system_instruction=SCENARIO_CLASSIFICATION_SYSTEM, response_schema=CLASSIFIER_SCHEMA,
            temperature=0.0, max_output_tokens=800, purpose="scenario_classification",
            model=get_settings().gemini_classifier_model or None))
        data = response.parsed or {}
    except Exception as exc:  # noqa: BLE001 - classification falls back to rules
        analysis.llm_error = getattr(exc, "kind", type(exc).__name__)
        return analysis
    llm_issues = list(dict.fromkeys(i for i in data.get("issues", []) if i in allowed_issues))
    rule_issues = list(analysis.issues)
    if llm_issues and not rule_issues:
        # Rules found nothing: Gemini's classification (closed list only) fills the gap.
        analysis.issues = llm_issues
        analysis.issue_sources = {i: "llm" for i in llm_issues}
        analysis.out_of_scope = False
        if analysis.categories == ["out_of_scope"]:
            analysis.categories = []
    elif llm_issues and set(llm_issues) & set(rule_issues):
        # Agreement on at least one issue: accept Gemini's additional issues (multi-issue scenarios).
        merged = rule_issues + [i for i in llm_issues if i not in rule_issues]
        suppressed = set().union(*(SUPPRESSES.get(i, set()) for i in merged))
        analysis.issues = [i for i in merged if i not in suppressed or i in rule_issues]
        analysis.issue_sources = {i: ("rules+llm" if i in llm_issues and i in rule_issues else
                                      "rules" if i in rule_issues else "llm") for i in analysis.issues}
    elif llm_issues:
        # Disagreement: keep the rule-based issues (specific triggers) and record Gemini's view for debugging.
        analysis.issue_sources = {**analysis.issue_sources, "_llm_disagreed": ",".join(llm_issues)}
    if analysis.out_of_scope:
        return analysis
    analysis.categories = list(dict.fromkeys(
        [c for c in data.get("categories", []) if c in CATEGORIES] + analysis.categories))
    if analysis.actor_type in {"unknown", "not_applicable"} and data.get("actor_type") in {"state", "private", "mixed"}:
        analysis.actor_type = data["actor_type"]
    analysis.incident = (data.get("incident_summary") or "")[:400] or None
    if not analysis.jurisdiction and data.get("jurisdiction") and data["jurisdiction"].lower() not in {"india", "unknown", ""}:
        analysis.jurisdiction = data["jurisdiction"][:80]
    analysis.asks_violation = analysis.asks_violation or bool(data.get("asks_violation"))
    issue_by_id = {i["id"]: i for i in kb["legal_issues"]}
    facts = list(analysis.missing_facts)
    for issue_id in analysis.issues:
        for fact in issue_by_id[issue_id].get("missing_facts", []):
            if fact not in facts:
                facts.append(fact)
    for fact in data.get("missing_facts", [])[:4]:
        if isinstance(fact, str) and fact and fact not in facts:
            facts.append(fact[:200])
    analysis.missing_facts = facts[:8]
    analysis.classifier = "hybrid"
    return analysis
