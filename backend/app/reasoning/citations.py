"""Deterministic citation and answer validation.

Checks every generated answer before it is shown as a legal answer:

* every evidence tag ([C1], [S2], [U1]) exists;
* every Article mentioned was retrieved (no Article from model memory);
* every statutory section mentioned is in the statutory evidence;
* every quotation of 25+ characters appears verbatim (after normalisation) in the evidence;
* no sentence asserts a violation definitively, and none contradicts the applicability
  assessment (e.g. "Article 21 may have been violated" when the actor is private);
* legal statements carry citations.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.knowledge.text_utils import normalise_for_match
from app.reasoning.applicability import NOT_DIRECT, OUTSIDE_SCOPE, Assessment
from app.retrieval.hybrid import Evidence

TAG = re.compile(r"\[((?:[CSU]\d+)(?:\s*,\s*[CSU]\d+)*)\]")
ARTICLE_MENTION = re.compile(
    r"\bArt(?:icles?|s?\.)\s+((?:\d{1,3}[A-Z]{0,3}(?:\s*\(\s*\w{1,3}\s*\))*)"
    r"(?:\s*(?:,|and|or|to|&)\s*\d{1,3}[A-Z]{0,3}(?:\s*\(\s*\w{1,3}\s*\))*)*)")
SECTION_MENTION = re.compile(r"\b(?:[Ss]ections?|[Ss]s?\.)\s+(\d{1,3}[A-Z]?)(?:\s*\(\s*\w{1,3}\s*\))*")
# Pair quotes of any length first (curly pairs, or straight quotes on one line) and filter by length
# afterwards; requiring a minimum length inside the regex mis-pairs a closing quote of a short term
# ("the State") with the opening quote of the next one.
QUOTE = re.compile(r"“([^”]{1,800})”|\"([^\"\n]{1,800})\"")
VIOLATION = re.compile(r"\b(violat\w*|infring\w*|breach\w*|unconstitutional)\b", re.I)
# An *affirmative* claim that a right was breached ("has been violated", "this violates your right",
# "is a clear violation of", "is unconstitutional"). Mentions such as "rather than an automatic
# constitutional violation" or "whether a violation occurred" are not claims.
CLAIM = re.compile(
    r"\b(?:has|have|was|were|is|are)\s+(?:clearly\s+|definitely\s+|certainly\s+|thus\s+|therefore\s+)?"
    r"(?:been\s+)?(?:violated|infringed|breached)\b"
    r"|\b(?:violated|violates|infringed|infringes|breached|breaches)\s+(?:your|my|his|her|their|the|article)\b"
    r"|\bis\s+(?:a\s+)?(?:clear\s+|direct\s+|definite\s+)?(?:violation|infringement|breach)\s+of\b"
    r"|\b(?:is|was|are|were)\s+(?:clearly\s+)?unconstitutional\b", re.I)
HEDGE = re.compile(r"\b(may|might|could|possibl\w*|potential\w*|whether|if|unless|would|depend\w*|unlikely|"
                   r"cannot|can't|not|no|only|question|ask|assess\w*|determine\w*|rather than|instead|generally|"
                   r"ordinarily|primarily|automatic\w*)\b", re.I)
NEGATION = re.compile(r"\b(not|no|unlikely|does not|doesn't|isn't|is not|cannot|can't|without|rather than|"
                      r"instead|rarely|generally not|would not|wouldn't)\b", re.I)


@dataclass
class ValidationResult:
    status: str = "passed"                 # passed | passed_with_warnings | failed
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    cited_ids: list[str] = field(default_factory=list)
    articles_mentioned: list[str] = field(default_factory=list)
    quotes_checked: int = 0
    quotes_verified: int = 0
    legal_sentences: int = 0
    cited_legal_sentences: int = 0

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def _split_sentences(text: str) -> list[str]:
    text = re.sub(r"\n+", "\n", text)
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z\[*(\"“])|\n", text)
    return [p.strip() for p in parts if p.strip()]


def _article_numbers(mention: str) -> list[tuple[str, list[str]]]:
    out = []
    for match in re.finditer(r"(\d{1,3}[A-Z]{0,3})((?:\s*\(\s*\w{1,3}\s*\))*)", mention):
        parts = re.findall(r"\(\s*(\w{1,3})\s*\)", match.group(2))
        out.append((match.group(1), parts))
    return out


def validate_answer(answer: str, evidence: list[Evidence], assessment: Assessment | None = None,
                    user_unknown_refs: list[str] | None = None) -> ValidationResult:
    """``user_unknown_refs``: provisions the user cited that do not exist (e.g. "Article 500"); the
    answer may mention them only to say so (the sentence must contain a negation)."""
    result = ValidationResult()
    unknown_numbers = {re.match(r"Article (\d{1,3}[A-Z]{0,3})", r).group(1) for r in (user_unknown_refs or [])
                       if re.match(r"Article \d", r)}
    ids = {e.evidence_id: e for e in evidence if e.evidence_id}
    constitution = [e for e in evidence if e.kind == "constitution"]
    statutes = [e for e in evidence if e.kind == "statute"]
    articles_in_evidence: dict[str, set[str]] = {}
    for e in constitution:
        m = re.match(r"Article (\d{1,3}[A-Z]{0,3})((?:\(\w+\))*)", e.label)
        if m:
            articles_in_evidence.setdefault(m.group(1), set()).add(m.group(2))
        for ref in re.findall(r"\barticles?\s+(\d{1,3}[A-Z]{0,3})", e.text, re.I):
            # Articles named inside retrieved text may be mentioned (they are part of the evidence).
            articles_in_evidence.setdefault(ref.upper(), set())
    section_numbers = set()
    for e in statutes:
        m = re.search(r"s\.(\d{1,3}[A-Z]?)", e.label)
        if m:
            section_numbers.add(m.group(1))
    for e in evidence:
        # Sections named inside evidence text (e.g. an amendment footnote "… Amendment) Act, 1978, s. 6")
        for ref in re.findall(r"\b(?:sections?|ss?\.)\s*(\d{1,3}[A-Z]?)\b", e.text, re.I):
            section_numbers.add(ref)

    # 1. tags exist
    for match in TAG.finditer(answer):
        for tag in re.split(r"\s*,\s*", match.group(1)):
            result.cited_ids.append(tag)
            if tag not in ids:
                result.errors.append(f"Citation [{tag}] does not refer to any retrieved evidence.")
    result.cited_ids = sorted(set(result.cited_ids), key=lambda t: (t[0], int(t[1:])))

    # 2. articles mentioned were retrieved
    sentences_by_pos = [(m.start(), m.end()) for m in re.finditer(r"[^.!?\n]+[.!?]?", answer)]
    for match in ARTICLE_MENTION.finditer(answer):
        for number, parts in _article_numbers(match.group(1)):
            number = number.upper()
            label = f"Article {number}" + "".join(f"({p})" for p in parts)
            result.articles_mentioned.append(label)
            if number in unknown_numbers and number not in articles_in_evidence:
                sentence = next((answer[s:e] for s, e in sentences_by_pos if s <= match.start() < e), "")
                if not re.search(r"\b(not|no|does not|doesn't|isn't|nonexistent|non-existent|exist\w*|only|"
                                 r"ends? at|up to|incorrect|mistaken)\b", sentence, re.I):
                    result.errors.append(f"{label} (cited by the user) is described without saying it does not exist.")
                continue
            if number not in articles_in_evidence:
                result.errors.append(f"{label} is mentioned but was not retrieved or verified.")
            elif parts:
                clause_suffix = "".join(f"({p})" for p in parts)
                have = articles_in_evidence[number]
                if have and "" not in have and not any(h.startswith(clause_suffix) or clause_suffix.startswith(h)
                                                       for h in have if h):
                    result.warnings.append(f"{label}: this clause was not itself retrieved; only "
                                           f"{', '.join('Article ' + number + h for h in sorted(have))} were.")
    result.articles_mentioned = sorted(set(result.articles_mentioned))

    # 3. sections
    for match in SECTION_MENTION.finditer(answer):
        if match.group(1) not in section_numbers:
            result.errors.append(f"Section {match.group(1)} is mentioned but is not in the statutory evidence.")

    # 4. quotations
    evidence_text = [normalise_for_match(e.text) for e in evidence]
    for match in QUOTE.finditer(answer):
        raw = (match.group(1) or match.group(2) or "").strip(" .,;…")
        quote = normalise_for_match(raw)
        if len(quote) < 25:
            continue
        result.quotes_checked += 1
        # Allow an ellipsis inside a quotation: every fragment must appear in the same evidence item.
        fragments = [normalise_for_match(f) for f in re.split(r"\s*(?:\.\.\.|…)\s*", raw) if len(f.strip()) >= 8]
        if any(quote in text for text in evidence_text) or (
                len(fragments) > 1 and any(all(f in text for f in fragments) for text in evidence_text)):
            result.quotes_verified += 1
        else:
            result.errors.append(f"Quotation not found verbatim in the evidence: \"{raw[:80]}…\"")

    # 5. violation claims
    not_direct = set()
    if assessment:
        for verdict in assessment.provisions:
            if verdict.verdict in {NOT_DIRECT, OUTSIDE_SCOPE}:
                m = re.match(r"Article (\d{1,3}[A-Z]{0,3})", verdict.label)
                if m:
                    not_direct.add(m.group(1))
        # An Article counts as "not directly applicable" only if none of its retrieved parts may be engaged.
        engaged = {re.match(r"Article (\d{1,3}[A-Z]{0,3})", v.label).group(1) for v in assessment.provisions
                   if v.verdict not in {NOT_DIRECT, OUTSIDE_SCOPE} and re.match(r"Article \d", v.label)
                   and v.role not in {"context", "restriction", "exception", "remedy"}}
        not_direct -= engaged
    for sentence in _split_sentences(answer):
        if not VIOLATION.search(sentence) or sentence.startswith("#"):
            continue
        if CLAIM.search(sentence) and not HEDGE.search(sentence):
            result.errors.append(f"Definitive violation claim without qualification: \"{sentence[:120]}\"")
            continue
        mentioned = {n.upper() for m in ARTICLE_MENTION.finditer(sentence) for n, _ in _article_numbers(m.group(1))}
        conflict = mentioned & not_direct
        if conflict and not NEGATION.search(sentence):
            result.errors.append(f"Suggests a violation of Article {', '.join(sorted(conflict))}, which the "
                                 f"applicability assessment found not directly applicable: \"{sentence[:120]}\"")

    # 6. legal statements carry citations
    paragraphs = [p for p in re.split(r"\n\s*\n", answer) if p.strip()]
    for paragraph in paragraphs:
        for sentence in _split_sentences(paragraph):
            if sentence.startswith("#"):
                continue
            if ARTICLE_MENTION.search(sentence) or SECTION_MENTION.search(sentence):
                result.legal_sentences += 1
                if TAG.search(sentence) or TAG.search(paragraph):
                    result.cited_legal_sentences += 1
    uncited = result.legal_sentences - result.cited_legal_sentences
    if uncited:
        message = f"{uncited} sentence(s) mention a provision without an evidence citation."
        if result.legal_sentences and uncited / result.legal_sentences > 0.5:
            result.errors.append(message)
        else:
            result.warnings.append(message)
    if evidence and not result.cited_ids and len(answer) > 300:
        result.errors.append("The answer does not cite any evidence.")

    result.status = "failed" if result.errors else ("passed_with_warnings" if result.warnings else "passed")
    return result
