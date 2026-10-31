"""Structure-aware parser for the official Constitution of India PDF.

The Legislative Department edition is laid out as:

* front matter (preface, table of contents) - skipped;
* the Preamble;
* Parts, optional Chapters, optional group sub-headings, Articles;
* Articles split into clauses ``(1)``, sub-clauses ``(a)``, provisos and Explanations;
* Schedules (the Seventh Schedule is split into its three Lists and their entries);
* Appendices.

Every page has a running header and per-page footnotes below a rule line. The footnotes
record amendments ("Subs. by the Constitution (Forty-fourth Amendment) Act, 1978 ...")
and are linked to numbered markers such as ``2[`` in the body text.

The parser never invents text. It only segments the source and records where each
segment came from (PDF page and printed page). Unrecognised layouts fall back to
coarser segments rather than guessed structure.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

PARSER_VERSION = "2.0.2"

ROMAN_PART = r"[IVXLC]+[A-Z]?"
SCHEDULE_ORDINALS = ["FIRST", "SECOND", "THIRD", "FOURTH", "FIFTH", "SIXTH", "SEVENTH", "EIGHTH",
                     "NINTH", "TENTH", "ELEVENTH", "TWELFTH"]

FOOTNOTE_RULE = re.compile(r"^\s*_{10,}\s*$")
MARKER = re.compile(r"(\d{1,2})\*?\[")                     # amendment marker such as 2[ or 3*[
STRIP_MARKERS = re.compile(r"(?<![\w(])\d{1,2}\*?\[")      # remove the number, keep the bracket
PART_HEADING = re.compile(rf"^\s*(?:\d{{1,2}}\*?\[)?\s*PART\s+({ROMAN_PART})\b\s*(.*)$")
CHAPTER_HEADING = re.compile(r"^\s*(?:\d{1,2}\*?\[)?\s*CHAPTER\s+([IVX]+[A-Z]?)\.?\s*[—–-]\s*(.*)$")
ARTICLE_HEADING = re.compile(r"^\s*((?:\d{1,2}\*?\[|\[)*)\s*(\d{1,3}[A-Z]{0,3})\.(?:\s+(\S.*)|\s*)$")
CLAUSE_START = re.compile(r"^\s*((?:\d{1,2}\*?\[)*)\s*\((\d{1,2}[A-Z]{0,2})\)\s*(.*)$")
SUBCLAUSE_START = re.compile(r"^\s*((?:\d{1,2}\*?\[)*)\s*\(([a-z]{1,2})\)\s*(.*)$")
SCHEDULE_HEADING = re.compile(rf"^\s*(?:\d{{1,2}}\*?\[)?\s*(?:THE\s+)?({'|'.join(SCHEDULE_ORDINALS)})\s+SCHEDULE\s*$")
APPENDIX_HEADING = re.compile(r"^\s*APPENDIX\s+([IVX]+)\s*$")
SEVENTH_LIST = re.compile(r"^\s*List\s+(I{1,3})\s*[—–-]\s*(Union|State|Concurrent)\s+List", re.I)
ENTRY_START = re.compile(r"^\s*((?:\d{1,2}\*?\[)*)\s*(\d{1,3}[A-Z]{0,2})\.\s+(\S.*)$")
ROMAN_ITEM = re.compile(r"^(?:i{1,3}|iv|v|vi{1,3}|ix|x|xi{1,3}|xiv|xv)$")
AMEND_ACT = re.compile(r"Constitution\s*\(([^)]*?Amendment)\)\s*Act,\s*(\d{4})", re.I)
AMEND_ACTION = re.compile(r"^\s*(Subs|Ins|Added|Omitted|Renumbered|Rep|Am|Substituted|Inserted)\b", re.I)
WEF = re.compile(r"w\.\s*e\.\s*f\.?\s*([\d]{1,2}-[\d]{1,2}-[\d]{4})")


@dataclass
class Footnote:
    number: str
    page: int
    text: str
    amendment_act: str | None = None
    action: str | None = None
    effective_from: str | None = None

    def as_dict(self) -> dict:
        return {"marker": self.number, "page": self.page, "note": self.text,
                "amendment_act": self.amendment_act, "action": self.action,
                "effective_from": self.effective_from}


@dataclass
class Line:
    text: str
    page: int
    printed_page: str | None


@dataclass
class Provision:
    """One structural unit of the Constitution with exact source text."""

    provision_id: str
    kind: str                       # preamble|part|chapter|article|clause|subclause|schedule|schedule_entry|appendix
    title: str | None
    text: str                       # exact source text, amendment markers preserved
    page_start: int
    page_end: int
    printed_page: str | None = None
    parent_id: str | None = None
    part_number: str | None = None
    part_title: str | None = None
    chapter: str | None = None
    chapter_title: str | None = None
    group_heading: str | None = None
    article_number: str | None = None
    clause_number: str | None = None
    subclause: str | None = None
    schedule: str | None = None
    status: str = "in_force"        # in_force|amended|omitted
    amendments: list[dict] = field(default_factory=list)
    ordinal: int = 0

    @property
    def clean_text(self) -> str:
        return clean_text(self.text)

    @property
    def label(self) -> str:
        return provision_label(self.kind, self.article_number, self.clause_number, self.subclause,
                               self.part_number, self.schedule, self.title)


@dataclass
class ParsedConstitution:
    provisions: list[Provision]
    footnotes: dict[int, dict[str, Footnote]]
    page_count: int
    parser_version: str = PARSER_VERSION

    def by_id(self) -> dict[str, Provision]:
        return {p.provision_id: p for p in self.provisions}

    def articles(self) -> list[Provision]:
        return [p for p in self.provisions if p.kind == "article"]


def clean_text(text: str) -> str:
    """Remove numeric amendment markers (``2[`` -> ``[``) for display; keep wording intact."""
    text = STRIP_MARKERS.sub("[", text)
    text = re.sub(r"(?<![\w(])\d{1,2}(?=\*)", "", text)   # "article 32 3***" -> "article 32 ***"
    return re.sub(r"[ \t]+", " ", text).strip()


def provision_label(kind: str, article: str | None, clause: str | None, sub: str | None,
                    part: str | None = None, schedule: str | None = None, title: str | None = None) -> str:
    if kind in {"article", "clause", "subclause"} and article:
        label = f"Article {article}"
        if clause:
            label += f"({clause})"
        if sub:
            label += f"({sub})"
        return label
    if kind == "part" and part:
        return f"Part {part}"
    if kind == "chapter":
        return f"Part {part}, {title or 'Chapter'}"
    if kind in {"schedule", "schedule_entry"}:
        return f"{schedule} Schedule" + (f", {title}" if title and kind == "schedule_entry" else "")
    if kind == "preamble":
        return "Preamble"
    return title or kind.title()


def article_sort_key(number: str) -> tuple[int, str]:
    match = re.fullmatch(r"(\d+)([A-Z]*)", number.upper())
    if not match:
        return (10_000, number)
    return (int(match.group(1)), match.group(2))


def _normalise(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


# --------------------------------------------------------------------------------------
# Page reading
# --------------------------------------------------------------------------------------

def _split_page(raw: str) -> tuple[list[str], list[str], str | None]:
    """Return (body lines, footnote lines, printed page number) without running headers."""
    lines = raw.splitlines()
    body: list[str] = []
    notes: list[str] = []
    in_notes = False
    for line in lines:
        if not in_notes and FOOTNOTE_RULE.match(line):
            in_notes = True
            continue
        (notes if in_notes else body).append(line)
    printed = None
    # Running header: "THE CONSTITUTION OF INDIA", "(Part III.—Fundamental Rights)", page number.
    cleaned: list[str] = []
    header_zone = True
    for index, line in enumerate(body):
        stripped = line.strip()
        if header_zone and index < 8:
            if not stripped:
                continue
            if re.fullmatch(r"THE CONSTITUTION OF\s+INDIA", stripped):
                continue
            if re.fullmatch(r"\((?:Part|Appendix|[A-Z][a-z]+ Schedule)[^)]*\)", stripped) or \
                    re.fullmatch(r"\([A-Z][a-z]+ Schedule\)", stripped):
                continue
            if re.fullmatch(r"\d{1,3}", stripped):
                printed = stripped
                continue
            header_zone = False
        cleaned.append(line.rstrip())
    return cleaned, notes, printed


def _parse_footnotes(lines: list[str], page: int, previous_act: str | None) -> tuple[dict[str, Footnote], str | None]:
    notes: dict[str, Footnote] = {}
    current: Footnote | None = None
    for line in lines:
        # Usually "2. Ins. by ...", occasionally printed without the period ("2 Ins. by ...").
        match = re.match(r"^\s*(\d{1,2})(?:\.\s+|\s+(?=[A-Z]))(.*)$", line)
        if match and (current is None or int(match.group(1)) == int(current.number) + 1):
            current = Footnote(number=match.group(1), page=page, text=match.group(2).strip())
            notes[current.number] = current
        elif current is not None and line.strip():
            current.text = f"{current.text} {line.strip()}"
    for note in notes.values():
        note.text = _normalise(note.text)
        act = AMEND_ACT.search(note.text)
        if act:
            note.amendment_act = f"Constitution ({_normalise(act.group(1))}) Act, {act.group(2)}"
            previous_act = note.amendment_act
        elif "ibid" in note.text.lower() and previous_act:
            note.amendment_act = previous_act
        action = AMEND_ACTION.match(note.text)
        if action:
            note.action = action.group(1).rstrip(".").title()
        elif "omitted" in note.text.lower():
            note.action = "Omitted"
        wef = WEF.search(note.text)
        if wef:
            note.effective_from = wef.group(1)
        elif "retrospective" in note.text.lower():
            note.effective_from = "with retrospective effect"
    return notes, previous_act


def read_pages(path: Path) -> tuple[list[tuple[int, list[str], str | None]], dict[int, dict[str, Footnote]], int]:
    import fitz  # PyMuPDF
    pages: list[tuple[int, list[str], str | None]] = []
    footnotes: dict[int, dict[str, Footnote]] = {}
    previous_act = None
    with fitz.open(path) as pdf:
        count = len(pdf)
        for index, page in enumerate(pdf, start=1):
            body, notes, printed = _split_page(page.get_text("text"))
            footnotes[index], previous_act = _parse_footnotes(notes, index, previous_act)
            pages.append((index, body, printed))
    return pages, footnotes, count


# --------------------------------------------------------------------------------------
# Section detection
# --------------------------------------------------------------------------------------

def _find_layout(pages: list[tuple[int, list[str], str | None]]) -> dict[str, int]:
    preamble = schedules = appendices = None
    for page, lines, _ in pages:
        joined = "\n".join(lines)
        if preamble is None and re.search(r"(?m)^\s*PREAMBLE\s*$", joined) and "WE, THE PEOPLE OF INDIA" in joined:
            preamble = page
        if preamble and schedules is None and page > preamble and \
                re.search(r"(?m)^\s*(?:\d{1,2}\[)?\s*FIRST SCHEDULE\s*$", joined) and "[Articles 1 and 4]" in joined:
            schedules = page
        if schedules and appendices is None and page > schedules and re.search(r"(?m)^\s*APPENDIX\s+I\s*$", joined):
            appendices = page
    if not preamble or not schedules:
        raise ValueError("The PDF does not have the expected Constitution of India layout "
                         "(Preamble and First Schedule headings were not found).")
    return {"preamble": preamble, "schedules": schedules, "appendices": appendices or len(pages) + 1}


def _lines(pages, start: int, end: int) -> list[Line]:
    out: list[Line] = []
    for page, body, printed in pages:
        if start <= page < end:
            out.extend(Line(text, page, printed) for text in body)
    return out


def _markers_in(lines: Iterable[Line], footnotes: dict[int, dict[str, Footnote]]) -> list[dict]:
    found: list[dict] = []
    seen: set[tuple[int, str]] = set()
    for line in lines:
        for match in MARKER.finditer(line.text):
            key = (line.page, match.group(1))
            note = footnotes.get(line.page, {}).get(match.group(1))
            if note and key not in seen:
                seen.add(key)
                found.append(note.as_dict())
    return found


# --------------------------------------------------------------------------------------
# Articles
# --------------------------------------------------------------------------------------

@dataclass
class _ArticleDraft:
    number: str
    heading_lines: list[Line]
    body: list[Line]
    part: str | None
    part_title: str | None
    chapter: str | None
    chapter_title: str | None
    group_heading: str | None


def _is_group_heading(text: str) -> bool:
    # Inserted sub-headings carry amendment markers, e.g. "4[Saving of Certain Laws]".
    t = re.sub(r"^\d{1,2}\*?\[", "", text.strip()).rstrip("]").strip()
    if not t or len(t) > 75 or t[0] in "(0123456789[*" or not t[0].isupper():
        return False
    if t[-1] in ".;:,—–-)]":
        return False
    if re.match(r"^(Provided|Explanation)", t):
        return False
    words = t.split()
    return len(words) <= 9 and t.upper() != t


def _plausible_title(rest: str) -> bool:
    rest = rest.strip()
    if not rest:
        return False
    return rest[0].isupper() or rest[0] in "[“\"" or bool(re.match(r"^\d{1,2}\*?\[", rest))


def _split_heading(heading_lines: list[Line], first_rest: str) -> tuple[str, str, int]:
    """Return (title, remainder of the heading text that belongs to the body, lines consumed)."""
    accumulated = first_rest
    consumed = 1
    # Most headings use an em dash; a few use "––" (Article 124C) or ".-" (Article 330A).
    dash = re.compile(r"—|–{2}|(?<=\.)\s?[-–](?=\s|\()")
    max_lines = 9 if not first_rest.strip() or len(first_rest.strip()) < 25 else 4
    while not dash.search(accumulated) and consumed < len(heading_lines) and consumed < max_lines:
        accumulated = f"{accumulated} {heading_lines[consumed].text.strip()}"
        consumed += 1
    found = dash.search(accumulated)
    if found:
        title, remainder = accumulated[:found.start()], accumulated[found.end():]
    else:
        # Omitted provisions: "31. [Compulsory acquisition of property.] Omitted by ..."
        match = re.match(r"^(\[[^\]]*\])\s*(.*)$", accumulated)
        if match:
            title, remainder = match.group(1), match.group(2)
        else:
            title, remainder = accumulated, ""
            consumed = 1
    title = _normalise(title).rstrip(" .")
    return title, remainder.strip(), consumed


def _article_ok(number: str, last: tuple[int, str] | None) -> bool:
    key = article_sort_key(number)
    if key[0] > 395:
        return False
    if last is None:
        return key[0] <= 2
    return key > last and key[0] <= last[0] + 6


def _collect_articles(lines: list[Line]) -> tuple[list[_ArticleDraft], list[dict]]:
    drafts: list[_ArticleDraft] = []
    containers: list[dict] = []
    part = part_title = chapter = chapter_title = None
    group: str | None = None
    pending: list[Line] = []
    current: _ArticleDraft | None = None
    last_key: tuple[int, str] | None = None
    index = 0
    while index < len(lines):
        line = lines[index]
        text = line.text
        part_match = PART_HEADING.match(text)
        if part_match and (not part_match.group(2).strip() or part_match.group(2).strip().startswith("—")
                           or part_match.group(2).strip().startswith(".")):
            part = part_match.group(1)
            chapter = chapter_title = None
            group = None
            title_parts = [part_match.group(2).strip(" .—")] if part_match.group(2).strip(" .—") else []
            look = index + 1
            while look < len(lines) and (lambda s: s and s.upper() == s and re.search(r"[A-Z]", s)
                                         and not CHAPTER_HEADING.match(s) and not ARTICLE_HEADING.match(s))(lines[look].text.strip()):
                title_parts.append(lines[look].text.strip())
                look += 1
            omitted_part = False
            if not title_parts and look < len(lines) and lines[look].text.strip().startswith("["):
                # e.g. "PART VII" followed by "[The States in Part B of the First Schedule]." (omitted Part)
                title_parts.append(lines[look].text.strip())
                omitted_part = True
                look += 1
            part_title = re.sub(r"\s*\d*\*(?:\s*\*)*", "", _normalise(clean_text(" ".join(title_parts)))).strip("[] .") or None
            containers.append({"kind": "part", "part": part, "title": part_title, "page": line.page,
                               "printed_page": line.printed_page, "lines": lines[index:look],
                               "omitted": omitted_part})
            current = None
            pending = []
            index = look
            continue
        chapter_match = CHAPTER_HEADING.match(text)
        if chapter_match:
            chapter = chapter_match.group(1)
            title_parts = [chapter_match.group(2).strip()]
            look = index + 1
            while (title_parts[-1].endswith(",") and look < len(lines)
                   and lines[look].text.strip().upper() == lines[look].text.strip()):
                title_parts.append(lines[look].text.strip())
                look += 1
            chapter_title = _normalise(clean_text(" ".join(title_parts))).strip("[] .")
            group = None
            containers.append({"kind": "chapter", "part": part, "chapter": chapter, "title": chapter_title,
                               "page": line.page, "printed_page": line.printed_page, "lines": lines[index:look]})
            current = None
            pending = []
            index = look
            continue
        heading = ARTICLE_HEADING.match(text)
        if heading and part and _article_ok(heading.group(2), last_key) and _plausible_title(
                heading.group(3) or (lines[index + 1].text if index + 1 < len(lines) else "")):
            # A short title-case line immediately before the heading is a group sub-heading.
            trailing = current.body if current else pending
            while trailing and not trailing[-1].text.strip():
                trailing.pop()
            if trailing and _is_group_heading(trailing[-1].text):
                group = _normalise(clean_text(trailing.pop().text)).strip("[] ")
            elif pending and any(_is_group_heading(l.text) for l in pending):
                group = _normalise(clean_text([l for l in pending if _is_group_heading(l.text)][-1].text)).strip("[] ")
            number = heading.group(2)
            current = _ArticleDraft(number=number, heading_lines=lines[index:index + 10], body=[],
                                    part=part, part_title=part_title, chapter=chapter,
                                    chapter_title=chapter_title, group_heading=group)
            drafts.append(current)
            last_key = article_sort_key(number)
            pending = []
            index += 1
            continue
        if current is not None:
            current.body.append(line)
        else:
            pending.append(line)
        index += 1
    return drafts, containers


def _finalise_article(draft: _ArticleDraft, footnotes, ordinal: int) -> list[Provision]:
    first = ARTICLE_HEADING.match(draft.heading_lines[0].text)
    assert first is not None
    rest = first.group(3) or ""
    title, remainder, consumed = _split_heading(draft.heading_lines, rest)
    body_lines = draft.body[consumed - 1:] if consumed > 1 else draft.body
    heading_used = draft.heading_lines[:consumed]
    heading_text = " ".join(l.text.strip() for l in heading_used)
    title_clean = _normalise(clean_text(title)).strip("[] .").replace("].", "").strip("[] .")
    article_id = f"COI:art:{draft.number}"
    all_lines = heading_used + [l for l in body_lines if l.text.strip()]
    page_start = draft.heading_lines[0].page
    page_end = max((l.page for l in all_lines), default=page_start)
    full_text = _normalise(" ".join(l.text.strip() for l in heading_used)) + "\n" + \
        "\n".join(l.text.strip() for l in body_lines if l.text.strip())
    # Omitted or substituted-away articles are printed as a bracketed title followed by an
    # editorial note ("[31. ...] Omitted by ...", "[232. Interpretation.—Articles 230, 231 and 232 subs. ...]").
    bare_bracket = "[" in first.group(1) and not re.search(r"\d", first.group(1))
    omitted = (bool(re.search(r"\bOmitted by\b", full_text[:400])) and len(clean_text(full_text)) < 600) \
        or title.startswith("[") or bare_bracket
    amendments = _markers_in(all_lines, footnotes)
    article = Provision(
        provision_id=article_id, kind="article", title=title_clean, text=full_text.strip(),
        page_start=page_start, page_end=page_end, printed_page=draft.heading_lines[0].printed_page,
        parent_id=None, part_number=draft.part, part_title=draft.part_title, chapter=draft.chapter,
        chapter_title=draft.chapter_title, group_heading=draft.group_heading, article_number=draft.number,
        status="omitted" if omitted else ("amended" if amendments else "in_force"),
        amendments=amendments, ordinal=ordinal,
    )
    out = [article]
    if omitted:
        return out
    # Body units: the heading remainder starts the body (e.g. "(1) India, that is Bharat,").
    units: list[list[Line]] = []
    start_lines = ([Line(remainder, draft.heading_lines[consumed - 1].page,
                         draft.heading_lines[consumed - 1].printed_page)] if remainder else [])
    stream = start_lines + [l for l in body_lines if l.text.strip()]
    clauses: list[tuple[str, list[Line], list[tuple[str, list[Line]]]]] = []
    lead: list[Line] = []              # text before the first clause (or the whole body)
    direct_subs: list[tuple[str, list[Line]]] = []   # sub-clauses with no numbered clause (e.g. 51A)
    expected_clause = 1
    last_clause_key: tuple[int, str] | None = None
    last_sub: str | None = None
    last_roman: str | None = None

    def clause_ok(num: str, rest_text: str) -> bool:
        key = article_sort_key(num)
        if re.match(r"^(of|or|and|to|in|shall|is|has|under)\b", rest_text):
            return False
        if last_clause_key is None:
            return key == (1, "")
        return key > last_clause_key and key[0] <= last_clause_key[0] + 1

    def sub_ok(letter: str, rest_text: str) -> bool:
        nonlocal last_roman
        if re.match(r"^(of|or|and)\s+(clause|sub-clause|article)", rest_text):
            return False
        if last_sub is None:
            return letter == "a"
        if len(letter) == 1 and ord(letter) == ord(last_sub[-1]) + 1 and len(last_sub) == 1:
            if ROMAN_ITEM.match(letter) and last_roman is not None:
                return False
            return True
        if len(letter) == 2 and letter[0] == last_sub[0] and last_sub in (letter[0], letter[:-1] + chr(ord(letter[-1]) - 1)):
            return not ROMAN_ITEM.match(letter)
        return False

    for line in stream:
        clause_match = CLAUSE_START.match(line.text)
        sub_match = SUBCLAUSE_START.match(line.text)
        if clause_match and clause_ok(clause_match.group(2), clause_match.group(3)):
            clauses.append((clause_match.group(2), [line], []))
            last_clause_key = article_sort_key(clause_match.group(2))
            last_sub = None
            last_roman = None
            continue
        if sub_match and sub_ok(sub_match.group(2), sub_match.group(3)):
            target = clauses[-1][2] if clauses else direct_subs
            target.append((sub_match.group(2), [line]))
            last_sub = sub_match.group(2)
            last_roman = None
            continue
        if sub_match and ROMAN_ITEM.match(sub_match.group(2)):
            last_roman = sub_match.group(2)
        # Continuation, proviso, Explanation or roman item: belongs to the open unit.
        if clauses:
            if clauses[-1][2] and not re.match(r"^\s*(Provided|Explanation)", line.text):
                clauses[-1][2][-1][1].append(line)
            else:
                if re.match(r"^\s*(Provided|Explanation)", line.text):
                    last_sub = last_sub  # provisos after sub-clauses belong to the clause
                clauses[-1][1].append(line)
        elif direct_subs:
            if re.match(r"^\s*(Provided|Explanation)", line.text):
                lead.append(line)
            else:
                direct_subs[-1][1].append(line)
        else:
            lead.append(line)
    del units, expected_clause

    def text_of(ls: list[Line]) -> str:
        return "\n".join(l.text.strip() for l in ls).strip()

    def status_of(ls: list[Line], text: str) -> tuple[str, list[dict]]:
        notes = _markers_in(ls, footnotes)
        if re.fullmatch(r"\W*\(?\w{0,3}\)?\s*\*[\s*\]]*", clean_text(text).strip("[] ")) or \
                re.search(r"^\W*\(\w+\)\s*\*\s*\*", clean_text(text)):
            return "omitted", notes
        if any((n.get("action") or "") == "Omitted" for n in notes) and len(clean_text(text)) < 40:
            return "omitted", notes
        return ("amended" if notes else "in_force"), notes

    for clause_number, clause_lines, subs in clauses:
        all_clause_lines = clause_lines + [l for _, sl in subs for l in sl]
        # Clause text keeps its sub-clauses so that a clause chunk is self-contained.
        clause_text = text_of(all_clause_lines)
        status, notes = status_of(all_clause_lines, text_of(clause_lines) if not subs else clause_text)
        clause_id = f"{article_id}:cl:{clause_number}"
        out.append(Provision(
            provision_id=clause_id, kind="clause", title=None, text=clause_text,
            page_start=all_clause_lines[0].page, page_end=all_clause_lines[-1].page,
            printed_page=all_clause_lines[0].printed_page, parent_id=article_id,
            part_number=draft.part, part_title=draft.part_title, chapter=draft.chapter,
            chapter_title=draft.chapter_title, group_heading=draft.group_heading,
            article_number=draft.number, clause_number=clause_number, status=status,
            amendments=notes, ordinal=ordinal,
        ))
        for letter, sub_lines in subs:
            sub_text = text_of(sub_lines)
            sub_status, sub_notes = status_of(sub_lines, sub_text)
            out.append(Provision(
                provision_id=f"{clause_id}:sub:{letter}", kind="subclause", title=None, text=sub_text,
                page_start=sub_lines[0].page, page_end=sub_lines[-1].page,
                printed_page=sub_lines[0].printed_page, parent_id=clause_id,
                part_number=draft.part, part_title=draft.part_title, chapter=draft.chapter,
                chapter_title=draft.chapter_title, group_heading=draft.group_heading,
                article_number=draft.number, clause_number=clause_number, subclause=letter,
                status=sub_status, amendments=sub_notes, ordinal=ordinal,
            ))
    for letter, sub_lines in direct_subs:
        sub_text = text_of(sub_lines)
        sub_status, sub_notes = status_of(sub_lines, sub_text)
        out.append(Provision(
            provision_id=f"{article_id}:sub:{letter}", kind="subclause", title=None, text=sub_text,
            page_start=sub_lines[0].page, page_end=sub_lines[-1].page,
            printed_page=sub_lines[0].printed_page, parent_id=article_id,
            part_number=draft.part, part_title=draft.part_title, chapter=draft.chapter,
            chapter_title=draft.chapter_title, group_heading=draft.group_heading,
            article_number=draft.number, subclause=letter, status=sub_status,
            amendments=sub_notes, ordinal=ordinal,
        ))
    return out


# --------------------------------------------------------------------------------------
# Schedules and appendices
# --------------------------------------------------------------------------------------

def _chunk_lines(lines: list[Line], limit: int = 1600) -> list[list[Line]]:
    groups: list[list[Line]] = []
    current: list[Line] = []
    size = 0
    for line in lines:
        if not line.text.strip():
            continue
        if current and size + len(line.text) > limit:
            groups.append(current)
            current, size = [], 0
        current.append(line)
        size += len(line.text) + 1
    if current:
        groups.append(current)
    return groups


def _parse_schedules(lines: list[Line], footnotes, ordinal_start: int) -> list[Provision]:
    out: list[Provision] = []
    blocks: list[tuple[str, list[Line]]] = []
    for line in lines:
        match = SCHEDULE_HEADING.match(line.text)
        if match:
            blocks.append((match.group(1).title(), [line]))
        elif blocks:
            blocks[-1][1].append(line)
    ordinal = ordinal_start
    for name, block in blocks:
        ordinal += 1
        sched_id = f"COI:sch:{SCHEDULE_ORDINALS.index(name.upper()) + 1}"
        content = [l for l in block if l.text.strip()]
        subtitle = next((l.text.strip() for l in content[1:4] if l.text.strip().startswith(("[", "("))), None)
        out.append(Provision(
            provision_id=sched_id, kind="schedule", title=f"{name} Schedule" + (f" {clean_text(subtitle)}" if subtitle else ""),
            text="\n".join(l.text.strip() for l in content[:6]), page_start=content[0].page,
            page_end=content[-1].page, printed_page=content[0].printed_page, schedule=name,
            amendments=_markers_in(content[:3], footnotes), ordinal=ordinal,
        ))
        if name.upper() == "SEVENTH":
            out.extend(_seventh_schedule_entries(sched_id, content, footnotes, ordinal))
        else:
            for n, group in enumerate(_chunk_lines(content), start=1):
                out.append(Provision(
                    provision_id=f"{sched_id}:seg:{n}", kind="schedule_entry", title=f"segment {n}",
                    text="\n".join(l.text.strip() for l in group), page_start=group[0].page,
                    page_end=group[-1].page, printed_page=group[0].printed_page, parent_id=sched_id,
                    schedule=name, amendments=_markers_in(group, footnotes), ordinal=ordinal,
                ))
    return out


def _seventh_schedule_entries(sched_id: str, content: list[Line], footnotes, ordinal: int) -> list[Provision]:
    out: list[Provision] = []
    list_name = None
    list_title = None
    entry: tuple[str, list[Line]] | None = None
    last_key: tuple[int, str] | None = None

    def flush():
        if entry and list_name:
            number, ls = entry
            text = "\n".join(l.text.strip() for l in ls)
            omitted = bool(re.search(r"\*\s*\*\s*\*", text)) and len(clean_text(text)) < 40
            notes = _markers_in(ls, footnotes)
            out.append(Provision(
                provision_id=f"{sched_id}:list:{list_name}:entry:{number}", kind="schedule_entry",
                title=f"List {list_name} ({list_title} List), Entry {number}", text=text,
                page_start=ls[0].page, page_end=ls[-1].page, printed_page=ls[0].printed_page,
                parent_id=sched_id, schedule="Seventh", status="omitted" if omitted else ("amended" if notes else "in_force"),
                amendments=notes, ordinal=ordinal,
            ))

    for line in content:
        list_match = SEVENTH_LIST.match(line.text)
        if list_match:
            flush()
            entry = None
            list_name, list_title = list_match.group(1), list_match.group(2).title()
            last_key = None
            continue
        entry_match = ENTRY_START.match(line.text)
        if list_name and entry_match:
            key = article_sort_key(entry_match.group(2))
            if (last_key is None and key[0] <= 2) or (last_key and key > last_key and key[0] <= last_key[0] + 3):
                flush()
                entry = (entry_match.group(2), [line])
                last_key = key
                continue
        if entry:
            entry[1].append(line)
    flush()
    return out


# --------------------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------------------

def parse_constitution(path: Path) -> ParsedConstitution:
    pages, footnotes, page_count = read_pages(path)
    layout = _find_layout(pages)
    provisions: list[Provision] = []

    # Preamble
    pre_lines = [l for l in _lines(pages, layout["preamble"], layout["preamble"] + 1) if l.text.strip()]
    start = next(i for i, l in enumerate(pre_lines) if l.text.strip() == "PREAMBLE")
    pre_text_lines = pre_lines[start + 1:]
    provisions.append(Provision(
        provision_id="COI:preamble", kind="preamble", title="Preamble",
        text="\n".join(l.text.strip() for l in pre_text_lines), page_start=layout["preamble"],
        page_end=layout["preamble"], printed_page=pre_lines[0].printed_page,
        amendments=_markers_in(pre_text_lines, footnotes),
        status="amended" if _markers_in(pre_text_lines, footnotes) else "in_force",
    ))

    body = _lines(pages, layout["preamble"] + 1, layout["schedules"])
    drafts, containers = _collect_articles(body)
    for container in containers:
        if container["kind"] == "part":
            pid = f"COI:part:{container['part']}"
            if any(p.provision_id == pid for p in provisions):
                continue
            text = "\n".join(l.text.strip() for l in container["lines"] if l.text.strip())
            notes = _markers_in(container["lines"], footnotes)
            provisions.append(Provision(
                provision_id=pid, kind="part", title=container["title"], text=text,
                page_start=container["page"], page_end=container["page"], printed_page=container["printed_page"],
                part_number=container["part"], part_title=container["title"], amendments=notes,
                status="omitted" if container.get("omitted") else ("amended" if notes else "in_force"),
            ))
        else:
            pid = f"COI:part:{container['part']}:ch:{container['chapter']}"
            if any(p.provision_id == pid for p in provisions):
                continue
            provisions.append(Provision(
                provision_id=pid, kind="chapter", title=f"Chapter {container['chapter']}: {container['title']}",
                text="\n".join(l.text.strip() for l in container["lines"]), page_start=container["page"],
                page_end=container["page"], printed_page=container["printed_page"],
                parent_id=f"COI:part:{container['part']}", part_number=container["part"],
                chapter=container["chapter"], chapter_title=container["title"],
                amendments=_markers_in(container["lines"], footnotes),
            ))
    seen_articles: set[str] = set()
    for ordinal, draft in enumerate(drafts, start=1):
        if draft.number in seen_articles:
            continue
        seen_articles.add(draft.number)
        provisions.extend(_finalise_article(draft, footnotes, ordinal))

    schedule_lines = _lines(pages, layout["schedules"], layout["appendices"])
    provisions.extend(_parse_schedules(schedule_lines, footnotes, len(drafts)))

    appendix_lines = _lines(pages, layout["appendices"], page_count + 1)
    current: tuple[str, list[Line]] | None = None
    appendices: list[tuple[str, list[Line]]] = []
    for line in appendix_lines:
        match = APPENDIX_HEADING.match(line.text)
        if match:
            current = (match.group(1), [])
            appendices.append(current)
        elif current:
            current[1].append(line)
    for name, ls in appendices:
        for n, group in enumerate(_chunk_lines(ls), start=1):
            provisions.append(Provision(
                provision_id=f"COI:appendix:{name}:seg:{n}", kind="appendix", title=f"Appendix {name}, segment {n}",
                text="\n".join(l.text.strip() for l in group), page_start=group[0].page, page_end=group[-1].page,
                printed_page=group[0].printed_page, schedule=None,
            ))
    return ParsedConstitution(provisions=provisions, footnotes=footnotes, page_count=page_count)
