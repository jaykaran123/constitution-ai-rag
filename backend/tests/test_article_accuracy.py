"""Structured extraction accuracy against the official Constitution PDF."""
from __future__ import annotations

import re

import pytest

from app.knowledge.constitution_parser import article_sort_key, clean_text, parse_constitution
from tests.conftest import CONSTITUTION_PDF


@pytest.fixture(scope="module")
def parsed():
    if not CONSTITUTION_PDF.exists():
        pytest.skip("Constitution PDF not installed")
    return parse_constitution(CONSTITUTION_PDF)


@pytest.fixture(scope="module")
def by_id(parsed):
    return parsed.by_id()


def text(p):
    return re.sub(r"\s+", " ", clean_text(p.text))


def test_every_article_number_from_1_to_395_is_present_except_part_vii(parsed):
    numbers = {article_sort_key(a.article_number)[0] for a in parsed.articles()}
    # Article 238 belonged to Part VII, which was omitted in 1956 and is not printed in the edition.
    assert set(range(1, 396)) - numbers == {238}
    assert len(parsed.articles()) >= 495


@pytest.mark.parametrize("article,title,part", [
    ("12", "Definition", "III"),
    ("14", "Equality before law", "III"),
    ("19", "Protection of certain rights regarding freedom of speech, etc", "III"),
    ("21", "Protection of life and personal liberty", "III"),
    ("21A", "Right to education", "III"),
    ("22", "Protection against arrest and detention in certain cases", "III"),
    ("23", "Prohibition of traffic in human beings and forced labour", "III"),
    ("32", "Remedies for enforcement of rights conferred by this Part", "III"),
    ("39A", "Equal justice and free legal aid", "IV"),
    ("51A", "Fundamental duties", "IVA"),
    ("226", "Power of High Courts to issue certain writs", "VI"),
    ("300A", "Persons not to be deprived of property save by authority of law", "XII"),
    ("368", "Power of Parliament to amend the Constitution and procedure therefor", "XX"),
    ("124C", "Power of Parliament to make law", "V"),
])
def test_article_titles_and_parts(by_id, article, title, part):
    p = by_id[f"COI:art:{article}"]
    assert p.title == title
    assert p.part_number == part


def test_article_21_exact_text_and_page(by_id):
    p = by_id["COI:art:21"]
    assert text(p) == ("21. Protection of life and personal liberty.—No person shall be deprived of his life or "
                       "personal liberty except according to procedure established by law.")
    assert p.page_start == 42 and p.printed_page == "11"
    assert not [x for x in by_id if x.startswith("COI:art:21:")]   # Article 21 has no clauses


def test_article_19_clause_and_subclause_structure(by_id):
    clauses = sorted(k.split(":")[-1] for k in by_id if re.fullmatch(r"COI:art:19:cl:\w+", k))
    assert clauses == ["1", "2", "3", "4", "5", "6"]
    subs = sorted(k.split(":")[-1] for k in by_id if k.startswith("COI:art:19:cl:1:sub:"))
    assert subs == list("abcdefg")
    assert text(by_id["COI:art:19:cl:1:sub:a"]).startswith("(a) to freedom of speech and expression")
    assert "reasonable restrictions" in text(by_id["COI:art:19:cl:2"])
    assert by_id["COI:art:19:cl:1:sub:f"].status == "omitted"


def test_article_22_clauses_and_exception(by_id):
    assert text(by_id["COI:art:22:cl:1"]).startswith("(1) No person who is arrested shall be detained in custody")
    assert "twenty-four hours" in text(by_id["COI:art:22:cl:2"])
    assert text(by_id["COI:art:22:cl:3"]).startswith("(3) Nothing in clauses (1) and (2) shall apply")
    assert {"COI:art:22:cl:3:sub:a", "COI:art:22:cl:3:sub:b"} <= set(by_id)


def test_article_51a_duties_are_direct_subclauses(by_id):
    duties = sorted(k.split(":")[-1] for k in by_id if k.startswith("COI:art:51A:sub:"))
    assert duties == list("abcdefghijk")


@pytest.mark.parametrize("pid", ["COI:art:2A", "COI:art:31", "COI:art:19:cl:1:sub:f", "COI:art:259", "COI:part:VII"])
def test_omitted_provisions_are_flagged(by_id, pid):
    assert by_id[pid].status == "omitted"


def test_amendment_history_is_attached(by_id):
    preamble = by_id["COI:preamble"]
    acts = {a["amendment_act"] for a in preamble.amendments}
    assert "Constitution (Forty-second Amendment) Act, 1976" in acts
    note = by_id["COI:art:19:cl:2"].amendments[0]
    assert note["amendment_act"] == "Constitution (First Amendment) Act, 1951" and note["action"] == "Subs"
    omitted_f = by_id["COI:art:19:cl:1:sub:f"].amendments[0]
    assert omitted_f["action"] == "Omitted" and omitted_f["effective_from"] == "20-6-1979"
    assert by_id["COI:art:21A"].status == "amended"   # inserted by the Eighty-sixth Amendment


def test_schedule_entries_are_not_articles(parsed, by_id):
    police = by_id["COI:sch:7:list:II:entry:2"]
    assert police.kind == "schedule_entry" and police.article_number is None
    assert "Police" in police.text
    # The old ingestion stored Seventh Schedule entry "19. Drugs and poisons" as Article 19.
    assert all("Drugs and poisons" not in a.text for a in parsed.articles())


def test_parts_and_chapters(by_id):
    assert by_id["COI:part:III"].title == "FUNDAMENTAL RIGHTS"
    assert by_id["COI:part:IXB"].title == "THE CO-OPERATIVE SOCIETIES"
    assert by_id["COI:art:300A"].chapter == "IV"
    assert by_id["COI:art:52"].chapter_title == "THE EXECUTIVE"
    assert by_id["COI:art:14"].group_heading == "Right to Equality"
    assert by_id["COI:art:23"].group_heading == "Right against Exploitation"
