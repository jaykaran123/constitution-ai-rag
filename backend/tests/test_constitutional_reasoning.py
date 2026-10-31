"""Constitutional applicability reasoning on the five primary scenarios and adversarial cases.

These tests are deterministic (rule-based classification, no LLM) and run against the real
ingested Constitution and knowledge graph.
"""
from __future__ import annotations

import pytest

from app.reasoning.answer import assign_evidence_ids
from app.reasoning.applicability import (APPLIES_ANY_ACTOR, MAY_ENGAGE, NOT_DIRECT, OMITTED, OUTSIDE_SCOPE, assess)
from app.retrieval.hybrid import HybridRetriever, ProvisionStore
from app.retrieval.query_understanding import analyse
from tests.conftest import run


def analyse_and_assess(db, question):
    by_id, _ = ProvisionStore.load(db)
    analysis = run(analyse(question, set(by_id), mode="rules"))
    result = run(HybridRetriever(db, None).retrieve(analysis, question, None))
    assign_evidence_ids(result)
    return analysis, result, assess(analysis, result)


def verdict(assessment, label):
    return next((p.verdict for p in assessment.provisions if p.label == label), None)


def labels(result):
    return {e.label for e in result.constitution}


# ----------------------------------------------------------------------------- primary scenarios
def test_case1_stolen_phone_is_ordinary_criminal_law(db):
    q = "Someone stole my mobile phone on a crowded bus. Has my Article 21 right been violated?"
    analysis, result, a = analyse_and_assess(db, q)
    assert analysis.actor_type == "private" and analysis.asks_violation
    assert "Article 21" in labels(result)
    assert verdict(a, "Article 21") == NOT_DIRECT                     # no automatic Article 21 violation
    assert a.primary_domain == "ordinary_law"
    assert a.constitutional_violation == "unlikely_on_stated_facts"
    assert "Theft" in a.ordinary_law
    assert any("303" in s.label for s in result.statutes)             # theft offence
    assert {"police_report_fir", "superintendent_of_police"} <= {r["id"] for r in result.remedies}
    assert a.premise_check and a.premise_check[0]["provision"] == "Article 21"
    assert "Article 12" in labels(result)                             # State vs private distinction


def test_case2_unlawful_arrest_engages_articles_21_and_22(db):
    q = "Police arrested me without telling me the grounds of arrest and refused access to a lawyer."
    analysis, result, a = analyse_and_assess(db, q)
    assert analysis.actor_type == "state"
    assert {"Article 21", "Article 22(1)"} <= labels(result)
    assert verdict(a, "Article 22(1)") == MAY_ENGAGE
    assert "Article 22(3)" in labels(result)                          # exceptions considered
    assert a.constitutional_violation == "possible_but_cannot_be_determined_without_more_facts"
    assert any("preventive detention" in f for f in a.missing_facts)  # factual qualification
    assert {"writ_high_court", "writ_supreme_court"} <= {r["id"] for r in result.remedies}


def test_case3_private_landlord_religion_article_15_scope(db):
    q = "My private landlord refused to rent me a flat because of my religion. Is Article 15 violated?"
    analysis, result, a = analyse_and_assess(db, q)
    assert analysis.actor_type == "private"
    assert {"Article 15(1)", "Article 15(2)"} <= labels(result)
    assert verdict(a, "Article 15(1)") == NOT_DIRECT                  # binds the State only
    assert verdict(a, "Article 15(2)") == OUTSIDE_SCOPE               # housing is not a listed matter
    assert verdict(a, "Article 15") == NOT_DIRECT
    assert a.constitutional_violation == "unlikely_on_stated_facts"
    assert "Article 15(4)" not in labels(result)                      # State-only exceptions dropped for private actor


def test_case4_forced_labour_engages_article_23(db):
    q = "My employer forces me to work without wages and refuses to let me leave."
    analysis, result, a = analyse_and_assess(db, q)
    assert "Article 23(1)" in labels(result)
    assert verdict(a, "Article 23(1)") == APPLIES_ANY_ACTOR           # not limited to the State
    statutes = " ".join(s.label for s in result.statutes)
    assert "146" in statutes and "BLSA" in statutes                   # labour / forced labour legislation
    assert "district_magistrate_bonded_labour" in {r["id"] for r in result.remedies}
    assert a.constitutional_violation == "possible_but_cannot_be_determined_without_more_facts"


def test_case5_private_college_speech_needs_19_2_and_article_12(db):
    q = "My private college threatened to suspend me because I criticized a government policy."
    analysis, result, a = analyse_and_assess(db, q)
    assert analysis.actor_type == "private"                           # "government policy" is not an actor
    assert {"Article 19(1)(a)", "Article 19(2)", "Article 12"} <= labels(result)
    assert verdict(a, "Article 19(1)(a)") == NOT_DIRECT
    assert a.constitutional_violation == "unlikely_on_stated_facts"
    assert any("government-run" in f for f in a.missing_facts)
    assert "institutional_appeal" in {r["id"] for r in result.remedies}


# ----------------------------------------------------------------------------- adversarial
def test_incorrect_article_number_is_reported(db):
    analysis, result, a = analyse_and_assess(db, "Does Article 500 protect my right to vote?")
    assert "Article 500" in analysis.unknown_refs
    assert any("not found" in n for n in a.notes)
    assert {"Article 326"} <= labels(result)                          # the real voting provision


def test_outdated_provision_is_flagged_as_omitted(db):
    analysis, result, a = analyse_and_assess(db, "Is the right to property still a fundamental right under Article 31?")
    assert verdict(a, "Article 31") == OMITTED
    assert "Article 300A" in labels(result)                           # current provision
    assert a.constitutional_violation == "not_a_violation_question"


def test_same_conduct_by_state_and_private_actor_differs(db):
    _, _, private = analyse_and_assess(db, "A private school refused my child admission because of our caste.")
    _, _, state = analyse_and_assess(db, "A government school refused my child admission because of our caste.")
    assert private.actor_type == "private" and state.actor_type == "state"
    assert verdict(state, "Article 15(1)") == MAY_ENGAGE
    assert verdict(private, "Article 15(1)") in {NOT_DIRECT, None}


def test_misleading_premise_untouchability_is_horizontal(db):
    _, result, a = analyse_and_assess(db, "My neighbour practises untouchability against me. The Constitution only binds "
                                          "the government, so nothing can be done, right?")
    assert "Article 17" in labels(result)
    assert verdict(a, "Article 17") == APPLIES_ANY_ACTOR


def test_multi_article_question(db):
    analysis, result, _ = analyse_and_assess(db, "Compare Article 14, Article 15 and Article 16.")
    assert {"Article 14", "Article 15", "Article 16"} <= labels(result)


def test_amendment_question_uses_article_368(db):
    _, result, _ = analyse_and_assess(db, "How can Parliament amend the Constitution?")
    assert "Article 368" in labels(result)


def test_no_constitutional_provision_directly_applies(db):
    analysis, result, a = analyse_and_assess(db, "Who won the cricket match yesterday?")
    assert analysis.out_of_scope


def test_rights_vs_directives(db):
    _, result, a = analyse_and_assess(db, "What is the difference between Fundamental Rights and Directive Principles?")
    assert {"Article 37"} <= labels(result) and any(l.startswith("Article 32") for l in labels(result))
    assert a.constitutional_violation == "not_a_violation_question"


@pytest.mark.parametrize("question,expected", [
    ("What does Article 19(1)(a) say?", {"Article 19(1)(a)", "Article 19(2)"}),
    ("Explain Article 21A", {"Article 21A"}),
    ("What are the fundamental duties?", {"Article 51A"}),
])
def test_explicit_lookups(db, question, expected):
    _, result, _ = analyse_and_assess(db, question)
    assert expected <= labels(result)


# ----------------------------------------------------------------------------- hybrid classification merge
def _hybrid(db, question, llm_json):
    import json
    from tests.conftest import FakeLLM
    by_id, _ = ProvisionStore.load(db)
    return run(analyse(question, set(by_id), llm=FakeLLM(classification=json.dumps(llm_json)), mode="hybrid"))


def _classification(issues, actor="unknown"):
    return {"issues": issues, "categories": [], "actor_type": actor, "incident_summary": "x",
            "asks_violation": False, "missing_facts": []}


def test_llm_cannot_replace_specific_rule_issues(db):
    # Regression (live eval A3): Gemini replaced "government taking property" with a generic overview.
    analysis = _hybrid(db, "Is the right to property still a fundamental right under Article 31?",
                       _classification(["fundamental_rights_overview"]))
    assert analysis.issues == ["property_deprivation_state"]
    assert analysis.issue_sources["_llm_disagreed"] == "fundamental_rights_overview"


def test_llm_adds_issues_when_it_agrees_with_rules(db):
    analysis = _hybrid(db, "Police arrested me and also refused to register my complaint about the theft.",
                       _classification(["arrest_safeguards", "police_refusal_complaint", "made_up_issue"]))
    assert {"arrest_safeguards", "police_refusal_complaint"} <= set(analysis.issues)
    assert "made_up_issue" not in analysis.issues           # closed list only


def test_llm_fills_gap_when_rules_find_nothing(db):
    question = "I am made to sweep the headman's house every day for nothing."
    rules_only = run(analyse(question, set(ProvisionStore.load(db)[0]), mode="rules"))
    assert not rules_only.issues
    analysis = _hybrid(db, question, _classification(["forced_labour"], actor="private"))
    assert analysis.issues == ["forced_labour"] and analysis.actor_type == "private"
    assert not analysis.out_of_scope


def test_classifier_failure_falls_back_to_rules(db):
    from app.llm.exceptions import LLMServerError
    from tests.conftest import FakeLLM
    by_id, _ = ProvisionStore.load(db)
    llm = FakeLLM(classification=LLMServerError("down"))
    analysis = run(analyse("My employer forces me to work without wages.", set(by_id), llm=llm, mode="hybrid"))
    assert analysis.issues == ["forced_labour"] and analysis.llm_error == "server_error"
