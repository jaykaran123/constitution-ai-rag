"""Citation and answer validation (deterministic, no LLM)."""
from __future__ import annotations

from app.reasoning.applicability import NOT_DIRECT, MAY_ENGAGE, Assessment, ProvisionVerdict
from app.reasoning.citations import validate_answer
from app.retrieval.hybrid import Evidence

ART21 = Evidence(provision_id="COI:art:21", kind="constitution", label="Article 21", evidence_id="C1",
                 text="21. Protection of life and personal liberty.—No person shall be deprived of his life or "
                      "personal liberty except according to procedure established by law.")
ART22_1 = Evidence(provision_id="COI:art:22:cl:1", kind="constitution", label="Article 22(1)", evidence_id="C2",
                   text="(1) No person who is arrested shall be detained in custody without being informed, as soon "
                        "as may be, of the grounds for such arrest nor shall he be denied the right to consult, and "
                        "to be defended by, a legal practitioner of his choice.")
BNS303 = Evidence(provision_id="StatutorySection:BNS:303:1", kind="statute", label="BNS s.303(1) Theft",
                  evidence_id="S1", verification="unverified",
                  text="Whoever, intending to take dishonestly any movable property out of the possession of any "
                       "person without that person's consent, moves that property in order to such taking, is said "
                       "to commit theft.")
EVIDENCE = [ART21, ART22_1, BNS303]


def private_theft_assessment() -> Assessment:
    return Assessment(actor_type="private", primary_domain="ordinary_law",
                      constitutional_violation="unlikely_on_stated_facts",
                      provisions=[ProvisionVerdict("C1", "Article 21", "explicit", "state", NOT_DIRECT, "")])


def test_well_cited_answer_passes():
    answer = ("### Short answer\nTheft by a private person is mainly a criminal-law matter [S1].\n\n"
              "Article 21 says that \"No person shall be deprived of his life or personal liberty except according "
              "to procedure established by law\" [C1]. It does not by itself make a theft a constitutional "
              "violation [C1]. Section 303 defines theft [S1].")
    result = validate_answer(answer, EVIDENCE, private_theft_assessment())
    assert result.status == "passed", result.errors
    assert result.quotes_checked == 1 and result.quotes_verified == 1
    assert result.cited_ids == ["C1", "S1"]


def test_unknown_evidence_tag_fails():
    result = validate_answer("Article 21 protects liberty [C9].", EVIDENCE)
    assert result.status == "failed"
    assert any("[C9]" in e for e in result.errors)


def test_article_not_retrieved_is_rejected():
    result = validate_answer("Article 300A also protects your phone [C1].", EVIDENCE)
    assert any("Article 300A" in e for e in result.errors)


def test_section_not_in_evidence_is_rejected():
    result = validate_answer("Under Section 379 the thief can be punished [S1].", EVIDENCE)
    assert any("Section 379" in e for e in result.errors)


def test_fabricated_quotation_is_rejected():
    answer = 'Article 22(1) says "every arrested person must be released within six hours of arrest" [C2].'
    result = validate_answer(answer, EVIDENCE)
    assert result.status == "failed"
    assert any("Quotation not found" in e for e in result.errors)


def test_quotation_matching_ignores_curly_quotes_and_spacing():
    answer = "Article 22(1) requires that a person be informed “as soon as may be, of the grounds for such   arrest” [C2]."
    assert validate_answer(answer, EVIDENCE).status == "passed"


def test_definitive_violation_claim_is_rejected():
    result = validate_answer("Your Article 21 right has been violated [C1].", EVIDENCE)
    assert any("Definitive violation claim" in e for e in result.errors)


def test_hedged_claim_contradicting_assessment_is_rejected():
    answer = "Article 21 may have been violated by the thief [C1]."
    result = validate_answer(answer, EVIDENCE, private_theft_assessment())
    assert any("not directly applicable" in e for e in result.errors)


def test_negated_claim_consistent_with_assessment_passes():
    answer = "A theft by a private person does not, by itself, violate Article 21 [C1]."
    assert validate_answer(answer, EVIDENCE, private_theft_assessment()).status == "passed"


def test_hedged_claim_allowed_when_provision_may_be_engaged():
    assessment = Assessment(actor_type="state", primary_domain="constitutional",
                            constitutional_violation="possible_but_cannot_be_determined_without_more_facts",
                            provisions=[ProvisionVerdict("C2", "Article 22(1)", "primary", "arresting_authority",
                                                         MAY_ENGAGE, "")])
    answer = "If the police did not tell you the grounds, Article 22(1) may have been violated [C2]."
    assert validate_answer(answer, EVIDENCE, assessment).status == "passed"


def test_uncited_legal_statements_fail():
    answer = "Article 21 protects life.\nArticle 22(1) protects arrested persons.\nArticle 21 is important."
    result = validate_answer(answer, EVIDENCE)
    assert result.status == "failed"
    assert any("without an evidence citation" in e for e in result.errors)


def test_clause_not_retrieved_is_a_warning_not_an_error():
    answer = "Article 22(2) requires production before a magistrate [C2]."
    result = validate_answer(answer, EVIDENCE)
    assert result.status == "passed_with_warnings"
    assert any("Article 22(2)" in w for w in result.warnings)


def test_answer_without_any_citation_fails():
    answer = "Here is a long explanation of rights. " * 12
    assert "does not cite any evidence" in " ".join(validate_answer(answer, EVIDENCE).errors)


def test_short_quoted_terms_do_not_create_false_quotations():
    # Regression (live run 2026-10-30): quoting a short term such as “the State” made the validator pair a
    # closing quote with the next opening quote and flag ordinary prose as a fabricated quotation.
    answer = ("Article 21 restrains “the State” [C1]. A thief is not “the State”, so the theft is a criminal "
              "matter [S1]. Article 22(1) requires that the person be informed “as soon as may be, of the grounds "
              "for such arrest” [C2].")
    result = validate_answer(answer, EVIDENCE)
    assert result.status == "passed", result.errors
    assert result.quotes_checked == 1 and result.quotes_verified == 1


def test_quotation_with_ellipsis_must_match_fragments_in_one_source():
    ok = 'Article 22(1) says "No person who is arrested shall be detained … of his choice" [C2].'
    assert validate_answer(ok, EVIDENCE).status == "passed"
    bad = 'Article 22(1) says "No person who is arrested shall be detained … within six hours" [C2].'
    assert validate_answer(bad, EVIDENCE).status == "failed"


def test_nonexistent_article_cited_by_user_may_only_be_mentioned_as_nonexistent():
    ok = "There is no Article 500 in the Constitution; elections are covered elsewhere [C1]."
    assert validate_answer(ok, EVIDENCE, user_unknown_refs=["Article 500"]).status == "passed"
    bad = "Article 500 guarantees your right to vote [C1]."
    result = validate_answer(bad, EVIDENCE, user_unknown_refs=["Article 500"])
    assert result.status == "failed"


# Regressions from the live evaluation run (30 Oct 2026) ------------------------------------------
def test_bracketed_capitalisation_in_quotes_is_accepted():
    answer = 'Article 22(1) provides that "[n]o person who is arrested shall be detained in custody without being informed" [C2].'
    assert validate_answer(answer, EVIDENCE).status == "passed"


def test_evidence_delimiters_copied_into_a_quote_are_ignored():
    answer = 'It says "<<<No person shall be deprived of his life or personal liberty except according to procedure>>>" [C1].'
    assert validate_answer(answer, EVIDENCE).status == "passed"


def test_section_named_inside_evidence_text_may_be_mentioned():
    omitted = Evidence(provision_id="COI:art:31", kind="constitution", label="Article 31", evidence_id="C3",
                       text="[31. Compulsory acquisition of property.] Omitted by the Constitution (Forty-fourth "
                            "Amendment) Act, 1978, s. 6 (w.e.f. 20-6-1979).")
    answer = "Article 31 was omitted by section 6 of the Forty-fourth Amendment Act [C3]."
    assert validate_answer(answer, EVIDENCE + [omitted]).status == "passed"


def test_mentioning_violation_without_claiming_one_is_allowed():
    for sentence in ["Theft is an ordinary criminal-law matter rather than an automatic constitutional violation [S1].",
                     "Most Fundamental Rights limit the State, so private wrongs are generally not constitutional violations [C1]."]:
        assert validate_answer(sentence, EVIDENCE).status == "passed", sentence


def test_affirmative_claims_are_still_caught():
    for sentence in ["The police action was unconstitutional [C2].", "This is a clear violation of Article 22(1) [C2].",
                     "The officer violated your rights under Article 22(1) [C2]."]:
        assert validate_answer(sentence, EVIDENCE).status == "failed", sentence


def test_quote_across_amendment_brackets_matches():
    # Regression (live eval P5): Article 19(2) reads "in the interests of 4[the sovereignty and integrity of India], the
    # security of the State"; a faithful quote without the editorial brackets must verify.
    clause = Evidence(provision_id="COI:art:19:cl:2", kind="constitution", label="Article 19(2)", evidence_id="C4",
                      text="3[(2) Nothing in sub-clause (a) of clause (1) shall affect the operation of any existing law, "
                           "or prevent the State from making any law, in so far as such law imposes reasonable restrictions "
                           "on the exercise of the right conferred by the said sub-clause in the interests of 4[the "
                           "sovereignty and integrity of India], the security of the State, friendly relations with "
                           "foreign States, public order, decency or morality, or in relation to contempt of court, "
                           "defamation or incitement to an offence.]")
    answer = ('Restrictions are allowed "in the interests of the sovereignty and integrity of India, the security of the '
              'State, friendly relations with foreign States, public order, decency or morality" [C4].')
    assert validate_answer(answer, [clause]).status == "passed"
