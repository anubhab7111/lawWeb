"""The claim-level grounding gate. Fixtures are modelled on real answers where the
previous gate flagged faithful statute text, resolved section numbers against the
wrong Act, and rewrote correct sentences."""

import asyncio
import json
from types import SimpleNamespace

from app.tools import grounding_verifier as gv

# The format app.tool_dispatch._budget_context produces. The first provision
# follows the block header with a SINGLE newline — a parser that splits on blank
# lines alone silently drops it.
CONTEXT = (
    "**Applicable Statutory Provisions** (each is tagged with its legal domain):\n"
    "• **Constitution of India Article 21** — Protection of life and personal liberty [constitutional]\n"
    "21. Protection of life and personal liberty. —No person shall be deprived of his life or "
    "personal liberty except according to procedure established by law.\n\n"
    "• **Constitution of India Article 19** — Protection of certain rights regarding freedom of speech [constitutional]\n"
    "19. Protection of certain rights regarding freedom of speech, etc.—(1) All citizens shall have "
    "the right to freedom of speech and expression, to assemble peaceably, to form associations or "
    "unions, to move freely throughout the territory of India, and to practise any profession, or to "
    "carry on any occupation, trade or business.\n\n"
    "• **Indian Contract Act § 10** — What agreements are contracts [civil]\n"
    "10. What agreements are contracts.—All agreements are contracts if they are made by the free "
    "consent of parties competent to contract, for a lawful consideration and with a lawful object, "
    "and are not hereby expressly declared to be void. Nothing herein contained shall affect any law "
    "in force in India by which any contract is required to be made in writing.\n\n"
    "• **Hindu Marriage Act § 13B** — Divorce by mutual consent [family]\n"
    "13B. Divorce by mutual consent.—(1) A petition for dissolution of marriage may be presented on "
    "the ground that they have been living separately for a period of one year or more, that they "
    "have not been able to live together and that they have mutually agreed that the marriage "
    "should be dissolved. (2) On the motion of both the parties made not earlier than six months "
    "after the date of the presentation of the petition and not later than eighteen months after "
    "the said date, the court shall grant a decree. Imprisonment may extend to seven years.\n\n"
    "• **Indian Contract Act § 56** — Agreement to do impossible act [civil]\n"
    "56. Agreement to do impossible act.—A contract to do an act which, after the contract is made, "
    "becomes impossible, or, by reason of some event which the promisor could not prevent, "
    "unlawful, becomes void when the act becomes impossible or unlawful, unless the parties agree.\n\n"
    "• **Shreya Singhal v. Union of India** (2015) — Supreme Court\n"
    "Restrictions on speech must be reasonable and proximate to public order."
)
SECTIONS = {"21", "19", "10", "13B", "56"}

DECOY_COPYRIGHT_56 = [SimpleNamespace(
    text="56. Protection of separate rights.—Subject to the provisions of this Act, where the "
         "several rights comprising the copyright in any work are owned by different persons.",
    act_name="Copyright Act")]


class StubRag:
    """find_section as the gate uses it: real text for a hinted lookup of a
    section we hold, and the Copyright Act decoy for an act-less "56" — what the
    old gate wrongly used as evidence for a Contract Act citation."""

    def find_section(self, act_hint, section, max_parts=1):
        if not act_hint:
            return DECOY_COPYRIGHT_56 if section == "56" else []
        for entry in gv._context_passages(CONTEXT):
            if entry[0] == section.upper():
                return [SimpleNamespace(text=entry[1], act_name=act_hint)]
        return []


def assess(answer):
    return gv.assess_grounding(answer, StubRag(), set(SECTIONS), CONTEXT)


def only_claim(answer):
    claims = assess(answer).claim_sentences
    assert len(claims) == 1, [c.text for c in claims]
    return claims[0]


# --- parsing the context -------------------------------------------------------


def test_the_first_provision_after_the_block_header_is_parsed():
    sections = [sec for sec, _ in gv._context_passages(CONTEXT)]
    assert sections == ["21", "19", "10", "13B", "56", ""]  # incl. the first, and case law (no section)


# --- faithful text must not be flagged ------------------------------------------


def test_verbatim_statute_with_must_is_supported():
    claim = only_claim("They must have been living separately for a period of one year or more.")
    assert claim.status == gv.SUPPORTED


def test_faithful_restatement_of_an_exception_is_not_a_contradiction():
    # "No person shall be deprived ... except according to procedure" restated with
    # "cannot ... without": the old dropped-qualifier rule called this CONTRADICTED.
    claim = only_claim(
        "Under Article 21 no person can be deprived of life or personal liberty without "
        "procedure established by law."
    )
    assert claim.status != gv.CONTRADICTED


def test_qualifiers_elsewhere_in_the_context_do_not_condemn_an_uncited_claim():
    # the context contains "unless"/"except" (other provisions); it must not
    # count against an unrelated obligation claim
    claim = only_claim(
        "The court must grant a decree on the motion of both parties made after six months."
    )
    assert claim.status == gv.SUPPORTED


def test_a_claim_resting_on_case_law_is_not_flagged_for_lacking_the_section_text():
    claim = only_claim(
        "Article 21 restrictions must be reasonable and proximate to public order (Shreya Singhal)."
    )
    assert claim.status != gv.UNGROUNDED


# --- real errors are still caught ------------------------------------------------


def test_a_qualifier_phrased_differently_from_the_source_is_not_invented():
    # Regression (live, 18-query eval set): §10 Contract Act's real exception is
    # "Nothing herein contained shall affect any law... requiring... in writing"
    # — no literal "unless"/"except". A faithful claim using "unless" to describe
    # that same exception was flagged as inventing a condition the source lacks.
    claim = only_claim(
        "Section 10 of the Contract Act makes agreements binding unless a specific "
        "law requires them to be in writing."
    )
    assert claim.status != gv.CONTRADICTED


def test_a_claim_with_no_qualifier_language_anywhere_in_the_source_is_still_caught():
    # The broadened check only forgives a DIFFERENTLY-WORDED qualifier — an
    # invented one against a source with no qualifying language at all must
    # still be caught.
    # Article 19 (added to this fixture for the multi-citation test below)
    # states no qualifier of any kind, so an invented one must still be caught.
    claim = only_claim(
        "Article 19 protects freedom of speech unless the President declares a "
        "national holiday."
    )
    assert claim.status == gv.CONTRADICTED


def test_a_claim_spanning_two_cited_provisions_is_checked_against_both():
    # Regression (live): a claim about Article 19 qualified by Article 21's
    # "except ..." was evidenced against whichever provision scored higher
    # generic word-overlap (Article 19, the longer one) and flagged
    # CONTRADICTED for "inventing" a qualifier Article 21 actually states.
    claim = only_claim(
        "All citizens' freedom of speech, expression, and right to move freely, "
        "form associations, or carry on any trade, occupation, profession or "
        "business is subject to lawful procedure, per Article 19 and Article 21."
    )
    assert claim.status != gv.CONTRADICTED


def test_a_negated_absolute_marker_does_not_assert_an_exceptionless_rule():
    # "No absolute right exists" AGREES with a qualified provision — the
    # opposite of "This right is absolute". Regression (live, 18-query eval
    # set): flagged CONTRADICTED against Article 21's real "except ..." clause
    # because the claim was correctly agreeing with it.
    claim = only_claim(
        "No absolute right exists under Article 21, which is subject to "
        "procedure established by law."
    )
    assert claim.status != gv.CONTRADICTED


def test_an_unnegated_absolute_marker_still_asserts_an_exceptionless_rule():
    claim = only_claim("Article 21 is an absolute right with no exceptions.")
    assert claim.status == gv.CONTRADICTED


def test_an_exceptionless_rule_about_a_qualified_provision_is_a_contradiction():
    claim = only_claim("Article 21 is an absolute right that can never be restricted under any procedure.")
    assert claim.status == gv.CONTRADICTED


def test_an_invented_condition_is_a_contradiction():
    claim = only_claim(
        "Under Section 13B of the Hindu Marriage Act a divorce by mutual consent must be granted "
        "within 24 hours of the petition unless the court objects."
    )
    assert claim.status == gv.CONTRADICTED


def test_a_quantity_the_provision_does_not_state_is_flagged():
    claim = only_claim("Section 13B of the Hindu Marriage Act allows a decree only after ten years.")
    assert claim.status == gv.CONTRADICTED
    assert "10 year" in claim.reason


def test_number_words_and_digits_are_the_same_quantity():
    ok = only_claim("Under Section 13B the petition must come not later than 18 months after the date.")
    assert ok.status == gv.SUPPORTED
    assert gv._quantities("imprisonment up to seven years") == gv._quantities("imprisonment up to 7 years")
    assert ("18", "month") in gv._quantities("not later than eighteen months")


def test_a_citation_to_a_section_that_was_not_retrieved_is_ungrounded():
    claim = only_claim("Section 999 of the Indian Evidence Act provides that chats carry a presumption of forgery.")
    assert claim.status == gv.UNGROUNDED
    assert "not among the retrieved provisions" in claim.reason


def test_a_miscited_section_cannot_ride_on_a_neighbouring_provisions_wording():
    # Worded exactly like retrieved Contract Act s.56, but attributed to a section
    # that was never retrieved. Letting a cited claim match ANY retrieved passage
    # would pass this on shared vocabulary alone.
    claim = only_claim(
        "Section 999 of the Indian Contract Act says a contract to do an act becomes void "
        "when the act becomes impossible or unlawful."
    )
    assert claim.status == gv.UNGROUNDED
    assert "not among the retrieved provisions" in claim.reason


def test_an_actless_citation_resolves_to_the_retrieved_provision_not_another_act():
    # rag.find_section("", "56") returns the Copyright Act; the retrieved provision is
    # the Contract Act's, and the claim is about that.
    claim = only_claim("Section 56 makes a contract void when the act becomes impossible or unlawful.")
    assert claim.status == gv.SUPPORTED
    assert "impossible" in claim.evidence and "copyright" not in claim.evidence.lower()


# --- what is (not) a claim -------------------------------------------------------


def test_headings_table_cells_hedges_and_list_lead_ins_are_not_claims():
    answer = (
        "#### B. General Legal Procedure (Article 21)\n"
        "| Claimed event | Section 56 | *Satyabrata Ghose v. Mugneeram Bangur* (1954), Section 56 |\n"
        "The retrieved context does not specify exact thresholds, but Article 21 implies procedure.\n"
        "In practice, a party claiming force majeure must prove the event:\n"
    )
    assert assess(answer).claim_sentences == []


def test_the_splitter_does_not_break_on_enumerators_and_abbreviations():
    def spans(text):
        return [text[a:b].strip() for a, b in gv._split_sentences(text)]

    assert spans("Ghose v. Mugneeram Bangur applies here.") == ["Ghose v. Mugneeram Bangur applies here."]
    assert spans("B. General procedure applies.") == ["B. General procedure applies."]
    # ...but a year ending a sentence is still a boundary
    assert spans("The Act, 1956. The court decides.") == ["The Act, 1956.", "The court decides."]


# --- adjudication: when text may be rewritten -------------------------------------


def _adjudicate(answer, reply_status, corrected="REWRITTEN."):
    async def llm(prompt):
        return json.dumps([{"index": 1, "status": reply_status, "corrected": corrected}])

    return asyncio.run(gv.ground_and_correct(answer, StubRag(), set(SECTIONS), CONTEXT, llm))


def test_a_sentence_the_deterministic_evidence_condemned_is_rewritten():
    answer = "Section 13B of the Hindu Marriage Act allows a decree only after ten years."
    text, report = _adjudicate(answer, gv.UNGROUNDED, "The retrieved text sets no ten-year period.")
    assert "ten-year period" in text and report.sentences[0].outcome == "corrected"
    assert report.confirmed_flagged


def test_an_llm_only_objection_flags_but_never_rewrites():
    # cited paraphrase, deterministically fine (overlap 0.25-0.6) -> sent for review;
    # the small model objects, but nothing else does.
    answer = "Article 21 requires the State to follow a fair procedure before depriving anyone of liberty."
    text, report = _adjudicate(answer, gv.CONTRADICTED, "This is wrong.")
    assert text == answer
    claim = report.claim_sentences[0]
    assert claim.status == gv.CONTRADICTED and claim.outcome == "unchanged"
    assert report.confirmed_flagged == []
    assert "could not confirm" in claim.reason


def test_a_supported_verdict_never_rewrites_whatever_wording_comes_back():
    answer = "Article 21 requires the State to follow a fair procedure before depriving anyone of liberty."
    text, report = _adjudicate(answer, gv.SUPPORTED, "Something entirely different.")
    assert text == answer and report.flagged == []


# --- the user-facing footer --------------------------------------------------------


def test_an_llm_only_objection_never_produces_a_footer():
    answer = "Article 21 requires the State to follow a fair procedure before depriving anyone of liberty."
    _, report = _adjudicate(answer, gv.CONTRADICTED)
    assert gv.grounding_footer(report) == ""


def test_a_confirmed_problem_produces_a_footer_with_an_honest_summary():
    answer = "Section 13B of the Hindu Marriage Act allows a decree only after ten years."
    report = assess(answer)
    report.llm_succeeded = True
    report.sentences[0].status = gv.UNGROUNDED  # what adjudication would leave
    footer = gv.grounding_footer(report)
    assert "Grounding check" in footer and "ten years" in footer
