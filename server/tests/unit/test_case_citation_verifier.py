import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.tools.case_citation_verifier import (
    case_verification_footer,
    iter_case_citations,
    verify_case_citations,
)
from app.tools.case_law_rag import normalize_case_name


class StubCaseRag:
    """find_case as case_citation_verifier uses it."""

    def __init__(self, known_names):
        self._known = list(known_names)

    def find_case(self, name):
        target = normalize_case_name(name)
        for known in self._known:
            if normalize_case_name(known) == target:
                return SimpleNamespace(case_name=known)
        return None


def test_extracts_name_and_year():
    citations = iter_case_citations(
        "As held in Shreya Singhal v. Union of India (2015), restrictions must be reasonable."
    )
    assert citations == [
        ("Shreya Singhal v. Union of India (2015)", "Shreya Singhal v. Union of India")
    ]


def test_does_not_extract_without_a_year():
    # Deliberate precision-over-recall choice: real citations almost always
    # carry a year; requiring one keeps ordinary "X v. Y" prose from being
    # flagged as an unverifiable case citation.
    assert iter_case_citations("Ghose v. Mugneeram Bangur applies here.") == []


def test_deduplicates_repeated_citations():
    text = (
        "Shreya Singhal v. Union of India (2015) held X. Later, "
        "Shreya Singhal v. Union of India (2015) was followed."
    )
    assert len(iter_case_citations(text)) == 1


def test_extracts_through_markdown_bold():
    citations = iter_case_citations(
        "**Shreya Singhal v. Union of India** (2015) is the leading case."
    )
    assert citations[0][1] == "Shreya Singhal v. Union of India"


def test_extracts_names_with_lowercase_connector_words():
    # No capitalized lead-in word directly before the party name: a preceding
    # capitalized word (e.g. a sentence-initial "See") is itself a valid
    # match for the party-phrase pattern and gets greedily swept in — a known
    # heuristic limitation, not the property this test checks.
    citations = iter_case_citations("Attorney General for India v. Satish (2021) is the authority.")
    assert citations[0][1] == "Attorney General for India v. Satish"


def test_a_cited_case_in_the_corpus_is_verified():
    report = verify_case_citations(
        "Shreya Singhal v. Union of India (2015) is the leading case.",
        StubCaseRag(["Shreya Singhal v. Union of India (2015)"]),
    )
    assert report.verified and not report.unverified


def test_a_fabricated_case_is_unverified_and_produces_a_footer():
    report = verify_case_citations(
        "Sharma v. Fictional Union Authority (2019) established this rule.",
        StubCaseRag(["Shreya Singhal v. Union of India (2015)"]),
    )
    assert report.unverified
    footer = case_verification_footer(report)
    assert "Sharma v. Fictional Union Authority" in footer
    assert "could not be verified" in footer


def test_no_citations_produces_no_footer():
    report = verify_case_citations(
        "This is general commentary with no case citation.", StubCaseRag([])
    )
    assert case_verification_footer(report) == ""
