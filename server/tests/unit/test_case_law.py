import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.tools.case_law_rag import (
    CaseLawRAGSystem,
    CaseRecord,
    _statute_overlap_score,
    find_case_in_cases,
    normalize_act,
    normalize_case_name,
)


def _case(case_id, name):
    return CaseRecord(
        case_id=case_id, case_name=name, citation="", court="",
        bench_size=1, court_rank=1, date="", status="reported",
    )


def test_normalize_act():
    assert normalize_act("Bharatiya Nyaya Sanhita BNS") == normalize_act("Bharatiya Nyaya Sanhita")
    assert normalize_act("Indian Contract Act") == normalize_act("Indian Contract")
    assert normalize_act("Insolvency and Bankruptcy Code IBC") == normalize_act("Insolvency and Bankruptcy")
    assert normalize_act("POCSO Act") == normalize_act("POCSO")


def test_statute_overlap_matches_index_names():
    case = [["Bharatiya Nyaya Sanhita", "103"], ["Indian Contract", "73"]]
    boost = {
        (normalize_act("Bharatiya Nyaya Sanhita BNS"), "103"),
        (normalize_act("Indian Contract Act"), "10"),
    }
    assert _statute_overlap_score(case, boost) == 0.5


def test_normalize_case_name_ignores_year_vs_spelling_and_whitespace():
    assert normalize_case_name("Ghose v. Mugneeram Bangur (1954)") == normalize_case_name(
        "ghose   vs  mugneeram bangur"
    )
    assert normalize_case_name("Anvar P.V. v. P.K. Basheer (2014)") == normalize_case_name(
        "Anvar P.V. versus P.K. Basheer"
    )


def test_find_case_exact_match_after_normalization():
    cases = {"1": _case("1", "Shreya Singhal v. Union of India (2015)")}
    found = find_case_in_cases(cases, "shreya singhal vs union of india")
    assert found is not None and found.case_id == "1"


def test_find_case_fuzzy_match_tolerates_a_dropped_word():
    cases = {"1": _case("1", "Shreya Singhal v. Union of India (2015)")}
    found = find_case_in_cases(cases, "Shreya Singhal v Union India")
    assert found is not None and found.case_id == "1"


def test_find_case_returns_none_for_an_unrelated_name():
    cases = {"1": _case("1", "Shreya Singhal v. Union of India (2015)")}
    assert find_case_in_cases(cases, "Kesavananda Bharati v. State of Kerala") is None


def test_find_case_returns_none_on_empty_corpus():
    assert find_case_in_cases({}, "Shreya Singhal v. Union of India") is None


def test_case_law_rag_system_find_case_delegates_to_the_cases_dict():
    system = CaseLawRAGSystem()
    system._cases = {"1": _case("1", "Shreya Singhal v. Union of India (2015)")}
    assert system.find_case("Shreya Singhal vs Union of India").case_id == "1"
    assert system.find_case("Nonexistent Case v. Nobody") is None
