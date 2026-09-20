import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.tools.case_law_rag import _statute_overlap_score, normalize_act


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
