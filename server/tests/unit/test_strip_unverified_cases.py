import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.tools.case_citation_verifier import NO_VERIFIED_CASE, strip_unverified_cases

KNOWN = {"lalita kumari v. govt. of u.p.", "suraj lamp & industries v. state of haryana"}


def find_case(name):
    """Like CaseLawRAGSystem.find_case: a trailing "(year)" is ignored."""
    return object() if re.sub(r"\s*\(\d{4}\)$", "", name).lower() in KNOWN else None


def test_table_cell_naming_an_unverified_case_is_blanked():
    text = ("| Element | Case Law |\n"
            "| Rent | *Suresh Kumar v. Delhi Rent Control Board* (2019) — invalidated agreement |\n"
            "| FIR | Lalita Kumari v. Govt. of U.P. (2014) |")
    out, removed = strip_unverified_cases(text, find_case)
    assert "Suresh Kumar" not in out and "invalidated agreement" not in out
    assert "| Rent | — |" in out
    assert "Lalita Kumari v. Govt. of U.P. (2014)" in out
    assert removed == ["Suresh Kumar v. Delhi Rent Control Board"]


def test_case_law_line_falls_back_to_none_in_database():
    out, _ = strip_unverified_cases("- **Case Law:** S.K. Sharma v. Delhi Rent Control Board — notice rule", find_case)
    assert out == f"- **Case Law:** {NO_VERIFIED_CASE}"


def test_prose_sentence_with_an_unverified_case_is_dropped_and_yearless_names_are_caught():
    text = "Registration is compulsory. In Rajesh Gupta v. Union of India the court voided it. Fix the deed."
    out, removed = strip_unverified_cases(text, find_case)
    assert out == "Registration is compulsory. Fix the deed."
    assert removed == ["Rajesh Gupta v. Union of India"]


def test_bullet_that_only_named_an_unverified_case_disappears():
    out, _ = strip_unverified_cases("Intro.\n- Fake Party v. Other Party (2001).\nEnd.", find_case)
    assert out == "Intro.\nEnd."


def test_verified_and_case_free_text_is_untouched():
    text = "See Suraj Lamp & Industries v. State of Haryana (2012). Section 17 requires registration."
    assert strip_unverified_cases(text, find_case) == (text, [])


def test_only_curated_precedents_found_in_the_index_are_listed(monkeypatch):
    import asyncio
    from types import SimpleNamespace

    import app.tools.case_law_rag as clr
    from app.tools.indian_law_rag import IndianLawRAGTool

    monkeypatch.setattr(clr, "get_case_law_rag_system",
                        lambda: SimpleNamespace(initialized=True, find_case=find_case))
    info = {"key_precedents": ["Lalita Kumari v. Govt. of U.P. (2014) — FIR registration",
                               "Punati Ramulu v. State (1993) — not indexed"]}
    notes = asyncio.run(IndianLawRAGTool()._verified_precedents(info))
    assert notes == ["Lalita Kumari v. Govt. of U.P. (2014) — FIR registration"]
