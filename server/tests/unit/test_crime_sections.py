import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.tools.criminal_rag import CriminalRAGSystem, SectionMatch, _fits_crime_type, _is_offence_section


def _match(act, section, title="t"):
    return SectionMatch(section=section, title=title, confidence=0.9, reasons=[], punishment="", definition="",
                        act_name=act)


def test_offence_sections_exclude_procedure_and_struck_down_adultery():
    assert _is_offence_section("Indian Penal Code", "420")
    assert _is_offence_section("Information Technology Act", "66D")
    assert not _is_offence_section("Code of Criminal Procedure", "41A")
    assert not _is_offence_section("Bharatiya Nagarik Suraksha Sanhita BNSS", "54")
    assert not _is_offence_section("Indian Penal Code", "497")


def test_section_from_unrelated_crime_family_does_not_fit_the_report():
    assert not _fits_crime_type("420", "rape")
    assert not _fits_crime_type("420", "theft")
    assert _fits_crime_type("420", "cybercrime")
    assert _fits_crime_type("379", "robbery")
    assert _fits_crime_type("506", "assault")
    assert _fits_crime_type("498A", "dowry")
    assert _fits_crime_type("420", "general")
    assert _fits_crime_type("120B", "rape")


def test_ipc_match_is_replaced_by_indexed_bns_section_and_deduplicated(monkeypatch):
    bns_chunk = SimpleNamespace(act_name="Bharatiya Nyaya Sanhita BNS", title="Cheating", text="318. Cheating.")
    monkeypatch.setattr("app.tools.unified_legal_rag.get_unified_rag_system",
                        lambda: SimpleNamespace(_chunks={"c1": bns_chunk}))
    rag = CriminalRAGSystem.__new__(CriminalRAGSystem)
    rag._pinned_chunk_ids = lambda pairs: ["c1"] if pairs == [("Bharatiya Nyaya Sanhita", "318")] else []

    out = rag._prefer_bns([_match("Indian Penal Code", "420"), _match("Bharatiya Nyaya Sanhita BNS", "318"),
                           _match("Indian Penal Code", "379", title="Theft")])

    assert [(m.act_name, m.section) for m in out] == [("Bharatiya Nyaya Sanhita BNS", "318"),
                                                       ("Indian Penal Code", "379")]
    assert "formerly IPC § 420" in out[0].title
    assert "now BNS § 303" in out[1].title
