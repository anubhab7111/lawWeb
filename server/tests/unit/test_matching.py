import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.text_match import contains_word
from app.tools.citation_verifier import iter_citation_occurrences
from app.tools.civil_rag import CivilRAGSystem
from app.tools.crime_reporter import detect_crime_type
from app.tools.criminal_rag import CriminalRAGSystem, extract_crime_features
from app.tools.document_classifier import DocumentClassifier
from app.tools.legal_query_parser import parse_legal_query
from app.tool_dispatch import infer_indian_kanoon_context_type


def test_word_boundaries():
    assert not contains_word("fired from job", "fire")
    assert contains_word("the house is on fire", "fire")
    assert contains_word("he was killed", "kill")
    assert contains_word("she was stabbed", "stab")
    assert not contains_word("establish the claim", "stab")


def test_fired_is_not_arson():
    assert detect_crime_type("my employer fired me from my job") != "arson"


def test_deadline_is_not_violence():
    f = extract_crime_features("deadline to establish the claim")
    assert not f.violence and not f.death


def test_civil_expander_no_false_triggers():
    civil = CivilRAGSystem()
    assert "RTI" not in civil._preprocess_query("What is a first information report?")
    assert "mortgage" not in civil._preprocess_query("What is the criminal charge for theft?")


def test_criminal_expander_no_false_fir():
    q = "Is my first firm allowed to register"
    assert CriminalRAGSystem()._preprocess_query(q) == q


def test_rent_agreement_not_fir():
    text = (
        "This agreement is between the party of the first part and the second part. "
        "The lessee shall pay rent monthly to the landlord for the premises."
    )
    assert DocumentClassifier().classify(text).document_type != "FIR"


def test_act_hint_needs_whole_word():
    pins = parse_legal_query("Section 138 for medical records dispute").pinned_sections
    assert all("Contract" not in act for act, _ in pins)
    pins = parse_legal_query("Section 138 ICA").pinned_sections
    assert pins[0][0] == "Indian Contract"


def test_ik_context_type():
    assert infer_indian_kanoon_context_type("first time offender") != "crpc"
    assert infer_indian_kanoon_context_type("a written contract about grapes") != "constitution"
    assert infer_indian_kanoon_context_type("anticipatory bail in an FIR") == "crpc"


def test_citation_act_hint_across_abbreviated_dot():
    occ = iter_citation_occurrences(
        "Under Sec. 420 of the IPC the offence is cheating. Section 438 of the CrPC gives bail."
    )
    assert [o.act_hint for o in occ] == ["Indian Penal Code", "Code of Criminal Procedure"]
