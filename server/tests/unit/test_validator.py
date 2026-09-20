from app.tools.document_classifier import DocumentClassifier
from app.tools.statutory_validator import CHECKLIST_REGISTRY, StatutoryValidator, format_score
from app.tools.indian_law_rag import DOCUMENT_LAW_MAP

BAIL = (
    "IN THE COURT OF SESSIONS JUDGE. Bail application under Section 439 CrPC. FIR No. 12/2025, "
    "Police Station Saket. The applicant was arrested on 3 May and is in judicial custody. "
    "Grounds for bail: the applicant is falsely implicated and will cooperate with the investigation. "
    "The applicant is ready to furnish surety. No earlier application has been filed."
)


def test_every_classifiable_type_has_a_checklist_and_law_map():
    from app.tools.document_classifier import DOCUMENT_PATTERNS

    for doc_type in DOCUMENT_PATTERNS:
        assert doc_type in CHECKLIST_REGISTRY, doc_type
        assert doc_type in DOCUMENT_LAW_MAP, doc_type


def test_bail_application_validates():
    assert DocumentClassifier().classify(BAIL).document_type == "Bail Application"
    result = StatutoryValidator().validate(BAIL, "Bail Application")
    assert result.compliance_score == 1.0
    assert result.passed + result.failed == result.total_checks


def test_unknown_type_is_not_scored():
    result = StatutoryValidator().validate("text", "Unknown")
    assert result.compliance_score is None
    assert format_score(result.compliance_score) == "not assessed"


def test_missing_elements_counted():
    result = StatutoryValidator().validate("A bail application.", "Bail Application")
    assert result.failed > 0 and result.compliance_score < 1.0
