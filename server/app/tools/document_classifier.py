"""
Document Classifier (Layer 1)
Deterministic classification of Indian legal documents into predefined types.
Uses keyword/pattern matching with optional LLM fallback at temperature=0.

Supported document types:
- Sale Deed
- FIR (First Information Report)
- Affidavit
- Agreement to Sell
- Power of Attorney
- Rent Agreement
- Notice (CrPC / CPC)
- Court Order / Judgment
- Will / Testament
- Partnership Deed
- Bail Application
- Complaint (under CrPC)
- Chargesheet
- Unknown
"""

import re
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class DocumentClassification:
    """Result of document type classification."""

    document_type: str
    confidence: float
    sub_type: Optional[str] = None
    matched_indicators: List[str] = field(default_factory=list)
    jurisdiction_hints: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "document_type": self.document_type,
            "confidence": round(self.confidence, 2),
            "sub_type": self.sub_type,
            "matched_indicators": self.matched_indicators,
            "jurisdiction_hints": self.jurisdiction_hints,
        }


# ============================================================================
# Pattern Definitions — each document type has weighted indicators
# ============================================================================

DOCUMENT_PATTERNS: Dict[str, Dict[str, Any]] = {
    "Sale Deed": {
        "primary_indicators": [
            r"sale\s*deed",
            r"conveyance\s*deed",
            r"deed\s*of\s*sale",
            r"transfer\s*of\s*property",
            r"immovable\s*property",
            r"registration\s*act",
        ],
        "secondary_indicators": [
            r"vendor\b",
            r"vendee\b",
            r"purchaser\b",
            r"conveyance",
            r"stamp\s*duty",
            r"sub[\s-]?registrar",
            r"schedule\s*of\s*property",
            r"sale\s*consideration",
            r"transfer\s*of\s*property\s*act",
            r"absolute\s*sale",
            r"freehold",
            r"market\s*value",
            r"registration\s*number",
        ],
        "weight": 1.0,
    },
    "FIR": {
        "primary_indicators": [
            r"first\s*information\s*report",
            r"\bfir\b",
            r"f\.?i\.?r\.?",
            r"information\s*received\s*at\s*p\.?s\.?",
            r"police\s*station\b.*\bcrime\s*no",
        ],
        "secondary_indicators": [
            r"complainant",
            r"accused\b",
            r"offence\b",
            r"section\s*\d+\s*(?:ipc|crpc|bnsa|bns)",
            r"investigating\s*officer",
            r"police\s*station",
            r"cognizable\s*offence",
            r"non[\s-]?cognizable",
            r"date\s*(?:and|&)\s*time\s*of\s*occurrence",
            r"general\s*diary\s*reference",
        ],
        "weight": 1.0,
    },
    "Affidavit": {
        "primary_indicators": [
            r"\baffidavit\b",
            r"sworn\s*statement",
            r"oath\s*commissioner",
            r"deponent\b",
            r"i\b.*\bsolemnly\s*(?:affirm|declare|state)",
        ],
        "secondary_indicators": [
            r"verification\s*clause",
            r"verified\s*at\b",
            r"before\s*me",
            r"notary\s*public",
            r"magistrate\b",
            r"stamp\s*paper",
            r"oath",
            r"true\s*to\s*(?:my|the\s*best\s*of\s*my)\s*knowledge",
            r"nothing\s*(?:material\s*)?has\s*been\s*(?:suppressed|concealed)",
        ],
        "weight": 1.0,
    },
    "Agreement to Sell": {
        "primary_indicators": [
            r"agreement\s*to\s*sell",
            r"agreement\s*for\s*sale",
            r"memorandum\s*of\s*understanding.*(?:property|sale)",
        ],
        "secondary_indicators": [
            r"earnest\s*money",
            r"token\s*amount",
            r"advance\s*payment",
            r"possession\s*(?:date|shall\s*be)",
            r"balance\s*(?:amount|consideration|payment)",
            r"buyer\b",
            r"seller\b",
            r"property\s*(?:described|situated)",
            r"specific\s*performance",
            r"time\s*(?:is|being)\s*(?:of\s*)?the\s*essence",
            r"party\s*of\s*the\s*(?:first|second)\s*part",
        ],
        "weight": 1.0,
    },
    "Power of Attorney": {
        "primary_indicators": [
            r"power\s*of\s*attorney",
            r"\bpoa\b",
            r"general\s*power\s*of\s*attorney",
            r"special\s*power\s*of\s*attorney",
            r"\bgpa\b.*(?:attorney|power|authority)",
            r"\bspa\b.*(?:attorney|power|authority)",
        ],
        "secondary_indicators": [
            r"principal\b",
            r"attorney\s*(?:in\s*fact|holder)",
            r"authorize\b",
            r"authorise\b",
            r"delegate\b",
            r"empower\b",
            r"act\s*on\s*(?:my|his|her|their)\s*behalf",
            r"power\s*of\s*attorney\s*act",
            r"irrevocable",
            r"revocable",
        ],
        "weight": 1.0,
    },
    "Rent Agreement": {
        "primary_indicators": [
            r"rent\s*agreement",
            r"lease\s*(?:deed|agreement)",
            r"tenancy\s*agreement",
            r"leave\s*and\s*licen[cs]e\s*agreement",
            r"rental\s*agreement",
        ],
        "secondary_indicators": [
            r"landlord\b",
            r"tenant\b",
            r"lessee\b",
            r"lessor\b",
            r"licensor\b",
            r"licensee\b",
            r"monthly\s*rent",
            r"security\s*deposit",
            r"rent\s*control\s*act",
            r"lock[\s-]?in\s*period",
            r"notice\s*period",
            r"maintenance\s*charges",
            r"premises\b",
            r"rent\s*escalation",
        ],
        "weight": 1.0,
    },
    "Notice (CrPC/CPC)": {
        "primary_indicators": [
            r"legal\s*notice",
            r"notice\s*under\s*section\s*\d+",
            r"demand\s*notice",
            r"cease\s*and\s*desist",
            r"notice\s*(?:u/?s|under)\s*\d+\s*(?:crpc|cpc|cr\.?p\.?c|c\.?p\.?c)",
            r"show\s*cause\s*notice",
            r"eviction\s*notice",
        ],
        "secondary_indicators": [
            r"hereby\s*(?:call\s*upon|demand|require|notify)",
            r"within\s*\d+\s*days",
            r"failing\s*which",
            r"legal\s*proceedings",
            r"advocate\b",
            r"on\s*behalf\s*of\s*(?:my|our)\s*client",
            r"reply\s*(?:within|to\s*this\s*notice)",
            r"without\s*prejudice",
            r"liability\b",
        ],
        "weight": 1.0,
    },
    "Court Order / Judgment": {
        "primary_indicators": [
            r"(?:in\s*the\s*)?(?:hon\'?ble|honourable)\s*(?:high\s*)?court",
            r"order\s*(?:sheet|dated)",
            r"judgment\b",
            r"decree\b",
            r"case\s*no\.?\s*\d+",
            r"writ\s*petition",
            r"civil\s*(?:suit|appeal|revision)",
            r"criminal\s*(?:appeal|revision|case|petition)",
        ],
        "secondary_indicators": [
            r"petitioner\b",
            r"respondent\b",
            r"appellant\b",
            r"plaintiff\b",
            r"defendant\b",
            r"versus\b",
            r"\bvs?\.\b",
            r"heard\s*(?:the|counsel|arguments)",
            r"disposed\s*of",
            r"dismissed\b",
            r"allowed\b",
            r"coram\b",
            r"bench\b",
            r"hon[\'\u2019]?ble\s*(?:justice|judge|mr\.\s*justice)",
        ],
        "weight": 1.0,
    },
    "Will / Testament": {
        "primary_indicators": [
            r"\bwill\b.*\btestament\b",
            r"last\s*will\s*and\s*testament",
            r"testamentary\b",
            r"codicil\b",
        ],
        "secondary_indicators": [
            r"testator\b",
            r"testatrix\b",
            r"executor\b",
            r"beneficiary\b",
            r"bequeath\b",
            r"devise\b",
            r"probate\b",
            r"succession\s*act",
            r"indian\s*succession\s*act",
            r"revoke\s*(?:all\s*)?(?:former|previous|earlier)\s*wills",
            r"sound\s*mind",
        ],
        "weight": 0.95,
    },
    "Partnership Deed": {
        "primary_indicators": [
            r"partnership\s*deed",
            r"deed\s*of\s*partnership",
            r"articles\s*of\s*partnership",
        ],
        "secondary_indicators": [
            r"partner\b",
            r"firm\s*name",
            r"profit[\s-]?sharing",
            r"loss[\s-]?sharing",
            r"capital\s*contribution",
            r"indian\s*partnership\s*act",
            r"partnership\s*act\s*(?:,\s*)?1932",
            r"dissolution\b",
            r"goodwill\b",
        ],
        "weight": 0.90,
    },
    "Bail Application": {
        "primary_indicators": [
            r"bail\s*application",
            r"application\s*(?:for|seeking)\s*(?:regular\s*|anticipatory\s*)?bail",
            r"anticipatory\s*bail",
            r"regular\s*bail",
        ],
        "secondary_indicators": [
            r"section\s*(?:437|438|439)\s*cr\.?p\.?c",
            r"accused\b",
            r"surety\b",
            r"undertaking\b",
            r"custody\b",
            r"remand\b",
            r"arrested\b",
            r"release\s*on\s*bail",
            r"conditions?\s*(?:of|for)\s*bail",
        ],
        "weight": 0.90,
    },
    "Complaint (CrPC)": {
        "primary_indicators": [
            r"criminal\s*complaint",
            r"complaint\s*under\s*section\s*\d+\s*cr\.?p\.?c",
            r"private\s*complaint",
        ],
        "secondary_indicators": [
            r"complainant\b",
            r"magistrate\b",
            r"cognizance\b",
            r"section\s*200\s*cr\.?p\.?c",
            r"section\s*156\(?3\)?\s*cr\.?p\.?c",
            r"offence\b",
        ],
        "weight": 0.85,
    },
    "Chargesheet": {
        "primary_indicators": [
            r"charge[\s-]?sheet",
            r"final\s*report\s*(?:under|u/?s)",
            r"challan\b",
        ],
        "secondary_indicators": [
            r"investigating\s*officer",
            r"section\s*173\s*cr\.?p\.?c",
            r"charge[\s-]?framed",
            r"prosecution\b",
            r"evidence\s*collected",
            r"witnesses\b",
            r"sanction\s*for\s*prosecution",
        ],
        "weight": 0.85,
    },
}


class DocumentClassifier:
    """
    Deterministic classifier for Indian legal documents.

    Layer 1 of the document analysis pipeline.
    Uses weighted pattern matching for reliable, reproducible classification.
    No LLM dependency — pure rule-based for consistency.
    """

    def __init__(self):
        self.patterns = DOCUMENT_PATTERNS

    def classify(self, document_text: str) -> DocumentClassification:
        """
        Classify a legal document into a predefined type.

        Args:
            document_text: Full text of the document

        Returns:
            DocumentClassification with type, confidence, and matched indicators
        """
        if not document_text or not document_text.strip():
            return DocumentClassification(
                document_type="Unknown",
                confidence=0.0,
                matched_indicators=["Empty document"],
            )

        text = document_text.lower()
        scores: Dict[str, Tuple[float, List[str]]] = {}

        for doc_type, config in self.patterns.items():
            primary = config["primary_indicators"]
            secondary = config["secondary_indicators"]
            type_weight = config["weight"]

            primary_matches = []
            secondary_matches = []

            # Primary indicators carry 3x weight
            for pattern in primary:
                if re.search(pattern, text, re.IGNORECASE):
                    primary_matches.append(pattern)

            # Secondary indicators carry 1x weight
            for pattern in secondary:
                if re.search(pattern, text, re.IGNORECASE):
                    secondary_matches.append(pattern)

            if not primary_matches and not secondary_matches:
                continue

            # Score calculation:
            # - Each primary match: 3 points
            # - Each secondary match: 1 point
            # - Normalize by total possible points
            total_possible = len(primary) * 3 + len(secondary)
            raw_score = len(primary_matches) * 3 + len(secondary_matches)
            normalized_score = (raw_score / total_possible) * type_weight

            # Bonus for having at least one primary match
            if primary_matches:
                normalized_score = min(1.0, normalized_score + 0.2)

            all_matches = [f"[P] {m}" for m in primary_matches] + [
                f"[S] {m}" for m in secondary_matches
            ]
            scores[doc_type] = (normalized_score, all_matches)

        if not scores:
            return DocumentClassification(
                document_type="Unknown",
                confidence=0.0,
                matched_indicators=["No known document patterns matched"],
            )

        # Pick highest scoring type
        best_type = max(scores, key=lambda k: scores[k][0])
        best_score, best_matches = scores[best_type]

        # Detect sub-type
        sub_type = self._detect_sub_type(best_type, text)

        # Detect jurisdiction hints
        jurisdiction_hints = self._detect_jurisdiction(text)

        return DocumentClassification(
            document_type=best_type,
            confidence=min(best_score, 0.99),  # Cap at 0.99 — never absolute
            sub_type=sub_type,
            matched_indicators=best_matches[:10],
            jurisdiction_hints=jurisdiction_hints,
        )

    def _detect_sub_type(self, doc_type: str, text: str) -> Optional[str]:
        """Detect sub-type within a document category."""
        sub_types: Dict[str, List[Tuple[str, str]]] = {
            "Power of Attorney": [
                (
                    r"general\s*power\s*of\s*attorney|\bgpa\b",
                    "General Power of Attorney",
                ),
                (
                    r"special\s*power\s*of\s*attorney|\bspa\b",
                    "Special Power of Attorney",
                ),
                (r"irrevocable", "Irrevocable Power of Attorney"),
            ],
            "Notice (CrPC/CPC)": [
                (r"eviction", "Eviction Notice"),
                (r"demand\s*notice", "Demand Notice"),
                (r"show\s*cause", "Show Cause Notice"),
                (r"cease\s*and\s*desist", "Cease and Desist Notice"),
                (
                    r"legal\s*notice.*recovery|recovery.*legal\s*notice",
                    "Recovery Notice",
                ),
                (
                    r"section\s*138.*negotiable|negotiable.*section\s*138|cheque\s*bounce",
                    "Cheque Bounce Notice (NI Act S.138)",
                ),
            ],
            "Court Order / Judgment": [
                (r"interim\s*order|ad[\s-]?interim", "Interim Order"),
                (r"final\s*(?:order|judgment|decree)", "Final Judgment"),
                (r"writ\s*(?:of\s*)?habeas\s*corpus", "Habeas Corpus"),
                (r"writ\s*(?:of\s*)?mandamus", "Mandamus"),
                (r"writ\s*(?:of\s*)?certiorari", "Certiorari"),
            ],
            "Affidavit": [
                (r"income\s*(?:tax|certificate)", "Income Affidavit"),
                (r"name\s*change|change\s*(?:of|in)\s*name", "Name Change Affidavit"),
                (r"address\s*(?:proof|verification)", "Address Verification Affidavit"),
                (r"gap\s*(?:certificate|period)", "Gap Certificate Affidavit"),
                (r"lost\s*(?:document|certificate)", "Lost Document Affidavit"),
            ],
            "Bail Application": [
                (r"anticipatory\s*bail|section\s*438", "Anticipatory Bail"),
                (r"regular\s*bail|section\s*(?:437|439)", "Regular Bail"),
                (r"default\s*bail|section\s*167\(?2\)?", "Default Bail"),
            ],
        }

        if doc_type not in sub_types:
            return None

        for pattern, sub_type_name in sub_types[doc_type]:
            if re.search(pattern, text, re.IGNORECASE):
                return sub_type_name

        return None

    def _detect_jurisdiction(self, text: str) -> List[str]:
        """Detect jurisdiction hints from the document."""
        hints = []

        # State-level jurisdiction
        states = [
            "delhi",
            "maharashtra",
            "karnataka",
            "tamil nadu",
            "uttar pradesh",
            "west bengal",
            "rajasthan",
            "gujarat",
            "madhya pradesh",
            "andhra pradesh",
            "telangana",
            "kerala",
            "punjab",
            "haryana",
            "bihar",
            "jharkhand",
            "odisha",
            "chhattisgarh",
            "assam",
            "goa",
            "himachal pradesh",
            "uttarakhand",
            "jammu",
            "kashmir",
        ]
        for state in states:
            if re.search(rf"\b{state}\b", text, re.IGNORECASE):
                hints.append(f"State: {state.title()}")

        # Court-level jurisdiction
        court_patterns = [
            (r"supreme\s*court", "Supreme Court of India"),
            (r"high\s*court", "High Court"),
            (r"district\s*court", "District Court"),
            (r"sessions\s*court", "Sessions Court"),
            (r"magistrate[\'\u2019]?s?\s*court", "Magistrate Court"),
            (r"consumer\s*(?:forum|court|commission)", "Consumer Forum"),
            (r"family\s*court", "Family Court"),
            (r"tribunal\b", "Tribunal"),
            (r"nclt\b", "NCLT"),
            (r"nclat\b", "NCLAT"),
        ]
        for pattern, court_name in court_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                hints.append(f"Court: {court_name}")

        return hints


# Singleton
_classifier: Optional[DocumentClassifier] = None


def get_document_classifier() -> DocumentClassifier:
    """Get or create the document classifier singleton."""
    global _classifier
    if _classifier is None:
        _classifier = DocumentClassifier()
    return _classifier
