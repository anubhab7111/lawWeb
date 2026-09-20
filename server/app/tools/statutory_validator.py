"""
Statutory Requirements Validator (Layer 2)
Pure rule-based engine — NO LLM dependency.

For each document type, validates against:
- Mandatory elements required by Indian statutes
- Standard drafting practices recognized by courts
- State-specific requirements (stamp duty, registration, etc.)

Sources of authority:
- Indian Registration Act, 1908
- Indian Stamp Act, 1899 (+ State Stamp Acts)
- Transfer of Property Act, 1882
- Indian Contract Act, 1872
- Code of Civil Procedure, 1908 (CPC)
- Code of Criminal Procedure, 1973 (CrPC) / BNSS 2023
- Indian Penal Code, 1860 (IPC) / BNS 2023
- Indian Evidence Act, 1872 / BSA 2023
- Power of Attorney Act, 1882
- Indian Partnership Act, 1932
- Indian Succession Act, 1925
- Negotiable Instruments Act, 1881
- Rent Control Acts (State-specific)
- Notaries Act, 1952
"""

import re
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field


@dataclass
class ChecklistItem:
    """A single statutory requirement."""

    element: str
    description: str
    statute_reference: str
    severity: str  # "mandatory" | "recommended" | "best_practice"
    present: bool = False
    details: str = ""


@dataclass
class StatutoryValidationResult:
    """Result of statutory validation against hard checklist."""

    document_type: str
    total_checks: int
    passed: int
    failed: int
    missing_elements: List[Dict[str, Any]] = field(default_factory=list)
    present_elements: List[Dict[str, Any]] = field(default_factory=list)
    non_compliance: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    compliance_score: Optional[float] = None  # None: no checklist for this type

    def to_dict(self) -> Dict[str, Any]:
        return {
            "document_type": self.document_type,
            "total_checks": self.total_checks,
            "passed": self.passed,
            "failed": self.failed,
            "missing_elements": self.missing_elements,
            "present_elements": self.present_elements,
            "non_compliance": self.non_compliance,
            "warnings": self.warnings,
            "compliance_score": None if self.compliance_score is None else round(self.compliance_score, 2),
        }


def format_score(score: Optional[float]) -> str:
    """Compliance score for display; None means no checklist exists for the type."""
    return "not assessed" if score is None else f"{score:.0%}"


# ============================================================================
# Statutory Checklists per Document Type
# ============================================================================

AFFIDAVIT_CHECKLIST: List[Dict[str, Any]] = [
    {
        "element": "Deponent Details",
        "description": "Full name, age, father's/husband's name, occupation, and address of the deponent",
        "statute": "Order XIX Rule 3 CPC; General practice",
        "severity": "mandatory",
        "patterns": [
            r"(?:i|deponent)\b.*\b(?:son|daughter|wife|s/o|d/o|w/o)\s*(?:of|:)",
            r"(?:name\s*of\s*(?:the\s*)?deponent|deponent\s*[:\-])",
            r"(?:aged?\s*(?:about\s*)?\d+|age\s*[:\-]?\s*\d+)",
            r"(?:resident|residing|address)\s*(?:of|at|[:\-])",
        ],
    },
    {
        "element": "Verification Clause",
        "description": "Verification stating contents are true to knowledge/belief, signed at specific place/date",
        "statute": "Order XIX Rule 3 CPC; Order VI Rule 15 CPC",
        "severity": "mandatory",
        "patterns": [
            r"verif(?:ied|ication)",
            r"true\s*to\s*(?:my|the\s*best\s*of\s*my)\s*knowledge",
            r"nothing\s*(?:material\s*)?has\s*been\s*(?:suppressed|concealed)",
        ],
    },
    {
        "element": "Place and Date",
        "description": "Place of execution and date must be mentioned",
        "statute": "Order XIX Rule 3 CPC",
        "severity": "mandatory",
        "patterns": [
            r"(?:place|verified\s*at)\s*[:\-]?\s*\w+",
            r"(?:date|dated)\s*[:\-]?\s*\d+",
        ],
    },
    {
        "element": "Oath Commissioner / Notary Attestation",
        "description": "Must be attested by Oath Commissioner, Notary Public, or Judicial Magistrate",
        "statute": "Notaries Act, 1952 Section 8; Oaths Act, 1969 Section 4",
        "severity": "mandatory",
        "patterns": [
            r"(?:oath\s*commissioner|notary\s*public|magistrate)",
            r"(?:before\s*me|attested\s*by|sworn\s*before)",
            r"(?:seal|stamp).*(?:notary|commissioner)",
        ],
    },
    {
        "element": "Stamp Paper",
        "description": "Must be on appropriate stamp paper (value varies by state)",
        "statute": "Indian Stamp Act, 1899 (Schedule I); State Stamp Acts",
        "severity": "mandatory",
        "patterns": [
            r"stamp\s*paper",
            r"(?:non[\s-]?judicial|e[\s-]?stamp)\s*(?:stamp\s*)?paper",
            r"stamp\s*(?:duty|value)",
            r"e[\s-]?stamp\s*certificate",
        ],
    },
    {
        "element": "Oath / Solemn Affirmation",
        "description": "Statement that contents are stated on oath/solemn affirmation",
        "statute": "Oaths Act, 1969 Section 4, 5, 6",
        "severity": "mandatory",
        "patterns": [
            r"(?:on\s*)?(?:solemn\s*)?(?:oath|affirm(?:ation)?)",
            r"solemnly\s*(?:affirm|declare|state)",
            r"do\s*(?:hereby\s*)?(?:solemnly\s*)?(?:affirm|declare|state\s*on\s*oath)",
        ],
    },
    {
        "element": "Jurisdiction Wording",
        "description": "Correct jurisdiction wording ('Before the [authority] at [place]')",
        "statute": "General drafting practice; Jurisdiction-specific requirements",
        "severity": "recommended",
        "patterns": [
            r"before\s*(?:the\s*)?(?:hon['\u2019]?ble|honourable)?\s*(?:court|authority|commissioner|magistrate|notary)",
            r"jurisdiction\b",
            r"(?:in\s*the\s*)?(?:matter\s*(?:of|before))",
        ],
    },
    {
        "element": "Signature of Deponent",
        "description": "Deponent's signature or thumb impression",
        "statute": "Order XIX Rule 3 CPC",
        "severity": "mandatory",
        "patterns": [
            r"(?:signature|signed|thumb\s*impression|deponent)",
            r"(?:sd/?[\-\s]|sign(?:ed)?)",
        ],
    },
]

SALE_DEED_CHECKLIST: List[Dict[str, Any]] = [
    {
        "element": "Vendor and Vendee Details",
        "description": "Complete details of seller(s) and buyer(s) — name, father's name, age, address",
        "statute": "Transfer of Property Act, 1882 Section 54; Indian Registration Act, 1908 Section 17",
        "severity": "mandatory",
        "patterns": [
            r"(?:vendor|seller|transferor)\b",
            r"(?:vendee|buyer|purchaser|transferee)\b",
            r"party\s*of\s*the\s*(?:first|second)\s*part",
        ],
    },
    {
        "element": "Property Description / Schedule",
        "description": "Complete description of immovable property — survey number, boundaries, area, measurements",
        "statute": "Transfer of Property Act, 1882 Section 54; Registration Act Section 21",
        "severity": "mandatory",
        "patterns": [
            r"(?:schedule\s*(?:of\s*)?property|property\s*schedule)",
            r"(?:survey\s*no|khasra\s*no|plot\s*no|flat\s*no|house\s*no)",
            r"(?:bounded\s*(?:by|on)|abuttals|boundaries)",
            r"(?:east|west|north|south)\s*(?:by|:)",
            r"(?:sq\.?\s*(?:ft|feet|meters|metres|yards)|hectare|acre|gunta|cent)",
        ],
    },
    {
        "element": "Sale Consideration",
        "description": "Clearly stated sale price/consideration amount",
        "statute": "Transfer of Property Act, 1882 Section 54",
        "severity": "mandatory",
        "patterns": [
            r"(?:sale\s*)?consideration\s*(?:amount|of|:)",
            r"(?:rs\.?|₹|inr)\s*[\d,]+",
            r"(?:total|agreed|purchase)\s*(?:price|amount|consideration)",
        ],
    },
    {
        "element": "Stamp Duty Payment",
        "description": "Proper stamp duty paid as per state stamp act and circle rate",
        "statute": "Indian Stamp Act, 1899 Section 3; State Stamp Acts",
        "severity": "mandatory",
        "patterns": [
            r"stamp\s*duty",
            r"(?:circle|guidance|market)\s*(?:rate|value)",
            r"e[\s-]?stamp",
            r"(?:registration|stamp)\s*(?:fees?|charges?)",
        ],
    },
    {
        "element": "Execution Clause",
        "description": "Signed and executed by parties in presence of witnesses",
        "statute": "Transfer of Property Act Section 54; Registration Act Section 32",
        "severity": "mandatory",
        "patterns": [
            r"(?:executed|signed)\s*(?:on|at|this)",
            r"(?:in\s*(?:the\s*)?presence\s*of|witnesses?)",
            r"(?:witness(?:es|eth)?)\s*(?:whereof|:)",
        ],
    },
    {
        "element": "Registration",
        "description": "Deed must be registered with Sub-Registrar",
        "statute": "Indian Registration Act, 1908 Section 17(1)(a) — compulsory for immovable property above Rs. 100",
        "severity": "mandatory",
        "patterns": [
            r"(?:sub[\s-]?)?registrar",
            r"registration\s*(?:number|no|#)",
            r"(?:registered|presented\s*for\s*registration)\s*(?:at|before|on)",
            r"book\s*no\b.*volume\s*no",
        ],
    },
    {
        "element": "Encumbrance Certificate",
        "description": "Confirmation that property is free from encumbrances",
        "statute": "General practice; Transfer of Property Act Section 55(1)(a)",
        "severity": "recommended",
        "patterns": [
            r"encumbrance\s*(?:certificate|free)",
            r"free\s*from\s*(?:all\s*)?(?:encumbrances|liens|charges|mortgages)",
            r"clear\s*(?:and\s*marketable\s*)?title",
        ],
    },
    {
        "element": "Witness Details",
        "description": "At least two witnesses with names and addresses",
        "statute": "Indian Registration Act, 1908 Section 32",
        "severity": "mandatory",
        "patterns": [
            r"witness(?:es)?\s*[:\-]",
            r"(?:in\s*(?:the\s*)?presence\s*of|attested\s*by)",
            r"witness\s*(?:no\.?\s*)?[12]\b",
        ],
    },
]

FIR_CHECKLIST: List[Dict[str, Any]] = [
    {
        "element": "FIR Number and Police Station",
        "description": "FIR number and name of police station where registered",
        "statute": "CrPC Section 154 / BNSS Section 173",
        "severity": "mandatory",
        "patterns": [
            r"(?:fir|f\.i\.r\.?)\s*(?:no\.?|number|#)\s*[:;]?\s*\d+",
            r"police\s*station\s*[:\-]?\s*\w+",
            r"p\.?s\.?\s*[:\-]?\s*\w+",
        ],
    },
    {
        "element": "Date and Time of Occurrence",
        "description": "Date and time when the offence was committed",
        "statute": "CrPC Section 154 / BNSS Section 173",
        "severity": "mandatory",
        "patterns": [
            r"(?:date|time)\s*(?:of|&)\s*(?:occurrence|incident|offence|crime)",
            r"(?:occurred|happened|took\s*place)\s*(?:on|at)",
        ],
    },
    {
        "element": "Date and Time of Reporting",
        "description": "Date and time when the FIR was registered",
        "statute": "CrPC Section 154 / BNSS Section 173",
        "severity": "mandatory",
        "patterns": [
            r"(?:date|time)\s*(?:of\s*)?(?:report(?:ing)?|registration|information)",
            r"(?:reported|registered|lodged)\s*(?:on|at)",
        ],
    },
    {
        "element": "Complainant Details",
        "description": "Full name, address, and contact of the complainant/informant",
        "statute": "CrPC Section 154 / BNSS Section 173",
        "severity": "mandatory",
        "patterns": [
            r"(?:complainant|informant)\s*(?:name|details|:)",
            r"(?:name\s*of\s*(?:the\s*)?(?:complainant|informant))",
        ],
    },
    {
        "element": "Place of Occurrence",
        "description": "Specific place/location where offence was committed",
        "statute": "CrPC Section 154 / BNSS Section 173",
        "severity": "mandatory",
        "patterns": [
            r"(?:place|location)\s*(?:of\s*)?(?:occurrence|incident|offence)",
            r"(?:at|near|in\s*front\s*of)\s*(?:the\s*)?(?:premises|house|shop|office)",
        ],
    },
    {
        "element": "Description of Offence / Facts",
        "description": "Detailed account of events constituting the offence",
        "statute": "CrPC Section 154 / BNSS Section 173",
        "severity": "mandatory",
        "patterns": [
            r"(?:brief\s*)?(?:facts|details|description|particulars)\s*(?:of\s*(?:the\s*)?)?(?:offence|crime|incident|complaint|case)",
            r"(?:gist\s*of\s*(?:the\s*)?complaint|substance\s*of\s*information)",
        ],
    },
    {
        "element": "Sections / Acts Applied",
        "description": "Relevant IPC/BNS sections and other applicable acts",
        "statute": "CrPC Section 154 / BNSS Section 173",
        "severity": "mandatory",
        "patterns": [
            r"(?:section|sec\.?|s\.)\s*\d+[A-Z]?\s*(?:ipc|bns|crpc|bnss|it\s*act)",
            r"(?:act|acts?)\s*(?:applied|sections?)\s*[:\-]",
            r"(?:under\s*)?(?:section|sec)\s*\d+",
        ],
    },
    {
        "element": "Accused Details",
        "description": "Known details of the accused, if available",
        "statute": "CrPC Section 154 / BNSS Section 173",
        "severity": "recommended",
        "patterns": [
            r"(?:accused|suspect)\s*(?:name|details|:)",
            r"(?:name\s*of\s*(?:the\s*)?accused)",
            r"(?:known|unknown)\s*(?:accused|person)",
        ],
    },
    {
        "element": "Signature of Informant",
        "description": "Signature or thumb impression of the complainant/informant",
        "statute": "CrPC Section 154(1) / BNSS Section 173",
        "severity": "mandatory",
        "patterns": [
            r"(?:signature|signed|thumb\s*impression)\s*(?:of\s*)?(?:informant|complainant)",
            r"(?:sd/?[\-\s]|sign(?:ed)?)",
        ],
    },
]

POWER_OF_ATTORNEY_CHECKLIST: List[Dict[str, Any]] = [
    {
        "element": "Principal Details",
        "description": "Full details of the person granting the power (principal/donor)",
        "statute": "Power of Attorney Act, 1882 Section 1A",
        "severity": "mandatory",
        "patterns": [
            r"(?:principal|donor|executant|grantor)\b",
            r"party\s*of\s*the\s*first\s*part",
            r"(?:i|we)\b.*\b(?:son|daughter|wife|s/o|d/o|w/o)\s*(?:of|:)",
        ],
    },
    {
        "element": "Attorney Details",
        "description": "Full details of the person receiving the power (attorney/agent/donee)",
        "statute": "Power of Attorney Act, 1882 Section 1A",
        "severity": "mandatory",
        "patterns": [
            r"(?:attorney|agent|donee|holder)\b",
            r"party\s*of\s*the\s*second\s*part",
            r"(?:appoint|nominate|constitute|authoriz[es])\b",
        ],
    },
    {
        "element": "Scope of Authority",
        "description": "Clear enumeration of powers granted (specific acts the attorney can perform)",
        "statute": "Power of Attorney Act, 1882 Section 2",
        "severity": "mandatory",
        "patterns": [
            r"(?:power|authority|right)\s*to\b",
            r"(?:authoriz(?:ed?|ing)|empower(?:ed?|ing))\s*to\b",
            r"(?:shall|may)\s*(?:have\s*(?:the\s*)?(?:power|authority|right))",
            r"(?:act|execute|sign|appear|represent|negotiate|sell|purchase|let|lease|mortgage)\s*(?:on|for)\s*(?:my|his|her|their)\s*behalf",
        ],
    },
    {
        "element": "Stamp Duty",
        "description": "Executed on proper stamp paper as per state stamp act",
        "statute": "Indian Stamp Act, 1899 Article 48; State Stamp Acts",
        "severity": "mandatory",
        "patterns": [
            r"stamp\s*(?:paper|duty)",
            r"e[\s-]?stamp",
            r"non[\s-]?judicial\s*stamp",
        ],
    },
    {
        "element": "Registration (if for immovable property)",
        "description": "POA authorizing sale of immovable property must be registered",
        "statute": "Indian Registration Act, 1908 Section 17(1)(g); Supreme Court in Suraj Lamp Industries v. State of Haryana (2012)",
        "severity": "mandatory",
        "patterns": [
            r"(?:sub[\s-]?)?registrar",
            r"registered\s*(?:at|before|on)",
            r"registration\s*(?:no|number|#)",
        ],
    },
    {
        "element": "Witness Attestation",
        "description": "Signed in presence of at least two witnesses",
        "statute": "General practice; Registration Act Section 32",
        "severity": "recommended",
        "patterns": [
            r"witness(?:es)?\s*[:\-]",
            r"(?:in\s*(?:the\s*)?presence\s*of|attested\s*by)",
        ],
    },
    {
        "element": "Revocability Clause",
        "description": "Whether POA is revocable or irrevocable must be stated",
        "statute": "Indian Contract Act, 1872 Section 201-203",
        "severity": "recommended",
        "patterns": [
            r"(?:ir)?revocable",
            r"(?:revoke|revocation|cancel(?:lation)?)\s*(?:of\s*(?:this\s*)?power)",
            r"(?:coupled\s*with\s*interest|interest\s*of\s*the\s*agent)",
        ],
    },
]

RENT_AGREEMENT_CHECKLIST: List[Dict[str, Any]] = [
    {
        "element": "Landlord / Lessor Details",
        "description": "Full name, address, and details of the property owner",
        "statute": "Transfer of Property Act, 1882 Section 105; State Rent Control Acts",
        "severity": "mandatory",
        "patterns": [
            r"(?:landlord|lessor|owner|licensor)\b",
            r"party\s*of\s*the\s*first\s*part",
        ],
    },
    {
        "element": "Tenant / Lessee Details",
        "description": "Full name, address, and details of the tenant",
        "statute": "Transfer of Property Act, 1882 Section 105; State Rent Control Acts",
        "severity": "mandatory",
        "patterns": [
            r"(?:tenant|lessee|licensee)\b",
            r"party\s*of\s*the\s*second\s*part",
        ],
    },
    {
        "element": "Property Description",
        "description": "Address, dimensions, and details of the rented premises",
        "statute": "Transfer of Property Act, 1882 Section 105",
        "severity": "mandatory",
        "patterns": [
            r"(?:premises|property|flat|house|apartment|office|shop)\s*(?:situated|located|at|bearing)",
            r"(?:address|location)\s*(?:of\s*(?:the\s*)?(?:premises|property))",
        ],
    },
    {
        "element": "Rent Amount",
        "description": "Monthly rent amount clearly stated",
        "statute": "Transfer of Property Act Section 105; State Rent Control Acts",
        "severity": "mandatory",
        "patterns": [
            r"(?:monthly\s*)?rent\s*(?:amount|of|:|\-)\s*(?:rs\.?|₹|inr)?\s*[\d,]+",
            r"(?:rs\.?|₹|inr)\s*[\d,]+\s*(?:per\s*month|monthly|p\.?m\.?)",
        ],
    },
    {
        "element": "Security Deposit",
        "description": "Security deposit amount and refund conditions",
        "statute": "State Rent Control Acts; General practice",
        "severity": "mandatory",
        "patterns": [
            r"(?:security|refundable)\s*deposit",
            r"(?:advance|caution)\s*(?:money|deposit|amount)",
        ],
    },
    {
        "element": "Tenure / Duration",
        "description": "Agreement period with start and end dates",
        "statute": "Transfer of Property Act Section 105, 106",
        "severity": "mandatory",
        "patterns": [
            r"(?:period|tenure|duration|term)\s*(?:of\s*(?:the\s*)?(?:agreement|lease|tenancy|licence))",
            r"(?:from|commencing|starting)\s*\d+.*(?:to|until|ending|expiring)\s*\d+",
            r"(?:for\s*a\s*period\s*of|for\s*\d+\s*(?:month|year))",
        ],
    },
    {
        "element": "Lock-in Period",
        "description": "Minimum period during which neither party can terminate",
        "statute": "General practice; Transfer of Property Act Section 106",
        "severity": "recommended",
        "patterns": [
            r"lock[\s-]?in\s*period",
            r"minimum\s*(?:period|tenure|stay)",
        ],
    },
    {
        "element": "Notice Period for Termination",
        "description": "Period of notice required for termination by either party",
        "statute": "Transfer of Property Act Section 106",
        "severity": "mandatory",
        "patterns": [
            r"notice\s*period",
            r"(?:one|two|three|\d+)\s*month(?:s)?\s*(?:prior\s*)?notice",
            r"(?:termination|vacating)\s*(?:upon|with|after)\s*(?:\d+|one|two|three)\s*month",
        ],
    },
    {
        "element": "Stamp Duty / Registration",
        "description": "Proper stamp duty and registration (mandatory if lease > 11 months in most states)",
        "statute": "Indian Registration Act Section 17(1)(d); Indian Stamp Act; State Stamp Acts",
        "severity": "mandatory",
        "patterns": [
            r"(?:stamp\s*(?:duty|paper)|e[\s-]?stamp)",
            r"(?:registered|registration)",
        ],
    },
    {
        "element": "Maintenance and Repairs",
        "description": "Allocation of responsibility for maintenance and repairs",
        "statute": "Transfer of Property Act Section 108; State Rent Control Acts",
        "severity": "recommended",
        "patterns": [
            r"(?:maintenance|repair|upkeep)\b",
            r"(?:structural|minor)\s*(?:repair|maintenance)",
        ],
    },
]

AGREEMENT_TO_SELL_CHECKLIST: List[Dict[str, Any]] = [
    {
        "element": "Seller and Buyer Details",
        "description": "Complete details of seller(s) and buyer(s)",
        "statute": "Indian Contract Act, 1872 Section 10; Transfer of Property Act Section 54",
        "severity": "mandatory",
        "patterns": [
            r"(?:seller|vendor|owner|transferor)\b",
            r"(?:buyer|purchaser|vendee|transferee)\b",
        ],
    },
    {
        "element": "Property Description",
        "description": "Complete property details with boundaries and survey numbers",
        "statute": "Transfer of Property Act Section 54",
        "severity": "mandatory",
        "patterns": [
            r"(?:schedule\s*(?:of\s*)?property|property\s*description)",
            r"(?:survey\s*no|khasra\s*no|plot\s*no|flat\s*no)",
            r"(?:bounded\s*(?:by|on)|boundaries)",
        ],
    },
    {
        "element": "Total Sale Consideration",
        "description": "Agreed total price for the property",
        "statute": "Indian Contract Act Section 2(d); Transfer of Property Act Section 54",
        "severity": "mandatory",
        "patterns": [
            r"(?:total\s*)?(?:sale\s*)?consideration\s*(?:amount|of|:)",
            r"(?:agreed|total|purchase)\s*(?:price|amount|value)",
        ],
    },
    {
        "element": "Earnest Money / Token Amount",
        "description": "Advance payment details and terms",
        "statute": "Indian Contract Act Section 2(d); General practice",
        "severity": "mandatory",
        "patterns": [
            r"(?:earnest|token)\s*(?:money|amount)",
            r"(?:advance\s*(?:payment|amount))",
            r"(?:booking\s*amount)",
        ],
    },
    {
        "element": "Payment Schedule",
        "description": "Timeline for balance payment instalments",
        "statute": "Indian Contract Act; General practice",
        "severity": "recommended",
        "patterns": [
            r"(?:payment\s*schedule|instalment|installment)",
            r"(?:balance\s*(?:amount|consideration|payment))",
        ],
    },
    {
        "element": "Possession Date / Timeline",
        "description": "Date or timeline for handover of possession",
        "statute": "Transfer of Property Act Section 55(1)(f)",
        "severity": "mandatory",
        "patterns": [
            r"(?:possession|handover|hand[\s-]?over)\s*(?:date|on|shall|will|to\s*be)",
            r"(?:vacant|physical|actual)\s*possession",
        ],
    },
    {
        "element": "Time is of the Essence Clause",
        "description": "Whether time is of the essence for performance",
        "statute": "Indian Contract Act Section 55",
        "severity": "recommended",
        "patterns": [
            r"time\s*(?:is|being|shall\s*be)\s*(?:of\s*)?the\s*essence",
        ],
    },
    {
        "element": "Title Warranty",
        "description": "Seller's warranty of clear and marketable title",
        "statute": "Transfer of Property Act Section 55(1)(a)",
        "severity": "mandatory",
        "patterns": [
            r"(?:clear|marketable|good|valid)\s*title",
            r"(?:free\s*from\s*(?:all\s*)?encumbrances)",
            r"(?:title\s*(?:deed|document|certificate))",
        ],
    },
    {
        "element": "Default / Breach Clause",
        "description": "Consequences of default by either party",
        "statute": "Indian Contract Act Section 73, 74",
        "severity": "recommended",
        "patterns": [
            r"(?:default|breach)\s*(?:by|of|clause)",
            r"(?:forfeiture|forfeit)",
            r"(?:specific\s*performance)",
            r"(?:liquidated\s*damages)",
        ],
    },
]

NOTICE_CHECKLIST: List[Dict[str, Any]] = [
    {
        "element": "Sender (Advocate / Client) Details",
        "description": "Name and address of the person issuing the notice or their advocate",
        "statute": "CPC Section 80 (for suits against government); General practice",
        "severity": "mandatory",
        "patterns": [
            r"(?:advocate|counsel|lawyer|attorney)\s*(?:for|on\s*behalf)",
            r"(?:on\s*behalf\s*of\s*(?:my|our)\s*client)",
            r"(?:client\s*(?:details|name|:))",
            r"(?:issued\s*by|from\s*the\s*office\s*of)",
        ],
    },
    {
        "element": "Recipient / Noticee Details",
        "description": "Name and address of the person to whom notice is addressed",
        "statute": "General practice; CPC Order V",
        "severity": "mandatory",
        "patterns": [
            r"(?:to|addressed\s*to|noticee)\s*[:\-]?\s*\w+",
            r"(?:shri|smt|ms|mr|m/s)\b.*\b(?:address|r/?o|s/?o|d/?o|w/?o)",
        ],
    },
    {
        "element": "Subject / Heading",
        "description": "Clear subject line indicating nature of notice",
        "statute": "General practice",
        "severity": "recommended",
        "patterns": [
            r"(?:subject|re|ref)\s*[:\-]",
            r"(?:in\s*the\s*matter\s*of)",
        ],
    },
    {
        "element": "Facts / Background",
        "description": "Statement of facts giving rise to the notice",
        "statute": "General practice; Specific statute under which notice is served",
        "severity": "mandatory",
        "patterns": [
            r"(?:facts|background|whereas|it\s*is\s*(?:stated|submitted))",
            r"(?:my\s*client\s*(?:states|submits|instructs))",
        ],
    },
    {
        "element": "Demand / Relief Sought",
        "description": "Clear statement of what is demanded or sought",
        "statute": "Dependent on underlying cause; CPC Section 80 for government notices",
        "severity": "mandatory",
        "patterns": [
            r"(?:demand|require|call\s*upon|request|direct(?:ed)?)\s*(?:you|the\s*noticee)",
            r"(?:hereby\s*(?:demand|require|call\s*upon))",
            r"(?:failing\s*which|otherwise)",
        ],
    },
    {
        "element": "Time / Period for Compliance",
        "description": "Specific time period given for compliance",
        "statute": "CPC Section 80 (60 days for government); Specific Acts",
        "severity": "mandatory",
        "patterns": [
            r"(?:within\s*\d+\s*days)",
            r"(?:period\s*of\s*\d+\s*days)",
            r"(?:on\s*or\s*before\s*\d+)",
        ],
    },
    {
        "element": "Consequence of Non-Compliance",
        "description": "Statement of legal consequences if notice is ignored",
        "statute": "General practice",
        "severity": "recommended",
        "patterns": [
            r"(?:legal\s*(?:action|proceedings|recourse))",
            r"(?:at\s*your\s*(?:own\s*)?(?:risk|cost|peril))",
            r"(?:constrained|compelled|obliged)\s*to\s*(?:initiate|file|approach)",
        ],
    },
    {
        "element": "Statutory Reference",
        "description": "Reference to the specific section/act under which notice is issued",
        "statute": "Depends on underlying cause of action",
        "severity": "mandatory",
        "patterns": [
            r"(?:under\s*(?:section|sec\.?|s\.)\s*\d+)",
            r"(?:section\s*\d+\s*(?:of\s*(?:the\s*)?)?(?:cpc|crpc|ipc|act))",
            r"(?:(?:as\s*)?(?:per|under)\s*(?:the\s*)?(?:provisions?\s*of|law))",
        ],
    },
]

COURT_ORDER_CHECKLIST: List[Dict[str, Any]] = [
    {
        "element": "Court Name and Jurisdiction",
        "description": "Name of the court and its jurisdictional details",
        "statute": "CPC Section 9, 15-20; Constitution of India Article 226, 227, 32",
        "severity": "mandatory",
        "patterns": [
            r"(?:in\s*the\s*(?:hon[\'\u2019]?ble\s*)?)?(?:high\s*court|supreme\s*court|district\s*court|sessions\s*court|magistrate|tribunal|consumer|family\s*court)",
            r"(?:at\s*\w+(?:\s*bench)?)",
        ],
    },
    {
        "element": "Case Number and Year",
        "description": "Case/petition number and filing year",
        "statute": "Court registry rules; General practice",
        "severity": "mandatory",
        "patterns": [
            r"(?:case|petition|suit|appeal|revision|writ)\s*(?:no\.?|number|#)\s*[:\-]?\s*\d+",
            r"(?:of\s*(?:the\s*year\s*)?\d{4}|\d{4})",
        ],
    },
    {
        "element": "Parties (Petitioner / Respondent)",
        "description": "Names and descriptions of parties to the case",
        "statute": "CPC Order I; CrPC",
        "severity": "mandatory",
        "patterns": [
            r"(?:petitioner|respondent|plaintiff|defendant|appellant|opposite\s*party|complainant)\b",
            r"versus|vs\.?",
        ],
    },
    {
        "element": "Coram (Presiding Judge/Bench)",
        "description": "Name(s) of the judge(s) hearing the case",
        "statute": "Court rules; General practice",
        "severity": "mandatory",
        "patterns": [
            r"(?:coram|before)\s*[:\-]?\s*(?:hon|justice|judge|shri|smt)",
            r"(?:hon[\'\u2019]?ble\s*(?:mr\.?\s*)?justice)\s*\w+",
            r"(?:bench\s*(?:of|:))",
        ],
    },
    {
        "element": "Date of Order",
        "description": "Date on which the order/judgment was passed",
        "statute": "Court rules; General practice",
        "severity": "mandatory",
        "patterns": [
            r"(?:dated?\s*[:\-]?\s*\d+[\./\-]\d+[\./\-]\d+)",
            r"(?:order\s*dated|judgment\s*dated|pronounced\s*on|decided\s*on|delivered\s*on)",
        ],
    },
    {
        "element": "Operative Part / Disposition",
        "description": "The actual order/direction/disposition of the court",
        "statute": "CPC Order XX; CrPC Section 354",
        "severity": "mandatory",
        "patterns": [
            r"(?:ordered\s*(?:that|as\s*follows)|it\s*is\s*(?:hereby\s*)?(?:ordered|directed|decreed))",
            r"(?:petition\s*is\s*(?:allowed|dismissed|disposed))",
            r"(?:appeal\s*(?:is|stands)\s*(?:allowed|dismissed))",
            r"(?:accordingly|disposed\s*of|stand(?:s)?\s*disposed)",
        ],
    },
    {
        "element": "Reasoning / Ratio Decidendi",
        "description": "Court's reasoning and legal analysis",
        "statute": "CPC Order XX Rule 4; CrPC Section 354(1)",
        "severity": "mandatory",
        "patterns": [
            r"(?:having\s*heard|having\s*considered|upon\s*(?:hearing|consideration))",
            r"(?:in\s*(?:my|our)\s*(?:view|opinion|considered\s*view))",
            r"(?:for\s*the\s*(?:reasons|foregoing)\s*(?:stated|discussed))",
        ],
    },
]

WILL_CHECKLIST: List[Dict[str, Any]] = [
    {
        "element": "Testator Details",
        "description": "Full name, age, address of the person making the will",
        "statute": "Indian Succession Act, 1925 Section 59",
        "severity": "mandatory",
        "patterns": [
            r"(?:testator|testatrix)\b",
            r"(?:i\b.*\bhereby\s*(?:make|declare|revoke).*(?:will|testament))",
            r"(?:last\s*will\s*and\s*testament\s*of)",
        ],
    },
    {
        "element": "Sound Mind Declaration",
        "description": "Declaration that testator is of sound mind and not under undue influence",
        "statute": "Indian Succession Act, 1925 Section 59",
        "severity": "mandatory",
        "patterns": [
            r"(?:sound\s*(?:disposing\s*)?mind)",
            r"(?:free\s*will|without\s*(?:any\s*)?(?:coercion|undue\s*influence|pressure))",
            r"(?:voluntarily|of\s*(?:my|his|her)\s*own\s*(?:free\s*)?(?:will|volition))",
        ],
    },
    {
        "element": "Beneficiary Details",
        "description": "Clear identification of beneficiaries",
        "statute": "Indian Succession Act, 1925 Section 74-82",
        "severity": "mandatory",
        "patterns": [
            r"(?:beneficiar(?:y|ies))\b",
            r"(?:bequeath|devise|give|leave)\s*(?:to|unto)",
            r"(?:my\s*(?:son|daughter|wife|husband|brother|sister|mother|father))",
        ],
    },
    {
        "element": "Property / Asset Description",
        "description": "Clear description of all properties and assets being bequeathed",
        "statute": "Indian Succession Act, 1925 Section 74",
        "severity": "mandatory",
        "patterns": [
            r"(?:property|asset|estate|belonging)",
            r"(?:movable|immovable)\s*property",
            r"(?:bank\s*account|fixed\s*deposit|shares|mutual\s*fund|jewel|ornament)",
        ],
    },
    {
        "element": "Executor Appointment",
        "description": "Appointment of executor to carry out the will",
        "statute": "Indian Succession Act, 1925 Section 59, 222",
        "severity": "recommended",
        "patterns": [
            r"(?:executor|executrix)\b",
            r"(?:appoint|nominate)\s*(?:as\s*)?executor",
        ],
    },
    {
        "element": "Revocation of Previous Wills",
        "description": "Clause revoking all previous wills and codicils",
        "statute": "Indian Succession Act, 1925 Section 62, 70",
        "severity": "recommended",
        "patterns": [
            r"(?:revoke|cancel)\s*(?:all\s*)?(?:former|previous|earlier|prior)\s*(?:wills|testament|codicil)",
            r"(?:this\s*is\s*my\s*last\s*(?:will|testament))",
        ],
    },
    {
        "element": "Two Witnesses",
        "description": "Will must be attested by at least two witnesses",
        "statute": "Indian Succession Act, 1925 Section 63",
        "severity": "mandatory",
        "patterns": [
            r"(?:witness(?:es)?)\s*[:\-]",
            r"(?:attested|signed)\s*(?:by|in\s*(?:the\s*)?presence\s*of)\s*(?:two|2)",
            r"(?:witness\s*(?:no\.?\s*)?[12])",
        ],
    },
    {
        "element": "Testator Signature",
        "description": "Testator's signature or mark at the foot of the will",
        "statute": "Indian Succession Act, 1925 Section 63(a)",
        "severity": "mandatory",
        "patterns": [
            r"(?:sign(?:ed|ature)?|mark)\s*(?:of\s*(?:the\s*)?(?:testator|testatrix))",
            r"(?:sd/?[\-\s]|sign(?:ed)?)",
        ],
    },
]


PARTNERSHIP_DEED_CHECKLIST: List[Dict[str, Any]] = [
    {
        "element": "Firm Name and Business",
        "description": "Name of the firm and the nature of its business",
        "statute": "Indian Partnership Act, 1932 Section 4",
        "severity": "mandatory",
        "patterns": [r"firm\s*name", r"name\s*of\s*the\s*firm", r"(?:carry|carrying)\s*on\s*(?:the\s*)?business", r"nature\s*of\s*(?:the\s*)?business"],
    },
    {
        "element": "Partners' Details",
        "description": "Names, addresses and capacities of all partners",
        "statute": "Indian Partnership Act, 1932 Section 4; Section 58",
        "severity": "mandatory",
        "patterns": [r"partners?\b.*\b(?:residing|resident|address)", r"first\s*partner", r"second\s*partner"],
    },
    {
        "element": "Capital Contribution",
        "description": "Capital contributed by each partner",
        "statute": "Indian Partnership Act, 1932 Section 13",
        "severity": "mandatory",
        "patterns": [r"capital\s*contribution", r"contribut(?:e|es|ed|ion)\s*(?:the\s*)?capital", r"capital\s*of\s*(?:rs\.?|inr|₹)"],
    },
    {
        "element": "Profit and Loss Sharing",
        "description": "Ratio in which profits and losses are shared",
        "statute": "Indian Partnership Act, 1932 Sections 13(b), 13(e)",
        "severity": "mandatory",
        "patterns": [r"profit[\s-]*(?:and|&)?[\s-]*loss", r"profit[\s-]?sharing", r"share\s*(?:the\s*)?profits?"],
    },
    {
        "element": "Duration / Dissolution",
        "description": "Whether the firm is at will or for a fixed term, and how it dissolves",
        "statute": "Indian Partnership Act, 1932 Sections 7, 39-44",
        "severity": "recommended",
        "patterns": [r"partnership\s*at\s*will", r"dissol(?:ve|ved|ution)", r"duration\s*of\s*(?:the\s*)?partnership"],
    },
    {
        "element": "Execution and Stamp Duty",
        "description": "Signed by all partners on appropriately stamped paper (registration of the firm is optional but advisable)",
        "statute": "Indian Stamp Act, 1899 / State Stamp Acts; Indian Partnership Act, 1932 Section 58",
        "severity": "recommended",
        "patterns": [r"stamp\s*paper", r"e[\s-]?stamp", r"witness(?:es)?", r"in\s*witness\s*whereof"],
    },
]

BAIL_APPLICATION_CHECKLIST: List[Dict[str, Any]] = [
    {
        "element": "Court and Case Details",
        "description": "Court, case/FIR number, police station and sections charged",
        "statute": "Section 437/439 CrPC; Sections 480/483 BNSS",
        "severity": "mandatory",
        "patterns": [r"fir\s*(?:no|number)", r"crime\s*no", r"police\s*station", r"case\s*no"],
    },
    {
        "element": "Statutory Provision Invoked",
        "description": "The provision under which bail is sought (regular, anticipatory or default bail)",
        "statute": "Sections 437, 438, 439 CrPC / Sections 480, 482, 483 BNSS",
        "severity": "mandatory",
        "patterns": [r"section\s*(?:437|438|439|167)\b", r"section\s*(?:480|482|483|187)\b", r"anticipatory\s*bail", r"regular\s*bail", r"default\s*bail"],
    },
    {
        "element": "Grounds for Bail",
        "description": "Specific grounds (false implication, no flight risk, cooperation with investigation, parity, custody period)",
        "statute": "Sections 437(1), 439 CrPC; Gurbaksh Singh Sibbia v. State of Punjab (1980)",
        "severity": "mandatory",
        "patterns": [r"grounds?\s*(?:for|of)\s*bail", r"falsely\s*implicated", r"no\s*(?:risk|chance)\s*of\s*(?:absconding|fleeing|tampering)", r"cooperate\s*with\s*the\s*investigation"],
    },
    {
        "element": "Custody Status",
        "description": "Date of arrest / period in custody, or apprehension of arrest for anticipatory bail",
        "statute": "Section 437 CrPC; Section 482 BNSS",
        "severity": "mandatory",
        "patterns": [r"(?:arrested|in\s*custody)\s*(?:on|since)", r"apprehen(?:d|sion)\s*of\s*arrest", r"judicial\s*custody", r"police\s*custody"],
    },
    {
        "element": "Undertaking / Conditions",
        "description": "Willingness to furnish surety and abide by conditions",
        "statute": "Section 437(3) CrPC; Section 480 BNSS",
        "severity": "recommended",
        "patterns": [r"surety", r"bail\s*bond", r"undertak(?:e|ing)", r"abide\s*by\s*(?:any|the)\s*conditions?"],
    },
    {
        "element": "Prior Applications",
        "description": "Whether any earlier bail application was filed or rejected",
        "statute": "Supreme Court practice directions",
        "severity": "recommended",
        "patterns": [r"(?:no\s*)?(?:earlier|previous|prior)\s*(?:bail\s*)?application", r"first\s*bail\s*application"],
    },
]

COMPLAINT_CHECKLIST: List[Dict[str, Any]] = [
    {
        "element": "Magistrate and Parties",
        "description": "Court addressed and full names/addresses of complainant and accused",
        "statute": "Section 200 CrPC; Section 223 BNSS",
        "severity": "mandatory",
        "patterns": [r"(?:before|to)\s*the\s*(?:hon\'?ble\s*)?(?:chief\s*)?(?:judicial\s*)?magistrate", r"complainant", r"accused"],
    },
    {
        "element": "Facts and Offence Alleged",
        "description": "Chronological facts with the offence(s) and sections alleged",
        "statute": "Section 200 CrPC; Section 223 BNSS",
        "severity": "mandatory",
        "patterns": [r"facts?\s*of\s*the\s*case", r"offen[cs]es?\s*(?:under|punishable)", r"section\s*\d+\s*(?:ipc|bns|of)"],
    },
    {
        "element": "Jurisdiction",
        "description": "Territorial jurisdiction of the court",
        "statute": "Sections 177-184 CrPC; Sections 197-203 BNSS",
        "severity": "mandatory",
        "patterns": [r"jurisdiction", r"cause\s*of\s*action\s*(?:arose|arisen)"],
    },
    {
        "element": "Prayer",
        "description": "Relief sought (cognizance, summoning the accused, or investigation under Section 156(3))",
        "statute": "Sections 156(3), 190, 200 CrPC; Sections 175(3), 210, 223 BNSS",
        "severity": "mandatory",
        "patterns": [r"prayer", r"it\s*is\s*(?:therefore\s*)?prayed", r"take\s*cognizance", r"summon\s*the\s*accused"],
    },
    {
        "element": "Verification / Affidavit",
        "description": "Verification of the complaint and supporting affidavit where seeking Section 156(3) directions",
        "statute": "Priyanka Srivastava v. State of U.P. (2015) 6 SCC 287",
        "severity": "recommended",
        "patterns": [r"verif(?:ied|ication)", r"affidavit"],
    },
]

CHARGESHEET_CHECKLIST: List[Dict[str, Any]] = [
    {
        "element": "Case Particulars",
        "description": "Police station, FIR number and date, sections of law",
        "statute": "Section 173(2) CrPC; Section 193(3) BNSS",
        "severity": "mandatory",
        "patterns": [r"fir\s*(?:no|number)", r"police\s*station", r"sections?\s*of\s*law"],
    },
    {
        "element": "Accused Details",
        "description": "Names of the accused, their status (arrested/bailed/absconding) and offences charged",
        "statute": "Section 173(2)(a) CrPC",
        "severity": "mandatory",
        "patterns": [r"accused", r"name\s*of\s*(?:the\s*)?accused"],
    },
    {
        "element": "Witness List",
        "description": "Names of the persons acquainted with the facts (prosecution witnesses)",
        "statute": "Section 173(2)(b) CrPC; Section 193(3)(b) BNSS",
        "severity": "mandatory",
        "patterns": [r"list\s*of\s*witnesses", r"witnesses", r"cited\s*witnesses"],
    },
    {
        "element": "Evidence and Documents",
        "description": "Documents and material objects relied upon",
        "statute": "Section 173(5) CrPC; Section 193(6) BNSS",
        "severity": "mandatory",
        "patterns": [r"(?:list\s*of\s*)?documents", r"material\s*(?:objects|exhibits)", r"seized", r"forensic|fsl"],
    },
    {
        "element": "Investigating Officer Signature",
        "description": "Report signed by the investigating officer / officer in charge",
        "statute": "Section 173(2) CrPC",
        "severity": "recommended",
        "patterns": [r"investigating\s*officer", r"officer\s*in\s*charge", r"forwarded\s*(?:by|to)"],
    },
    {
        "element": "Sanction (where required)",
        "description": "Sanction for prosecution where the law requires it (e.g. public servants)",
        "statute": "Section 197 CrPC; Section 218 BNSS",
        "severity": "best_practice",
        "patterns": [r"sanction\s*for\s*prosecution", r"sanction\s*order"],
    },
]

# ============================================================================
# Checklist Registry
# ============================================================================

CHECKLIST_REGISTRY: Dict[str, List[Dict[str, Any]]] = {
    "Affidavit": AFFIDAVIT_CHECKLIST,
    "Sale Deed": SALE_DEED_CHECKLIST,
    "FIR": FIR_CHECKLIST,
    "Power of Attorney": POWER_OF_ATTORNEY_CHECKLIST,
    "Rent Agreement": RENT_AGREEMENT_CHECKLIST,
    "Agreement to Sell": AGREEMENT_TO_SELL_CHECKLIST,
    "Notice (CrPC/CPC)": NOTICE_CHECKLIST,
    "Court Order / Judgment": COURT_ORDER_CHECKLIST,
    "Will / Testament": WILL_CHECKLIST,
    "Partnership Deed": PARTNERSHIP_DEED_CHECKLIST,
    "Bail Application": BAIL_APPLICATION_CHECKLIST,
    "Complaint (CrPC)": COMPLAINT_CHECKLIST,
    "Chargesheet": CHARGESHEET_CHECKLIST,
}


class StatutoryValidator:
    """
    Rule-based statutory requirements validator (Layer 2).

    NO LLM dependency. Pure pattern matching against statutory checklists.
    LLMs reason — rules verify.
    """

    def __init__(self):
        self.checklists = CHECKLIST_REGISTRY

    def validate(
        self, document_text: str, document_type: str
    ) -> StatutoryValidationResult:
        """
        Validate a document against its statutory checklist.

        Args:
            document_text: Full text of the document
            document_type: Classified document type from Layer 1

        Returns:
            StatutoryValidationResult with detailed compliance analysis
        """
        checklist = self.checklists.get(document_type)
        if not checklist:
            return StatutoryValidationResult(
                document_type=document_type,
                total_checks=0,
                passed=0,
                failed=0,
                warnings=[
                    f"No statutory checklist available for document type: {document_type}. "
                    f"Supported types: {', '.join(self.checklists.keys())}"
                ],
                compliance_score=None,
            )

        text = document_text.lower()
        missing_elements: List[Dict[str, Any]] = []
        present_elements: List[Dict[str, Any]] = []
        non_compliance: List[Dict[str, Any]] = []
        warnings: List[str] = []

        mandatory_total = 0
        mandatory_passed = 0

        for item in checklist:
            element = item["element"]
            description = item["description"]
            statute = item["statute"]
            severity = item["severity"]
            patterns = item["patterns"]

            # Check if any pattern matches
            found = False
            matched_patterns = []

            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    found = True
                    matched_patterns.append(pattern)

            if severity == "mandatory":
                mandatory_total += 1

            entry = {
                "element": element,
                "description": description,
                "statute_reference": statute,
                "severity": severity,
            }

            if found:
                if severity == "mandatory":
                    mandatory_passed += 1
                present_elements.append(entry)
            else:
                if severity == "mandatory":
                    missing_elements.append(entry)
                elif severity == "recommended":
                    non_compliance.append(entry)

        # Additional non-compliance checks
        additional_nc = self._check_additional_compliance(document_type, text)
        non_compliance.extend(additional_nc)

        total_checks = len(checklist)
        passed = len(present_elements)
        failed = total_checks - passed

        # Compliance score: mandatory items weighted heavily
        if mandatory_total > 0:
            compliance_score = mandatory_passed / mandatory_total
        else:
            compliance_score = 1.0 if passed > 0 else 0.0

        return StatutoryValidationResult(
            document_type=document_type,
            total_checks=total_checks,
            passed=passed,
            failed=failed,
            missing_elements=missing_elements,
            present_elements=present_elements,
            non_compliance=non_compliance,
            warnings=warnings,
            compliance_score=compliance_score,
        )

    def _check_additional_compliance(
        self, document_type: str, text: str
    ) -> List[Dict[str, Any]]:
        """Run additional rule-based compliance checks specific to document type."""
        issues = []

        if document_type == "Affidavit":
            # Check if affidavit uses first person consistently
            if not re.search(
                r"\bi\b.*\b(?:state|affirm|declare|depose)\b", text, re.IGNORECASE
            ):
                issues.append(
                    {
                        "element": "First Person Narration",
                        "description": "Affidavit should be in first person (I state/affirm/declare/depose...)",
                        "statute_reference": "Order XIX Rule 3 CPC — standard drafting practice",
                        "severity": "recommended",
                    }
                )

        elif document_type == "Sale Deed":
            # Check if consideration amount is in words AND figures
            has_amount_figures = bool(re.search(r"(?:rs\.?|₹|inr)\s*[\d,]+", text))
            has_amount_words = bool(
                re.search(
                    r"(?:rupees?\s*(?:one|two|three|four|five|six|seven|eight|nine|ten|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|lakh|crore))",
                    text,
                    re.IGNORECASE,
                )
            )
            if has_amount_figures and not has_amount_words:
                issues.append(
                    {
                        "element": "Consideration in Words",
                        "description": "Sale consideration should be stated both in figures and words to avoid disputes",
                        "statute_reference": "General drafting practice; Indian Stamp Act valuation requirements",
                        "severity": "recommended",
                    }
                )

        elif document_type == "Rent Agreement":
            # Check for rent escalation clause
            if not re.search(
                r"(?:escalation|increment|increase|hike|revision)\s*(?:of\s*)?(?:rent|rental)",
                text,
            ):
                issues.append(
                    {
                        "element": "Rent Escalation Clause",
                        "description": "No rent escalation/increment clause found. Standard practice is 5-10% annual escalation.",
                        "statute_reference": "General practice; State Rent Control Acts",
                        "severity": "recommended",
                    }
                )

        elif document_type == "Power of Attorney":
            # Check for duration/validity period
            if not re.search(
                r"(?:valid(?:ity)?|duration|effective|in\s*force)\s*(?:for|till|until|period|from)",
                text,
            ):
                issues.append(
                    {
                        "element": "Validity Period",
                        "description": "No validity period or duration specified for the Power of Attorney",
                        "statute_reference": "Power of Attorney Act, 1882; General practice",
                        "severity": "recommended",
                    }
                )

        return issues


# Singleton
_validator: Optional[StatutoryValidator] = None


def get_statutory_validator() -> StatutoryValidator:
    """Get or create the statutory validator singleton."""
    global _validator
    if _validator is None:
        _validator = StatutoryValidator()
    return _validator
