"""
Indian Law RAG Tool (Layer 2 — Tool 3: retrieve_indian_law_context)

RAG-based retrieval of Indian legal context for document validation.
Searches across:
- Bare Acts (via Indian Kanoon API)
- State-specific Stamp Acts
- Case law and judicial precedents
- Statutory requirements and forms

Uses the existing IndianKanoonTool + CrimeRAGSystem as underlying data sources
but provides a validation-focused interface.
"""

import re
import asyncio
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field


@dataclass
class LawReference:
    """A single reference to an Indian law, act, or case."""

    title: str
    act_name: str
    section: str
    relevance: str  # Why this reference is relevant to the validation
    excerpt: str
    url: str = ""
    source_type: str = ""  # "bare_act", "case_law", "stamp_act", "rules_forms"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "act_name": self.act_name,
            "section": self.section,
            "relevance": self.relevance,
            "excerpt": self.excerpt,
            "url": self.url,
            "source_type": self.source_type,
        }


@dataclass
class IndianLawContext:
    """Aggregated legal context for document validation."""

    document_type: str
    references: List[LawReference] = field(default_factory=list)
    applicable_acts: List[str] = field(default_factory=list)
    applicable_sections: List[str] = field(default_factory=list)
    state_specific_notes: List[str] = field(default_factory=list)
    precedent_notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "document_type": self.document_type,
            "references": [r.to_dict() for r in self.references],
            "applicable_acts": self.applicable_acts,
            "applicable_sections": self.applicable_sections,
            "state_specific_notes": self.state_specific_notes,
            "precedent_notes": self.precedent_notes,
        }


# ============================================================================
# Mapping: Document Type → Applicable Acts and Key Sections
# ============================================================================

DOCUMENT_LAW_MAP: Dict[str, Dict[str, Any]] = {
    "Affidavit": {
        "acts": [
            "Code of Civil Procedure, 1908",
            "Oaths Act, 1969",
            "Notaries Act, 1952",
            "Indian Stamp Act, 1899",
        ],
        "key_sections": [
            "Order XIX CPC — Affidavits",
            "Order VI Rule 15 CPC — Verification of pleadings",
            "Section 4 Oaths Act — Persons empowered to administer oaths",
            "Section 8 Notaries Act — Functions of notaries",
        ],
        "search_queries": [
            "affidavit verification clause mandatory CPC Order XIX",
            "defective affidavit not on proper stamp paper",
            "affidavit without notary attestation validity",
        ],
        "key_precedents": [
            "A.K. K. Nambiar v. Union of India (1970) — Affidavit as evidence",
            "Barium Chemicals Ltd v. Company Law Board (1967) — Requirements of valid affidavit",
        ],
    },
    "Sale Deed": {
        "acts": [
            "Transfer of Property Act, 1882",
            "Indian Registration Act, 1908",
            "Indian Stamp Act, 1899",
            "Indian Contract Act, 1872",
        ],
        "key_sections": [
            "Section 54 TPA — Sale of immovable property",
            "Section 17(1)(a) Registration Act — Documents of which registration is compulsory",
            "Section 55 TPA — Rights and liabilities of buyer and seller",
            "Section 3 Indian Stamp Act — Instruments chargeable with duty",
        ],
        "search_queries": [
            "sale deed registration compulsory Section 17 Registration Act",
            "sale deed stamp duty deficiency consequences",
            "unregistered sale deed validity Transfer of Property Act",
        ],
        "key_precedents": [
            "Suraj Lamp & Industries v. State of Haryana (2012) — Sale through GPA/POA void",
            "K. Ramaswami Gounder v. Arumugam (1999) — Registration of sale deed mandatory",
        ],
    },
    "FIR": {
        "acts": [
            "Code of Criminal Procedure, 1973 (CrPC)",
            "Bharatiya Nagarik Suraksha Sanhita, 2023 (BNSS)",
            "Indian Penal Code, 1860 (IPC)",
            "Bharatiya Nyaya Sanhita, 2023 (BNS)",
        ],
        "key_sections": [
            "Section 154 CrPC / Section 173 BNSS — Information in cognizable cases",
            "Section 155 CrPC / Section 174 BNSS — Non-cognizable cases",
            "Section 156 CrPC / Section 175 BNSS — Police officer's power to investigate",
            "Section 157 CrPC / Section 176 BNSS — Procedure for investigation",
        ],
        "search_queries": [
            "FIR mandatory requirements Section 154 CrPC",
            "delayed FIR effect on prosecution",
            "FIR registered without jurisdiction",
        ],
        "key_precedents": [
            "Lalita Kumari v. Govt. of U.P. (2014) — Mandatory registration of FIR for cognizable offence",
            "State of Andhra Pradesh v. Punati Ramulu (1993) — FIR not encyclopedia of events",
        ],
    },
    "Power of Attorney": {
        "acts": [
            "Power of Attorney Act, 1882",
            "Indian Registration Act, 1908",
            "Indian Stamp Act, 1899",
            "Indian Contract Act, 1872",
        ],
        "key_sections": [
            "Section 1A PA Act — Definitions",
            "Section 2 PA Act — Execution of power of attorney",
            "Section 17(1)(g) Registration Act — POA for immovable property",
            "Section 201-203 Indian Contract Act — Termination of agency",
        ],
        "search_queries": [
            "power of attorney immovable property registration required",
            "irrevocable power of attorney coupled with interest",
            "GPA sale immovable property Supreme Court",
        ],
        "key_precedents": [
            "Suraj Lamp & Industries v. State of Haryana (2012) — POA-based sale is not valid sale",
            "Bryant v. Powis (1871) — Scope of authority of attorney",
        ],
    },
    "Rent Agreement": {
        "acts": [
            "Transfer of Property Act, 1882",
            "Indian Registration Act, 1908",
            "Indian Stamp Act, 1899",
            "State Rent Control Acts",
        ],
        "key_sections": [
            "Section 105 TPA — Lease defined",
            "Section 106 TPA — Duration and termination of leases",
            "Section 107 TPA — Lease how made",
            "Section 108 TPA — Rights and liabilities of lessor and lessee",
            "Section 17(1)(d) Registration Act — Lease registration if more than one year",
        ],
        "search_queries": [
            "rent agreement registration mandatory Section 17 Registration Act",
            "lease agreement more than 11 months stamp duty",
            "unregistered lease agreement eviction",
        ],
        "key_precedents": [
            "Anthony v. K.C. Ittoop (2000) — Unregistered lease inadmissible as evidence",
            "S.S. Grewal v. Desh Raj (1968) — Leave and licence vs lease",
        ],
    },
    "Agreement to Sell": {
        "acts": [
            "Transfer of Property Act, 1882",
            "Indian Contract Act, 1872",
            "Specific Relief Act, 1963",
            "Indian Registration Act, 1908",
        ],
        "key_sections": [
            "Section 54 TPA — Agreement to sell vs sale",
            "Section 55 Indian Contract Act — Time as essence of contract",
            "Section 10, 16 Specific Relief Act — Specific performance",
            "Section 73, 74 Indian Contract Act — Damages for breach",
        ],
        "search_queries": [
            "agreement to sell does not create interest in property Section 54 TPA",
            "specific performance agreement to sell limitation",
            "forfeiture earnest money agreement to sell",
        ],
        "key_precedents": [
            "Suraj Lamp & Industries v. State of Haryana (2012)",
            "Chand Rani v. Kamal Rani (1993) — Time as essence in agreement to sell",
        ],
    },
    "Notice (CrPC/CPC)": {
        "acts": [
            "Code of Civil Procedure, 1908",
            "Code of Criminal Procedure, 1973",
            "Limitation Act, 1963",
        ],
        "key_sections": [
            "Section 80 CPC — Notice to government before suit",
            "Section 138 Negotiable Instruments Act — Cheque bounce notice",
            "Section 91 CrPC — Order to produce document",
            "Section 106 Transfer of Property Act — Notice for lease termination",
        ],
        "search_queries": [
            "legal notice mandatory before filing suit Section 80 CPC",
            "cheque bounce notice Section 138 NI Act requirements",
            "effect of non-compliance with legal notice",
        ],
        "key_precedents": [
            "State of Punjab v. Geeta Iron (1978) — Strict compliance with Section 80 CPC",
            "Dashrath Rupsingh Rathod v. State of Maharashtra (2014) — Section 138 NI Act jurisdiction",
        ],
    },
    "Court Order / Judgment": {
        "acts": [
            "Code of Civil Procedure, 1908",
            "Code of Criminal Procedure, 1973",
            "Constitution of India",
        ],
        "key_sections": [
            "Section 33, Order XX CPC — Judgment and decree",
            "Section 354 CrPC — Language of judgment",
            "Article 141 Constitution — Law declared by Supreme Court binding",
            "Article 226 Constitution — Power of High Courts to issue writs",
        ],
        "search_queries": [
            "judgment without reasons void CrPC Section 354",
            "decree without judgment validity",
            "court order requirements under CPC",
        ],
        "key_precedents": [],
    },
    "Will / Testament": {
        "acts": [
            "Indian Succession Act, 1925",
            "Indian Registration Act, 1908",
        ],
        "key_sections": [
            "Section 59 ISA — Person capable of making wills",
            "Section 63 ISA — Execution of unprivileged wills",
            "Section 68 Indian Evidence Act — Proof of will attested",
            "Section 18 Registration Act — Optional registration of wills",
        ],
        "search_queries": [
            "will without two witnesses validity Indian Succession Act Section 63",
            "unregistered will validity Indian law",
            "suspicious circumstances will attestation",
        ],
        "key_precedents": [
            "H. Venkatachala Iyengar v. B.N. Thimmajamma (1959) — Suspicious circumstances in wills",
            "Jaswant Kaur v. Amrit Kaur (1977) — Attestation requirements for wills",
        ],
    },
}


class IndianLawRAGTool:
    """
    RAG-based tool for retrieving Indian law context relevant to document validation.

    Combines:
    1. Static law mapping (acts, sections, precedents per document type)
    2. Indian Kanoon API for live statute and case law retrieval
    3. Domain-specific FAISS RAG (criminal / civil / constitutional)

    This tool feeds into Layer 3 (LLM-based defect explanation).
    """

    def __init__(
        self,
        indian_kanoon_tool=None,
        crime_rag=None,           # Legacy; kept for backward compatibility
        criminal_rag=None,        # CriminalRAGSystem
        civil_rag=None,           # CivilRAGSystem
        constitutional_rag=None,  # ConstitutionalRAGSystem
    ):
        self.indian_kanoon = indian_kanoon_tool
        # Prefer new domain-specific instances; fall back to legacy crime_rag
        self.criminal_rag = criminal_rag or crime_rag
        self.civil_rag = civil_rag
        self.constitutional_rag = constitutional_rag
        self.crime_rag = self.criminal_rag  # backward-compat alias
        self.law_map = DOCUMENT_LAW_MAP

    async def retrieve_context(
        self,
        document_type: str,
        missing_elements: List[Dict[str, Any]],
        non_compliance: List[Dict[str, Any]],
        document_text: str = "",
        jurisdiction_hints: List[str] = None,
    ) -> IndianLawContext:
        """
        Retrieve Indian law context relevant to the document validation findings.

        Args:
            document_type: Classified document type
            missing_elements: Missing mandatory elements from Layer 2
            non_compliance: Non-compliance issues from Layer 2
            document_text: Original document text for additional context
            jurisdiction_hints: State/court hints from classifier

        Returns:
            IndianLawContext with references, acts, sections, and notes
        """
        context = IndianLawContext(document_type=document_type)

        # 1. Get static law mapping
        law_info = self.law_map.get(document_type, {})
        context.applicable_acts = law_info.get("acts", [])
        context.applicable_sections = law_info.get("key_sections", [])
        context.precedent_notes = law_info.get("key_precedents", [])

        # 2. Generate state-specific notes if jurisdiction detected
        if jurisdiction_hints:
            context.state_specific_notes = self._get_state_notes(
                document_type, jurisdiction_hints
            )

        # 3. Search Indian Kanoon for missing elements (async, non-blocking)
        if self.indian_kanoon and missing_elements:
            api_refs = await self._search_missing_elements_context(
                document_type, missing_elements
            )
            context.references.extend(api_refs)

        # 4. Search FAISS RAG for additional legal context
        if self.crime_rag and document_text:
            rag_refs = await self._search_rag_context(document_type, document_text)
            context.references.extend(rag_refs)

        # 5. Add static references for missing elements
        static_refs = self._build_static_references(
            document_type, missing_elements, non_compliance
        )
        context.references.extend(static_refs)

        return context

    def _get_state_notes(
        self, document_type: str, jurisdiction_hints: List[str]
    ) -> List[str]:
        """Generate state-specific notes based on jurisdiction."""
        notes = []

        # Extract state from hints
        states_found = []
        for hint in jurisdiction_hints:
            if hint.startswith("State: "):
                states_found.append(hint.replace("State: ", "").lower())

        # State-specific stamp duty and registration notes
        stamp_duty_notes: Dict[str, Dict[str, str]] = {
            "delhi": {
                "Sale Deed": "Delhi: Stamp duty is 6% for male owners, 4% for female owners. Registration fee is 1%.",
                "Rent Agreement": "Delhi: Rent agreements must be registered if lease period exceeds 11 months. E-stamping mandatory.",
                "Affidavit": "Delhi: Affidavit on Rs. 10/- stamp paper. E-stamp accepted.",
                "Power of Attorney": "Delhi: GPA stamp duty is Rs. 100/- for general, varies for property-related POA.",
            },
            "maharashtra": {
                "Sale Deed": "Maharashtra: Stamp duty is 5% in metro areas (Mumbai, Pune). Additional 1% metro cess. LBT applicable.",
                "Rent Agreement": "Maharashtra: Leave and licence agreement registration mandatory. Stamp duty 0.25% of annual rent.",
                "Affidavit": "Maharashtra: Affidavit on Rs. 100/- stamp paper as per Maharashtra Stamp Act.",
                "Power of Attorney": "Maharashtra: POA stamp duty as per Article 48 of Maharashtra Stamp Act.",
            },
            "karnataka": {
                "Sale Deed": "Karnataka: Stamp duty is 5% + 1% surcharge. Registration fee is 1%.",
                "Rent Agreement": "Karnataka: Rent agreement on stamp paper of 1% of annual rent. Registration mandatory for >11 months.",
                "Affidavit": "Karnataka: Affidavit on Rs. 20/- stamp paper.",
            },
            "tamil nadu": {
                "Sale Deed": "Tamil Nadu: Stamp duty is 7%. Registration fee is 4%. Total cost is 11%.",
                "Rent Agreement": "Tamil Nadu: Stamp duty is 1% of annual rent or advance, whichever is higher.",
            },
            "uttar pradesh": {
                "Sale Deed": "Uttar Pradesh: Stamp duty is 7% for male, 6% for female. Registration fee 1%.",
                "Rent Agreement": "UP: Rent agreement on Rs. 100/- stamp paper. Registration optional for ≤11 months.",
                "Affidavit": "UP: Affidavit on Rs. 10/- stamp paper.",
            },
            "west bengal": {
                "Sale Deed": "West Bengal: Stamp duty ranges from 5-8% depending on area. Registration fee 1%.",
            },
            "rajasthan": {
                "Sale Deed": "Rajasthan: Stamp duty is 5% for male, 4% for female. DLC rate applicable.",
            },
            "gujarat": {
                "Sale Deed": "Gujarat: Stamp duty is 4.9%. Registration fee is 1%.",
                "Rent Agreement": "Gujarat: Rent agreement stamp duty is 1% of annual rent.",
            },
        }

        for state in states_found:
            state_notes = stamp_duty_notes.get(state, {})
            note = state_notes.get(document_type)
            if note:
                notes.append(note)

        return notes

    async def _search_missing_elements_context(
        self,
        document_type: str,
        missing_elements: List[Dict[str, Any]],
    ) -> List[LawReference]:
        """Search Indian Kanoon for context on missing mandatory elements."""
        references = []

        if not self.indian_kanoon:
            return references

        # Build targeted search queries from missing elements
        search_tasks = []
        for element in missing_elements[:3]:  # Limit to 3 to avoid API overload
            statute_ref = element.get("statute_reference", "")
            element_name = element.get("element", "")
            query = (
                f"{element_name} {statute_ref} {document_type} mandatory requirement"
            )
            search_tasks.append(self._search_single_element(query, element_name))

        # Execute searches in parallel
        if search_tasks:
            results = await asyncio.gather(*search_tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, list):
                    references.extend(result)

        return references

    async def _search_single_element(
        self, query: str, element_name: str
    ) -> List[LawReference]:
        """Search Indian Kanoon for a single element."""
        refs = []
        try:
            result = await self.indian_kanoon.answer_legal_query(query, "statute")
            for doc in result.get("results", [])[:2]:
                refs.append(
                    LawReference(
                        title=doc.title,
                        act_name=self._extract_act_name(doc.title),
                        section=self._extract_section(doc.title),
                        relevance=f"Relevant to missing element: {element_name}",
                        excerpt=doc.excerpt[:300] if doc.excerpt else "",
                        url=doc.url,
                        source_type="indian_kanoon",
                    )
                )
        except Exception as e:
            print(f"Indian Kanoon search error for '{element_name}': {e}")

        return refs

    async def _search_rag_context(
        self, document_type: str, document_text: str
    ) -> List[LawReference]:
        """Search local FAISS RAG for additional legal context.

        Routes to the appropriate domain RAG based on document type:
        - FIR / criminal documents  -> criminal_rag
        - Civil / contract docs     -> civil_rag (if available)
        - Default                   -> criminal_rag (legacy behaviour)
        """
        refs = []

        criminal_doc_types = {"FIR"}
        civil_doc_types = {
            "Sale Deed", "Rent Agreement", "Agreement to Sell",
            "Power of Attorney", "Will / Testament", "Notice (CrPC/CPC)",
            "Affidavit",
        }

        rag_instance = None
        if document_type in criminal_doc_types and self.criminal_rag:
            rag_instance = self.criminal_rag
        elif document_type in civil_doc_types and self.civil_rag:
            rag_instance = self.civil_rag
        elif self.criminal_rag:
            rag_instance = self.criminal_rag

        if not rag_instance or not getattr(rag_instance, "initialized", False):
            return refs

        try:
            query = f"{document_type} legal requirements validity Indian law"

            if hasattr(rag_instance, "retrieve_context"):
                # Criminal RAG returns CrimeContext with relevant_passages
                context = await rag_instance.retrieve_context(query, k=3)
                for passage in getattr(context, "relevant_passages", [])[:2]:
                    refs.append(
                        LawReference(
                            title=f"RAG: {document_type} legal context",
                            act_name=self._extract_act_name(passage),
                            section=self._extract_section(passage),
                            relevance=f"Local knowledge base reference for {document_type}",
                            excerpt=passage[:300],
                            url="",
                            source_type="local_rag",
                        )
                    )
            elif hasattr(rag_instance, "retrieve"):
                # Civil / Constitutional RAG returns LegalContext with chunks
                context = await rag_instance.retrieve(query, k=3)
                for chunk in context.chunks[:2]:
                    refs.append(
                        LawReference(
                            title=f"{chunk.act_name} § {chunk.section_number}",
                            act_name=chunk.act_name,
                            section=chunk.section_number,
                            relevance=f"Relevant to {document_type}: {chunk.title}",
                            excerpt=chunk.text[:300],
                            url="",
                            source_type="local_rag",
                        )
                    )
        except Exception as e:
            print(f"RAG search error for {document_type}: {e}")

        return refs

    def _build_static_references(
        self,
        document_type: str,
        missing_elements: List[Dict[str, Any]],
        non_compliance: List[Dict[str, Any]],
    ) -> List[LawReference]:
        """Build static references from the statute references in checklist items."""
        refs = []

        for item in missing_elements:
            statute = item.get("statute_reference", "")
            if statute:
                refs.append(
                    LawReference(
                        title=f"Missing: {item['element']}",
                        act_name=self._extract_act_name(statute),
                        section=self._extract_section(statute),
                        relevance=item.get("description", ""),
                        excerpt=f"Required by: {statute}",
                        url="",
                        source_type="statutory_checklist",
                    )
                )

        for item in non_compliance:
            statute = item.get("statute_reference", "")
            if statute:
                refs.append(
                    LawReference(
                        title=f"Non-compliance: {item['element']}",
                        act_name=self._extract_act_name(statute),
                        section=self._extract_section(statute),
                        relevance=item.get("description", ""),
                        excerpt=f"Recommended by: {statute}",
                        url="",
                        source_type="statutory_checklist",
                    )
                )

        return refs

    def _extract_act_name(self, text: str) -> str:
        """Extract act name from a text string."""
        act_patterns = [
            r"((?:Indian\s+)?(?:Registration|Stamp|Contract|Evidence|Succession|Partnership)\s+Act(?:\s*,?\s*\d{4})?)",
            r"(Transfer\s+of\s+Property\s+Act(?:\s*,?\s*\d{4})?)",
            r"(Code\s+of\s+(?:Civil|Criminal)\s+Procedure(?:\s*,?\s*\d{4})?)",
            r"(Power\s+of\s+Attorney\s+Act(?:\s*,?\s*\d{4})?)",
            r"(Oaths\s+Act(?:\s*,?\s*\d{4})?)",
            r"(Notaries\s+Act(?:\s*,?\s*\d{4})?)",
            r"(Specific\s+Relief\s+Act(?:\s*,?\s*\d{4})?)",
            r"(Negotiable\s+Instruments\s+Act(?:\s*,?\s*\d{4})?)",
            r"(Limitation\s+Act(?:\s*,?\s*\d{4})?)",
            r"(Bharatiya\s+(?:Nyaya|Nagarik)\s+Sanhita(?:\s*,?\s*\d{4})?)",
            r"((?:IPC|CrPC|CPC|BNS|BNSS|BSA|TPA)\b)",
        ]
        for pattern in act_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)
        return ""

    def _extract_section(self, text: str) -> str:
        """Extract section number from a text string."""
        patterns = [
            r"(?:Section|Sec\.?|S\.)\s*(\d+[A-Z]?(?:\(\d+\))?)",
            r"(?:Order\s+[IVXLC]+(?:\s+Rule\s+\d+)?)",
            r"(?:Article\s+\d+[A-Z]?)",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(0)
        return ""


# Singleton
_indian_law_rag: Optional[IndianLawRAGTool] = None


def get_indian_law_rag(
    indian_kanoon_tool=None,
    crime_rag=None,
    criminal_rag=None,
    civil_rag=None,
    constitutional_rag=None,
) -> IndianLawRAGTool:
    """Get or create the Indian law RAG tool singleton.

    Accepts both the legacy crime_rag parameter and the new domain-specific
    criminal_rag / civil_rag / constitutional_rag parameters.
    """
    global _indian_law_rag
    if _indian_law_rag is None:
        _indian_law_rag = IndianLawRAGTool(
            indian_kanoon_tool=indian_kanoon_tool,
            crime_rag=crime_rag,
            criminal_rag=criminal_rag,
            civil_rag=civil_rag,
            constitutional_rag=constitutional_rag,
        )
    return _indian_law_rag
