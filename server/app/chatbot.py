"""
LangGraph-based legal chatbot implementation.
This module defines the chatbot workflow using LangGraph for state management and routing.
"""

import asyncio
import contextvars
import json
import re
from asyncio.events import AbstractEventLoop
from functools import lru_cache
from typing import Any, AsyncGenerator, Dict, List, Literal, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from app.config import get_settings
from app.prompts import (
    CONVERSATION_PROMPT,
    CRIME_REPORT_PROMPT,
    DOCUMENT_ANALYSIS_PROMPT,
    DOCUMENT_VALIDATION_INTENT_KEYWORDS,
    DOCUMENT_VALIDATION_UPLOAD_PROMPT,
    GENERAL_QUERY_PROMPT,
    INDIAN_LAW_SEARCH_PROMPT,
    INTENT_CLASSIFICATION_PROMPT,
    LAWYER_SEARCH_PROMPT,
    RELEVANCE_CHECK_PROMPT,
)
from app.state import (
    ChatState,
    CrimeReportInfo,
    DocumentInfo,
    DocumentValidationInfo,
    LawyerInfo,
    Message,
)
from app.tools.crime_reporter import detect_crime_type
from app.tools.document_classifier import get_document_classifier
from app.tools.document_extractor import get_document_extractor
from app.tools.indian_kanoon import get_indian_kanoon_tool
from app.tools.indian_law_rag import get_indian_law_rag
from app.tools.lawyer_finder import get_lawyer_finder
from app.tools.legal_defect_analyzer import get_legal_defect_analyzer
from app.tools.statutory_validator import get_statutory_validator

# ============================================================================
# Pydantic Models for Structured Routing
# ============================================================================


class RoutingDecision(BaseModel):
    """Structured output for routing logic."""

    primary_intent: Literal[
        "document_analysis",
        "crime_report",
        "find_lawyer",
        "general_query",
        "non_legal",
    ] = Field(description="The main intent classification")

    confidence: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Confidence score between 0 and 1"
    )

    reasoning: str = Field(
        default="", description="Brief explanation of why this route was chosen"
    )

    secondary_intents: List[str] = Field(
        default_factory=list, description="Additional intents for multi-intent queries"
    )

    extracted_entities: List[str] = Field(
        default_factory=list, description="Extracted legal terms, acts, or sections"
    )

    requires_tools: List[str] = Field(
        default_factory=list,
        description="Tools needed: indian_kanoon, crime_rag, lawyer_finder, document_analyzer",
    )


class DomainClassification(BaseModel):
    """Stage 1: Domain-level classification (Legal vs Non-Legal)."""

    is_legal: bool = Field(description="Whether the query is related to legal matters")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    legal_indicators: List[str] = Field(
        default_factory=list, description="Legal terms or concepts found in the query"
    )


class ToolSelection(BaseModel):
    """Determines which tools should be used for a given query."""

    use_indian_kanoon: bool = Field(
        default=False, description="Use for case law, precedents, judgments"
    )
    use_crime_rag: bool = Field(
        default=False, description="Use for IPC/CrPC/BNS sections, punishments, criminal procedures"
    )
    use_civil_rag: bool = Field(
        default=False,
        description="Use for civil/contract/property/tort law sections (Contract Act, TPA, CPC)",
    )
    use_constitutional_rag: bool = Field(
        default=False,
        description="Use for constitutional Articles, fundamental rights, directive principles",
    )
    use_lawyer_finder: bool = Field(
        default=False, description="Use for finding lawyers"
    )
    use_document_analyzer: bool = Field(
        default=False, description="Use for document analysis/validation"
    )
    use_llm_only: bool = Field(
        default=True, description="Use LLM directly without tools"
    )
    reasoning: str = Field(default="", description="Why these tools were selected")


# ============================================================================
# Keyword Banks for Fast Routing (Zero-Latency Layer)
# ============================================================================

# Non-legal patterns (casual conversation)
NON_LEGAL_PATTERNS = frozenset(
    [
        "my favorite",
        "my favourite",
        "i like",
        "i love",
        "favorite color",
        "favourite color",
        "best food",
        "favorite movie",
        "what is your",
        "how are you",
        "good morning",
        "good night",
        "hello",
        "hi there",
        "weather today",
        "tell me a joke",
        "sing a song",
        "who are you",
        "your name",
        "thank you",
        "thanks",
        "bye",
        "goodbye",
    ]
)

# Legal domain indicators
LEGAL_DOMAIN_KEYWORDS = frozenset(
    [
        "law",
        "legal",
        "court",
        "judge",
        "crime",
        "police",
        "lawyer",
        "attorney",
        "ipc",
        "crpc",
        "section",
        "act",
        "right",
        "constitution",
        "case",
        "fir",
        "bail",
        "arrest",
        "prosecution",
        "verdict",
        "judgment",
        "statute",
        "offence",
        "offense",
        "punishment",
        "penalty",
        "fine",
        "imprisonment",
        "contract",
        "agreement",
        "deed",
        "property",
        "tenant",
        "landlord",
        "divorce",
        "custody",
        "maintenance",
        "alimony",
        "will",
        "testament",
        "inheritance",
        "defamation",
        "fraud",
        "cheating",
        "theft",
        "robbery",
    ]
)

# Document validation keywords
VALIDATION_KEYWORDS = frozenset(
    [
        "validate",
        "validity",
        "check validity",
        "verify",
        "statutory compliance",
        "defects",
        "legal defects",
        "is this valid",
        "check compliance",
        "missing elements",
        "properly drafted",
        "drafting defects",
        "formal defects",
        "mandatory requirements",
        "stamp duty compliance",
        "review this document",
        "check this document",
        "is this correct",
        "is this proper",
    ]
)

# Personal crime report indicators
PERSONAL_CRIME_INDICATORS = frozenset(
    [
        "i was attacked",
        "i was scammed",
        "i was robbed",
        "i was cheated",
        "someone stole",
        "someone attacked",
        "someone threatened",
        "i am victim",
        "i need help",
        "help me report",
        "file fir",
        "report crime",
        "happened to me",
        "i have been",
        "they took my",
        "my money was",
    ]
)

# Legal analysis indicators (theoretical questions)
LEGAL_ANALYSIS_KEYWORDS = frozenset(
    [
        "which section",
        "what section",
        "which ipc",
        "what ipc",
        "sections apply",
        "offences apply",
        "laws apply",
        "act apply",
        "procedural steps",
        "procedure under",
        "cognizable",
        "non-cognizable",
        "bailable",
        "non-bailable",
        "sanction required",
        "sanction for prosecution",
        "legal implications",
        "legal consequences",
        "jurisdiction",
        "which court",
        "competent court",
        "extradition",
        "can both",
        "both apply",
        "what happens if",
        "explain the",
    ]
)

# Case law / precedent search keywords (→ Indian Kanoon)
CASE_SEARCH_KEYWORDS = frozenset(
    [
        "case",
        "judgment",
        "judgement",
        "ruling",
        "verdict",
        "precedent",
        "citation",
        "court held",
        "supreme court",
        "high court",
        "landmark",
        "case law",
        "decided by",
        "vs",
        "v/s",
        "versus",
        "petitioner",
        "respondent",
        "appellant",
        "similar cases",
        "previous cases",
    ]
)

# IPC/CrPC/Statute keywords (→ Crime RAG)
STATUTE_KEYWORDS = frozenset(
    [
        "ipc",
        "indian penal code",
        "crpc",
        "criminal procedure",
        "punishment",
        "imprisonment",
        "fine",
        "penalty",
        "forgery",
        "trespass",
        "assault",
        "threat",
        "intimidation",
        "fraud",
        "cheating",
        "theft",
        "robbery",
        "bribery",
        "cyber",
        "hacking",
        "identity theft",
        "money laundering",
        "defamation",
        "kidnapping",
        "murder",
        "hurt",
        "grievous",
        "it act",
        "information technology",
        "prevention of corruption",
        "poca",
        "pmla",
    ]
)

# Crime type keywords for multi-offense detection
CRIME_TYPE_KEYWORDS = frozenset(
    [
        "forgery",
        "forged",
        "trespass",
        "trespassed",
        "assault",
        "assaulted",
        "threat",
        "threatened",
        "bribe",
        "bribery",
        "fraud",
        "cyber",
        "identity theft",
        "launder",
        "laundering",
        "cheating",
        "theft",
        "robbery",
        "murder",
        "kidnapping",
        "extortion",
        "blackmail",
        "defamation",
        "harassment",
        "stalking",
        "dowry",
        "domestic violence",
    ]
)

# Lawyer search keywords
LAWYER_SEARCH_KEYWORDS = frozenset(
    [
        "lawyer",
        "attorney",
        "advocate",
        "legal help",
        "representation",
        "law firm",
        "find lawyer",
        "need lawyer",
        "consult lawyer",
        "hire lawyer",
        "legal counsel",
    ]
)

# ============================================================================
# Enhanced Tool Selection Keywords
# ============================================================================

# Keywords that STRONGLY indicate Indian Kanoon is needed
INDIAN_KANOON_STRONG_INDICATORS = frozenset(
    [
        "case law",
        "case laws",
        "precedent",
        "precedents",
        "landmark case",
        "supreme court held",
        "high court held",
        "court ruling",
        "court rulings",
        "judgment in",
        "judgement in",
        "cited in",
        "ratio decidendi",
        "obiter dicta",
        "similar case",
        "similar cases",
        "relevant cases",
        "leading case",
        "authority",
        "binding precedent",
        "persuasive precedent",
        "case citation",
        "air ",
        "scr ",
        "scc ",
        "all ",  # Case report abbreviations
        # Substantive law queries that need authoritative sources
        "constitutionality",
        "constitutional validity",
        "unconstitutional",
        "basic structure doctrine",
        "fundamental right",
        "article 19",
        "article 21",
        "article 14",
        "article 32",
        "article 226",
        "article 356",
        "puttaswamy",
        "kesavananda",
        "right to privacy",
        "marital rape",
        "section 65b",
        "admissibility",
        "electronic evidence",
        "anticipatory bail",
        "quash",
        "quashing",
        "section 482",
        "cryptocurrency",
        "rbi ban",
    ]
)

# Keywords that MODERATELY indicate Indian Kanoon might be useful
INDIAN_KANOON_MODERATE_INDICATORS = frozenset(
    [
        "case",
        "judgment",
        "judgement",
        "ruling",
        "verdict",
        "decided",
        "vs",
        "v/s",
        "versus",
        "petitioner",
        "respondent",
        "appellant",
        "supreme court",
        "high court",
        "district court",
        "sessions court",
        "tribunal",
        "bench",
        "division bench",
        "constitution bench",
        # Substantive law areas that benefit from case law
        "constitution",
        "constitutional",
        "parliament",
        "legislature",
        "fundamental rights",
        "article",
        "contract act",
        "transfer of property",
        "succession act",
        "hindu succession",
        "special marriage act",
        "hindu marriage act",
        "evidence act",
        "consumer protection",
        "arbitration",
        "negotiable instruments",
        "partnership act",
        "it act",
        "information technology act",
        "rera",
        "fema",
        "pmla",
        "sebi",
        "rbi",
        "dv act",
        "domestic violence act",
        "prevention of corruption",
        "bail",
        "fir",
        "enforceable",
        "valid",
        "void",
        "voidable",
        "legal position",
        "legal status",
        "legally",
    ]
)

# Keywords that STRONGLY indicate Crime RAG is needed
CRIME_RAG_STRONG_INDICATORS = frozenset(
    [
        "which section",
        "what section",
        "applicable section",
        "sections apply",
        "ipc section",
        "crpc section",
        "under which",
        "punishable under",
        "punishment for",
        "penalty for",
        "imprisonment for",
        "fine for",
        "cognizable",
        "non-cognizable",
        "bailable",
        "non-bailable",
        "compoundable",
        "non-compoundable",
        "triable by",
        "investigation",
        "chargesheet",
        "fir for",
        "file fir",
        "police complaint",
    ]
)

# Keywords that MODERATELY indicate Crime RAG might be useful
CRIME_RAG_MODERATE_INDICATORS = frozenset(
    [
        "ipc",
        "crpc",
        "indian penal code",
        "criminal procedure",
        "offence",
        "offense",
        "crime",
        "criminal",
        "penal",
        "forgery",
        "theft",
        "robbery",
        "assault",
        "murder",
        "kidnapping",
        "cheating",
        "fraud",
        "defamation",
        "trespass",
        "hurt",
        "grievous",
        "extortion",
        "bribery",
        "corruption",
        "cyber crime",
        "hacking",
    ]
)

# Keywords indicating NO tools needed (LLM sufficient)
# NOTE: These are VERY narrow — only truly non-legal-domain definitional queries
# qualify. Any query touching specific acts, sections, rights, or legal concepts
# MUST still use tools for grounding.
LLM_ONLY_INDICATORS = frozenset(
    [
        "general advice",
        "guidance",
        "help me understand",
    ]
)

# Strong LLM-only patterns — extremely narrow, only meta/definitional queries
# that are about learning the structure of law, not about specific legal questions
LLM_ONLY_STRONG_PATTERNS: frozenset[str] = frozenset(
    [
        "what is the meaning",
        "what does the term",
        "explain the concept of jurisprudence",
    ]
)

# ============================================================================
# LEGAL SUBSTANCE KEYWORDS — Force tool use for any substantive legal question
# These override LLM_ONLY when present, ensuring RAG grounding
# ============================================================================

# Constitutional & fundamental rights keywords (→ Indian Kanoon)
CONSTITUTIONAL_KEYWORDS = frozenset(
    [
        "article",
        "fundamental right",
        "fundamental rights",
        "constitution",
        "constitutional",
        "amendment",
        "basic structure",
        "parliament",
        "legislature",
        "writ",
        "habeas corpus",
        "mandamus",
        "certiorari",
        "prohibition",
        "quo warranto",
        "right to privacy",
        "right to life",
        "right to equality",
        "free speech",
        "freedom of speech",
        "freedom of expression",
        "public order",
        "reasonable restriction",
        "directive principles",
        "dpsp",
        "preamble",
        "federalism",
        "president",
        "governor",
        "president's rule",
        "article 356",
        "article 19",
        "article 21",
        "article 14",
        "article 32",
        "article 226",
        "article 370",
        "article 370",
        "ninth schedule",
        "seventh schedule",
        "union list",
        "concurrent list",
        "state list",
        "surveillance",
        "proportionality",
        "puttaswamy",
        "kesavananda",
        "minerva mills",
        "maneka gandhi",
        "golaknath",
    ]
)

# Civil / Contract law keywords (→ Indian Kanoon)
CIVIL_LAW_KEYWORDS = frozenset(
    [
        "contract",
        "agreement",
        "enforceable",
        "void",
        "voidable",
        "consideration",
        "breach",
        "specific performance",
        "damages",
        "indemnity",
        "guarantee",
        "coercion",
        "undue influence",
        "misrepresentation",
        "mistake",
        "frustration",
        "force majeure",
        "non-compete",
        "restraint of trade",
        "liquidated damages",
        "injunction",
        "arbitration",
        "mediation",
        "consumer protection",
        "tort",
        "negligence",
        "defamation",
        "nuisance",
        "indian contract act",
        "section 10",
        "section 23",
        "section 25",
        "section 27",
        "section 56",
        "section 73",
        "section 74",
        "sale of goods",
        "negotiable instruments",
        "partnership",
        "llp",
        "oral agreement",
        "oral contract",
        "stamp duty",
        "registration",
        "admissible",
        "admissibility",
        "evidence",
        "section 65b",
        "electronic evidence",
        "digital evidence",
        "whatsapp",
        "electronic record",
    ]
)

# Property law keywords (→ Indian Kanoon)
PROPERTY_LAW_KEYWORDS = frozenset(
    [
        "property",
        "ancestral property",
        "coparcenary",
        "partition",
        "sale deed",
        "gift deed",
        "will",
        "testament",
        "succession",
        "inheritance",
        "legal heir",
        "legal heirs",
        "hindu succession",
        "transfer of property",
        "easement",
        "mortgage",
        "lease",
        "tenancy",
        "tenant",
        "landlord",
        "rent control",
        "eviction",
        "encumbrance",
        "benami",
        "rera",
        "real estate",
        "mutation",
        "land revenue",
        "stridhan",
        "joint family",
        "huf",
    ]
)

# Family law keywords (→ Indian Kanoon)
FAMILY_LAW_KEYWORDS = frozenset(
    [
        "divorce",
        "custody",
        "maintenance",
        "alimony",
        "domestic violence",
        "dowry",
        "marriage",
        "matrimonial",
        "judicial separation",
        "mutual consent",
        "cruelty",
        "desertion",
        "restitution of conjugal rights",
        "live-in",
        "live in partner",
        "cohabitation",
        "hindu marriage act",
        "special marriage act",
        "muslim personal law",
        "guardianship",
        "adoption",
        "juvenile",
        "child marriage",
        "marital rape",
        "section 498a",
        "protection of women",
        "dv act",
    ]
)

# Technology & modern law keywords (→ Indian Kanoon)
TECH_LAW_KEYWORDS = frozenset(
    [
        "cryptocurrency",
        "crypto",
        "bitcoin",
        "blockchain",
        "artificial intelligence",
        "ai liability",
        "data protection",
        "personal data",
        "gdpr",
        "pdp bill",
        "dpdp",
        "social media",
        "online",
        "internet",
        "deepfake",
        "it act",
        "information technology act",
        "section 66a",
        "section 67",
        "section 43",
        "intermediary",
        "safe harbour",
        "takedown",
        "right to be forgotten",
        "aadhaar",
        "rbi",
        "fema",
        "pmla",
        "sebi",
    ]
)

# Criminal procedure & bail keywords (→ Crime RAG + Indian Kanoon)
CRIMINAL_PROCEDURE_KEYWORDS = frozenset(
    [
        "fir",
        "bail",
        "anticipatory bail",
        "regular bail",
        "quash",
        "quashing",
        "section 482",
        "section 438",
        "section 439",
        "section 154",
        "section 200",
        "section 320",
        "compoundable",
        "non-compoundable",
        "chargesheet",
        "investigation",
        "cognizable",
        "non-cognizable",
        "complainant",
        "withdrawal",
        "compound",
        "plea bargaining",
        "discharge",
        "acquittal",
        "conviction",
        "appeal",
        "revision",
        "review",
        "habeas corpus",
        "remand",
        "police custody",
        "judicial custody",
        "bhajan lal",
    ]
)


@lru_cache()
def get_llm() -> ChatOllama:
    """Get cached LLM instance for better performance."""
    settings = get_settings()
    return ChatOllama(
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        base_url=settings.ollama_base_url,
        num_predict=1024,  # Balanced: enough for detailed answers, faster inference
        timeout=35.0,  # Tighter timeout for snappier responses
    )


@lru_cache()
def get_fast_llm() -> ChatOllama:
    """Get cached LLM for classification tasks (using local Ollama)."""
    settings = get_settings()
    return ChatOllama(
        model=settings.llm_model,
        temperature=0,
        base_url=settings.ollama_base_url,
        num_predict=128,  # Reduced from 256 for faster classification
        timeout=15.0,  # Reduced from 30s
    )


# Context variable for streaming queue - when set, invoke_llm_safely streams tokens
_stream_queue_var: contextvars.ContextVar[asyncio.Queue | None] = (
    contextvars.ContextVar("stream_queue", default=None)
)


async def invoke_llm_safely(llm: ChatOllama, prompt: str) -> str:
    """Safely invoke LLM with proper error handling. Supports streaming via context queue."""
    queue = _stream_queue_var.get(None)

    if queue is not None:
        # Streaming mode - use astream and push chunks to queue
        try:
            full_response = ""
            async for chunk in llm.astream([HumanMessage(content=prompt)]):
                token = chunk.content if hasattr(chunk, "content") else str(chunk)
                if token:
                    full_response += token
                    await queue.put(token)
            return full_response
        except Exception as e:
            print(f"LLM streaming error: {e}")
            raise
    else:
        # Normal (non-streaming) mode
        try:
            loop: AbstractEventLoop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None, lambda: llm.invoke([HumanMessage(content=prompt)])
            )
            return response.content
        except Exception as e:
            print(f"LLM invocation error: {e}")
            raise


# ============================================================================
# Node Functions
# ============================================================================


def _fast_keyword_check(text: str, keywords: frozenset) -> bool:
    """Fast O(n) keyword matching against a frozenset."""
    text_lower = text.lower()
    return any(kw in text_lower for kw in keywords)


def _count_keyword_matches(text: str, keywords: frozenset) -> int:
    """Count how many keywords match in the text."""
    text_lower = text.lower()
    return sum(1 for kw in keywords if kw in text_lower)


def _extract_legal_entities(text: str) -> List[str]:
    """Extract legal terms, acts, and sections from text."""
    entities = []
    text_lower = text.lower()

    # Extract IPC/CrPC sections
    import re

    section_patterns = [
        r"section\s+(\d+[a-z]?)",
        r"ipc\s+(\d+[a-z]?)",
        r"crpc\s+(\d+[a-z]?)",
    ]
    for pattern in section_patterns:
        matches = re.findall(pattern, text_lower)
        for match in matches:
            entities.append(f"Section {match}")

    # Extract act names
    act_keywords = [
        "indian penal code",
        "ipc",
        "crpc",
        "criminal procedure code",
        "it act",
        "information technology act",
        "prevention of corruption act",
        "pmla",
        "aadhaar act",
        "contract act",
        "transfer of property act",
        "evidence act",
        "motor vehicles act",
        "negotiable instruments act",
    ]
    for act in act_keywords:
        if act in text_lower:
            entities.append(act.title())

    return list(set(entities))


def _determine_tools_needed(
    text: str, intent: str, has_document: bool
) -> ToolSelection:
    """
    Enhanced tool selection with weighted scoring system.

    CORE PRINCIPLE: Any substantive legal question MUST use tools for grounding.
    LLM-only is reserved for truly non-legal definitional meta-questions.

    Tool Selection Logic (Weighted Scoring):
    - indian_kanoon: Case law, precedents, judgments, constitutional questions,
      civil/property/family law, tech law — anything needing authoritative sources
      * Strong indicators: +2.0 each
      * Moderate indicators: +1.0 each
      * Substantive law domain matches: +1.5 each
      * Threshold: >= 1.5 to activate

    - crime_rag: IPC/CrPC sections, punishments, procedures, criminal process
      * Strong indicators: +2.0 each
      * Moderate indicators: +1.0 each
      * Criminal procedure keywords: +1.5
      * Multi-offense detection: +1.5
      * Threshold: >= 1.5 to activate

    - lawyer_finder: Finding lawyers (intent-based)
    - document_analyzer: Document analysis/validation (intent/document-based)
    - llm_only: ONLY when no legal substance keywords match at all
    """
    text_lower = text.lower()

    # =========================================================================
    # Weighted Scoring for Indian Kanoon
    # =========================================================================
    indian_kanoon_score = 0.0
    ik_reasons = []

    # Strong indicators (weight: 2.0 each, max 3)
    strong_ik_matches = sum(
        1 for kw in INDIAN_KANOON_STRONG_INDICATORS if kw in text_lower
    )
    if strong_ik_matches > 0:
        indian_kanoon_score += min(strong_ik_matches * 2.0, 6.0)
        ik_reasons.append(f"{strong_ik_matches} strong case law indicators")

    # Moderate indicators (weight: 1.0 each, max 5)
    moderate_ik_matches = sum(
        1 for kw in INDIAN_KANOON_MODERATE_INDICATORS if kw in text_lower
    )
    if moderate_ik_matches > 0:
        indian_kanoon_score += min(moderate_ik_matches * 1.0, 5.0)
        ik_reasons.append(f"{moderate_ik_matches} moderate case law indicators")

    # Substantive law domain keywords (weight: 1.5 each, max 4.5)
    # These are CRITICAL — they catch constitutional, civil, property, family,
    # and tech law queries that the old system missed entirely
    constitutional_matches = sum(
        1 for kw in CONSTITUTIONAL_KEYWORDS if kw in text_lower
    )
    civil_matches = sum(1 for kw in CIVIL_LAW_KEYWORDS if kw in text_lower)
    property_matches = sum(1 for kw in PROPERTY_LAW_KEYWORDS if kw in text_lower)
    family_matches = sum(1 for kw in FAMILY_LAW_KEYWORDS if kw in text_lower)
    tech_matches = sum(1 for kw in TECH_LAW_KEYWORDS if kw in text_lower)

    substantive_matches = (
        constitutional_matches
        + civil_matches
        + property_matches
        + family_matches
        + tech_matches
    )
    if substantive_matches > 0:
        indian_kanoon_score += min(substantive_matches * 1.5, 4.5)
        domain_parts = []
        if constitutional_matches:
            domain_parts.append(f"constitutional({constitutional_matches})")
        if civil_matches:
            domain_parts.append(f"civil({civil_matches})")
        if property_matches:
            domain_parts.append(f"property({property_matches})")
        if family_matches:
            domain_parts.append(f"family({family_matches})")
        if tech_matches:
            domain_parts.append(f"tech({tech_matches})")
        ik_reasons.append(f"substantive law: {', '.join(domain_parts)}")

    # =========================================================================
    # Weighted Scoring for Crime RAG
    # =========================================================================
    crime_rag_score = 0.0
    rag_reasons = []

    # Strong indicators (weight: 2.0 each, max 3)
    strong_rag_matches = sum(
        1 for kw in CRIME_RAG_STRONG_INDICATORS if kw in text_lower
    )
    if strong_rag_matches > 0:
        crime_rag_score += min(strong_rag_matches * 2.0, 6.0)
        rag_reasons.append(f"{strong_rag_matches} strong statute indicators")

    # Moderate indicators (weight: 1.0 each, max 3)
    moderate_rag_matches = sum(
        1 for kw in CRIME_RAG_MODERATE_INDICATORS if kw in text_lower
    )
    if moderate_rag_matches > 0:
        crime_rag_score += min(moderate_rag_matches * 1.0, 3.0)
        rag_reasons.append(f"{moderate_rag_matches} moderate statute indicators")

    # Criminal procedure keywords (weight: 1.5 each, max 3)
    crim_proc_matches = sum(1 for kw in CRIMINAL_PROCEDURE_KEYWORDS if kw in text_lower)
    if crim_proc_matches > 0:
        crime_rag_score += min(crim_proc_matches * 1.5, 4.5)
        rag_reasons.append(f"{crim_proc_matches} criminal procedure indicators")

    # Multi-offense detection bonus
    crime_count = _count_keyword_matches(text, CRIME_TYPE_KEYWORDS)
    if crime_count >= 2:
        crime_rag_score += 1.5 + (crime_count - 2) * 0.5  # Bonus for complexity
        rag_reasons.append(f"multi-offense scenario ({crime_count} crimes)")

    # Intent-based boost
    if intent == "crime_report":
        crime_rag_score += 2.0
        rag_reasons.append("crime report intent")

    # =========================================================================
    # LLM-Only Check — STRICT: only when NO legal substance is detected
    # =========================================================================
    llm_only_matches = sum(1 for kw in LLM_ONLY_INDICATORS if kw in text_lower)
    strong_llm_only = any(pattern in text_lower for pattern in LLM_ONLY_STRONG_PATTERNS)

    # Check if query has ANY legal substance that needs grounding
    has_legal_substance = (
        strong_ik_matches > 0
        or moderate_ik_matches > 0
        or strong_rag_matches > 0
        or moderate_rag_matches > 0
        or substantive_matches > 0
        or crim_proc_matches > 0
        or crime_count > 0
    )

    # LLM-only ONLY when: no legal substance AND explicitly a meta-question
    is_simple_query = not has_legal_substance and (
        strong_llm_only or llm_only_matches >= 2
    )

    # =========================================================================
    # Determine Final Tool Selection (Threshold: 1.5)
    # =========================================================================
    ACTIVATION_THRESHOLD = 1.5

    use_indian_kanoon = indian_kanoon_score >= ACTIVATION_THRESHOLD
    use_crime_rag = crime_rag_score >= ACTIVATION_THRESHOLD
    use_lawyer_finder = intent == "find_lawyer"
    use_document_analyzer = has_document or intent == "document_analysis"

    # If has legal substance but neither tool scored enough, force Indian Kanoon
    # as the default grounding tool for substantive legal questions
    if has_legal_substance and not use_indian_kanoon and not use_crime_rag:
        use_indian_kanoon = True
        ik_reasons.append("forced: substantive legal query needs grounding")

    # LLM-only if no tools needed OR if it's a simple general query
    use_llm_only = is_simple_query or not (
        use_indian_kanoon or use_crime_rag or use_lawyer_finder or use_document_analyzer
    )

    # If LLM-only but intent requires tools, override
    if use_llm_only and intent in (
        "crime_report",
        "document_analysis",
        "find_lawyer",
    ):
        use_llm_only = False
        if intent == "crime_report":
            use_crime_rag = True
        elif intent == "document_analysis":
            use_document_analyzer = True
        elif intent == "find_lawyer":
            use_lawyer_finder = True

    # =========================================================================
    # Build Detailed Reasoning
    # =========================================================================
    reasons = []

    if use_indian_kanoon:
        reasons.append(
            f"indian_kanoon (score: {indian_kanoon_score:.1f}) - {'; '.join(ik_reasons)}"
        )

    if use_crime_rag:
        reasons.append(
            f"crime_rag (score: {crime_rag_score:.1f}) - {'; '.join(rag_reasons)}"
        )

    if use_lawyer_finder:
        reasons.append("lawyer_finder - lawyer search requested")

    if use_document_analyzer:
        reasons.append("document_analyzer - document processing required")

    if use_llm_only:
        reasons.append(
            f"llm_only - general query (simple indicators: {llm_only_matches})"
        )

    # =========================================================================
    # Domain Isolation Guard — CRITICAL for legal safety
    # =========================================================================
    # If a query is primarily about civil or constitutional law, criminal RAG
    # must be disabled to prevent IPC punishment clauses from contaminating
    # the context (the root cause of hallucinations reported in crime_rag.py).
    # "Strong" criminal signals (explicit IPC section numbers, punishment
    # keywords) are the only case where criminal RAG is allowed alongside
    # civil/constitutional signals.
    # =========================================================================

    # Determine sub-domain routing from existing substantive match counts
    use_constitutional_rag = constitutional_matches > 0
    use_civil_rag = (
        civil_matches > 0 or property_matches > 0
        or family_matches > 0 or tech_matches > 0
    )

    # Apply the isolation guard: force crime_rag OFF when civil/constitutional
    # signals dominate AND there are no strong explicit criminal statute signals.
    if (use_constitutional_rag or use_civil_rag) and strong_rag_matches == 0:
        use_crime_rag = False
        if use_constitutional_rag:
            rag_reasons.append(
                "constitutional domain detected — criminal RAG disabled (domain isolation)"
            )
        if use_civil_rag:
            rag_reasons.append(
                "civil/property/tech domain detected — criminal RAG disabled (domain isolation)"
            )

    return ToolSelection(
        use_indian_kanoon=use_indian_kanoon,
        use_crime_rag=use_crime_rag,
        use_civil_rag=use_civil_rag,
        use_constitutional_rag=use_constitutional_rag,
        use_lawyer_finder=use_lawyer_finder,
        use_document_analyzer=use_document_analyzer,
        use_llm_only=use_llm_only,
        reasoning=" | ".join(reasons) if reasons else "default routing",
    )


async def _stage1_domain_check(text: str) -> DomainClassification:
    """
    Stage 1: Hierarchical Routing - Domain Check
    Determines if the query is Legal or Non-Legal.
    Uses fast keyword matching (zero-latency).
    """
    text_lower = text.lower()

    # Check for non-legal patterns first
    is_non_legal = _fast_keyword_check(text, NON_LEGAL_PATTERNS)

    # Check for legal domain indicators
    legal_matches = [kw for kw in LEGAL_DOMAIN_KEYWORDS if kw in text_lower]
    has_legal_context = len(legal_matches) > 0

    # If clearly non-legal and no legal context
    if is_non_legal and not has_legal_context:
        return DomainClassification(is_legal=False, confidence=0.9, legal_indicators=[])

    # If has legal indicators
    if has_legal_context:
        confidence = min(0.95, 0.6 + len(legal_matches) * 0.1)
        return DomainClassification(
            is_legal=True,
            confidence=confidence,
            legal_indicators=legal_matches[:5],  # Top 5 indicators
        )

    # Ambiguous - assume legal with lower confidence
    return DomainClassification(is_legal=True, confidence=0.5, legal_indicators=[])


def _stage2_specialization(
    text: str, has_document: bool, domain: DomainClassification
) -> RoutingDecision:
    """
    Stage 2: Hierarchical Routing - Specialization
    If Legal, determine: Transactional (Documents), Informational (General), or Personal (Crime).
    Uses keyword-based fast routing with confidence scoring.
    """
    text_lower = text.lower()

    # If non-legal, return immediately
    if not domain.is_legal:
        return RoutingDecision(
            primary_intent="non_legal",
            confidence=domain.confidence,
            reasoning="Query is not related to legal matters",
            extracted_entities=[],
            requires_tools=[],
        )

    # Extract entities for context
    entities = _extract_legal_entities(text)

    # =========================================================================
    # Priority 1: Document Context (if document is present)
    # =========================================================================
    if has_document:
        # Check if validation is requested
        wants_validation = _fast_keyword_check(text, VALIDATION_KEYWORDS)
        reasoning = (
            "Document present with validation request"
            if wants_validation
            else "Document present for analysis"
        )
        return RoutingDecision(
            primary_intent="document_analysis",
            confidence=0.95 if wants_validation else 0.9,
            reasoning=reasoning,
            extracted_entities=entities,
            requires_tools=["document_analyzer", "indian_kanoon"],
        )

    # =========================================================================
    # Priority 2: Personal Crime Reports (urgent/immediate)
    # =========================================================================
    is_personal_crime = _fast_keyword_check(text, PERSONAL_CRIME_INDICATORS)
    if is_personal_crime:
        # Check for multi-intent (crime + lawyer)
        needs_lawyer = _fast_keyword_check(text, LAWYER_SEARCH_KEYWORDS)
        secondary = ["find_lawyer"] if needs_lawyer else []

        return RoutingDecision(
            primary_intent="crime_report",
            confidence=0.9,
            reasoning="Personal crime incident reported",
            secondary_intents=secondary,
            extracted_entities=entities,
            requires_tools=["crime_rag"],
        )

    # =========================================================================
    # Priority 3: Lawyer Search
    # =========================================================================
    needs_lawyer = _fast_keyword_check(text, LAWYER_SEARCH_KEYWORDS)
    if needs_lawyer:
        # Check for multi-intent (lawyer + crime context)
        has_crime_context = _count_keyword_matches(text, CRIME_TYPE_KEYWORDS) > 0
        secondary = []
        tools = ["lawyer_finder"]

        if has_crime_context:
            secondary.append("crime_report")
            tools.append("crime_rag")

        return RoutingDecision(
            primary_intent="find_lawyer",
            confidence=0.85,
            reasoning="Lawyer search requested",
            secondary_intents=secondary,
            extracted_entities=entities,
            requires_tools=tools,
        )

    # =========================================================================
    # Priority 4: Legal Analysis / General Query
    # =========================================================================
    is_legal_analysis = _fast_keyword_check(text, LEGAL_ANALYSIS_KEYWORDS)
    crime_count = _count_keyword_matches(text, CRIME_TYPE_KEYWORDS)
    is_multi_offense = crime_count >= 2
    needs_case_law = _fast_keyword_check(text, CASE_SEARCH_KEYWORDS)
    needs_statute = _fast_keyword_check(text, STATUTE_KEYWORDS)

    # Determine tools needed
    tools = []
    if needs_case_law:
        tools.append("indian_kanoon")
    if needs_statute or is_multi_offense:
        tools.append("crime_rag")

    # Calculate confidence based on indicators
    confidence = 0.7
    if is_legal_analysis:
        confidence += 0.1
    if is_multi_offense:
        confidence += 0.1
    if len(entities) > 0:
        confidence += 0.05
    confidence = min(0.95, confidence)

    reasoning_parts = []
    if is_legal_analysis:
        reasoning_parts.append("legal analysis question")
    if is_multi_offense:
        reasoning_parts.append(f"multi-offense scenario ({crime_count} crimes)")
    if needs_case_law:
        reasoning_parts.append("requires case law search")
    if needs_statute:
        reasoning_parts.append("requires statute lookup")
    if not reasoning_parts:
        reasoning_parts.append("general legal query")

    return RoutingDecision(
        primary_intent="general_query",
        confidence=confidence,
        reasoning="; ".join(reasoning_parts),
        extracted_entities=entities,
        requires_tools=tools if tools else [],
    )


async def _llm_routing_fallback(
    text: str, history: List[Message], has_document: bool
) -> RoutingDecision:
    """
    LLM-based routing fallback for ambiguous cases.
    Uses structured output for confidence scoring.
    """
    try:
        llm = get_fast_llm()

        # Build context from history
        history_text = ""
        if history:
            recent = history[-4:]  # Last 2 exchanges
            history_text = "\n".join(
                [f"{m['role']}: {m['content'][:100]}" for m in recent]
            )

        prompt = f"""Analyze this legal query and classify the intent. Respond in JSON format.

Query: {text}
Document Uploaded: {has_document}
Recent History: {history_text}

Classification options:
- "crime_report": Personal incident (theft, assault) seeking help
- "general_query": Legal questions, IPC/CrPC analysis, multi-offense scenarios
- "document_analysis": Analyzing or validating a legal document
- "find_lawyer": Looking for legal representation
- "non_legal": Not related to law

Respond ONLY with a JSON object like:
{{"intent": "general_query", "confidence": 0.8, "reasoning": "Legal analysis question about sections", "tools": ["crime_rag"]}}

Available tools: indian_kanoon (case law), crime_rag (IPC/CrPC), lawyer_finder, document_analyzer"""

        response = await invoke_llm_safely(llm, prompt)

        # Parse JSON response
        import json

        # Try to extract JSON from response
        response_clean = response.strip()
        if response_clean.startswith("```"):
            response_clean = response_clean.split("```")[1]
            if response_clean.startswith("json"):
                response_clean = response_clean[4:]

        try:
            data = json.loads(response_clean)
            intent = data.get("intent", "general_query")
            confidence = float(data.get("confidence", 0.6))
            reasoning = data.get("reasoning", "LLM classification")
            tools = data.get("tools", [])

            # Validate intent
            valid_intents = [
                "document_analysis",
                "crime_report",
                "find_lawyer",
                "general_query",
                "non_legal",
            ]
            if intent == "document_validation":
                intent = "document_analysis"
            if intent not in valid_intents:
                intent = "general_query"

            return RoutingDecision(
                primary_intent=intent,
                confidence=confidence,
                reasoning=reasoning,
                requires_tools=tools,
            )
        except json.JSONDecodeError:
            pass

        # Fallback: try to extract intent from text
        response_lower = response.lower()
        for intent in [
            "crime_report",
            "document_analysis",
            "find_lawyer",
            "non_legal",
        ]:
            if intent in response_lower:
                return RoutingDecision(
                    primary_intent=intent,
                    confidence=0.6,
                    reasoning="LLM classification (text extraction)",
                    requires_tools=[],
                )

        return RoutingDecision(
            primary_intent="general_query",
            confidence=0.5,
            reasoning="LLM fallback - could not parse response",
            requires_tools=[],
        )

    except Exception as e:
        print(f"LLM routing fallback error: {e}")
        return RoutingDecision(
            primary_intent="general_query",
            confidence=0.4,
            reasoning=f"LLM error fallback: {str(e)}",
            requires_tools=[],
        )


async def classify_intent(state: ChatState) -> ChatState:
    """
    Hybrid Intent Classification with Hierarchical Routing.

    Architecture:
    1. Fast Keyword-Based Pre-Classification (Zero-Latency)
    2. Stage 1: Domain Check (Legal vs Non-Legal)
    3. Stage 2: Specialization (Document/Crime/Lawyer/General)
    4. Confidence-Based LLM Fallback (for ambiguous cases)
    5. Tool Selection based on intent and query analysis

    Returns enriched state with:
    - intent: Primary classification
    - routing_confidence: Confidence score (0-1)
    - routing_reasoning: Explanation
    - secondary_intents: For multi-intent queries
    - selected_tools: Tools to be used by handlers
    - extracted_entities: Legal terms found
    """
    user_input = state["current_input"]
    has_document = bool(state.get("document_content"))
    messages = state.get("messages", [])
    input_lower = user_input.lower()

    print(f"[Router] Input: {user_input[:100]}...")
    print(f"[Router] Has document: {has_document}")

    # =========================================================================
    # FAST PATH: Document with short query → Document Analysis
    # =========================================================================
    if has_document and len(user_input.split()) < 5:
        tools = _determine_tools_needed(user_input, "document_analysis", has_document)
        print(f"[Router] Fast path: document_analysis (short query with document)")
        return {
            **state,
            "intent": "document_analysis",
            "routing_confidence": 0.95,
            "routing_reasoning": "Document present with brief query",
            "selected_tools": ["document_analyzer", "indian_kanoon"],
            "active_document_context": True,
            "is_ambiguous": False,
        }

    # =========================================================================
    # STAGE 1: Domain Check (Legal vs Non-Legal)
    # =========================================================================
    domain = await _stage1_domain_check(user_input)
    print(
        f"[Router] Stage 1 - Domain: is_legal={domain.is_legal}, confidence={domain.confidence:.2f}"
    )

    # =========================================================================
    # STAGE 2: Specialization
    # =========================================================================
    decision = _stage2_specialization(user_input, has_document, domain)
    print(
        f"[Router] Stage 2 - Decision: {decision.primary_intent}, confidence={decision.confidence:.2f}"
    )
    print(f"[Router] Reasoning: {decision.reasoning}")
    print(f"[Router] Tools: {decision.requires_tools}")

    # =========================================================================
    # CONFIDENCE-BASED LLM FALLBACK
    # =========================================================================
    if decision.confidence < 0.6:
        print(
            f"[Router] Low confidence ({decision.confidence:.2f}), using LLM fallback..."
        )
        llm_decision = await _llm_routing_fallback(user_input, messages, has_document)

        # Use LLM decision if it has higher confidence
        if llm_decision.confidence > decision.confidence:
            print(
                f"[Router] LLM improved: {llm_decision.primary_intent}, confidence={llm_decision.confidence:.2f}"
            )
            decision = llm_decision
        else:
            print(f"[Router] Keeping keyword-based decision")

    # =========================================================================
    # TOOL SELECTION (refine based on final decision)
    # =========================================================================
    tools = _determine_tools_needed(user_input, decision.primary_intent, has_document)

    # Merge tool lists
    selected_tools = list(
        set(
            decision.requires_tools
            + (["indian_kanoon"] if tools.use_indian_kanoon else [])
            + (["crime_rag"] if tools.use_crime_rag else [])
            + (["lawyer_finder"] if tools.use_lawyer_finder else [])
            + (["document_analyzer"] if tools.use_document_analyzer else [])
        )
    )

    print(f"[Router] Final: intent={decision.primary_intent}, tools={selected_tools}")

    # =========================================================================
    # BUILD ENRICHED STATE
    # =========================================================================
    return {
        **state,
        "intent": decision.primary_intent,
        "routing_confidence": decision.confidence,
        "routing_reasoning": decision.reasoning,
        "is_ambiguous": decision.confidence < 0.6,
        "secondary_intents": decision.secondary_intents,
        "extracted_entities": decision.extracted_entities,
        "selected_tools": selected_tools,
        "active_document_context": has_document,
    }


async def handle_document_analysis(state: ChatState) -> ChatState:
    """
    Handle document analysis and validation requests.
    Analyzes uploaded documents and provides structured insights.
    If the user asks for validation/compliance checking, runs the 3-layer
    validation pipeline (classification → statutory checklist → legal reasoning).
    Otherwise uses the enhanced analysis pipeline with IndianKanoon and RAG.
    """
    document_content = state.get("document_content", "")
    document_type = state.get("document_type", "unknown")
    user_query = state.get("current_input", "")

    # If no document content, redirect to general query handler instead of showing upload prompt
    if not document_content:
        # Check if user is explicitly asking to upload
        input_lower = user_query.lower()
        if any(
            kw in input_lower
            for kw in ["upload", "i will upload", "how to upload", "can i upload"]
        ):
            response = """I can help you analyze and validate legal documents and images!

Please upload a document (PDF, DOCX, TXT) or image (JPG, PNG) and I'll provide:
- Document type identification and OCR extraction (for images)
- Summary of key points
- Relevant legal references from IndianKanoon
- Statutory compliance validation and defect analysis
- Crime reporting guidance (if applicable)
- Legal implications and concerns
- Suggested next steps

You can upload your document using the upload feature."""

            return {
                **state,
                "response": response,
                "messages": state["messages"]
                + [{"role": "assistant", "content": response}],
            }
        else:
            # Reroute to general query since no document was actually provided
            return await handle_general_query(state)

    # Check if user is asking for validation/compliance checking
    wants_validation = _fast_keyword_check(user_query, VALIDATION_KEYWORDS)
    if wants_validation:
        return await _handle_document_validation(state)

    # ALWAYS use Indian Kanoon API for document analysis (priority)
    # Run Indian Kanoon and Crime RAG initialization in parallel for better latency
    indian_kanoon = None
    indian_kanoon_results = []
    crime_rag = None

    async def init_indian_kanoon():
        """Initialize Indian Kanoon in parallel."""
        try:
            indian_kanoon_tool = get_indian_kanoon_tool()
            await indian_kanoon_tool.initialize()
            doc_summary = document_content[:500]
            ik_result = await indian_kanoon_tool.search_and_analyze(doc_summary)
            results = ik_result.get("documents", [])
            print(
                f"Indian Kanoon found {len(results)} relevant legal references for document"
            )
            return indian_kanoon_tool, results
        except Exception as e:
            print(f"Indian Kanoon search error in document analysis: {e}")
            return None, []

    async def init_crime_rag():
        """Initialize Crime RAG in parallel."""
        try:
            from app.tools.crime_rag import get_rag_system

            rag_system = get_rag_system()
            await rag_system.initialize()
            return rag_system
        except Exception:
            return None

    # Run both initializations in parallel
    ik_task = asyncio.create_task(init_indian_kanoon())
    rag_task = asyncio.create_task(init_crime_rag())

    # Wait for both to complete
    (indian_kanoon, indian_kanoon_results), crime_rag = await asyncio.gather(
        ik_task, rag_task
    )

    # Track whether at least one RAG source succeeded (compulsory RAG)
    rag_succeeded = bool(indian_kanoon_results) or (
        crime_rag is not None and crime_rag.initialized
    )

    # Use the enhanced document analysis pipeline
    try:
        from app.tools.document_analysis_pipeline import get_document_analysis_pipeline

        llm = get_llm()

        # Create pipeline and analyze
        pipeline = get_document_analysis_pipeline(llm, indian_kanoon, crime_rag)
        result = await pipeline.analyze_document(
            document_text=document_content,
            document_type=document_type,
            user_query=user_query,
        )

        # Format the response
        response_parts = [result.summary]

        if result.key_points:
            response_parts.append("\n\n**Key Points:**")
            for i, point in enumerate(result.key_points, 1):
                response_parts.append(f"{i}. {point}")

        # Prioritize Indian Kanoon results
        if indian_kanoon_results:
            response_parts.append(
                "\n\n**Relevant Legal References from Indian Kanoon:**"
            )
            for ref in indian_kanoon_results[:5]:
                response_parts.append(f"\n• **{ref.title}**")
                response_parts.append(f"  {ref.excerpt[:150]}...")
                response_parts.append(f"  [View on IndianKanoon]({ref.url})")
        elif result.legal_references:
            response_parts.append("\n\n**Relevant Legal References:**")
            for ref in result.legal_references[:3]:
                response_parts.append(f"\n• **{ref['title']}**")
                response_parts.append(f"  {ref['excerpt'][:150]}...")
                response_parts.append(f"  [View on IndianKanoon]({ref['url']})")

        if result.crime_context:
            response_parts.append("\n\n**Crime Reporting Context:**")
            passages = result.crime_context.get("relevant_passages", [])
            for passage in passages[:2]:
                response_parts.append(f"• {passage[:200]}...")

        if result.warnings:
            response_parts.append("\n\n**Note:**")
            for warning in result.warnings:
                response_parts.append(f"⚠️ {warning}")

        response = "\n".join(response_parts)

        # Compulsory RAG: if retrieval failed, prepend disclaimer
        if not rag_succeeded:
            response = (
                "⚠️ **Legal database retrieval was unavailable.** The following analysis "
                "is based on the document text alone without authoritative legal references. "
                "Please retry or consult a qualified legal practitioner.\n\n" + response
            )

        return {
            **state,
            "response": response,
            "document_info": {
                "text": (
                    document_content[:1000] + "..."
                    if len(document_content) > 1000
                    else document_content
                ),
                "summary": result.summary,
                "key_points": result.key_points,
                "document_type": document_type,
                "legal_references": result.legal_references,
                "confidence": result.confidence,
            },
            "messages": state["messages"]
            + [{"role": "assistant", "content": response}],
        }
    except Exception as e:
        # Fallback to basic analysis
        error_msg = f"Enhanced analysis unavailable: {str(e)}"
        print(error_msg)

        # Basic fallback analysis
        llm = get_llm()
        max_chars = 15000
        doc_text = document_content[:max_chars]
        if len(document_content) > max_chars:
            doc_text += (
                "\n\n[Document truncated for analysis. Full document is longer.]"
            )

        prompt = DOCUMENT_ANALYSIS_PROMPT.format(document_text=doc_text)
        analysis = await invoke_llm_safely(llm, prompt)

        # Compulsory RAG: always prepend disclaimer when using fallback path
        analysis = (
            "⚠️ **Legal database retrieval was unavailable.** The following analysis "
            "is based on the document text alone without authoritative legal references. "
            "Please retry or consult a qualified legal practitioner.\n\n" + analysis
        )

        return {
            **state,
            "response": analysis,
            "document_info": {
                "text": (
                    document_content[:1000] + "..."
                    if len(document_content) > 1000
                    else document_content
                ),
                "summary": analysis[:500],
                "key_points": [],
                "document_type": document_type,
            },
            "messages": state["messages"]
            + [{"role": "assistant", "content": analysis}],
        }


async def handle_crime_report(state: ChatState) -> ChatState:
    """
    Handle crime reporting and guidance requests.
    Uses two-stage legal RAG pipeline:
    1. Extract crime features (violence, intent, weapon, etc.)
    2. Retrieve IPC sections via vector recall → legal reranking → legal constraints
    3. Feed structured IPC sections to LLM for court-safe response
    """
    user_input = state["current_input"]
    crime_details = state.get("crime_details") or user_input

    # Detect crime type using keyword matching
    identified_crime = detect_crime_type(crime_details)

    # Two-stage RAG: retrieve IPC/BNS sections with legal reranking
    rag_sections_text = ""
    rag_result = None
    rag_succeeded = False  # Compulsory RAG tracking
    try:
        # ── Use CriminalRAGSystem (not the old monolithic CrimeRAGSystem) ──
        from app.tools.criminal_rag import extract_crime_features, get_criminal_rag_system

        rag_system = get_criminal_rag_system()
        await rag_system.initialize()

        if rag_system.initialized:
            # Extract crime features for metadata-aware retrieval
            features = extract_crime_features(crime_details)
            print(
                f"Crime features: violence={features.violence}, death={features.death}, "
                f"weapon={features.weapon}, intent={features.intent}, "
                f"property={features.property_loss}, trespass={features.trespass}, "
                f"threat={features.threat}"
            )

            # Legal retrieval with minimality (fewer, more accurate sections)
            rag_result = await rag_system.retrieve_sections(
                crime_details,
                crime_type=identified_crime,
                features=features,
                k=2,  # Legal minimality: 1-2 primary chargeable sections
            )

            if rag_result.ipc_sections:
                rag_succeeded = True  # RAG returned results
                # Build section reference with title and punishment for the LLM
                section_lines = []
                for match in rag_result.ipc_sections:
                    section_lines.append(
                        f"• IPC Section {match.section} ({match.title})\n  Punishment: {match.punishment}"
                    )
                rag_sections_text = "\n".join(section_lines)
                print(
                    f"RAG retrieved {len(rag_result.ipc_sections)} IPC sections for '{identified_crime}' "
                    f"(avg confidence: {rag_result.confidence:.0%}, "
                    f"sections: {[m.section for m in rag_result.ipc_sections]})"
                )
    except Exception as e:
        print(f"RAG lookup error (non-critical): {e}")
        import traceback

        traceback.print_exc()

    # Build prompt for the finetuned LLM
    llm = get_llm()

    rag_section = ""
    if rag_sections_text:
        rag_section = f"""\n\nAPPLICABLE IPC SECTIONS:
{rag_sections_text}"""

    # Compulsory RAG: when RAG failed, instruct LLM not to fabricate sections
    no_rag_warning = ""
    if not rag_succeeded:
        no_rag_warning = (
            "\n\nWARNING: Legal statute retrieval was unavailable. "
            "Do NOT fabricate or guess specific IPC/CrPC section numbers. "
            "Instead, refer to offences by name and recommend consulting a lawyer "
            "for precise statutory references."
        )

    prompt = f"""Indian law assistant. User reporting a crime. You MUST respond with ALL 4 sections in this EXACT format:

**Crime:** [2-4 word crime name]

**Statute:** [IPC sections from data below, e.g. "IPC Section 379 (Theft)"]

**Punishment:** [Copy punishment from data below]

**Further Steps:** [Steps: call 100/112, file FIR, preserve evidence]

Crime reported: {crime_details}
Type: {identified_crime}{rag_section}{no_rag_warning}

IMPORTANT: All 4 sections (Crime, Statute, Punishment, Further Steps) are REQUIRED. Use the IPC sections provided above."""

    try:
        final_response = await invoke_llm_safely(llm, prompt)
    except Exception as e:
        print(f"LLM error in crime report: {e}")
        final_response = f"""**Crime:** {identified_crime.replace("_", " ").title()}

**Statute:** Please consult with police or a lawyer for applicable IPC/CrPC sections.

**Punishment:** Varies based on the specific offense and severity. Consult a lawyer for details.

**Further Steps to be Taken:** If in immediate danger, call 100 (Police) or 112 (Emergency). Visit the nearest police station to file an FIR under CrPC Section 154. Preserve all evidence including photographs, documents, and witness contact information. Consult a criminal lawyer for legal guidance."""

    # Compulsory RAG: if RAG failed, prepend visible disclaimer
    if not rag_succeeded:
        final_response = (
            "⚠️ **Legal statute retrieval was unavailable.** The following guidance "
            "is general and may not cite accurate IPC/CrPC section numbers. "
            "Please consult a lawyer for precise statutory references.\n\n"
            + final_response
        )

    return {
        **state,
        "response": final_response,
        "crime_details": crime_details,
        "crime_report": {
            "crime_type": identified_crime,
        },
        "messages": state["messages"]
        + [{"role": "assistant", "content": final_response}],
    }


async def handle_find_lawyer(state: ChatState) -> ChatState:
    """
    Handle lawyer search requests.
    Finds relevant lawyers based on user needs and location.
    """
    user_input = state["current_input"]
    lawyer_query = state.get("lawyer_query") or user_input

    # Get lawyer finder tool
    finder = get_lawyer_finder()

    # Search for lawyers
    lawyers = finder.search_by_query(lawyer_query, limit=5)
    formatted_results = finder.format_lawyer_results(lawyers)

    # Optionally use Indian Kanoon to provide legal context for lawyer search
    legal_context = ""
    query_lower = lawyer_query.lower()
    if any(
        kw in query_lower
        for kw in ["criminal", "civil", "family", "property", "divorce", "ipc", "case"]
    ):
        try:
            ik_tool = get_indian_kanoon_tool()
            await ik_tool.initialize()

            # Get relevant legal context
            ik_result = await ik_tool.search_and_analyze(lawyer_query, max_results=2)
            docs = ik_result.get("documents", [])
            if docs:
                legal_context = "\n\n**Relevant Legal Context:**\n"
                for doc in docs[:2]:
                    legal_context += f"• {doc.title}\n"
                print(f"Added Indian Kanoon legal context to lawyer search")
        except Exception as e:
            print(f"Indian Kanoon error for lawyer search: {e}")

    # Enhance with LLM for personalized recommendations
    try:
        llm = get_llm()
        prompt = LAWYER_SEARCH_PROMPT.format(
            query=lawyer_query, lawyer_results=formatted_results
        )
        # Add legal context if available
        if legal_context:
            prompt = f"{prompt}\n\n{legal_context}"

        final_response = await invoke_llm_safely(llm, prompt)
    except Exception:
        # Use formatted results directly if LLM fails
        final_response = f"""Based on your request, I found some lawyers who might be able to help:

{formatted_results}

**Tips for choosing a lawyer:**
1. Schedule consultations with 2-3 lawyers before deciding
2. Ask about their experience with cases like yours
3. Discuss fees and payment structure upfront
4. Trust your instincts about communication style

Would you like me to search with different criteria?"""

    # Convert to LawyerInfo format
    lawyers_info: List[LawyerInfo] = [
        {
            "name": l.name,
            "specialization": l.specialization,
            "location": l.location,
            "contact": l.contact,
            "rating": l.rating,
            "experience_years": l.experience_years,
        }
        for l in lawyers
    ]

    return {
        **state,
        "response": final_response,
        "lawyer_query": lawyer_query,
        "lawyers_found": lawyers_info,
        "messages": state["messages"]
        + [{"role": "assistant", "content": final_response}],
    }


async def handle_general_query(state: ChatState) -> ChatState:
    """
    Handle general legal questions and complex legal analysis.

    Uses the selected_tools from routing to determine which tools to invoke:
    - indian_kanoon: Case law, precedents, court decisions
    - crime_rag: IPC/CrPC sections, punishments, procedures
    - LLM only: General explanations (when selected_tools is empty)

    This handles:
    - Multi-offense scenarios (forgery + assault + threat + trespass)
    - Cross-act questions (IPC + Prevention of Corruption Act + IT Act)
    - Procedural questions (cognizable/non-cognizable, CrPC procedures)
    - Sanction requirements, jurisdictional questions
    """
    user_input = state["current_input"]
    messages = state.get("messages", [])

    # Get tools selected by the router
    selected_tools = state.get("selected_tools", [])
    extracted_entities = state.get("extracted_entities", [])
    routing_reasoning = state.get("routing_reasoning", "")

    print(f"[GeneralQuery] Selected tools: {selected_tools}")
    print(f"[GeneralQuery] Extracted entities: {extracted_entities}")

    # Build conversation context from recent messages (last 3-4 exchanges)
    conversation_context = ""
    if len(messages) > 1:
        recent_messages = messages[-6:]  # Last 3 exchanges (user + assistant)
        context_parts = []
        for msg in recent_messages:
            role = msg["role"]
            content = msg["content"][:200]  # Truncate long messages
            context_parts.append(f"{role.upper()}: {content}")
        conversation_context = "\n".join(context_parts)

    input_lower = user_input.lower()

    # =========================================================================
    # TOOL EXECUTION BASED ON ROUTER SELECTION
    # =========================================================================

    # Determine which tools to use from selected_tools (from router)
    use_indian_kanoon = "indian_kanoon" in selected_tools
    use_crime_rag = "crime_rag" in selected_tools
    # New domain-specific flags (set by _determine_tools_needed via ToolSelection)
    use_civil_rag = state.get("use_civil_rag", False)
    use_constitutional_rag = state.get("use_constitutional_rag", False)

    # If civil/constitutional flags were captured in ToolSelection but not propagated
    # to state yet, fall back to keyword matching.
    if not use_civil_rag and not use_constitutional_rag and not use_crime_rag:
        use_civil_rag = _fast_keyword_check(user_input, CIVIL_LAW_KEYWORDS) or \
                        _fast_keyword_check(user_input, PROPERTY_LAW_KEYWORDS) or \
                        _fast_keyword_check(user_input, TECH_LAW_KEYWORDS)
        use_constitutional_rag = _fast_keyword_check(user_input, CONSTITUTIONAL_KEYWORDS)

    # Fallback: If no tools selected but query has legal substance, force tools
    if not selected_tools:
        # Check all substantive law domains
        has_substantive_law = (
            _fast_keyword_check(user_input, CONSTITUTIONAL_KEYWORDS)
            or _fast_keyword_check(user_input, CIVIL_LAW_KEYWORDS)
            or _fast_keyword_check(user_input, PROPERTY_LAW_KEYWORDS)
            or _fast_keyword_check(user_input, FAMILY_LAW_KEYWORDS)
            or _fast_keyword_check(user_input, TECH_LAW_KEYWORDS)
            or _fast_keyword_check(user_input, CASE_SEARCH_KEYWORDS)
            or _fast_keyword_check(user_input, LEGAL_DOMAIN_KEYWORDS)
        )
        # Check criminal law domain
        has_criminal_law = _fast_keyword_check(
            user_input, STATUTE_KEYWORDS
        ) or _fast_keyword_check(user_input, CRIMINAL_PROCEDURE_KEYWORDS)
        crime_count = _count_keyword_matches(user_input, CRIME_TYPE_KEYWORDS)

        # Force Indian Kanoon for any substantive legal question
        if has_substantive_law:
            use_indian_kanoon = True
        # Force Crime RAG for criminal matters
        if has_criminal_law or crime_count >= 2:
            use_crime_rag = True
        # If nothing matched but it's routed to general_query, still try IK
        if not use_indian_kanoon and not use_crime_rag:
            use_indian_kanoon = True  # Default: use IK for any legal query

    print(
        f"[GeneralQuery] Tool execution: indian_kanoon={use_indian_kanoon}, crime_rag={use_crime_rag}"
    )

    # Calculate multi-offense for RAG k parameter
    crime_count = _count_keyword_matches(user_input, CRIME_TYPE_KEYWORDS)
    is_multi_offense = crime_count >= 2

    # =========================================================================
    # PARALLEL TOOL EXECUTION
    # =========================================================================

    indian_kanoon_results = ""
    rag_sections_text = ""
    rag_result = None

    async def fetch_indian_kanoon():
        """Fetch case law from Indian Kanoon."""
        try:
            ik_tool = get_indian_kanoon_tool()
            await ik_tool.initialize()

            # Determine context type for targeted search
            context_type = "general"
            if "ipc" in input_lower or "penal code" in input_lower:
                context_type = "ipc"
            elif "crpc" in input_lower or "criminal procedure" in input_lower:
                context_type = "crpc"
            elif any(
                kw in input_lower
                for kw in (
                    "article",
                    "constitution",
                    "fundamental right",
                    "directive principle",
                    "writ",
                    "preamble",
                    "amendment",
                    "right to privacy",
                    "right to life",
                    "right to equality",
                    "freedom of speech",
                    "puttaswamy",
                    "kesavananda",
                    "parliament",
                    "basic structure",
                    "public order",
                    "central law",
                    "state government",
                    "surveillance",
                )
            ):
                context_type = "constitution"
            elif any(
                kw in input_lower
                for kw in (
                    "bail",
                    "anticipatory bail",
                    "fir",
                    "quash",
                    "cognizable",
                    "complainant",
                    "criminal case",
                    "compoundable",
                    "withdraw",
                    "marital rape",
                    "rape",
                    "economic offence",
                )
            ):
                context_type = "crpc"
            elif any(
                kw in input_lower
                for kw in (
                    "contract",
                    "agreement",
                    "oral agreement",
                    "force majeure",
                    "non-compete",
                    "restraint of trade",
                    "breach",
                    "coercion",
                    "undue influence",
                    "enforceable",
                    "voidable",
                    "consideration",
                    "sale of goods",
                    "partnership",
                    "negotiable instrument",
                    "specific relief",
                    "limitation act",
                    "arbitration",
                    "consumer protection",
                    "insolvency",
                )
            ):
                context_type = "statute"
            elif any(
                kw in input_lower
                for kw in (
                    "property",
                    "ancestral",
                    "heir",
                    "coparcener",
                    "partition",
                    "transfer of property",
                    "registration act",
                    "easement",
                    "succession",
                    "hindu marriage",
                    "special marriage",
                    "maintenance",
                    "divorce",
                    "custody",
                    "adoption",
                    "domestic violence",
                    "dowry",
                    "live-in",
                    "family",
                )
            ):
                context_type = "statute"
            elif any(
                kw in input_lower
                for kw in (
                    "evidence",
                    "admissible",
                    "whatsapp",
                    "electronic record",
                    "certificate",
                    "witness",
                )
            ):
                context_type = "statute"
            elif any(
                kw in input_lower
                for kw in (
                    "crypto",
                    "cryptocurrency",
                    "cyber",
                    "data protection",
                    "it act",
                    "information technology",
                    "ai system",
                    "artificial intelligence",
                    "online",
                    "digital",
                    "photos shared",
                    "privacy",
                    "fema",
                    "pmla",
                    "rbi",
                    "sebi",
                    "companies act",
                    "prevention of corruption",
                )
            ):
                context_type = "statute"

            result = await ik_tool.answer_legal_query(user_input, context_type)
            return result.get("formatted_results", "")
        except Exception as e:
            print(f"Indian Kanoon error: {e}")
            return ""

    async def fetch_rag_sections():
        """Fetch IPC/BNS/CrPC sections from the criminal RAG system."""
        try:
            from app.tools.criminal_rag import extract_crime_features, get_criminal_rag_system

            rag_system = get_criminal_rag_system()
            await rag_system.initialize()

            if not rag_system.initialized:
                return None, ""

            features = extract_crime_features(user_input)
            k = 5 if is_multi_offense else 3

            rag_result_inner = await rag_system.retrieve_sections(
                user_input,
                crime_type="general",
                features=features,
                k=k,
            )

            if rag_result_inner.ipc_sections:
                MIN_CONFIDENCE = 0.40
                relevant_sections = [
                    m for m in rag_result_inner.ipc_sections
                    if m.confidence >= MIN_CONFIDENCE
                ]
                dropped = len(rag_result_inner.ipc_sections) - len(relevant_sections)
                if dropped:
                    print(
                        f"RAG relevance filter: dropped {dropped} low-confidence sections "
                        f"(threshold={MIN_CONFIDENCE})"
                    )
                if not relevant_sections:
                    return None, ""

                section_lines = []
                for match in relevant_sections:
                    section_info = (
                        f"• **Section {match.section}** ({match.title})\n"
                        f"  - Punishment: {match.punishment}\n"
                    )
                    section_lines.append(section_info)

                sections_text = "\n".join(section_lines)
                print(
                    f"Criminal RAG: {len(relevant_sections)} sections retrieved: "
                    f"{[m.section for m in relevant_sections]}"
                )
                return rag_result_inner, sections_text

            return None, ""
        except Exception as e:
            print(f"Criminal RAG lookup error: {e}")
            import traceback; traceback.print_exc()
            return None, ""

    async def fetch_civil_sections():
        """Fetch civil/contract/property law provisions from the civil RAG system."""
        try:
            from app.tools.civil_rag import get_civil_rag_system

            civil_system = get_civil_rag_system()
            await civil_system.initialize()

            if not civil_system.initialized:
                return ""

            context = await civil_system.retrieve(user_input, k=4, min_score=0.30)
            if not context.chunks:
                return ""

            lines = []
            for chunk in context.chunks:
                lines.append(
                    f"• **{chunk.act_name} § {chunk.section_number}** — {chunk.title}\n"
                    f"  {chunk.text[:300]}"
                )
            text = "\n\n".join(lines)
            print(
                f"Civil RAG: {len(context.chunks)} provisions retrieved "
                f"(confidence: {context.confidence:.2%})"
            )
            return text
        except Exception as e:
            print(f"Civil RAG error: {e}")
            return ""

    async def fetch_constitutional_articles():
        """Fetch constitutional Articles from the constitutional RAG system."""
        try:
            from app.tools.constitutional_rag import get_constitutional_rag_system

            const_system = get_constitutional_rag_system()
            await const_system.initialize()

            if not const_system.initialized:
                return ""

            context = await const_system.retrieve(user_input, k=3, min_score=0.30)
            if not context.chunks:
                return ""

            lines = []
            for chunk in context.chunks:
                lines.append(
                    f"• **{chunk.section_number}** — {chunk.title}\n"
                    f"  {chunk.text[:400]}"
                )
            text = "\n\n".join(lines)
            print(
                f"Constitutional RAG: {len(context.chunks)} Articles retrieved "
                f"(confidence: {context.confidence:.2%})"
            )
            return text
        except Exception as e:
            print(f"Constitutional RAG error: {e}")
            return ""

    # Execute tools in parallel based on router selection
    tasks = []
    task_names = []
    rag_succeeded = False  # Compulsory RAG tracking

    # New domain-specific RAG context containers
    civil_context_text = ""
    constitutional_context_text = ""

    if use_indian_kanoon:
        tasks.append(fetch_indian_kanoon())
        task_names.append("indian_kanoon")

    if use_crime_rag:
        tasks.append(fetch_rag_sections())
        task_names.append("rag")

    if use_civil_rag:
        tasks.append(fetch_civil_sections())
        task_names.append("civil_rag")

    if use_constitutional_rag:
        tasks.append(fetch_constitutional_articles())
        task_names.append("constitutional_rag")

    if tasks:
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"Tool {task_names[i]} failed: {result}")
                continue

            if task_names[i] == "indian_kanoon":
                indian_kanoon_results = result if result else ""
                if indian_kanoon_results:
                    rag_succeeded = True
            elif task_names[i] == "rag":
                if result and len(result) == 2:
                    rag_result, rag_sections_text = result
                    if rag_sections_text:
                        rag_succeeded = True
            elif task_names[i] == "civil_rag":
                civil_context_text = result if result else ""
                if civil_context_text:
                    rag_succeeded = True
            elif task_names[i] == "constitutional_rag":
                constitutional_context_text = result if result else ""
                if constitutional_context_text:
                    rag_succeeded = True

    # =========================================================================
    # BUILD PROMPT WITH RETRIEVED CONTEXT
    # =========================================================================

    try:
        llm = get_llm()

        # Build context sections
        context_parts = []

        if rag_sections_text:
            context_parts.append(
                f"""**Applicable Criminal Law Sections (IPC/BNS/CrPC):**
{rag_sections_text}"""
            )

        if civil_context_text:
            context_parts.append(
                f"""**Applicable Civil Law Provisions:**
{civil_context_text}

NOTE: The above are civil law provisions. They govern remedies, obligations, and rights — NOT criminal punishments. Do NOT describe civil breaches as criminal offences."""
            )

        if constitutional_context_text:
            context_parts.append(
                f"""**Applicable Constitutional Provisions:**
{constitutional_context_text}"""
            )

        if indian_kanoon_results:
            context_parts.append(
                f"""**Relevant Case Law & Precedents:**
{indian_kanoon_results}"""
            )

        retrieved_context = "\n\n".join(context_parts) if context_parts else ""


        # Choose appropriate prompt based on context
        if retrieved_context:
            # Use enhanced prompt with retrieved legal context
            prompt = f"""You are a knowledgeable Indian legal assistant. Answer the following legal query comprehensively using ONLY the retrieved legal context below.

**User Query:** {user_input}

{retrieved_context}

**CRITICAL ACCURACY RULES:**
- You MUST base your answer on the retrieved context above. Cite specific sections, articles, case names, and provisions that appear in the context.
- If the retrieved context does not cover a particular aspect of the query, say "I don't have specific references for this aspect" rather than guessing.
- NEVER fabricate or guess section numbers, article numbers, or case citations.
- If the legal position has changed or is contested, explicitly state that.
- Cite landmark cases BY NAME when they appear in the retrieved context.

**Instructions:**
1. Directly answer the user's question using information from the retrieved context
2. Cite the specific legal provisions, sections, or case law from the context that support your answer
3. Explain the legal principles and reasoning clearly
4. If multiple provisions or cases apply, explain how they relate to each other
5. Note any exceptions, limitations, or conditions that apply
6. If the retrieved context includes case law, reference the relevant holdings

Provide a comprehensive, well-structured answer. Use headers and bullet points for clarity.

End with: "This is general legal information. For specific advice on your situation, please consult a lawyer registered with the Bar Council of India."
"""
        else:
            # No retrieved context — tools returned empty or were not used.
            # Use general prompt with extra caution about ungrounded claims.
            prompt = GENERAL_QUERY_PROMPT.format(query=user_input)
            if use_indian_kanoon or use_crime_rag:
                # Tools were attempted but returned no results — CRITICAL anti-hallucination warning
                prompt += """

🚨 CRITICAL WARNING: Legal database searches returned NO RELEVANT RESULTS for this query.

You MUST follow these rules strictly:
1. DO NOT cite ANY specific IPC/CrPC section numbers (e.g., DO NOT say "Section 420 IPC" or "Section 438 CrPC")
2. DO NOT cite specific Article numbers from the Constitution
3. DO NOT cite specific case names or citations
4. Refer to laws ONLY by their full Act name (e.g., "Indian Penal Code, 1860" or "Code of Criminal Procedure, 1973")
5. Use general legal principles and concepts ONLY
6. Start your answer with: "I could not retrieve specific statutory references from my legal database for this query."
7. ALWAYS recommend: "Please consult a qualified lawyer registered with the Bar Council of India for specific statutory citations and authoritative legal advice."

If you cite ANY specific section number, article number, or case citation, you are HALLUCINATING."""

        # Add conversation context if available
        if conversation_context:
            prompt = f"""Previous conversation context:
{conversation_context}

{prompt}"""

        final_response = await invoke_llm_safely(llm, prompt)

    except Exception as e:
        print(f"LLM error in general query: {e}")
        final_response = """I apologize, but I'm having trouble processing your request right now.

In the meantime, I can help you with:
1. **Document Analysis** - Upload a legal document for analysis
2. **Crime Reporting** - Get guidance on reporting crimes and next steps
3. **Find a Lawyer** - Search for attorneys based on your needs

Please try rephrasing your question or selecting one of the options above."""

    # Compulsory RAG: if tools were requested but retrieval failed, prepend disclaimer
    if (use_indian_kanoon or use_crime_rag) and not rag_succeeded:
        final_response = (
            "⚠️ **I was unable to retrieve authoritative legal references for this query.** "
            "The response below is based on general knowledge and may not contain "
            "accurate statutory citations. Please verify with a qualified legal practitioner.\n\n"
            + final_response
        )

    return {
        **state,
        "response": final_response,
        "messages": state["messages"]
        + [{"role": "assistant", "content": final_response}],
    }


async def _handle_document_validation(state: ChatState) -> ChatState:
    """
    Internal handler for document validation using the 3-layer pipeline.
    Called by handle_document_analysis when validation is requested.

    Layer 1: Document Classification (deterministic, rule-based)
    Layer 2: Statutory Checklist Validation (rule-based, no LLM)
    Layer 3: Legal Reasoning & Defect Explanation (LLM-based)

    Output is framed as identifying potential issues — NEVER provides
    binding legal opinions or states "this document is legally valid."
    """
    document_content = state.get("document_content", "")
    user_query = state.get("current_input", "")

    # If no document content, show upload prompt
    if not document_content:
        from app.prompts import DOCUMENT_VALIDATION_UPLOAD_PROMPT

        response = DOCUMENT_VALIDATION_UPLOAD_PROMPT
        return {
            **state,
            "response": response,
            "messages": state["messages"]
            + [{"role": "assistant", "content": response}],
        }

    try:
        # ================================================================
        # Layer 1: Document Classification (deterministic)
        # ================================================================
        classifier = get_document_classifier()
        classification = classifier.classify(document_content)

        print(
            f"[Layer 1] Document classified as: {classification.document_type} "
            f"(confidence: {classification.confidence:.2f})"
        )

        # ================================================================
        # Layer 2: Statutory Checklist Validation (rule-based, no LLM)
        # ================================================================
        validator = get_statutory_validator()
        validation = validator.validate(document_content, classification.document_type)

        print(
            f"[Layer 2] Statutory validation: {validation.passed}/{validation.total_checks} passed, "
            f"compliance score: {validation.compliance_score:.0%}"
        )

        # ================================================================
        # Layer 2.5: Retrieve Indian Law Context (RAG)
        # ================================================================
        # Initialize Indian Kanoon and Crime RAG in parallel
        indian_kanoon = None
        crime_rag = None

        async def init_ik():
            try:
                ik_tool = get_indian_kanoon_tool()
                await ik_tool.initialize()
                return ik_tool
            except Exception as e:
                print(f"Indian Kanoon init error: {e}")
                return None

        async def init_rag():
            try:
                from app.tools.crime_rag import get_rag_system

                rag_system = get_rag_system()
                await rag_system.initialize()
                return rag_system
            except Exception:
                return None

        indian_kanoon, crime_rag = await asyncio.gather(init_ik(), init_rag())

        # Get Indian law context via RAG tool
        law_rag = get_indian_law_rag(indian_kanoon, crime_rag)
        law_context = await law_rag.retrieve_context(
            document_type=classification.document_type,
            missing_elements=validation.missing_elements,
            non_compliance=validation.non_compliance,
            document_text=document_content[:2000],
            jurisdiction_hints=classification.jurisdiction_hints,
        )

        print(
            f"[Layer 2.5] Retrieved {len(law_context.references)} law references, "
            f"{len(law_context.applicable_acts)} applicable acts"
        )

        # ================================================================
        # Layer 3: Legal Reasoning & Defect Explanation (LLM)
        # ================================================================
        llm = get_llm()
        analyzer = get_legal_defect_analyzer(llm)
        result = await analyzer.analyze_defects(
            classification=classification,
            validation=validation,
            law_context=law_context,
            document_text=document_content[:5000],
        )

        response = result["formatted_response"]

        print(
            f"[Layer 3] Analysis complete. Defects: {result['defect_count']}, "
            f"Compliance: {result['compliance_score']:.0%}"
        )

        # Build validation info for state
        validation_info: DocumentValidationInfo = {
            "classified_type": classification.document_type,
            "classification_confidence": classification.confidence,
            "sub_type": classification.sub_type,
            "jurisdiction_hints": classification.jurisdiction_hints,
            "compliance_score": validation.compliance_score,
            "total_checks": validation.total_checks,
            "passed": validation.passed,
            "failed": validation.failed,
            "missing_elements": validation.missing_elements,
            "present_elements": validation.present_elements,
            "non_compliance": validation.non_compliance,
            "llm_analysis": result["llm_analysis"],
            "applicable_acts": law_context.applicable_acts,
            "applicable_sections": law_context.applicable_sections,
            "precedent_notes": law_context.precedent_notes,
            "state_specific_notes": law_context.state_specific_notes,
            "reasoning_trace": result.get("reasoning_trace"),
        }

        return {
            **state,
            "response": response,
            "document_validation": validation_info,
            "messages": state["messages"]
            + [{"role": "assistant", "content": response}],
        }

    except Exception as e:
        print(f"Document validation error: {e}")
        import traceback

        traceback.print_exc()

        # Fallback: try basic classification and validation without LLM
        try:
            classifier = get_document_classifier()
            classification = classifier.classify(document_content)
            validator = get_statutory_validator()
            validation = validator.validate(
                document_content, classification.document_type
            )

            fallback_parts = [
                "**⚠️ Disclaimer:** This analysis is for informational purposes only and does not constitute a binding legal opinion.",
                "",
                f"## 📄 Document Classification",
                f"**Type:** {classification.document_type}",
                f"**Confidence:** {classification.confidence:.0%}",
                "",
                f"## 📊 Statutory Compliance: {validation.compliance_score:.0%}",
            ]

            if validation.missing_elements:
                fallback_parts.append("\n## ❌ Missing Mandatory Elements")
                for item in validation.missing_elements:
                    fallback_parts.append(
                        f"- **{item['element']}** — {item['description']}"
                    )
                    fallback_parts.append(f"  📜 *{item['statute_reference']}*")

            if validation.non_compliance:
                fallback_parts.append("\n## ⚠️ Non-Compliance")
                for item in validation.non_compliance:
                    fallback_parts.append(
                        f"- **{item['element']}** — {item['description']}"
                    )

            fallback_parts.append(
                "\n---\n*Detailed legal analysis temporarily unavailable. "
                "The above findings are based on statutory checklist validation. "
                "Please consult a qualified legal practitioner for comprehensive review.*"
            )

            response = "\n".join(fallback_parts)
        except Exception:
            response = (
                "I apologize, but I encountered an error while validating your document. "
                "Please try again or consult a qualified legal practitioner for document review."
            )

        return {
            **state,
            "response": response,
            "error": str(e),
            "messages": state["messages"]
            + [{"role": "assistant", "content": response}],
        }


async def handle_non_legal_query(state: ChatState) -> ChatState:
    """
    Handle non-legal queries with a polite rejection message.
    """
    response = """I'm a legal assistance chatbot specializing in Indian law. I can help you with:

• Legal questions and advice
• Crime reporting guidance
• Document analysis (contracts, agreements, etc.)
• Finding lawyers
• Understanding Indian laws (IPC, CrPC, IT Act, etc.)

For other topics, I may not be the best resource. Please ask me a legal question!"""

    return {
        **state,
        "response": response,
        "messages": state["messages"] + [{"role": "assistant", "content": response}],
    }


# ============================================================================
# Router Function
# ============================================================================


def route_by_intent(
    state: ChatState,
) -> Literal[
    "document_analysis",
    "crime_report",
    "find_lawyer",
    "general_query",
    "non_legal",
]:
    """Route to the appropriate handler based on classified intent."""
    intent = state.get("intent")
    if intent in (
        "document_analysis",
        "crime_report",
        "find_lawyer",
        "general_query",
        "non_legal",
    ):
        return intent
    return "general_query"


# ============================================================================
# Graph Builder
# ============================================================================


def build_legal_chatbot_graph() -> StateGraph:
    """
    Build the LangGraph workflow for the legal chatbot.

    Graph structure:
    START -> classify_intent -> [route_by_intent] -> handler -> END
    """
    # Create the graph
    workflow = StateGraph(ChatState)

    # Add nodes
    workflow.add_node("classify_intent", classify_intent)
    workflow.add_node("document_analysis", handle_document_analysis)
    workflow.add_node("crime_report", handle_crime_report)
    workflow.add_node("find_lawyer", handle_find_lawyer)
    workflow.add_node("general_query", handle_general_query)
    workflow.add_node("non_legal", handle_non_legal_query)

    # Set entry point
    workflow.set_entry_point("classify_intent")

    # Add conditional routing
    workflow.add_conditional_edges(
        "classify_intent",
        route_by_intent,
        {
            "document_analysis": "document_analysis",
            "crime_report": "crime_report",
            "find_lawyer": "find_lawyer",
            "general_query": "general_query",
            "non_legal": "non_legal",
        },
    )

    # All handlers go to END
    workflow.add_edge("document_analysis", END)
    workflow.add_edge("crime_report", END)
    workflow.add_edge("find_lawyer", END)
    workflow.add_edge("general_query", END)
    workflow.add_edge("non_legal", END)

    return workflow


# ============================================================================
# Chatbot Class
# ============================================================================


class LegalChatbot:
    """
    Main chatbot class that wraps the LangGraph workflow.
    Provides a clean interface for the API layer.
    """

    def __init__(self):
        workflow = build_legal_chatbot_graph()
        self.graph = workflow.compile()
        self._sessions: Dict[str, List[Message]] = {}

    def _get_session_messages(self, session_id: str) -> List[Message]:
        """Get or create session message history."""
        if session_id not in self._sessions:
            self._sessions[session_id] = []
        return self._sessions[session_id]

    def _add_message(self, session_id: str, message: Message):
        """Add a message to session history."""
        if session_id not in self._sessions:
            self._sessions[session_id] = []
        self._sessions[session_id].append(message)

        # Keep only last 20 messages for memory efficiency
        if len(self._sessions[session_id]) > 20:
            self._sessions[session_id] = self._sessions[session_id][-20:]

    async def stream_chat(
        self,
        message: str,
        session_id: str = "default",
        document_content: Optional[str] = None,
        document_type: Optional[str] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream chat response token by token.
        Yields dicts: {"type": "token", "content": "..."} or {"type": "done", ...}
        """
        # Get session history
        messages = self._get_session_messages(session_id)

        # Add user message to history
        user_message: Message = {"role": "user", "content": message}
        self._add_message(session_id, user_message)

        # Build initial state
        initial_state: ChatState = {
            "messages": messages,
            "current_input": message,
            "conversation_context": None,
            "intent": None,
            "document_content": document_content,
            "document_type": document_type or "unknown",
            "document_info": None,
            "document_validation": None,
            "crime_details": None,
            "crime_report": None,
            "lawyer_query": None,
            "lawyers_found": None,
            "response": None,
            "session_id": session_id,
            "error": None,
        }

        # Phase 1: Classification (non-streaming)
        classified_state = await classify_intent(initial_state)
        intent = route_by_intent(classified_state)

        # Phase 2: Run handler with streaming
        handler_map = {
            "document_analysis": handle_document_analysis,
            "crime_report": handle_crime_report,
            "find_lawyer": handle_find_lawyer,
            "general_query": handle_general_query,
            "non_legal": handle_non_legal_query,
        }

        handler = handler_map.get(intent, handle_general_query)

        # Set up streaming queue
        queue: asyncio.Queue = asyncio.Queue()
        tokens_streamed = False

        async def run_handler():
            _stream_queue_var.set(queue)
            try:
                return await handler(classified_state)
            except Exception as e:
                print(f"Handler error during streaming: {e}")
                return {
                    **classified_state,
                    "response": f"I apologize, but I encountered an error processing your request. Please try again.",
                    "error": str(e),
                }
            finally:
                await queue.put(None)  # Signal completion

        task = asyncio.create_task(run_handler())

        # Yield tokens as they arrive
        while True:
            chunk = await queue.get()
            if chunk is None:
                break
            tokens_streamed = True
            yield {"type": "token", "content": chunk}

        # Wait for handler to complete and get result
        result = await task

        # If no tokens were streamed (non-LLM path), yield full response
        if not tokens_streamed and result.get("response"):
            yield {"type": "token", "content": result["response"]}

        # Add assistant response to session history
        response_text = result.get("response", "")
        if response_text:
            assistant_message: Message = {
                "role": "assistant",
                "content": response_text,
            }
            self._add_message(session_id, assistant_message)

        # Yield completion event with metadata
        yield {
            "type": "done",
            "session_id": session_id,
            "intent": result.get("intent") or intent,
            "lawyers_found": result.get("lawyers_found"),
            "document_info": result.get("document_info"),
            "document_validation": result.get("document_validation"),
            "crime_report": result.get("crime_report"),
        }

    async def chat(
        self,
        message: str,
        session_id: str = "default",
        document_content: Optional[str] = None,
        document_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Process a chat message and return the response.

        Args:
            message: User's message
            session_id: Session identifier for conversation context
            document_content: Optional document content if user uploaded a file
            document_type: Type of uploaded document (pdf, image_ocr, etc.)

        Returns:
            Dict containing response and any additional data
        """
        # Get session history
        messages = self._get_session_messages(session_id)

        # Add user message to history
        user_message: Message = {"role": "user", "content": message}
        self._add_message(session_id, user_message)

        # Build initial state
        initial_state: ChatState = {
            "messages": messages,
            "current_input": message,
            "conversation_context": None,
            "intent": None,
            "document_content": document_content,
            "document_type": document_type or "unknown",  # Pass document type to state
            "document_info": None,
            "document_validation": None,
            "crime_details": None,
            "crime_report": None,
            "lawyer_query": None,
            "lawyers_found": None,
            "response": None,
            "session_id": session_id,
            "error": None,
        }

        # Run the graph
        result = await self.graph.ainvoke(initial_state)

        # Add assistant response to history
        if result.get("response"):
            assistant_message: Message = {
                "role": "assistant",
                "content": result["response"],
            }
            self._add_message(session_id, assistant_message)

        # Return structured response
        return {
            "response": result.get(
                "response", "I'm sorry, I couldn't process your request."
            ),
            "intent": result.get("intent"),
            "document_info": result.get("document_info"),
            "document_validation": result.get("document_validation"),
            "crime_report": result.get("crime_report"),
            "lawyers_found": result.get("lawyers_found"),
            "error": result.get("error"),
        }

    def clear_session(self, session_id: str):
        """Clear a session's message history."""
        if session_id in self._sessions:
            del self._sessions[session_id]

    def get_session_history(self, session_id: str) -> List[Message]:
        """Get the message history for a session."""
        return self._get_session_messages(session_id).copy()


# Singleton instance
_chatbot: Optional[LegalChatbot] = None


def get_chatbot() -> LegalChatbot:
    """Get or create the chatbot instance."""
    global _chatbot
    if _chatbot is None:
        _chatbot = LegalChatbot()
    return _chatbot
