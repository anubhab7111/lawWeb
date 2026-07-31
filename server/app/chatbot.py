"""
LangGraph-based legal chatbot implementation.
This module defines the chatbot workflow using LangGraph for state management and routing.
"""

import asyncio
import time
import contextvars
import re
from asyncio.events import AbstractEventLoop
from functools import lru_cache
from typing import Any, AsyncGenerator, Dict, List, Literal, Optional

from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from app.config import get_settings
from app.prompts import (
    DOCUMENT_ANALYSIS_PROMPT,
    DOCUMENT_VALIDATION_UPLOAD_PROMPT,
    GENERAL_QUERY_PROMPT,
    LAWYER_SEARCH_PROMPT,
    QUERY_REWRITE_PROMPT,
)
from app.state import (
    ChatState,
    DocumentValidationInfo,
    LawyerInfo,
    Message,
)
from app.intent_classifier import classify_intent_embedding
from app.tool_dispatch import RAG_TOOL_REGISTRY, infer_indian_kanoon_context_type
from app.tools.crime_reporter import detect_crime_type
from app.tools.document_classifier import get_document_classifier
from app.tools.indian_kanoon import get_indian_kanoon_tool
from app.tools.indian_law_rag import get_indian_law_rag
from app.tools.lawyer_finder import get_lawyer_finder
from app.tools.legal_defect_analyzer import get_legal_defect_analyzer
from app.tools.statutory_validator import get_statutory_validator

# ============================================================================
# Pydantic Models for Structured Routing
# ============================================================================


class DomainClassification(BaseModel):
    """Stage 1: Domain-level classification (Legal vs Non-Legal)."""

    is_legal: bool = Field(description="Whether the query is related to legal matters")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    legal_indicators: List[str] = Field(
        default_factory=list, description="Legal terms or concepts found in the query"
    )


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

# ============================================================================
# Domain keyword banks — sole remaining consumer is _infer_domain_hint
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
        num_ctx=6144,  # Ollama defaults to 2048, which silently clips grounded prompts
        num_predict=1024,  # Balanced: enough for detailed answers, faster inference
        timeout=35.0,  # Tighter timeout for snappier responses
        reasoning=False,  # qwen3 defaults to thinking mode; keep responses direct
        keep_alive="1h",  # loading the 14B model is the OOM-prone step — do it rarely
    )


@lru_cache()
def get_fast_llm() -> ChatOllama:
    """Get cached small LLM for classification/routing tasks (local Ollama)."""
    settings = get_settings()
    return ChatOllama(
        model=settings.fast_llm_model,
        temperature=0,
        base_url=settings.ollama_base_url,
        num_ctx=4096,  # rewrite/classification prompts include conversation history
        num_predict=128,  # Reduced from 256 for faster classification
        timeout=15.0,  # Reduced from 30s
        reasoning=False,  # classification needs the raw JSON, not a thinking preamble
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


async def _rewrite_query_for_retrieval(
    messages: List[Message], current_input: str
) -> str:
    """
    Condense conversation history + the latest message into one standalone
    retrieval query using the fast LLM. First turns (no prior exchange)
    return the input unchanged with zero added latency; any failure or
    degenerate output also falls back to the raw input.
    """
    # Exclude the current input if it's already the last history entry
    prior = messages
    if prior and prior[-1]["role"] == "user" and prior[-1]["content"] == current_input:
        prior = prior[:-1]
    if not any(m["role"] == "assistant" for m in prior):
        return current_input

    history_lines = [
        f"{m['role'].upper()}: {m['content'][:300]}" for m in prior[-4:]
    ]
    prompt = QUERY_REWRITE_PROMPT.format(
        history="\n".join(history_lines), question=current_input
    )

    try:
        loop = asyncio.get_event_loop()
        response = await asyncio.wait_for(
            loop.run_in_executor(
                None, lambda: get_fast_llm().invoke([HumanMessage(content=prompt)])
            ),
            timeout=8.0,
        )
        rewritten = str(response.content).strip().strip('"').strip()
        if not rewritten or len(rewritten) > 300 or "\n" in rewritten:
            return current_input
        if rewritten.lower() != current_input.lower():
            print(f"[Router] Retrieval query rewritten: {rewritten[:120]}")
        return rewritten
    except Exception as e:
        print(f"[Router] Query rewrite failed ({e}) — using raw input.")
        return current_input


# ============================================================================
# Primary Router (embedding-based) + deterministic policy layer
# ============================================================================

# The only keyword signals treated as "strong, unambiguous criminal" for
# _infer_domain_hint below — explicit section/procedure references, not
# generic crime-adjacent words. Deliberately narrow.
STRONG_CRIMINAL_SIGNAL_KEYWORDS = frozenset(
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

# Hard-wired per-intent tool sets — metadata for state["selected_tools"];
# each handler still calls its specific tool_dispatch.invoke_* function(s)
# directly (bespoke prompt assembly per handler), it doesn't loop over this
# dict generically. find_lawyer deliberately omits indian_kanoon here — that
# handler keeps its own cheap local keyword gate (purely locational lawyer
# searches get no benefit from case-law retrieval).
INTENT_TOOL_MAP: Dict[str, List[str]] = {
    "document_analysis": ["indian_kanoon"],
    "crime_report": ["crime_sections"],
    "general_query": ["indian_kanoon", "statute_context"],
    "find_lawyer": ["lawyer_finder"],
    "non_legal": [],
}


def _infer_domain_hint(text: str) -> Optional[Literal["criminal"]]:
    """
    Deterministic domain bias for the unified statute retrieval: "criminal"
    only when explicit criminal-statute vocabulary is present AND no other
    legal domain signal is competing for it without a *strong*, explicit
    criminal-statute signal (section number, "punishable under", etc.).
    Prevents IPC punishment clauses from contaminating civil/constitutional/
    property/family/tech retrieval context — purely keyword-driven, so it
    never depends on a classifier's guess being right.
    """
    has_criminal_signal = (
        _fast_keyword_check(text, STATUTE_KEYWORDS)
        or _fast_keyword_check(text, CRIMINAL_PROCEDURE_KEYWORDS)
        or _count_keyword_matches(text, CRIME_TYPE_KEYWORDS) > 0
    )
    if not has_criminal_signal:
        return None

    has_other_domain_signal = (
        _fast_keyword_check(text, CONSTITUTIONAL_KEYWORDS)
        or _fast_keyword_check(text, CIVIL_LAW_KEYWORDS)
        or _fast_keyword_check(text, PROPERTY_LAW_KEYWORDS)
        or _fast_keyword_check(text, FAMILY_LAW_KEYWORDS)
        or _fast_keyword_check(text, TECH_LAW_KEYWORDS)
    )
    has_strong_criminal_signal = _fast_keyword_check(
        text, STRONG_CRIMINAL_SIGNAL_KEYWORDS
    )

    if has_other_domain_signal and not has_strong_criminal_signal:
        return None
    return "criminal"


_GROUNDING_UNAVAILABLE_DISCLAIMER = (
    "⚠️ **I was unable to retrieve authoritative legal references for this query.** "
    "The response below is based on general knowledge and may not contain accurate "
    "statutory citations. Please verify with a qualified legal practitioner.\n\n"
)

_GROUNDING_UNAVAILABLE_PROMPT_WARNING = """

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


def _apply_compulsory_rag_policy(rag_succeeded: bool) -> tuple:
    """
    Single shared implementation of the "grounding unavailable" pattern,
    used identically across handle_document_analysis, handle_crime_report,
    and handle_general_query. Returns (disclaimer_prefix, prompt_warning):
    - disclaimer_prefix: prepend to the final response when rag_succeeded
      is False (empty string when grounding succeeded — prepend is a no-op).
    - prompt_warning: append to the generation prompt when rag_succeeded is
      False, instructing the LLM not to fabricate citations it wasn't given.
    """
    if rag_succeeded:
        return "", ""
    return _GROUNDING_UNAVAILABLE_DISCLAIMER, _GROUNDING_UNAVAILABLE_PROMPT_WARNING


async def classify_intent(state: ChatState) -> ChatState:
    """
    Intent classification and tool selection.

    Architecture:
    1. Fast path: document + very short query -> document_analysis
    2. Zero-latency keyword pre-filter: non-legal short-circuit (skips the
       embedding classifier entirely for casual chat)
    3. Primary router: embedding nearest-centroid classification
       (classify_intent_embedding) — no LLM anywhere in this path
    4. Deterministic domain_hint inference (_infer_domain_hint), independent
       of the classifier
    5. History-aware query rewrite for retrieval (unchanged)

    Returns enriched state with:
    - intent: Primary classification
    - routing_confidence: Confidence score (0-1)
    - routing_reasoning: Explanation
    - secondary_intents: For multi-intent queries
    - selected_tools: Tools to be used by handlers (from INTENT_TOOL_MAP)
    - domain_hint: Soft bias for unified statute retrieval
    - extracted_entities: Legal terms found
    """
    user_input = state["current_input"]
    has_document = bool(state.get("document_content"))
    messages = state.get("messages", [])

    print(f"[Router] Input: {user_input[:100]}...")
    print(f"[Router] Has document: {has_document}")

    # =========================================================================
    # FAST PATH: Document with short query → Document Analysis
    # =========================================================================
    if has_document and len(user_input.split()) < 5:
        print(f"[Router] Fast path: document_analysis (short query with document)")
        return {
            **state,
            "intent": "document_analysis",
            "routing_confidence": 0.95,
            "routing_reasoning": "Document present with brief query",
            "selected_tools": INTENT_TOOL_MAP["document_analysis"],
            "domain_hint": None,
            "active_document_context": True,
            "is_ambiguous": False,
        }

    # =========================================================================
    # ZERO-LATENCY PRE-FILTER: Domain Check (Legal vs Non-Legal)
    # =========================================================================
    domain = await _stage1_domain_check(user_input)
    print(
        f"[Router] Stage 1 - Domain: is_legal={domain.is_legal}, confidence={domain.confidence:.2f}"
    )
    if not domain.is_legal:
        print(f"[Router] Non-legal short-circuit — skipping embedding classifier")
        return {
            **state,
            "intent": "non_legal",
            "routing_confidence": domain.confidence,
            "routing_reasoning": "Query is not related to legal matters",
            "selected_tools": [],
            "domain_hint": None,
            "active_document_context": has_document,
        }

    # =========================================================================
    # PRIMARY ROUTER: embedding nearest-centroid classification
    # =========================================================================
    result = await classify_intent_embedding(user_input, has_document)
    # Ambiguous classifications default to general_query (still grounded,
    # just not the specific handler) rather than falling back to an LLM —
    # no model call anywhere in this routing path.
    intent = "general_query" if result.is_ambiguous else result.primary_intent

    domain_hint = _infer_domain_hint(user_input)
    entities = _extract_legal_entities(user_input)
    print(
        f"[Router] Decision: intent={intent}, confidence={result.confidence:.3f}, "
        f"margin={result.margin:.3f}, ambiguous={result.is_ambiguous}, "
        f"domain_hint={domain_hint}, secondary={result.secondary_intents}"
    )

    # =========================================================================
    # HISTORY-AWARE QUERY REWRITE (multi-turn only; no-op on first turns)
    # =========================================================================
    retrieval_query = await _rewrite_query_for_retrieval(messages, user_input)

    # =========================================================================
    # BUILD ENRICHED STATE
    # =========================================================================
    return {
        **state,
        "retrieval_query": retrieval_query,
        "intent": intent,
        "routing_confidence": result.confidence,
        "routing_reasoning": result.reasoning,
        "is_ambiguous": result.is_ambiguous,
        "secondary_intents": result.secondary_intents,
        "extracted_entities": entities,
        "selected_tools": INTENT_TOOL_MAP.get(intent, []),
        "domain_hint": domain_hint,
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
            ik_result = await RAG_TOOL_REGISTRY["indian_kanoon"](doc_summary)
            results = ik_result.raw.get("results", []) if ik_result.raw else []
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
            from app.tools.criminal_rag import get_criminal_rag_system

            rag_system = get_criminal_rag_system()
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
    2. Retrieve IPC/BNS sections via FAISS semantic search, sorted by score
    3. Feed structured IPC sections to LLM for court-safe response
    """
    user_input = state["current_input"]
    # Prefer the history-aware standalone query for retrieval on follow-ups
    crime_details = (
        state.get("crime_details") or state.get("retrieval_query") or user_input
    )

    # Detect crime type using keyword matching
    identified_crime = detect_crime_type(crime_details)

    # Retrieve IPC/BNS sections via the shared dispatcher (legal minimality:
    # k=2, fewer/more-accurate chargeable sections)
    ik_result = await RAG_TOOL_REGISTRY["crime_sections"](
        crime_details, crime_type=identified_crime, k=2
    )
    rag_sections_text = ik_result.context_text
    rag_succeeded = ik_result.succeeded

    # Build prompt for the finetuned LLM
    llm = get_llm()

    rag_section = ""
    if rag_sections_text:
        rag_section = f"""\n\nAPPLICABLE IPC SECTIONS:
{rag_sections_text}"""

    # Compulsory RAG: when RAG failed, instruct LLM not to fabricate sections
    disclaimer_prefix, no_rag_warning = _apply_compulsory_rag_policy(rag_succeeded)

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
    if disclaimer_prefix:
        final_response = disclaimer_prefix + final_response

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

    # Optionally use Indian Kanoon to provide legal context for lawyer search —
    # purely locational searches ("find a lawyer near me") get no benefit
    # from case-law retrieval, so this stays gated on a cheap keyword check
    # rather than running unconditionally like general_query's tools.
    legal_context = ""
    query_lower = lawyer_query.lower()
    if any(
        kw in query_lower
        for kw in ["criminal", "civil", "family", "property", "divorce", "ipc", "case"]
    ):
        ik_result = await RAG_TOOL_REGISTRY["indian_kanoon"](lawyer_query)
        if ik_result.succeeded:
            docs = ik_result.raw.get("results", [])
            if docs:
                legal_context = "\n\n**Relevant Legal Context:**\n"
                for doc in docs[:2]:
                    legal_context += f"• {doc.title}\n"
                print(f"Added Indian Kanoon legal context to lawyer search")

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


async def _verify_response_citations(response_text: str) -> str:
    """
    Non-LLM post-generation check: does every 'Section N of the X Act' /
    'Article N' claim in the answer actually exist in the indexed corpus
    under the cited act? Appends a footer flagging mismatches; silent when
    everything verifies. Never raises — a verifier bug must not break chat.
    """
    try:
        from app.tools.citation_verifier import verification_footer, verify_citations
        from app.tools.unified_legal_rag import get_unified_rag_system

        rag = get_unified_rag_system()
        if not rag.initialized:
            return response_text
        report = verify_citations(response_text, rag)
        if report.checks:
            print(
                f"[CitationVerify] {len(report.verified)}/{len(report.checks)} "
                f"citations verified"
            )
        return response_text + verification_footer(report)
    except Exception as e:
        print(f"[CitationVerify] Skipped due to error: {e}")
        return response_text


async def handle_general_query(state: ChatState) -> ChatState:
    """
    Handle general legal questions and complex legal analysis.

    Always runs both Indian Kanoon case-law search and unified statute
    retrieval — general_query's tool set is hard-wired (INTENT_TOOL_MAP),
    not decided per-query, since the unified index already covers every
    legal domain and always ran unconditionally in practice.

    This handles:
    - Multi-offense scenarios (forgery + assault + threat + trespass)
    - Cross-act questions (IPC + Prevention of Corruption Act + IT Act)
    - Procedural questions (cognizable/non-cognizable, CrPC procedures)
    - Sanction requirements, jurisdictional questions
    """
    user_input = state["current_input"]
    messages = state.get("messages", [])

    # Standalone query for retrieval (rewritten from conversation history
    # when this is a follow-up turn); generation still sees the raw input.
    retrieval_query = state.get("retrieval_query") or user_input
    domain_hint = state.get("domain_hint")
    extracted_entities = state.get("extracted_entities", [])

    print(f"[GeneralQuery] domain_hint={domain_hint} extracted_entities={extracted_entities}")

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

    # Multi-offense bumps the statute-retrieval k parameter
    crime_count = _count_keyword_matches(user_input, CRIME_TYPE_KEYWORDS)
    is_multi_offense = crime_count >= 2

    async def _fast_llm_invoke(prompt: str) -> str:
        loop = asyncio.get_event_loop()
        response = await asyncio.wait_for(
            loop.run_in_executor(
                None, lambda: get_fast_llm().invoke([HumanMessage(content=prompt)])
            ),
            timeout=8.0,
        )
        return str(response.content)

    # =========================================================================
    # PARALLEL TOOL EXECUTION — both tools always run for general_query
    # =========================================================================
    context_type = infer_indian_kanoon_context_type(user_input)
    ik_task = RAG_TOOL_REGISTRY["indian_kanoon"](retrieval_query, context_type)
    statute_task = RAG_TOOL_REGISTRY["statute_context"](
        retrieval_query,
        k=10 if is_multi_offense else 8,
        domain_hint=["criminal"] if domain_hint == "criminal" else None,
        fast_llm_invoke=_fast_llm_invoke,
    )
    ik_result, statute_result = await asyncio.gather(
        ik_task, statute_task, return_exceptions=True
    )

    indian_kanoon_results = ""
    rag_sections_text = ""
    case_law_text = ""
    rag_succeeded = False  # Compulsory RAG tracking

    if isinstance(ik_result, Exception):
        print(f"Tool indian_kanoon failed: {ik_result}")
    else:
        indian_kanoon_results = ik_result.context_text
        rag_succeeded = rag_succeeded or ik_result.succeeded

    if isinstance(statute_result, Exception):
        print(f"Tool statute_context failed: {statute_result}")
    else:
        rag_sections_text = statute_result.context_text
        case_law_text = (statute_result.raw or {}).get("case_law_text", "")
        rag_succeeded = rag_succeeded or statute_result.succeeded

    # =========================================================================
    # BUILD PROMPT WITH RETRIEVED CONTEXT
    # =========================================================================

    disclaimer_prefix, prompt_warning = _apply_compulsory_rag_policy(rag_succeeded)

    try:
        llm = get_llm()

        # Build context sections
        context_parts = []

        if rag_sections_text:
            context_parts.append(
                f"""**Applicable Statutory Provisions** (each is tagged with its legal domain, e.g. [criminal], [civil], [family]):
{rag_sections_text}

NOTE: Only provisions tagged [criminal] define offences and punishments. Civil/constitutional/other provisions govern rights, remedies, and obligations — do NOT describe them as criminal offences."""
            )

        if case_law_text:
            context_parts.append(
                f"""**Judicial Interpretation** (curated landmark judgments, ordered by court authority — cite the case NAME, do not invent citations beyond what's shown):
{case_law_text}"""
            )

        if indian_kanoon_results:
            context_parts.append(f"""**Relevant Case Law & Precedents:**
{indian_kanoon_results[:3000]}""")

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
            # No retrieved context — tools returned empty.
            # Use general prompt with extra caution about ungrounded claims.
            prompt = GENERAL_QUERY_PROMPT.format(query=user_input) + prompt_warning

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

    # Compulsory RAG: if retrieval failed across all sources, prepend disclaimer
    if disclaimer_prefix:
        final_response = disclaimer_prefix + final_response
    elif rag_sections_text:
        # Only check citations when statute context was actually retrieved —
        # the no-context path already forbids specific citations by prompt.
        final_response = await _verify_response_citations(final_response)

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

    # If no document content, show upload prompt
    if not document_content:
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
                from app.tools.criminal_rag import get_criminal_rag_system

                rag_system = get_criminal_rag_system()
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
        self._session_last_access: Dict[str, float] = {}

    def _evict_stale_sessions(self):
        """Drop sessions idle beyond the TTL and cap the total session count."""
        settings = get_settings()
        now = time.monotonic()
        for sid in [
            sid
            for sid, last in self._session_last_access.items()
            if now - last > settings.session_ttl_seconds
        ]:
            self._sessions.pop(sid, None)
            self._session_last_access.pop(sid, None)

        overflow = len(self._sessions) - settings.max_sessions
        if overflow > 0:
            oldest = sorted(
                self._session_last_access, key=self._session_last_access.get
            )[:overflow]
            for sid in oldest:
                self._sessions.pop(sid, None)
                self._session_last_access.pop(sid, None)

    def _get_session_messages(self, session_id: str) -> List[Message]:
        """Get or create session message history."""
        self._evict_stale_sessions()
        if session_id not in self._sessions:
            self._sessions[session_id] = []
        self._session_last_access[session_id] = time.monotonic()
        return self._sessions[session_id]

    def _add_message(self, session_id: str, message: Message):
        """Add a message to session history."""
        if session_id not in self._sessions:
            self._sessions[session_id] = []
        self._sessions[session_id].append(message)
        self._session_last_access[session_id] = time.monotonic()

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
        self._sessions.pop(session_id, None)
        self._session_last_access.pop(session_id, None)

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
