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

from app.config import get_settings
from app.prompts import (
    CASE_LAW_CONTEXT_BLOCK,
    CRIME_REPORT_FALLBACK,
    CRIME_REPORT_PROMPT,
    DOC_RAG_UNAVAILABLE_DISCLAIMER,
    DOCUMENT_ANALYSIS_PROMPT,
    DOCUMENT_UPLOAD_HELP,
    DOCUMENT_VALIDATION_UPLOAD_PROMPT,
    GENERAL_QUERY_ERROR,
    GENERAL_QUERY_PROMPT,
    GROUNDED_QUERY_PROMPT,
    GROUNDING_UNAVAILABLE_DISCLAIMER,
    GROUNDING_UNAVAILABLE_PROMPT_WARNING,
    INDIAN_KANOON_CONTEXT_BLOCK,
    LAWYER_SEARCH_FALLBACK,
    LAWYER_SEARCH_PROMPT,
    NON_LEGAL_RESPONSE,
    QUERY_REWRITE_PROMPT,
    STATUTE_CONTEXT_BLOCK,
)
from app.routing_keywords import CRIME_TYPE_KEYWORDS
from app.state import (
    ChatState,
    DocumentValidationInfo,
    LawyerInfo,
    Message,
)
from app.intent_classifier import (
    classify_document_subintent_embedding,
    classify_domain_hint_embedding,
    classify_intent_embedding,
)
from app.tool_dispatch import RAG_TOOL_REGISTRY, infer_indian_kanoon_context_type
from app.tools.crime_reporter import detect_crime_type
from app.tools.document_classifier import get_document_classifier
from app.tools.indian_kanoon import get_indian_kanoon_tool
from app.tools.indian_law_rag import get_indian_law_rag
from app.tools.lawyer_recommender import (
    format_lawyer_results,
    recommend_lawyers as recommend_lawyers_core,
)
from app.tools.legal_defect_analyzer import get_legal_defect_analyzer
from app.tools.statutory_validator import get_statutory_validator


# Main-LLM context window / output reservation. Kept as module constants so the
# prompt-budget guard (_fit_context_blocks) sizes the input against the same
# window the model runs with.
LLM_NUM_CTX = 6144  # Ollama defaults to 2048, which silently clips grounded prompts
LLM_NUM_PREDICT = 1024  # tokens reserved for the answer
_PROMPT_SAFETY_MARGIN = 256  # headroom for chat scaffolding the estimate can't see
_MAX_QUERY_CHARS = 8000  # clamp on user-derived text so one huge query can't overflow


@lru_cache()
def get_llm() -> ChatOllama:
    """Get cached LLM instance for better performance."""
    settings = get_settings()
    return ChatOllama(
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        base_url=settings.ollama_base_url,
        num_ctx=LLM_NUM_CTX,
        num_predict=LLM_NUM_PREDICT,
        timeout=35.0,  # Tighter timeout for snappier responses
        reasoning=False,  # qwen3 defaults to thinking mode; keep responses direct
        keep_alive="1h",  # loading the 14B model is the OOM-prone step — do it rarely
    )


def _fit_context_blocks(context_parts: list, reserved_tokens: int) -> str:
    """
    Join retrieved-context blocks (already in priority order: statute → case
    law → Indian Kanoon) without exceeding the model's input budget. Drops
    lower-priority blocks and truncates the last kept one, so the instruction
    template and user query always survive — this is what stops Ollama from
    silently front-truncating the grounded statute block on long prompts.
    """
    from app.metrics.engineering_metrics import count_tokens_approx

    budget = LLM_NUM_CTX - LLM_NUM_PREDICT - _PROMPT_SAFETY_MARGIN - reserved_tokens
    if budget <= 0:
        return ""

    kept: list = []
    used = 0
    for block in context_parts:
        block_tokens = count_tokens_approx(block)
        if used + block_tokens <= budget:
            kept.append(block)
            used += block_tokens
        else:
            remaining = budget - used
            if remaining > 50:  # ~4 chars/token — keep a useful truncated head
                kept.append(block[: remaining * 4])
            break
    return "\n\n".join(kept)


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


@lru_cache()
def get_fast_llm_prose() -> ChatOllama:
    """Same small model as get_fast_llm(), but for short natural-language
    generation (case summaries, plain-language explanations) rather than
    JSON classification. qwen3:4b keeps thinking even with reasoning=False
    (verified directly against the Ollama API — `think: false` is not
    honored by this model/build) and the thinking preamble alone commonly
    runs 300-500 tokens before the real answer starts, so num_predict needs
    real headroom above get_fast_llm()'s 128 or the response gets cut off
    mid-thought before ever reaching the answer."""
    settings = get_settings()
    return ChatOllama(
        model=settings.fast_llm_model,
        temperature=0,
        base_url=settings.ollama_base_url,
        num_ctx=4096,
        num_predict=900,
        timeout=45.0,
        reasoning=False,
    )


_THINK_TAG_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def strip_reasoning_tags(text: str) -> str:
    """Strip a qwen3 thinking preamble from a response. Two shapes seen in
    practice: a full <think>...</think> pair, or — what this model's chat
    template actually produces — only the closing </think> tag, since the
    opening tag is injected into the prompt template rather than generated
    (confirmed against the raw Ollama API). In the latter case a naive
    <think>...</think> regex matches nothing, so fall back to keeping only
    what follows the last </think>."""
    if "</think>" in text:
        return text.rsplit("</think>", 1)[-1].strip()
    return _THINK_TAG_RE.sub("", text).strip()


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
    "find_lawyer": ["lawyer_recommender"],
    "non_legal": [],
}


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
    return GROUNDING_UNAVAILABLE_DISCLAIMER, GROUNDING_UNAVAILABLE_PROMPT_WARNING


async def classify_intent(state: ChatState) -> ChatState:
    """
    Intent classification and tool selection.

    Architecture — fully embedding-based, no keyword gates or hardcoded
    shortcuts anywhere in this path:
    1. Primary router: embedding nearest-centroid classification over 5
       classes, including non_legal (classify_intent_embedding). A
       has_document boost/penalty biases document_analysis appropriately —
       verified empirically that this alone handles even single-word
       document-attached queries ("review", "thoughts?") confidently, so no
       separate word-count fast path is needed.
    2. Domain-hint inference (classify_domain_hint_embedding), an
       independent binary embedding classifier
    3. History-aware query rewrite for retrieval (unchanged)

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
    # PRIMARY ROUTER: embedding nearest-centroid classification, 5-way
    # (document_analysis / crime_report / find_lawyer / general_query /
    # non_legal) — no separate keyword-based domain gate. non_legal is a
    # real competing class here, not a hardcoded-pattern fallback.
    # =========================================================================
    result = await classify_intent_embedding(user_input, has_document)
    # Ambiguous ties among the four *legal* intents default to general_query
    # (still grounded, just not the specific handler) rather than falling
    # back to an LLM — no model call anywhere in this routing path. But if
    # non_legal is itself the top-scoring class, trust it even when the
    # margin is thin: collapsing an ambiguous non_legal read into
    # general_query would silently reintroduce the old "assume legal when
    # unsure" bias non_legal was added to remove.
    intent = (
        "general_query"
        if result.is_ambiguous and result.primary_intent != "non_legal"
        else result.primary_intent
    )

    if intent == "non_legal":
        print(
            f"[Router] Non-legal (confidence={result.confidence:.3f}, "
            f"margin={result.margin:.3f}) — skipping domain_hint/entities/retrieval"
        )
        return {
            **state,
            "intent": "non_legal",
            "routing_confidence": result.confidence,
            "routing_reasoning": result.reasoning,
            "is_ambiguous": result.is_ambiguous,
            "selected_tools": [],
            "domain_hint": None,
            "active_document_context": has_document,
        }

    domain_hint = await classify_domain_hint_embedding(user_input)
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
            response = DOCUMENT_UPLOAD_HELP

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
    subintent = await classify_document_subintent_embedding(user_query)
    if subintent == "validation":
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

    # Track whether at least one RAG source succeeded (compulsory RAG).
    # Provisional: crime_rag's own per-document grounding (result.crime_context,
    # below) isn't available yet — it's folded in once the pipeline returns,
    # since crime_rag.initialized only means "the shared index loaded at
    # some point in this process's life," not "retrieved something for this
    # document."
    rag_succeeded = bool(indian_kanoon_results)

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

        # Fold in crime RAG's actual per-document grounding now that the
        # pipeline has run, instead of the process-lifetime .initialized flag.
        rag_succeeded = rag_succeeded or bool(
            result.crime_context and result.crime_context.get("relevant_passages")
        )

        # Compulsory RAG: if retrieval failed, prepend disclaimer
        if not rag_succeeded:
            response = DOC_RAG_UNAVAILABLE_DISCLAIMER + response

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
        analysis = DOC_RAG_UNAVAILABLE_DISCLAIMER + analysis

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

    prompt = CRIME_REPORT_PROMPT.format(
        crime_details=crime_details[:_MAX_QUERY_CHARS],
        identified_crime=identified_crime,
        rag_section=rag_section,
        no_rag_warning=no_rag_warning,
    )

    try:
        final_response = await invoke_llm_safely(llm, prompt)
    except Exception as e:
        print(f"LLM error in crime report: {e}")
        final_response = CRIME_REPORT_FALLBACK.format(
            crime_name=identified_crime.replace("_", " ").title()
        )

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

    # Real Postgres-backed recommendation (pgvector semantic search + weighted
    # rating/success_rate score). ChatState has no session plumbing, and this
    # is the only DB access chatbot.py needs, so open one locally rather than
    # threading a Session through the whole graph.
    from app.db.engine import get_engine
    from sqlmodel import Session as DBSession

    with DBSession(get_engine()) as session:
        lawyers = await recommend_lawyers_core(session, problem_description=lawyer_query, limit=5)
    formatted_results = format_lawyer_results(lawyers)

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
        final_response = LAWYER_SEARCH_FALLBACK.format(
            formatted_results=formatted_results
        )

    # Convert to LawyerInfo format
    lawyers_info: List[LawyerInfo] = [
        {
            "name": l.name,
            "specialization": l.specialty,
            "location": l.location,
            "contact": None,
            "rating": l.rating,
            "experience_years": l.experience,
            "hourly_rate": l.hourly_rate,
            "success_rate": l.success_rate,
            "bio": l.bio,
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


async def _verify_response_citations(
    response_text: str, retrieved_sections=None
) -> str:
    """
    Non-LLM post-generation check: does every 'Section N of the X Act' /
    'Article N' claim in the answer actually exist in the indexed corpus
    under the cited act? When ``retrieved_sections`` is given, citations that
    exist but were not retrieved for this query are flagged as ungrounded.
    Appends a footer flagging mismatches; silent when everything verifies.
    Never raises — a verifier bug must not break chat.
    """
    try:
        from app.tools.citation_verifier import verification_footer, verify_citations
        from app.tools.unified_legal_rag import get_unified_rag_system

        rag = get_unified_rag_system()
        if not rag.initialized:
            return response_text
        report = verify_citations(response_text, rag, retrieved_sections)
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
                STATUTE_CONTEXT_BLOCK.format(rag_sections_text=rag_sections_text)
            )

        if case_law_text:
            context_parts.append(
                CASE_LAW_CONTEXT_BLOCK.format(case_law_text=case_law_text)
            )

        if indian_kanoon_results:
            context_parts.append(
                INDIAN_KANOON_CONTEXT_BLOCK.format(
                    indian_kanoon_results=indian_kanoon_results[:3000]
                )
            )

        from app.metrics.engineering_metrics import count_tokens_approx

        user_input_for_prompt = user_input[:_MAX_QUERY_CHARS]
        # Reserve budget for the fixed scaffolding (instruction template ~500
        # tokens) plus the query and conversation history, then fit the context
        # blocks into whatever input budget remains.
        reserved = (
            count_tokens_approx(user_input_for_prompt)
            + count_tokens_approx(conversation_context or "")
            + 500
        )
        retrieved_context = (
            _fit_context_blocks(context_parts, reserved) if context_parts else ""
        )

        # Choose appropriate prompt based on context
        if retrieved_context:
            # Use enhanced prompt with retrieved legal context
            prompt = GROUNDED_QUERY_PROMPT.format(
                user_query=user_input_for_prompt,
                retrieved_context=retrieved_context,
            )
        else:
            # No retrieved context — tools returned empty.
            # Use general prompt with extra caution about ungrounded claims.
            prompt = (
                GENERAL_QUERY_PROMPT.format(query=user_input_for_prompt)
                + prompt_warning
            )

        # Add conversation context if available
        if conversation_context:
            prompt = f"""Previous conversation context:
{conversation_context}

{prompt}"""

        final_response = await invoke_llm_safely(llm, prompt)

    except Exception as e:
        print(f"LLM error in general query: {e}")
        final_response = GENERAL_QUERY_ERROR

    # Compulsory RAG: if retrieval failed across all sources, prepend disclaimer
    if disclaimer_prefix:
        final_response = disclaimer_prefix + final_response
    else:
        # Check citations whenever any RAG source succeeded — not just
        # when statute context specifically was retrieved. When only
        # Indian Kanoon case-law search succeeded, retrieved_sections is
        # None (no statute chunks to compare against), so this still
        # existence-checks any "Section N"/"Article N" claim in the answer
        # against the indexed corpus without the "was it actually
        # retrieved" check that requires a real statute_result.
        retrieved_sections = (
            (statute_result.raw or {}).get("retrieved_sections")
            if not isinstance(statute_result, Exception)
            else None
        )
        final_response = await _verify_response_citations(
            final_response, retrieved_sections
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
    response = NON_LEGAL_RESPONSE

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
        self._active_stream_tasks: Dict[str, asyncio.Task] = {}

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
        # Get session history — snapshot into a new list so a concurrent
        # request for the same session_id (double-submit/retry) appending
        # via _add_message() can't mutate the list this run is still reading.
        messages = list(self._get_session_messages(session_id))

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
        # A new stream supersedes any still-running one for this session — cancel
        # the old task first so it isn't orphaned (unstoppable, still burning
        # LLM compute) by the dict overwrite below.
        previous = self._active_stream_tasks.get(session_id)
        if previous is not None and not previous.done():
            previous.cancel()
        self._active_stream_tasks[session_id] = task

        accumulated = ""
        stopped = False
        superseded = False
        try:
            # Yield tokens as they arrive
            while True:
                chunk = await queue.get()
                if chunk is None:
                    break
                tokens_streamed = True
                accumulated += chunk
                yield {"type": "token", "content": chunk}

            # Wait for handler to complete and get result
            try:
                result = await task
            except asyncio.CancelledError:
                # stop_stream() cancelled the handler mid-generation — the
                # tokens already yielded above are everything the user saw,
                # so save that partial text as the assistant turn instead of
                # dropping it (keeps conversation context coherent).
                stopped = True
                result = {"intent": intent, "response": accumulated}
        finally:
            if self._active_stream_tasks.get(session_id) is task:
                self._active_stream_tasks.pop(session_id, None)
            else:
                # A newer stream_chat() call for this session_id superseded
                # us (and cancelled us) before we finished — don't let our
                # stale partial response land in history after the newer,
                # already-completed turn.
                superseded = True

        # If no tokens were streamed (non-LLM path), yield full response
        if not tokens_streamed and result.get("response"):
            yield {"type": "token", "content": result["response"]}

        # Add assistant response (or partial, if stopped) to session history
        response_text = result.get("response", "") or accumulated
        if response_text and not superseded:
            assistant_message: Message = {
                "role": "assistant",
                "content": response_text,
            }
            self._add_message(session_id, assistant_message)

        if stopped:
            yield {
                "type": "stopped",
                "session_id": session_id,
                "intent": result.get("intent") or intent,
                "response": response_text,
            }
            return

        # Yield completion event with metadata
        yield {
            "type": "done",
            "session_id": session_id,
            "intent": result.get("intent") or intent,
            "response": response_text,
            "lawyers_found": result.get("lawyers_found"),
            "document_info": result.get("document_info"),
            "document_validation": result.get("document_validation"),
            "crime_report": result.get("crime_report"),
        }

    def stop_stream(self, session_id: str) -> bool:
        """Cancel an in-flight stream_chat() generation for this session, if
        any. Called from the /api/chat/stream/stop endpoint (the Stop button)
        rather than relying on HTTP disconnect detection, since the handler
        task is a detached asyncio task that a closed response body alone
        would not cancel."""
        task = self._active_stream_tasks.get(session_id)
        if task is not None and not task.done():
            task.cancel()
            return True
        return False

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
        # Get session history — snapshot into a new list so a concurrent
        # request for the same session_id can't mutate the list this run
        # is still reading (see stream_chat for the same fix).
        messages = list(self._get_session_messages(session_id))

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

    def has_session(self, session_id: str) -> bool:
        """Whether session_id is already live in the in-memory cache."""
        self._evict_stale_sessions()
        return session_id in self._sessions

    def seed_session(self, session_id: str, messages: List[Message]) -> None:
        """
        Prime in-memory state from DB-loaded history, but only if this
        session_id isn't already live (avoids clobbering an active
        conversation with a stale DB read). Called by the chat router for an
        authenticated user whose session_id isn't yet in this process (fresh
        restart, or a session_id that predates this process's uptime).
        """
        self._evict_stale_sessions()
        if session_id not in self._sessions:
            # Matches _add_message's cap — DB may hold the full untruncated
            # transcript, but the live LangGraph context window is unaffected.
            self._sessions[session_id] = list(messages[-20:])
            self._session_last_access[session_id] = time.monotonic()


# Singleton instance
_chatbot: Optional[LegalChatbot] = None


def get_chatbot() -> LegalChatbot:
    """Get or create the chatbot instance."""
    global _chatbot
    if _chatbot is None:
        _chatbot = LegalChatbot()
    return _chatbot
