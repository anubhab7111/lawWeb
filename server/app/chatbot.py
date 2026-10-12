"""LangGraph-based legal chatbot implementation."""

import asyncio
import contextvars
import json
import logging
import re
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from typing import (
    Annotated,
    Any,
    AsyncGenerator,
    Awaitable,
    Callable,
    Dict,
    FrozenSet,
    List,
    Literal,
    Optional,
    Tuple,
    TypedDict,
)

from app.checkpointing import get_checkpointer
from app.config import get_settings
from app.intent_classifier import (
    classify_document_subintent_embedding,
    classify_domain_hint_embedding,
    classify_intent_embedding,
)
from app.logging_config import request_id_var
from app.multilingual import postprocess_response, preprocess_query
from app.prompts import (
    CASE_LAW_CONTEXT_BLOCK,
    CLARIFY_GENERIC,
    CLARIFY_LAW_OR_LAWYER,
    CLARIFY_LAW_OR_REPORT,
    CLARIFY_PREFIX,
    CONCISE_ANSWER_SUFFIX,
    CRIME_DETAILS_ASK,
    CRIME_DETAILS_PREFIX,
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
    LAWYER_DETAILS_ASK,
    LAWYER_DETAILS_PREFIX,
    LAWYER_NONE_NEARBY_NOTE,
    LAWYER_SEARCH_FALLBACK,
    LAWYER_SEARCH_PROMPT,
    NON_LEGAL_RESPONSE,
    QUERY_REWRITE_PROMPT,
    REGENERATION_FEEDBACK_BLOCK,
    ROUTE_TIEBREAK_INTENT_DESCRIPTIONS,
    ROUTE_TIEBREAK_PROMPT,
    ROUTE_TIEBREAK_UNSURE,
    SERIOUS_CRIME_HELPLINES,
    SERIOUS_CRIMES,
    STATUTE_CONTEXT_BLOCK,
    sanitize_untrusted_document,
)
from app.routing_keywords import CRIME_TYPE_KEYWORDS
from app.state import ChatState, DocumentValidationInfo, LawyerInfo, Message
from app.text_match import contains_word, count_words
from app.tool_dispatch import RAG_TOOL_REGISTRY, ToolInvocationResult, select_tools
from app.tools.crime_reporter import classify_crime_type
from app.tools.document_classifier import classify_document, get_document_classifier
from app.tools.followup import (
    crime_report_too_thin,
    find_location,
    missing_lawyer_details,
)
from app.tools.indian_law_rag import get_indian_law_rag
from app.tools.lawyer_recommender import format_lawyer_results
from app.tools.lawyer_recommender import recommend_lawyers as recommend_lawyers_core
from app.tools.legal_defect_analyzer import get_legal_defect_analyzer
from app.tools.statutory_validator import format_score, get_statutory_validator
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, StateGraph

logger = logging.getLogger(__name__)

LLM_NUM_CTX = 8192


LLM_NUM_PREDICT = 4608
_PROMPT_SAFETY_MARGIN = 256
_MAX_QUERY_CHARS = 8000


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
        timeout=210.0,
        reasoning=False,
        keep_alive="1h",
    )


def _truncate_block(block: str, max_chars: int) -> str:
    """Cut a context block to max_chars at a provision boundary (" • " entries) or, failing that, the last line/sentence end."""
    if len(block) <= max_chars:
        return block
    head = block[:max_chars]
    cut = head.rfind("\n\n• ")
    if cut > 0:
        return head[:cut]
    for sep in ("\n", ". "):
        cut = head.rfind(sep)
        if cut > max_chars // 2:
            return head[: cut + 1].rstrip()
    return head.rsplit(" ", 1)[0]


@lru_cache()
def get_retry_llm() -> ChatOllama:
    """get_llm() with a higher temperature, used for the single retry after a give-up."""
    settings = get_settings()
    return ChatOllama(
        model=settings.llm_model,
        temperature=settings.llm_retry_temperature,
        base_url=settings.ollama_base_url,
        num_ctx=LLM_NUM_CTX,
        num_predict=LLM_NUM_PREDICT,
        timeout=210.0,
        reasoning=False,
        keep_alive="1h",
    )


def _fit_context_blocks(
    context_parts: list, reserved_tokens: int, max_tokens: Optional[int] = None
) -> str:
    """Join retrieved-context blocks (already in priority order: statute → case law) without exceeding the model's input budget."""
    from app.metrics.engineering_metrics import count_tokens_approx

    budget = LLM_NUM_CTX - LLM_NUM_PREDICT - _PROMPT_SAFETY_MARGIN - reserved_tokens
    if max_tokens is not None:
        budget = min(budget, max_tokens)
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
            if remaining > 50:
                kept.append(_truncate_block(block, remaining * 4))
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
        num_ctx=LLM_NUM_CTX,
        num_predict=128,
        timeout=15.0,
        reasoning=False,
    )


@lru_cache()
def get_fast_llm_prose() -> ChatOllama:
    """Same small model as get_fast_llm(), but for short natural-language generation (case summaries, plain-language explanations) rather than JSON classification."""
    settings = get_settings()
    return ChatOllama(
        model=settings.fast_llm_model,
        temperature=0,
        base_url=settings.ollama_base_url,
        num_ctx=LLM_NUM_CTX,
        num_predict=900,
        timeout=45.0,
        reasoning=False,
    )


@lru_cache()
def get_grounding_correction_llm() -> ChatOllama:
    """Dedicated LLM for grounding_verifier's claim-correction call — same small model as get_fast_llm_prose(), but grammar-constrained to CORRECTION_RESPONSE_SCHEMA via Ollama's..."""
    from app.tools.grounding_verifier import CORRECTION_RESPONSE_SCHEMA

    settings = get_settings()
    return ChatOllama(
        model=settings.fast_llm_model,
        temperature=0,
        base_url=settings.ollama_base_url,
        num_ctx=LLM_NUM_CTX,
        num_predict=1200,
        timeout=60.0,
        reasoning=False,
        format=CORRECTION_RESPONSE_SCHEMA,
    )


_THINK_TAG_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def strip_reasoning_tags(text: str) -> str:
    """Strip a qwen3 thinking preamble from a response."""
    if "</think>" in text:
        return text.rsplit("</think>", 1)[-1].strip()
    return _THINK_TAG_RE.sub("", text).strip()


_stream_queue_var: contextvars.ContextVar[asyncio.Queue | None] = (
    contextvars.ContextVar("stream_queue", default=None)
)


_LLM_TIMEOUT_SECONDS = 180


_THINKING_BUFFER_SAFETY_CAP = 20000


_INCOMPLETE_GENERATION_NOTE = (
    "I wasn't able to finish formulating a complete answer to that in time. "
    "Please try again, or rephrase the question — this can happen with more "
    "complex or multi-part questions."
)


class LLMUnavailableError(RuntimeError):
    """Raised without calling Ollama while the circuit breaker is open."""


class ChatBusyError(RuntimeError):
    """Raised when settings.chat_max_concurrent chats are already in flight."""


class _LLMCircuitBreaker:
    """Fail fast when Ollama is down or wedged."""

    def __init__(self) -> None:
        self.failures = 0
        self.open_until = 0.0
        self._half_open = False

    def check(self) -> None:
        now = time.monotonic()
        if now < self.open_until:
            raise LLMUnavailableError(
                f"LLM circuit open for another {self.open_until - now:.0f}s"
            )
        if self.open_until:
            self.open_until = 0.0
            self._half_open = True

    def record_success(self) -> None:
        self.failures = 0
        self._half_open = False

    def record_failure(self) -> None:
        settings = get_settings()
        self.failures += 1
        if self._half_open or self.failures >= settings.llm_breaker_failures:
            self.open_until = time.monotonic() + settings.llm_breaker_cooldown_seconds
            self.failures = 0
            self._half_open = False
            logger.error(
                "LLM circuit opened for %ss after repeated failures",
                settings.llm_breaker_cooldown_seconds,
            )


_llm_breaker = _LLMCircuitBreaker()


async def emit_event(event_type: str, **payload: Any) -> None:
    """Push a non-token event (status, reset, replace, ...) to the live SSE stream."""
    queue = _stream_queue_var.get(None)
    if queue is not None:
        await queue.put({"type": event_type, **payload})


async def emit_text(text: str) -> None:
    """Stream literal text (e.g."""
    queue = _stream_queue_var.get(None)
    if queue is not None and text:
        await queue.put(text)


async def invoke_llm_safely(
    llm: ChatOllama, prompt: str, stream: bool = False, notify_incomplete: bool = True
) -> str:
    """Safely invoke LLM with proper error handling."""
    _llm_breaker.check()
    queue = _stream_queue_var.get(None) if stream else None

    if queue is not None:

        visible_response = ""

        async def _drain() -> str:
            nonlocal visible_response
            in_thinking = get_settings().llm_thinking
            buffer = ""
            async for chunk in llm.astream([HumanMessage(content=prompt)]):
                token = chunk.content if hasattr(chunk, "content") else str(chunk)
                if not token:
                    continue
                if not in_thinking:
                    visible_response += token
                    await queue.put(token)
                    continue
                buffer += token
                if "</think>" in buffer:
                    after = buffer.split("</think>", 1)[1]
                    in_thinking = False
                    buffer = ""
                    if after:
                        visible_response += after
                        await queue.put(after)
                elif len(buffer) > _THINKING_BUFFER_SAFETY_CAP:

                    logger.warning(
                        "thinking exceeded %s chars without closing — giving up",
                        _THINKING_BUFFER_SAFETY_CAP,
                    )
                    visible_response = _INCOMPLETE_GENERATION_NOTE
                    if notify_incomplete:
                        await queue.put(_INCOMPLETE_GENERATION_NOTE)
                    return visible_response
            if in_thinking and buffer:

                logger.warning(
                    "generation ended mid-thinking (%s buffered chars, no closing tag)",
                    len(buffer),
                )
                visible_response = _INCOMPLETE_GENERATION_NOTE
                if notify_incomplete:
                    await queue.put(_INCOMPLETE_GENERATION_NOTE)
            return visible_response

        try:
            result = await asyncio.wait_for(_drain(), timeout=_LLM_TIMEOUT_SECONDS)
            _llm_breaker.record_success()
            return result
        except asyncio.TimeoutError:

            _llm_breaker.record_failure()
            note = "\n\n*(Response generation took too long and was cut short.)*"
            visible_response += note
            await queue.put(note)
            logger.warning(
                "generation exceeded %ss, returning partial output",
                _LLM_TIMEOUT_SECONDS,
            )
            return visible_response
        except Exception:
            _llm_breaker.record_failure()
            logger.exception("LLM streaming error")
            raise
    else:

        try:
            response = await asyncio.wait_for(
                llm.ainvoke([HumanMessage(content=prompt)]),
                timeout=_LLM_TIMEOUT_SECONDS,
            )
            _llm_breaker.record_success()
            raw = response.content

            if getattr(llm, "format", None):
                return strip_reasoning_tags(raw)
            if "</think>" not in raw and not get_settings().llm_thinking:
                return raw.strip()
            if "</think>" not in raw:

                logger.warning(
                    "non-streaming generation ended without </think> (%s chars) — giving up",
                    len(raw),
                )
                return _INCOMPLETE_GENERATION_NOTE
            return strip_reasoning_tags(raw)
        except asyncio.TimeoutError:
            _llm_breaker.record_failure()
            logger.warning("generation exceeded %ss", _LLM_TIMEOUT_SECONDS)
            raise
        except Exception:
            _llm_breaker.record_failure()
            logger.exception("LLM invocation error")
            raise


def _count_keyword_matches(text: str, keywords: frozenset) -> int:
    """Count how many keywords match in the text."""
    return count_words(text, keywords)


def _extract_legal_entities(text: str) -> List[str]:
    """Extract legal terms, acts, and sections from text."""
    entities = []
    text_lower = text.lower()

    section_patterns = [
        r"section\s+(\d+[a-z]?)",
        r"ipc\s+(\d+[a-z]?)",
        r"crpc\s+(\d+[a-z]?)",
    ]
    for pattern in section_patterns:
        matches = re.findall(pattern, text_lower)
        for match in matches:
            entities.append(f"Section {match}")

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
        if contains_word(text_lower, act):
            entities.append(act.title())

    return list(set(entities))


async def _invoke_fast_text(prompt: str, timeout: float) -> str:
    """Short auxiliary generation with the thinking preamble stripped."""
    return await asyncio.wait_for(
        invoke_llm_safely(get_fast_llm_prose(), prompt, stream=False), timeout=timeout
    )


async def _rewrite_query_for_retrieval(
    messages: List[Message], current_input: str
) -> str:
    """Condense conversation history + the latest message into one standalone retrieval query using the fast LLM."""

    prior = messages
    if prior and prior[-1]["role"] == "user" and prior[-1]["content"] == current_input:
        prior = prior[:-1]
    if not any(m["role"] == "assistant" for m in prior):
        return current_input

    history_lines = [f"{m['role'].upper()}: {m['content'][:300]}" for m in prior[-4:]]
    prompt = QUERY_REWRITE_PROMPT.format(
        history="\n".join(history_lines), question=current_input
    )

    try:
        rewritten = (
            (await _invoke_fast_text(prompt, timeout=25.0)).strip().strip('"').strip()
        )
        if (
            rewritten == _INCOMPLETE_GENERATION_NOTE
            or not rewritten
            or len(rewritten) > 300
            or "\n" in rewritten
        ):
            return current_input
        if rewritten.lower() != current_input.lower():
            logger.info(f"Retrieval query rewritten: {rewritten[:120]}")
        return rewritten
    except Exception as e:
        logger.warning(f"[Router] Query rewrite failed ({e}) — using raw input.")
        return current_input


def _apply_compulsory_rag_policy(rag_succeeded: bool) -> tuple:
    """Single shared implementation of the "grounding unavailable" pattern, used identically across handle_document_analysis, handle_crime_report, and handle_general_query."""
    if rag_succeeded:
        return "", ""
    return GROUNDING_UNAVAILABLE_DISCLAIMER, GROUNDING_UNAVAILABLE_PROMPT_WARNING


_UPLOAD_HELP_KEYWORDS = ("upload", "i will upload", "how to upload", "can i upload")
_ACTION_INTENTS = frozenset({"find_lawyer", "crime_report"})
_CLARIFY_MAX_WORDS = 25


def _asks_about_uploading(text: str) -> bool:
    lowered = text.lower()
    return any(kw in lowered for kw in _UPLOAD_HELP_KEYWORDS)


def _merge_trace(state: ChatState, **entries: Any) -> Dict[str, Any]:
    return {**(state.get("trace") or {}), **entries}


def _looks_like_an_explicit_question(text: str) -> bool:
    """A '?' means the user already wrote out a specific legal question, not a raw situational statement — see _clarification_for."""
    return "?" in text


def _ambiguous_action_contenders(result) -> FrozenSet[str]:
    """The near-tied intents worth escalating over — only when an *action* intent (report a crime / find a lawyer) is among them; two explanatory intents tying is not worth..."""
    contenders = {result.primary_intent, *result.secondary_intents} - {
        "non_legal",
        "document_analysis",
    }
    if len(contenders) < 2 or not (contenders & _ACTION_INTENTS):
        return frozenset()
    return frozenset(contenders)


def _clarification_gate(
    result, has_document: bool, messages: List[Message], user_input: str
) -> FrozenSet[str]:
    """Non-empty iff this turn is genuinely torn between an *action* flow (report a crime / find a lawyer) and another one — worth escalating, first to the LLM tie-breaker..."""
    if not get_settings().clarify_on_ambiguous or not result.is_ambiguous:
        return frozenset()
    if has_document or len(user_input.split()) > _CLARIFY_MAX_WORDS:
        return frozenset()
    if _looks_like_an_explicit_question(user_input):
        return frozenset()
    if (
        messages
        and messages[-1]["role"] == "assistant"
        and messages[-1]["content"].startswith(CLARIFY_PREFIX)
    ):
        return frozenset()
    return _ambiguous_action_contenders(result)


def _resume_after_followup(messages: List[Message], user_input: str) -> Optional[tuple]:
    """(intent, original request + this reply) when the last assistant turn was a missing-detail question; None otherwise."""
    if (
        len(messages) < 2
        or messages[-1]["role"] != "assistant"
        or messages[-2]["role"] != "user"
    ):
        return None
    asked = messages[-1]["content"]
    for prefix, intent in (
        (LAWYER_DETAILS_PREFIX, "find_lawyer"),
        (CRIME_DETAILS_PREFIX, "crime_report"),
    ):
        if asked.startswith(prefix):
            return intent, f"{messages[-2]['content'].strip()} {user_input.strip()}"
    return None


def _clarify_question(contenders: FrozenSet[str]) -> str:
    if contenders == {"general_query", "find_lawyer"}:
        return CLARIFY_LAW_OR_LAWYER
    if contenders == {"general_query", "crime_report"}:
        return CLARIFY_LAW_OR_REPORT
    return CLARIFY_GENERIC


def _tiebreak_options(candidates: Tuple[str, ...]) -> str:
    return "\n".join(
        f"- {c}: the user {ROUTE_TIEBREAK_INTENT_DESCRIPTIONS[c]}" for c in candidates
    )


@lru_cache()
def _get_tiebreak_llm(candidates: Tuple[str, ...]) -> ChatOllama:
    """Schema-constrained tie-breaker for a near-tied route (see _resolve_ambiguity_with_llm) — same pattern as get_grounding_correction_llm(): format= is what actually gets qwen3:4b..."""
    settings = get_settings()
    schema = {
        "type": "object",
        "properties": {
            "intent": {
                "type": "string",
                "enum": list(candidates) + [ROUTE_TIEBREAK_UNSURE],
            }
        },
        "required": ["intent"],
    }
    return ChatOllama(
        model=settings.fast_llm_model,
        temperature=0,
        base_url=settings.ollama_base_url,
        num_ctx=LLM_NUM_CTX,
        num_predict=64,
        timeout=15.0,
        reasoning=False,
        format=schema,
    )


async def _resolve_ambiguity_with_llm(
    contenders: FrozenSet[str], user_input: str
) -> Tuple[Optional[str], Dict[str, Any]]:
    """Cascade routing, tier 2: one fast, schema-constrained call to resolve a near-tied route before asking the user."""
    settings = get_settings()
    if not settings.route_tiebreak_enabled:
        return None, {"tried": False}
    key = tuple(sorted(contenders))
    prompt = ROUTE_TIEBREAK_PROMPT.format(
        options=_tiebreak_options(key),
        message=user_input[:500],
        unsure=ROUTE_TIEBREAK_UNSURE,
    )
    t0 = time.monotonic()
    picked = None
    try:
        raw = await asyncio.wait_for(
            invoke_llm_safely(_get_tiebreak_llm(key), prompt, stream=False),
            timeout=settings.route_tiebreak_timeout_seconds,
        )
        picked = json.loads(raw).get("intent")
    except Exception as e:
        logger.warning("Route tie-break failed (%s) — falling back to clarification", e)
    elapsed = round(time.monotonic() - t0, 2)
    resolved = picked if picked in key else None
    return resolved, {
        "tried": True,
        "seconds": elapsed,
        "candidates": list(key),
        "raw": picked,
        "resolved": resolved,
    }


async def classify_intent(state: ChatState) -> ChatState:
    """Intent classification and tool selection."""
    user_input = state["current_input"]
    has_document = bool(state.get("document_content"))
    messages = state.get("messages", [])

    logger.info("Router input=%.100r has_document=%s", user_input, has_document)

    resumed = _resume_after_followup(messages, user_input)
    if resumed:
        intent, merged = resumed
        logger.info("Router: reply to a %s follow-up — resuming that flow", intent)
        await emit_event("routing", intent=intent)
        return {
            **state,
            "intent": intent,
            "followup_answered": True,
            "retrieval_query": merged,
            "lawyer_query": merged if intent == "find_lawyer" else None,
            "crime_details": merged if intent == "crime_report" else None,
            "selected_tools": select_tools(intent, merged),
            "domain_hint": await classify_domain_hint_embedding(merged),
            "trace": _merge_trace(
                state, routing={"intent": intent, "followup_answered": True}
            ),
        }

    result = await classify_intent_embedding(user_input, has_document)

    intent = result.primary_intent
    if result.is_ambiguous and intent != "non_legal":

        intent = "document_analysis" if has_document else "general_query"

    if intent == "document_analysis" and not has_document:
        if not _asks_about_uploading(user_input):
            intent = "general_query"

    routing_trace = {
        "intent": intent,
        "top_intent": result.primary_intent,
        "confidence": round(result.confidence, 3),
        "margin": round(result.margin, 3),
        "ambiguous": result.is_ambiguous,
        "secondary": list(result.secondary_intents),
    }
    base = {
        **state,
        "routing_confidence": result.confidence,
        "routing_reasoning": result.reasoning,
        "is_ambiguous": result.is_ambiguous,
        "active_document_context": has_document,
    }

    if intent == "non_legal":
        logger.info(
            "Router: non-legal (confidence=%.3f, margin=%.3f) — skipping retrieval",
            result.confidence,
            result.margin,
        )
        await emit_event("routing", intent="non_legal")
        return {
            **base,
            "intent": "non_legal",
            "selected_tools": [],
            "domain_hint": None,
            "trace": _merge_trace(state, routing=routing_trace),
        }

    contenders = _clarification_gate(result, has_document, messages, user_input)
    if contenders:

        try:
            resolved, tiebreak_trace = await _resolve_ambiguity_with_llm(
                contenders, user_input
            )
        except Exception as e:
            logger.exception("Route tie-break raised unexpectedly (%s)", e)
            resolved, tiebreak_trace = None, {"tried": True, "error": str(e)}
        routing_trace["tiebreak"] = tiebreak_trace
        if resolved:
            logger.info(
                "Router: tie-break resolved %s -> %s (%.1fs)",
                sorted(contenders),
                resolved,
                tiebreak_trace.get("seconds", 0.0),
            )
            intent = resolved
            routing_trace["intent"] = intent

        else:
            logger.info(
                "Router: ambiguous %s — asking a clarifying question",
                sorted(contenders),
            )
            await emit_event("routing", intent="clarify")
            return {
                **base,
                "intent": "clarify",
                "response": _clarify_question(contenders),
                "clarification": True,
                "selected_tools": [],
                "domain_hint": None,
                "trace": _merge_trace(
                    state, routing={**routing_trace, "intent": "clarify"}
                ),
            }

    domain_hint = await classify_domain_hint_embedding(user_input)
    entities = _extract_legal_entities(user_input)
    logger.info(
        "Router decision: intent=%s confidence=%.3f margin=%.3f ambiguous=%s "
        "domain_hint=%s secondary=%s",
        intent,
        result.confidence,
        result.margin,
        result.is_ambiguous,
        domain_hint,
        result.secondary_intents,
    )

    retrieval_query = await _rewrite_query_for_retrieval(messages, user_input)
    selected_tools = select_tools(intent, retrieval_query)

    await emit_event("routing", intent=intent)
    return {
        **base,
        "retrieval_query": retrieval_query,
        "intent": intent,
        "secondary_intents": result.secondary_intents,
        "extracted_entities": entities,
        "selected_tools": selected_tools,
        "domain_hint": domain_hint,
        "trace": _merge_trace(
            state,
            routing={
                **routing_trace,
                "domain_hint": domain_hint,
                "tools": selected_tools,
                "retrieval_query": retrieval_query,
            },
        ),
    }


async def handle_document_analysis(state: ChatState) -> ChatState:
    """Handle document analysis and validation requests."""
    document_content = state.get("document_content", "")
    document_type = state.get("document_type", "unknown")
    user_query = state.get("current_input", "")

    if not document_content:
        response = DOCUMENT_UPLOAD_HELP
        return {
            **state,
            "response": response,
            "messages": state["messages"]
            + [{"role": "assistant", "content": response}],
        }

    subintent = await classify_document_subintent_embedding(user_query)
    if subintent == "validation":
        return await _handle_document_validation(state)

    try:
        from app.tools.criminal_rag import get_criminal_rag_system

        crime_rag = get_criminal_rag_system()
        await crime_rag.initialize()
    except Exception:
        crime_rag = None

    try:
        from app.tools.document_analysis_pipeline import get_document_analysis_pipeline

        llm = get_llm()

        pipeline = get_document_analysis_pipeline(llm, crime_rag=crime_rag)
        result = await pipeline.analyze_document(
            document_text=document_content,
            document_type=document_type,
            user_query=user_query,
        )

        response_parts = [result.summary]

        if result.key_points:
            response_parts.append("\n\n**Key Points:**")
            for i, point in enumerate(result.key_points, 1):
                response_parts.append(f"{i}. {point}")

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

        rag_succeeded = bool(
            result.crime_context and result.crime_context.get("relevant_passages")
        )

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
                "confidence": result.confidence,
            },
            "messages": state["messages"]
            + [{"role": "assistant", "content": response}],
        }
    except Exception as e:

        error_msg = f"Enhanced analysis unavailable: {str(e)}"
        logger.warning(error_msg)

        llm = get_llm()
        max_chars = 15000
        doc_text = document_content[:max_chars]
        if len(document_content) > max_chars:
            doc_text += (
                "\n\n[Document truncated for analysis. Full document is longer.]"
            )

        prompt = DOCUMENT_ANALYSIS_PROMPT.format(
            document_text=sanitize_untrusted_document(doc_text)
        )

        await emit_text(DOC_RAG_UNAVAILABLE_DISCLAIMER)
        analysis = await invoke_llm_safely(llm, prompt, stream=True)
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


def _ask_followup(state: ChatState, question: str, flow: str) -> ChatState:
    """Answer this turn with a missing-detail question; the next turn resumes `flow`."""
    return {
        **state,
        "response": question,
        "messages": state["messages"] + [{"role": "assistant", "content": question}],
        "trace": _merge_trace(state, followup={"asked": True, "flow": flow}),
    }


async def handle_crime_report(state: ChatState) -> ChatState:
    """Handle crime reporting and guidance requests."""
    user_input = state["current_input"]

    crime_details = (
        state.get("crime_details") or state.get("retrieval_query") or user_input
    )
    identified_crime = await classify_crime_type(crime_details)
    serious = identified_crime in SERIOUS_CRIMES
    if (
        not serious
        and not state.get("followup_answered")
        and crime_report_too_thin(crime_details)
    ):
        return _ask_followup(state, CRIME_DETAILS_ASK, "crime_report")

    ik_result = await RAG_TOOL_REGISTRY["crime_sections"](
        crime_details, crime_type=identified_crime, k=2
    )
    rag_sections_text = ik_result.context_text
    rag_succeeded = ik_result.succeeded

    llm = get_llm()

    rag_section = ""
    if rag_sections_text:
        rag_section = f"""\n\nAPPLICABLE PROVISIONS (cite the Act named with each section):
{rag_sections_text}"""

    disclaimer_prefix, no_rag_warning = _apply_compulsory_rag_policy(rag_succeeded)

    prompt = CRIME_REPORT_PROMPT.format(
        crime_details=crime_details[:_MAX_QUERY_CHARS],
        identified_crime=identified_crime,
        rag_section=rag_section,
        no_rag_warning=no_rag_warning,
    )

    await emit_text(disclaimer_prefix)
    fell_back = False
    try:
        final_response = await invoke_llm_safely(llm, prompt, stream=True)
    except Exception as e:
        logger.error("LLM error in crime report: %s", e)
        fell_back = True
        final_response = CRIME_REPORT_FALLBACK.format(
            crime_name=identified_crime.replace("_", " ").title()
        )

    if disclaimer_prefix:
        final_response = disclaimer_prefix + final_response
    if serious:
        final_response = SERIOUS_CRIME_HELPLINES[identified_crime] + final_response
    if serious or fell_back:
        await emit_event("replace", content=final_response)

    return {
        **state,
        "response": final_response,
        "crime_details": crime_details,
        "crime_report": {
            "crime_type": identified_crime,
        },
        "messages": state["messages"]
        + [{"role": "assistant", "content": final_response}],
        "trace": _merge_trace(
            state,
            crime_report={
                "crime_type": identified_crime,
                "rag_succeeded": rag_succeeded,
            },
        ),
    }


async def handle_find_lawyer(state: ChatState) -> ChatState:
    """Handle lawyer search requests."""
    user_input = state["current_input"]
    lawyer_query = (state.get("lawyer_query") or user_input)[:_MAX_QUERY_CHARS]
    tools = state.get("selected_tools") or []
    if not state.get("followup_answered"):
        missing = missing_lawyer_details(state.get("retrieval_query") or lawyer_query)
        if missing:
            return _ask_followup(
                state,
                LAWYER_DETAILS_PREFIX + LAWYER_DETAILS_ASK[tuple(missing)],
                "find_lawyer",
            )
    location = find_location(state.get("retrieval_query") or lawyer_query)

    from app.db.engine import get_engine
    from sqlmodel import Session as DBSession

    location_note = ""
    with DBSession(get_engine()) as session:
        lawyers = (
            await recommend_lawyers_core(
                session, problem_description=lawyer_query, location=location, limit=5
            )
            if location
            else []
        )
        if not lawyers:
            if location:
                location_note = LAWYER_NONE_NEARBY_NOTE.format(
                    location=location.title()
                )
            lawyers = await recommend_lawyers_core(
                session, problem_description=lawyer_query, limit=5
            )
        formatted_results = format_lawyer_results(lawyers)
        lawyers_info: List[LawyerInfo] = [
            {
                "id": l.id,
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

    try:
        llm = get_llm()
        prompt = LAWYER_SEARCH_PROMPT.format(
            query=lawyer_query, lawyer_results=formatted_results
        )
        final_response = await invoke_llm_safely(llm, prompt, stream=True)
    except Exception:

        final_response = LAWYER_SEARCH_FALLBACK.format(
            formatted_results=formatted_results
        )
    if location_note:
        final_response = location_note + final_response
        await emit_event("replace", content=final_response)

    return {
        **state,
        "response": final_response,
        "lawyer_query": lawyer_query,
        "lawyers_found": lawyers_info,
        "messages": state["messages"]
        + [{"role": "assistant", "content": final_response}],
        "trace": _merge_trace(
            state, lawyer_search={"candidates": len(lawyers_info), "tools": tools}
        ),
    }


async def _verify_response_citations(
    response_text: str,
    retrieved_sections=None,
    retrieved_context_text: str = "",
    llm_invoke: Optional[Callable[[str], Awaitable[str]]] = None,
    citation_only: bool = False,
) -> tuple:
    """Post-generation grounding gate."""
    try:
        from app.tools.case_citation_verifier import (
            case_verification_footer,
            verify_case_citations,
        )
        from app.tools.case_law_rag import get_case_law_rag_system
        from app.tools.citation_verifier import verification_footer, verify_citations
        from app.tools.grounding_verifier import ground_and_correct, grounding_footer
        from app.tools.unified_legal_rag import get_unified_rag_system

        rag = get_unified_rag_system()
        case_rag = get_case_law_rag_system()

        if citation_only:
            text = response_text
            if rag.initialized:
                citation_report = verify_citations(
                    response_text, rag, retrieved_sections
                )
                if citation_report.checks:
                    logger.info(
                        "CitationVerify (no-retrieval path): %s/%s citations verified",
                        len(citation_report.verified),
                        len(citation_report.checks),
                    )
                text += verification_footer(citation_report)
            if case_rag.initialized:
                text += case_verification_footer(
                    verify_case_citations(response_text, case_rag)
                )
            return text, None

        if not rag.initialized:
            return response_text, None

        corrected_text, report = await ground_and_correct(
            response_text,
            rag,
            retrieved_sections=retrieved_sections,
            retrieved_context_text=retrieved_context_text,
            llm_invoke=llm_invoke,
        )
        if report.citation_report.checks:
            logger.info(
                "CitationVerify: %s/%s citations verified",
                len(report.citation_report.verified),
                len(report.citation_report.checks),
            )
        if report.claim_sentences:
            corrected_count = sum(
                1 for s in report.sentences if s.outcome == "corrected"
            )
            logger.info(
                "GroundingGate: confidence=%.2f flagged=%s/%s corrected=%s",
                report.overall_score,
                len(report.flagged),
                len(report.claim_sentences),
                corrected_count,
            )
        text = (
            corrected_text
            + verification_footer(report.citation_report)
            + grounding_footer(report)
        )
        if case_rag.initialized:
            text += case_verification_footer(
                verify_case_citations(corrected_text, case_rag)
            )
        return (
            text,
            report,
        )
    except Exception:
        logger.exception("Citation verification skipped")
        return response_text, None


_QUESTION_SPLIT_RE = re.compile(r"(?<=\?)\s+")
_ENUM_SPLIT_RE = re.compile(r"(?:^|\s)\(?(?:[a-d]|[1-4])[\).]\s+(?=[A-Za-z])")
_MIN_SUBQUESTION_WORDS = 5
_MAX_SUBQUESTIONS = 3
_ENTRY_SPLIT_RE = re.compile(r"\n\n(?=• )")


def decompose_question(text: str) -> List[str]:
    """Split a multi-part question into standalone sub-questions so retrieval can search for each part instead of one blended query."""
    text = text.strip()

    def _qualifying(parts: List[str]) -> List[str]:
        return [p.strip() for p in parts if len(p.split()) >= _MIN_SUBQUESTION_WORDS]

    questions = _qualifying(
        [p for p in _QUESTION_SPLIT_RE.split(text) if p.strip().endswith("?")]
    )
    if len(questions) >= 2:
        return questions[:_MAX_SUBQUESTIONS]
    enumerated = _qualifying(_ENUM_SPLIT_RE.split(text)[1:])
    if len(enumerated) >= 2:
        return enumerated[:_MAX_SUBQUESTIONS]
    return []


def _merge_entries(texts: List[str]) -> str:
    """Join formatted context blocks, dropping entries (keyed by their first line, e.g."""
    seen: set = set()
    out: List[str] = []
    for text in texts:
        for entry in _ENTRY_SPLIT_RE.split(text or ""):
            entry = entry.strip()
            if not entry:
                continue
            key = entry.split("\n", 1)[0]
            if key not in seen:
                seen.add(key)
                out.append(entry)
    return "\n\n".join(out)


def _merge_statute_results(results: list) -> ToolInvocationResult:
    """Fold several statute_context results (one per query) into one, in priority order."""
    ok = [r for r in results if isinstance(r, ToolInvocationResult) and r.succeeded]
    if not ok:
        return ToolInvocationResult(
            name="statute_context",
            succeeded=False,
            context_text="",
            raw={"case_law_text": "", "confidence": 0.0, "chunk_count": 0},
        )
    if len(ok) == 1:
        return ok[0]
    sections: set = set()
    for r in ok:
        sections |= set((r.raw or {}).get("retrieved_sections") or ())
    text = _merge_entries([r.context_text for r in ok])
    return ToolInvocationResult(
        name="statute_context",
        succeeded=True,
        context_text=text,
        raw={
            "case_law_text": _merge_entries(
                [(r.raw or {}).get("case_law_text", "") for r in ok]
            ),
            "retrieved_sections": sections,
            "confidence": max((r.raw or {}).get("confidence", 0.0) for r in ok),
            "chunk_count": max(
                len(_ENTRY_SPLIT_RE.split(text)) if text else 0,
                *((r.raw or {}).get("chunk_count", 0) for r in ok),
            ),
        },
    )


def _time_elapsed(state: ChatState) -> float:
    started = state.get("started_at")
    return time.monotonic() - started if started else 0.0


def _tool_result(state: ChatState, name: str) -> Optional[ToolInvocationResult]:
    result = (state.get("tool_results") or {}).get(name)
    return result if isinstance(result, ToolInvocationResult) else None


async def gq_plan(state: ChatState) -> ChatState:
    """Split multi-part questions into sub-questions and reset loop counters."""
    sub_questions = decompose_question(state["current_input"])
    if sub_questions:
        logger.info("Plan: %s sub-questions", len(sub_questions))
    return {
        **state,
        "sub_questions": sub_questions,
        "retrieval_attempts": 0,
        "regen_count": 0,
        "regen_pending": False,
        "extra_queries": [],
        "trace": _merge_trace(state, plan={"sub_questions": sub_questions}),
    }


async def gq_retrieve(state: ChatState) -> ChatState:
    """Run the tools chosen by the router (state["selected_tools"]) in parallel."""
    user_input = state["current_input"]
    retrieval_query = state.get("retrieval_query") or user_input
    tools = state.get("selected_tools") or []
    prior = dict(state.get("tool_results") or {})
    regenerating = bool(state.get("regen_feedback"))
    widen = regenerating or (state.get("retrieval_attempts") or 0) > 0
    domain_hint = state.get("domain_hint")

    await emit_event(
        "status", stage="retrieval", label="Searching statutes and case law…"
    )

    is_multi_offense = _count_keyword_matches(user_input, CRIME_TYPE_KEYWORDS) >= 2

    jobs: Dict[str, Any] = {}
    statute_queries: List[str] = []
    if "statute_context" in tools:
        extra = list(state.get("extra_queries") or [])
        if regenerating and extra:
            statute_queries = extra
        else:
            statute_queries = [retrieval_query] + [
                q for q in (state.get("sub_questions") or []) if q != retrieval_query
            ]
        primary_k = 12 if widen else (10 if is_multi_offense else 8)
        for i, query in enumerate(statute_queries):
            is_primary = i == 0 and not regenerating
            jobs[f"statute_context:{i}"] = RAG_TOOL_REGISTRY["statute_context"](
                query,
                k=primary_k if is_primary else 4,
                domain_hint=(
                    ["criminal"] if domain_hint == "criminal" and not widen else None
                ),
                doctrine_assist=is_primary and not widen,
                with_case_law=i == 0,
            )
    names = list(jobs)
    results = await asyncio.gather(*jobs.values(), return_exceptions=True)
    by_name = dict(zip(names, results))

    statute_results = [r for n, r in by_name.items() if n.startswith("statute_context")]
    for name, r in by_name.items():
        if isinstance(r, Exception):
            logger.warning("Tool %s failed: %s", name, r)
    if statute_results:
        merged = _merge_statute_results(
            ([_tool_result_from(prior, "statute_context")] if regenerating else [])
            + statute_results
        )
        prior["statute_context"] = merged
    rag_succeeded = any(
        isinstance(r, ToolInvocationResult) and r.succeeded for r in prior.values()
    )
    return {
        **state,
        "tool_results": prior,
        "rag_succeeded": rag_succeeded,
        "trace": _merge_trace(
            state,
            retrieval={
                **(state.get("trace") or {}).get("retrieval", {}),
                "queries": statute_queries,
                "tools": tools,
            },
        ),
    }


def _tool_result_from(results: Dict[str, Any], name: str) -> Any:
    r = results.get(name)
    return r if isinstance(r, ToolInvocationResult) else None


async def gq_grade(state: ChatState) -> ChatState:
    """Turn retrieval quality into a decision."""
    settings = get_settings()
    statute = _tool_result(state, "statute_context")
    confidence = 0.0
    chunk_count = 0
    if statute is None:

        grade = "good" if state.get("rag_succeeded") else "none"
    elif not statute.succeeded:
        grade = "none"
    else:
        raw = statute.raw or {}
        confidence = float(raw.get("confidence", 0.0))
        chunk_count = int(raw.get("chunk_count", 0))
        weak = chunk_count < 3 or confidence < settings.retrieval_min_confidence
        grade = "weak" if weak else "good"

    logger.info(
        "Retrieval grade=%s confidence=%.3f provisions=%s attempts=%s",
        grade,
        confidence,
        chunk_count,
        state.get("retrieval_attempts") or 0,
    )
    sections = (
        sorted((statute.raw or {}).get("retrieved_sections") or ()) if statute else []
    )
    return {
        **state,
        "retrieval_grade": grade,
        "retrieval_confidence": confidence,
        "retrieved_sections": (
            (statute.raw or {}).get("retrieved_sections") if statute else None
        ),
        "trace": _merge_trace(
            state,
            retrieval={
                **(state.get("trace") or {}).get("retrieval", {}),
                "grade": grade,
                "confidence": round(confidence, 3),
                "provisions": chunk_count,
                "attempts": (state.get("retrieval_attempts") or 0) + 1,
                "sections": sections[:20],
            },
        ),
    }


def _route_after_grade(state: ChatState) -> Literal["gq_rewrite", "gq_generate"]:
    settings = get_settings()
    can_retry = (state.get("retrieval_attempts") or 0) < 1 and _time_elapsed(
        state
    ) < settings.request_budget_seconds / 3
    if state.get("retrieval_grade") in ("weak", "none") and can_retry:
        return "gq_rewrite"
    return "gq_generate"


async def gq_rewrite(state: ChatState) -> ChatState:
    """Second chance for weak/empty retrieval: gq_retrieve re-runs the same query widened (no domain filter, larger k)."""
    await emit_event("status", stage="retrieval", label="Broadening the search…")
    return {
        **state,
        "retrieval_attempts": 1,
        "sub_questions": [],
    }


_CONCISE_CONTEXT_TOKENS = 1400


def _build_answer_prompt(state: ChatState, *, concise: bool = False) -> tuple:
    """Assemble the grounded prompt for the general-query answer."""
    user_input = state["current_input"]
    messages = state.get("messages", [])
    regenerating = bool(state.get("regen_pending"))

    conversation_context = ""
    if len(messages) > 1:
        conversation_context = "\n".join(
            f"{m['role'].upper()}: {m['content'][:200]}" for m in messages[-6:]
        )

    statute = _tool_result(state, "statute_context")
    rag_sections_text = statute.context_text if statute else ""
    case_law_text = (statute.raw or {}).get("case_law_text", "") if statute else ""
    _, prompt_warning = _apply_compulsory_rag_policy(bool(state.get("rag_succeeded")))

    context_parts = []
    if rag_sections_text:
        context_parts.append(
            STATUTE_CONTEXT_BLOCK.format(rag_sections_text=rag_sections_text)
        )
    if case_law_text:
        context_parts.append(CASE_LAW_CONTEXT_BLOCK.format(case_law_text=case_law_text))
    from app.metrics.engineering_metrics import count_tokens_approx

    user_input_for_prompt = user_input[:_MAX_QUERY_CHARS]
    feedback = state.get("regen_feedback") if regenerating else None

    reserved = (
        count_tokens_approx(user_input_for_prompt)
        + count_tokens_approx(conversation_context or "")
        + count_tokens_approx(feedback or "")
        + 500
    )
    retrieved_context = (
        _fit_context_blocks(
            context_parts,
            reserved,
            max_tokens=_CONCISE_CONTEXT_TOKENS if concise else None,
        )
        if context_parts
        else ""
    )

    if retrieved_context:
        prompt = GROUNDED_QUERY_PROMPT.format(
            user_query=user_input_for_prompt,
            retrieved_context=retrieved_context,
        )
        if feedback:
            prompt += REGENERATION_FEEDBACK_BLOCK.format(feedback=feedback)
    else:

        prompt = (
            GENERAL_QUERY_PROMPT.format(query=user_input_for_prompt) + prompt_warning
        )

    if concise:
        prompt += CONCISE_ANSWER_SUFFIX
    if conversation_context:
        prompt = f"""Previous conversation context:
{conversation_context}

{prompt}"""
    return prompt, retrieved_context


def _concise_gates(state: ChatState) -> Dict[str, bool]:
    """The individual conditions behind _prefers_concise, exposed separately so gq_generate can log which gate(s) block the fast path on a given turn — the "concise first" path was..."""
    settings = get_settings()
    user_input = state["current_input"]
    return {
        "good_retrieval": state.get("retrieval_grade") == "good",
        "single_part": not bool(state.get("sub_questions")),
        "short_query": len(user_input.split())
        <= settings.concise_first_max_query_words,
        "not_multi_offense": _count_keyword_matches(user_input, CRIME_TYPE_KEYWORDS)
        < 2,
    }


def _prefers_concise(state: ChatState) -> bool:
    """Whether to answer with the concise prompt on the first attempt."""
    settings = get_settings()
    if not settings.concise_first_enabled:
        return False
    return all(_concise_gates(state).values())


def _can_retry_giveup(state: ChatState) -> bool:
    settings = get_settings()
    return (
        settings.llm_giveup_retry_enabled
        and _time_elapsed(state) < settings.llm_giveup_retry_max_elapsed_seconds
    )


async def gq_generate(state: ChatState) -> ChatState:
    """Build the grounded prompt from what retrieval returned and generate, streaming to the client."""
    regenerating = bool(state.get("regen_pending"))
    disclaimer_prefix, _ = _apply_compulsory_rag_policy(
        bool(state.get("rag_succeeded"))
    )

    if regenerating:
        await emit_event("reset")
        await emit_event(
            "status",
            stage="regenerating",
            label="Some claims weren't supported — redrafting from the sources…",
        )
    else:
        await emit_event("status", stage="generating", label="Drafting the answer…")
    await emit_text(disclaimer_prefix)

    retrieved_context = ""
    failed = False
    attempts = 1
    retried = False
    concise = False
    concise_gates = _concise_gates(state)
    try:
        retry_possible = get_settings().llm_giveup_retry_enabled
        concise = _prefers_concise(state)
        prompt, retrieved_context = _build_answer_prompt(state, concise=concise)
        answer = await invoke_llm_safely(
            get_llm(), prompt, stream=True, notify_incomplete=not retry_possible
        )
        if answer == _INCOMPLETE_GENERATION_NOTE and retry_possible:
            if _can_retry_giveup(state):
                logger.warning(
                    "Generation gave up — retrying once with a concise prompt"
                )
                await emit_event(
                    "status",
                    stage="retrying",
                    label="That was taking too long to work out — trying a simpler pass…",
                )
                prompt, retrieved_context = _build_answer_prompt(state, concise=True)
                answer = await invoke_llm_safely(get_retry_llm(), prompt, stream=True)
                attempts, retried = 2, True
            else:
                await emit_text(answer)

        if answer == _INCOMPLETE_GENERATION_NOTE:
            failed = True
    except Exception as e:
        logger.error("LLM error in general query: %s", e)
        answer = GENERAL_QUERY_ERROR
        failed = True
        await emit_text(answer)

    return {
        **state,
        "response": disclaimer_prefix + answer,
        "retrieved_context": retrieved_context,
        "regen_pending": False,
        "error": "generation_failed" if failed else state.get("error"),
        "trace": _merge_trace(
            state,
            generation={
                "variant": "concise" if (concise or retried) else "full",
                "attempts": attempts,
                "giveup_retry": retried,
                "failed": failed,
                "concise_gates": concise_gates,
            },
        ),
    }


def _regeneration_plan(report) -> tuple:
    """(feedback text, targeted retrieval queries) for the flagged claims."""
    flagged = report.confirmed_flagged[:3]
    feedback = "\n".join(
        f'- "{s.text.strip()[:240]}"' + (f" — {s.reason}" if s.reason else "")
        for s in flagged
    )
    queries: List[str] = []
    for s in report.confirmed_flagged:
        for citation in s.citations:
            if citation not in queries:
                queries.append(citation)
    return feedback, queries[:3]


async def gq_verify(state: ChatState) -> ChatState:
    """Post-generation grounding gate."""
    settings = get_settings()
    response = state.get("response") or ""
    finished = {
        **state,
        "messages": state["messages"] + [{"role": "assistant", "content": response}],
    }

    if state.get("error") == "generation_failed":
        return {
            **finished,
            "trace": _merge_trace(
                state, grounding={"verified": False, "reason": "generation_failed"}
            ),
        }

    citation_only = not state.get("rag_succeeded")

    await emit_event(
        "status", stage="verifying", label="Checking citations against the statutes…"
    )

    adjudication = {"seconds": 0.0, "calls": 0, "prompt_chars": 0, "output_chars": 0}

    async def _grounding_correction_invoke(prompt: str) -> str:
        t0 = time.monotonic()
        adjudication["calls"] += 1
        adjudication["prompt_chars"] += len(prompt)
        try:
            out = strip_reasoning_tags(
                await invoke_llm_safely(
                    get_grounding_correction_llm(), prompt, stream=False
                )
            )
            adjudication["output_chars"] += len(out)
            return out
        finally:
            adjudication["seconds"] += time.monotonic() - t0

    verify_started = time.monotonic()
    final_text, report = await _verify_response_citations(
        response,
        state.get("retrieved_sections"),
        retrieved_context_text=state.get("retrieved_context") or "",
        llm_invoke=_grounding_correction_invoke,
        citation_only=citation_only,
    )
    verify_seconds = time.monotonic() - verify_started

    score = report.overall_score if report is not None else None

    can_regenerate = (
        not citation_only
        and settings.grounding_retry_enabled
        and report is not None
        and report.llm_succeeded
        and report.confirmed_flagged
        and score is not None
        and score < settings.grounding_retry_threshold
        and (state.get("regen_count") or 0) < 1
        and _time_elapsed(state) < settings.request_budget_seconds
    )
    grounding_trace = {
        "verified": report is not None,
        "adjudicated": bool(report is not None and report.llm_succeeded),
        "score": score,
        "flagged": len(report.flagged) if report is not None else 0,
        "regenerated": bool(state.get("regen_count")),
        "verify_seconds": round(verify_seconds, 2),
        "adjudication": {
            **adjudication,
            "seconds": round(adjudication["seconds"], 2),
        },
    }
    if citation_only:
        grounding_trace["reason"] = "no_retrieval_citation_only"
    if can_regenerate:
        feedback, queries = _regeneration_plan(report)
        logger.info(
            "Grounding score %.2f < %.2f — regenerating once (targeted: %s)",
            score,
            settings.grounding_retry_threshold,
            queries,
        )
        return {
            **state,
            "regen_pending": True,
            "regen_count": 1,
            "regen_feedback": feedback,
            "extra_queries": queries,
            "retrieval_attempts": 1,
            "grounding_score": score,
            "trace": _merge_trace(
                state, grounding={**grounding_trace, "regenerating": True}
            ),
        }

    if final_text != response:

        await emit_event("replace", content=final_text)
    return {
        **state,
        "response": final_text,
        "grounding_score": score,
        "messages": state["messages"] + [{"role": "assistant", "content": final_text}],
        "trace": _merge_trace(state, grounding=grounding_trace),
    }


def _route_after_verify(state: ChatState) -> Literal["gq_retrieve", "end"]:
    return "gq_retrieve" if state.get("regen_pending") else "end"


async def _handle_document_validation(state: ChatState) -> ChatState:
    """Internal handler for document validation using the 3-layer pipeline."""
    document_content = state.get("document_content", "")

    if not document_content:
        response = DOCUMENT_VALIDATION_UPLOAD_PROMPT
        return {
            **state,
            "response": response,
            "messages": state["messages"]
            + [{"role": "assistant", "content": response}],
        }

    try:

        await emit_event("status", stage="classify", label="Reading the document…")
        classification = await classify_document(document_content)

        logger.info(
            f"[Layer 1] Document classified as: {classification.document_type} "
            f"(confidence: {classification.confidence:.2f})"
        )

        validator = get_statutory_validator()
        validation = validator.validate(document_content, classification.document_type)
        await emit_event(
            "status", stage="retrieval", label="Checking the law that applies…"
        )

        logger.info(
            f"[Layer 2] Statutory validation: {validation.passed}/{validation.total_checks} passed, "
            f"compliance score: {format_score(validation.compliance_score)}"
        )

        crime_rag = None

        async def init_rag():
            try:
                from app.tools.criminal_rag import get_criminal_rag_system

                rag_system = get_criminal_rag_system()
                await rag_system.initialize()
                return rag_system
            except Exception:
                return None

        async def init_civil():
            try:
                from app.tools.civil_rag import get_civil_rag_system

                rag_system = get_civil_rag_system()
                await rag_system.initialize()
                return rag_system
            except Exception:
                return None

        analyzer = get_legal_defect_analyzer(get_llm())
        think_task = asyncio.create_task(analyzer.think(classification))

        crime_rag, civil_rag = await asyncio.gather(init_rag(), init_civil())

        law_rag = get_indian_law_rag(crime_rag=crime_rag, civil_rag=civil_rag)
        law_context = await law_rag.retrieve_context(
            document_type=classification.document_type,
            missing_elements=validation.missing_elements,
            non_compliance=validation.non_compliance,
            document_text=document_content[:2000],
            jurisdiction_hints=classification.jurisdiction_hints,
        )

        logger.info(
            f"[Layer 2.5] Retrieved {len(law_context.references)} law references, "
            f"{len(law_context.applicable_acts)} applicable acts"
        )

        await emit_event("status", stage="generation", label="Writing the analysis…")
        result = await analyzer.analyze_defects(
            classification=classification,
            validation=validation,
            law_context=law_context,
            document_text=document_content[:5000],
            think_output=await think_task,
            stream=True,
        )

        response = result["formatted_response"]

        await emit_event("replace", content=response)

        logger.info(
            f"[Layer 3] Analysis complete. Defects: {result['defect_count']}, "
            f"Compliance: {format_score(result['compliance_score'])}"
        )

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
        logger.exception("Document validation error: %s", e)

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
                f"## 📊 Statutory Compliance: {format_score(validation.compliance_score)}",
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
            await emit_event("replace", content=response)
        except Exception:
            response = (
                "I apologize, but I encountered an error while validating your document. "
                "Please try again or consult a qualified legal practitioner for document review."
            )
            await emit_event("replace", content=response)

        return {
            **state,
            "response": response,
            "error": str(e),
            "messages": state["messages"]
            + [{"role": "assistant", "content": response}],
        }


async def handle_non_legal_query(state: ChatState) -> ChatState:
    """Handle non-legal queries with a polite rejection message."""
    response = NON_LEGAL_RESPONSE

    return {
        **state,
        "response": response,
        "messages": state["messages"] + [{"role": "assistant", "content": response}],
    }


async def handle_clarification(state: ChatState) -> ChatState:
    """Ask the one clarifying question classify_intent put in state["response"]."""
    response = state.get("response") or CLARIFY_GENERIC
    return {
        **state,
        "response": response,
        "messages": state["messages"] + [{"role": "assistant", "content": response}],
    }


def route_by_intent(
    state: ChatState,
) -> Literal[
    "document_analysis",
    "crime_report",
    "find_lawyer",
    "general_query",
    "non_legal",
    "clarify",
]:
    """Route to the appropriate handler based on classified intent."""
    intent = state.get("intent")
    if intent in (
        "document_analysis",
        "crime_report",
        "find_lawyer",
        "general_query",
        "non_legal",
        "clarify",
    ):
        return intent
    return "general_query"


def build_legal_chatbot_graph() -> StateGraph:
    """Build the LangGraph workflow for the legal chatbot."""
    workflow = StateGraph(ChatState)

    workflow.add_node("classify_intent", classify_intent)
    workflow.add_node("document_analysis", handle_document_analysis)
    workflow.add_node("crime_report", handle_crime_report)
    workflow.add_node("find_lawyer", handle_find_lawyer)
    workflow.add_node("non_legal", handle_non_legal_query)
    workflow.add_node("clarify", handle_clarification)
    workflow.add_node("gq_plan", gq_plan)
    workflow.add_node("gq_retrieve", gq_retrieve)
    workflow.add_node("gq_grade", gq_grade)
    workflow.add_node("gq_rewrite", gq_rewrite)
    workflow.add_node("gq_generate", gq_generate)
    workflow.add_node("gq_verify", gq_verify)

    workflow.set_entry_point("classify_intent")

    workflow.add_conditional_edges(
        "classify_intent",
        route_by_intent,
        {
            "document_analysis": "document_analysis",
            "crime_report": "crime_report",
            "find_lawyer": "find_lawyer",
            "general_query": "gq_plan",
            "non_legal": "non_legal",
            "clarify": "clarify",
        },
    )

    workflow.add_edge("gq_plan", "gq_retrieve")
    workflow.add_edge("gq_retrieve", "gq_grade")
    workflow.add_conditional_edges(
        "gq_grade",
        _route_after_grade,
        {"gq_rewrite": "gq_rewrite", "gq_generate": "gq_generate"},
    )
    workflow.add_edge("gq_rewrite", "gq_retrieve")
    workflow.add_edge("gq_generate", "gq_verify")
    workflow.add_conditional_edges(
        "gq_verify",
        _route_after_verify,
        {"gq_retrieve": "gq_retrieve", "end": END},
    )

    for terminal in (
        "document_analysis",
        "crime_report",
        "find_lawyer",
        "non_legal",
        "clarify",
    ):
        workflow.add_edge(terminal, END)

    return workflow


_MAX_SESSION_MESSAGES = 20


def _append_capped(
    left: Optional[List[Message]], right: Optional[List[Message]]
) -> List[Message]:
    return ((left or []) + (right or []))[-_MAX_SESSION_MESSAGES:]


class SessionState(TypedDict):
    """The only state that is checkpointed: conversation history."""

    messages: Annotated[List[Message], _append_capped]


@dataclass
class _TurnContext:
    """Per-turn inputs and output, passed through a contextvar rather than graph state so none of it is checkpointed (see SessionState)."""

    session_id: str
    started_at: float
    document_content: Optional[str] = None
    document_type: Optional[str] = None
    result: Optional[Dict[str, Any]] = None


_turn_var: contextvars.ContextVar[Optional[_TurnContext]] = contextvars.ContextVar(
    "turn_context", default=None
)


class LegalChatbot:
    """Main chatbot class that wraps the LangGraph workflow."""

    def __init__(self, checkpointer: Optional[BaseCheckpointSaver] = None):
        self._checkpointer = (
            checkpointer if checkpointer is not None else InMemorySaver()
        )
        self._turn_graph = build_legal_chatbot_graph().compile(checkpointer=False)

        session = StateGraph(SessionState)
        session.add_node("run_turn", self._run_turn)
        session.set_entry_point("run_turn")
        session.add_edge("run_turn", END)
        self.graph = session.compile(checkpointer=self._checkpointer)

        self._active_stream_tasks: Dict[str, asyncio.Task] = {}
        self._in_flight = 0

        self._last_seen: Dict[str, float] = {}

    def _acquire_slot(self) -> None:

        if self._in_flight >= get_settings().chat_max_concurrent:
            raise ChatBusyError(
                "The assistant is busy right now. Please try again shortly."
            )
        self._in_flight += 1

    def _release_slot(self) -> None:
        self._in_flight = max(0, self._in_flight - 1)

    def _config(self, session_id: str, touch: bool = True) -> Dict[str, Any]:
        """thread_id is the memory key, which already carries the account scope (`user:<id>:<session>` / `guest:<session>`), so one account cannot read another's conversation by guessing..."""
        if touch:
            self._touch(session_id)
        return {
            "configurable": {"thread_id": session_id},
            "metadata": {"last_active": datetime.now(timezone.utc).isoformat()},
        }

    def _touch(self, session_id: str) -> None:
        if not isinstance(self._checkpointer, InMemorySaver):
            return
        settings = get_settings()
        now = time.monotonic()
        self._last_seen[session_id] = now
        stale = [
            sid
            for sid, seen in self._last_seen.items()
            if now - seen > settings.session_ttl_seconds
        ]
        overflow = len(self._last_seen) - settings.max_sessions
        if overflow > 0:
            stale += sorted(self._last_seen, key=self._last_seen.get)[:overflow]
        for sid in set(stale) - {session_id}:
            self._checkpointer.delete_thread(sid)
            self._last_seen.pop(sid, None)

    async def get_session_history(self, session_id: str) -> List[Message]:
        """The live (20-message-capped) conversation window for a session."""
        snapshot = await self.graph.aget_state(self._config(session_id, touch=False))
        return list((snapshot.values or {}).get("messages") or [])

    async def has_session(self, session_id: str) -> bool:
        """Whether this session already has a conversation checkpoint."""
        return bool(await self.get_session_history(session_id))

    async def seed_session(self, session_id: str, messages: List[Message]) -> None:
        """Prime a thread from DB-loaded history, but only if it has none yet (avoids clobbering an active conversation with a stale DB read)."""
        if messages and not await self.has_session(session_id):
            await self.graph.aupdate_state(
                self._config(session_id),
                {"messages": messages[-_MAX_SESSION_MESSAGES:]},
                as_node="run_turn",
            )

    async def _append_assistant(self, session_id: str, content: str) -> None:
        await self.graph.aupdate_state(
            self._config(session_id),
            {"messages": [{"role": "assistant", "content": content}]},
            as_node="run_turn",
        )

    async def clear_session(self, session_id: str) -> None:
        """Forget a session's history and stop any in-flight generation for it (otherwise it would keep running and write into a session the user just cleared)."""
        self.stop_stream(session_id)
        self._last_seen.pop(session_id, None)
        await self._checkpointer.adelete_thread(session_id)

    async def _run_turn(self, state: SessionState) -> Dict[str, Any]:
        """The outer graph's only node: run the workflow for this turn."""
        ctx = _turn_var.get()
        if ctx is None:
            raise RuntimeError("chat turn started without a turn context")
        *history, current = state["messages"]
        turn_state: ChatState = {
            "messages": history,
            "current_input": current["content"],
            "conversation_context": None,
            "intent": None,
            "document_content": ctx.document_content,
            "document_type": ctx.document_type or "unknown",
            "document_info": None,
            "document_validation": None,
            "crime_details": None,
            "crime_report": None,
            "lawyer_query": None,
            "lawyers_found": None,
            "response": None,
            "session_id": ctx.session_id,
            "error": None,
            "started_at": ctx.started_at,
            "trace": {},
        }
        result = await self._turn_graph.ainvoke(turn_state)
        ctx.result = result
        response = result.get("response")
        return (
            {"messages": [{"role": "assistant", "content": response}]}
            if response
            else {}
        )

    async def stream_chat(
        self,
        message: str,
        session_id: str = "default",
        document_content: Optional[str] = None,
        document_type: Optional[str] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream a chat turn (see _stream_chat), holding a concurrency slot for its whole lifetime — including when the client disconnects."""
        self._acquire_slot()
        inner = self._stream_chat(message, session_id, document_content, document_type)
        try:
            async for event in inner:
                yield event
        finally:

            await inner.aclose()
            self._release_slot()

    async def _stream_chat(
        self,
        message: str,
        session_id: str = "default",
        document_content: Optional[str] = None,
        document_type: Optional[str] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream chat response token by token."""
        request_id_var.set(uuid.uuid4().hex[:8])

        english_message, lang = await preprocess_query(message)
        translate_out = (
            lang.is_reliable and lang.language != get_settings().default_language
        )

        ctx = _TurnContext(
            session_id=session_id,
            started_at=time.monotonic(),
            document_content=document_content,
            document_type=document_type,
        )
        config = self._config(session_id)
        user_input = {"messages": [{"role": "user", "content": english_message}]}

        queue: asyncio.Queue = asyncio.Queue()
        tokens_streamed = False

        async def run_graph():
            _stream_queue_var.set(queue)
            _turn_var.set(ctx)
            try:
                await self.graph.ainvoke(user_input, config)
                return ctx.result or {}
            except Exception as e:
                logger.exception("Graph error during streaming")
                return {
                    "response": "I apologize, but I encountered an error processing your request. Please try again.",
                    "error": str(e),
                }
            finally:
                await queue.put(None)

        task = asyncio.create_task(run_graph())

        previous = self._active_stream_tasks.get(session_id)
        if previous is not None and not previous.done():
            previous.cancel()
        self._active_stream_tasks[session_id] = task

        accumulated = ""
        intent_seen: Optional[str] = None
        stopped = False
        superseded = False
        try:

            while True:
                item = await queue.get()
                if item is None:
                    break
                if isinstance(item, dict):
                    kind = item.get("type")
                    if kind == "routing":
                        intent_seen = item.get("intent")
                        continue
                    if kind == "reset":
                        accumulated = ""
                    elif kind == "replace":
                        accumulated = item.get("content", "")
                    if translate_out and kind in ("reset", "replace"):
                        continue
                    yield item
                    continue
                tokens_streamed = True
                accumulated += item
                if not translate_out:
                    yield {"type": "token", "content": item}

            try:
                result = await task
            except asyncio.CancelledError:

                stopped = True
                result = {"intent": intent_seen, "response": accumulated}
        finally:
            if self._active_stream_tasks.get(session_id) is task:
                self._active_stream_tasks.pop(session_id, None)

                if not task.done():
                    task.cancel()
            else:

                superseded = True

        english_text = result.get("response", "") or accumulated
        intent = result.get("intent") or intent_seen

        if stopped and english_text and not superseded:
            await self._append_assistant(session_id, english_text)

        if translate_out:
            response_text = await postprocess_response(english_text, lang)
            if response_text:
                yield {"type": "token", "content": response_text}
        else:
            response_text = english_text
            if not tokens_streamed and response_text:
                yield {"type": "token", "content": response_text}

        if superseded:

            yield {"type": "superseded", "session_id": session_id}
            return

        if stopped:
            yield {
                "type": "stopped",
                "session_id": session_id,
                "intent": intent,
                "response": response_text,
                "response_en": english_text,
                "query_en": english_message,
                "language": lang.language,
            }
            return

        yield {
            "type": "done",
            "session_id": session_id,
            "intent": intent,
            "response": response_text,
            "response_en": english_text,
            "query_en": english_message,
            "language": lang.language,
            "lawyers_found": result.get("lawyers_found"),
            "document_info": result.get("document_info"),
            "document_validation": result.get("document_validation"),
            "crime_report": result.get("crime_report"),
            "trace": result.get("trace"),
        }

    def stop_stream(self, session_id: str) -> bool:
        """Cancel an in-flight stream_chat() generation for this session, if any."""
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
        """Process a chat turn (see _chat) while holding a concurrency slot."""
        self._acquire_slot()
        try:
            return await self._chat(
                message, session_id, document_content, document_type
            )
        finally:
            self._release_slot()

    async def _chat(
        self,
        message: str,
        session_id: str = "default",
        document_content: Optional[str] = None,
        document_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Process a chat message and return the response."""
        request_id_var.set(uuid.uuid4().hex[:8])

        english_message, lang = await preprocess_query(message)

        ctx = _TurnContext(
            session_id=session_id,
            started_at=time.monotonic(),
            document_content=document_content,
            document_type=document_type,
        )
        _turn_var.set(ctx)
        await self.graph.ainvoke(
            {"messages": [{"role": "user", "content": english_message}]},
            self._config(session_id),
        )
        result = ctx.result or {}

        english_response = result.get("response")

        display_response = (
            await postprocess_response(english_response or "", lang)
            or "I'm sorry, I couldn't process your request."
        )

        return {
            "response": display_response,
            "response_en": english_response,
            "query_en": english_message,
            "language": lang.language,
            "language_confidence": lang.confidence,
            "intent": result.get("intent"),
            "document_info": result.get("document_info"),
            "document_validation": result.get("document_validation"),
            "crime_report": result.get("crime_report"),
            "lawyers_found": result.get("lawyers_found"),
            "error": result.get("error"),
            "trace": result.get("trace"),
        }


_chatbot: Optional[LegalChatbot] = None


def get_chatbot() -> LegalChatbot:
    """Get or create the chatbot instance, bound to the app's checkpointer (Postgres once app startup has initialised it; in-memory otherwise)."""
    global _chatbot
    if _chatbot is None:
        _chatbot = LegalChatbot(get_checkpointer())
    return _chatbot
