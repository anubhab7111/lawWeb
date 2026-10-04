"""
LangGraph-based legal chatbot implementation.
This module defines the chatbot workflow using LangGraph for state management and routing.
"""

import asyncio
import json
import logging
import time
import contextvars
import re
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

from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, StateGraph

from app.checkpointing import get_checkpointer
from app.config import get_settings
from app.prompts import (
    CASE_LAW_CONTEXT_BLOCK,
    CLARIFY_GENERIC,
    CONCISE_ANSWER_SUFFIX,
    CLARIFY_LAW_OR_LAWYER,
    CLARIFY_LAW_OR_REPORT,
    CLARIFY_PREFIX,
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
    REGENERATION_FEEDBACK_BLOCK,
    ROUTE_TIEBREAK_INTENT_DESCRIPTIONS,
    ROUTE_TIEBREAK_PROMPT,
    ROUTE_TIEBREAK_UNSURE,
    STATUTE_CONTEXT_BLOCK,
    STATUTE_QUERY_REWRITE_PROMPT,
    sanitize_untrusted_document,
)
from app.routing_keywords import CRIME_TYPE_KEYWORDS
from app.text_match import contains_word, count_words
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
from app.multilingual import preprocess_query, postprocess_response
from app.logging_config import request_id_var
from app.tool_dispatch import (
    RAG_TOOL_REGISTRY,
    ToolInvocationResult,
    infer_indian_kanoon_context_type,
    select_tools,
)
from app.tools.crime_reporter import detect_crime_type
from app.tools.document_classifier import get_document_classifier
from app.tools.indian_kanoon import get_indian_kanoon_tool
from app.tools.indian_law_rag import get_indian_law_rag
from app.tools.lawyer_recommender import (
    format_lawyer_results,
    recommend_lawyers as recommend_lawyers_core,
)
from app.tools.legal_defect_analyzer import get_legal_defect_analyzer
from app.tools.statutory_validator import format_score, get_statutory_validator


logger = logging.getLogger(__name__)

LLM_NUM_CTX = 8192  # Ollama defaults to 2048, which silently clips grounded prompts
# 8192 is the largest window that still keeps qwen3:4b 100% on the 4GB GPU
# (measured live via `ollama ps`: 3.65GB resident, 0 CPU spill; 10240+ spills
# ~0.64GB to CPU and slows generation). num_ctx was 6144 with num_predict 3072,
# which left ~3072 tokens for thinking+answer — and on complex, multi-part
# grounded questions (privacy limits, FIR quashing, marital rape, force
# majeure) qwen3:4b routinely spent that entire budget re-drafting inside its
# own <think> block and never emitted the closing </think>, so the whole
# generation was discarded and the user got _INCOMPLETE_GENERATION_NOTE
# instead of an answer (measured: 4/18 eval queries gave up this way). Raising
# the window to 8192 and num_predict to 4608 gives thinking room to finish and
# still answer — verified over the full 18-prompt eval: all four give-up queries
# now return full grounded answers (0/18 give-ups vs 4/18), lifting answer
# relevance 0.61->0.67 and rag-triad 0.56->0.59, at a mean-latency cost of
# ~93s->116s (p95 113s->154s). The extra 1024 tokens of
# window also grows the _fit_context_blocks budget (num_ctx - num_predict) from
# 3072 to 3584, so retrieved context is not starved to pay for it.
LLM_NUM_PREDICT = 4608  # tokens reserved for thinking + the answer
_PROMPT_SAFETY_MARGIN = 256  # headroom for chat scaffolding the estimate can't see
_MAX_QUERY_CHARS = 8000  # clamp on user-derived text so one huge query can't overflow


@lru_cache()
def get_llm() -> ChatOllama:
    """Get cached LLM instance for better performance.

    reasoning=False, not True: verified live (2 empty answers out of 3
    identical requests) that with reasoning=True, Ollama's `thinking` field
    and `content` field are genuinely separate per-chunk, and qwen3:4b's
    thinking-phase length is variable enough that it can consume the whole
    num_predict budget before ever starting the answer, leaving `content`
    completely empty with no error raised. With reasoning=False the same
    thinking text lands inline in `content` (ending in `</think>`, same as
    get_fast_llm_prose() already relies on), which invoke_llm_safely()
    strips/filters — and, critically, always has *something* to fall back
    to if generation gets cut off mid-thought, instead of nothing.
    """
    settings = get_settings()
    return ChatOllama(
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        base_url=settings.ollama_base_url,
        num_ctx=LLM_NUM_CTX,
        num_predict=LLM_NUM_PREDICT,
        timeout=210.0,  # kept above _LLM_TIMEOUT_SECONDS (the real, enforced cap)
        reasoning=False,
        keep_alive="1h",  # loading the 14B model is the OOM-prone step — do it rarely
    )


def _truncate_block(block: str, max_chars: int) -> str:
    """Cut a context block to max_chars at a provision boundary ("\n\n• "
    entries) or, failing that, the last line/sentence end. Cutting mid-word
    would hand the model a half-quoted section it may cite as if complete."""
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
    """get_llm() with a higher temperature, used for the single retry after a
    give-up. At temperature 0.1 the same prompt tends to retrace the same
    reasoning; a different sample plus a smaller prompt gives it a real chance.
    Same model and num_ctx, so Ollama doesn't reload anything."""
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
    """
    Join retrieved-context blocks (already in priority order: statute → case
    law → Indian Kanoon) without exceeding the model's input budget. Drops
    lower-priority blocks and truncates the last kept one, so the instruction
    template and user query always survive — this is what stops Ollama from
    silently front-truncating the grounded statute block on long prompts.
    """
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
            if remaining > 50:  # ~4 chars/token — keep a useful truncated head
                kept.append(_truncate_block(block, remaining * 4))
            break
    return "\n\n".join(kept)


@lru_cache()
def get_fast_llm() -> ChatOllama:
    """Get cached small LLM for classification/routing tasks (local Ollama).

    num_ctx=LLM_NUM_CTX, not a smaller value: fast_llm_model is the same
    Ollama model tag as get_llm() (both "qwen3:4b" — see config.py), and
    Ollama reloads a model whenever a request asks for a different num_ctx
    than what's currently loaded (verified live via `ollama ps`, ~1.7-2s per
    reload). A single chat request calls both get_llm() and get_fast_llm()
    in sequence, so a mismatched num_ctx here was forcing a reload on every
    handoff between them — pure added latency for no benefit.
    """
    settings = get_settings()
    return ChatOllama(
        model=settings.fast_llm_model,
        temperature=0,
        base_url=settings.ollama_base_url,
        num_ctx=LLM_NUM_CTX,
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
    mid-thought before ever reaching the answer.

    num_ctx=LLM_NUM_CTX, same reasoning as get_fast_llm(): matching the
    context size of get_llm()'s "qwen3:4b" instance avoids a model reload
    every time a single request pipeline hands off between the two.
    """
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
    """Dedicated LLM for grounding_verifier's claim-correction call — same
    small model as get_fast_llm_prose(), but grammar-constrained to
    CORRECTION_RESPONSE_SCHEMA via Ollama's structured-output support. Free
    prompting alone ("respond with ONLY a JSON array") does not work for
    this task: qwen3:4b reliably reasons through each claim in prose instead
    and gets cut off by num_predict before emitting any JSON, so the
    correction step silently never fired. Kept separate from
    get_fast_llm_prose() since that's shared with bare_act_explorer.py and
    case_summarizer.py for free-text generation, which format= would break.
    num_predict has headroom above get_fast_llm_prose()'s 900 for the
    worst-case _MAX_LLM_CORRECTIONS-sized batch."""
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


# Wall-clock cap on a single answer generation (asyncio.wait_for enforces it;
# ChatOllama's own timeout= kwarg is a silent no-op). Sized to the LLM_NUM_PREDICT
# budget: 4608 tokens at qwen3:4b's measured ~30-35 tok/s can take ~130-150s of
# generation on the hardest multi-part grounded questions. At 120s those exact
# queries (privacy limits, basic-structure doctrine, marital rape, contract
# under economic pressure) were killed mid-answer and served GENERAL_QUERY_ERROR
# — a 386-char "having trouble" fallback the LLM judge scores 0 for relevance and
# recall. 180s lets the full think+answer finish (verified live) with headroom
# below the worst case, trading ~30-50s of tail latency on the 4 hardest queries
# for a real answer instead of an error.
_LLM_TIMEOUT_SECONDS = 180

# get_llm()'s prompt template always opens an implicit <think> block server-side
# (confirmed against the raw Ollama API — see strip_reasoning_tags), so every
# generation starts as thinking and only becomes a real answer once the model
# emits the closing </think>. Measured live: on multi-provision questions
# qwen3:4b can spend its *entire* LLM_NUM_PREDICT budget re-drafting the answer
# inside its own thinking before ever closing the tag — so "no </think> yet"
# is not a rare edge case, it's a real failure mode that must not be shown to
# the user as if it were the answer (see _INCOMPLETE_GENERATION_NOTE below).
# This cap is sized comfortably above the worst-case character output for
# LLM_NUM_PREDICT tokens (~4-5 chars/token) — it fires only as an absolute
# backstop against a truly runaway/never-closing generation, not during
# normal (if verbose) thinking.
_THINKING_BUFFER_SAFETY_CAP = 20000

# Shown instead of raw thinking text whenever generation ends (hits the safety
# cap, exhausts num_predict, or the stream errors) without ever producing a
# closed </think> — i.e. the model never actually finished formulating an
# answer. Deliberately not a truncated dump of the buffered thinking: that
# text is internal reasoning-in-progress (drafts, self-corrections, "let me
# check..."), not a legal answer, and showing it verbatim is worse for a demo
# than an honest "please retry".
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
    """Fail fast when Ollama is down or wedged. Without it every request waits
    out the full generation timeout (up to 180s) before failing; after a few
    consecutive failures we reject immediately for a cooldown, then let one
    probe request through (half-open) — a single failure there re-opens it."""

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
    """Push a non-token event (status, reset, replace, ...) to the live SSE
    stream. No-op outside a streaming request, so nodes can call it freely."""
    queue = _stream_queue_var.get(None)
    if queue is not None:
        await queue.put({"type": event_type, **payload})


async def emit_text(text: str) -> None:
    """Stream literal text (e.g. a disclaimer) ahead of the generated answer."""
    queue = _stream_queue_var.get(None)
    if queue is not None and text:
        await queue.put(text)


async def invoke_llm_safely(
    llm: ChatOllama, prompt: str, stream: bool = False, notify_incomplete: bool = True
) -> str:
    """Safely invoke LLM with proper error handling.

    stream=False is the default: streaming into the user's token queue is the
    deliberate exception (final answer generation), never something a helper
    can do by accident. The queue lives in a contextvar that asyncio tasks
    inherit, so a streaming default would let any auxiliary call made from
    inside a chat handler (query rewrite, grounding fact-check, a summariser)
    dump its raw output into the user's answer. Pass stream=True only for the
    call whose text is the answer.

    notify_incomplete=False keeps the give-up note (see below) out of the
    stream — the return value is still the note — for a caller that will retry.

    Every model here runs with reasoning=False (see get_llm()), so a
    thinking preamble arrives inline in the token stream ending with a
    literal `</think>`, not as a separate field, and every generation starts
    inside that preamble (the opening tag is injected server-side). Both
    branches below withhold/strip it: streaming buffers tokens until the
    closing tag is seen so the preamble is never shown live; non-streaming
    strips everything up to and including it. If generation ends — safety
    cap, num_predict exhausted, or the model just stops — without a closing
    tag ever appearing, that means the model never actually finished
    formulating an answer (measured live: it can spend its whole budget
    re-drafting inside the thinking phase), so both branches return
    _INCOMPLETE_GENERATION_NOTE rather than the raw buffered thinking text —
    showing that verbatim would leak internal monologue/drafts to the user.
    """
    _llm_breaker.check()
    queue = _stream_queue_var.get(None) if stream else None

    if queue is not None:
        # Streaming mode - use astream and push chunks to queue, filtering
        # out a leading thinking preamble before anything reaches the queue.
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
                    # Absolute backstop against a runaway/never-closing
                    # generation. Stop consuming the stream entirely rather
                    # than falling through to the normal per-token path —
                    # otherwise the model's continued thinking output would
                    # keep arriving as if it were real content.
                    logger.warning(
                        "thinking exceeded %s chars without closing — giving up",
                        _THINKING_BUFFER_SAFETY_CAP,
                    )
                    visible_response = _INCOMPLETE_GENERATION_NOTE
                    if notify_incomplete:
                        await queue.put(_INCOMPLETE_GENERATION_NOTE)
                    return visible_response
            if in_thinking and buffer:
                # Stream ended (hit num_predict or stopped) before a closing
                # tag ever showed up — the buffered text is unfinished
                # thinking, not an answer.
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
            # Same treatment as a user-initiated Stop: keep whatever was
            # already streamed to the client rather than discarding it.
            _llm_breaker.record_failure()
            note = "\n\n*(Response generation took too long and was cut short.)*"
            visible_response += note
            await queue.put(note)
            logger.warning(
                "generation exceeded %ss, returning partial output", _LLM_TIMEOUT_SECONDS
            )
            return visible_response
        except Exception:
            _llm_breaker.record_failure()
            logger.exception("LLM streaming error")
            raise
    else:
        # Non-streaming mode. ainvoke (not run_in_executor + invoke): on
        # timeout wait_for cancels the in-flight HTTP request itself, where a
        # thread would keep blocking a shared executor worker until the
        # client's own (longer) timeout fired.
        try:
            response = await asyncio.wait_for(
                llm.ainvoke([HumanMessage(content=prompt)]),
                timeout=_LLM_TIMEOUT_SECONDS,
            )
            _llm_breaker.record_success()
            raw = response.content
            # A schema-constrained model (format=...) decodes straight into the
            # grammar and never emits a think block, so a missing </think> is
            # the normal shape of its output — not a give-up. Treating it as one
            # discarded every grounding-correction reply (valid JSON, ~3s) and
            # left claim-level correction silently switched off.
            if getattr(llm, "format", None):
                return strip_reasoning_tags(raw)
            if "</think>" not in raw and not get_settings().llm_thinking:
                return raw.strip()
            if "</think>" not in raw:
                # Never closed the thinking phase — raw is entirely internal
                # monologue, not an answer (see docstring).
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


# ============================================================================
# Node Functions
# ============================================================================


def _count_keyword_matches(text: str, keywords: frozenset) -> int:
    """Count how many keywords match in the text."""
    return count_words(text, keywords)


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
        if contains_word(text_lower, act):
            entities.append(act.title())

    return list(set(entities))


async def _invoke_fast_text(prompt: str, timeout: float) -> str:
    """Short auxiliary generation with the thinking preamble stripped. Uses the
    prose-sized budget: the 128-token classification LLM spends its whole budget
    on thinking and never reaches the answer."""
    return await asyncio.wait_for(
        invoke_llm_safely(get_fast_llm_prose(), prompt, stream=False), timeout=timeout
    )


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

    history_lines = [f"{m['role'].upper()}: {m['content'][:300]}" for m in prior[-4:]]
    prompt = QUERY_REWRITE_PROMPT.format(
        history="\n".join(history_lines), question=current_input
    )

    try:
        rewritten = (
            await _invoke_fast_text(prompt, timeout=25.0)
        ).strip().strip('"').strip()
        if rewritten == _INCOMPLETE_GENERATION_NOTE or not rewritten or len(rewritten) > 300 or "\n" in rewritten:
            return current_input
        if rewritten.lower() != current_input.lower():
            logger.info(f"Retrieval query rewritten: {rewritten[:120]}")
        return rewritten
    except Exception as e:
        logger.warning(f"[Router] Query rewrite failed ({e}) — using raw input.")
        return current_input


# ============================================================================
# Primary Router (embedding-based) + deterministic policy layer
# ============================================================================

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


_UPLOAD_HELP_KEYWORDS = ("upload", "i will upload", "how to upload", "can i upload")
_ACTION_INTENTS = frozenset({"find_lawyer", "crime_report"})
_CLARIFY_MAX_WORDS = 25  # a long, detailed message has enough to answer directly


def _asks_about_uploading(text: str) -> bool:
    lowered = text.lower()
    return any(kw in lowered for kw in _UPLOAD_HELP_KEYWORDS)


def _merge_trace(state: ChatState, **entries: Any) -> Dict[str, Any]:
    return {**(state.get("trace") or {}), **entries}


def _looks_like_an_explicit_question(text: str) -> bool:
    """A '?' means the user already wrote out a specific legal question,
    not a raw situational statement — see _clarification_for."""
    return "?" in text


def _ambiguous_action_contenders(result) -> FrozenSet[str]:
    """The near-tied intents worth escalating over — only when an *action*
    intent (report a crime / find a lawyer) is among them; two explanatory
    intents tying is not worth interrupting for."""
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
    """Non-empty iff this turn is genuinely torn between an *action* flow
    (report a crime / find a lawyer) and another one — worth escalating,
    first to the LLM tie-breaker (_resolve_ambiguity_with_llm), then, only if
    that is also unsure, to a clarifying question. Guessing wrong here is
    worse than for two explanatory flows: the user gets a confident answer to
    a question they did not ask, and cannot tell.

    Deliberately narrow so it does not nag: it never fires with an attached
    document, on long messages, right after its own previous question, or on
    an explicit written-out question — only on a raw situational statement
    ("my landlord is cheating me"), where there genuinely is no question to
    answer yet. This last guard is load-bearing, not decorative: measured
    against the project's own 18-query eval set, embedding-classifier margins
    of 0.001-0.03 between "explain the law" and "find a lawyer"/"report a
    crime" are common noise on ordinary legal questions — before this guard,
    6 of 18 canonical questions ("Can an FIR be quashed by the High Court? On
    what grounds?") were intercepted and never answered. Every one of those
    six, like every false trigger found, was an explicit written-out question."""
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
    """Schema-constrained tie-breaker for a near-tied route (see
    _resolve_ambiguity_with_llm) — same pattern as get_grounding_correction_llm():
    format= is what actually gets qwen3:4b to answer with the label instead of
    reasoning through it in prose. One tiny cached instance per distinct
    candidate-set (at most a handful of combinations of the 3 legal intents
    that can tie), reusing fast_llm_model/num_ctx so Ollama never reloads
    between this and get_fast_llm()/get_grounding_correction_llm()."""
    settings = get_settings()
    schema = {
        "type": "object",
        "properties": {
            "intent": {"type": "string", "enum": list(candidates) + [ROUTE_TIEBREAK_UNSURE]}
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
    """Cascade routing, tier 2: one fast, schema-constrained call to resolve a
    near-tied route before asking the user. Only reached from classify_intent
    when _clarification_gate already found the turn worth interrupting for —
    so this never runs on the common, unambiguous case; the fast path stays a
    pure embedding lookup with zero LLM calls. Returns (resolved_intent,
    trace); resolved_intent is None when the model is also unsure, returns
    something outside the offered candidates, or the call fails/times out —
    the caller then falls back to the existing clarifying question. Must
    never raise or block routing."""
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
        "tried": True, "seconds": elapsed, "candidates": list(key),
        "raw": picked, "resolved": resolved,
    }


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
    2. Clarification: a near-tie between an action intent (report a crime,
       find a lawyer) and another flow becomes one clarifying question
       instead of a silent guess (_clarification_for).
    3. Domain-hint inference (classify_domain_hint_embedding), an
       independent binary embedding classifier
    4. History-aware query rewrite for retrieval
    5. Tool selection (tool_dispatch.select_tools) — the handlers execute
       exactly the tools chosen here.

    Returns enriched state with:
    - intent: Primary classification ("clarify" when a question is asked)
    - routing_confidence / routing_reasoning / is_ambiguous / secondary_intents
    - selected_tools: Tools the handler will run
    - domain_hint: Soft bias for unified statute retrieval
    - extracted_entities: Legal terms found
    - trace["routing"]: what was decided and why
    """
    user_input = state["current_input"]
    has_document = bool(state.get("document_content"))
    messages = state.get("messages", [])

    logger.info("Router input=%.100r has_document=%s", user_input, has_document)

    result = await classify_intent_embedding(user_input, has_document)
    # Ambiguous ties among the four *legal* intents default to general_query
    # (still grounded, just not the specific handler) rather than falling
    # back to an LLM — no model call anywhere in this routing path. But if
    # non_legal is itself the top-scoring class, trust it even when the
    # margin is thin: collapsing an ambiguous non_legal read into
    # general_query would silently reintroduce the old "assume legal when
    # unsure" bias non_legal was added to remove.
    intent = result.primary_intent
    if result.is_ambiguous and intent != "non_legal":
        # An attached document must not be silently dropped by the fallback.
        intent = "document_analysis" if has_document else "general_query"

    # No document attached means nothing to analyse: answer as a general legal
    # question, unless the user is asking how uploading works. Decided here so
    # the reported intent matches the flow that actually runs.
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
        # _resolve_ambiguity_with_llm has its own internal try/except and should
        # never raise — this is defense in depth so an unexpected failure there
        # degrades to the clarifying question rather than 500ing the whole turn.
        try:
            resolved, tiebreak_trace = await _resolve_ambiguity_with_llm(contenders, user_input)
        except Exception as e:
            logger.exception("Route tie-break raised unexpectedly (%s)", e)
            resolved, tiebreak_trace = None, {"tried": True, "error": str(e)}
        routing_trace["tiebreak"] = tiebreak_trace
        if resolved:
            logger.info(
                "Router: tie-break resolved %s -> %s (%.1fs)",
                sorted(contenders), resolved, tiebreak_trace.get("seconds", 0.0),
            )
            intent = resolved
            routing_trace["intent"] = intent
            # falls through to the normal flow below — no clarifying question
        else:
            logger.info("Router: ambiguous %s — asking a clarifying question", sorted(contenders))
            await emit_event("routing", intent="clarify")
            return {
                **base,
                "intent": "clarify",
                "response": _clarify_question(contenders),
                "clarification": True,
                "selected_tools": [],
                "domain_hint": None,
                "trace": _merge_trace(state, routing={**routing_trace, "intent": "clarify"}),
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

    # History-aware query rewrite (multi-turn only; no-op on first turns).
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

    # classify_intent only routes here without a document when the user is
    # asking how uploading works (anything else is rerouted to general_query
    # there, so the reported intent matches the flow that runs).
    if not document_content:
        response = DOCUMENT_UPLOAD_HELP
        return {
            **state,
            "response": response,
            "messages": state["messages"]
            + [{"role": "assistant", "content": response}],
        }

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
            logger.info(
                f"Indian Kanoon found {len(results)} relevant legal references for document"
            )
            return indian_kanoon_tool, results
        except Exception as e:
            logger.warning(f"Indian Kanoon search error in document analysis: {e}")
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
        logger.warning(error_msg)

        # Basic fallback analysis
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
        analysis = await invoke_llm_safely(llm, prompt, stream=True)

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
        rag_section = f"""\n\nAPPLICABLE PROVISIONS (cite the Act named with each section):
{rag_sections_text}"""

    # Compulsory RAG: when RAG failed, instruct LLM not to fabricate sections
    disclaimer_prefix, no_rag_warning = _apply_compulsory_rag_policy(rag_succeeded)

    prompt = CRIME_REPORT_PROMPT.format(
        crime_details=crime_details[:_MAX_QUERY_CHARS],
        identified_crime=identified_crime,
        rag_section=rag_section,
        no_rag_warning=no_rag_warning,
    )

    await emit_text(disclaimer_prefix)
    try:
        final_response = await invoke_llm_safely(llm, prompt, stream=True)
    except Exception as e:
        logger.error("LLM error in crime report: %s", e)
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
        "trace": _merge_trace(
            state,
            crime_report={"crime_type": identified_crime, "rag_succeeded": rag_succeeded},
        ),
    }


async def handle_find_lawyer(state: ChatState) -> ChatState:
    """
    Handle lawyer search requests.
    Finds relevant lawyers based on user needs and location.
    """
    user_input = state["current_input"]
    lawyer_query = (state.get("lawyer_query") or user_input)[:_MAX_QUERY_CHARS]
    tools = state.get("selected_tools") or []

    # Real Postgres-backed recommendation (pgvector semantic search + weighted
    # rating/success_rate score). ChatState has no session plumbing, and this
    # is the only DB access chatbot.py needs, so open one locally rather than
    # threading a Session through the whole graph. recommend_lawyers runs its
    # blocking query in a worker thread; everything that reads the ORM rows
    # happens inside the `with` so none is touched after the session closes.
    from app.db.engine import get_engine
    from sqlmodel import Session as DBSession

    with DBSession(get_engine()) as session:
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

    # select_tools() adds indian_kanoon only when the request names a legal
    # area — purely locational searches ("find a lawyer near me") get no
    # benefit from case-law retrieval.
    legal_context = ""
    if "indian_kanoon" in tools:
        ik_result = await RAG_TOOL_REGISTRY["indian_kanoon"](lawyer_query)
        if ik_result.succeeded:
            docs = ik_result.raw.get("results", [])
            if docs:
                legal_context = "\n\n**Relevant Legal Context:**\n"
                for doc in docs[:2]:
                    legal_context += f"• {doc.title}\n"
                logger.info("Added Indian Kanoon legal context to lawyer search")

    # Enhance with LLM for personalized recommendations
    try:
        llm = get_llm()
        prompt = LAWYER_SEARCH_PROMPT.format(
            query=lawyer_query, lawyer_results=formatted_results
        )
        if legal_context:
            prompt = f"{prompt}\n\n{legal_context}"

        final_response = await invoke_llm_safely(llm, prompt, stream=True)
    except Exception:
        # Use formatted results directly if LLM fails
        final_response = LAWYER_SEARCH_FALLBACK.format(
            formatted_results=formatted_results
        )

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
    """
    Post-generation grounding gate.

    citation_only=True runs ONLY the deterministic statute-citation-existence
    check, never the claim-level grounding pass — used by gq_verify when
    this query's own retrieval failed but generation still produced a real
    answer. Claim-level grounding needs retrieved context text to judge
    claims against (none exists on that path), but citation existence is
    checkable regardless: verify_citations() queries the whole persistent
    corpus index (rag.find_section()), not what this specific query
    retrieved. Always returns report=None in this mode — nothing here
    should ever trigger a regeneration, since there is no better context to
    retarget retrieval at.

    citation_only=False (default, unchanged from before) runs the full
    two-layer gate:

    1. citation_verifier (unchanged): does every 'Section N of the X Act' /
       'Article N' in the answer exist in the indexed corpus under the cited
       act, and was it among what was actually retrieved for this query?
    2. grounding_verifier: goes past the citation token to the claim built
       around it — splits the answer into sentences, checks each cited or
       high-risk-absolute claim against its evidence text, and (only for
       what's flagged) uses one batched LLM call to rewrite the unsupported
       part from evidence alone. Supported sentences are never touched.

    Returns (text, report): the answer with advisory footers from both layers
    (silent when everything checks out) and the GroundingReport the caller
    can act on — or (response_text, None) when the gate was skipped,
    citation_only was requested, or an error occurred. Never raises — a
    verifier bug must not break chat.
    """
    try:
        from app.tools.grounding_verifier import ground_and_correct, grounding_footer
        from app.tools.citation_verifier import verify_citations, verification_footer
        from app.tools.case_citation_verifier import (
            case_verification_footer,
            verify_case_citations,
        )
        from app.tools.unified_legal_rag import get_unified_rag_system
        from app.tools.case_law_rag import get_case_law_rag_system

        rag = get_unified_rag_system()
        case_rag = get_case_law_rag_system()

        if citation_only:
            text = response_text
            if rag.initialized:
                citation_report = verify_citations(response_text, rag, retrieved_sections)
                if citation_report.checks:
                    logger.info(
                        "CitationVerify (no-retrieval path): %s/%s citations verified",
                        len(citation_report.verified),
                        len(citation_report.checks),
                    )
                text += verification_footer(citation_report)
            if case_rag.initialized:
                text += case_verification_footer(verify_case_citations(response_text, case_rag))
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
            text += case_verification_footer(verify_case_citations(corrected_text, case_rag))
        return (
            text,
            report,
        )
    except Exception:
        logger.exception("Citation verification skipped")
        return response_text, None


# ============================================================================
# General-query agentic loop
#
#   gq_plan -> gq_retrieve -> gq_grade -+-> gq_generate -> gq_verify -+-> END
#                  ^                    |                             |
#                  +---- gq_rewrite <---+ (weak/empty retrieval)      |
#                  +-------------------------------------------------+
#                                        (grounding score low: regenerate once)
#
# Both loops are bounded (one retry each) and skipped once the request has
# used its wall-clock budget, so the worst case stays predictable.
# ============================================================================

_QUESTION_SPLIT_RE = re.compile(r"(?<=\?)\s+")
_ENUM_SPLIT_RE = re.compile(r"(?:^|\s)\(?(?:[a-d]|[1-4])[\).]\s+(?=[A-Za-z])")
_MIN_SUBQUESTION_WORDS = 5
_MAX_SUBQUESTIONS = 3
_ENTRY_SPLIT_RE = re.compile(r"\n\n(?=• )")


def decompose_question(text: str) -> List[str]:
    """Split a multi-part question into standalone sub-questions so retrieval
    can search for each part instead of one blended query.

    Deterministic on purpose: a 4B model spends seconds thinking through even
    a trivial decomposition, and this box has none to spare. Only genuinely
    separable parts are returned — a fragment like "On what grounds?" depends
    on the sentence before it, so it is dropped, and a query with no
    separable structure returns [] and retrieval runs on the full text alone.
    The full query is always searched too, so decomposition can only add
    recall."""
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
    """Join formatted context blocks, dropping entries (keyed by their first
    line, e.g. "• **IPC § 420** — Cheating") already seen in an earlier one."""
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
    """Fold several statute_context results (one per query) into one, in
    priority order. Exceptions and empty results are skipped."""
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
            # The retriever's own counts are authoritative (the formatted text
            # can hold fewer entries than chunks after budgeting); a merge must
            # never report fewer provisions than its best input.
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
    """Run the tools chosen by the router (state["selected_tools"]) in
    parallel. The statute search is repeated per sub-question and, on a
    regeneration pass, aimed at the citations the grounding gate could not
    support."""
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

    async def _fast_llm_invoke(prompt: str) -> str:
        return await _invoke_fast_text(prompt, timeout=25.0)

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
                # Widening drops the domain bias; a second-chance search must
                # not be limited by the guess that just under-delivered.
                domain_hint=(
                    ["criminal"] if domain_hint == "criminal" and not widen else None
                ),
                # Only the primary query pays for the LLM query parser.
                fast_llm_invoke=(
                    _fast_llm_invoke if is_primary and not widen else None
                ),
                # Sub-questions add statutes; authorities come from the first
                # query only (each extra case-law hop cost ~10s of CPU reranking).
                with_case_law=i == 0,
            )
    if "indian_kanoon" in tools and "indian_kanoon" not in prior:
        jobs["indian_kanoon"] = RAG_TOOL_REGISTRY["indian_kanoon"](
            retrieval_query, infer_indian_kanoon_context_type(user_input)
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
    if "indian_kanoon" in by_name:
        ik = by_name["indian_kanoon"]
        prior["indian_kanoon"] = ik

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
    """Turn retrieval quality into a decision. Emptiness alone is not enough:
    non-empty but off-topic provisions produce a fluent, confidently-cited
    answer about the wrong law. "weak" = fewer than 3 provisions, or a mean
    reranker score under settings.retrieval_min_confidence."""
    settings = get_settings()
    statute = _tool_result(state, "statute_context")
    confidence = 0.0
    chunk_count = 0
    if statute is None:
        # Statute search wasn't selected; judge on whether anything came back.
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
    sections = sorted((statute.raw or {}).get("retrieved_sections") or ()) if statute else []
    return {
        **state,
        "retrieval_grade": grade,
        "retrieval_confidence": confidence,
        "retrieved_sections": (statute.raw or {}).get("retrieved_sections") if statute else None,
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
    can_retry = (
        (state.get("retrieval_attempts") or 0) < 1
        and _time_elapsed(state) < settings.request_budget_seconds / 3
    )
    if state.get("retrieval_grade") in ("weak", "none") and can_retry:
        return "gq_rewrite"
    return "gq_generate"


async def gq_rewrite(state: ChatState) -> ChatState:
    """Second-chance query for weak/empty retrieval: restate the question as a
    statute-oriented keyword query (falls back to the original on any
    failure — the widened, unfiltered search still runs)."""
    await emit_event(
        "status", stage="retrieval", label="Broadening the search…"
    )
    original = state.get("retrieval_query") or state["current_input"]
    rewritten = original
    try:
        candidate = (
            await _invoke_fast_text(
                STATUTE_QUERY_REWRITE_PROMPT.format(question=original[:1000]),
                timeout=25.0,
            )
        ).strip().strip('"').strip()
        if (
            candidate
            and candidate != _INCOMPLETE_GENERATION_NOTE
            and len(candidate) <= 300
            and "\n" not in candidate
        ):
            rewritten = candidate
    except Exception as e:
        logger.warning("Retrieval rewrite failed (%s) — reusing the original query", e)
    logger.info("Retrieval rewrite: %.120r -> %.120r", original, rewritten)
    return {
        **state,
        "retrieval_query": rewritten,
        "retrieval_attempts": 1,
        "sub_questions": [],
    }


_CONCISE_CONTEXT_TOKENS = 1400  # retry pass: a much smaller context block


def _build_answer_prompt(state: ChatState, *, concise: bool = False) -> tuple:
    """Assemble the grounded prompt for the general-query answer. Returns
    (prompt, retrieved_context).

    concise=True builds the give-up retry variant: statute and case-law blocks
    only (Indian Kanoon excerpts are the lowest-priority block), a fraction of
    the context budget, and an instruction to answer directly. The give-up is
    the model spending its whole budget deliberating inside <think>, so
    re-running the identical prompt is just latency; less to reason over and an
    explicit ask for a short answer is what can change the outcome."""
    user_input = state["current_input"]
    messages = state.get("messages", [])
    regenerating = bool(state.get("regen_pending"))

    conversation_context = ""
    if len(messages) > 1:
        conversation_context = "\n".join(
            f"{m['role'].upper()}: {m['content'][:200]}" for m in messages[-6:]
        )

    statute = _tool_result(state, "statute_context")
    kanoon = _tool_result(state, "indian_kanoon")
    rag_sections_text = statute.context_text if statute else ""
    case_law_text = (statute.raw or {}).get("case_law_text", "") if statute else ""
    indian_kanoon_results = kanoon.context_text if kanoon else ""
    _, prompt_warning = _apply_compulsory_rag_policy(bool(state.get("rag_succeeded")))

    context_parts = []
    if rag_sections_text:
        context_parts.append(
            STATUTE_CONTEXT_BLOCK.format(rag_sections_text=rag_sections_text)
        )
    if case_law_text:
        context_parts.append(CASE_LAW_CONTEXT_BLOCK.format(case_law_text=case_law_text))
    if indian_kanoon_results and not concise:
        context_parts.append(
            INDIAN_KANOON_CONTEXT_BLOCK.format(
                indian_kanoon_results=indian_kanoon_results[:3000]
            )
        )

    from app.metrics.engineering_metrics import count_tokens_approx

    user_input_for_prompt = user_input[:_MAX_QUERY_CHARS]
    feedback = state.get("regen_feedback") if regenerating else None
    # Reserve budget for the fixed scaffolding (instruction template ~500
    # tokens) plus the query, conversation history and any regeneration
    # feedback, then fit the context blocks into whatever input budget remains.
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
        # No retrieved context — tools returned empty. Use the general prompt
        # with extra caution about ungrounded claims.
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


def _prefers_concise(state: ChatState) -> bool:
    """Whether to answer with the concise prompt on the first attempt.

    Only for a simple question that retrieval covered well: graded "good", one
    part, short, and not a multi-offense scenario. Everything else keeps the full
    prompt, since it is depth on those that the concise answer would cost. The
    reason to do it at all: the full prompt gives up (never closes its <think>
    block) on a meaningful share of queries and takes 2+ minutes when it doesn't,
    where the concise one measured ~6x faster (see docs/chatbot-production-readiness.md)."""
    settings = get_settings()
    if not settings.concise_first_enabled:
        return False
    user_input = state["current_input"]
    return (
        state.get("retrieval_grade") == "good"
        and not state.get("sub_questions")
        and len(user_input.split()) <= settings.concise_first_max_query_words
        and _count_keyword_matches(user_input, CRIME_TYPE_KEYWORDS) < 2
    )


def _can_retry_giveup(state: ChatState) -> bool:
    settings = get_settings()
    return (
        settings.llm_giveup_retry_enabled
        and _time_elapsed(state) < settings.llm_giveup_retry_max_elapsed_seconds
    )


async def gq_generate(state: ChatState) -> ChatState:
    """Build the grounded prompt from what retrieval returned and generate,
    streaming to the client. The "grounding unavailable" disclaimer is streamed
    *before* the answer — it is known the moment retrieval returns, and must
    not be something that only arrives with the final event.

    If the model gives up (spends its whole budget thinking and never answers),
    retry once with a trimmed, concise variant of the prompt before showing the
    user a "please try again" note. The give-up note is withheld from the stream
    on the first attempt so the user never sees it when the retry succeeds."""
    regenerating = bool(state.get("regen_pending"))
    disclaimer_prefix, _ = _apply_compulsory_rag_policy(bool(state.get("rag_succeeded")))

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
    try:
        retry_possible = get_settings().llm_giveup_retry_enabled
        concise = _prefers_concise(state)
        prompt, retrieved_context = _build_answer_prompt(state, concise=concise)
        answer = await invoke_llm_safely(
            get_llm(), prompt, stream=True, notify_incomplete=not retry_possible
        )
        if answer == _INCOMPLETE_GENERATION_NOTE and retry_possible:
            if _can_retry_giveup(state):
                logger.warning("Generation gave up — retrying once with a concise prompt")
                await emit_event(
                    "status",
                    stage="retrying",
                    label="That was taking too long to work out — trying a simpler pass…",
                )
                prompt, retrieved_context = _build_answer_prompt(state, concise=True)
                answer = await invoke_llm_safely(get_retry_llm(), prompt, stream=True)
                attempts, retried = 2, True
            else:
                await emit_text(answer)  # no time left: show the note we withheld
        # Still no answer: the text is a canned "please retry" note, not
        # something to fact-check (and a "verified, score 1.0" trace for it
        # would mislead).
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
            },
        ),
    }


def _regeneration_plan(report) -> tuple:
    """(feedback text, targeted retrieval queries) for the flagged claims."""
    flagged = report.confirmed_flagged[:3]
    feedback = "\n".join(
        f"- \"{s.text.strip()[:240]}\"" + (f" — {s.reason}" if s.reason else "")
        for s in flagged
    )
    queries: List[str] = []
    for s in report.confirmed_flagged:
        for citation in s.citations:
            if citation not in queries:
                queries.append(citation)
    return feedback, queries[:3]


async def gq_verify(state: ChatState) -> ChatState:
    """Post-generation grounding gate. Corrects unsupported sentences in
    place; if the answer is still poorly supported it loops back once to
    regenerate with retrieval aimed at the unsupported citations.

    When this query's retrieval failed (rag_succeeded is False) but
    generation still produced a real answer, runs a citation-only pass
    instead of skipping verification altogether — see
    _verify_response_citations(citation_only=True). No regeneration is
    ever triggered on that path: there is no better context to retarget
    retrieval at.
    """
    settings = get_settings()
    response = state.get("response") or ""
    finished = {
        **state,
        "messages": state["messages"] + [{"role": "assistant", "content": response}],
    }

    # Generation itself failed: the text is a canned apology, nothing to
    # check against anything.
    if state.get("error") == "generation_failed":
        return {
            **finished,
            "trace": _merge_trace(state, grounding={"verified": False, "reason": "generation_failed"}),
        }

    citation_only = not state.get("rag_succeeded")

    await emit_event(
        "status", stage="verifying", label="Checking citations against the statutes…"
    )

    async def _grounding_correction_invoke(prompt: str) -> str:
        raw = await invoke_llm_safely(
            get_grounding_correction_llm(), prompt, stream=False
        )
        return strip_reasoning_tags(raw)

    final_text, report = await _verify_response_citations(
        response,
        state.get("retrieved_sections"),
        retrieved_context_text=state.get("retrieved_context") or "",
        llm_invoke=_grounding_correction_invoke,
        citation_only=citation_only,
    )

    score = report.overall_score if report is not None else None
    # Only an *adjudicated* report may trigger a regeneration. The deterministic
    # word-overlap pass alone flags most fluent paraphrases (grounding_footer
    # already refuses to show those to users for the same reason), and a
    # regeneration costs another minute or more of GPU time. citation_only is
    # also checked explicitly (not just relied on via report is None): defense
    # in depth so a future change to _verify_response_citations's return
    # contract can't silently reopen regeneration on the no-retrieval path.
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
            "trace": _merge_trace(state, grounding={**grounding_trace, "regenerating": True}),
        }

    if final_text != response:
        # Corrections and footers land in the client immediately, not only
        # once the terminal event arrives.
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

        logger.info(
            f"[Layer 1] Document classified as: {classification.document_type} "
            f"(confidence: {classification.confidence:.2f})"
        )

        # ================================================================
        # Layer 2: Statutory Checklist Validation (rule-based, no LLM)
        # ================================================================
        validator = get_statutory_validator()
        validation = validator.validate(document_content, classification.document_type)

        logger.info(
            f"[Layer 2] Statutory validation: {validation.passed}/{validation.total_checks} passed, "
            f"compliance score: {format_score(validation.compliance_score)}"
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
                logger.warning(f"Indian Kanoon init error: {e}")
                return None

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

        indian_kanoon, crime_rag, civil_rag = await asyncio.gather(
            init_ik(), init_rag(), init_civil()
        )

        # Get Indian law context via RAG tool
        law_rag = get_indian_law_rag(indian_kanoon, crime_rag, civil_rag=civil_rag)
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

        logger.info(
            f"[Layer 3] Analysis complete. Defects: {result['defect_count']}, "
            f"Compliance: {format_score(result['compliance_score'])}"
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
        logger.exception("Document validation error: %s", e)

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


async def handle_clarification(state: ChatState) -> ChatState:
    """Ask the one clarifying question classify_intent put in state["response"]."""
    response = state.get("response") or CLARIFY_GENERIC
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


# ============================================================================
# Graph Builder
# ============================================================================


def build_legal_chatbot_graph() -> StateGraph:
    """
    Build the LangGraph workflow for the legal chatbot. The one graph serves
    both /api/chat and /api/chat/stream — streaming is a contextvar side
    channel (see invoke_llm_safely / emit_event), not a second code path.

    START -> classify_intent -> [route_by_intent] -+-> document_analysis -> END
                                                   +-> crime_report     -> END
                                                   +-> find_lawyer      -> END
                                                   +-> non_legal        -> END
                                                   +-> clarify          -> END
                                                   +-> general_query loop:
       gq_plan -> gq_retrieve -> gq_grade -> gq_generate -> gq_verify -> END
                     ^              | weak/empty            | score low
                     +-- gq_rewrite-+                        |
                     +---------------------------------------+ (regenerate once)
    """
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


# ============================================================================
# Chatbot Class
# ============================================================================

_MAX_SESSION_MESSAGES = 20  # live conversation window per session


def _append_capped(left: Optional[List[Message]], right: Optional[List[Message]]) -> List[Message]:
    return ((left or []) + (right or []))[-_MAX_SESSION_MESSAGES:]


class SessionState(TypedDict):
    """The only state that is checkpointed: conversation history.

    The per-turn graph's own state (tool results holding retriever objects and
    sets, up-to-10MB uploaded document text, exceptions) never reaches the
    checkpointer — LangGraph's serializer fails on the first arbitrary object in
    a tool's `raw` payload and flattens exceptions to strings. Keeping history
    in a small outer graph means only plain, JSON-safe messages are persisted."""

    messages: Annotated[List[Message], _append_capped]


@dataclass
class _TurnContext:
    """Per-turn inputs and output, passed through a contextvar rather than graph
    state so none of it is checkpointed (see SessionState)."""

    session_id: str
    started_at: float
    document_content: Optional[str] = None
    document_type: Optional[str] = None
    result: Optional[Dict[str, Any]] = None


_turn_var: contextvars.ContextVar[Optional[_TurnContext]] = contextvars.ContextVar(
    "turn_context", default=None
)


class LegalChatbot:
    """
    Main chatbot class that wraps the LangGraph workflow.
    Provides a clean interface for the API layer.

    Two graphs: `self._turn_graph` is the workflow (classify -> handlers, see
    build_legal_chatbot_graph), compiled with checkpointer=False so its state is
    never persisted — and so it cannot inherit the outer graph's checkpointer,
    which a subgraph compiled with checkpointer=None would. `self.graph` is a
    one-node outer graph whose only state is the message history; it is compiled
    with the checkpointer, keyed by thread_id == the session's memory key.
    """

    def __init__(self, checkpointer: Optional[BaseCheckpointSaver] = None):
        self._checkpointer = checkpointer if checkpointer is not None else InMemorySaver()
        self._turn_graph = build_legal_chatbot_graph().compile(checkpointer=False)

        session = StateGraph(SessionState)
        session.add_node("run_turn", self._run_turn)
        session.set_entry_point("run_turn")
        session.add_edge("run_turn", END)
        self.graph = session.compile(checkpointer=self._checkpointer)

        self._active_stream_tasks: Dict[str, asyncio.Task] = {}
        self._in_flight = 0
        # Only used to bound the in-process fallback saver; Postgres threads are
        # reaped by app.jobs.chat_threads instead.
        self._last_seen: Dict[str, float] = {}

    # -- concurrency ----------------------------------------------------------

    def _acquire_slot(self) -> None:
        # One GPU serves every request; past this many concurrent chats the
        # extra ones would only queue behind Ollama, so refuse them quickly.
        if self._in_flight >= get_settings().chat_max_concurrent:
            raise ChatBusyError("The assistant is busy right now. Please try again shortly.")
        self._in_flight += 1

    def _release_slot(self) -> None:
        self._in_flight = max(0, self._in_flight - 1)

    # -- session memory (checkpoint-backed) -----------------------------------

    def _config(self, session_id: str, touch: bool = True) -> Dict[str, Any]:
        """thread_id is the memory key, which already carries the account scope
        (`user:<id>:<session>` / `guest:<session>`), so one account cannot read
        another's conversation by guessing a session id. Only writes count as
        activity (touch=True)."""
        if touch:
            self._touch(session_id)
        return {
            "configurable": {"thread_id": session_id},
            # Stamped into every checkpoint; idle-thread cleanup reads it.
            "metadata": {"last_active": datetime.now(timezone.utc).isoformat()},
        }

    def _touch(self, session_id: str) -> None:
        if not isinstance(self._checkpointer, InMemorySaver):
            return
        settings = get_settings()
        now = time.monotonic()
        self._last_seen[session_id] = now
        stale = [
            sid for sid, seen in self._last_seen.items()
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
        """
        Prime a thread from DB-loaded history, but only if it has none yet
        (avoids clobbering an active conversation with a stale DB read). Called
        by the chat router for an authenticated user whose thread has no
        checkpoint (a session that predates checkpointing, or whose idle thread
        was cleaned up) — chat_messages is the durable transcript.
        """
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
        """Forget a session's history and stop any in-flight generation for it
        (otherwise it would keep running and write into a session the user just
        cleared)."""
        self.stop_stream(session_id)
        self._last_seen.pop(session_id, None)
        await self._checkpointer.adelete_thread(session_id)

    # -- the turn -------------------------------------------------------------

    async def _run_turn(self, state: SessionState) -> Dict[str, Any]:
        """The outer graph's only node: run the workflow for this turn. The last
        message is the user turn that was just appended; everything before it is
        history."""
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
        return {"messages": [{"role": "assistant", "content": response}]} if response else {}

    async def stream_chat(
        self,
        message: str,
        session_id: str = "default",
        document_content: Optional[str] = None,
        document_type: Optional[str] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream a chat turn (see _stream_chat), holding a concurrency slot
        for its whole lifetime — including when the client disconnects."""
        self._acquire_slot()
        inner = self._stream_chat(message, session_id, document_content, document_type)
        try:
            async for event in inner:
                yield event
        finally:
            # Close the inner generator deterministically: its `finally` is
            # what cancels the graph task when the client has gone away.
            await inner.aclose()
            self._release_slot()

    async def _stream_chat(
        self,
        message: str,
        session_id: str = "default",
        document_content: Optional[str] = None,
        document_type: Optional[str] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream chat response token by token.

        Runs the same compiled graph as chat(); the graph's nodes push tokens
        and progress events into a queue (see invoke_llm_safely / emit_event)
        that this generator relays. Yields dicts:
          {"type": "token", "content": "..."}     answer text
          {"type": "status", "stage", "label"}    progress for the UI
          {"type": "reset"}                       discard text streamed so far
          {"type": "replace", "content": "..."}   corrected full text
          {"type": "done"|"stopped", ...}         terminal, with metadata
        """
        request_id_var.set(uuid.uuid4().hex[:8])
        # Multilingual layer. Translate the query to English up front so
        # routing/retrieval/reasoning run in English and memory stays canonical.
        # Hybrid streaming: English replies stream token by token; a
        # non-English reply cannot stream because it is translated only after
        # the full English answer exists, so for those we suppress per-token
        # output and emit one translated message at the end.
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
                await queue.put(None)  # Signal completion

        task = asyncio.create_task(run_graph())
        # A new stream supersedes any still-running one for this session — cancel
        # the old task first so it isn't orphaned (unstoppable, still burning
        # LLM compute) by the dict overwrite below.
        previous = self._active_stream_tasks.get(session_id)
        if previous is not None and not previous.done():
            previous.cancel()
        self._active_stream_tasks[session_id] = task

        accumulated = ""
        intent_seen: Optional[str] = None
        stopped = False
        superseded = False
        try:
            # Relay tokens and events as they arrive. For a non-English reply
            # we still drain the queue (to accumulate the full English answer)
            # but do not emit per-token — the client receives one translated
            # message after generation completes.
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

            # Wait for the graph to complete and get its result
            try:
                result = await task
            except asyncio.CancelledError:
                # stop_stream() cancelled the graph mid-generation — the
                # tokens already yielded above are everything the user saw.
                stopped = True
                result = {"intent": intent_seen, "response": accumulated}
        finally:
            if self._active_stream_tasks.get(session_id) is task:
                self._active_stream_tasks.pop(session_id, None)
                # Reached without the task finishing means the consumer went
                # away (client disconnect closes this generator). Deregistering
                # alone would leave the graph generating with nothing able to
                # stop it, so cancel it here.
                if not task.done():
                    task.cancel()
            else:
                # A newer stream_chat() call for this session_id superseded
                # us (and cancelled us) before we finished — don't let our
                # stale partial response land in history after the newer,
                # already-completed turn.
                superseded = True

        # English answer (canonical) — from the graph, or accumulated tokens.
        english_text = result.get("response", "") or accumulated
        intent = result.get("intent") or intent_seen

        # A completed turn's reply was already checkpointed by the graph. A
        # stopped one never got that far, so save the partial text the user saw
        # as the assistant turn (keeps conversation context coherent).
        if stopped and english_text and not superseded:
            await self._append_assistant(session_id, english_text)

        # Client-facing text: translated for non-English, else the English
        # answer. For English replies that streamed token-by-token, the client
        # already has the text; only the non-streamed/non-English cases need a
        # full-text token emission below.
        if translate_out:
            response_text = await postprocess_response(english_text, lang)
            if response_text:
                yield {"type": "token", "content": response_text}
        else:
            response_text = english_text
            if not tokens_streamed and response_text:
                yield {"type": "token", "content": response_text}

        if superseded:
            # A newer stream_chat() call for this session_id already started
            # (or finished) before we did — our result is stale. Emitting a
            # normal "stopped"/"done" event here would make the router
            # persist this superseded partial into the transcript, possibly
            # landing it in the DB after the newer, already-completed turn.
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

        # Yield completion event with metadata
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
        """Cancel an in-flight stream_chat() generation for this session, if
        any. Called from the /api/chat/stream/stop endpoint (the Stop button)."""
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
            return await self._chat(message, session_id, document_content, document_type)
        finally:
            self._release_slot()

    async def _chat(
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
        request_id_var.set(uuid.uuid4().hex[:8])
        # Multilingual layer (no-op for English / when disabled): detect the
        # input language and translate the query to English so the entire
        # downstream pipeline — routing, retrieval, reasoning — runs in English.
        # Conversation memory therefore stays canonical-English.
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

        # The reply is already in the checkpoint (English canonical — memory is
        # language-independent); only the client-facing copy is translated.
        english_response = result.get("response")

        # Translate the final answer back into the user's language (no-op for
        # English / when disabled). Falls back to English text on failure.
        display_response = (
            await postprocess_response(english_response or "", lang)
            or "I'm sorry, I couldn't process your request."
        )

        # Return structured response. response_en/query_en are the canonical
        # English texts for language-independent persistence; response is the
        # user-facing (possibly translated) text.
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


# Singleton instance
_chatbot: Optional[LegalChatbot] = None


def get_chatbot() -> LegalChatbot:
    """Get or create the chatbot instance, bound to the app's checkpointer
    (Postgres once app startup has initialised it; in-memory otherwise)."""
    global _chatbot
    if _chatbot is None:
        _chatbot = LegalChatbot(get_checkpointer())
    return _chatbot
