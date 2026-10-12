"""Centralized tool invocation for the legal chatbot."""

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from app.tools.base_legal_rag import compress_chunks_for_context

logger = logging.getLogger(__name__)


@dataclass
class ToolInvocationResult:
    """Uniform result shape returned by every invoke_* function below."""

    name: str
    succeeded: bool
    context_text: str
    raw: Any = None


def _format_case_law(cases, max_chars: int = 4000, per_case_chars: int = 1200) -> str:
    """Format curated landmark judgments in authority order for the prompt."""
    parts = []
    used = 0
    for c in cases:
        citation = (c.citation or "").split(",")[0].strip()
        entry = (
            f"• **{c.case_name}** ({citation or c.court}, {c.date[:4] if c.date else '?'})"
            f" — {c.court}\n{c.text[:per_case_chars]}"
        )
        if used + len(entry) > max_chars and parts:
            break
        parts.append(entry)
        used += len(entry)
    return "\n\n".join(parts)


def _format_passage(p, max_chars: int = 900) -> str:
    when = p.citation or str(p.year)
    return (
        f"• **{p.case_title}** ({when}) — Supreme Court of India\n{p.text[:max_chars]}"
    )


_RESERVED_LANDMARKS = 2


async def _reranked_case_law_text(query, cases, passages, max_chars: int = 5000) -> str:
    """Landmark cases and judgment passages ranked together by the cross-encoder, so the prompt gets the most relevant authority whichever index found it."""
    from app.tools.base_legal_rag import _get_shared_reranker, cached_predict
    from app.tools.case_law_rag import rerank_header, rerank_source
    from app.tools.unified_rerank import Candidate, rerank

    candidates = [
        Candidate(
            "case_law", c.case_id, rerank_source(c), header=rerank_header(c), payload=c
        )
        for c in cases
    ] + [
        Candidate(
            "judgment",
            p.chunk_id,
            p.text,
            header=p.case_title,
            payload=p,
            group=p.doc_id,
        )
        for p in passages
    ]
    reranker = await _get_shared_reranker()
    if reranker is not None:
        loop = asyncio.get_event_loop()
        predict = lambda pairs: cached_predict(reranker, pairs)  # noqa: E731
        ranked = await loop.run_in_executor(
            None,
            lambda: rerank(
                query,
                candidates,
                predict,
                top_k=6,
                min_relative=0.15,
                per_source_cap={"judgment": 4, "case_law": 3},
            ),
        )
    else:
        ranked = candidates[:6]

    reserved = [c for c in candidates if c.source == "case_law"][:_RESERVED_LANDMARKS]
    for c in reserved:
        if c not in ranked:
            ranked.append(c)
    while len(ranked) > 6 and any(c.source == "judgment" for c in ranked):
        ranked.remove(next(c for c in reversed(ranked) if c.source == "judgment"))

    parts, used = [], 0
    for c in ranked:
        if c.source == "case_law":
            entry = _format_case_law([c.payload], per_case_chars=700)
        else:
            entry = _format_passage(c.payload, max_chars=700)
        if used + len(entry) > max_chars and parts:
            continue
        parts.append(entry)
        used += len(entry)
    return "\n\n".join(parts)


def _budget_context(chunks, max_chars: int = 12000, per_chunk_chars: int = 2500) -> str:
    """Format retrieved statute chunks for the prompt in rerank order, filling a fixed character budget (≈3k tokens inside the 6k num_ctx window) instead of blindly truncating every..."""
    parts = []
    used = 0
    for chunk in chunks:
        sec = chunk.section_number
        sec_label = sec if sec.lower().startswith("article") else f"§ {sec}"
        text = chunk.text[:per_chunk_chars]
        entry = (
            f"• **{chunk.act_name} {sec_label}** — {chunk.title} "
            f"[{chunk.domain}]\n{text}"
        )
        if used + len(entry) > max_chars and parts:
            break
        parts.append(entry)
        used += len(entry)
    return "\n\n".join(parts)


async def invoke_statute_context(
    query: str,
    k: int = 8,
    domain_hint: Optional[list] = None,
    doctrine_assist: bool = False,
    with_case_law: bool = True,
) -> ToolInvocationResult:
    """Understanding-first statute + case-law retrieval: parse the legal question, fill via the unified hybrid index, then hop to the curated landmark-judgment corpus for cases..."""
    try:
        from app.tools.legal_retrieval import (
            retrieve_case_law,
            retrieve_judgment_passages,
            retrieve_statutes,
        )

        context, parsed = await retrieve_statutes(
            query,
            k=k,
            domains_hint=domain_hint,
            doctrine_assist=doctrine_assist,
        )
        if not context.chunks:
            return ToolInvocationResult(
                name="statute_context",
                succeeded=False,
                context_text="",
                raw={"case_law_text": "", "confidence": 0.0, "chunk_count": 0},
            )

        compressed = await compress_chunks_for_context(query, context.chunks)
        text = _budget_context(compressed)
        logger.info(
            f"Unified RAG: {len(context.chunks)} provisions retrieved "
            f"(confidence: {context.confidence:.2%}): "
            f"{[f'{c.act_name} §{c.section_number}' for c in context.chunks]}"
        )

        case_text = ""
        cases, passages = [], []
        if with_case_law:
            cases, passages = await asyncio.gather(
                retrieve_case_law(query, parsed, context.chunks),
                retrieve_judgment_passages(query, context.chunks),
                return_exceptions=True,
            )
            if isinstance(cases, BaseException):
                logger.warning(f"Case law lookup error: {cases}")
                cases = []
            if isinstance(passages, BaseException):
                logger.warning(f"Judgment passage lookup error: {passages}")
                passages = []
        if cases or passages:
            case_text = await _reranked_case_law_text(query, cases, passages)
            logger.info(
                f"Case law: {len(cases)} landmark cases + {len(passages)} judgment passages retrieved"
            )

        retrieved_sections = {
            c.section_number.replace("Article", "").replace("§", "").strip().upper()
            for c in context.chunks
            if not c.section_number.startswith("part ")
        }

        return ToolInvocationResult(
            name="statute_context",
            succeeded=True,
            context_text=text,
            raw={
                "case_law_text": case_text,
                "retrieved_sections": retrieved_sections,
                "confidence": context.confidence,
                "chunk_count": len(context.chunks),
            },
        )
    except Exception as e:
        logger.warning(f"Unified RAG lookup error: {e}")
        import traceback

        traceback.print_exc()
        return ToolInvocationResult(
            name="statute_context",
            succeeded=False,
            context_text="",
            raw={"case_law_text": ""},
        )


async def invoke_crime_sections(
    query: str, crime_type: str = "", k: int = 2
) -> ToolInvocationResult:
    """Fetch IPC/BNS sections + punishments for a described crime."""
    try:
        from app.tools.criminal_rag import (
            extract_crime_features,
            get_criminal_rag_system,
        )

        rag_system = get_criminal_rag_system()
        await rag_system.initialize()

        if not rag_system.initialized:
            return ToolInvocationResult(
                name="crime_sections", succeeded=False, context_text=""
            )

        features = extract_crime_features(query)
        logger.info(
            f"Crime features: violence={features.violence}, death={features.death}, "
            f"weapon={features.weapon}, intent={features.intent}, "
            f"property={features.property_loss}, trespass={features.trespass}, "
            f"threat={features.threat}"
        )

        rag_result = await rag_system.retrieve_sections(
            query, crime_type=crime_type, features=features, k=k, offences_only=True
        )

        if not rag_result.ipc_sections:
            return ToolInvocationResult(
                name="crime_sections", succeeded=False, context_text="", raw=rag_result
            )

        section_lines = [
            f"• {match.act_name} § {match.section} ({match.title})\n  Punishment: {match.punishment}"
            for match in rag_result.ipc_sections
        ]
        sections_text = "\n".join(section_lines)
        logger.info(
            f"RAG retrieved {len(rag_result.ipc_sections)} IPC sections for '{crime_type}' "
            f"(avg confidence: {rag_result.confidence:.0%}, "
            f"sections: {[m.section for m in rag_result.ipc_sections]})"
        )
        return ToolInvocationResult(
            name="crime_sections",
            succeeded=True,
            context_text=sections_text,
            raw=rag_result,
        )
    except Exception as e:
        logger.warning(f"RAG lookup error (non-critical): {e}")
        import traceback

        traceback.print_exc()
        return ToolInvocationResult(
            name="crime_sections", succeeded=False, context_text=""
        )


async def invoke_bare_act_lookup(query: str) -> ToolInvocationResult:
    try:
        from app.tools.bare_act_explorer import explore_bare_act

        result = await explore_bare_act(query, explain=False)
        if not result.matches:
            return ToolInvocationResult(
                name="bare_act_lookup", succeeded=False, context_text=""
            )

        text = _budget_context(result.matches)
        return ToolInvocationResult(
            name="bare_act_lookup", succeeded=True, context_text=text, raw=result
        )
    except Exception as e:
        logger.warning(f"Bare act lookup error: {e}")
        return ToolInvocationResult(
            name="bare_act_lookup", succeeded=False, context_text=""
        )


INTENT_TOOL_MAP: Dict[str, List[str]] = {
    "document_analysis": [],
    "crime_report": ["crime_sections"],
    "general_query": ["statute_context"],
    "find_lawyer": ["lawyer_recommender"],
    "non_legal": [],
}


def select_tools(intent: str, query: str) -> List[str]:
    """Return the deterministic tool policy for one request."""
    return list(INTENT_TOOL_MAP.get(intent, []))


RAG_TOOL_REGISTRY: Dict[str, Callable] = {
    "statute_context": invoke_statute_context,
    "crime_sections": invoke_crime_sections,
    "bare_act_lookup": invoke_bare_act_lookup,
}
