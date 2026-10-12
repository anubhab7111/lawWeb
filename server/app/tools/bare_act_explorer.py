"""Bare Act Explorer — "Section 302" or "arrest without warrant" -> section text, landmark judgments, plain-language AI explanation."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional

from app.config import get_settings
from app.tools.base_legal_rag import LegalChunk
from app.tools.case_law_rag import CaseRecord, get_case_law_rag_system
from app.tools.unified_legal_rag import get_unified_rag_system
from cachetools import TTLCache

_SECTION_QUERY_RE = re.compile(
    r"^\s*(?:section|sec\.?|§|article|art\.?)\s+(\d{1,4}[A-Za-z]{0,3})\s*$",
    re.IGNORECASE,
)


@dataclass
class BareActResult:
    query: str
    is_section_lookup: bool
    matches: List[LegalChunk] = field(default_factory=list)
    landmark_judgments: List[CaseRecord] = field(default_factory=list)
    explanation: Optional[str] = None
    ambiguous: bool = False


_explanation_cache = TTLCache(maxsize=512, ttl=get_settings().cache_ttl_seconds)


def _parse_section_query(query: str) -> Optional[str]:
    m = _SECTION_QUERY_RE.match(query.strip())
    return m.group(1) if m else None


async def explore_bare_act(
    query: str,
    act_hint: str = "",
    k: int = 5,
    explain: bool = True,
) -> BareActResult:
    section = _parse_section_query(query)
    rag = get_unified_rag_system()
    await rag.initialize()

    if section is not None:
        matches = rag.find_section(act_hint, section, max_parts=5)

        distinct_acts = {c.act_name for c in matches}
        result = BareActResult(
            query=query,
            is_section_lookup=True,
            matches=matches,
            ambiguous=len(distinct_acts) > 1 and not act_hint,
        )
    else:
        context = await rag.retrieve(query, k=k)
        result = BareActResult(
            query=query, is_section_lookup=False, matches=context.chunks
        )

    if result.matches:
        top = result.matches[0]
        try:
            case_rag = get_case_law_rag_system()
            if await case_rag.initialize():
                result.landmark_judgments = await case_rag.retrieve(
                    f"{top.act_name} section {top.section_number}",
                    k=3,
                    boost_statutes=[(top.act_name, top.section_number)],
                )
        except Exception as e:
            print(f"[BareActExplorer] landmark judgment lookup failed: {e}")

        if explain:
            result.explanation = await _explain(top)

    return result


async def _explain(chunk: LegalChunk) -> str:
    cache_key = f"{chunk.act_name}::{chunk.section_number}"
    cached = _explanation_cache.get(cache_key)
    if cached is not None:
        return cached

    from app.chatbot import get_fast_llm_prose, invoke_llm_safely, strip_reasoning_tags

    prompt = (
        "Explain the following statutory provision in plain, simple language "
        "for someone with no legal background. 3-5 sentences. Do not invent "
        "facts beyond the text given.\n\n"
        f"{chunk.act_name}, Section {chunk.section_number} — {chunk.title}\n"
        f"{chunk.text[:2000]}"
    )
    try:
        explanation = await invoke_llm_safely(get_fast_llm_prose(), prompt)
        explanation = strip_reasoning_tags(explanation or "")
    except Exception as e:
        print(f"[BareActExplorer] explanation LLM call failed: {e}")
        explanation = ""

    if explanation:
        _explanation_cache[cache_key] = explanation
    return explanation
