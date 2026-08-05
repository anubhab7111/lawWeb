"""
Centralized tool invocation for the legal chatbot.

Every handler in app.chatbot that needs case law, statutes, IPC/BNS sections,
or lawyer results goes through the functions here instead of calling the
underlying RAG/API classes ad hoc. Having one implementation per tool means
a bug fixed here (or a call-signature change in the underlying tool) is
fixed everywhere at once.
"""

from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, Optional

from app.tools.base_legal_rag import compress_chunks_for_context
from app.tools.indian_kanoon import get_indian_kanoon_tool


@dataclass
class ToolInvocationResult:
    """Uniform result shape returned by every invoke_* function below."""

    name: str
    succeeded: bool  # True when non-empty grounding content was retrieved
    context_text: str  # pre-formatted text, ready to splice into a prompt
    raw: Any = None  # structured payload for callers needing more than text


# ============================================================================
# Indian Kanoon (case law / precedent search)
# ============================================================================

_IK_CONTEXT_TYPE_KEYWORDS = (
    (
        "constitution",
        (
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
        ),
    ),
    (
        "crpc",
        (
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
        ),
    ),
    (
        "statute",
        (
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
            "evidence",
            "admissible",
            "whatsapp",
            "electronic record",
            "certificate",
            "witness",
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
        ),
    ),
)


def infer_indian_kanoon_context_type(text: str) -> str:
    """Heuristic: pick the Indian Kanoon search context that best matches the query."""
    text_lower = text.lower()
    if "ipc" in text_lower or "penal code" in text_lower:
        return "ipc"
    if "crpc" in text_lower or "criminal procedure" in text_lower:
        return "crpc"
    for context_type, keywords in _IK_CONTEXT_TYPE_KEYWORDS:
        if any(kw in text_lower for kw in keywords):
            return context_type
    return "general"


async def invoke_indian_kanoon(
    query: str, context_type: str = "general"
) -> ToolInvocationResult:
    """Fetch case law / precedents from Indian Kanoon."""
    try:
        ik_tool = get_indian_kanoon_tool()
        await ik_tool.initialize()
        result = await ik_tool.answer_legal_query(query, context_type)
        formatted = result.get("formatted_results", "")
        return ToolInvocationResult(
            name="indian_kanoon",
            succeeded=bool(formatted),
            context_text=formatted,
            raw=result,
        )
    except Exception as e:
        print(f"Indian Kanoon error: {e}")
        return ToolInvocationResult(
            name="indian_kanoon", succeeded=False, context_text=""
        )


# ============================================================================
# Unified statute + case-law retrieval
# ============================================================================


def _format_case_law(cases, max_chars: int = 4000, per_case_chars: int = 1200) -> str:
    """Format curated landmark judgments in authority order for the prompt."""
    parts = []
    used = 0
    for c in cases:
        entry = (
            f"• **{c.case_name}** ({c.citation or c.court}, {c.date[:4] if c.date else '?'})"
            f" — {c.court}\n{c.text[:per_case_chars]}"
        )
        if used + len(entry) > max_chars and parts:
            break
        parts.append(entry)
        used += len(entry)
    return "\n\n".join(parts)


def _budget_context(chunks, max_chars: int = 12000, per_chunk_chars: int = 2500) -> str:
    """
    Format retrieved statute chunks for the prompt in rerank order, filling
    a fixed character budget (≈3k tokens inside the 6k num_ctx window)
    instead of blindly truncating every chunk to a few hundred characters.
    """
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
    fast_llm_invoke: Optional[Callable[[str], Awaitable[str]]] = None,
) -> ToolInvocationResult:
    """
    Understanding-first statute + case-law retrieval: parse the legal
    question, fill via the unified hybrid index, then hop to the curated
    landmark-judgment corpus for cases interpreting those same provisions.

    context_text carries the statute text; raw["case_law_text"] carries the
    paired case-law text (kept separate since handlers give each its own
    prompt framing).
    """
    try:
        from app.tools.legal_retrieval import retrieve_case_law, retrieve_statutes

        context, parsed = await retrieve_statutes(
            query,
            k=k,
            domains_hint=domain_hint,
            llm_invoke=fast_llm_invoke,
        )
        if not context.chunks:
            return ToolInvocationResult(
                name="statute_context",
                succeeded=False,
                context_text="",
                raw={"case_law_text": ""},
            )

        compressed = await compress_chunks_for_context(query, context.chunks)
        text = _budget_context(compressed)
        print(
            f"Unified RAG: {len(context.chunks)} provisions retrieved "
            f"(confidence: {context.confidence:.2%}): "
            f"{[f'{c.act_name} §{c.section_number}' for c in context.chunks]}"
        )

        case_text = ""
        try:
            cases = await retrieve_case_law(query, parsed, context.chunks)
            if cases:
                case_text = _format_case_law(cases)
                print(
                    f"Case law: {len(cases)} judgments retrieved: "
                    f"{[c.case_name for c in cases]}"
                )
        except Exception as e:
            print(f"Case law lookup error: {e}")

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
            },
        )
    except Exception as e:
        print(f"Unified RAG lookup error: {e}")
        import traceback

        traceback.print_exc()
        return ToolInvocationResult(
            name="statute_context",
            succeeded=False,
            context_text="",
            raw={"case_law_text": ""},
        )


# ============================================================================
# Criminal RAG (IPC/BNS section retrieval for crime reports)
# ============================================================================


async def invoke_crime_sections(
    query: str, crime_type: str = "", k: int = 2
) -> ToolInvocationResult:
    """Fetch IPC/BNS sections + punishments for a described crime."""
    try:
        from app.tools.criminal_rag import extract_crime_features, get_criminal_rag_system

        rag_system = get_criminal_rag_system()
        await rag_system.initialize()

        if not rag_system.initialized:
            return ToolInvocationResult(
                name="crime_sections", succeeded=False, context_text=""
            )

        features = extract_crime_features(query)
        print(
            f"Crime features: violence={features.violence}, death={features.death}, "
            f"weapon={features.weapon}, intent={features.intent}, "
            f"property={features.property_loss}, trespass={features.trespass}, "
            f"threat={features.threat}"
        )

        rag_result = await rag_system.retrieve_sections(
            query, crime_type=crime_type, features=features, k=k
        )

        if not rag_result.ipc_sections:
            return ToolInvocationResult(
                name="crime_sections", succeeded=False, context_text="", raw=rag_result
            )

        section_lines = [
            f"• IPC Section {match.section} ({match.title})\n  Punishment: {match.punishment}"
            for match in rag_result.ipc_sections
        ]
        sections_text = "\n".join(section_lines)
        print(
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
        print(f"RAG lookup error (non-critical): {e}")
        import traceback

        traceback.print_exc()
        return ToolInvocationResult(name="crime_sections", succeeded=False, context_text="")


# ============================================================================
# Bare Act Explorer (bonus chatbot reachability — primary UX is the
# standalone /api/bare-acts router; see app/tools/bare_act_explorer.py)
# ============================================================================


async def invoke_bare_act_lookup(query: str) -> ToolInvocationResult:
    try:
        from app.tools.bare_act_explorer import explore_bare_act

        result = await explore_bare_act(query, explain=False)
        if not result.matches:
            return ToolInvocationResult(name="bare_act_lookup", succeeded=False, context_text="")

        text = _budget_context(result.matches)
        return ToolInvocationResult(
            name="bare_act_lookup", succeeded=True, context_text=text, raw=result
        )
    except Exception as e:
        print(f"Bare act lookup error: {e}")
        return ToolInvocationResult(name="bare_act_lookup", succeeded=False, context_text="")


RAG_TOOL_REGISTRY: Dict[str, Callable] = {
    "indian_kanoon": invoke_indian_kanoon,
    "statute_context": invoke_statute_context,
    "crime_sections": invoke_crime_sections,
    "bare_act_lookup": invoke_bare_act_lookup,
}
