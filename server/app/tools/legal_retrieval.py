"""
legal_retrieval.py — adaptive statute retrieval with legal query understanding.

The single entry point `retrieve_statutes()` used by both the chatbot and
the retrieval evaluation:

  1. Parse the query (citations + doctrine ontology, optional fast-LLM assist)
  2. Deterministically fetch PINNED sections (known doctrine law — no
     embedding roulette)
  3. Fill the remaining slots via the hybrid index (BM25 + dense + reranker)
  4. Merge, dedupe by (act, section), pins first

query_type shapes the mix:
  - section_lookup: pins only (plus hybrid only if pins missed)
  - doctrine:       pins first, hybrid fills the rest
  - comparison:     pins from both codes, small hybrid fill
  - general:        hybrid as before
"""

from __future__ import annotations

import re
from typing import Awaitable, Callable, List, Optional, Tuple

from app.tools.base_legal_rag import LegalChunk, LegalContext
from app.tools.case_law_rag import CaseRecord, get_case_law_rag_system
from app.tools.legal_query_parser import (
    ParsedLegalQuery,
    parse_legal_query,
    parse_legal_query_llm,
)
from app.tools.unified_legal_rag import get_unified_rag_system


_SECTION_REF_RE = re.compile(r"\bsections?\s+(\d{1,4}[A-Z]{0,2})\b", re.IGNORECASE)


def _related_sections(chunk: LegalChunk, max_refs: int = 3) -> List[str]:
    """
    Sections related to a provision within the SAME act: numeric neighbours
    (statutes are drafted so adjacent sections form one scheme, e.g.
    §437/§438/§439 bail) plus explicit cross-references in the text
    ("subject to section N", "notwithstanding section N").
    """
    out: List[str] = []
    sec = chunk.section_number.lower().replace("article", "").strip()
    if sec.isdigit():
        n = int(sec)
        out.extend(str(m) for m in (n - 1, n + 1) if m > 0)
    self_norm = sec.upper()
    for m in _SECTION_REF_RE.finditer(chunk.text[:2500]):
        ref = m.group(1).upper()
        if ref != self_norm and ref not in out:
            out.append(ref)
        if len(out) >= 2 + max_refs:
            break
    return out


def _expand_context(
    rag, merged: List[LegalChunk], max_related: int = 4
) -> List[LegalChunk]:
    """Attach neighbour/cross-referenced sections of the top hits."""
    seen = {(c.act_name, c.section_number) for c in merged}
    related: List[LegalChunk] = []
    for chunk in merged[:3]:
        if len(related) >= max_related:
            break
        for sec in _related_sections(chunk):
            if len(related) >= max_related:
                break
            for rc in rag.find_section(chunk.act_name, sec, max_parts=1):
                key = (rc.act_name, rc.section_number)
                if key in seen:
                    continue
                seen.add(key)
                related.append(
                    LegalChunk(
                        chunk_id=rc.chunk_id,
                        domain=rc.domain,
                        act_name=rc.act_name,
                        section_number=rc.section_number,
                        title=rc.title,
                        text=rc.text,
                        source_file=rc.source_file,
                        has_punishment=rc.has_punishment,
                        domains=list(rc.domains),
                        score=0.3,  # supporting context, not a primary hit
                    )
                )
    return merged + related


def _dedupe_by_section(chunks: List[LegalChunk]) -> List[LegalChunk]:
    seen: set = set()
    out: List[LegalChunk] = []
    for c in chunks:
        key = (c.act_name, c.section_number)
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


def _drop_prose(chunks: List[LegalChunk]) -> List[LegalChunk]:
    """Exclude prose sliding-window chunks (``section_number`` like "part 3").
    These are notifications/commentary, not citable statute sections, and can
    outrank real provisions — especially in the unfiltered confidence-gate retry.
    Pinned sections never use the "part N" format, so they are unaffected."""
    return [c for c in chunks if not c.section_number.startswith("part ")]


async def retrieve_statutes(
    query: str,
    k: int = 8,
    domains_hint: Optional[List[str]] = None,
    llm_invoke: Optional[Callable[[str], Awaitable[str]]] = None,
) -> Tuple[LegalContext, ParsedLegalQuery]:
    """
    Understanding-first statute retrieval over the unified hybrid index.

    Returns (LegalContext, ParsedLegalQuery). Pinned chunks carry score 1.0
    and come first; hybrid results keep their reranker-relative scores.
    """
    rag = get_unified_rag_system()
    if not await rag.initialize():
        return LegalContext(domain="unified", query=query), ParsedLegalQuery()

    if llm_invoke is not None:
        parsed = await parse_legal_query_llm(query, llm_invoke)
    else:
        parsed = parse_legal_query(query)

    # ── Deterministic pins ──────────────────────────────────────
    pinned: List[LegalChunk] = []
    for act_hint, section in parsed.pinned_sections:
        for chunk in rag.find_section(act_hint, section):
            pinned.append(
                LegalChunk(
                    chunk_id=chunk.chunk_id,
                    domain=chunk.domain,
                    act_name=chunk.act_name,
                    section_number=chunk.section_number,
                    title=chunk.title,
                    text=chunk.text,
                    source_file=chunk.source_file,
                    has_punishment=chunk.has_punishment,
                    domains=list(chunk.domains),
                    score=1.0,  # deterministically identified as governing law
                )
            )
    pinned = _dedupe_by_section(pinned)

    # ── Hybrid fill ─────────────────────────────────────────────
    if parsed.query_type == "section_lookup" and pinned:
        hybrid_k = 0
    elif parsed.query_type == "comparison" and pinned:
        hybrid_k = max(0, k - len(pinned))
    else:
        hybrid_k = k if not pinned else max(2, k - len(pinned))

    hybrid_chunks: List[LegalChunk] = []
    if hybrid_k > 0:
        search_query = query
        if parsed.expansion_terms:
            search_query = query + " " + " ".join(parsed.expansion_terms)
        context = await rag.retrieve(
            search_query,
            k=hybrid_k,
            domains=domains_hint or (parsed.domains or None),
        )
        hybrid_chunks = _drop_prose(context.chunks)

    merged = _dedupe_by_section(pinned + hybrid_chunks)

    # ── Confidence gate ─────────────────────────────────────────
    # Thin evidence (fewer than 3 primary provisions) → ONE broadened retry:
    # no domain filter, larger k, relaxed score cutoff. A grounded answer
    # from adjacent law beats a fluent answer from parametric memory.
    if len(merged) < 3 and parsed.query_type != "section_lookup":
        search_query = query
        if parsed.expansion_terms:
            search_query = query + " " + " ".join(parsed.expansion_terms)
        print(
            f"[LegalRetrieval] Confidence gate: only {len(merged)} provisions — "
            f"broadening retrieval."
        )
        context = await rag.retrieve(
            search_query, k=max(k, 12), domains=None, min_score=0.1
        )
        merged = _dedupe_by_section(merged + _drop_prose(context.chunks))

    merged = merged[: max(k, len(pinned))]
    merged = _expand_context(rag, merged)
    avg_conf = sum(c.score for c in merged) / len(merged) if merged else 0.0
    sources = list({f"{c.act_name} § {c.section_number}" for c in merged})

    if parsed.query_type != "general" or parsed.pinned_sections:
        print(
            f"[LegalRetrieval] type={parsed.query_type} "
            f"doctrines={parsed.doctrines} pins={len(pinned)} "
            f"hybrid={len(hybrid_chunks)}"
        )

    return (
        LegalContext(
            domain="unified",
            query=query,
            chunks=merged,
            sources=sources,
            confidence=round(avg_conf, 3),
        ),
        parsed,
    )


async def retrieve_case_law(
    query: str,
    parsed: ParsedLegalQuery,
    statute_chunks: List[LegalChunk],
    k: int = 3,
    query_issues: Optional[List[str]] = None,
) -> List[CaseRecord]:
    """
    Second hop of statute -> case retrieval: query the curated landmark-
    judgment corpus, composite-scoring by semantic relevance + issue
    overlap (query_issues, when the caller has run FIRAC extraction on an
    uploaded document — see similar_cases.py) + statute overlap (cases
    whose `statutes_cited` overlap the sections already retrieved for this
    query) + doctrine overlap (parsed.doctrines). query_issues defaults to
    None for callers without a structured FIRAC query (e.g. the general
    chatbot), which degrades gracefully rather than distorting ranking —
    see case_law_rag.CaseLawRAGSystem.retrieve. Silently returns [] if the
    case-law corpus isn't built yet.
    """
    case_rag = get_case_law_rag_system()
    if not await case_rag.initialize():
        return []

    boost = [(c.act_name, c.section_number) for c in statute_chunks]
    search_query = query
    if parsed.doctrines:
        search_query = query + " " + " ".join(parsed.doctrines)
    return await case_rag.retrieve(
        search_query,
        k=k,
        boost_statutes=boost,
        query_issues=query_issues,
        query_doctrines=parsed.doctrines,
    )
