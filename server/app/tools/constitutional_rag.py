"""
constitutional_rag.py — Constitutional Law Domain RAG System

Handles: Constitution of India (and any future constitutional instruments).
Source PDFs: app/data/bare_acts/constitutional/

Key design decisions
--------------------
* Article-aware parser: The Constitution uses "Article N" / "Article NA"
  headers, not the numeric "N. Title" pattern used by the IPC. The base
  class's generic parser must be overridden to match these headers.

* NO punishment-clause filter. Fundamental rights (Articles 14, 19, 21,
  32) and Directive Principles have no criminal penalty clause; filtering
  them out would result in an empty index.

* _preprocess_query() maps constitutional vocabulary (fundamental rights,
  writs, amendments) to Article numbers to improve retrieval precision.

* Completely isolated from criminal_rag.py: a constitutional query
  (e.g. "Is surveillance legal under Article 21?") will NEVER receive
  context from the IPC punishment index.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional

from app.tools.base_legal_rag import (
    BaseLegalRAGSystem,
    LegalChunk,
    LegalContext,
    _infer_act_name,
    _schedule_boundary,
)


class ConstitutionalRAGSystem(BaseLegalRAGSystem):
    """
    Constitutional law RAG: indexes the Constitution of India.

    Filtering policy:
      - ALL Articles are indexed (no punishment-clause filter).
      - Article-aware chunking via overridden _parse_legal_sections().
    """

    @property
    def domain_name(self) -> str:
        return "constitutional"

    @property
    def pdf_subdir(self) -> str:
        return "constitutional"

    # ── Adapter: delegate storage/retrieval to the unified index ──

    async def initialize(self) -> bool:
        from app.tools.unified_legal_rag import get_unified_rag_system

        self.initialized = await get_unified_rag_system().initialize()
        return self.initialized

    async def retrieve(
        self,
        query: str,
        k: int = 4,
        min_score: float = 0.25,
        domains: Optional[List[str]] = None,
        use_reranker: bool = True,
    ) -> LegalContext:
        from app.tools.unified_legal_rag import get_unified_rag_system

        context = await get_unified_rag_system().retrieve(
            query,
            k=k,
            min_score=min_score,
            domains=domains or [self.domain_name],
            use_reranker=use_reranker,
        )
        return LegalContext(
            domain=self.domain_name,
            query=query,
            chunks=context.chunks,
            sources=context.sources,
            confidence=context.confidence,
        )

    # ── Overrides ────────────────────────────────────────────────

    def _parse_legal_sections(
        self, full_text: str, source_file: str
    ) -> List[LegalChunk]:
        """
        Article-aware constitutional parser.

        Matches headers of the form:
          "Article 21.  Protection of life and personal liberty."
          "Article 226.  Power of High Courts to issue certain writs."
          "21A.  Right to education."
        """
        chunks: List[LegalChunk] = []
        act_name = _infer_act_name(source_file)

        # Stop before any Schedule: schedules restart their own paragraph
        # numbering (1, 2, 3, ...), which otherwise collides with real
        # Article numbers and pollutes the index with schedule text.
        boundary = _schedule_boundary(full_text)
        search_text = full_text[:boundary] if boundary else full_text

        # Primary pattern: "Article 19" / "Article 19A". \s* (not \s+) after the
        # period for consistency with the base parser — some source PDFs run
        # the body straight on with no space after the number/period. An
        # optional "N[" prefix is tolerated: amended Articles are often
        # annotated with a footnote-bracket marker, e.g. "3[226. Power of...".
        article_pattern = re.compile(
            r"\n\s*(?:Article\s+)?(?:\d+\[)?(\d{1,3}[A-Z]?)\.\s*([^\n.—]{3,}?)(?:[.—])\s*",
            re.MULTILINE,
        )

        matches = list(article_pattern.finditer(search_text))

        if not matches:
            # Fallback: treat the whole document as one chunk
            chunk_id = f"CON_CONSTITUTION_FULL"
            chunks.append(
                LegalChunk(
                    chunk_id=chunk_id,
                    domain=self.domain_name,
                    act_name=act_name,
                    section_number="",
                    title=act_name,
                    text=full_text[:2000],
                    source_file=source_file,
                    has_punishment=False,
                )
            )
            return chunks

        for i, match in enumerate(matches):
            art_num = match.group(1).strip()
            title = match.group(2).strip().rstrip(".")

            start = match.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(search_text)
            raw = search_text[start:end].strip()
            raw = re.sub(r"\s+", " ", raw)

            if len(raw) < 20:
                continue

            # Prefix by Act (not just "CON_ART_"): this domain indexes 3 PDFs
            # (Constitution, Representation of the People Act, Protection of
            # Human Rights Act) whose section/article numbers legitimately
            # overlap — without an Act-specific prefix, e.g. Constitution
            # Article 21 and Human Rights Act Section 21 collide under the
            # same chunk_id and one silently shadows the other.
            act_prefix = re.sub(r"[^A-Z0-9]", "", act_name.upper())[:6]
            chunk_id = f"CON_{act_prefix}_{art_num}"

            # Only the Constitution itself has "Articles"; the other statutes
            # in this domain (RoPA, PHRA) have plain sections and citing them
            # as Articles would be legally wrong.
            is_constitution = "constitution" in Path(source_file).stem.lower()
            sec_label = f"Article {art_num}" if is_constitution else art_num

            chunks.append(
                LegalChunk(
                    chunk_id=chunk_id,
                    domain=self.domain_name,
                    act_name=act_name,
                    section_number=sec_label,
                    title=title,
                    text=raw,
                    source_file=source_file,
                    # Some penal provisions exist in the Const. (e.g. Art. 105 on contempt)
                    # but has_punishment is informational only here; no filtering applied.
                    has_punishment=False,
                )
            )

        print(f"[constitutional] Parsed {len(chunks)} Articles from {source_file}")
        return chunks

    def _preprocess_query(self, query: str) -> str:
        """
        Expand constitutional queries with Article numbers and doctrine names.
        """
        q = query.lower()
        terms: List[str] = []

        # Fundamental rights
        if any(w in q for w in ["right to equality", "equal protection", "article 14"]):
            terms.extend(["Article 14", "right to equality", "equal protection of law"])
        if any(
            w in q
            for w in [
                "free speech",
                "freedom of speech",
                "article 19",
                "expression",
                "press freedom",
            ]
        ):
            terms.extend(
                [
                    "Article 19",
                    "freedom of speech and expression",
                    "reasonable restriction",
                ]
            )
        if any(
            w in q
            for w in [
                "right to life",
                "personal liberty",
                "article 21",
                "privacy",
                "surveillance",
                "dignity",
            ]
        ):
            terms.extend(
                ["Article 21", "right to life", "personal liberty", "right to privacy"]
            )
        if any(w in q for w in ["right to education", "article 21a"]):
            terms.extend(["Article 21A", "right to education"])
        if any(
            w in q
            for w in [
                "protection against arrest",
                "double jeopardy",
                "self-incrimination",
                "article 20",
            ]
        ):
            terms.extend(["Article 20", "protection against arbitrary arrest"])
        if any(
            w in q
            for w in ["article 32", "supreme court writ", "constitutional remedy"]
        ):
            terms.extend(["Article 32", "right to constitutional remedies"])
        if any(w in q for w in ["article 226", "high court writ"]):
            terms.extend(["Article 226", "power of high courts to issue writs"])

        # Writs
        if any(w in q for w in ["habeas corpus"]):
            terms.extend(["habeas corpus", "Article 32", "Article 226"])
        if any(w in q for w in ["mandamus"]):
            terms.extend(["mandamus", "writ of mandamus"])
        if any(w in q for w in ["certiorari"]):
            terms.extend(["certiorari", "writ of certiorari"])

        # Federalism
        if any(w in q for w in ["president rule", "article 356", "emergency"]):
            terms.extend(["Article 356", "President's rule", "emergency provisions"])
        if any(
            w in q
            for w in ["union list", "state list", "concurrent list", "seventh schedule"]
        ):
            terms.extend(
                [
                    "Seventh Schedule",
                    "legislative lists",
                    "Article 246",
                    "concurrent list",
                ]
            )

        # Amendment / basic structure
        if any(w in q for w in ["amendment", "article 368", "basic structure"]):
            terms.extend(
                [
                    "Article 368",
                    "constitutional amendment",
                    "basic structure doctrine",
                    "Kesavananda Bharati",
                ]
            )

        # Directive Principles
        if any(
            w in q
            for w in ["dpsp", "directive principles", "article 36", "welfare state"]
        ):
            terms.extend(
                [
                    "Directive Principles",
                    "Article 36",
                    "Article 37",
                    "Article 38",
                    "Article 39",
                ]
            )

        # Parliament / legislature
        if any(
            w in q
            for w in ["parliament", "lok sabha", "rajya sabha", "legislative power"]
        ):
            terms.extend(
                ["Article 79", "Parliament", "Article 105", "legislative powers"]
            )

        if terms:
            return query + " " + " ".join(terms)
        return query


# ─────────────────────────────────────────────────────────────
# Singleton
# ─────────────────────────────────────────────────────────────

_constitutional_rag: Optional[ConstitutionalRAGSystem] = None


def get_constitutional_rag_system() -> ConstitutionalRAGSystem:
    """Get or create the ConstitutionalRAGSystem singleton."""
    global _constitutional_rag
    if _constitutional_rag is None:
        _constitutional_rag = ConstitutionalRAGSystem()
    return _constitutional_rag
