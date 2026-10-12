"""constitutional_rag.py — Constitutional Law Domain RAG System Handles: Constitution of India (and any future constitutional instruments)."""

from __future__ import annotations

import re
from pathlib import Path
from app.text_match import any_word
from typing import List, Optional

from app.tools.base_legal_rag import (
    BaseLegalRAGSystem,
    LegalChunk,
    LegalContext,
    _infer_act_name,
    _schedule_boundary,
)


class ConstitutionalRAGSystem(BaseLegalRAGSystem):
    """Constitutional law RAG: indexes the Constitution of India."""

    @property
    def domain_name(self) -> str:
        return "constitutional"

    @property
    def pdf_subdir(self) -> str:
        return "constitutional"


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


    def _parse_legal_sections(
        self, full_text: str, source_file: str
    ) -> List[LegalChunk]:
        """Article-aware constitutional parser."""
        chunks: List[LegalChunk] = []
        act_name = _infer_act_name(source_file)


        boundary = _schedule_boundary(full_text)
        search_text = full_text[:boundary] if boundary else full_text


        article_pattern = re.compile(
            r"\n\s*(?:Article\s+)?(?:\d+\[)?(\d{1,3}[A-Z]{0,2})\.\s*([^\n.—]{3,}?)(?:[.—])\s*",
            re.MULTILINE,
        )

        matches = list(article_pattern.finditer(search_text))

        if not matches:

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


            act_prefix = re.sub(r"[^A-Z0-9]", "", act_name.upper())
            chunk_id = f"CON_{act_prefix}_{art_num}"


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


                    has_punishment=False,
                )
            )

        print(f"[constitutional] Parsed {len(chunks)} Articles from {source_file}")
        return chunks

    def _preprocess_query(self, query: str) -> str:
        """Expand constitutional queries with Article numbers and doctrine names."""
        q = query.lower()
        terms: List[str] = []


        if any_word(q, ["right to equality", "equal protection", "article 14"]):
            terms.extend(["Article 14", "right to equality", "equal protection of law"])
        if any_word(q, [
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
        if any_word(q, [
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
        if any_word(q, ["right to education", "article 21a"]):
            terms.extend(["Article 21A", "right to education"])
        if any_word(q, [
                "protection against arrest",
                "double jeopardy",
                "self-incrimination",
                "article 20",
            ]
        ):
            terms.extend(["Article 20", "protection against arbitrary arrest"])
        if any_word(q, ["article 32", "supreme court writ", "constitutional remedy"]
        ):
            terms.extend(["Article 32", "right to constitutional remedies"])
        if any_word(q, ["article 226", "high court writ"]):
            terms.extend(["Article 226", "power of high courts to issue writs"])


        if any_word(q, ["habeas corpus"]):
            terms.extend(["habeas corpus", "Article 32", "Article 226"])
        if any_word(q, ["mandamus"]):
            terms.extend(["mandamus", "writ of mandamus"])
        if any_word(q, ["certiorari"]):
            terms.extend(["certiorari", "writ of certiorari"])


        if any_word(q, ["president rule", "article 356", "emergency"]):
            terms.extend(["Article 356", "President's rule", "emergency provisions"])
        if any_word(q, ["union list", "state list", "concurrent list", "seventh schedule"]
        ):
            terms.extend(
                [
                    "Seventh Schedule",
                    "legislative lists",
                    "Article 246",
                    "concurrent list",
                ]
            )


        if any_word(q, ["amendment", "article 368", "basic structure"]):
            terms.extend(
                [
                    "Article 368",
                    "constitutional amendment",
                    "basic structure doctrine",
                    "Kesavananda Bharati",
                ]
            )


        if any_word(q, ["dpsp", "directive principles", "article 36", "welfare state"]
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


        if any_word(q, ["parliament", "lok sabha", "rajya sabha", "legislative power"]
        ):
            terms.extend(
                ["Article 79", "Parliament", "Article 105", "legislative powers"]
            )

        if terms:
            return query + " " + " ".join(terms)
        return query


_constitutional_rag: Optional[ConstitutionalRAGSystem] = None


def get_constitutional_rag_system() -> ConstitutionalRAGSystem:
    """Get or create the ConstitutionalRAGSystem singleton."""
    global _constitutional_rag
    if _constitutional_rag is None:
        _constitutional_rag = ConstitutionalRAGSystem()
    return _constitutional_rag
