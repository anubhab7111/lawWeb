"""
civil_rag.py — Civil Law Domain RAG System

Handles: Indian Contract Act 1872, Transfer of Property Act 1882,
         Code of Civil Procedure 1908, Specific Relief Act 1963,
         Negotiable Instruments Act 1881, Limitation Act 1963,
         Protection of Women from Domestic Violence Act 2005,
         Right to Information Act 2005, and others.

Source PDFs: app/data/bare_acts/civil/

Key design decisions
--------------------
* NO punishment-clause filter. Civil law is built on rights,
  obligations, and remedies — not criminal sentences. Sections like
  Indian Contract Act § 10 ("What agreements are contracts") have no
  "shall be punished" clause but are the foundation of contract law.
  Filtering them out was the root cause of hallucinations.

* _preprocess_query() expands civil vocabulary (breach of contract,
  damages, remedies, etc.) without ever mapping to IPC sections.

* retrieve() is the universal entry-point from the chatbot.
  The result is returned as a LegalContext (new type) rather than the
  old RAGResult (criminal-specific type).
"""

from __future__ import annotations

import re
from typing import List, Optional

from app.tools.base_legal_rag import BaseLegalRAGSystem, LegalChunk, LegalContext


class CivilRAGSystem(BaseLegalRAGSystem):
    """
    Civil law RAG: indexes all PDFs under app/data/bare_acts/civil/.

    Filtering policy:
      - ALL sections are indexed (no punishment-clause filter).
      - This ensures foundational provisions like ICA §10, Specific
        Relief Act §10–16, and CPC Order rules are available.
    """

    @property
    def domain_name(self) -> str:
        return "civil"

    @property
    def pdf_subdir(self) -> str:
        return "civil"

    # No override of _parse_legal_sections() — base implementation is correct
    # for civil law (generic parser, no punishment filter).

    def _preprocess_query(self, query: str) -> str:
        """
        Expand civil law queries with domain vocabulary.

        Deliberately civil-only — never maps to criminal section headings.
        """
        q = query.lower()
        terms: List[str] = []

        # Contract law
        if any(w in q for w in ["breach", "breached", "default", "non-performance"]):
            terms.extend(["breach of contract", "damages", "Section 73", "Section 74"])
        if any(w in q for w in ["contract", "agreement", "enforceable"]):
            terms.extend(["valid contract", "Section 10", "consideration",
                          "free consent"])
        if any(w in q for w in ["void", "voidable"]):
            terms.extend(["void agreement", "voidable contract",
                          "coercion", "undue influence", "misrepresentation"])
        if any(w in q for w in ["force majeure", "frustration", "impossibility"]):
            terms.extend(["frustration of contract", "Section 56",
                          "supervening impossibility"])
        if any(w in q for w in ["specific performance"]):
            terms.extend(["specific performance", "Specific Relief Act",
                          "Section 10 Specific Relief"])
        if any(w in q for w in ["injunction"]):
            terms.extend(["injunction", "temporary injunction", "permanent injunction"])
        if any(w in q for w in ["damages", "compensation", "indemnify"]):
            terms.extend(["liquidated damages", "unliquidated damages",
                          "Section 73", "Section 74"])

        # Property / landlord-tenant
        if any(w in q for w in ["lease", "rent", "tenant", "landlord", "eviction"]):
            terms.extend(["lease", "Section 105 Transfer of Property Act",
                          "tenancy", "eviction"])
        if any(w in q for w in ["sale deed", "property sale", "conveyance"]):
            terms.extend(["sale of immovable property", "Section 54",
                          "Transfer of Property Act"])
        if any(w in q for w in ["mortgage", "hypothecation", "charge"]):
            terms.extend(["mortgage", "Section 58", "Transfer of Property Act"])

        # Negotiable instruments
        if any(w in q for w in ["cheque", "bounce", "dishonour", "138", "ni act"]):
            terms.extend(["cheque dishonour", "Section 138 Negotiable Instruments Act",
                          "penalty for dishonour"])
        if any(w in q for w in ["promissory note", "bill of exchange"]):
            terms.extend(["promissory note", "bill of exchange",
                          "Negotiable Instruments Act"])

        # Civil procedure
        if any(w in q for w in ["suit", "plaint", "civil court", "limitation"]):
            terms.extend(["code of civil procedure", "limitation period",
                          "Limitation Act"])
        if any(w in q for w in ["injunction", "stay order"]):
            terms.extend(["temporary injunction", "Order 39 CPC"])

        # RTI
        if any(w in q for w in ["rti", "information", "public authority", "disclosure"]):
            terms.extend(["right to information", "RTI Act", "public authority",
                          "Section 6"])

        # AI / technology liability (civil tort framing — NOT criminal)
        if any(w in q for w in ["ai", "artificial intelligence", "algorithm", "automated"]):
            if any(w in q for w in ["liable", "liability", "loss", "damage"]):
                terms.extend(["civil liability", "negligence tort",
                               "duty of care", "damages for loss"])

        # Corporate / financial loss (civil framing)
        if any(w in q for w in ["financial loss", "economic loss", "monetary loss"]):
            terms.extend(["civil damages", "breach of contract damages",
                          "Section 73 Contract Act"])

        # Cryptocurrency / blockchain (civil/regulatory framing — NOT criminal)
        if any(w in q for w in ["cryptocurrency", "crypto", "bitcoin", "blockchain"]):
            terms.extend(["virtual asset", "contract enforceability",
                          "civil liability", "regulatory compliance"])

        if terms:
            return query + " " + " ".join(terms)
        return query


# ─────────────────────────────────────────────────────────────
# Singleton
# ─────────────────────────────────────────────────────────────

_civil_rag: Optional[CivilRAGSystem] = None


def get_civil_rag_system() -> CivilRAGSystem:
    """Get or create the CivilRAGSystem singleton."""
    global _civil_rag
    if _civil_rag is None:
        _civil_rag = CivilRAGSystem()
    return _civil_rag
