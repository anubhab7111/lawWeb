"""
criminal_rag.py — Criminal Law Domain RAG System

Handles: IPC 1860, BNS 2023, CrPC 1973, BNSS 2023, Indian Evidence Act, BSA 2023
Source PDFs: app/data/bare_acts/criminal/

Key design decisions
--------------------
* Punishment-clause filter IS kept: for criminal law, only chargeable sections
  (those with an explicit "shall be punished" clause) are indexed. This prevents
  the LLM from citing procedural definitions as offences.

* _preprocess_query() is SAFE: it maps genuine criminal vocabulary only.
  REMOVED dangerous mappings:
    - AI / algorithm / financial-loss  →  causing death by negligence  ❌
    - cryptocurrency / blockchain      →  criminal breach of trust      ❌
    - "fraud/scam/financial" (generic) →  cheating                     ❌
  KEPT safe mappings:
    - stabbed / slash                  →  grievous hurt / hurt          ✅
    - killed / dead                    →  culpable homicide / murder    ✅
    - kidnap / abduct                  →  kidnapping / abduction        ✅
    - sexual assault / rape            →  rape / sexual intent          ✅

* retrieve_sections() retains the same signature as the old CrimeRAGSystem
  so the chatbot.py handle_crime_report node works with zero changes (aside
  from swapping the import).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from app.tools.base_legal_rag import (
    BaseLegalRAGSystem,
    LegalChunk,
    LegalContext,
    _extract_punishment,
    _infer_act_name,
)


# ─────────────────────────────────────────────────────────────
# Data models (kept for backward compatibility with chatbot.py)
# ─────────────────────────────────────────────────────────────


@dataclass
class CrimeFeatures:
    """Extracted legal signals from a crime description (unchanged API)."""

    violence: bool = False
    death: bool = False
    weapon: str = ""
    intent: str = "unknown"  # intentional | reckless | negligent | unknown
    property_loss: bool = False
    sexual: bool = False
    fraud: bool = False
    domestic: bool = False
    trespass: bool = False
    fire: bool = False
    kidnapping: bool = False
    threat: bool = False


@dataclass
class SectionMatch:
    """A matched IPC/BNS section with confidence and reasoning."""

    section: str
    title: str
    confidence: float
    reasons: List[str]
    punishment: str
    definition: str
    review_required: bool = False


@dataclass
class RAGResult:
    """Final output of the criminal RAG pipeline."""

    crime_type: str
    ipc_sections: List[SectionMatch]
    sources: List[str]
    confidence: float


@dataclass
class CrimeContext:
    """Legacy context format for backward compatibility."""

    crime_type: str
    relevant_passages: List[str]
    sources: List[str]
    confidence: float


# ─────────────────────────────────────────────────────────────
# Crime Feature Extraction  (same logic as the old crime_rag.py)
# ─────────────────────────────────────────────────────────────


def extract_crime_features(text: str) -> CrimeFeatures:
    """Convert crime description into structured legal signals."""
    t = text.lower()
    f = CrimeFeatures()

    violence_words = [
        "hit", "beat", "attack", "assault", "stab", "slash", "punch", "kick",
        "injure", "wound", "hurt", "violence", "physical", "bleed", "fracture",
        "broken bone",
    ]
    f.violence = any(w in t for w in violence_words)

    death_words = ["kill", "murder", "dead", "death", "died", "homicide",
                   "body found", "corpse"]
    f.death = any(w in t for w in death_words)

    weapons = {
        "knife": ["knife", "stabbed", "stabbing", "blade"],
        "gun": ["gun", "shot", "shooting", "firearm", "pistol", "rifle", "bullet"],
        "acid": ["acid attack", "acid thrown", "acid"],
        "stick": ["stick", "rod", "bat", "lathi"],
        "explosive": ["bomb", "explosive", "blast"],
        "vehicle": ["run over", "hit by car", "vehicle"],
    }
    for weapon, keywords in weapons.items():
        if any(w in t for w in keywords):
            f.weapon = weapon
            f.violence = True
            break

    intentional_words = [
        "deliberately", "intentionally", "planned", "premeditated",
        "purposely", "wilfully", "willfully", "on purpose",
    ]
    reckless_words = [
        "reckless", "rashly", "negligent", "careless",
        "speeding", "drunk driving", "rash driving",
    ]
    if any(w in t for w in intentional_words):
        f.intent = "intentional"
    elif any(w in t for w in reckless_words):
        f.intent = "reckless"
    elif f.death and not f.violence:
        f.intent = "negligent"
    elif f.violence or f.death:
        f.intent = "intentional"

    property_words = [
        "stolen", "theft", "robbed", "took my", "snatched",
        "missing property", "cheated money", "misappropriated", "embezzled",
        "property taken", "grabbed", "encroached", "illegally taken",
    ]
    f.property_loss = any(w in t for w in property_words)

    sexual_words = [
        "rape", "molest", "sexual assault", "groping", "stalking",
        "sexual harassment", "indecent", "obscene",
    ]
    f.sexual = any(w in t for w in sexual_words)

    fraud_words = [
        "fraud", "scam", "cheated", "deceived", "forged",
        "fake", "forgery", "counterfeit", "swindled", "duped",
    ]
    f.fraud = any(w in t for w in fraud_words)

    domestic_words = [
        "husband", "wife", "in-laws", "dowry", "domestic",
        "marital", "spouse", "marriage", "matrimonial",
    ]
    f.domestic = any(w in t for w in domestic_words)

    trespass_words = [
        "trespass", "encroach", "illegal entry", "broke into",
        "entered my", "occupied my land", "illegally taken",
        "land grabbed", "land taken", "property grabbed",
    ]
    f.trespass = any(w in t for w in trespass_words)

    fire_words = ["fire", "arson", "set fire", "burnt", "burning",
                  "flames", "house fire", "on fire"]
    f.fire = any(w in t for w in fire_words)

    kidnap_words = [
        "kidnap", "abduct", "ransom", "taken away", "missing child", "hostage",
    ]
    f.kidnapping = any(w in t for w in kidnap_words)

    threat_words = [
        "threatened", "threatening", "threat", "intimidate", "intimidation",
        "will kill", "warned me", "death threat",
    ]
    f.threat = any(w in t for w in threat_words)

    return f


# ─────────────────────────────────────────────────────────────
# Criminal RAG System
# ─────────────────────────────────────────────────────────────


class CriminalRAGSystem(BaseLegalRAGSystem):
    """
    Criminal law RAG: indexes IPC, BNS, CrPC, BNSS, Evidence Act.

    Filtering policy:
      - Only sections with an explicit punishment clause are indexed.
      - Query preprocessing maps genuine criminal vocabulary only —
        civil/tech/AI queries are NOT touched.
    """

    @property
    def domain_name(self) -> str:
        return "criminal"

    @property
    def pdf_subdir(self) -> str:
        return "criminal"

    # ── Overrides ────────────────────────────────────────────────

    def _parse_legal_sections(
        self, full_text: str, source_file: str
    ) -> List[LegalChunk]:
        """
        Criminal law parser with punishment-clause filter.

        LEGAL FILTER (appropriate for criminal domain only):
        Sections without a "shall be punished" or "shall be punishable"
        clause are excluded — we only want chargeable sections in the
        criminal index.
        """
        base_chunks = super()._parse_legal_sections(full_text, source_file)

        # Apply criminal-specific filter: only index sections with punishment
        filtered: List[LegalChunk] = []
        for chunk in base_chunks:
            if chunk.has_punishment:
                filtered.append(chunk)
            # else: skip definition-only sections — appropriate for IPC/BNS

        return filtered

    def _preprocess_query(self, query: str) -> str:
        """
        Safe criminal vocabulary enhancement.

        SAFETY RULE: Only expand terms that are unambiguously criminal acts
        involving physical harm, sexual violence, theft, or explicit fraud.
        Do NOT map civil torts, financial instruments, or technology concepts
        to criminal section headings.
        """
        q = query.lower()
        terms: List[str] = []

        # Physical violence → relevant IPC headings
        if any(w in q for w in ["stabbed", "slash", "cut with knife", "blade"]):
            terms.extend(["grievous hurt", "hurt", "dangerous weapon"])
        if any(w in q for w in ["beaten", "punched", "hit", "physically assaulted"]):
            terms.extend(["hurt", "voluntarily causing hurt"])
        if any(w in q for w in ["killed", "murdered", "dead", "death"]):
            terms.extend(["culpable homicide", "murder", "causing death"])

        # Sexual offences
        if any(w in q for w in ["rape", "sexual assault", "molest"]):
            terms.extend(["rape", "sexual intent", "outraging modesty"])
        if any(w in q for w in ["stalking", "following woman", "monitor woman"]):
            terms.extend(["stalking", "following woman"])

        # Kidnapping / abduction
        if any(w in q for w in ["kidnap", "abduct", "hostage", "ransom"]):
            terms.extend(["kidnapping", "abduction", "ransom"])

        # Domestic violence / dowry (genuinely criminal provisions)
        if any(w in q for w in ["dowry", "498a", "cruelty by husband"]):
            terms.extend(["cruelty by husband", "dowry death", "abetment of suicide"])

        # Explicit criminal fraud / forgery (only when combined with criminal act verbs)
        if any(w in q for w in ["forged document", "forged signature", "fake document"]):
            terms.extend(["forgery", "using forged document"])
        if any(w in q for w in ["cheated me", "cheated out of", "deceived me into"]):
            terms.extend(["cheating", "dishonestly inducing delivery of property"])

        # FIR / procedure queries
        if any(w in q for w in ["fir", "police complaint", "cognizable", "arrest"]):
            terms.extend(["cognizable offence", "complaint", "investigation"])

        # Bail
        if any(w in q for w in ["bail", "anticipatory bail", "custody"]):
            terms.extend(["bail", "custody", "arrest"])

        # Arson
        if any(w in q for w in ["set fire", "arson", "burnt my house"]):
            terms.extend(["arson", "fire to property"])

        # Criminal trespass (breaking and entering — not civil land disputes)
        if any(w in q for w in ["broke into", "illegal entry", "trespassed into house"]):
            terms.extend(["criminal trespass", "house-breaking"])

        if terms:
            return query + " " + " ".join(terms)
        return query

    def _build_search_query(
        self, query: str, crime_type: str, features: CrimeFeatures
    ) -> str:
        """Build an enhanced search with feature signals (criminal context only)."""
        parts = [query, query, query]  # 3× weight for original query

        if features.violence and features.death:
            parts.append("murder culpable homicide")
        elif features.violence:
            parts.append("hurt grievous hurt assault")
        elif features.death:
            parts.append("culpable homicide causing death")

        if features.property_loss and features.fraud:
            parts.append("cheating criminal breach of trust")
        elif features.property_loss:
            parts.append("theft stolen property")
        elif features.fraud:
            parts.append("cheating dishonestly inducing delivery")

        if features.sexual:
            parts.append("rape sexual assault outraging modesty")
        if features.kidnapping:
            parts.append("kidnapping abduction")
        if features.threat:
            parts.append("criminal intimidation threat")
        if features.weapon:
            parts.append(f"{features.weapon} dangerous weapon")

        return " ".join(parts)

    # ── Main retrieval entry-point (backward-compatible API) ────

    async def retrieve_sections(
        self,
        query: str,
        crime_type: str = "",
        features: Optional[CrimeFeatures] = None,
        k: int = 2,
    ) -> RAGResult:
        """
        Full criminal RAG pipeline.

        Maintains the same signature as the old CrimeRAGSystem.retrieve_sections()
        so chatbot.py nodes need only change the import.
        """
        import asyncio

        if not self.initialized:
            await self.initialize()

        if not self.initialized or not self.vector_store:
            return RAGResult(
                crime_type=crime_type or "general",
                ipc_sections=[],
                sources=[],
                confidence=0.0,
            )

        if features is None:
            features = extract_crime_features(query)

        try:
            search_query = self._build_search_query(
                self._preprocess_query(query), crime_type, features
            )

            import asyncio as _asyncio
            loop = _asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None,
                lambda: self.vector_store.similarity_search_with_score(
                    search_query, k=20
                ),
            )

            MIN_CONFIDENCE = 0.40
            matches: List[SectionMatch] = []
            seen: set = set()

            for doc, distance in results:
                sec_num = doc.metadata.get("section_number", "")
                if not sec_num or sec_num in seen:
                    continue
                seen.add(sec_num)

                cached = self._chunks.get(doc.metadata.get("chunk_id", ""))

                # Punishment clause required for criminal index (sanity check)
                punishment = (
                    ""
                    if not cached
                    else (cached.text[:250] if cached.has_punishment else "")
                ) or doc.metadata.get("punishment", "")
                if not punishment or len(punishment) < 10:
                    continue

                confidence = max(0.0, 1.0 - (distance / 2.0))
                if confidence < MIN_CONFIDENCE:
                    continue

                title = cached.title if cached else doc.metadata.get("title", "")
                definition = cached.text if cached else doc.page_content[:300]

                matches.append(
                    SectionMatch(
                        section=sec_num,
                        title=title,
                        confidence=round(min(confidence, 1.0), 2),
                        reasons=["Chargeable criminal section with punishment clause"],
                        punishment=punishment,
                        definition=definition,
                        review_required=confidence < 0.6,
                    )
                )

                if len(matches) >= k * 2:
                    break

            matches.sort(key=lambda m: m.confidence, reverse=True)
            matches = matches[:k]

            avg_conf = (
                sum(m.confidence for m in matches) / len(matches) if matches else 0.0
            )
            sources = list({f"Section {m.section}" for m in matches})

            return RAGResult(
                crime_type=crime_type or "general",
                ipc_sections=matches,
                sources=sources,
                confidence=round(avg_conf, 2),
            )

        except Exception as e:
            print(f"[criminal] Retrieval error: {e}")
            import traceback; traceback.print_exc()
            return RAGResult(
                crime_type=crime_type or "general",
                ipc_sections=[],
                sources=[],
                confidence=0.0,
            )

    async def retrieve_context(
        self, query: str, k: int = 5, crime_type: str = ""
    ) -> CrimeContext:
        """Legacy-compatible interface (used by indian_law_rag.py)."""
        features = extract_crime_features(query)
        result = await self.retrieve_sections(query, crime_type=crime_type,
                                              features=features, k=k)
        passages = []
        sources = []
        for match in result.ipc_sections:
            passages.append(
                f"Section {match.section} — {match.title}\n"
                f"{match.definition}\nPunishment: {match.punishment}"
            )
            sources.append(f"Section {match.section}")
        return CrimeContext(
            crime_type=result.crime_type,
            relevant_passages=passages,
            sources=sources,
            confidence=result.confidence,
        )


# ─────────────────────────────────────────────────────────────
# Singleton
# ─────────────────────────────────────────────────────────────

_criminal_rag: Optional[CriminalRAGSystem] = None


def get_criminal_rag_system() -> CriminalRAGSystem:
    """Get or create the CriminalRAGSystem singleton."""
    global _criminal_rag
    if _criminal_rag is None:
        _criminal_rag = CriminalRAGSystem()
    return _criminal_rag
