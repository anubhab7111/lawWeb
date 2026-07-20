"""
criminal_rag.py — Criminal Law Domain RAG System

Handles: IPC 1860, BNS 2023, CrPC 1973, BNSS 2023, Indian Evidence Act, BSA 2023
Source PDFs: app/data/bare_acts/criminal/

Key design decisions
--------------------
* Punishment-clause filter is applied PER-ACT, not uniformly. Substantive penal
  statutes (IPC, BNS, NDPS, POCSO, SC/ST Atrocities Act, PMLA) only index
  chargeable sections (those with an explicit "shall be punished"/"punishable"
  clause) — this prevents the LLM from citing procedural definitions as
  offences. Procedural/evidentiary codes (CrPC, BNSS, Indian Evidence Act, BSA)
  have no offence-creating clauses by nature — applying the same filter to them
  would index almost nothing (e.g. CrPC §482 quashing-FIR power, §438
  anticipatory bail — routinely cited sections — have no punishment clause and
  would be silently dropped). For those Acts, all parsed sections are indexed.

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

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from app.tools.base_legal_rag import (
    BaseLegalRAGSystem,
    LegalChunk,
    LegalContext,
    _extract_punishment,
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
        "hit",
        "beat",
        "attack",
        "assault",
        "stab",
        "slash",
        "punch",
        "kick",
        "injure",
        "wound",
        "hurt",
        "violence",
        "physical",
        "bleed",
        "fracture",
        "broken bone",
    ]
    f.violence = any(w in t for w in violence_words)

    death_words = [
        "kill",
        "murder",
        "dead",
        "death",
        "died",
        "homicide",
        "body found",
        "corpse",
    ]
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
        "deliberately",
        "intentionally",
        "planned",
        "premeditated",
        "purposely",
        "wilfully",
        "willfully",
        "on purpose",
    ]
    reckless_words = [
        "reckless",
        "rashly",
        "negligent",
        "careless",
        "speeding",
        "drunk driving",
        "rash driving",
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
        "stolen",
        "theft",
        "robbed",
        "took my",
        "snatched",
        "missing property",
        "cheated money",
        "misappropriated",
        "embezzled",
        "property taken",
        "grabbed",
        "encroached",
        "illegally taken",
    ]
    f.property_loss = any(w in t for w in property_words)

    sexual_words = [
        "rape",
        "molest",
        "sexual assault",
        "groping",
        "stalking",
        "sexual harassment",
        "indecent",
        "obscene",
    ]
    f.sexual = any(w in t for w in sexual_words)

    fraud_words = [
        "fraud",
        "scam",
        "cheated",
        "deceived",
        "forged",
        "fake",
        "forgery",
        "counterfeit",
        "swindled",
        "duped",
    ]
    f.fraud = any(w in t for w in fraud_words)

    domestic_words = [
        "husband",
        "wife",
        "in-laws",
        "dowry",
        "domestic",
        "marital",
        "spouse",
        "marriage",
        "matrimonial",
    ]
    f.domestic = any(w in t for w in domestic_words)

    trespass_words = [
        "trespass",
        "encroach",
        "illegal entry",
        "broke into",
        "entered my",
        "occupied my land",
        "illegally taken",
        "land grabbed",
        "land taken",
        "property grabbed",
    ]
    f.trespass = any(w in t for w in trespass_words)

    fire_words = [
        "fire",
        "arson",
        "set fire",
        "burnt",
        "burning",
        "flames",
        "house fire",
        "on fire",
    ]
    f.fire = any(w in t for w in fire_words)

    kidnap_words = [
        "kidnap",
        "abduct",
        "ransom",
        "taken away",
        "missing child",
        "hostage",
    ]
    f.kidnapping = any(w in t for w in kidnap_words)

    threat_words = [
        "threatened",
        "threatening",
        "threat",
        "intimidate",
        "intimidation",
        "will kill",
        "warned me",
        "death threat",
    ]
    f.threat = any(w in t for w in threat_words)

    return f


# ─────────────────────────────────────────────────────────────
# Criminal RAG System
# ─────────────────────────────────────────────────────────────


class CriminalRAGSystem(BaseLegalRAGSystem):
    """
    Criminal law RAG: indexes IPC, BNS, CrPC, BNSS, Evidence Act, BSA, NDPS,
    POCSO, SC/ST Atrocities Act, PMLA.

    Filtering policy:
      - Substantive penal statutes: only sections with an explicit punishment
        clause are indexed (see PUNISHMENT_FILTERED_ACTS below).
      - Procedural/evidentiary codes: no offence-creating clauses exist by
        design, so all parsed sections are indexed unfiltered.
      - Query preprocessing maps genuine criminal vocabulary only —
        civil/tech/AI queries are NOT touched.
    """

    # Substantive penal statutes — apply the punishment-clause filter.
    # Everything else parsed from bare_acts/criminal/ (procedural/evidentiary
    # codes: CrPC, BNSS, Indian Evidence Act, BSA) is indexed unfiltered.
    PUNISHMENT_FILTERED_ACTS = {
        "indian_penal_code_1860",
        "bharatiya_nyaya_sanhita_bns_2023",
        "ndps_act_1985",
        "pocso_act_2012",
        "sc_st_prevention_of_atrocities_act_1989",
        "prevention_of_money_laundering_act_pmla_2002",
    }

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
        Criminal law parser with a per-Act punishment-clause filter.

        Substantive penal statutes (PUNISHMENT_FILTERED_ACTS) only keep
        sections with a "shall be punished"/"punishable" clause, so the LLM
        can't cite a definition as a chargeable offence. Procedural/
        evidentiary codes (CrPC, BNSS, Evidence Act, BSA) have no such clauses
        by nature — filtering them the same way would index almost nothing,
        including routinely-cited sections like CrPC §482 (quashing FIRs) or
        §438 (anticipatory bail) — so those are indexed unfiltered.
        """
        base_chunks = super()._parse_legal_sections(full_text, source_file)

        stem = Path(source_file).stem.lower()
        if stem not in self.PUNISHMENT_FILTERED_ACTS:
            return base_chunks

        # Apply criminal-specific filter: only index sections with punishment
        filtered: List[LegalChunk] = []
        for chunk in base_chunks:
            if chunk.has_punishment:
                filtered.append(chunk)
            # else: skip definition-only sections — appropriate for IPC/BNS

        # Fallback: some penal acts (e.g. PMLA, POCSO scans) have few
        # sections whose punishment clause the regex can extract — a strict
        # filter would nearly empty them. Better an unfiltered act than an
        # unretrievable one.
        if len(filtered) < max(10, len(base_chunks) // 4):
            print(
                f"  [criminal] Punishment filter kept only {len(filtered)}/"
                f"{len(base_chunks)} sections of {source_file} — keeping all."
            )
            return base_chunks

        return filtered

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
        if any(
            w in q for w in ["forged document", "forged signature", "fake document"]
        ):
            terms.extend(["forgery", "using forged document"])
        if any(w in q for w in ["cheated me", "cheated out of", "deceived me into"]):
            terms.extend(["cheating", "dishonestly inducing delivery of property"])

        # Electronic evidence (WhatsApp/email/CCTV → statutory vocabulary)
        if any(
            w in q
            for w in [
                "whatsapp",
                "electronic evidence",
                "digital evidence",
                "email as evidence",
                "chats admissible",
                "cctv",
                "call recording",
            ]
        ):
            terms.extend(
                [
                    "admissibility of electronic records",
                    "electronic record",
                    "certificate",
                ]
            )

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
        if any(
            w in q for w in ["broke into", "illegal entry", "trespassed into house"]
        ):
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
        Full criminal RAG pipeline over the unified hybrid index
        (BM25 + dense + reranker, filtered to the criminal domain).

        Maintains the same signature as the old CrimeRAGSystem.retrieve_sections()
        so chatbot.py nodes need only change the import.
        """
        from app.tools.unified_legal_rag import get_unified_rag_system

        unified = get_unified_rag_system()
        if not await unified.initialize():
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

            chunks = await unified._hybrid_search(
                search_query=search_query,
                rerank_query=query,
                k=k * 2,
                min_score=0.25,
                domains=[self.domain_name],
            )

            matches: List[SectionMatch] = []
            seen: set = set()

            for chunk in chunks:
                sec_num = chunk.section_number
                if not sec_num or sec_num in seen:
                    continue
                seen.add(sec_num)

                # Punishment clause is only required for substantive penal
                # statutes (IPC/BNS/NDPS/POCSO/etc.) — matches _parse_legal_sections'
                # per-Act policy. Procedural/evidentiary codes (CrPC, BNSS,
                # Evidence Act, BSA) have no such clause by design; requiring
                # one here would silently drop routinely-cited sections like
                # CrPC §438 (anticipatory bail) or §482 (inherent powers).
                requires_punishment = (
                    Path(chunk.source_file).stem.lower()
                    in self.PUNISHMENT_FILTERED_ACTS
                )

                punishment = _extract_punishment(chunk.text) or (
                    chunk.text[:250] if chunk.has_punishment else ""
                )
                if requires_punishment and (not punishment or len(punishment) < 10):
                    continue

                matches.append(
                    SectionMatch(
                        section=sec_num,
                        title=chunk.title,
                        confidence=round(min(chunk.score, 1.0), 2),
                        reasons=(
                            ["Chargeable criminal section with punishment clause"]
                            if requires_punishment
                            else ["Procedural/evidentiary criminal law section"]
                        ),
                        punishment=punishment,
                        definition=chunk.text,
                        review_required=chunk.score < 0.6,
                    )
                )

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
            import traceback

            traceback.print_exc()
            return RAGResult(
                crime_type=crime_type or "general",
                ipc_sections=[],
                sources=[],
                confidence=0.0,
            )

    async def get_relevant_context(self, query: str, top_k: int = 3) -> dict:
        """Dict-style context used by the document analysis pipeline."""
        context = await self.retrieve_context(query, k=top_k)
        return {
            "passages": context.relevant_passages,
            "sources": context.sources,
            "crime_type": context.crime_type,
        }

    async def retrieve_context(
        self, query: str, k: int = 5, crime_type: str = ""
    ) -> CrimeContext:
        """Legacy-compatible interface (used by indian_law_rag.py)."""
        features = extract_crime_features(query)
        result = await self.retrieve_sections(
            query, crime_type=crime_type, features=features, k=k
        )
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
