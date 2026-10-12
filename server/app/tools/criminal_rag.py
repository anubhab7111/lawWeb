"""criminal_rag.py — Criminal Law Domain RAG System Handles: IPC 1860, BNS 2023, CrPC 1973, BNSS 2023, Indian Evidence Act, BSA 2023 Source PDFs: app/data/bare_acts/criminal/ Key..."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import List, Optional

import numpy as np
from app.text_match import any_word
from app.tools.base_legal_rag import (
    BaseLegalRAGSystem,
    LegalChunk,
    LegalContext,
    _extract_punishment,
)
from app.tools.crime_reporter import ANIMAL_WORDS


@dataclass
class CrimeFeatures:
    """Extracted legal signals from a crime description (unchanged API)."""

    violence: bool = False
    death: bool = False
    weapon: str = ""
    intent: str = "unknown"
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
    act_name: str = ""


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
    f.violence = any_word(t, violence_words)

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
    f.death = any_word(t, death_words)

    weapons = {
        "knife": ["knife", "stabbed", "stabbing", "blade"],
        "gun": ["gun", "shot", "shooting", "firearm", "pistol", "rifle", "bullet"],
        "acid": ["acid attack", "acid thrown", "acid"],
        "stick": ["stick", "rod", "bat", "lathi"],
        "explosive": ["bomb", "explosive", "blast"],
        "vehicle": ["run over", "hit by car", "vehicle"],
    }
    for weapon, keywords in weapons.items():
        if any_word(t, keywords):
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
    if any_word(t, intentional_words):
        f.intent = "intentional"
    elif any_word(t, reckless_words):
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
    f.property_loss = any_word(t, property_words)

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
    f.sexual = any_word(t, sexual_words)

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
    f.fraud = any_word(t, fraud_words)

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
    f.domestic = any_word(t, domestic_words)

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
    f.trespass = any_word(t, trespass_words)

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
    f.fire = any_word(t, fire_words)

    kidnap_words = [
        "kidnap",
        "abduct",
        "ransom",
        "taken away",
        "missing child",
        "hostage",
    ]
    f.kidnapping = any_word(t, kidnap_words)

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
    f.threat = any_word(t, threat_words)

    return f


_CHARGEABLE_PUNISHMENT_RE = re.compile(
    r"\b(?:be|shall be)\s+punished\s+(?:with|in the same manner|as|under)\b",
    re.IGNORECASE,
)


_DOCTRINE_DESCRIPTIONS = {
    "forgery": (
        "Forgery (IPC 463, 468, 471): making a false document or part of a document with intent "
        "to cause damage or injury, to cheat, or to support a claim; using as genuine a document "
        "known to be forged. Forgery of a court record, public register, will, valuable security "
        "or power of attorney.",
        ["forgery", "using forged document"],
    ),
    "abetment": (
        "Abetment (IPC 107, 109, 120B): instigating a person to do a thing, hiring or procuring "
        "another person to commit an offence, conspiring with others to commit it, or intentionally "
        "aiding it. A person who abets an offence or joins a criminal conspiracy is punished with the "
        "punishment provided for the offence committed in consequence.",
        ["abetment", "punishment of abetment"],
    ),
    "cheating": (
        "Cheating (IPC 415, 417, 420): deceiving a person, fraudulently or dishonestly, to induce "
        "them to deliver property, consent to retention of property, or do something causing "
        "damage. Falsely pretending to be another person, or dishonest concealment of facts.",
        ["cheating", "dishonestly inducing delivery of property"],
    ),
    "theft": (
        "Theft (IPC 378, 379, 380): dishonestly taking movable property out of the possession of "
        "another person without consent. Theft from a building or dwelling house, and theft by a "
        "clerk or servant of their employer's property.",
        ["theft", "dishonestly taking movable property"],
    ),
    "criminal_intimidation": (
        "Criminal intimidation (IPC 503, 506): threatening another person with injury to their "
        "person, reputation or property, with intent to cause alarm or to make them do or omit an "
        "act. A threat to cause death or grievous hurt.",
        ["criminal intimidation", "threat of injury"],
    ),
    "breach_of_trust": (
        "Criminal breach of trust (IPC 405, 406, 409): a person entrusted with property or with "
        "dominion over property dishonestly misappropriates or converts it to his own use, or uses "
        "it in violation of a legal contract or trust. Includes a banker, agent or public servant.",
        ["criminal breach of trust", "misappropriation of entrusted property"],
    ),
    "extortion": (
        "Extortion (IPC 383, 384): intentionally putting a person in fear of injury and thereby "
        "dishonestly inducing them to deliver property or a valuable security. Threats to publish "
        "defamatory material or to accuse of an offence in order to obtain money.",
        ["extortion", "putting in fear to deliver property"],
    ),
    "defamation": (
        "Defamation (IPC 499, 500): making or publishing an imputation concerning a person, by "
        "words, signs or visible representation, intending to harm or knowing it will harm their "
        "reputation. Punishment for defamation.",
        ["defamation", "imputation harming reputation"],
    ),
    "dowry_cruelty": (
        "Dowry death and cruelty (IPC 304B, 498A): death of a woman within seven years of marriage "
        "after cruelty or harassment by her husband or his relatives over a dowry demand. A husband "
        "or relative of the husband subjecting a woman to cruelty.",
        ["dowry death", "cruelty by husband"],
    ),
    "kidnapping": (
        "Kidnapping and abduction (IPC 359, 363, 366): taking a person out of India or out of lawful "
        "guardianship, especially a minor. Kidnapping or abducting a woman to compel her marriage or "
        "forced illicit intercourse.",
        ["kidnapping", "abduction"],
    ),
}
DOCTRINE_MATCH_MIN_SCORE = 0.50


OFFENCE_DOMAINS = ["criminal", "consumer_cyber_ip", "family", "civil"]
NON_OFFENCE_ACTS = (
    "Code of Criminal Procedure",
    "Bharatiya Nagarik Suraksha Sanhita",
    "Indian Evidence Act",
    "Bharatiya Sakshya Adhiniyam",
)
EXTRA_OFFENCE_ACTS = (
    "Information Technology Act",
    "Dowry Prohibition Act",
    "Protection of Women from Domestic Violence Act",
)
OFFENCE_POOL = 30


def _is_offence_act(act_name: str) -> bool:
    if act_name.startswith(NON_OFFENCE_ACTS):
        return False
    return act_name.startswith(EXTRA_OFFENCE_ACTS) or act_name in _CRIMINAL_ACT_NAMES


STRUCK_DOWN_IPC = {"497"}
_IPC_CRIME_TYPES: Optional[dict] = None


def _ipc_crime_type(section: str) -> Optional[str]:
    global _IPC_CRIME_TYPES
    if _IPC_CRIME_TYPES is None:
        mapping = json.loads(
            (
                Path(__file__).resolve().parents[1] / "data" / "crime_type_map.json"
            ).read_text()
        )
        _IPC_CRIME_TYPES = {
            k[4:]: v for k, v in mapping.items() if k.startswith("IPC:")
        }
    return _IPC_CRIME_TYPES.get(str(section))


RELATED_CRIME_TYPES = [
    {"theft", "robbery"},
    {"fraud", "cybercrime"},
    {"assault", "murder", "threat", "robbery"},
    {"harassment", "rape", "kidnapping", "cybercrime", "threat"},
    {"domestic_violence", "dowry", "assault", "threat", "murder"},
    {"property_damage", "land_dispute", "arson", "threat"},
]


CRIME_TYPE_SECTIONS = {
    "theft": "379",
    "robbery": "392",
    "assault": "323",
    "threat": "503",
    "fraud": "420",
    "harassment": "354",
    "kidnapping": "363",
    "rape": "376",
    "domestic_violence": "498A",
    "dowry": "304B",
    "property_damage": "427",
    "land_dispute": "447",
    "arson": "436",
}


def _fits_crime_type_pair(section_type: str, crime_type: str) -> bool:
    return section_type == crime_type or any(
        {section_type, crime_type} <= g for g in RELATED_CRIME_TYPES
    )


def _fits_crime_type(section: str, crime_type: Optional[str]) -> bool:
    """A classifier-suggested IPC section from an unrelated crime family (cheating on a rape report) is noise; unknown or "general" on either side always fits."""
    section_type = _ipc_crime_type(section)
    if not section_type or not crime_type or crime_type == "general":
        return True
    return _fits_crime_type_pair(section_type, crime_type)


def _section_crime_types(act_name: str, section: str) -> set:
    """Crime types of a retrieved section; a BNS section takes those of its IPC sources."""
    from app.tools.fact_statutes import _translation

    if act_name == "Indian Penal Code":
        ipc = [str(section)]
    elif act_name.startswith("Bharatiya Nyaya Sanhita"):
        ipc = _translation()["ipc_bns"]["new_to_old"].get(str(section), [])
    else:
        return set()
    return {t for t in map(_ipc_crime_type, ipc) if t}


def _fits_report_lead(section: str, prob: float, crime_type: Optional[str]) -> bool:
    """A classifier lead on a short report from a merely related crime (a threat section on a rape report) needs a confident classifier."""
    if not _fits_crime_type(section, crime_type):
        return False
    return (
        _ipc_crime_type(section) in (None, crime_type)
        or crime_type in (None, "", "general")
        or prob >= CLASSIFIER_RELATED_MIN_PROB
    )


def _is_offence_section(act_name: str, section: str) -> bool:
    return _is_offence_act(act_name) and not (
        act_name == "Indian Penal Code" and str(section) in STRUCK_DOWN_IPC
    )


_SUBSECTION_RE = re.compile(r"\((\d{1,2})\)\s*(?=[A-Z])")
_GAZETTE_HEADER_RE = re.compile(r"Sec\. \d+\] THE GAZETTE OF INDIA EXTRAORDINARY \d*_*")


_MARGIN_NOTE_RE = re.compile(r"(?:^|(?<=\s))[A-Z][^.;:()]{0,80}\.\s*$")
_STATUTE_WORDS_RE = re.compile(r"\b(shall|[Ww]hoever|punished)\b")


def _strip_page_breaks(text: str) -> str:
    pages = _GAZETTE_HEADER_RE.split(text)
    for i, page in enumerate(pages[:-1]):
        while (
            (m := _MARGIN_NOTE_RE.search(page))
            and not _STATUTE_WORDS_RE.search(m.group(0))
            and len(m.group(0).split()) <= 12
        ):
            page = page[: m.start()].rstrip()
        pages[i] = page
    return " ".join(p.strip() for p in pages)


def _section_punishment(text: str) -> str:
    """Every subsection's punishment with its condition."""
    parts = _SUBSECTION_RE.split(_strip_page_breaks(text))
    clauses = []
    for num, body in zip(parts[1::2], parts[2::2]):
        punishment = _extract_punishment(body, max_len=160)
        if punishment:
            condition = re.split(r"\bshall\b", body, maxsplit=1)[0].strip().rstrip(",")
            if len(condition) > 160:
                condition = condition[:160].rsplit(" ", 1)[0] + "..."
            clause = f"({num}) {condition}: {punishment}"
            if clause not in clauses:
                clauses.append(clause)
    return "; ".join(clauses) if len(clauses) > 1 else _extract_punishment(text)


_CRIMINAL_ACT_NAMES = {
    "Indian Penal Code",
    "Bharatiya Nyaya Sanhita BNS",
    "NDPS Act",
    "Juvenile Justice Act",
    "Prevention of Money Laundering Act PMLA",
    "Unlawful Activities Prevention Act UAPA",
    "Arms Act",
    "POCSO Act",
    "Immigration and Foreigners Act",
    "SC ST Prevention of Atrocities Act",
    "Prevention of Corruption Act",
}


CLASSIFIER_DIR = (
    Path(__file__).resolve().parent.parent / "data" / "criminal_section_classifier"
)
CLASSIFIER_TOP_N = 10
CLASSIFIER_FIRST = 5
CLASSIFIER_LONG_QUERY_WORDS = 40
CLASSIFIER_SHORT_MIN_PROB = 0.15
CLASSIFIER_RELATED_MIN_PROB = 0.3
REPORT_NOISE_MIN_SCORE = 0.4


class CriminalRAGSystem(BaseLegalRAGSystem):
    """Criminal law RAG: indexes IPC, BNS, CrPC, BNSS, Evidence Act, BSA, NDPS, POCSO, SC/ST Atrocities Act, PMLA."""

    PUNISHMENT_FILTERED_ACTS = {
        "indian_penal_code_1860",
        "bharatiya_nyaya_sanhita_bns_2023",
        "ndps_act_1985",
        "pocso_act_2012",
        "sc_st_prevention_of_atrocities_act_1989",
        "prevention_of_money_laundering_act_pmla_2002",
    }

    INDEX_ALL_SECTIONS = {
        "indian_penal_code_1860",
        "bharatiya_nyaya_sanhita_bns_2023",
    }

    _doctrine_cache: Optional[tuple] = None
    _pin_index: Optional[dict] = None
    _classifier: Optional[tuple] = None
    _query_vec_cache: Optional[tuple] = None

    def _query_vector(self, query: str) -> Optional[np.ndarray]:
        from app.tools.unified_legal_rag import get_unified_rag_system

        if self._query_vec_cache and self._query_vec_cache[0] == query:
            return self._query_vec_cache[1]
        embeddings = get_unified_rag_system().embeddings
        if embeddings is None:
            return None
        q = np.array(embeddings.embed_query(query))
        q = q / np.linalg.norm(q)
        self._query_vec_cache = (query, q)
        return q

    def _classified_sections(self, query: str) -> List[tuple]:
        """(act, section, probability) for the classifier's top IPC sections, each followed by its BNS equivalent when one is mapped; empty if the classifier is not installed."""
        if self._classifier is None:
            weights = CLASSIFIER_DIR / "weights.npz"
            if not weights.exists():
                self._classifier = ()
                return []
            z = np.load(weights)
            bns_map = json.loads((CLASSIFIER_DIR / "bns_map.json").read_text())
            self._classifier = (
                z["W"],
                z["b"],
                [str(label) for label in z["labels"]],
                bns_map,
            )
        if not self._classifier:
            return []
        W, b, labels, bns_map = self._classifier
        q = self._query_vector(query)
        if q is None:
            return []

        logits = W @ q + b
        probs = np.exp(logits - logits.max())
        probs /= probs.sum()
        triples = []
        for i in np.argsort(-logits):
            if labels[i] == "NONE":
                continue
            triples.append(("Indian Penal Code", labels[i], float(probs[i])))
            if labels[i] in bns_map:
                triples.append(
                    ("Bharatiya Nyaya Sanhita", bns_map[labels[i]], float(probs[i]))
                )
            if (
                sum(1 for act, _, _ in triples if act == "Indian Penal Code")
                >= CLASSIFIER_TOP_N
            ):
                break
        return triples

    def _prefer_bns(self, matches: List[SectionMatch]) -> List[SectionMatch]:
        """Replace each IPC match by its BNS equivalent (official comparative table): the indexed BNS text when there is one, else the IPC text labelled with the BNS number."""
        from app.tools.fact_statutes import _translation
        from app.tools.unified_legal_rag import get_unified_rag_system

        ipc_to_bns = _translation()["ipc_bns"]["old_to_new"]
        chunks = get_unified_rag_system()._chunks
        out, seen = [], set()
        for m in matches:
            if m.act_name == "Indian Penal Code" and m.section in ipc_to_bns:
                bns = ipc_to_bns[m.section][0]
                cids = self._pinned_chunk_ids([("Bharatiya Nyaya Sanhita", bns)])
                if cids:
                    c = chunks[cids[0]]
                    m = replace(
                        m,
                        act_name=c.act_name,
                        section=bns,
                        title=f"{c.title} (formerly IPC § {m.section})",
                        punishment=_section_punishment(
                            self._full_section_text(c.act_name, bns, c.text)
                        )
                        or m.punishment,
                        definition=c.text,
                    )
                else:
                    m = replace(
                        m,
                        title=f"{m.title} (now BNS § {bns}; offences before 1 July 2024 stay under IPC)",
                    )
            key = (
                ("BNS", m.section)
                if m.act_name.startswith("Bharatiya Nyaya")
                else (m.act_name, m.section)
            )
            if key not in seen:
                seen.add(key)
                out.append(m)
        return out

    @staticmethod
    def _fits_report(chunk, crime_type: str) -> bool:
        """On a short report with a known crime type, a reranked section must be that crime's own, or score well and come from a related or untyped offence (cheating on a burglary report..."""
        types = _section_crime_types(chunk.act_name, chunk.section_number)
        if crime_type in types:
            return True
        if types and not any(_fits_crime_type_pair(t, crime_type) for t in types):
            return False
        return chunk.score >= REPORT_NOISE_MIN_SCORE

    def _full_section_text(self, act_name: str, section: str, fallback: str) -> str:
        """A long section is indexed as parts (…_p1, …_p2); later subsections, and their punishments, are only in the later parts."""
        from app.tools.unified_legal_rag import get_unified_rag_system

        chunks = get_unified_rag_system()._chunks
        cids = [c for c in self._pinned_chunk_ids([(act_name, section)]) if c in chunks]
        if len(cids) < 2:
            return fallback
        part = lambda cid: (
            int(cid.rsplit("_p", 1)[1]) if re.search(r"_p\d+$", cid) else 0
        )
        return " ".join(chunks[c].text for c in sorted(cids, key=part))

    def _pinned_chunk_ids(self, sections: List[tuple]) -> List[str]:
        """Chunk ids for the given (act, section) pairs."""
        if not sections:
            return []
        from app.tools.unified_legal_rag import get_unified_rag_system

        if self._pin_index is None:
            index: dict = {}
            for cid, chunk in get_unified_rag_system()._chunks.items():
                key = (chunk.act_name, str(chunk.section_number))
                index.setdefault(key, []).append(cid)
            self._pin_index = index
        return [
            cid
            for (act, section), cids in self._pin_index.items()
            for act_key, sec in sections
            if act_key in act and section == sec
            for cid in cids
        ]

    @property
    def domain_name(self) -> str:
        return "criminal"

    @property
    def pdf_subdir(self) -> str:
        return "criminal"

    def _parse_legal_sections(
        self, full_text: str, source_file: str
    ) -> List[LegalChunk]:
        """Criminal law parser with a per-Act punishment-clause filter."""
        base_chunks = super()._parse_legal_sections(full_text, source_file)

        stem = Path(source_file).stem.lower()
        if stem not in self.PUNISHMENT_FILTERED_ACTS or stem in self.INDEX_ALL_SECTIONS:
            return base_chunks

        filtered: List[LegalChunk] = []
        for chunk in base_chunks:
            if chunk.has_punishment or _CHARGEABLE_PUNISHMENT_RE.search(chunk.text):
                filtered.append(chunk)

        if len(filtered) < max(10, len(base_chunks) // 4):
            print(
                f"  [criminal] Punishment filter kept only {len(filtered)}/"
                f"{len(base_chunks)} sections of {source_file} — keeping all."
            )
            return base_chunks

        return filtered

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
        """Safe criminal vocabulary enhancement."""
        q = query.lower()
        terms: List[str] = []

        if any_word(q, ["stabbed", "slash", "cut with knife", "blade"]):
            terms.extend(["grievous hurt", "hurt", "dangerous weapon"])
        if any_word(q, ["beaten", "punched", "hit", "physically assaulted"]):
            terms.extend(["hurt", "voluntarily causing hurt"])
        if any_word(q, ["killed", "murdered", "dead", "death"]):
            terms.extend(["culpable homicide", "murder", "causing death"])

        if any_word(q, ["rape", "sexual assault", "molest"]):
            terms.extend(["rape", "sexual intent", "outraging modesty"])
        if any_word(q, ["stalking", "following woman", "monitor woman"]):
            terms.extend(["stalking", "following woman"])

        if any_word(q, ["kidnap", "abduct", "hostage", "ransom"]):
            terms.extend(["kidnapping", "abduction", "ransom"])

        if any_word(q, ["dowry", "498a", "cruelty by husband"]):
            terms.extend(["cruelty by husband", "dowry death", "abetment of suicide"])

        if any_word(q, ["forged", "forgery", "forge", "fake document"]):
            terms.extend(["forgery", "using forged document"])
        if any_word(q, ["cheated me", "cheated out of", "deceived me into"]):
            terms.extend(["cheating", "dishonestly inducing delivery of property"])

        if any_word(
            q,
            [
                "whatsapp",
                "electronic evidence",
                "digital evidence",
                "email as evidence",
                "chats admissible",
                "cctv",
                "call recording",
            ],
        ):
            terms.extend(
                [
                    "admissibility of electronic records",
                    "electronic record",
                    "certificate",
                ]
            )

        if any_word(q, ["fir", "police complaint", "cognizable", "arrest"]):
            terms.extend(["cognizable offence", "complaint", "investigation"])

        if any_word(q, ["bail", "anticipatory bail", "custody"]):
            terms.extend(["bail", "custody", "arrest"])

        if any_word(q, ["set fire", "arson", "burnt my house"]):
            terms.extend(["arson", "fire to property"])

        if any_word(q, ["broke into", "illegal entry", "trespassed into house"]):
            terms.extend(["criminal trespass", "house-breaking"])

        terms.extend(self._doctrine_terms(query))

        if terms:
            return query + " " + " ".join(terms)
        return query

    def _matched_doctrines(self, query: str) -> List[str]:
        """Doctrine families whose description is similar enough to the query."""
        from app.tools.unified_legal_rag import get_unified_rag_system

        embeddings = get_unified_rag_system().embeddings
        if embeddings is None:
            return []
        if self._doctrine_cache is None:
            names = list(_DOCTRINE_DESCRIPTIONS)
            vecs = np.array(
                embeddings.embed_documents(
                    [_DOCTRINE_DESCRIPTIONS[n][0] for n in names]
                )
            )
            self._doctrine_cache = (
                names,
                vecs / np.linalg.norm(vecs, axis=1, keepdims=True),
            )
        names, doc_vecs = self._doctrine_cache
        q = self._query_vector(query)
        if q is None:
            return []
        scores = doc_vecs @ q
        return [
            name
            for name, score in zip(names, scores)
            if score >= DOCTRINE_MATCH_MIN_SCORE
        ]

    def _doctrine_terms(self, query: str) -> List[str]:
        """Expansion terms for doctrine families whose description the query matches."""
        return [
            term
            for name in self._matched_doctrines(query)
            for term in _DOCTRINE_DESCRIPTIONS[name][1]
        ]

    def _build_search_query(
        self, query: str, crime_type: str, features: CrimeFeatures
    ) -> str:
        """Build an enhanced search with feature signals (criminal context only)."""
        parts = [query, query, query]

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

    async def retrieve_sections(
        self,
        query: str,
        crime_type: str = "",
        features: Optional[CrimeFeatures] = None,
        k: int = 2,
        offences_only: bool = False,
    ) -> RAGResult:
        """Full criminal RAG pipeline over the unified hybrid index (BM25 + dense + reranker, filtered to the criminal domain)."""
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

            classified = self._classified_sections(query)
            chunks = await unified._hybrid_search(
                search_query=search_query,
                rerank_query=query,
                k=OFFENCE_POOL if offences_only else k * 2,
                min_score=0.0,
                domains=OFFENCE_DOMAINS if offences_only else [self.domain_name],
                extra_candidates=self._pinned_chunk_ids(
                    [(a, s) for a, s, _ in classified]
                ),
            )
            if offences_only:
                chunks = [
                    c
                    for c in chunks
                    if _is_offence_section(c.act_name, c.section_number)
                ]

            short = len(query.split()) < CLASSIFIER_LONG_QUERY_WORDS
            lead_sections = [
                (a, s)
                for a, s, p in classified
                if a == "Indian Penal Code"
                and (not short or p >= CLASSIFIER_SHORT_MIN_PROB)
                and (not offences_only or _is_offence_section(a, s))
                and not (
                    offences_only and short and not _fits_report_lead(s, p, crime_type)
                )
            ][:CLASSIFIER_FIRST]

            core = (
                CRIME_TYPE_SECTIONS.get(crime_type) if offences_only and short else None
            )
            if (
                core
                and crime_type == "property_damage"
                and any_word(query, ANIMAL_WORDS)
            ):
                core = "429"
            if core and ("Indian Penal Code", core) not in lead_sections:
                lead_sections = [("Indian Penal Code", core)] + lead_sections[
                    : CLASSIFIER_FIRST - 1
                ]
            typed_report = offences_only and short and crime_type not in ("", "general")
            reranked_score = {(c.act_name, c.section_number): c.score for c in chunks}
            lead = []
            for pair in lead_sections:
                cids = sorted(self._pinned_chunk_ids([pair]))
                if cids:
                    chunk = unified._chunks[cids[0]]
                    lead.append(
                        replace(
                            chunk,
                            score=reranked_score.get(
                                (chunk.act_name, chunk.section_number), 0.5
                            ),
                        )
                    )
            chunks = lead + chunks
            lead_rank = {(c.act_name, c.section_number): i for i, c in enumerate(lead)}

            matches: List[SectionMatch] = []
            seen: set = set()

            for chunk in chunks:
                sec_num = chunk.section_number
                if not sec_num or (chunk.act_name, sec_num) in seen:
                    continue
                seen.add((chunk.act_name, sec_num))

                requires_punishment = (
                    Path(chunk.source_file).stem.lower()
                    in self.PUNISHMENT_FILTERED_ACTS
                )

                full_text = self._full_section_text(chunk.act_name, sec_num, chunk.text)
                punishment = _section_punishment(full_text) or (
                    chunk.text[:250]
                    if chunk.has_punishment
                    or _CHARGEABLE_PUNISHMENT_RE.search(chunk.text)
                    else ""
                )
                is_lead = (chunk.act_name, sec_num) in lead_rank
                if (
                    requires_punishment
                    and not is_lead
                    and (not punishment or len(punishment) < 10)
                ):
                    continue
                if (
                    not is_lead
                    and typed_report
                    and not self._fits_report(chunk, crime_type)
                ):
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
                        act_name=chunk.act_name,
                    )
                )

            matches.sort(
                key=lambda m: (
                    lead_rank.get((m.act_name, m.section), len(lead_rank)),
                    -m.confidence,
                )
            )
            if offences_only:
                matches = self._prefer_bns(matches)
            matches = matches[:k]

            avg_conf = (
                sum(m.confidence for m in matches) / len(matches) if matches else 0.0
            )
            sources = list({f"{m.act_name} § {m.section}" for m in matches})

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
                f"{match.act_name} § {match.section} — {match.title}\n"
                f"{match.definition}\nPunishment: {match.punishment}"
            )
            sources.append(f"{match.act_name} § {match.section}")
        return CrimeContext(
            crime_type=result.crime_type,
            relevant_passages=passages,
            sources=sources,
            confidence=result.confidence,
        )


_criminal_rag: Optional[CriminalRAGSystem] = None


def get_criminal_rag_system() -> CriminalRAGSystem:
    """Get or create the CriminalRAGSystem singleton."""
    global _criminal_rag
    if _criminal_rag is None:
        _criminal_rag = CriminalRAGSystem()
    return _criminal_rag
