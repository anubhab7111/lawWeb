"""
Legal RAG system for IPC section retrieval.

Implements a legally-aware pipeline:
1. Source Canonicalization — parse IPC bare act into atomic section chunks
2. Legal-Aware Chunking — one IPC section = one chunk (the atomic legal unit)
3. Vector Retrieval — semantic search for relevant sections
4. Legal Filtering — remove definition-only sections, apply minimality principle
5. Return only chargeable punishment sections
"""

import asyncio
import json
import os
import pickle
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_community.vectorstores import FAISS
    from langchain_core.documents import Document
    from langchain_ollama import OllamaEmbeddings

    HAS_RAG_DEPS = True
except ImportError as e:
    HAS_RAG_DEPS = False
    print(f"RAG dependencies import error: {e}")


# ─────────────────────────────────────────────────────────────
# 1. Data Models
# ─────────────────────────────────────────────────────────────


@dataclass
class IPCSection:
    """Canonical representation of one IPC section."""

    section_id: str  # e.g. "IPC_302"
    section_number: str  # e.g. "302"
    title: str  # e.g. "Punishment for murder"
    definition: str  # Full section text
    punishment: str  # Extracted punishment clause
    is_definition_only: bool = False  # True if no punishment clause
    ingredients: List[str] = field(default_factory=list)


@dataclass
class CrimeFeatures:
    """Extracted legal signals from a crime description."""

    violence: bool = False
    death: bool = False
    weapon: str = ""
    intent: str = "unknown"  # intentional, reckless, negligent, unknown
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
    """A matched IPC section with confidence and reasoning."""

    section: str
    title: str
    confidence: float
    reasons: List[str]
    punishment: str
    definition: str
    review_required: bool = False


@dataclass
class RAGResult:
    """Final output of the RAG pipeline."""

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
# 2. IPC Section Parser
# ─────────────────────────────────────────────────────────────


def _extract_punishment(text: str, max_len: int = 250) -> str:
    """Extract punishment clause from section text, truncated to max_len."""
    # Try "shall be punished with"
    match = re.search(
        r"shall be punished with\s+(.+?)(?:\.\s*[A-Z]|\.\s*$|$)",
        text,
        re.DOTALL | re.IGNORECASE,
    )
    if match:
        punishment = match.group(1).strip()
        punishment = re.sub(r"\s+", " ", punishment)
        # Stop at various delimiters
        for stopper in ["Illustration", "Explanation", "STATE AMENDMENT", " Of "]:
            idx = punishment.find(stopper)
            if idx > 0:
                punishment = punishment[:idx].strip().rstrip(".")
        # Truncate to max length at word boundary
        if len(punishment) > max_len:
            punishment = punishment[:max_len].rsplit(" ", 1)[0] + "..."
        return punishment

    # Try "shall be punishable"
    match = re.search(
        r"shall[^.]*?be punishable\s+(.+?)(?:\.\s*[A-Z]|\.\s*$|$)",
        text,
        re.DOTALL | re.IGNORECASE,
    )
    if match:
        punishment = re.sub(r"\s+", " ", match.group(1).strip())
        if len(punishment) > max_len:
            punishment = punishment[:max_len].rsplit(" ", 1)[0] + "..."
        return punishment

    return ""


def _extract_ingredients(text: str) -> List[str]:
    """Extract actus reus elements from section text."""
    ingredients = []
    whoever_match = re.search(
        r"Whoever\s+(.+?)(?:shall be punished|$)", text, re.DOTALL | re.IGNORECASE
    )
    if whoever_match:
        action_text = whoever_match.group(1).strip()
        parts = re.split(r",\s+(?:and|or)\s+|,\s+", action_text)
        for part in parts:
            part = part.strip()
            if 10 < len(part) < 200:
                ingredients.append(part)
    return ingredients[:5]


def _parse_ipc_sections(full_text: str) -> List[IPCSection]:
    """
    Parse IPC bare act text into canonical section objects.
    Each section = one atomic legal unit.
    """
    sections: List[IPCSection] = []

    header_pattern = re.compile(
        r"\n\s*(\d{1,3}[A-Z]{0,2})\.\s+([^.—\n]+(?:\.[^—\n]*)?)[.—]\s*", re.MULTILINE
    )

    matches = list(header_pattern.finditer(full_text))
    if not matches:
        return sections

    for i, match in enumerate(matches):
        section_number = match.group(1).strip()
        title = match.group(2).strip().rstrip(".")

        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)
        raw_text = full_text[start:end].strip()

        # Clean up
        raw_text = re.sub(r"\n\d{1,3}\s*\n", "\n", raw_text)
        raw_text = re.sub(r"\s+", " ", raw_text)

        if len(raw_text) < 30:
            continue

        punishment = _extract_punishment(raw_text)
        ingredients = _extract_ingredients(raw_text)

        # Determine if this is a definition-only section (no punishment clause)
        is_definition_only = not punishment or len(punishment) < 10

        ipc_section = IPCSection(
            section_id=f"IPC_{section_number}",
            section_number=section_number,
            title=title,
            definition=raw_text,
            punishment=punishment,
            is_definition_only=is_definition_only,
            ingredients=ingredients,
        )
        sections.append(ipc_section)

    return sections


# ─────────────────────────────────────────────────────────────
# 3. Crime Feature Extraction
# ─────────────────────────────────────────────────────────────


def extract_crime_features(text: str) -> CrimeFeatures:
    """
    Convert crime description into structured legal signals.
    These features are used to enhance the search query.
    """
    text_lower = text.lower()
    features = CrimeFeatures()

    # Violence
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
    features.violence = any(w in text_lower for w in violence_words)

    # Death
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
    features.death = any(w in text_lower for w in death_words)

    # Weapon
    weapons = {
        "knife": ["knife", "stabbed", "stabbing", "blade"],
        "gun": ["gun", "shot", "shooting", "firearm", "pistol", "rifle", "bullet"],
        "acid": ["acid attack", "acid thrown", "acid"],
        "stick": ["stick", "rod", "bat", "lathi"],
        "explosive": ["bomb", "explosive", "blast"],
        "vehicle": ["run over", "hit by car", "vehicle"],
    }
    for weapon, keywords in weapons.items():
        if any(w in text_lower for w in keywords):
            features.weapon = weapon
            features.violence = True
            break

    # Intent
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
    if any(w in text_lower for w in intentional_words):
        features.intent = "intentional"
    elif any(w in text_lower for w in reckless_words):
        features.intent = "reckless"
    elif features.death and not features.violence:
        features.intent = "negligent"
    elif features.violence or features.death:
        features.intent = "intentional"
    else:
        features.intent = "unknown"

    # Property
    property_words = [
        "stolen",
        "theft",
        "robbed",
        "took my",
        "snatched",
        "missing property",
        "lost money",
        "cheated money",
        "fraud",
        "scam",
        "misappropriated",
        "embezzled",
        "land taken",
        "property taken",
        "grabbed",
        "encroached",
        "land dispute",
        "illegally taken",
    ]
    features.property_loss = any(w in text_lower for w in property_words)

    # Sexual
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
    features.sexual = any(w in text_lower for w in sexual_words)

    # Fraud
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
    features.fraud = any(w in text_lower for w in fraud_words)

    # Domestic
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
    features.domestic = any(w in text_lower for w in domestic_words)

    # Trespass
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
    features.trespass = any(w in text_lower for w in trespass_words)

    # Fire / Arson
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
    features.fire = any(w in text_lower for w in fire_words)

    # Kidnapping
    kidnap_words = [
        "kidnap",
        "abduct",
        "ransom",
        "taken away",
        "missing child",
        "hostage",
    ]
    features.kidnapping = any(w in text_lower for w in kidnap_words)

    # Threat / Criminal Intimidation
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
    features.threat = any(w in text_lower for w in threat_words)

    return features


# ─────────────────────────────────────────────────────────────
# 4. CrimeRAGSystem
# ─────────────────────────────────────────────────────────────


class CrimeRAGSystem:
    """
    Legal RAG system:
    1. Parse IPC bare act into canonical sections
    2. Embed [section_number + title + definition]
    3. Vector retrieval based on semantic similarity
    4. Return sections with punishment from PDF
    """

    def __init__(self, data_dir: str = "app/data"):
        self.data_dir = Path(data_dir)
        self.vector_store_path = self.data_dir / "vector_store.pkl"
        self.faiss_index_path = self.data_dir / "faiss_index"
        self.sections_cache_path = self.data_dir / "ipc_sections.json"
        self.vector_store = None
        self.embeddings = None
        self.initialized = False
        self._init_lock = asyncio.Lock()
        self._sections: Dict[str, IPCSection] = {}

    async def initialize(self) -> bool:
        """Initialize the RAG system."""
        if self.initialized:
            return True

        if not HAS_RAG_DEPS:
            return False

        async with self._init_lock:
            if self.initialized:
                return True

            try:
                self.embeddings = OllamaEmbeddings(
                    model="nomic-embed-text",
                    base_url="http://localhost:11434",
                )

                if await self._should_rebuild():
                    print("Building legal vector store from IPC bare act...")
                    await self._build_vectorstore()
                else:
                    print("Loading existing legal vector store...")
                    await self._load_vectorstore()
                    self._load_sections_cache()

                self.initialized = True
                print(
                    f"Legal RAG ready: {len(self._sections)} canonical IPC sections indexed"
                )
                return True

            except Exception as e:
                print(f"Error initializing legal RAG system: {e}")
                import traceback

                traceback.print_exc()
                return False

    async def _should_rebuild(self) -> bool:
        """Check if vector store needs rebuilding."""
        if not self.faiss_index_path.exists() or not self.vector_store_path.exists():
            return True
        if not self.sections_cache_path.exists():
            return True
        vectorstore_mtime = self.vector_store_path.stat().st_mtime
        for src_file in list(self.data_dir.rglob("*.pdf")) + list(
            self.data_dir.rglob("*.txt")
        ):
            if src_file.stat().st_mtime > vectorstore_mtime:
                return True
        return False

    async def _build_vectorstore(self):
        """
        Build canonical legal vector store:
        1. Load IPC bare act PDF
        2. Parse into canonical sections
        3. Extract punishment for each section
        4. Embed [section_number + title + definition]
        """
        ipc_pdf = (
            self.data_dir / "bare_acts" / "criminal" / "Indian_Penal_Code_1860.pdf"
        )
        if not ipc_pdf.exists():
            # Fallback: search recursively for any PDF containing "Penal" or "IPC"
            pdf_files = [
                f
                for f in self.data_dir.rglob("*.pdf")
                if "penal" in f.name.lower() or "ipc" in f.name.lower()
            ]
            if not pdf_files:
                # Last resort: any PDF in the criminal directory
                pdf_files = list(
                    (self.data_dir / "bare_acts" / "criminal").rglob("*.pdf")
                )
            ipc_pdf = pdf_files[0] if pdf_files else None

        if not ipc_pdf:
            print("ERROR: No IPC PDF found!")
            return

        # Load and extract text (skip TOC pages)
        loader = PyPDFLoader(str(ipc_pdf))
        all_pages = loader.load()
        print(f"Loaded {len(all_pages)} pages from {ipc_pdf.name}")

        toc_end = 13
        for i, page in enumerate(all_pages):
            if "shall be punished" in page.page_content.lower() and i > 5:
                toc_end = i
                break

        full_text = "\n".join(page.page_content for page in all_pages[toc_end:])

        # Parse into canonical IPC sections
        parsed_sections = _parse_ipc_sections(full_text)
        print(f"Parsed {len(parsed_sections)} canonical IPC sections")

        # Store in memory and cache
        self._sections = {s.section_number: s for s in parsed_sections}
        self._save_sections_cache()

        # Create embedding documents
        documents = []
        for section in parsed_sections:
            # Embed: section number + title + definition
            embed_text = (
                f"Section {section.section_number}. {section.title}. "
                f"{section.definition}"
            )

            meta = {
                "source": ipc_pdf.name,
                "section_number": section.section_number,
                "section_id": section.section_id,
                "title": section.title,
                "punishment": section.punishment,
                "type": "ipc_section",
            }

            documents.append(
                Document(
                    page_content=embed_text,
                    metadata=meta,
                )
            )

        print(f"Created {len(documents)} embedding documents")

        loop = asyncio.get_event_loop()
        self.vector_store = await loop.run_in_executor(
            None, lambda: FAISS.from_documents(documents, self.embeddings)
        )

        await self._save_vectorstore()

    def _save_sections_cache(self):
        """Save parsed sections to JSON cache."""
        cache = {}
        for num, sec in self._sections.items():
            cache[num] = {
                "section_id": sec.section_id,
                "section_number": sec.section_number,
                "title": sec.title,
                "definition": sec.definition[:500],
                "punishment": sec.punishment,
                "is_definition_only": sec.is_definition_only,
                "ingredients": sec.ingredients,
            }
        with open(self.sections_cache_path, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2, ensure_ascii=False)
        print(f"Sections cache saved: {len(cache)} sections")

    def _load_sections_cache(self):
        """Load sections from JSON cache."""
        if not self.sections_cache_path.exists():
            return
        try:
            with open(self.sections_cache_path, "r", encoding="utf-8") as f:
                cache = json.load(f)
            for num, data in cache.items():
                self._sections[num] = IPCSection(
                    section_id=data["section_id"],
                    section_number=data["section_number"],
                    title=data["title"],
                    definition=data["definition"],
                    punishment=data["punishment"],
                    is_definition_only=data.get(
                        "is_definition_only", not data["punishment"]
                    ),
                    ingredients=data.get("ingredients", []),
                )
            print(f"Loaded {len(self._sections)} sections from cache")
        except Exception as e:
            print(f"Error loading sections cache: {e}")

    async def _save_vectorstore(self):
        """Save vector store to disk."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: self.vector_store.save_local(str(self.faiss_index_path)),
        )
        with open(self.vector_store_path, "wb") as f:
            pickle.dump({"timestamp": os.path.getmtime(self.data_dir)}, f)
        print(f"Vector store saved to {self.faiss_index_path}")

    async def _load_vectorstore(self):
        """Load vector store from disk."""
        try:
            loop = asyncio.get_event_loop()
            self.vector_store = await loop.run_in_executor(
                None,
                lambda: FAISS.load_local(
                    str(self.faiss_index_path),
                    self.embeddings,
                    allow_dangerous_deserialization=True,
                ),
            )
            print(f"Vector store loaded from {self.faiss_index_path}")
        except Exception as e:
            print(f"Error loading vector store: {e}. Rebuilding...")
            await self._build_vectorstore()

    # ─────────────────────────────────────────────────────────
    # Main Retrieval Pipeline
    # ─────────────────────────────────────────────────────────

    async def retrieve_sections(
        self,
        query: str,
        crime_type: str = "",
        features: Optional[CrimeFeatures] = None,
        k: int = 2,  # Default to 2 for legal minimality
    ) -> RAGResult:
        """
        Vector-based RAG retrieval with legal filtering:
        1. Preprocess query to extract key legal terms
        2. Build targeted search query
        3. Vector similarity search
        4. Apply strict relevance threshold (0.40+)
        5. Filter out definition-only sections
        6. Return only top-k chargeable sections
        """
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
            # Preprocess query to extract key legal concepts
            preprocessed_query = self._preprocess_query(query, features)

            # Build enhanced search query
            search_query = self._build_search_query(
                preprocessed_query, crime_type, features
            )

            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None,
                lambda: self.vector_store.similarity_search_with_score(
                    search_query,
                    k=20,  # Get more candidates for strict filtering
                ),
            )

            # Process results with STRICT legal filtering
            matches: List[SectionMatch] = []
            seen_sections: set = set()

            # STRICT RELEVANCE THRESHOLD - only keep highly relevant results
            MIN_CONFIDENCE_THRESHOLD = 0.40

            for doc, distance in results:
                section_number = doc.metadata.get("section_number", "")
                if not section_number or section_number in seen_sections:
                    continue
                seen_sections.add(section_number)

                # Get cached section data
                cached = self._sections.get(section_number)

                # LEGAL FILTER 1: Skip definition-only sections (no punishment)
                if cached and cached.is_definition_only:
                    continue

                punishment = (
                    cached.punishment if cached else doc.metadata.get("punishment", "")
                )

                # LEGAL FILTER 2: Must have a punishment clause to be chargeable
                if not punishment or len(punishment) < 10:
                    continue

                # FAISS L2 distance → similarity score
                confidence = max(0.0, 1.0 - (distance / 2.0))

                # RELEVANCE FILTER 3: Apply strict threshold
                if confidence < MIN_CONFIDENCE_THRESHOLD:
                    continue

                title = cached.title if cached else doc.metadata.get("title", "")
                definition = cached.definition if cached else doc.page_content[:300]

                matches.append(
                    SectionMatch(
                        section=section_number,
                        title=title,
                        confidence=round(min(confidence, 1.0), 2),
                        reasons=["Chargeable section with punishment"],
                        punishment=punishment,
                        definition=definition,
                        review_required=confidence < 0.6,
                    )
                )

                # LEGAL MINIMALITY: Stop after we have enough high-quality matches
                if len(matches) >= k * 2:  # Get 2x to allow for best selection
                    break

            # Sort by confidence and take top k
            matches.sort(key=lambda m: m.confidence, reverse=True)
            matches = matches[:k]

            avg_conf = (
                sum(m.confidence for m in matches) / len(matches) if matches else 0.0
            )
            sources = list(set(f"IPC Section {m.section}" for m in matches))

            return RAGResult(
                crime_type=crime_type or "general",
                ipc_sections=matches,
                sources=sources,
                confidence=round(avg_conf, 2),
            )

        except Exception as e:
            print(f"Error in legal retrieval: {e}")
            import traceback

            traceback.print_exc()
            return RAGResult(
                crime_type=crime_type or "general",
                ipc_sections=[],
                sources=[],
                confidence=0.0,
            )

    def _preprocess_query(self, query: str, features: CrimeFeatures) -> str:
        """
        Extract key legal concepts from query for better retrieval.
        Focus on the CORE legal issue, not just keywords.
        """
        query_lower = query.lower()

        # Extract explicit legal terms (acts, sections, procedures)
        legal_terms = []

        # Bail-related queries - especially economic offences
        if "bail" in query_lower or "anticipatory" in query_lower:
            legal_terms.extend(["bail", "custody", "arrest"])
            if "economic" in query_lower:
                # Economic offences = white-collar crimes - emphasize these heavily
                legal_terms.extend(
                    [
                        "cheating",
                        "cheating",
                        "dishonestly inducing delivery",
                        "criminal breach of trust",
                        "criminal breach of trust",
                        "forgery",
                        "forgery of valuable security",
                        "using forged document",
                        "fraud",
                        "misappropriation",
                    ]
                )

        # FIR/procedure queries
        if "fir" in query_lower or "quash" in query_lower:
            legal_terms.extend(["cognizable", "complaint", "investigation"])

        # Cryptocurrency / blockchain / digital asset queries
        if any(
            term in query_lower
            for term in [
                "cryptocurrency",
                "crypto",
                "bitcoin",
                "blockchain",
                "digital asset",
            ]
        ):
            # Crypto crimes map to traditional fraud/forgery sections
            legal_terms.extend(
                [
                    "cheating",
                    "cheating",
                    "dishonestly inducing delivery of property",
                    "criminal breach of trust",
                    "criminal breach of trust",
                    "forgery",
                    "using as genuine a forged document",
                    "fraud",
                    "conspiracy",
                    "conspiracy to cheat",
                ]
            )

        # AI / technology / automated system liability
        if any(
            term in query_lower
            for term in [
                "ai system",
                "artificial intelligence",
                "ai",
                "automated",
                "algorithm",
            ]
        ):
            if (
                "financial loss" in query_lower
                or "loss" in query_lower
                or "liable" in query_lower
            ):
                # AI causing loss = cheating or negligence
                legal_terms.extend(
                    [
                        "cheating",
                        "cheating",
                        "dishonestly inducing",
                        "causing death by negligence",
                        "rash or negligent act",
                    ]
                )

        # Privacy / photo / image sharing without consent
        if any(
            term in query_lower for term in ["photo", "image", "picture", "video"]
        ) and (
            "online" in query_lower
            or "shared" in query_lower
            or "consent" in query_lower
        ):
            # Photo sharing = voyeurism, defamation, insulting modesty
            legal_terms.extend(
                [
                    "voyeurism",
                    "voyeurism",
                    "watching or capturing image of woman",
                    "defamation",
                    "defamation",
                    "insult intended to provoke breach of peace",
                    "word gesture or act intended to insult modesty",
                    "criminal intimidation",
                ]
            )

        # Marital/domestic
        if "marital" in query_lower or "rape" in query_lower or "spouse" in query_lower:
            legal_terms.extend(
                [
                    "rape",
                    "sexual intercourse",
                    "consent",
                    "sexual intercourse by husband upon wife",
                ]
            )

        # General fraud/scam/financial queries not caught above
        if (
            any(term in query_lower for term in ["fraud", "scam", "financial", "money"])
            and not legal_terms
        ):
            legal_terms.extend(["cheating", "criminal breach of trust", "forgery"])

        # Combine original query with extracted terms
        if legal_terms:
            return query + " " + " ".join(legal_terms)
        return query

    def _build_search_query(
        self, query: str, crime_type: str, features: CrimeFeatures
    ) -> str:
        """
        Build targeted search query emphasizing the ORIGINAL QUERY.
        Only add minimal, highly-relevant legal context.
        """
        # PRIMARY: Use the preprocessed query (3x weight)
        parts = [query, query, query]

        # Add ONLY the most relevant feature keywords (not generic "punishment")
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

        if features.weapon and features.weapon != "":
            parts.append(f"{features.weapon} dangerous weapon")

        return " ".join(parts)

    # ─────────────────────────────────────────────────────────
    # Legacy-compatible interface
    # ─────────────────────────────────────────────────────────

    async def retrieve_context(
        self, query: str, k: int = 5, crime_type: str = ""
    ) -> CrimeContext:
        """Legacy-compatible interface."""
        features = extract_crime_features(query)
        result = await self.retrieve_sections(
            query, crime_type=crime_type, features=features, k=k
        )

        passages = []
        sources = []
        for match in result.ipc_sections:
            passage = (
                f"IPC Section {match.section} — {match.title}\n"
                f"{match.definition}\n"
                f"Punishment: {match.punishment}"
            )
            passages.append(passage)
            sources.append(f"IPC Section {match.section}")

        return CrimeContext(
            crime_type=result.crime_type,
            relevant_passages=passages,
            sources=sources,
            confidence=result.confidence,
        )


# ─────────────────────────────────────────────────────────────
# Singleton
# ─────────────────────────────────────────────────────────────

_rag_system: Optional[CrimeRAGSystem] = None


def get_rag_system() -> CrimeRAGSystem:
    """Get or create the RAG system singleton."""
    global _rag_system
    if _rag_system is None:
        _rag_system = CrimeRAGSystem()
    return _rag_system


# ─────────────────────────────────────────────────────────────
# Backward-compatibility shim — Multi-Domain RAG Migration
# ─────────────────────────────────────────────────────────────
# All new code should import directly from:
#   app.tools.criminal_rag  →  CriminalRAGSystem, get_criminal_rag_system
#   app.tools.civil_rag     →  CivilRAGSystem, get_civil_rag_system
#   app.tools.constitutional_rag → ConstitutionalRAGSystem, get_constitutional_rag_system
#
# This shim keeps any remaining callers of get_rag_system() working
# without modification. get_rag_system() now delegates to the new
# CriminalRAGSystem so the old monolithic vector store is no longer used.
# ─────────────────────────────────────────────────────────────
try:
    from app.tools.criminal_rag import (  # noqa: F401, E402
        CriminalRAGSystem,
        get_criminal_rag_system,
    )

    def get_rag_system() -> "CriminalRAGSystem":  # type: ignore[misc]
        """Backward-compat: returns CriminalRAGSystem (replaces old monolithic store)."""
        return get_criminal_rag_system()

except ImportError:
    # Fallback if criminal_rag is not yet importable (e.g., missing deps)
    pass
