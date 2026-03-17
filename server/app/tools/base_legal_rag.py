"""
base_legal_rag.py — Abstract Base for Multi-Domain Legal RAG Systems

Provides the shared FAISS infrastructure for all domain-specific RAG classes:
  - CriminalRAGSystem  (criminal_rag.py)
  - CivilRAGSystem     (civil_rag.py)
  - ConstitutionalRAGSystem (constitutional_rag.py)

Design principles
-----------------
1. One FAISS index per legal domain → complete isolation between domains.
2. Generic _parse_legal_sections() preserves ALL sections (including
   definition-only ones) so civil and constitutional law can be indexed.
3. Subclasses override _parse_legal_sections() and/or _preprocess_query()
   to apply domain-appropriate filtering.
4. Each index is stored under  app/data/faiss_index/<domain>/
   so indexes don't collide.
"""

import asyncio
import json
import os
import pickle
import re
from abc import ABC, abstractmethod
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
# Shared Data Models
# ─────────────────────────────────────────────────────────────


@dataclass
class LegalChunk:
    """One atomic legal provision retrieved from the vector store."""

    chunk_id: str          # e.g. "ICA_10", "CON_ART21", "CPC_151"
    domain: str            # "criminal" | "civil" | "constitutional"
    act_name: str          # e.g. "Indian Contract Act, 1872"
    section_number: str    # e.g. "10", "21", "302"
    title: str             # e.g. "What agreements are contracts"
    text: str              # Full section / Article text
    source_file: str       # PDF filename
    score: float = 0.0     # Cosine similarity (higher = more relevant)
    has_punishment: bool = False  # True only for criminal sections


@dataclass
class LegalContext:
    """Aggregated retrieval result from a domain RAG system."""

    domain: str
    query: str
    chunks: List[LegalChunk] = field(default_factory=list)
    sources: List[str] = field(default_factory=list)
    confidence: float = 0.0


# ─────────────────────────────────────────────────────────────
# Shared Helpers
# ─────────────────────────────────────────────────────────────


def _extract_punishment(text: str, max_len: int = 250) -> str:
    """Extract a punishment clause from section text, if present."""
    match = re.search(
        r"shall be punished with\s+(.+?)(?:\.\s*[A-Z]|\.\s*$|$)",
        text,
        re.DOTALL | re.IGNORECASE,
    )
    if match:
        punishment = re.sub(r"\s+", " ", match.group(1).strip())
        for stopper in ["Illustration", "Explanation", "STATE AMENDMENT", " Of "]:
            idx = punishment.find(stopper)
            if idx > 0:
                punishment = punishment[:idx].strip().rstrip(".")
        if len(punishment) > max_len:
            punishment = punishment[:max_len].rsplit(" ", 1)[0] + "..."
        return punishment

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


def _infer_act_name(filename: str) -> str:
    """Derive a human-readable Act name from its PDF filename."""
    stem = Path(filename).stem.replace("_", " ")
    # Strip trailing year artifacts like "1872" already in the stem
    stem = re.sub(r"\s+\d{4}$", "", stem)
    return stem


# ─────────────────────────────────────────────────────────────
# Abstract Base Class
# ─────────────────────────────────────────────────────────────


class BaseLegalRAGSystem(ABC):
    """
    Abstract base for domain-specific legal RAG systems.

    Subclasses MUST implement:
        domain_name     (str property)  — e.g. "criminal"
        pdf_subdir      (str property)  — subdirectory under bare_acts/

    Subclasses MAY override:
        _parse_legal_sections()  — to apply domain-specific chunking rules
        _preprocess_query()      — to add domain vocabulary to the query
    """

    # ── Subclass contracts ──────────────────────────────────────

    @property
    @abstractmethod
    def domain_name(self) -> str:
        """Short domain identifier, e.g. 'criminal', 'civil', 'constitutional'."""
        ...

    @property
    @abstractmethod
    def pdf_subdir(self) -> str:
        """Subdirectory under app/data/bare_acts/ holding this domain's PDFs."""
        ...

    # ── Init ────────────────────────────────────────────────────

    def __init__(self, data_dir: str = "app/data"):
        self.data_dir = Path(data_dir)
        self._bare_acts_dir = self.data_dir / "bare_acts" / self.pdf_subdir
        # Each domain gets its own FAISS index directory
        self._faiss_dir = self.data_dir / "faiss_index" / self.domain_name
        self._meta_path = self._faiss_dir / "meta.pkl"
        self._cache_path = self._faiss_dir / "sections.json"

        self.vector_store: Optional[Any] = None
        self.embeddings: Optional[Any] = None
        self.initialized: bool = False
        self._init_lock = asyncio.Lock()
        self._chunks: Dict[str, LegalChunk] = {}  # chunk_id → LegalChunk

    # ── Public API ───────────────────────────────────────────────

    async def initialize(self) -> bool:
        """Initialize (or resume from cache) this domain's vector store."""
        if self.initialized:
            return True
        if not HAS_RAG_DEPS:
            print(f"[{self.domain_name}] RAG dependencies not available.")
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
                    print(f"[{self.domain_name}] Building vector store from PDFs …")
                    await self._build_vectorstore()
                else:
                    print(f"[{self.domain_name}] Loading cached vector store …")
                    await self._load_vectorstore()
                    self._load_chunk_cache()

                self.initialized = True
                print(
                    f"[{self.domain_name}] RAG ready — "
                    f"{len(self._chunks)} legal chunks indexed."
                )
                return True
            except Exception as e:
                print(f"[{self.domain_name}] Initialization error: {e}")
                import traceback; traceback.print_exc()
                return False

    async def retrieve(
        self,
        query: str,
        k: int = 4,
        min_score: float = 0.30,
    ) -> LegalContext:
        """
        Core retrieval: semantic search over the domain's vector store.

        Args:
            query:     User query (will be preprocessed before embedding).
            k:         Maximum chunks to return.
            min_score: Minimum cosine similarity (FAISS L2 converted).

        Returns:
            LegalContext with matched chunks sorted by score.
        """
        if not self.initialized:
            await self.initialize()

        if not self.initialized or not self.vector_store:
            return LegalContext(domain=self.domain_name, query=query)

        try:
            search_query = self._preprocess_query(query)

            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None,
                lambda: self.vector_store.similarity_search_with_score(
                    search_query, k=k * 4
                ),
            )

            chunks: List[LegalChunk] = []
            seen: set = set()

            for doc, distance in results:
                chunk_id = doc.metadata.get("chunk_id", "")
                if not chunk_id or chunk_id in seen:
                    continue
                seen.add(chunk_id)

                # Convert L2 distance to a [0, 1] score (higher = better)
                score = max(0.0, 1.0 - (distance / 2.0))
                if score < min_score:
                    continue

                # Retrieve rich object from cache or reconstruct from metadata
                cached = self._chunks.get(chunk_id)
                if cached:
                    chunk = LegalChunk(
                        chunk_id=cached.chunk_id,
                        domain=cached.domain,
                        act_name=cached.act_name,
                        section_number=cached.section_number,
                        title=cached.title,
                        text=cached.text,
                        source_file=cached.source_file,
                        has_punishment=cached.has_punishment,
                        score=round(min(score, 1.0), 3),
                    )
                else:
                    chunk = LegalChunk(
                        chunk_id=chunk_id,
                        domain=self.domain_name,
                        act_name=doc.metadata.get("act_name", ""),
                        section_number=doc.metadata.get("section_number", ""),
                        title=doc.metadata.get("title", ""),
                        text=doc.page_content[:500],
                        source_file=doc.metadata.get("source", ""),
                        has_punishment=bool(doc.metadata.get("punishment", "")),
                        score=round(min(score, 1.0), 3),
                    )
                chunks.append(chunk)

                if len(chunks) >= k:
                    break

            chunks.sort(key=lambda c: c.score, reverse=True)
            avg_conf = sum(c.score for c in chunks) / len(chunks) if chunks else 0.0
            sources = list({f"{c.act_name} § {c.section_number}" for c in chunks})

            return LegalContext(
                domain=self.domain_name,
                query=query,
                chunks=chunks,
                sources=sources,
                confidence=round(avg_conf, 3),
            )

        except Exception as e:
            print(f"[{self.domain_name}] Retrieval error: {e}")
            return LegalContext(domain=self.domain_name, query=query)

    # ── Domain hooks (override in subclasses) ──────────────────

    def _preprocess_query(self, query: str) -> str:
        """
        Optionally expand the query with domain-specific vocabulary.
        Base implementation returns the query unchanged.
        Subclasses override for domain-aware enhancement.
        """
        return query

    def _parse_legal_sections(
        self, full_text: str, source_file: str
    ) -> List[LegalChunk]:
        """
        Generic legal section parser.

        Splits text on numbered section headers (e.g. "10. What agreements …")
        and creates one LegalChunk per section.

        KEY DIFFERENCE from the old crime_rag.py parser:
        - NO punishment-clause filter (LEGAL FILTER 1 / LEGAL FILTER 2 are GONE).
        - Definition-only sections like Indian Contract Act § 10 ARE indexed.
        - Subclasses can apply stricter filters for their domain.
        """
        chunks: List[LegalChunk] = []
        act_name = _infer_act_name(source_file)

        # Match  "  10.  What agreements are contracts"  style headers
        header_pattern = re.compile(
            r"\n\s*(\d{1,3}[A-Z]{0,2})\.\s+([^.\n\u2014]{3,}?)(?:[.\u2014])\s*",
            re.MULTILINE,
        )
        matches = list(header_pattern.finditer(full_text))

        if not matches:
            # Fallback: treat the entire text as a single chunk
            if len(full_text.strip()) > 50:
                chunk_id = f"{self.domain_name.upper()[:3]}_{Path(source_file).stem}_FULL"
                chunks.append(
                    LegalChunk(
                        chunk_id=chunk_id,
                        domain=self.domain_name,
                        act_name=act_name,
                        section_number="",
                        title=act_name,
                        text=full_text[:2000],
                        source_file=source_file,
                        has_punishment=bool(_extract_punishment(full_text)),
                    )
                )
            return chunks

        for i, match in enumerate(matches):
            sec_num = match.group(1).strip()
            title = match.group(2).strip().rstrip(".")

            start = match.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)
            raw = full_text[start:end].strip()
            raw = re.sub(r"\n\d{1,3}\s*\n", "\n", raw)
            raw = re.sub(r"\s+", " ", raw)

            if len(raw) < 30:
                continue

            punishment = _extract_punishment(raw)
            # Derive a prefix from the act's stem for unique chunk IDs
            prefix = re.sub(r"[^A-Z0-9]", "", act_name.upper())[:6]
            chunk_id = f"{self.domain_name.upper()[:3]}_{prefix}_{sec_num}"

            chunks.append(
                LegalChunk(
                    chunk_id=chunk_id,
                    domain=self.domain_name,
                    act_name=act_name,
                    section_number=sec_num,
                    title=title,
                    text=raw,
                    source_file=source_file,
                    has_punishment=bool(punishment),
                )
            )

        return chunks

    # ── Private: Vector Store Lifecycle ────────────────────────

    async def _should_rebuild(self) -> bool:
        """True if the index is absent, stale, or the source PDFs changed."""
        if not self._faiss_dir.exists() or not self._meta_path.exists():
            return True
        if not self._cache_path.exists():
            return True
        meta_mtime = self._meta_path.stat().st_mtime
        for pdf in self._bare_acts_dir.rglob("*.pdf"):
            if pdf.stat().st_mtime > meta_mtime:
                return True
        return False

    async def _build_vectorstore(self):
        """
        1. Load all PDFs from the domain's bare_acts subdirectory.
        2. Parse into legal chunks via _parse_legal_sections().
        3. Embed chunk text and build a FAISS index.
        4. Persist index + JSON cache.
        """
        if not self._bare_acts_dir.exists():
            print(
                f"[{self.domain_name}] WARNING: PDFs directory not found — "
                f"{self._bare_acts_dir}"
            )
            return

        pdf_files = list(self._bare_acts_dir.rglob("*.pdf"))
        if not pdf_files:
            print(f"[{self.domain_name}] WARNING: No PDFs found in {self._bare_acts_dir}")
            return

        print(f"[{self.domain_name}] Indexing {len(pdf_files)} PDF(s)…")

        all_chunks: List[LegalChunk] = []
        for pdf_path in pdf_files:
            try:
                loader = PyPDFLoader(str(pdf_path))
                pages = loader.load()
                full_text = "\n".join(p.page_content for p in pages)
                parsed = self._parse_legal_sections(full_text, pdf_path.name)
                print(
                    f"  {pdf_path.name}: {len(pages)} pages → {len(parsed)} chunks"
                )
                all_chunks.extend(parsed)
            except Exception as e:
                print(f"  ERROR loading {pdf_path.name}: {e}")

        if not all_chunks:
            print(f"[{self.domain_name}] No chunks parsed — aborting build.")
            return

        self._chunks = {c.chunk_id: c for c in all_chunks}
        self._save_chunk_cache()

        documents: List[Document] = []
        for chunk in all_chunks:
            embed_text = (
                f"Section {chunk.section_number}. {chunk.title}. {chunk.text}"
            )
            documents.append(
                Document(
                    page_content=embed_text,
                    metadata={
                        "chunk_id": chunk.chunk_id,
                        "act_name": chunk.act_name,
                        "section_number": chunk.section_number,
                        "title": chunk.title,
                        "source": chunk.source_file,
                        "domain": chunk.domain,
                        "punishment": "yes" if chunk.has_punishment else "",
                    },
                )
            )

        print(
            f"[{self.domain_name}] Embedding {len(documents)} documents … "
            f"(this may take several minutes)"
        )
        loop = asyncio.get_event_loop()
        self.vector_store = await loop.run_in_executor(
            None, lambda: FAISS.from_documents(documents, self.embeddings)
        )
        await self._save_vectorstore()
        print(f"[{self.domain_name}] Vector store built and saved.")

    def _save_chunk_cache(self):
        """Persist parsed chunk metadata to JSON."""
        self._faiss_dir.mkdir(parents=True, exist_ok=True)
        cache = {}
        for cid, chunk in self._chunks.items():
            cache[cid] = {
                "chunk_id": chunk.chunk_id,
                "domain": chunk.domain,
                "act_name": chunk.act_name,
                "section_number": chunk.section_number,
                "title": chunk.title,
                "text": chunk.text[:600],
                "source_file": chunk.source_file,
                "has_punishment": chunk.has_punishment,
            }
        with open(self._cache_path, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2, ensure_ascii=False)
        print(f"[{self.domain_name}] Chunk cache saved: {len(cache)} entries.")

    def _load_chunk_cache(self):
        """Load chunk metadata from JSON cache."""
        if not self._cache_path.exists():
            return
        try:
            with open(self._cache_path, "r", encoding="utf-8") as f:
                cache = json.load(f)
            for cid, data in cache.items():
                self._chunks[cid] = LegalChunk(
                    chunk_id=data["chunk_id"],
                    domain=data["domain"],
                    act_name=data["act_name"],
                    section_number=data["section_number"],
                    title=data["title"],
                    text=data["text"],
                    source_file=data["source_file"],
                    has_punishment=data.get("has_punishment", False),
                )
            print(
                f"[{self.domain_name}] Loaded {len(self._chunks)} chunks from cache."
            )
        except Exception as e:
            print(f"[{self.domain_name}] Error loading chunk cache: {e}")

    async def _save_vectorstore(self):
        """Persist FAISS index and write a mtime-stamped meta file."""
        self._faiss_dir.mkdir(parents=True, exist_ok=True)
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: self.vector_store.save_local(str(self._faiss_dir)),
        )
        with open(self._meta_path, "wb") as f:
            pickle.dump({"domain": self.domain_name}, f)
        print(f"[{self.domain_name}] FAISS index saved to {self._faiss_dir}")

    async def _load_vectorstore(self):
        """Load FAISS index from disk."""
        try:
            loop = asyncio.get_event_loop()
            self.vector_store = await loop.run_in_executor(
                None,
                lambda: FAISS.load_local(
                    str(self._faiss_dir),
                    self.embeddings,
                    allow_dangerous_deserialization=True,
                ),
            )
            print(f"[{self.domain_name}] FAISS index loaded from {self._faiss_dir}")
        except Exception as e:
            print(f"[{self.domain_name}] Could not load index ({e}) — rebuilding…")
            await self._build_vectorstore()
