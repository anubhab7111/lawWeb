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
import math
import pickle
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_community.vectorstores import FAISS
    from langchain_community.embeddings import HuggingFaceBgeEmbeddings
    from langchain_core.documents import Document

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

    chunk_id: str  # e.g. "ICA_10", "CON_ART21", "CPC_151"
    domain: str  # primary domain, e.g. "criminal", "family"
    act_name: str  # e.g. "Indian Contract Act, 1872"
    section_number: str  # e.g. "10", "21", "302"
    title: str  # e.g. "What agreements are contracts"
    text: str  # Full section / Article text
    source_file: str  # PDF filename
    score: float = 0.0  # Relevance score (higher = more relevant)
    has_punishment: bool = False  # True only for criminal sections
    # All domains this provision belongs to (a deduped PDF may live in
    # several bare_acts subdirectories, e.g. constitutional + election).
    domains: List[str] = field(default_factory=list)

    def domain_list(self) -> List[str]:
        return self.domains or [self.domain]


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


_ORDINALS = (
    r"(?:FIRST|SECOND|THIRD|FOURTH|FIFTH|SIXTH|SEVENTH|EIGHTH|NINTH|TENTH|"
    r"ELEVENTH|TWELFTH)"
)
_SCHEDULE_HEADING_RE = re.compile(
    r"\n\s*\d*\[?" + _ORDINALS + r"\s+SCHEDULE\s*\n?\s*\[Articles?\s+\d"
)

# Genuine substantive section/article text essentially never contains these —
# they are the standard conventions Indian bare-act PDFs use for amendment
# footnotes ("Subs. by the ... Amendment Act ... (w.e.f. ...)") and Schedule
# paragraph headings, both of which can share a number with a real section.
_NOISE_RE = re.compile(
    r"w\.e\.f\.|subs\s*\.?\s*by|\bins\s*\.?\s*by\b|\brep\.\s*by\b|omitted by|schedule"
    # Amendment-act appendices bundled into bare-act PDFs ("38. Amendment of
    # section 438. In section 438 of the principal Act, ... shall be
    # inserted") reuse their own section numbering and otherwise shadow the
    # real sections they amend.
    r"|amendment of section|of the principal act|shall be inserted"
    r"|shall be substituted",
    re.IGNORECASE,
)


def _schedule_boundary(full_text: str) -> Optional[int]:
    """
    Position where a genuine Schedule begins (e.g. "1[FIRST SCHEDULE
    [Articles 1 and 4]"), if any. Schedules/appendices reuse their own
    paragraph numbering (1, 2, 3, ...) which otherwise collides with real
    section/article numbers, so callers should stop header-matching here.
    """
    m = _SCHEDULE_HEADING_RE.search(full_text)
    return m.start() if m else None


def _is_noise_match(title: str, raw: str) -> bool:
    """True if a candidate section/article match looks like a footnote or a
    Schedule paragraph rather than genuine section/article body text."""
    return bool(_NOISE_RE.search(title) or _NOISE_RE.search(raw[:250]))


def _make_bge_embeddings(device: str) -> Any:
    from app.config import get_settings

    settings = get_settings()
    return HuggingFaceBgeEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"device": device},
        # BGE-M3 (the multilingual default) uses NO query-instruction prefix,
        # unlike bge-large-en. Passing the en-v1.5 default prefix to M3
        # silently degrades cross-lingual retrieval, so it is configurable and
        # blank by default — see config.embedding_query_instruction.
        query_instruction=settings.embedding_query_instruction,
        # Small batch_size: a few thousand legal-section chunks at
        # once can OOM a consumer GPU (e.g. 4GB VRAM) if encoded
        # in one big batch.
        encode_kwargs={"normalize_embeddings": True, "batch_size": 16},
    )


_shared_embeddings: Optional[Any] = None
_shared_embeddings_lock = asyncio.Lock()


async def _get_shared_embeddings() -> Any:
    """
    Every domain (civil/criminal/constitutional) uses the identical embedding
    model — load it once and share the instance instead of one copy per
    domain. Loading 3 separate 1.3GB copies concurrently (e.g. on first
    query, when chatbot.py fetches all domains via asyncio.gather) exhausts a
    4GB consumer GPU; a single shared instance uses the memory once.
    """
    global _shared_embeddings
    if _shared_embeddings is not None:
        return _shared_embeddings
    async with _shared_embeddings_lock:
        if _shared_embeddings is None:
            from app.config import get_settings

            device = get_settings().embeddings_device
            if device not in ("cuda", "cpu"):
                # auto: only claim the GPU when it's big enough to also leave
                # Ollama room for LLM layer offload. On a ~4GB card, giving
                # the VRAM to the LLM instead cuts answer latency far more
                # than GPU query-embedding saves. (Bulk index *building* is a
                # different workload — see build_offline() below, which runs
                # before the LLM claims any VRAM and forces GPU regardless.)
                device = "cpu"
                try:
                    import torch

                    if torch.cuda.is_available():
                        free_bytes, total_bytes = torch.cuda.mem_get_info()
                        if total_bytes >= 6 * 1024**3 and free_bytes > 3.5 * 1024**3:
                            device = "cuda"
                except ImportError:
                    pass
                except Exception as e:
                    # A genuinely full/contended GPU (e.g. Ollama holding VRAM)
                    # can make the mem_get_info() probe itself raise a CUDA
                    # runtime error rather than just report low free memory —
                    # that must fall back to CPU like any other "GPU unusable"
                    # case, not crash RAG initialization entirely.
                    print(f"[rag] CUDA device probe failed ({e}) — using CPU.")
            print(f"[rag] Embeddings device: {device}")
            _shared_embeddings = _make_bge_embeddings(device)
        return _shared_embeddings


_shared_reranker: Optional[Any] = None
_shared_reranker_failed: bool = False
_shared_reranker_lock = asyncio.Lock()


async def _get_shared_reranker() -> Optional[Any]:
    """
    Shared cross-encoder reranker (same singleton pattern as the embeddings).
    Returns None if the model can't be loaded — callers must degrade to the
    fused-retrieval ordering in that case.
    """
    global _shared_reranker, _shared_reranker_failed
    if _shared_reranker is not None or _shared_reranker_failed:
        return _shared_reranker
    async with _shared_reranker_lock:
        if _shared_reranker is not None or _shared_reranker_failed:
            return _shared_reranker
        try:
            from sentence_transformers import CrossEncoder

            from app.config import get_settings

            device = get_settings().reranker_device
            if device not in ("cuda", "cpu"):
                device = "cpu"
                try:
                    import torch

                    if torch.cuda.is_available():
                        free_bytes, total_bytes = torch.cuda.mem_get_info()
                        # Same policy as the embeddings: on a small (<6GB)
                        # card the VRAM is worth more to Ollama's LLM layer
                        # offload; CPU reranking costs ~1-3s/query.
                        if (
                            total_bytes >= 6 * 1024**3
                            and free_bytes > 2.8 * 1024**3
                        ):
                            device = "cuda"
                except ImportError:
                    pass

            model_name = get_settings().reranker_model
            loop = asyncio.get_event_loop()
            try:
                _shared_reranker = await loop.run_in_executor(
                    None,
                    lambda: CrossEncoder(model_name, device=device, max_length=512),
                )
            except Exception as e:
                if device == "cpu":
                    raise
                print(f"[rag] Reranker failed on cuda ({e}) — retrying on cpu.")
                device = "cpu"
                _shared_reranker = await loop.run_in_executor(
                    None,
                    lambda: CrossEncoder(model_name, device=device, max_length=512),
                )
            print(f"[rag] Reranker loaded: {model_name} on {device}")
        except Exception as e:
            _shared_reranker_failed = True
            print(f"[rag] Reranker unavailable ({e}) — falling back to fused order.")
        return _shared_reranker


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, float(x)))))


_BM25_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _bm25_tokenize(text: str) -> List[str]:
    return _BM25_TOKEN_RE.findall(text.lower())


def _split_long_text(text: str, max_chars: int = 2000, overlap: int = 200) -> List[str]:
    """
    Split an oversized section body into overlapping parts on sentence/word
    boundaries. bge-large truncates input at 512 tokens (~2000 chars), so a
    very long section embedded whole silently loses its tail; splitting keeps
    every part retrievable while the shared section metadata ties them back
    to the same provision.
    """
    if len(text) <= max_chars:
        return [text]

    parts: List[str] = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            # Prefer a sentence boundary in the last 40% of the window,
            # then any whitespace, before a hard cut.
            window = text[start:end]
            cut = max(window.rfind(". "), window.rfind("; "))
            if cut < int(max_chars * 0.6):
                cut = window.rfind(" ")
            if cut > int(max_chars * 0.5):
                end = start + cut + 1
        parts.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return [p for p in parts if p]


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.;:])\s+(?=[A-Z0-9(])")
# Abbreviations that precede a period/section-number in Indian statutory
# text (Rs. 500, S. 302, u/s. 34, Art. 21, v.) — without this, splitting
# on the period would strand the abbreviation and its number as two
# separately-scored, individually meaningless fragments.
_ABBR = {
    "rs", "s", "ss", "art", "arts", "no", "u/s", "mr", "dr", "cr",
    "govt", "ors", "anr", "v", "sec", "secs",
}
# A handful of source PDFs duplicate the section header verbatim before the
# body (e.g. "419. Punishment for X. 419. Punishment for X.--Whoever..."),
# which would otherwise split into a bare "419." fragment carrying no
# content of its own — exactly the kind of empty fragment that can crowd
# out the one sentence with the actual operative text.
_BARE_SECTION_NUM_RE = re.compile(r"^\d{1,4}[A-Za-z]{0,3}\.?$")


def _split_into_sentences(text: str) -> List[str]:
    """
    Crude sentence/clause split for extractive scoring — same boundary
    heuristics as _split_long_text, plus guards so a false split (an
    abbreviation, a bare repeated section number) rejoins into the
    previous fragment instead of standing alone.
    """
    raw_parts = _SENTENCE_SPLIT_RE.split(text)
    parts: List[str] = []
    for part in raw_parts:
        part = part.strip()
        if not part:
            continue
        if parts:
            prev_words = parts[-1].split()
            last_word = re.sub(r"[^a-z/]", "", prev_words[-1].lower()) if prev_words else ""
            if last_word in _ABBR or _BARE_SECTION_NUM_RE.match(prev_words[-1]):
                parts[-1] = parts[-1] + " " + part
                continue
        parts.append(part)
    return parts or [text]


async def compress_chunks_for_context(
    query: str, chunks: List["LegalChunk"], max_sentences: int = 3
) -> List["LegalChunk"]:
    """
    Replace each chunk's full section text with just the max_sentences
    sentences most relevant to `query`, scored by the shared cross-encoder
    reranker (already resident in memory for chunk-level reranking) — a
    small LLM's context window is better spent on a few precise sentences
    than a whole section full of definitions/provisos it doesn't need.

    Sentences are kept in original document order, not score order, so the
    excerpt still reads as a coherent (if partial) provision rather than a
    shuffled list of fragments. Chunks already short enough, or all chunks
    when the reranker is unavailable/fails, pass through unmodified.
    """
    reranker = await _get_shared_reranker()
    if reranker is None:
        return chunks

    per_chunk_sentences = [_split_into_sentences(c.text) for c in chunks]
    pairs: List[Tuple[str, str]] = []
    owner: List[int] = []
    for i, sentences in enumerate(per_chunk_sentences):
        if len(sentences) <= max_sentences:
            continue  # already short enough — nothing to gain from scoring
        for sent in sentences:
            pairs.append((query, sent))
            owner.append(i)

    if not pairs:
        return chunks

    loop = asyncio.get_event_loop()
    try:
        logits = await loop.run_in_executor(
            None, lambda: reranker.predict(pairs, batch_size=16)
        )
    except Exception as e:
        print(f"[compress_chunks_for_context] Sentence rerank failed ({e}) — using full chunks.")
        return chunks

    scores_by_chunk: Dict[int, List[float]] = {}
    for idx, score in zip(owner, logits):
        scores_by_chunk.setdefault(idx, []).append(float(score))

    out: List[LegalChunk] = []
    for i, chunk in enumerate(chunks):
        if i not in scores_by_chunk:
            out.append(chunk)
            continue
        sentences = per_chunk_sentences[i]
        scores = scores_by_chunk[i]
        # Anchor on the opening sentence — where a section's operative "(1)"
        # rule usually lives — then fill the rest by score. Pure top-N-by-score
        # can strand a later sub-clause ("(2) Notwithstanding...") without the
        # "(1)" it refers back to, which reads as an incomplete/misleading
        # excerpt even though each kept sentence scored well on its own.
        fill = sorted(
            (j for j in range(1, len(sentences))), key=lambda j: scores[j], reverse=True
        )[: max_sentences - 1]
        top_idx = [0] + fill
        keep = sorted(top_idx)  # restore document order, not score order
        out.append(replace(chunk, text=" ".join(sentences[j] for j in keep)))
    return out


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
        self._bm25: Optional[Any] = None  # rank_bm25.BM25Okapi over _chunks
        self._bm25_ids: List[str] = []  # chunk_id per BM25 corpus row

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
                self.embeddings = await _get_shared_embeddings()
                if await self._should_rebuild():
                    print(f"[{self.domain_name}] Building vector store from PDFs …")
                    await self._build_vectorstore()
                else:
                    print(f"[{self.domain_name}] Loading cached vector store …")
                    await self._load_vectorstore()
                    self._load_chunk_cache()

                if self.vector_store is None:
                    print(
                        f"[{self.domain_name}] Build produced no vector store — "
                        f"will retry on next call."
                    )
                    return False

                self._build_bm25()
                self.initialized = True
                print(
                    f"[{self.domain_name}] RAG ready — "
                    f"{len(self._chunks)} legal chunks indexed."
                )
                return True
            except Exception as e:
                print(f"[{self.domain_name}] Initialization error: {e}")
                import traceback

                traceback.print_exc()
                return False

    async def build_offline(self, device: str) -> bool:
        """
        Force-build (or confirm up-to-date) this domain's index using an
        explicit, throwaway embeddings instance — independent of the shared
        query-time singleton (`_get_shared_embeddings`).

        Used by the offline-indexing startup step: it's called with
        device="cuda" while the LLM hasn't claimed any VRAM yet, then the
        embeddings instance is dropped and GPU memory freed before the
        server starts serving, so Ollama gets the full card afterward.
        Query-time embedding later still goes through the normal shared
        singleton (CPU on small GPUs, per _get_shared_embeddings), unaffected
        by whatever happened here.
        """
        if not HAS_RAG_DEPS:
            return False
        if not await self._should_rebuild():
            print(f"[{self.domain_name}] Index already up to date — skipping build.")
            return True
        embeddings = _make_bge_embeddings(device)
        self.embeddings = embeddings
        try:
            print(f"[{self.domain_name}] Offline build on {device} …")
            await self._build_vectorstore()
            return self.vector_store is not None
        finally:
            self.embeddings = None
            self.vector_store = None
            del embeddings
            import gc

            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

    async def retrieve(
        self,
        query: str,
        k: int = 4,
        min_score: float = 0.25,
        domains: Optional[List[str]] = None,
        use_reranker: bool = True,
    ) -> LegalContext:
        """
        Hybrid retrieval: dense (FAISS) + sparse (BM25) candidates fused with
        Reciprocal Rank Fusion, then reranked by a cross-encoder.

        Args:
            query:        User query (preprocessed before dense/sparse search).
            k:            Maximum chunks to return.
            min_score:    Minimum relevance RELATIVE to the top hit (reranker
                          probabilities are ordinal, not calibrated).
            domains:      Optional domain filter (e.g. ["criminal"]); matches
                          any of a chunk's domains.
            use_reranker: Disable to skip the cross-encoder stage.

        Returns:
            LegalContext with matched chunks sorted by score.
        """
        if not self.initialized:
            await self.initialize()

        if not self.initialized or not self.vector_store:
            return LegalContext(domain=self.domain_name, query=query)

        try:
            chunks = await self._hybrid_search(
                search_query=self._preprocess_query(query),
                rerank_query=query,
                k=k,
                min_score=min_score,
                domains=domains,
                use_reranker=use_reranker,
            )

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

    async def _hybrid_search(
        self,
        search_query: str,
        rerank_query: str,
        k: int,
        min_score: float,
        domains: Optional[List[str]] = None,
        use_reranker: bool = True,
        candidate_pool: int = 30,
        rerank_pool: int = 20,
    ) -> List[LegalChunk]:
        """
        Shared hybrid pipeline: dense + BM25 → RRF fusion → cross-encoder
        rerank → top-k LegalChunks (score-filtered, sorted desc).

        `search_query` may be keyword-expanded; `rerank_query` should be the
        natural-language query, which cross-encoders score more reliably.
        """
        loop = asyncio.get_event_loop()
        domain_set = set(domains) if domains else None

        def _md_domains(md: Dict[str, Any]) -> List[str]:
            raw = md.get("domains") or md.get("domain") or ""
            return [d for d in raw.split(",") if d]

        # ── Dense candidates ────────────────────────────────────
        search_kwargs: Dict[str, Any] = {"k": candidate_pool}
        if domain_set:
            # FAISS post-filters, so overfetch before the filter is applied.
            search_kwargs["fetch_k"] = candidate_pool * 5
            search_kwargs["filter"] = lambda md: bool(
                set(_md_domains(md)) & domain_set
            )
        dense_results = await loop.run_in_executor(
            None,
            lambda: self.vector_store.similarity_search_with_score(
                search_query, **search_kwargs
            ),
        )

        dense_rank: Dict[str, int] = {}
        dense_score: Dict[str, float] = {}
        for rank, (doc, distance) in enumerate(dense_results):
            cid = doc.metadata.get("chunk_id", "")
            if cid and cid not in dense_rank:
                dense_rank[cid] = rank
                # Legacy L2→[0,1] conversion, kept for the no-reranker fallback
                dense_score[cid] = max(0.0, 1.0 - (float(distance) / 2.0))

        # ── Sparse (BM25) candidates ────────────────────────────
        sparse_rank: Dict[str, int] = {}
        if self._bm25 is not None:
            tokens = _bm25_tokenize(search_query)
            if tokens:
                scores = self._bm25.get_scores(tokens)
                order = sorted(
                    range(len(scores)), key=lambda i: scores[i], reverse=True
                )
                rank = 0
                for idx in order:
                    if scores[idx] <= 0 or rank >= candidate_pool:
                        break
                    cid = self._bm25_ids[idx]
                    chunk = self._chunks.get(cid)
                    if domain_set and (
                        not chunk or not (set(chunk.domain_list()) & domain_set)
                    ):
                        continue
                    sparse_rank[cid] = rank
                    rank += 1

        # ── Reciprocal Rank Fusion ──────────────────────────────
        fused: Dict[str, float] = {}
        for cid, rank in dense_rank.items():
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (60 + rank)
        for cid, rank in sparse_rank.items():
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (60 + rank)

        candidates = [
            cid
            for cid in sorted(fused, key=lambda c: fused[c], reverse=True)
            if cid in self._chunks
        ][:rerank_pool]
        if not candidates:
            return []

        # ── Cross-encoder rerank (graceful fallback to fused order) ──
        # Cross-encoder probabilities are ordinal, not calibrated: a clearly
        # relevant pair may score 0.04 while garbage scores 0.0001. So the
        # reranker decides the ORDER, and filtering uses confidence RELATIVE
        # to the best candidate (with an absolute garbage floor), not the raw
        # probability. Reported chunk.score = prob / top_prob.
        RERANK_GARBAGE_FLOOR = 1e-4
        scored: List[Tuple[str, float]] = []
        reranker = await _get_shared_reranker() if use_reranker else None
        if reranker is not None:
            try:
                pairs = [
                    (rerank_query, self._chunks[cid].text[:2000])
                    for cid in candidates
                ]
                logits = await loop.run_in_executor(
                    None, lambda: reranker.predict(pairs, batch_size=16)
                )
                raw = [float(s) for s in logits]
                # Some sentence-transformers versions already apply sigmoid to
                # single-label cross-encoders; only normalize raw logits.
                if any(s < 0.0 or s > 1.0 for s in raw):
                    raw = [_sigmoid(s) for s in raw]
                top = max(raw) if raw else 0.0
                if top >= RERANK_GARBAGE_FLOOR:
                    scored = [
                        (cid, prob / top)
                        for cid, prob in zip(candidates, raw)
                        if prob >= RERANK_GARBAGE_FLOOR
                    ]
                    scored.sort(key=lambda pair: pair[1], reverse=True)
                # else: nothing plausibly relevant — fall back to fused order
            except Exception as e:
                print(f"[{self.domain_name}] Rerank failed ({e}) — using fused order.")
                scored = []
        if not scored:
            # Fused order with legacy dense scores; BM25-only candidates get a
            # floor just above typical min_score so exact-term hits survive.
            scored = [(cid, dense_score.get(cid, 0.35)) for cid in candidates]

        chunks: List[LegalChunk] = []
        seen_sections: set = set()
        for cid, score in scored:
            if score < min_score:
                continue
            cached = self._chunks[cid]
            # A long section split into parts can flood top-k with near
            # duplicates; keep only the best-scored part per section.
            section_key = (cached.act_name, cached.section_number)
            if section_key in seen_sections:
                continue
            seen_sections.add(section_key)
            chunks.append(
                LegalChunk(
                    chunk_id=cached.chunk_id,
                    domain=cached.domain,
                    act_name=cached.act_name,
                    section_number=cached.section_number,
                    title=cached.title,
                    text=cached.text,
                    source_file=cached.source_file,
                    has_punishment=cached.has_punishment,
                    domains=list(cached.domains),
                    score=round(min(score, 1.0), 3),
                )
            )
            if len(chunks) >= k:
                break

        return chunks

    def find_section(
        self, act_hint: str, section: str, max_parts: int = 2
    ) -> List[LegalChunk]:
        """
        Deterministic lookup of a statutory provision by act-name hint and
        section number (e.g. ("Indian Penal Code", "420")). Returns up to
        `max_parts` chunks (long sections are stored as ordered parts).
        """
        hint = act_hint.lower().strip()
        want = section.lower().replace("article", "").strip()
        hits = [
            c
            for c in self._chunks.values()
            if c.section_number.lower().replace("article", "").strip() == want
            and (not hint or hint in c.act_name.lower())
        ]
        hits.sort(key=lambda c: c.chunk_id)  # parts in order (_p1, _p2, …)
        return hits[:max_parts]

    def _build_bm25(self):
        """Build the in-memory BM25 index over the parsed chunk cache."""
        self._bm25 = None
        self._bm25_ids = []
        if not self._chunks:
            return
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            print(f"[{self.domain_name}] rank_bm25 not installed — dense-only mode.")
            return
        corpus: List[List[str]] = []
        for cid, chunk in self._chunks.items():
            self._bm25_ids.append(cid)
            corpus.append(
                _bm25_tokenize(
                    f"{chunk.act_name} Section {chunk.section_number} "
                    f"{chunk.title} {chunk.text}"
                )
            )
        self._bm25 = BM25Okapi(corpus)
        print(f"[{self.domain_name}] BM25 index built over {len(corpus)} chunks.")

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

        # Stop before any Schedule: schedules restart their own paragraph
        # numbering (1, 2, 3, ...), which otherwise collides with real
        # section numbers and pollutes the index with schedule text.
        boundary = _schedule_boundary(full_text)
        search_text = full_text[:boundary] if boundary else full_text

        # Match  "  10.  What agreements are contracts"  style headers.
        # \s* (not \s+) after the number: some Acts (e.g. BNS 2023) run the
        # subsection straight on, e.g. "103.(1) Whoever commits murder...".
        # An optional "N[" prefix is tolerated: amended sections are often
        # annotated with a footnote-bracket marker, e.g. "3[226. Power of...".
        # The title may also terminate at end-of-line: several bare-act PDFs
        # (e.g. Consumer Protection Act 2019) run the section body straight
        # on and the first sentence wraps before any period, which otherwise
        # makes the header unmatchable and merges sections. TOC lines match
        # too as a result, but the chunk-id dedup below already prefers the
        # longest clean candidate, so the body wins over the TOC entry.
        header_pattern = re.compile(
            r"\n\s*(?:\d+\[)?(\d{1,3}[A-Z]{0,2})\.\s*([^.\n\u2014]{3,}?)(?:[.\u2014]|(?=\n))\s*",
            re.MULTILINE,
        )
        matches = list(header_pattern.finditer(search_text))

        if not matches:
            # Fallback: treat the entire text as a single chunk
            if len(full_text.strip()) > 50:
                chunk_id = (
                    f"{self.domain_name.upper()[:3]}_{Path(source_file).stem}_FULL"
                )
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
            end = matches[i + 1].start() if i + 1 < len(matches) else len(search_text)
            raw = search_text[start:end].strip()
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
        """True if the index is absent, stale, or the source PDFs changed.

        Staleness is decided by a (relative_path -> size) fingerprint stored
        in meta.pkl, not filesystem mtimes: a fresh `git checkout`/clone/
        worktree resets every file's mtime to checkout time even when its
        content is unchanged, which would otherwise force a full re-embed
        on every environment that just checked the repo out.
        """
        if not self._faiss_dir.exists() or not self._meta_path.exists():
            return True
        if not self._cache_path.exists():
            return True
        try:
            with open(self._meta_path, "rb") as f:
                meta = pickle.load(f)
        except Exception:
            return True
        stored_fingerprint = meta.get("pdf_fingerprint")
        if stored_fingerprint is None:
            return True  # meta.pkl predates fingerprinting — rebuild once
        # Vectors are model-specific: an index built with a different embedding
        # model must be rebuilt, or queries embedded by the new model would be
        # scored against incompatible stored vectors.
        from app.config import get_settings

        if meta.get("embedding_model") != get_settings().embedding_model:
            return True
        return stored_fingerprint != self._current_pdf_fingerprint()

    def _current_pdf_fingerprint(self) -> Dict[str, int]:
        """(relative_path -> byte size) for every source PDF, cheap to compute."""
        return {
            str(p.relative_to(self._bare_acts_dir)): p.stat().st_size
            for p in self._bare_acts_dir.rglob("*.pdf")
        }

    def _collect_pdf_sources(self) -> List[Tuple[Path, str, List[str]]]:
        """
        Enumerate (pdf_path, primary_domain, domains) triples to index.
        Base implementation: every PDF under the domain's directory belongs
        to this domain. The unified system overrides this to walk all
        domains and dedupe PDFs that appear in several of them.
        """
        return [
            (p, self.domain_name, [self.domain_name])
            for p in sorted(self._bare_acts_dir.rglob("*.pdf"))
        ]

    def _parse_pdf(
        self, full_text: str, source_file: str, primary_domain: str
    ) -> List[LegalChunk]:
        """Parse one PDF's text into chunks. Overridable per-domain dispatch."""
        return self._parse_legal_sections(full_text, source_file)

    async def _build_vectorstore(self):
        """
        1. Collect source PDFs (deduped across domains by subclasses).
        2. Parse into legal chunks via _parse_pdf().
        3. Split oversized sections into overlapping parts.
        4. Embed chunk text and build a FAISS index.
        5. Persist index + JSON cache.
        """
        if not self._bare_acts_dir.exists():
            print(
                f"[{self.domain_name}] WARNING: PDFs directory not found — "
                f"{self._bare_acts_dir}"
            )
            return

        pdf_sources = self._collect_pdf_sources()
        if not pdf_sources:
            print(
                f"[{self.domain_name}] WARNING: No PDFs found in {self._bare_acts_dir}"
            )
            return

        print(f"[{self.domain_name}] Indexing {len(pdf_sources)} PDF(s)…")

        all_chunks: List[LegalChunk] = []
        for pdf_path, primary_domain, domains in pdf_sources:
            try:
                loader = PyPDFLoader(str(pdf_path))
                pages = loader.load()
                full_text = "\n".join(p.page_content for p in pages)
                parsed = self._parse_pdf(full_text, pdf_path.name, primary_domain)
                for c in parsed:
                    c.domains = list(domains)
                print(f"  {pdf_path.name}: {len(pages)} pages → {len(parsed)} chunks")
                all_chunks.extend(parsed)
            except Exception as e:
                print(f"  ERROR loading {pdf_path.name}: {e}")

        if not all_chunks:
            print(f"[{self.domain_name}] No chunks parsed — aborting build.")
            return

        # A given chunk_id can legitimately collide across a document (e.g. a
        # footnote/amendment citation like "1. Subs. by Act 3 of 1951..." or a
        # table-of-contents entry re-using the same section number). Prefer a
        # candidate that doesn't look like footnote/schedule noise; among
        # equally "clean" candidates, keep the longest (genuine section bodies
        # are almost always longer than a TOC line or citation).
        self._chunks = {}
        best_score: Dict[str, tuple] = {}
        for c in all_chunks:
            score = (0 if _is_noise_match(c.title, c.text) else 1, len(c.text))
            if c.chunk_id not in best_score or score > best_score[c.chunk_id]:
                best_score[c.chunk_id] = score
                self._chunks[c.chunk_id] = c

        self._chunks = self._split_oversized_chunks(self._chunks)
        self._save_chunk_cache()

        documents: List[Document] = []
        for chunk in self._chunks.values():
            sec_label = (
                chunk.section_number
                if chunk.section_number.lower().startswith("article")
                else f"Section {chunk.section_number}"
            )
            embed_text = f"{chunk.act_name}. {sec_label}. {chunk.title}. {chunk.text}"
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
                        "domains": ",".join(chunk.domain_list()),
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

    @staticmethod
    def _split_oversized_chunks(
        chunks: Dict[str, LegalChunk],
    ) -> Dict[str, LegalChunk]:
        """
        Replace any chunk whose body exceeds the embedder's useful window
        with overlapping "_pN" part-chunks carrying the same section metadata.
        """
        result: Dict[str, LegalChunk] = {}
        n_split = 0
        for cid, chunk in chunks.items():
            parts = _split_long_text(chunk.text)
            if len(parts) == 1:
                result[cid] = chunk
                continue
            n_split += 1
            for n, part in enumerate(parts, 1):
                pid = f"{cid}_p{n}"
                result[pid] = LegalChunk(
                    chunk_id=pid,
                    domain=chunk.domain,
                    act_name=chunk.act_name,
                    section_number=chunk.section_number,
                    title=chunk.title,
                    text=part,
                    source_file=chunk.source_file,
                    has_punishment=chunk.has_punishment,
                    domains=list(chunk.domains),
                )
        if n_split:
            print(
                f"  Split {n_split} oversized sections into overlapping parts "
                f"({len(chunks)} → {len(result)} chunks)."
            )
        return result

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
                "text": chunk.text,
                "source_file": chunk.source_file,
                "has_punishment": chunk.has_punishment,
                "domains": chunk.domain_list(),
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
                    domains=data.get("domains") or [data["domain"]],
                )
            print(f"[{self.domain_name}] Loaded {len(self._chunks)} chunks from cache.")
        except Exception as e:
            print(f"[{self.domain_name}] Error loading chunk cache: {e}")

    async def _save_vectorstore(self):
        """Persist FAISS index and write a mtime-stamped meta file."""
        from app.config import get_settings

        self._faiss_dir.mkdir(parents=True, exist_ok=True)
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: self.vector_store.save_local(str(self._faiss_dir)),
        )
        with open(self._meta_path, "wb") as f:
            pickle.dump(
                {
                    "domain": self.domain_name,
                    "pdf_fingerprint": self._current_pdf_fingerprint(),
                    "embedding_model": get_settings().embedding_model,
                },
                f,
            )
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
