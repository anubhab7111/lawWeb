"""base_legal_rag.py — Abstract Base for Multi-Domain Legal RAG Systems Provides the shared FAISS infrastructure for all domain-specific RAG classes: - CriminalRAGSystem..."""

import asyncio
import json
import math
import pickle
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from app.ingest.paths import statutes_dir

try:
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_community.vectorstores import FAISS
    from langchain_community.embeddings import HuggingFaceBgeEmbeddings
    from langchain_core.documents import Document

    HAS_RAG_DEPS = True
except ImportError as e:
    HAS_RAG_DEPS = False
    print(f"RAG dependencies import error: {e}")


@dataclass
class LegalChunk:
    """One atomic legal provision retrieved from the vector store."""

    chunk_id: str
    domain: str
    act_name: str
    section_number: str
    title: str
    text: str
    source_file: str
    score: float = 0.0
    has_punishment: bool = False


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

    stem = re.sub(r"\s+\d{4}$", "", stem)
    return stem


_ORDINALS = (
    r"(?:FIRST|SECOND|THIRD|FOURTH|FIFTH|SIXTH|SEVENTH|EIGHTH|NINTH|TENTH|"
    r"ELEVENTH|TWELFTH)"
)
_SCHEDULE_HEADING_RE = re.compile(
    r"\n\s*\d*\[?" + _ORDINALS + r"\s+SCHEDULE\s*\n?\s*\[Articles?\s+\d"
)


_NOISE_RE = re.compile(
    r"w\.e\.f\.|subs\s*\.?\s*by|\bins\s*\.?\s*by\b|\brep\.\s*by\b|omitted by|schedule"


    r"|amendment of section|of the principal act|shall be inserted"
    r"|shall be substituted",
    re.IGNORECASE,
)


def _schedule_boundary(full_text: str) -> Optional[int]:
    """Position where a genuine Schedule begins (e.g."""
    m = _SCHEDULE_HEADING_RE.search(full_text)
    return m.start() if m else None


RERANK_TEXT_CHARS = 1800
_BLEND_K = 10


def _rerank_text(chunk: "LegalChunk") -> str:
    sec = chunk.section_number
    label = sec if sec.lower().startswith(("article", "part")) else f"Section {sec}"
    return f"{chunk.act_name}, {label}: {chunk.title}. {chunk.text[:RERANK_TEXT_CHARS]}"


def get_settings_blend() -> float:
    from app.config import get_settings

    return get_settings().rerank_blend


def blend_rerank_order(
    scored: List[Tuple[str, float]], fused_order: List[str], weight: float
) -> List[Tuple[str, float]]:
    """Order reranked candidates by a weighted reciprocal-rank blend of the cross-encoder's order and the fused retrieval order."""
    by_rerank = sorted(scored, key=lambda pair: pair[1], reverse=True)
    if weight >= 1.0:
        return by_rerank
    rerank_rank = {cid: r for r, (cid, _s) in enumerate(by_rerank)}
    fused_rank = {cid: r for r, cid in enumerate(fused_order)}
    blended = {
        cid: weight / (_BLEND_K + rerank_rank[cid]) + (1 - weight) / (_BLEND_K + fused_rank.get(cid, len(fused_order)))
        for cid, _s in scored
    }
    return sorted(scored, key=lambda pair: blended[pair[0]], reverse=True)


def _is_noise_match(title: str, raw: str) -> bool:
    """True if a candidate section/article match looks like a footnote or a Schedule paragraph rather than genuine section/article body text."""
    return bool(_NOISE_RE.search(title) or _NOISE_RE.search(raw[:250]))


def _make_bge_embeddings(device: str) -> Any:
    from app.config import get_settings

    settings = get_settings()
    return HuggingFaceBgeEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"device": device},


        query_instruction=settings.embedding_query_instruction,


        encode_kwargs={"normalize_embeddings": True, "batch_size": 16},
    )


_shared_embeddings: Optional[Any] = None
_shared_embeddings_lock = asyncio.Lock()


async def _get_shared_embeddings() -> Any:
    """Every domain (civil/criminal/constitutional) uses the identical embedding model — load it once and share the instance instead of one copy per domain."""
    global _shared_embeddings
    if _shared_embeddings is not None:
        return _shared_embeddings
    async with _shared_embeddings_lock:
        if _shared_embeddings is None:
            from app.config import get_settings

            device = get_settings().embeddings_device
            if device not in ("cuda", "cpu"):


                device = "cpu"
                try:
                    import torch

                    if torch.cuda.is_available():
                        free_bytes, _total_bytes = torch.cuda.mem_get_info()
                        reserve_bytes = get_settings().ollama_vram_reserve_gb * 1024**3


                        embeddings_headroom_bytes = 2.0 * 1024**3
                        if free_bytes > reserve_bytes + embeddings_headroom_bytes:
                            device = "cuda"
                except ImportError:
                    pass
                except Exception as e:
                    print(f"[rag] CUDA availability check failed ({e}) — using CPU.")
            print(f"[rag] Embeddings device: {device}")
            try:
                _shared_embeddings = _make_bge_embeddings(device)
            except Exception as e:
                if device == "cpu":
                    raise
                print(f"[rag] Embeddings failed on cuda ({e}) — retrying on cpu.")
                device = "cpu"
                _shared_embeddings = _make_bge_embeddings(device)
        return _shared_embeddings


_shared_reranker: Optional[Any] = None
_shared_reranker_failed: bool = False
_shared_reranker_lock = asyncio.Lock()


_RERANK_CACHE: Dict[Tuple[str, str], float] = {}
_RERANK_CACHE_MAX = 4096


_PAIR_CACHE: Dict[Tuple[str, str], float] = {}


def cached_predict(reranker: Any, pairs: List[Tuple[str, str]], batch_size: int = 16) -> List[float]:
    scores = {p: _PAIR_CACHE[p] for p in pairs if p in _PAIR_CACHE}
    todo = [p for p in dict.fromkeys(pairs) if p not in scores]
    if todo:
        scores.update(zip(todo, (float(s) for s in reranker.predict(todo, batch_size=batch_size))))
        if len(_PAIR_CACHE) > _RERANK_CACHE_MAX:
            _PAIR_CACHE.clear()
        _PAIR_CACHE.update(scores)
    return [scores[p] for p in pairs]


async def _get_shared_reranker() -> Optional[Any]:
    """Shared cross-encoder reranker (same singleton pattern as the embeddings)."""
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
                        free_bytes, _total_bytes = torch.cuda.mem_get_info()
                        settings = get_settings()
                        reserve_bytes = settings.ollama_vram_reserve_gb * 1024**3
                        heavy = settings.reranker_model_gpu != settings.reranker_model
                        reranker_headroom_bytes = (2.5 if heavy else 1.0) * 1024**3
                        if free_bytes > reserve_bytes + reranker_headroom_bytes:
                            device = "cuda"
                except ImportError:
                    pass
                except Exception as e:
                    print(f"[rag] CUDA availability check failed ({e}) — reranker on CPU.")

            settings = get_settings()
            model_name = settings.reranker_model_gpu if device == "cuda" else settings.reranker_model
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
                model_name = settings.reranker_model
                _shared_reranker = await loop.run_in_executor(
                    None,
                    lambda: CrossEncoder(model_name, device=device, max_length=512),
                )
            print(f"[rag] Reranker loaded: {model_name} on {device}")
        except Exception as e:
            _shared_reranker_failed = True
            print(f"[rag] Reranker unavailable ({e}) — falling back to fused order.")
        return _shared_reranker


def file_fingerprint(path: Path) -> str:
    """Size plus a hash of the head and tail — detects same-size edits that a size-only check misses, without reading whole PDFs."""
    import hashlib

    size = path.stat().st_size
    with open(path, "rb") as f:
        head = f.read(65536)
        f.seek(max(size - 65536, 0))
        tail = f.read(65536)
    return f"{size}:{hashlib.sha1(head + tail).hexdigest()[:12]}"


_WORD_RE = re.compile(r"[A-Za-z]+")


def _split_word_count(text: str, vocab: set) -> int:
    """Adjacent token pairs that only form a vocabulary word when joined ("Magi strate"): pypdf's plain mode splits words at kerning gaps."""
    toks = _WORD_RE.findall(text)
    n = 0
    for a, b in zip(toks, toks[1:]):
        joined = (a + b).lower()
        if len(joined) >= 6 and joined in vocab and (a.lower() not in vocab or b.lower() not in vocab):
            n += 1
    return n


def _choose_extraction(plain: str, layout: Optional[str], parse) -> Tuple[str, List["LegalChunk"]]:
    """Prefer layout-mode text only when it parses to exactly the same sections and has fewer split words: layout mode fixes kerning splits but loses or invents section headers in..."""
    plain_chunks = parse(plain)
    if not layout:
        return plain, plain_chunks
    layout_chunks = parse(layout)
    if {c.section_number for c in layout_chunks} != {c.section_number for c in plain_chunks}:
        return plain, plain_chunks
    counts: Dict[str, int] = {}
    for w in _WORD_RE.findall(plain + "\n" + layout):
        w = w.lower()
        counts[w] = counts.get(w, 0) + 1
    vocab = {w for w, c in counts.items() if c >= 3 and len(w) >= 2}
    if _split_word_count(layout, vocab) < _split_word_count(plain, vocab):
        return layout, layout_chunks
    return plain, plain_chunks


def _natural_key(chunk: "LegalChunk"):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", chunk.chunk_id)]


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, float(x)))))


_BM25_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _bm25_tokenize(text: str) -> List[str]:
    return _BM25_TOKEN_RE.findall(text.lower())


def _split_long_text(text: str, max_chars: int = 2000, overlap: int = 200) -> List[str]:
    """Split an oversized section body into overlapping parts on sentence/word boundaries."""
    if len(text) <= max_chars:
        return [text]

    parts: List[str] = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):


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


_ABBR = {
    "rs", "s", "ss", "art", "arts", "no", "u/s", "mr", "dr", "cr",
    "govt", "ors", "anr", "v", "sec", "secs",
}


_BARE_SECTION_NUM_RE = re.compile(r"^\d{1,4}[A-Za-z]{0,3}\.?$")


def _split_into_sentences(text: str) -> List[str]:
    """Crude sentence/clause split for extractive scoring — same boundary heuristics as _split_long_text, plus guards so a false split (an abbreviation, a bare repeated section..."""
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


_COMPRESS_CANDIDATES = 8
_FUNCTION_WORDS = frozenset(
    "the and for with that this from shall any such under which who whom what when where"
    " are was were has have had been being may can not all its his her their into upon".split()
)


def _content_words(text: str) -> set:
    """6-letter stems of a text's content words (crude, suffix-insensitive)."""
    return {w[:6] for w in re.findall(r"[a-z]{3,}", text.lower()) if w not in _FUNCTION_WORDS}


async def compress_chunks_for_context(
    query: str, chunks: List["LegalChunk"], max_sentences: int = 3
) -> List["LegalChunk"]:
    """Replace each chunk's full section text with just the max_sentences sentences most relevant to `query`, scored by the shared cross-encoder reranker (already resident in memory..."""
    reranker = await _get_shared_reranker()
    if reranker is None:
        return chunks

    per_chunk_sentences = [_split_into_sentences(c.text) for c in chunks]
    query_words = _content_words(query)
    pairs: List[Tuple[str, str]] = []
    owner: List[Tuple[int, int]] = []
    for i, sentences in enumerate(per_chunk_sentences):
        if len(sentences) <= max_sentences:
            continue


        candidates = sorted(
            range(1, len(sentences)),
            key=lambda j: -len(query_words & _content_words(sentences[j])),
        )[:_COMPRESS_CANDIDATES]
        for j in candidates:
            pairs.append((query, sentences[j]))
            owner.append((i, j))

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

    scores_by_chunk: Dict[int, Dict[int, float]] = {}
    for (i, j), score in zip(owner, logits):
        scores_by_chunk.setdefault(i, {})[j] = float(score)

    out: List[LegalChunk] = []
    for i, chunk in enumerate(chunks):
        if i not in scores_by_chunk:
            out.append(chunk)
            continue
        sentences = per_chunk_sentences[i]
        scores = scores_by_chunk[i]


        fill = sorted(scores, key=lambda j: scores[j], reverse=True)[: max_sentences - 1]
        top_idx = [0] + fill
        keep = sorted(top_idx)
        out.append(replace(chunk, text=" ".join(sentences[j] for j in keep)))
    return out


class BaseLegalRAGSystem(ABC):
    """Abstract base for domain-specific legal RAG systems."""


    @property
    @abstractmethod
    def domain_name(self) -> str:
        """Short domain identifier, e.g."""
        ...

    @property
    @abstractmethod
    def pdf_subdir(self) -> str:
        """Subdirectory under app/data/bare_acts/ holding this domain's PDFs."""
        ...


    def __init__(self, data_dir: Optional[str] = None):


        self.data_dir = Path(data_dir or "app/data")
        self._statutes_dir = Path(data_dir) if data_dir else statutes_dir()
        self._bare_acts_dir = self._statutes_dir / "bare_acts" / self.pdf_subdir

        self._faiss_dir = self.data_dir / "faiss_index" / self.domain_name
        self._meta_path = self._faiss_dir / "meta.pkl"
        self._cache_path = self._faiss_dir / "sections.json"

        self.vector_store: Optional[Any] = None
        self.embeddings: Optional[Any] = None
        self.initialized: bool = False
        self._init_lock = asyncio.Lock()
        self._chunks: Dict[str, LegalChunk] = {}
        self._bm25: Optional[Any] = None
        self._bm25_ids: List[str] = []


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
                    self._warn_if_chunk_cache_mismatch()

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
        """Force-build (or confirm up-to-date) this domain's index using an explicit, throwaway embeddings instance — independent of the shared query-time singleton..."""
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
        acts: Optional[List[str]] = None,
        candidate_pool: int = 30,
        rerank_pool: int = 20,
    ) -> LegalContext:
        """Hybrid retrieval: dense (FAISS) + sparse (BM25) candidates fused with Reciprocal Rank Fusion, then reranked by a cross-encoder."""
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
                acts=acts,
                candidate_pool=candidate_pool,
                rerank_pool=rerank_pool,
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
        acts: Optional[List[str]] = None,
        extra_candidates: Optional[List[str]] = None,
    ) -> List[LegalChunk]:
        """Shared hybrid pipeline: dense + BM25 → RRF fusion → cross-encoder rerank → top-k LegalChunks (score-filtered, sorted desc)."""
        loop = asyncio.get_event_loop()
        domain_set = set(domains) if domains else None
        act_set = set(acts) if acts else None

        def _md_domains(md: Dict[str, Any]) -> List[str]:
            raw = md.get("domains") or md.get("domain") or ""
            return [d for d in raw.split(",") if d]


        search_kwargs: Dict[str, Any] = {"k": candidate_pool}
        if domain_set or act_set:


            search_kwargs["fetch_k"] = candidate_pool * (60 if act_set else 5)
            search_kwargs["filter"] = lambda md: (
                (not domain_set or bool(set(_md_domains(md)) & domain_set))
                and (not act_set or md.get("act_name") in act_set)
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

                dense_score[cid] = max(0.0, 1.0 - (float(distance) / 2.0))


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
                    if act_set and (not chunk or chunk.act_name not in act_set):
                        continue
                    sparse_rank[cid] = rank
                    rank += 1


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
        if extra_candidates:
            candidate_set = set(candidates)
            candidates += [
                cid
                for cid in dict.fromkeys(extra_candidates)
                if cid in self._chunks and cid not in candidate_set
            ]
        if not candidates:
            return []


        RERANK_GARBAGE_FLOOR = 1e-4
        scored: List[Tuple[str, float]] = []
        reranker = await _get_shared_reranker() if use_reranker else None
        if reranker is not None:
            try:


                todo = [cid for cid in candidates if (rerank_query, cid) not in _RERANK_CACHE]
                if todo:
                    pairs = [(rerank_query, _rerank_text(self._chunks[cid])) for cid in todo]
                    logits = await loop.run_in_executor(
                        None, lambda: reranker.predict(pairs, batch_size=16)
                    )
                    fresh = [float(s) for s in logits]


                    if any(s < 0.0 or s > 1.0 for s in fresh):
                        fresh = [_sigmoid(s) for s in fresh]
                    if len(_RERANK_CACHE) > _RERANK_CACHE_MAX:
                        _RERANK_CACHE.clear()
                    _RERANK_CACHE.update({(rerank_query, cid): s for cid, s in zip(todo, fresh)})
                raw = [_RERANK_CACHE[(rerank_query, cid)] for cid in candidates]
                top = max(raw) if raw else 0.0
                if top >= RERANK_GARBAGE_FLOOR:
                    scored = [
                        (cid, prob / top)
                        for cid, prob in zip(candidates, raw)
                        if prob >= RERANK_GARBAGE_FLOOR
                    ]
                    scored = blend_rerank_order(scored, candidates, get_settings_blend())

            except Exception as e:
                print(f"[{self.domain_name}] Rerank failed ({e}) — using fused order.")
                scored = []
        if not scored:


            scored = [(cid, dense_score.get(cid, 0.35)) for cid in candidates]

        chunks: List[LegalChunk] = []
        seen_sections: set = set()
        for cid, score in scored:
            if score < min_score:
                continue
            cached = self._chunks[cid]


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
        """Deterministic lookup of a statutory provision by act-name hint and section number (e.g."""
        hint = act_hint.lower().strip()
        want = section.lower().replace("article", "").strip()
        hits = [
            c
            for c in self._chunks.values()
            if c.section_number.lower().replace("article", "").strip() == want
            and (not hint or hint in c.act_name.lower())
        ]
        hits.sort(key=_natural_key)
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


    def _preprocess_query(self, query: str) -> str:
        """Optionally expand the query with domain-specific vocabulary."""
        return query

    def _parse_legal_sections(
        self, full_text: str, source_file: str
    ) -> List[LegalChunk]:
        """Generic legal section parser."""
        chunks: List[LegalChunk] = []
        act_name = _infer_act_name(source_file)


        boundary = _schedule_boundary(full_text)
        search_text = full_text[:boundary] if boundary else full_text


        header_pattern = re.compile(
            r"\n\s*(?:\d+\*?\[)?(\d{1,3}[A-Z]{0,2})\.\s*([^.\n\u2014]{3,}?)(?:[.\u2014]|(?=\n))\s*",
            re.MULTILINE,
        )
        matches = list(header_pattern.finditer(search_text))

        if not matches:

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


            prefix = re.sub(r"[^A-Z0-9]", "", act_name.upper())
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


    async def _should_rebuild(self) -> bool:
        """True if the index is absent, stale, or the source PDFs changed."""
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
            return True


        from app.config import get_settings

        if meta.get("embedding_model") != get_settings().embedding_model:
            return True
        if not any(self._bare_acts_dir.rglob("*.pdf")):


            print(
                f"[{self.domain_name}] Source PDFs unavailable ({self._bare_acts_dir}) "
                f"— serving the existing index without a staleness check."
            )
            return False
        return stored_fingerprint != self._current_pdf_fingerprint()

    def _current_pdf_fingerprint(self) -> Dict[str, str]:
        """(relative_path -> size:hash) for every source PDF, cheap to compute."""
        return {
            str(p.relative_to(self._bare_acts_dir)): file_fingerprint(p)
            for p in self._bare_acts_dir.rglob("*.pdf")
        }

    def _collect_pdf_sources(self) -> List[Tuple[Path, str, List[str]]]:
        """Enumerate (pdf_path, primary_domain, domains) triples to index."""
        return [
            (p, self.domain_name, [self.domain_name])
            for p in sorted(self._bare_acts_dir.rglob("*.pdf"))
        ]

    def _parse_pdf(
        self, full_text: str, source_file: str, primary_domain: str
    ) -> List[LegalChunk]:
        """Parse one PDF's text into chunks."""
        return self._parse_legal_sections(full_text, source_file)

    async def _build_vectorstore(self):
        """1."""
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
                pages = PyPDFLoader(str(pdf_path)).load()
                plain = "\n".join(p.page_content for p in pages)
                try:
                    layout_pages = PyPDFLoader(str(pdf_path), extraction_mode="layout").load()
                    layout = "\n".join(
                        line.strip()
                        for p in layout_pages
                        for line in p.page_content.split("\n")
                    )
                except Exception:
                    layout = None
                full_text, parsed = _choose_extraction(
                    plain,
                    layout,
                    lambda t: self._parse_pdf(t, pdf_path.name, primary_domain),
                )
                for c in parsed:
                    c.domains = list(domains)
                mode = "layout" if full_text is layout else "plain"
                print(f"  {pdf_path.name}: {len(pages)} pages → {len(parsed)} chunks ({mode})")
                all_chunks.extend(parsed)
            except Exception as e:
                print(f"  ERROR loading {pdf_path.name}: {e}")

        if not all_chunks:
            print(f"[{self.domain_name}] No chunks parsed — aborting build.")
            return


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
        """Replace any chunk whose body exceeds the embedder's useful window with overlapping "_pN" part-chunks carrying the same section metadata."""
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

    def _warn_if_chunk_cache_mismatch(self):
        """sections.json from a different build than the FAISS files (e.g."""
        if self.vector_store is None:
            return
        doc_ids = {d.metadata.get("chunk_id") for d in self.vector_store.docstore._dict.values()}
        missing = len(doc_ids - self._chunks.keys())
        if missing:
            print(
                f"[{self.domain_name}] WARNING: {missing}/{len(doc_ids)} indexed chunks are missing "
                f"from {self._cache_path} — their dense hits are dropped. Copy the matching "
                f"sections.json or run: python rebuild_rag_indices.py --domain {self.domain_name}"
            )

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
