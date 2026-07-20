"""
case_law_rag.py — curated landmark-judgment index, separate from the
statute index (different data shape: a handful of long prose documents
with authority metadata, not thousands of short numbered sections).

Retrieval is hybrid (BM25 + dense → RRF → cross-encoder rerank), same
building blocks as base_legal_rag.py, then RE-ORDERED by authority:
court tier first (Constitution Bench > Supreme Court > High Court),
recency second, relevance third within a tier — so a later Constitution
Bench does not lose to a more "semantically similar" High Court ruling.
Cases whose `statutes_cited` overlap the caller's already-retrieved
statute sections get a small relevance boost (multi-hop: statute -> case).
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from app.tools.base_legal_rag import (
    _bm25_tokenize,
    _get_shared_embeddings,
    _get_shared_reranker,
    _sigmoid,
    _split_long_text,
)

CASE_LAW_DIR = Path(__file__).resolve().parent.parent / "data" / "case_law"
FAISS_DIR = Path(__file__).resolve().parent.parent / "data" / "faiss_index" / "case_law"


@dataclass
class CaseChunk:
    chunk_id: str
    case_name: str
    citation: str
    court: str
    bench_size: int
    court_rank: int
    date: str
    status: str
    doctrines: List[str] = field(default_factory=list)
    statutes_cited: List[List[str]] = field(default_factory=list)
    text: str = ""
    url: str = ""
    score: float = 0.0


class CaseLawRAGSystem:
    """Hybrid-retrieval, authority-ranked index over app/data/case_law/."""

    def __init__(self):
        self.vector_store: Optional[Any] = None
        self.embeddings: Optional[Any] = None
        self.initialized: bool = False
        self._init_lock = asyncio.Lock()
        self._chunks: Dict[str, CaseChunk] = {}
        self._bm25: Optional[Any] = None
        self._bm25_ids: List[str] = []

    async def initialize(self) -> bool:
        if self.initialized:
            return True
        async with self._init_lock:
            if self.initialized:
                return True
            try:
                self.embeddings = await _get_shared_embeddings()
                if await self._should_rebuild():
                    await self._build_vectorstore()
                else:
                    await self._load_vectorstore()
                    self._load_chunk_cache()
                if self.vector_store is None:
                    return False
                self._build_bm25()
                self.initialized = True
                print(f"[case_law] RAG ready — {len(self._chunks)} chunks.")
                return True
            except Exception as e:
                print(f"[case_law] Initialization error: {e}")
                import traceback

                traceback.print_exc()
                return False

    async def _should_rebuild(self) -> bool:
        meta_path = FAISS_DIR / "meta.pkl"
        if not FAISS_DIR.exists() or not meta_path.exists():
            return True
        meta_mtime = meta_path.stat().st_mtime
        return any(
            p.stat().st_mtime > meta_mtime for p in CASE_LAW_DIR.glob("*.json")
        )

    async def _build_vectorstore(self):
        from langchain_community.vectorstores import FAISS
        from langchain_core.documents import Document

        case_files = sorted(CASE_LAW_DIR.glob("*.json"))
        if not case_files:
            print(f"[case_law] No cases found in {CASE_LAW_DIR} — skipping build.")
            return

        self._chunks = {}
        for path in case_files:
            try:
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as e:
                print(f"  ERROR loading {path.name}: {e}")
                continue
            parts = _split_long_text(data.get("text", ""), max_chars=3000, overlap=300)
            for n, part in enumerate(parts, 1):
                cid = f"{path.stem}_p{n}"
                self._chunks[cid] = CaseChunk(
                    chunk_id=cid,
                    case_name=data["case_name"],
                    citation=data.get("citation", ""),
                    court=data.get("court", ""),
                    bench_size=data.get("bench_size", 1),
                    court_rank=data.get("court_rank", 1),
                    date=data.get("date", ""),
                    status=data.get("status", "reported"),
                    doctrines=data.get("doctrines", []),
                    statutes_cited=data.get("statutes_cited", []),
                    text=part,
                    url=data.get("url", ""),
                )

        if not self._chunks:
            print("[case_law] No chunks parsed — aborting build.")
            return

        self._save_chunk_cache()
        documents = [
            Document(
                page_content=f"{c.case_name}. {c.citation}. {c.text}",
                metadata={"chunk_id": cid},
            )
            for cid, c in self._chunks.items()
        ]
        print(f"[case_law] Embedding {len(documents)} chunks …")
        loop = asyncio.get_event_loop()
        self.vector_store = await loop.run_in_executor(
            None, lambda: FAISS.from_documents(documents, self.embeddings)
        )
        FAISS_DIR.mkdir(parents=True, exist_ok=True)
        await loop.run_in_executor(
            None, lambda: self.vector_store.save_local(str(FAISS_DIR))
        )
        import pickle

        with open(FAISS_DIR / "meta.pkl", "wb") as f:
            pickle.dump({"n_cases": len(case_files)}, f)
        print(f"[case_law] Vector store built and saved ({len(case_files)} cases).")

    def _save_chunk_cache(self):
        FAISS_DIR.mkdir(parents=True, exist_ok=True)
        cache = {
            cid: {
                "case_name": c.case_name,
                "citation": c.citation,
                "court": c.court,
                "bench_size": c.bench_size,
                "court_rank": c.court_rank,
                "date": c.date,
                "status": c.status,
                "doctrines": c.doctrines,
                "statutes_cited": c.statutes_cited,
                "text": c.text,
                "url": c.url,
            }
            for cid, c in self._chunks.items()
        }
        with open(FAISS_DIR / "sections.json", "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2, ensure_ascii=False)

    def _load_chunk_cache(self):
        cache_path = FAISS_DIR / "sections.json"
        if not cache_path.exists():
            return
        with open(cache_path, encoding="utf-8") as f:
            cache = json.load(f)
        for cid, d in cache.items():
            self._chunks[cid] = CaseChunk(chunk_id=cid, **d)

    async def _load_vectorstore(self):
        from langchain_community.vectorstores import FAISS

        loop = asyncio.get_event_loop()
        try:
            self.vector_store = await loop.run_in_executor(
                None,
                lambda: FAISS.load_local(
                    str(FAISS_DIR), self.embeddings, allow_dangerous_deserialization=True
                ),
            )
        except Exception as e:
            print(f"[case_law] Could not load index ({e}) — rebuilding…")
            await self._build_vectorstore()

    def _build_bm25(self):
        self._bm25 = None
        self._bm25_ids = []
        if not self._chunks:
            return
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            print("[case_law] rank_bm25 not installed — dense-only mode.")
            return
        corpus = []
        for cid, c in self._chunks.items():
            self._bm25_ids.append(cid)
            corpus.append(_bm25_tokenize(f"{c.case_name} {c.citation} {c.text}"))
        self._bm25 = BM25Okapi(corpus)

    # ── Retrieval ────────────────────────────────────────────────

    async def retrieve(
        self,
        query: str,
        k: int = 4,
        boost_statutes: Optional[List[Tuple[str, str]]] = None,
        min_score: float = 0.15,
    ) -> List[CaseChunk]:
        """
        Hybrid retrieval, RRF-fused, cross-encoder reranked, then re-ordered
        by (statute overlap boost, court authority tier, recency, relevance)
        — so a governing Constitution Bench outranks a merely-similar High
        Court note, and cases interpreting the statutes already in context
        surface first (the statute -> case multi-hop).
        """
        if not self.initialized:
            await self.initialize()
        if not self.initialized or not self.vector_store:
            return []

        loop = asyncio.get_event_loop()
        candidate_pool = 20

        dense_results = await loop.run_in_executor(
            None,
            lambda: self.vector_store.similarity_search_with_score(
                query, k=candidate_pool
            ),
        )
        dense_rank = {}
        for rank, (doc, _dist) in enumerate(dense_results):
            cid = doc.metadata.get("chunk_id", "")
            if cid and cid not in dense_rank:
                dense_rank[cid] = rank

        sparse_rank = {}
        if self._bm25 is not None:
            tokens = _bm25_tokenize(query)
            if tokens:
                scores = self._bm25.get_scores(tokens)
                order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
                for rank, idx in enumerate(order[:candidate_pool]):
                    if scores[idx] <= 0:
                        break
                    sparse_rank[self._bm25_ids[idx]] = rank

        fused: Dict[str, float] = {}
        for cid, rank in dense_rank.items():
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (60 + rank)
        for cid, rank in sparse_rank.items():
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (60 + rank)
        candidates = [c for c in sorted(fused, key=fused.get, reverse=True) if c in self._chunks][:15]
        if not candidates:
            return []

        reranker = await _get_shared_reranker()
        relevance: Dict[str, float] = {}
        if reranker is not None:
            try:
                pairs = [(query, self._chunks[cid].text[:2000]) for cid in candidates]
                logits = await loop.run_in_executor(
                    None, lambda: reranker.predict(pairs, batch_size=16)
                )
                raw = [float(s) for s in logits]
                if any(s < 0.0 or s > 1.0 for s in raw):
                    raw = [_sigmoid(s) for s in raw]
                top = max(raw) if raw else 0.0
                if top >= 1e-4:
                    relevance = {cid: p / top for cid, p in zip(candidates, raw)}
            except Exception as e:
                print(f"[case_law] Rerank failed ({e}) — using fused order.")
        if not relevance:
            relevance = {cid: 0.5 for cid in candidates}

        boost_keys: Set[Tuple[str, str]] = set()
        if boost_statutes:
            boost_keys = {(a.lower(), s.lower()) for a, s in boost_statutes}

        def sort_key(cid: str):
            c = self._chunks[cid]
            rel = relevance.get(cid, 0.0)
            if rel < min_score:
                return None
            overlap = any(
                (a.lower(), s.lower()) in boost_keys for a, s in c.statutes_cited
            )
            return (overlap, c.court_rank, c.date, rel)

        scored = [(cid, sort_key(cid)) for cid in candidates]
        scored = [(cid, key) for cid, key in scored if key is not None]
        scored.sort(key=lambda t: t[1], reverse=True)

        # Dedupe to one chunk per case (multiple parts of the same judgment
        # shouldn't crowd out other relevant cases in a short context block).
        seen_cases: set = set()
        out: List[CaseChunk] = []
        for cid, _key in scored:
            c = self._chunks[cid]
            if c.case_name in seen_cases:
                continue
            seen_cases.add(c.case_name)
            result = CaseChunk(**{**c.__dict__, "score": round(relevance.get(cid, 0.0), 3)})
            out.append(result)
            if len(out) >= k:
                break
        return out


_case_law_rag: Optional[CaseLawRAGSystem] = None


def get_case_law_rag_system() -> CaseLawRAGSystem:
    global _case_law_rag
    if _case_law_rag is None:
        _case_law_rag = CaseLawRAGSystem()
    return _case_law_rag
