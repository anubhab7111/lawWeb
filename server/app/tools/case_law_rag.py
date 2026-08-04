"""
case_law_rag.py — curated landmark-judgment index, separate from the
statute index (different data shape: a handful of long prose documents
with authority metadata, not thousands of short numbered sections).

Each case is indexed as ONE document — a compact FIRAC summary (facts,
issues, holding, ratio decidendi, generated offline by
generate_case_firac.py into the case JSON's `firac` field), not raw
judgment prose. Retrieval is hybrid (BM25 + dense → RRF → cross-encoder
rerank against that summary), then re-scored by a composite of semantic
relevance, issue overlap, statute overlap and doctrine overlap, and
finally tie-broken by authority (court tier first, recency second) — so a
later Constitution Bench does not lose to a merely "semantically similar"
High Court ruling, and a case actually deciding the caller's legal issues
outranks one that only shares surface vocabulary.
"""

from __future__ import annotations

import asyncio
import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from app.tools.base_legal_rag import (
    _bm25_tokenize,
    _get_shared_embeddings,
    _get_shared_reranker,
    _make_bge_embeddings,
    _sigmoid,
)

CASE_LAW_DIR = Path(__file__).resolve().parent.parent / "data" / "case_law"
FAISS_DIR = Path(__file__).resolve().parent.parent / "data" / "faiss_index" / "case_law"

# Composite reranking weights — must sum to 1.0. Semantic similarity still
# dominates (it's the only signal general-chatbot callers without a FIRAC
# query have), but issue/statute/doctrine overlap now materially move the
# ranking instead of only breaking ties.
_W_SEMANTIC = 0.40
_W_ISSUE = 0.30
_W_STATUTE = 0.20
_W_DOCTRINE = 0.10

_ISSUE_STOPWORDS = frozenset(
    {
        "whether", "is", "are", "the", "of", "a", "an", "in", "to", "for",
        "and", "or", "under", "does", "do", "can", "was", "be", "by",
        "on", "with", "as", "that", "this",
    }
)


@dataclass
class CaseRecord:
    case_id: str
    case_name: str
    citation: str
    court: str
    bench_size: int
    court_rank: int
    date: str
    status: str
    doctrines: List[str] = field(default_factory=list)
    statutes_cited: List[List[str]] = field(default_factory=list)
    facts: str = ""
    issues: List[str] = field(default_factory=list)
    rules: List[str] = field(default_factory=list)
    holding: str = ""
    ratio_decidendi: str = ""
    firac_domain: str = ""
    summary: str = ""
    text: str = ""
    url: str = ""
    score: float = 0.0
    score_breakdown: Dict[str, float] = field(default_factory=dict)


def _build_summary(firac: dict) -> str:
    facts = firac.get("facts", "")
    issues = firac.get("issues") or []
    holding = firac.get("holding", "")
    ratio = firac.get("ratio_decidendi", "")
    parts = []
    if facts:
        parts.append(f"Facts: {facts}")
    if issues:
        parts.append(f"Issues: {'; '.join(issues)}")
    if holding:
        parts.append(f"Holding: {holding}")
    if ratio:
        parts.append(f"Ratio: {ratio}")
    return " ".join(parts)


def _issue_overlap_score(query_issues: List[str], case_issues: List[str]) -> float:
    """Best-match token-Jaccard, averaged over query issues. Cheap (no model
    calls) — both lists are short phrases, so this is a tiny nested loop over
    at most a few dozen candidates x a handful of issues, not a bottleneck."""
    if not query_issues or not case_issues:
        return 0.0

    def toks(s: str) -> Set[str]:
        return {t for t in _bm25_tokenize(s) if t not in _ISSUE_STOPWORDS}

    case_toks = [toks(i) for i in case_issues]
    case_toks = [t for t in case_toks if t]
    if not case_toks:
        return 0.0

    best_scores = []
    for qi in query_issues:
        qt = toks(qi)
        if not qt:
            continue
        best_scores.append(max(len(qt & ct) / len(qt | ct) for ct in case_toks))
    return sum(best_scores) / len(best_scores) if best_scores else 0.0


def _statute_overlap_score(
    case_statutes: List[List[str]], boost_keys: Set[Tuple[str, str]]
) -> float:
    """Recall against the caller's boost set, not Jaccard against the case's
    full citation list — a case citing 10 statutes but hitting both boosted
    ones should score as well as one citing only those 2."""
    if not case_statutes or not boost_keys:
        return 0.0
    case_keys = {(a.lower(), s.lower()) for a, s in case_statutes}
    return len(case_keys & boost_keys) / len(boost_keys)


def _doctrine_overlap_score(case_doctrines: List[str], query_doctrines: List[str]) -> float:
    if not case_doctrines or not query_doctrines:
        return 0.0
    cd = {d.lower() for d in case_doctrines}
    qd = {d.lower() for d in query_doctrines}
    return len(cd & qd) / len(qd)


class CaseLawRAGSystem:
    """Hybrid-retrieval, authority-ranked index over app/data/case_law/."""

    def __init__(self):
        self.vector_store: Optional[Any] = None
        self.embeddings: Optional[Any] = None
        self.initialized: bool = False
        self._init_lock = asyncio.Lock()
        self._cases: Dict[str, CaseRecord] = {}
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
                    self._load_case_cache()
                if self.vector_store is None:
                    return False
                self._build_bm25()
                self.initialized = True
                print(f"[case_law] RAG ready — {len(self._cases)} cases.")
                return True
            except Exception as e:
                print(f"[case_law] Initialization error: {e}")
                import traceback

                traceback.print_exc()
                return False

    async def build_offline(self, device: str) -> bool:
        """
        Force-build (or confirm up-to-date) using a throwaway embeddings
        instance independent of the shared query-time singleton. See
        BaseLegalRAGSystem.build_offline for the full rationale — same
        offline-indexing-then-free-GPU pattern, applied to the case-law
        index.
        """
        if not await self._should_rebuild():
            print("[case_law] Index already up to date — skipping build.")
            return True
        embeddings = _make_bge_embeddings(device)
        self.embeddings = embeddings
        try:
            print(f"[case_law] Offline build on {device} …")
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

    async def _should_rebuild(self) -> bool:
        """True if the index is absent or the source case JSON changed.

        Fingerprint-based (path -> size), not mtime-based: a fresh
        checkout/worktree resets file mtimes to checkout time even when
        content is unchanged, which would otherwise force a spurious
        full re-embed. Adding/changing the `firac` block changes a case
        file's byte size, so this also picks up FIRAC backfills for free.
        """
        meta_path = FAISS_DIR / "meta.pkl"
        if not FAISS_DIR.exists() or not meta_path.exists():
            return True
        try:
            with open(meta_path, "rb") as f:
                meta = pickle.load(f)
        except Exception:
            return True
        stored_fingerprint = meta.get("case_fingerprint")
        if stored_fingerprint is None:
            return True  # meta.pkl predates fingerprinting — rebuild once
        # Vectors are model-specific — a different embedding model forces a
        # rebuild (see base_legal_rag._should_rebuild).
        from app.config import get_settings

        if meta.get("embedding_model") != get_settings().embedding_model:
            return True
        return stored_fingerprint != self._current_case_fingerprint()

    @staticmethod
    def _current_case_fingerprint() -> Dict[str, int]:
        return {p.name: p.stat().st_size for p in CASE_LAW_DIR.glob("*.json")}

    async def _build_vectorstore(self):
        from langchain_community.vectorstores import FAISS
        from langchain_core.documents import Document

        from app.config import get_settings

        case_files = sorted(CASE_LAW_DIR.glob("*.json"))
        if not case_files:
            print(f"[case_law] No cases found in {CASE_LAW_DIR} — skipping build.")
            return

        self._cases = {}
        missing_firac = 0
        for path in case_files:
            try:
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as e:
                print(f"  ERROR loading {path.name}: {e}")
                continue

            case_id = path.stem
            firac = data.get("firac") or {}
            if firac.get("issues") or firac.get("holding"):
                summary = _build_summary(firac)
            else:
                missing_firac += 1
                summary = data.get("text", "")[:1500]

            self._cases[case_id] = CaseRecord(
                case_id=case_id,
                case_name=data["case_name"],
                citation=data.get("citation", ""),
                court=data.get("court", ""),
                bench_size=data.get("bench_size", 1),
                court_rank=data.get("court_rank", 1),
                date=data.get("date", ""),
                status=data.get("status", "reported"),
                doctrines=data.get("doctrines", []),
                statutes_cited=data.get("statutes_cited", []),
                facts=firac.get("facts", ""),
                issues=firac.get("issues", []),
                rules=firac.get("rules", []),
                holding=firac.get("holding", ""),
                ratio_decidendi=firac.get("ratio_decidendi", ""),
                firac_domain=firac.get("domain", ""),
                summary=summary,
                text=data.get("text", ""),
                url=data.get("url", ""),
            )

        if not self._cases:
            print("[case_law] No cases parsed — aborting build.")
            return
        if missing_firac:
            print(
                f"[case_law] {missing_firac} case(s) missing 'firac' — "
                f"falling back to raw-text summary; run generate_case_firac.py."
            )

        self._save_case_cache()
        documents = [
            Document(
                page_content=(
                    f"{c.case_name}. {c.citation}. {c.court}. "
                    f"Domain: {c.firac_domain}. Doctrines: {', '.join(c.doctrines)}. "
                    f"{c.summary} "
                    f"Statutes: {', '.join(f'{a} s.{s}' for a, s in c.statutes_cited)}."
                ),
                metadata={"case_id": cid},
            )
            for cid, c in self._cases.items()
        ]
        print(f"[case_law] Embedding {len(documents)} cases …")
        loop = asyncio.get_event_loop()
        self.vector_store = await loop.run_in_executor(
            None, lambda: FAISS.from_documents(documents, self.embeddings)
        )
        FAISS_DIR.mkdir(parents=True, exist_ok=True)
        await loop.run_in_executor(
            None, lambda: self.vector_store.save_local(str(FAISS_DIR))
        )
        with open(FAISS_DIR / "meta.pkl", "wb") as f:
            pickle.dump(
                {
                    "n_cases": len(case_files),
                    "case_fingerprint": self._current_case_fingerprint(),
                    "embedding_model": get_settings().embedding_model,
                },
                f,
            )
        print(f"[case_law] Vector store built and saved ({len(case_files)} cases).")

    def _save_case_cache(self):
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
                "facts": c.facts,
                "issues": c.issues,
                "rules": c.rules,
                "holding": c.holding,
                "ratio_decidendi": c.ratio_decidendi,
                "firac_domain": c.firac_domain,
                "summary": c.summary,
                "text": c.text,
                "url": c.url,
            }
            for cid, c in self._cases.items()
        }
        with open(FAISS_DIR / "cases.json", "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2, ensure_ascii=False)

    def _load_case_cache(self):
        cache_path = FAISS_DIR / "cases.json"
        if not cache_path.exists():
            return
        with open(cache_path, encoding="utf-8") as f:
            cache = json.load(f)
        for cid, d in cache.items():
            self._cases[cid] = CaseRecord(case_id=cid, **d)

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
        if not self._cases:
            return
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            print("[case_law] rank_bm25 not installed — dense-only mode.")
            return
        corpus = []
        for cid, c in self._cases.items():
            self._bm25_ids.append(cid)
            corpus.append(
                _bm25_tokenize(f"{c.case_name} {c.citation} {' '.join(c.doctrines)} {c.summary}")
            )
        self._bm25 = BM25Okapi(corpus)

    # ── Retrieval ────────────────────────────────────────────────

    async def retrieve(
        self,
        query: str,
        k: int = 4,
        boost_statutes: Optional[List[Tuple[str, str]]] = None,
        query_issues: Optional[List[str]] = None,
        query_doctrines: Optional[List[str]] = None,
        min_score: float = 0.15,
    ) -> List[CaseRecord]:
        """
        Hybrid retrieval, RRF-fused, cross-encoder reranked against each
        case's FIRAC summary, then re-scored by a weighted composite of
        semantic relevance + issue overlap + statute overlap + doctrine
        overlap, tie-broken by (court authority tier, recency).

        query_issues/query_doctrines are optional — callers without a
        structured FIRAC query (e.g. the general chatbot's statute->case
        multi-hop) simply get 0.0 contribution from those terms for every
        candidate, which degrades ranking to semantic+statute-weighted
        rather than distorting it.
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
            cid = doc.metadata.get("case_id", "")
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
        candidates = [c for c in sorted(fused, key=fused.get, reverse=True) if c in self._cases][:15]
        if not candidates:
            return []

        reranker = await _get_shared_reranker()
        relevance: Dict[str, float] = {}
        if reranker is not None:
            try:
                pairs = [(query, self._cases[cid].summary[:2000]) for cid in candidates]
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
        query_issues = query_issues or []
        query_doctrines = query_doctrines or []

        breakdowns: Dict[str, Dict[str, float]] = {}

        def sort_key(cid: str):
            c = self._cases[cid]
            rel = relevance.get(cid, 0.0)
            if rel < min_score:
                return None
            issue_o = _issue_overlap_score(query_issues, c.issues)
            stat_o = _statute_overlap_score(c.statutes_cited, boost_keys)
            doc_o = _doctrine_overlap_score(c.doctrines, query_doctrines)
            composite = (
                _W_SEMANTIC * rel
                + _W_ISSUE * issue_o
                + _W_STATUTE * stat_o
                + _W_DOCTRINE * doc_o
            )
            breakdowns[cid] = {
                "semantic": round(rel, 3),
                "issue": round(issue_o, 3),
                "statute": round(stat_o, 3),
                "doctrine": round(doc_o, 3),
            }
            return (round(composite, 4), c.court_rank, c.date)

        scored = [(cid, sort_key(cid)) for cid in candidates]
        scored = [(cid, key) for cid, key in scored if key is not None]
        scored.sort(key=lambda t: t[1], reverse=True)

        out: List[CaseRecord] = []
        for cid, key in scored[:k]:
            c = self._cases[cid]
            result = CaseRecord(
                **{**c.__dict__, "score": key[0], "score_breakdown": breakdowns[cid]}
            )
            out.append(result)
        return out


_case_law_rag: Optional[CaseLawRAGSystem] = None


def get_case_law_rag_system() -> CaseLawRAGSystem:
    global _case_law_rag
    if _case_law_rag is None:
        _case_law_rag = CaseLawRAGSystem()
    return _case_law_rag
