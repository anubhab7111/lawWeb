"""precedent_rag.py — statute identification from similar past cases."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

from app.metrics.iltur_eval import fact_windows

INDEX_DIR = Path(__file__).resolve().parent.parent / "data" / "faiss_index" / "precedent"
INDEX_FILE = "index.faiss"
META_FILE = "meta.json"
WINDOW_CASE_FILE = "window_case_idx.npy"

INDEX_WINDOWS = 4
QUERY_WINDOWS = 8
NEIGHBORS = 20
VOTE_POWER = 8.0


def encode_texts(embeddings: Any, texts: Sequence[str], batch_size: int = 64) -> np.ndarray:
    """L2-normalised float32 embeddings for texts, using the loaded BGE model."""
    client = getattr(embeddings, "client", None)
    if client is not None:
        return np.asarray(
            client.encode(
                list(texts),
                batch_size=batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            ),
            dtype=np.float32,
        )
    vecs = np.asarray(embeddings.embed_documents(list(texts)), dtype=np.float32)
    return vecs / np.maximum(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-12)


def vote_labels(
    neighbor_sims: np.ndarray,
    neighbor_cases: np.ndarray,
    case_labels: Sequence[Sequence[str]],
    exclude: Optional[Set[int]] = None,
    power: float = VOTE_POWER,
) -> Dict[str, float]:
    """Similarity-weighted label votes from per-query-window neighbours."""
    scores: Dict[str, float] = defaultdict(float)
    q = neighbor_sims.shape[0]
    for sims, cases in zip(neighbor_sims, neighbor_cases):
        best: Dict[int, float] = {}
        for sim, case in zip(sims, cases):
            case = int(case)
            if case < 0 or (exclude and case in exclude):
                continue
            if sim > best.get(case, -1.0):
                best[case] = float(sim)
        for case, sim in best.items():
            weight = max(sim, 0.0) ** power
            for label in case_labels[case]:
                scores[label] += weight / q
    return dict(scores)


def rank_votes(scores: Dict[str, float]) -> List[Tuple[str, float]]:
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))


class PrecedentIndex:
    def __init__(self, index_dir: Path = INDEX_DIR):
        self.dir = index_dir
        self._index: Any = None
        self._window_case: Optional[np.ndarray] = None
        self.meta: Dict = {}

    @property
    def available(self) -> bool:
        return all((self.dir / f).exists() for f in (INDEX_FILE, META_FILE, WINDOW_CASE_FILE))

    def load(self) -> None:
        import faiss

        if self._index is not None:
            return
        self.meta = json.loads((self.dir / META_FILE).read_text())
        from app.config import get_settings

        if self.meta.get("embedding_model") != get_settings().embedding_model:
            raise RuntimeError(
                f"precedent index built with {self.meta.get('embedding_model')}, "
                f"but embedding_model is {get_settings().embedding_model}; rebuild it"
            )
        self._index = faiss.read_index(str(self.dir / INDEX_FILE), faiss.IO_FLAG_MMAP)
        self._window_case = np.load(self.dir / WINDOW_CASE_FILE, mmap_mode="r")

    def load_labels(self) -> List[List[str]]:
        """Label set of every indexed case (train+dev)."""
        self.load()
        return self.meta["case_labels"]

    @property
    def case_ids(self) -> List[str]:
        self.load()
        return self.meta["case_ids"]

    def case_index(self, case_ids: Iterable[str]) -> Set[int]:
        lookup = {cid: i for i, cid in enumerate(self.case_ids)}
        return {lookup[c] for c in case_ids if c in lookup}

    def search(self, query_vecs: np.ndarray, k: int = NEIGHBORS, extra: int = 0):
        """(sims, owning case idx) of the k nearest windows per query vector."""
        self.load()
        sims, ids = self._index.search(np.ascontiguousarray(query_vecs, dtype=np.float32), k + extra)
        cases = np.where(ids >= 0, self._window_case[np.clip(ids, 0, None)], -1)
        return sims, cases

    def rank_sections(
        self,
        query_vecs: np.ndarray,
        k: int = NEIGHBORS,
        power: float = VOTE_POWER,
        exclude: Optional[Set[int]] = None,
    ) -> List[Tuple[str, float]]:

        sims, cases = self.search(query_vecs, k, extra=k if exclude else 0)
        return rank_votes(vote_labels(sims, cases, self.meta["case_labels"], exclude, power))


_index: Optional[PrecedentIndex] = None


def get_precedent_index() -> PrecedentIndex:
    global _index
    if _index is None:
        _index = PrecedentIndex()
    return _index


async def retrieve_precedent_sections(
    facts: Sequence[str],
    k: int = NEIGHBORS,
    power: float = VOTE_POWER,
    exclude_case_ids: Optional[Iterable[str]] = None,
) -> List[Tuple[str, float]]:
    """Candidate sections (number, score) voted by the most similar past cases."""
    from app.tools.base_legal_rag import _get_shared_embeddings

    index = get_precedent_index()
    if not index.available:
        return []
    windows = fact_windows(facts, window=4, stride=3, max_windows=QUERY_WINDOWS)
    if not windows:
        return []
    embeddings = await _get_shared_embeddings()
    vecs = encode_texts(embeddings, windows)
    exclude = index.case_index(exclude_case_ids) if exclude_case_ids else None
    return index.rank_sections(vecs, k=k, power=power, exclude=exclude)
