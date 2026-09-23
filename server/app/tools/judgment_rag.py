"""
judgment_rag.py — retrieval over chunked Supreme Court judgments.

Dense (FAISS, fp16 scalar-quantised inner product) and lexical (SQLite FTS5 BM25)
retrieval over judgment passages, fused with RRF. Passages carry the case name,
citation and the statute sections they cite, so results can be quoted with a
citation and boosted toward specific provisions ("passages that discuss IPC 302").

Stored under app/data/faiss_index/judgments/ — vectors, chunk text and metadata are
all inside the repo tree, so serving never touches the corpus drive. Loaded lazily
and memory-mapped.
"""

from __future__ import annotations

import json
import re
import sqlite3
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

INDEX_DIR = Path(__file__).resolve().parent.parent / "data" / "faiss_index" / "judgments"
INDEX_FILE = "index.faiss"
DB_FILE = "chunks.sqlite"
META_FILE = "meta.json"
RRF_K = 60
_FTS_TOKEN = re.compile(r"[A-Za-z0-9]{3,}")

SCHEMA = """
CREATE TABLE chunks (
    id INTEGER PRIMARY KEY, chunk_id TEXT UNIQUE, doc_id TEXT, year INTEGER,
    case_title TEXT, citation TEXT, role TEXT, sections_cited TEXT, text TEXT
);
CREATE TABLE chunk_sections (chunk_row INTEGER, section TEXT);
CREATE INDEX chunk_sections_section ON chunk_sections(section);
CREATE VIRTUAL TABLE chunks_fts USING fts5(text, content='chunks', content_rowid='id');
"""


@dataclass
class JudgmentPassage:
    chunk_id: str
    doc_id: str
    year: int
    case_title: str
    citation: str
    role: str
    sections_cited: List[str]
    text: str
    score: float = 0.0
    sources: List[str] = field(default_factory=list)


def fts_query(text: str, max_terms: int = 24) -> str:
    """OR-query of the text's distinctive tokens, safe for FTS5 syntax."""
    seen: List[str] = []
    for token in _FTS_TOKEN.findall(text.lower()):
        if token not in seen:
            seen.append(token)
    return " OR ".join(f'"{t}"' for t in seen[:max_terms])


def create_database(path: Path) -> sqlite3.Connection:
    path.unlink(missing_ok=True)
    con = sqlite3.connect(path)
    con.executescript(SCHEMA)
    return con


def insert_chunks(
    con: sqlite3.Connection, start_row: int, chunks: Sequence[Dict], rebuild_fts: bool = True
) -> None:
    for offset, c in enumerate(chunks):
        row = start_row + offset
        con.execute(
            "INSERT INTO chunks VALUES (?,?,?,?,?,?,?,?,?)",
            (row, c["chunk_id"], c["doc_id"], c["year"], c["case_title"], c.get("citation", ""),
             c["role"], json.dumps(c["sections_cited"]), c["text"]),
        )
        con.executemany(
            "INSERT INTO chunk_sections VALUES (?,?)", [(row, s) for s in c["sections_cited"]]
        )
    if rebuild_fts:
        con.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild')")


class JudgmentIndex:
    def __init__(self, index_dir: Path = INDEX_DIR):
        self.dir = index_dir
        self._index: Any = None
        self._db: Optional[sqlite3.Connection] = None
        self.meta: Dict = {}

    @property
    def available(self) -> bool:
        return all((self.dir / f).exists() for f in (INDEX_FILE, DB_FILE, META_FILE))

    def load(self) -> None:
        import faiss

        if self._index is not None:
            return
        self.meta = json.loads((self.dir / META_FILE).read_text())
        from app.config import get_settings

        if self.meta.get("embedding_model") != get_settings().embedding_model:
            raise RuntimeError(
                f"judgment index built with {self.meta.get('embedding_model')}, "
                f"but embedding_model is {get_settings().embedding_model}; rebuild it"
            )
        self._index = faiss.read_index(str(self.dir / INDEX_FILE), faiss.IO_FLAG_MMAP)
        self._db = sqlite3.connect(f"file:{self.dir / DB_FILE}?mode=ro", uri=True, check_same_thread=False)

    def _fetch(self, rows: Sequence[int]) -> Dict[int, JudgmentPassage]:
        assert self._db is not None
        if not rows:
            return {}
        marks = ",".join("?" * len(rows))
        out = {}
        for r in self._db.execute(
            f"SELECT id, chunk_id, doc_id, year, case_title, citation, role, sections_cited, text "
            f"FROM chunks WHERE id IN ({marks})", list(rows)
        ):
            out[r[0]] = JudgmentPassage(r[1], r[2], r[3], r[4], r[5], r[6], json.loads(r[7]), r[8])
        return out

    def dense(self, query_vecs: np.ndarray, k: int) -> List[List[tuple]]:
        self.load()
        sims, ids = self._index.search(np.ascontiguousarray(query_vecs, dtype=np.float32), k)
        return [[(int(i), float(s)) for i, s in zip(row_i, row_s) if i >= 0] for row_i, row_s in zip(ids, sims)]

    def lexical(self, query: str, k: int) -> List[int]:
        self.load()
        assert self._db is not None
        match = fts_query(query)
        if not match:
            return []
        rows = self._db.execute(
            "SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ? ORDER BY bm25(chunks_fts) LIMIT ?",
            (match, k),
        ).fetchall()
        return [r[0] for r in rows]

    def rows_citing(self, sections: Sequence[str], limit: int = 200) -> List[int]:
        self.load()
        assert self._db is not None
        if not sections:
            return []
        marks = ",".join("?" * len(sections))
        return [
            r[0]
            for r in self._db.execute(
                f"SELECT DISTINCT chunk_row FROM chunk_sections WHERE section IN ({marks}) LIMIT ?",
                [*sections, limit],
            )
        ]

    def retrieve(
        self,
        query_text: str,
        query_vecs: np.ndarray,
        k: int = 6,
        boost_sections: Sequence[str] = (),
        pool: int = 40,
    ) -> List[JudgmentPassage]:
        """Hybrid retrieval: dense + lexical fused by RRF; passages citing one of
        boost_sections (e.g. "IPC:302") get an extra rank-fusion vote."""
        scores: Dict[int, float] = defaultdict(float)
        sources: Dict[int, set] = defaultdict(set)
        for hits in self.dense(query_vecs, pool):
            for rank, (row, _sim) in enumerate(hits, start=1):
                scores[row] += 1.0 / (RRF_K + rank) / len(query_vecs)
                sources[row].add("dense")
        for rank, row in enumerate(self.lexical(query_text, pool), start=1):
            scores[row] += 1.0 / (RRF_K + rank)
            sources[row].add("lexical")
        if boost_sections:
            cited = set(self.rows_citing(boost_sections))
            for row in list(scores):
                if row in cited:
                    scores[row] += 1.0 / (RRF_K + 1)
                    sources[row].add("cites")
        top = sorted(scores, key=lambda r: -scores[r])[:k]
        passages = self._fetch(top)
        result = []
        for row in top:
            p = passages[row]
            p.score, p.sources = scores[row], sorted(sources[row])
            result.append(p)
        return result


_index: Optional[JudgmentIndex] = None


def get_judgment_index() -> JudgmentIndex:
    global _index
    if _index is None:
        _index = JudgmentIndex()
    return _index
