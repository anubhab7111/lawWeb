#!/usr/bin/env python3
"""
Build the judgment retrieval index (app/tools/judgment_rag.py) from the chunked
Supreme Court corpus.

Selection: every chunk of every processed judgment — the whole corpus is
retrievable. Each chunk also records `doc_sections`, the bare section numbers of all
IPC and CrPC provisions its judgment cites (IL-TUR labels are bare numbers whose
cases mix both codes), which section voting uses.

Nothing is excluded at build time. Judgments overlapping IL-TUR test/dev cases
(app/ingest/decontaminate.py) are masked only when the benchmark is scored.

The index is IVF with 8-bit scalar quantisation (about 1 GB for the ~1.07M chunks),
searched with a configurable nprobe.

Three resumable phases, year by year (memory stays small):
  select    <corpus>/builds/judgments/selected/year=YYYY.jsonl.gz
  embed     <corpus>/builds/judgments/emb/year=YYYY.npz   (fp16, keyed by chunk id)
  assemble  app/data/faiss_index/judgments/  (FAISS IVF-SQ8 + SQLite/FTS5 + meta.json)

Run with Ollama idle:
    EMBEDDINGS_DEVICE=cuda python -m app.ingest.build_judgment_index
"""

from __future__ import annotations

import argparse
import gzip
import json
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np

from app.ingest.paths import corpus_path
from app.ingest.thermal import ThermalGuard
from app.tools import judgment_rag as jr
from app.tools import precedent_rag as pr

MAX_SEQ_LENGTH = 256
NUMBERED_ACTS = ("IPC:", "CrPC:")


def _years() -> List[int]:
    root = corpus_path("judgments", "sc", "chunks")
    return sorted(
        int(p.name.split("=")[1]) for p in root.glob("year=*") if (p / "_DONE.json").exists()
    )


def _read_chunks(year: int) -> List[Dict]:
    path = corpus_path("judgments", "sc", "chunks", f"year={year}", "chunks.jsonl.gz")
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def select_year(year: int) -> List[Dict]:
    chunks = _read_chunks(year)
    doc_sections: Dict[str, set] = {}
    for c in chunks:
        doc_sections.setdefault(c["doc_id"], set()).update(
            s.split(":", 1)[1] for s in c["sections_cited"] if s.startswith(NUMBERED_ACTS)
        )
    return [{**c, "doc_sections": sorted(doc_sections[c["doc_id"]])} for c in chunks]


def embed_text(chunk: Dict) -> str:
    return f"{chunk['case_title']} ({chunk['year']}). {chunk['text']}"


def phase_select(years: List[int]) -> Dict[int, int]:
    out = corpus_path("builds", "judgments", "selected")
    out.mkdir(parents=True, exist_ok=True)
    counts = {}
    for year in years:
        target = out / f"year={year}.jsonl.gz"
        if not target.exists():
            chosen = select_year(year)
            with gzip.open(target.with_suffix(".tmp"), "wt", encoding="utf-8") as f:
                for c in chosen:
                    f.write(json.dumps(c, ensure_ascii=False) + "\n")
            target.with_suffix(".tmp").rename(target)
        with gzip.open(target, "rt", encoding="utf-8") as f:
            counts[year] = sum(1 for _ in f)
    print(f"[judgment-index] selected {sum(counts.values())} chunks from {sum(1 for n in counts.values() if n)} years")
    return counts


def _read_selected(sel: Path, year: int) -> List[Dict]:
    with gzip.open(sel / f"year={year}.jsonl.gz", "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def _load_vectors(path: Path) -> Dict[str, np.ndarray]:
    if not path.exists():
        return {}
    data = np.load(path, allow_pickle=False)
    return dict(zip((str(i) for i in data["ids"]), data["vecs"]))


def _save_vectors(path: Path, ids: List[str], vecs: np.ndarray) -> None:
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez(tmp, ids=np.array(ids), vecs=vecs)
    tmp.rename(path)


def migrate_embeddings(years: List[int]) -> None:
    """One-time: year=YYYY.npy (positional, tied to that year's selection) ->
    year=YYYY.npz keyed by chunk id, so a changed selection reuses old vectors.
    Must run before the selection is regenerated."""
    sel = corpus_path("builds", "judgments", "selected")
    emb = corpus_path("builds", "judgments", "emb")
    for year in years:
        old = emb / f"year={year}.npy"
        if not old.exists():
            continue
        chunks = _read_selected(sel, year) if (sel / f"year={year}.jsonl.gz").exists() else []
        vecs = np.load(old)
        if len(chunks) == len(vecs):
            _save_vectors(emb / f"year={year}.npz", [c["chunk_id"] for c in chunks], vecs)
        old.unlink()


def phase_embed(years: List[int], device: str) -> None:
    """Embed only chunks without a cached vector; vectors are cached per chunk id."""
    from app.tools.base_legal_rag import _make_bge_embeddings

    sel = corpus_path("builds", "judgments", "selected")
    emb = corpus_path("builds", "judgments", "emb")
    emb.mkdir(parents=True, exist_ok=True)

    pending = {}
    for year in years:
        chunks = _read_selected(sel, year)
        cached = _load_vectors(emb / f"year={year}.npz")
        missing = [c for c in chunks if c["chunk_id"] not in cached]
        if missing or not (emb / f"year={year}.npz").exists():
            pending[year] = (chunks, cached, missing)
    if not pending:
        return

    embeddings = _make_bge_embeddings(device)
    embeddings.client.max_seq_length = MAX_SEQ_LENGTH
    if device == "cuda":
        embeddings.client.half()
    guard = ThermalGuard()
    started = time.time()
    for done, (year, (chunks, cached, missing)) in enumerate(pending.items(), start=1):
        if missing:
            texts = [embed_text(c) for c in missing]
            parts = []
            for i in range(0, len(texts), 512):
                parts.append(pr.encode_texts(embeddings, texts[i : i + 512], batch_size=32).astype(np.float16))
                guard.step()
            fresh = np.concatenate(parts)
            cached.update({c["chunk_id"]: v for c, v in zip(missing, fresh)})
        vecs = np.stack([cached[c["chunk_id"]] for c in chunks]) if chunks else np.empty((0, 1024), dtype=np.float16)
        _save_vectors(emb / f"year={year}.npz", [c["chunk_id"] for c in chunks], vecs)
        print(f"[judgment-index] {year}: {len(missing)} new / {len(chunks)} chunks embedded ({done}/{len(pending)}, {time.time() - started:.0f}s)")


def _year_vectors(sel: Path, emb: Path, year: int):
    chunks = _read_selected(sel, year)
    if not chunks:
        return [], None
    cached = _load_vectors(emb / f"year={year}.npz")
    return chunks, np.stack([cached[c["chunk_id"]] for c in chunks]).astype(np.float32)


def train_ivf(years: List[int], dim: int, nlist: int, sample: int, seed: int = 0):
    """IVF-SQ8 index trained on a random sample of all chunk vectors."""
    import faiss

    sel = corpus_path("builds", "judgments", "selected")
    emb = corpus_path("builds", "judgments", "emb")
    rng = np.random.default_rng(seed)
    parts, total = [], 0
    for year in years:
        _, vecs = _year_vectors(sel, emb, year)
        if vecs is None:
            continue
        total += len(vecs)
        take = rng.random(len(vecs)) < min(1.0, sample / 1.07e6 * 1.2)
        parts.append(vecs[take])
    train = np.concatenate(parts)[:sample]
    quantizer = faiss.IndexFlatIP(dim)
    index = faiss.IndexIVFScalarQuantizer(quantizer, dim, nlist, faiss.ScalarQuantizer.QT_8bit, faiss.METRIC_INNER_PRODUCT)
    index.train(train)
    print(f"[judgment-index] IVF trained: nlist {nlist} on {len(train)} of {total} vectors")
    return index


def phase_assemble(years: List[int], out_dir: Optional[Path] = None, nlist: int = 4096, sample: int = 200_000) -> None:
    import faiss

    from app.config import get_settings

    out_dir = out_dir or jr.INDEX_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    sel = corpus_path("builds", "judgments", "selected")
    emb = corpus_path("builds", "judgments", "emb")
    index = None
    con = jr.create_database(out_dir / jr.DB_FILE)
    row = 0
    docs: Set[str] = set()
    for year in years:
        chunks, vecs = _year_vectors(sel, emb, year)
        if vecs is None:
            continue
        if index is None:
            index = train_ivf(years, vecs.shape[1], nlist, sample)
        index.add_with_ids(vecs, np.arange(row, row + len(chunks), dtype=np.int64))
        jr.insert_chunks(con, row, chunks, rebuild_fts=False)
        docs.update(c["doc_id"] for c in chunks)
        row += len(chunks)
        con.commit()
    assert index is not None, "nothing to assemble"
    print(f"[judgment-index] building full-text index over {row} chunks ...", flush=True)
    con.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild')")
    con.execute("CREATE INDEX chunks_doc ON chunks(doc_id)")
    con.commit()
    con.close()
    faiss.write_index(index, str(out_dir / jr.INDEX_FILE))
    (out_dir / jr.META_FILE).write_text(
        json.dumps(
            {
                "embedding_model": get_settings().embedding_model,
                "chunks": row,
                "docs": len(docs),
                "years": [years[0], years[-1]],
                "index": f"IVF{nlist},SQ8",
                "source": "Supreme Court of India judgments, full corpus (IL-TUR overlap masked only at evaluation)",
            }
        )
    )
    print(f"[judgment-index] index: {row} chunks from {len(docs)} judgments -> {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--years", type=int, nargs="*")
    parser.add_argument("--select-only", action="store_true")
    parser.add_argument("--embed-only", action="store_true")
    parser.add_argument("--reselect", action="store_true", help="discard the cached selection (after changing the criteria)")
    parser.add_argument("--nlist", type=int, default=4096)
    parser.add_argument("--out-dir", type=Path, help="write the index elsewhere (smoke tests)")
    args = parser.parse_args()

    years = args.years or _years()
    print(f"[judgment-index] {len(years)} processed years")
    migrate_embeddings(years)
    if args.reselect:
        shutil.rmtree(corpus_path("builds", "judgments", "selected"), ignore_errors=True)
    phase_select(years)
    if args.select_only:
        return
    phase_embed(years, args.device)
    if args.embed_only:
        return
    phase_assemble(years, args.out_dir, nlist=args.nlist)


if __name__ == "__main__":
    main()
