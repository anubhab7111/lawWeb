#!/usr/bin/env python3
"""
Build the judgment retrieval index (app/tools/judgment_rag.py) from the chunked
Supreme Court corpus.

Selection: all chunks of judgments that cite at least one IPC section and were not
flagged by decontamination (app/ingest/decontaminate.py) — the criminal-law
corpus IL-TUR statute identification draws on. The decontamination report is
required; there is no way to build without it short of --skip-decontam-check.

Three resumable phases, year by year (memory stays small):
  select    <corpus>/builds/judgments/selected/year=YYYY.jsonl.gz
  embed     <corpus>/builds/judgments/emb/year=YYYY.npy   (fp16)
  assemble  app/data/faiss_index/judgments/  (FAISS + SQLite/FTS5 + meta.json)

Run with Ollama idle:
    EMBEDDINGS_DEVICE=cuda python -m app.ingest.build_judgment_index
"""

from __future__ import annotations

import argparse
import gzip
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np

from app.ingest.paths import corpus_path
from app.tools import judgment_rag as jr
from app.tools import precedent_rag as pr

MAX_SEQ_LENGTH = 384


def _years() -> List[int]:
    root = corpus_path("judgments", "sc", "chunks")
    return sorted(
        int(p.name.split("=")[1]) for p in root.glob("year=*") if (p / "_DONE.json").exists()
    )


def _read_chunks(year: int) -> List[Dict]:
    path = corpus_path("judgments", "sc", "chunks", f"year={year}", "chunks.jsonl.gz")
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def load_flagged(skip_check: bool) -> Set[str]:
    report = corpus_path("builds", "decontam", "sc_flagged.json")
    if not report.exists():
        if skip_check:
            return set()
        raise SystemExit(
            f"{report} not found — run `python -m app.ingest.decontaminate` first "
            "(judgments must be checked against the IL-TUR test split before indexing)."
        )
    return set(json.loads(report.read_text())["flagged_docs"])


def select_year(year: int, flagged: Set[str]) -> List[Dict]:
    chunks = _read_chunks(year)
    citing: Dict[str, bool] = {}
    for c in chunks:
        citing[c["doc_id"]] = citing.get(c["doc_id"], False) or any(
            s.startswith("IPC:") for s in c["sections_cited"]
        )
    return [c for c in chunks if citing[c["doc_id"]] and c["doc_id"] not in flagged]


def embed_text(chunk: Dict) -> str:
    return f"{chunk['case_title']} ({chunk['year']}). {chunk['text']}"


def phase_select(years: List[int], flagged: Set[str]) -> Dict[int, int]:
    out = corpus_path("builds", "judgments", "selected")
    out.mkdir(parents=True, exist_ok=True)
    counts = {}
    for year in years:
        target = out / f"year={year}.jsonl.gz"
        if not target.exists():
            chosen = select_year(year, flagged)
            with gzip.open(target.with_suffix(".tmp"), "wt", encoding="utf-8") as f:
                for c in chosen:
                    f.write(json.dumps(c, ensure_ascii=False) + "\n")
            target.with_suffix(".tmp").rename(target)
        with gzip.open(target, "rt", encoding="utf-8") as f:
            counts[year] = sum(1 for _ in f)
    print(f"[judgment-index] selected {sum(counts.values())} chunks from {sum(1 for n in counts.values() if n)} years")
    return counts


def phase_embed(years: List[int], device: str) -> None:
    from app.tools.base_legal_rag import _make_bge_embeddings

    sel = corpus_path("builds", "judgments", "selected")
    emb = corpus_path("builds", "judgments", "emb")
    emb.mkdir(parents=True, exist_ok=True)
    todo = [y for y in years if not (emb / f"year={y}.npy").exists()]
    if not todo:
        return
    embeddings = _make_bge_embeddings(device)
    embeddings.client.max_seq_length = MAX_SEQ_LENGTH
    if device == "cuda":
        embeddings.client.half()
    started = time.time()
    for done, year in enumerate(todo, start=1):
        with gzip.open(sel / f"year={year}.jsonl.gz", "rt", encoding="utf-8") as f:
            texts = [embed_text(json.loads(line)) for line in f]
        vecs = (
            pr.encode_texts(embeddings, texts, batch_size=32).astype(np.float16)
            if texts
            else np.empty((0, 1024), dtype=np.float16)
        )
        tmp = emb / f"year={year}.tmp.npy"
        np.save(tmp, vecs)
        tmp.rename(emb / f"year={year}.npy")
        print(f"[judgment-index] embedded {year}: {len(texts)} chunks ({done}/{len(todo)}, {time.time() - started:.0f}s)")


def phase_assemble(years: List[int], out_dir: Optional[Path] = None) -> None:
    import faiss

    from app.config import get_settings

    out_dir = out_dir or jr.INDEX_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    sel = corpus_path("builds", "judgments", "selected")
    emb = corpus_path("builds", "judgments", "emb")
    con = jr.create_database(out_dir / jr.DB_FILE)
    index: Optional[faiss.Index] = None
    row = 0
    docs: Set[str] = set()
    for year in years:
        vecs = np.load(emb / f"year={year}.npy")
        if not len(vecs):
            continue
        with gzip.open(sel / f"year={year}.jsonl.gz", "rt", encoding="utf-8") as f:
            chunks = [json.loads(line) for line in f]
        assert len(chunks) == len(vecs), f"{year}: {len(chunks)} chunks vs {len(vecs)} vectors"
        if index is None:
            index = faiss.IndexScalarQuantizer(
                vecs.shape[1], faiss.ScalarQuantizer.QT_fp16, faiss.METRIC_INNER_PRODUCT
            )
        index.add(vecs.astype(np.float32))
        jr.insert_chunks(con, row, chunks, rebuild_fts=False)
        docs.update(c["doc_id"] for c in chunks)
        row += len(chunks)
    assert index is not None, "nothing to assemble"
    con.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild')")
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
                "source": "Supreme Court of India judgments (IPC-citing, decontaminated)",
            }
        )
    )
    print(f"[judgment-index] index: {row} chunks from {len(docs)} judgments -> {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--years", type=int, nargs="*")
    parser.add_argument("--skip-decontam-check", action="store_true")
    parser.add_argument("--select-only", action="store_true")
    parser.add_argument("--out-dir", type=Path, help="write the index elsewhere (smoke tests)")
    args = parser.parse_args()

    years = args.years or _years()
    flagged = load_flagged(args.skip_decontam_check)
    print(f"[judgment-index] {len(years)} processed years; {len(flagged)} judgments flagged as contaminated")
    phase_select(years, flagged)
    if args.select_only:
        return
    phase_embed(years, args.device)
    phase_assemble(years, args.out_dir)


if __name__ == "__main__":
    main()
