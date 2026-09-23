#!/usr/bin/env python3
"""
Build the IL-TUR precedent index (app/tools/precedent_rag.py) from train+dev.

Windows are embedded in shards written to <corpus>/builds/precedent/, so a
killed run resumes at the first missing shard. Once every shard exists they are
assembled into a FAISS fp16 index under app/data/faiss_index/precedent/.

Run with Ollama idle (the embedder needs ~1.2GB of VRAM in fp16):
    EMBEDDINGS_DEVICE=cuda python -m app.ingest.build_precedent_index
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from app.ingest.iltur_export import assert_disjoint, lsi_dir
from app.ingest.paths import corpus_path
from app.metrics.iltur_eval import fact_windows
from app.tools import precedent_rag as pr

SHARD_SIZE = 16384


def load_cases() -> Tuple[List[str], List[List[str]], List[List[str]]]:
    """(case ids, fact sentences, labels) for train then dev — never test."""
    root = lsi_dir()
    train, dev = pd.read_parquet(root / "train.parquet"), pd.read_parquet(root / "dev.parquet")
    test_ids = pd.read_parquet(root / "test.parquet", columns=["id"])["id"].tolist()
    assert_disjoint(
        {"train": train["id"].tolist(), "dev": dev["id"].tolist(), "test": test_ids}
    )
    both = pd.concat([train, dev], ignore_index=True)
    return (
        both["id"].tolist(),
        [list(s) for s in both["sentences"]],
        [list(x) for x in both["labels"]],
    )


def make_windows(sentences: List[List[str]], per_case: int) -> Tuple[List[str], np.ndarray]:
    texts: List[str] = []
    owner: List[int] = []
    for case_idx, sents in enumerate(sentences):
        for window in fact_windows(sents, window=4, stride=3, max_windows=per_case):
            texts.append(window)
            owner.append(case_idx)
    return texts, np.asarray(owner, dtype=np.int32)


def embed_shards(texts: List[str], shard_dir: Path, device: str) -> None:
    from app.tools.base_legal_rag import _make_bge_embeddings

    shard_dir.mkdir(parents=True, exist_ok=True)
    n_shards = (len(texts) + SHARD_SIZE - 1) // SHARD_SIZE
    todo = [i for i in range(n_shards) if not (shard_dir / f"shard_{i:04d}.npy").exists()]
    if not todo:
        return
    embeddings = _make_bge_embeddings(device)
    client = embeddings.client
    client.max_seq_length = 256
    if device == "cuda":
        client.half()
    started = time.time()
    for done, i in enumerate(todo, start=1):
        chunk = texts[i * SHARD_SIZE : (i + 1) * SHARD_SIZE]
        vecs = pr.encode_texts(embeddings, chunk, batch_size=64).astype(np.float16)
        tmp = shard_dir / f"shard_{i:04d}.tmp.npy"
        np.save(tmp, vecs)
        tmp.rename(shard_dir / f"shard_{i:04d}.npy")
        elapsed = time.time() - started
        print(f"[precedent] shard {i + 1}/{n_shards} ({done}/{len(todo)} this run, {elapsed:.0f}s)")


def assemble(shard_dir: Path, owner: np.ndarray, case_ids, labels, out_dir: Path) -> None:
    import faiss

    from app.config import get_settings

    vecs = np.concatenate(
        [np.load(p) for p in sorted(shard_dir.glob("shard_*.npy")) if ".tmp" not in p.name]
    )
    assert len(vecs) == len(owner), f"{len(vecs)} vectors for {len(owner)} windows"
    index = faiss.IndexScalarQuantizer(
        vecs.shape[1], faiss.ScalarQuantizer.QT_fp16, faiss.METRIC_INNER_PRODUCT
    )
    index.add(vecs.astype(np.float32))
    out_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(out_dir / pr.INDEX_FILE))
    np.save(out_dir / pr.WINDOW_CASE_FILE, owner)
    (out_dir / pr.META_FILE).write_text(
        json.dumps(
            {
                "embedding_model": get_settings().embedding_model,
                "windows": int(len(owner)),
                "cases": len(case_ids),
                "windows_per_case": pr.INDEX_WINDOWS,
                "case_ids": case_ids,
                "case_labels": labels,
                "source": "IL-TUR lsi train+dev",
            }
        )
    )
    print(f"[precedent] index: {len(owner)} windows over {len(case_ids)} cases -> {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--windows-per-case", type=int, default=pr.INDEX_WINDOWS)
    parser.add_argument("--limit-cases", type=int, help="smoke test on the first N cases")
    args = parser.parse_args()

    case_ids, sentences, labels = load_cases()
    if args.limit_cases:
        case_ids, sentences, labels = (x[: args.limit_cases] for x in (case_ids, sentences, labels))
    texts, owner = make_windows(sentences, args.windows_per_case)
    print(f"[precedent] {len(case_ids)} cases -> {len(texts)} windows")

    tag = f"w{args.windows_per_case}" + (f"_n{args.limit_cases}" if args.limit_cases else "")
    shard_dir = corpus_path("builds", "precedent", tag)
    embed_shards(texts, shard_dir, args.device)
    out_dir = pr.INDEX_DIR if not args.limit_cases else pr.INDEX_DIR.with_name("precedent_smoke")
    assemble(shard_dir, owner, case_ids, labels, out_dir)


if __name__ == "__main__":
    main()
